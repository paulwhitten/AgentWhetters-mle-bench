"""
Purple ML Agent — OpenAI-backed Kaggle competition solver.

Receives:
  - A text message containing the benchmark instructions
  - A file attachment: competition.tar.gz  (contents: home/data/description.md,
    home/data/train.csv, home/data/test.csv, …)

Runs:
  1. Extracts the tar into a temp working directory
  2. Reads description.md to understand the task
  3. Asks OpenAI to write self-contained Python ML code
  4. Executes the code (subprocess) in the working directory
  5. Optionally validates the submission with the green agent
  6. Returns submission.csv as a task artifact

Usage:
  uv run src/purple/server.py --host 127.0.0.1 --port 9022
"""

import argparse
import ast
import asyncio
import base64
import contextlib
import io
import json
import logging
import os
import re
import subprocess
import sys
import tarfile
import tempfile
import textwrap
import time
from pathlib import Path

# Ensure the package root (src/) is on sys.path so that
# ``from purple.ml_toolkit import ...`` works regardless
# of how the server process is launched.
_src_dir = str(Path(__file__).resolve().parent.parent)
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

import uvicorn
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.apps import A2AStarletteApplication
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
    Artifact,
    DataPart,
    FilePart,
    FileWithBytes,
    Part,
    TaskArtifactUpdateEvent,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
    TextPart,
)
from a2a.utils import new_agent_text_message
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_tar(data: bytes, dest: Path) -> None:
    """Extract a .tar.gz byte payload into *dest*."""
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        tar.extractall(dest)


def _read_description(work_dir: Path) -> str:
    """Return description.md text (searches recursively for the file)."""
    candidates = sorted(work_dir.rglob("description.md"))
    if candidates:
        return candidates[0].read_text(errors="replace")
    return "(no description.md found)"


# Maps regex pattern (case-insensitive) → (sklearn_scoring_string, human_label)
_METRIC_PATTERNS: list[tuple[str, str, str]] = [
    # Probability-based metrics (order matters — check specific phrases first)
    (r"mean\s+column[- ]?wise\s+(ROC\s+)?AUC", "roc_auc", "mean_column_auc"),
    (r"log\s*loss|logarithmic\s+loss", "neg_log_loss", "log_loss"),
    (r"area\s+under\s+(the\s+)?ROC\s+curve|AUC[\s-]?ROC|ROC\s+AUC", "roc_auc", "roc_auc"),
    (r"\bAUC\b", "roc_auc", "roc_auc"),
    # Error-based metrics
    (r"root\s+mean\s+squared\s+error|\bRMSE\b", "neg_root_mean_squared_error", "rmse"),
    (r"mean\s+squared\s+error|\bMSE\b", "neg_mean_squared_error", "mse"),
    (r"mean\s+absolute\s+error|\bMAE\b", "neg_mean_absolute_error", "mae"),
    (r"\bRMSLE\b|root\s+mean\s+squared\s+log", "neg_root_mean_squared_log_error", "rmsle"),
    # Classification metrics
    (r"classification\s+accuracy|\baccuracy\b", "accuracy", "accuracy"),
    (r"\bF1[\s-]?score\b|\bF1\b", "f1", "f1"),
    (r"Matthews\s+Correlation|MCC", "matthews_corrcoef", "mcc"),
    (r"Cohen.?s?\s+Kappa|\bkappa\b", "cohen_kappa_score", "kappa"),
    # R² last (very generic word)
    (r"\bR\s*²\b|\bR-?squared\b|\bR2\b", "r2", "r2"),
]


def _detect_competition_metric(
    description: str, eda_meta: dict
) -> tuple[str, str]:
    """Detect the competition's evaluation metric.

    Returns (sklearn_scoring_string, human_label).

    Strategy:
      1. Parse description.md for metric keywords
      2. Fall back to EDA-based inference (target_is_proba → roc_auc, etc.)
      3. Default: accuracy for classification, neg_root_mean_squared_error for regression
    """
    # --- Tier 1: parse description text ---
    if description and description != "(no description.md found)":
        # Look only in "Evaluation" / "Metric" sections if possible
        desc_lower = description.lower()
        # Try to isolate the evaluation section
        eval_section = description
        for marker in ("## evaluation", "## metric", "### metric", "### evaluation",
                        "evaluation\n", "metric\n"):
            idx = desc_lower.find(marker)
            if idx >= 0:
                # Take ~1500 chars from the marker (enough for the metric description)
                eval_section = description[idx:idx + 1500]
                break

        for pattern, sklearn_name, label in _METRIC_PATTERNS:
            if re.search(pattern, eval_section, re.IGNORECASE):
                return sklearn_name, label

    # --- Tier 2: EDA-based inference ---
    target_is_proba = eda_meta.get("target_is_proba", False)
    target_is_bool = eda_meta.get("target_is_bool", False)
    target_nunique = eda_meta.get("target_nunique", 0)
    target_dtype = eda_meta.get("target_dtype", "")
    columns = eda_meta.get("columns", {})
    target_col = eda_meta.get("target_col", "")

    is_binary = target_is_bool or (
        columns.get(target_col, {}).get("role") in ("BINARY_BOOL", "BINARY_NUMERIC")
    ) or (target_nunique == 2)
    is_multiclass = (not is_binary and target_nunique > 2
                     and (target_dtype.startswith("int") or target_dtype == "object"))
    is_multi_target = len(eda_meta.get("target_cols", [])) > 1

    if is_multi_target:
        return "roc_auc", "roc_auc"
    if target_is_proba:
        return "roc_auc", "roc_auc"
    if is_binary:
        return "accuracy", "accuracy"
    if is_multiclass:
        return "accuracy", "accuracy"
    # Regression
    return "neg_root_mean_squared_error", "rmse"


def _find_data_dir(work_dir: Path) -> Path:
    """Return the directory that contains description.md."""
    candidates = sorted(work_dir.rglob("description.md"))
    if candidates:
        return candidates[0].parent
    return work_dir


def _preload_data_context(data_dir: Path) -> str:
    """Pre-compute a data-context summary so the model skips exploration rounds.

    Returns a multi-line string containing:
    - Directory listing
    - train.csv schema (dtypes, nulls, shape, first 3 rows)
    - sample_submission.csv header and shape
    """
    import pandas as pd

    lines: list[str] = []

    # 1. Directory listing
    try:
        entries = sorted(p.name for p in data_dir.iterdir())
        lines.append(f"=== Data directory ({data_dir}) ===")
        lines.append("\n".join(f"  {e}" for e in entries))
    except Exception as exc:
        lines.append(f"(Could not list data directory: {exc})")

    # 2. train.csv schema
    train_path = data_dir / "train.csv"
    if train_path.exists():
        try:
            df = pd.read_csv(train_path, nrows=5)
            full_shape = sum(1 for _ in open(train_path)) - 1  # row count minus header
            lines.append(f"\n=== train.csv ({full_shape} rows, {len(df.columns)} columns) ===")
            lines.append("Columns and dtypes:")
            for col in df.columns:
                null_flag = ""
                # Check nulls on a larger sample for accuracy
                lines.append(f"  {col}: {df[col].dtype}")
            # First 3 rows
            lines.append(f"\nFirst 3 rows:\n{df.head(3).to_string()}")
            # Null counts from full file (read only first 1000 rows for speed)
            df_nulls = pd.read_csv(train_path, nrows=1000)
            null_counts = df_nulls.isnull().sum()
            has_nulls = null_counts[null_counts > 0]
            if len(has_nulls) > 0:
                lines.append(f"\nNull counts (first 1000 rows):")
                for col, cnt in has_nulls.items():
                    lines.append(f"  {col}: {cnt}")
        except Exception as exc:
            lines.append(f"(Could not read train.csv: {exc})")

    # 3. sample_submission.csv
    sample_path = data_dir / "sample_submission.csv"
    if sample_path.exists():
        try:
            samp = pd.read_csv(sample_path, nrows=3)
            samp_full = sum(1 for _ in open(sample_path)) - 1
            lines.append(f"\n=== sample_submission.csv ({samp_full} rows, {len(samp.columns)} columns) ===")
            lines.append(f"Columns: {list(samp.columns)}")
            lines.append(f"First 3 rows:\n{samp.head(3).to_string()}")
        except Exception as exc:
            lines.append(f"(Could not read sample_submission.csv: {exc})")

    # 4. test.csv shape
    test_path = data_dir / "test.csv"
    if test_path.exists():
        try:
            test_cols = pd.read_csv(test_path, nrows=1)
            test_rows = sum(1 for _ in open(test_path)) - 1
            lines.append(f"\n=== test.csv ({test_rows} rows, {len(test_cols.columns)} columns) ===")
            lines.append(f"Columns: {list(test_cols.columns)}")
        except Exception as exc:
            lines.append(f"(Could not read test.csv: {exc})")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Deterministic EDA gate — runs before planning, no LLM involved
# ---------------------------------------------------------------------------

def _run_eda(data_dir: Path) -> tuple[str, dict]:
    """Profile every CSV in *data_dir* and return a deterministic report string.

    Resilient to:
    - Encoding issues (utf-8 → latin1 → cp1252 → binary fallback)
    - Mixed-type / dirty columns (per-column try/except)
    - Large files (samples first 100k rows for speed, counts full rows cheaply)
    - Constant columns, ID columns, boolean-as-string columns
    - Numeric-stored-as-string columns ("1,234" / "$4.50")
    - Structured-string columns with detectable delimiters ("A/12/B")
    - Skewed numeric distributions

    Enrichments for feature engineering guidance:
    - Role tags: [ID], [CONSTANT], [BINARY], [BOOL_STR], [NUMERIC_AS_STR],
                 [STRUCTURED_STR delimiter=X], [HIGH_CARD], [CONTINUOUS], [CATEGORICAL]
    - Cross-file column diff (which columns are train-only = target candidates)
    - Skew flag for numeric columns (suggests log-transform when |skew| > 5)
    - Delimiter sniffing for structured strings
    - OOM prevention: explicit NEVER pd.get_dummies on flagged columns
    """
    import pandas as pd
    import numpy as np

    ENCODINGS = ["utf-8", "latin1", "cp1252"]
    SAMPLE_ROWS = 100_000
    HIGH_CARD_THRESHOLD = 50
    SKEW_THRESHOLD = 5.0
    NUMERIC_STR_THRESHOLD = 0.90   # fraction parseable as float → numeric-as-str
    DELIMITERS = ["/", "-", "_", "|", ":", ";", "."]

    def _read_resilient(path: Path) -> "pd.DataFrame | None":
        """Try multiple encodings; return None if all fail."""
        for enc in ENCODINGS:
            try:
                return pd.read_csv(path, nrows=SAMPLE_ROWS, encoding=enc, low_memory=False)
            except UnicodeDecodeError:
                continue
            except Exception:
                return None
        # Last resort: errors='replace'
        try:
            return pd.read_csv(
                path, nrows=SAMPLE_ROWS, encoding="utf-8", encoding_errors="replace",
                low_memory=False,
            )
        except Exception:
            return None

    def _count_rows(path: Path) -> int:
        """Count data rows (excluding header) without loading the file."""
        try:
            with open(path, encoding="utf-8", errors="replace") as fh:
                return max(0, sum(1 for _ in fh) - 1)
        except Exception:
            return -1

    def _sniff_delimiter(series: "pd.Series") -> "str | None":
        """Return the most consistent delimiter in a string series, or None."""
        sample = series.dropna().astype(str).head(200)
        if len(sample) < 10:
            return None
        for delim in DELIMITERS:
            counts = sample.str.count(re.escape(delim))
            # Consistent if >80% of values have at least 1 occurrence of the same count
            mode_count = int(counts.mode().iloc[0]) if len(counts.mode()) > 0 else 0
            if mode_count > 0 and (counts == mode_count).mean() > 0.80:
                return delim
        return None

    # Regex that matches any valid Python float after noise stripping.
    # Handles: integers, decimals, leading-dot, scientific notation, +/- prefix.
    # Applied AFTER stripping currency/thousands/percent/whitespace noise.
    _FLOAT_RE = re.compile(r'^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$')
    # Noise characters to strip before numeric check
    _NOISE_RE = re.compile(r'[\$,€£%\s_]')

    def _is_numeric_as_str(series: "pd.Series") -> bool:
        """True if an object column is ≥90% parseable as float.

        Uses two independent methods on the SAME random sample — both must
        agree at the threshold for a positive result (belt-and-suspenders):

        Method 1 — regex + float() try/except:
            Transparent, fast, explicit about what "parseable" means.
            Regex: ^[+-]?(\\d+\\.?\\d*|\\.\\d+)([eE][+-]?\\d+)?$
            Confirmed by Python's float() as the definitive ground truth.

        Method 2 — pd.to_numeric(..., errors='coerce'):
            Official pandas/C ground truth. Handles exotic float formats
            (e.g. hex floats, locale variants) that regex might miss.

        Sample size: 200 random non-null values (random_state=42 for
        reproducibility). At n=200 and 90% threshold, the 95% CI margin of
        error is ±4.2%, making the result statistically unambiguous.
        Head-of-file bias is avoided by random sampling.
        """
        nonnull = series.dropna().astype(str)
        if len(nonnull) == 0:
            return False

        # Draw up to 200 random values (reproducible)
        n = min(250, len(nonnull))
        sample = nonnull.sample(n=n, random_state=42) if len(nonnull) > n else nonnull

        # --- Strip noise characters from all sampled values once ---
        cleaned = sample.apply(lambda v: _NOISE_RE.sub("", v).strip())

        # --- Method 1: regex gate + float() ground truth ---
        m1_parseable = 0
        for val in cleaned:
            if not val or val in ("-", "+", "."):
                continue
            if not _FLOAT_RE.match(val):
                continue
            try:
                float(val)
                m1_parseable += 1
            except ValueError:
                pass
        m1_ratio = m1_parseable / len(sample)

        # --- Method 2: pd.to_numeric official ground truth ---
        m2_ratio = pd.to_numeric(cleaned, errors="coerce").notna().mean()

        # Both methods must independently agree at the threshold
        return m1_ratio >= NUMERIC_STR_THRESHOLD and m2_ratio >= NUMERIC_STR_THRESHOLD

    def _profile_column(
        col: str, series: "pd.Series", nrows: int,
    ) -> tuple[list[str], dict]:
        """Return (EDA text lines, structured metadata dict). Never raises."""
        out: list[str] = []
        meta: dict = {"name": col, "role": "UNKNOWN", "dtype": str(series.dtype),
                       "nunique": 0, "null_count": 0, "null_pct": 0.0}
        try:
            dtype = series.dtype
            nunique = int(series.nunique(dropna=False))
            null_count = int(series.isnull().sum())
            null_pct = 100.0 * null_count / len(series) if len(series) > 0 else 0.0
            meta.update(nunique=nunique, null_count=null_count, null_pct=null_pct)

            # --- Role classification ---
            roles: list[str] = []
            id_structure_hint: str | None = None

            if nunique <= 1:
                roles.append("CONSTANT → drop this column, zero predictive value")
                meta["role"] = "CONSTANT"
            else:
                n_sample = len(series)
                # ID detection: compare nunique against the SAMPLE size (not the
                # full file row count), because nunique is computed on the sample.
                # Float columns are NEVER flagged as IDs by cardinality alone —
                # continuous features are naturally all-unique; model context
                # (competition description) determines whether a float is an ID.
                is_id = (
                    n_sample > 10
                    and nunique == n_sample
                    and not pd.api.types.is_float_dtype(dtype)
                )
                if is_id:
                    roles.append("ID — unique per row, drop before modeling")
                    meta["role"] = "ID"
                    # Check if the ID itself encodes group/hierarchy structure
                    if not pd.api.types.is_numeric_dtype(dtype):
                        id_delim = _sniff_delimiter(series)
                        if id_delim:
                            nonnull_id = series.dropna()
                            sample_str = str(nonnull_id.iloc[0]) if len(nonnull_id) > 0 else "?"
                            n_parts = len(sample_str.split(id_delim))
                            meta["structured_id"] = {"delimiter": id_delim, "n_parts": n_parts, "example": sample_str}
                            id_structure_hint = (
                                f"    STRUCTURED ID: delimiter='{id_delim}' "
                                f"(e.g. '{sample_str}', {n_parts} parts) "
                                f"→ Extract at the TOP LEVEL of the script (NOT inside preprocess), "
                                f"BEFORE saving or dropping '{col}': "
                                f"train['_g'] = train['{col}'].str.split('{id_delim}').str[0]; "
                                f"train['GroupSize'] = train.groupby('_g')['{col}'].transform('count'); "
                                f"train['IsSolo'] = (train['GroupSize']==1).astype(int); "
                                f"train = train.drop(columns=['_g']); "
                                f"test['_g'] = test['{col}'].str.split('{id_delim}').str[0]; "
                                f"test['GroupSize'] = test.groupby('_g')['{col}'].transform('count'); "
                                f"test['IsSolo'] = (test['GroupSize']==1).astype(int); "
                                f"test = test.drop(columns=['_g']). "
                                f"THEN save '{col}' as IDs and drop it. Keep GroupSize and IsSolo."
                            )
                elif pd.api.types.is_bool_dtype(dtype):
                    roles.append("BINARY bool")
                    meta["role"] = "BINARY_BOOL"
                elif pd.api.types.is_numeric_dtype(dtype):
                    if nunique == 2:
                        roles.append("BINARY numeric")
                        meta["role"] = "BINARY_NUMERIC"
                    elif nunique <= 10:
                        roles.append(f"ORDINAL/CATEGORICAL ({nunique} values)")
                        meta["role"] = "ORDINAL"
                    else:
                        roles.append("CONTINUOUS")
                        meta["role"] = "CONTINUOUS"
                else:
                    # object / string dtype
                    str_values = series.dropna().astype(str)
                    bool_like = {"true", "false", "yes", "no", "t", "f", "0", "1"}
                    if nunique == 2 or (
                        nunique <= 4
                        and set(str_values.str.lower().unique()) <= bool_like
                    ):
                        roles.append("BOOL_STR → use .map({'True':1,'False':0,...}).fillna(0).astype(int)")
                        meta["role"] = "BOOL_STR"
                    elif _is_numeric_as_str(series):
                        roles.append("NUMERIC_AS_STR → strip noise chars then pd.to_numeric(..., errors='coerce')")
                        meta["role"] = "NUMERIC_AS_STR"
                    else:
                        delim = _sniff_delimiter(series)
                        avg_len = str_values.str.len().mean() if len(str_values) > 0 else 0
                        # Only flag as STRUCTURED_STR if there are enough unique values
                        # to make splitting worthwhile. A column with ≤20 unique values
                        # is a categorical feature whose string content happens to contain
                        # delimiter characters (e.g. 'TRAPPIST-1e', 'PSO J318.5-22') —
                        # splitting it produces meaningless sub-columns.
                        if delim and avg_len < 30 and nunique > 20:
                            split_sample = str_values.head(200).str.split(re.escape(delim), expand=True)
                            n_parts = split_sample.shape[1]
                            # Detect which parts are numeric (>50% of non-null values convert)
                            part_types: list[str] = []
                            for pi in range(n_parts):
                                part_col = split_sample[pi].dropna()
                                if len(part_col) == 0:
                                    part_types.append("categorical")
                                    continue
                                num_ok = pd.to_numeric(part_col, errors="coerce").notna().sum()
                                part_types.append("numeric" if num_ok > 0.5 * len(part_col) else "categorical")
                            roles.append(
                                f"STRUCTURED_STR delimiter='{delim}' (~{n_parts} parts) "
                                f"→ split into {n_parts} new feature columns, "
                                f"convert numeric parts with pd.to_numeric(..., errors='coerce')"
                            )
                            meta["role"] = "STRUCTURED_STR"
                            meta["structured_str"] = {"delimiter": delim, "n_parts": n_parts, "part_types": part_types}
                        elif avg_len > 50:
                            roles.append(
                                f"FREE_TEXT (avg_len={avg_len:.0f}) → drop or extract length feature only"
                            )
                            meta["role"] = "FREE_TEXT"
                        elif nunique > HIGH_CARD_THRESHOLD:
                            # Check for NAME_FORMAT: "Firstname Lastname" where last token
                            # has much lower cardinality than the full value → FamilySize signal.
                            # Use a small sample ONLY for the format check (median word count),
                            # then compute cardinality stats on the FULL series — otherwise the
                            # repeat_rate will be near-zero on a 500-row sample and the check fails.
                            name_hint: str | None = None
                            avg_token_count: float = 1.0
                            try:
                                fmt_sample = str_values.sample(
                                    min(500, len(str_values)), random_state=42
                                ) if len(str_values) > 500 else str_values
                                token_counts = fmt_sample.str.split(" ").str.len()
                                avg_token_count = float(token_counts.median())
                                if 1.8 <= avg_token_count <= 2.5:
                                    # Compute on FULL series, not sample
                                    last_tokens_full = str_values.str.split(" ").str[-1]
                                    last_nunique = int(last_tokens_full.nunique())
                                    compression = last_nunique / max(nunique, 1)
                                    repeat_rate = float(
                                        (last_tokens_full.value_counts() > 1).mean()
                                    )
                                    if compression < 0.50 and repeat_rate > 0.30:
                                        meta["name_format"] = True
                                        name_hint = (
                                            f"NAME_FORMAT ('Firstname Lastname') — last token has "
                                            f"~{last_nunique} unique values ({compression*100:.0f}% of full column), "
                                            f"{repeat_rate*100:.0f}% of family names repeat → "
                                            f"Extract at the TOP LEVEL (NOT inside preprocess), "
                                            f"BEFORE dropping '{col}': "
                                            f"train['_ln'] = train['{col}'].str.split(' ').str[-1].fillna('Unknown'); "
                                            f"train['FamilySize'] = train.groupby('_ln')['{col}'].transform('count'); "
                                            f"train = train.drop(columns=['_ln']); "
                                            f"test['_ln'] = test['{col}'].str.split(' ').str[-1].fillna('Unknown'); "
                                            f"test['FamilySize'] = test.groupby('_ln')['{col}'].transform('count'); "
                                            f"test = test.drop(columns=['_ln']). "
                                            f"LabelEncode or drop '{col}' itself."
                                        )
                            except Exception:
                                pass
                            if name_hint:
                                roles.append(
                                    f"HIGH_CARD ({nunique} unique) | {name_hint}"
                                )
                                meta["role"] = "HIGH_CARD"
                            elif 2.5 < avg_token_count <= 7 and avg_len <= 50:
                                # Short free text: 3–7 word strings — product titles, addresses,
                                # movie titles, short labels.  Upper bound ≤7 words keeps longer
                                # review excerpts from landing here instead of FREE_TEXT.
                                # LabelEncoding is safe but loses all semantic signal.
                                roles.append(
                                    f"SHORT_FREE_TEXT (avg_len={avg_len:.0f}, ~{avg_token_count:.1f} words/value) → "
                                    f"extract: df['{col}_len'] = df['{col}'].str.len(); "
                                    f"df['{col}_nwords'] = df['{col}'].str.split().str.len(); "
                                    f"then LabelEncode or drop '{col}' itself. "
                                    f"NEVER pd.get_dummies (OOM risk: {nunique} × {nrows} columns)"
                                )
                                meta["role"] = "SHORT_FREE_TEXT"
                            else:
                                roles.append(
                                    f"HIGH_CARD ({nunique} unique) → LabelEncoder ONLY, "
                                    f"NEVER pd.get_dummies (OOM risk: {nunique} × {nrows} columns)"
                                )
                                meta["role"] = "HIGH_CARD"
                        else:
                            roles.append(f"CATEGORICAL ({nunique} values) → LabelEncoder")
                            meta["role"] = "CATEGORICAL"

            role_str = " | ".join(roles)
            out.append(
                f"  [{col}]  dtype={dtype}  unique={nunique}  "
                f"null={null_count}({null_pct:.1f}%)  → {role_str}"
            )
            if id_structure_hint:
                out.append(id_structure_hint)

            # --- Stats ---
            if pd.api.types.is_numeric_dtype(dtype) and not pd.api.types.is_bool_dtype(dtype):
                nonnull = series.dropna()
                if len(nonnull) > 1:
                    try:
                        skew = float(nonnull.skew())
                        meta["skewed"] = abs(skew) > SKEW_THRESHOLD
                        skew_flag = (
                            f"  *** SKEWED (skew={skew:.2f}) → consider log1p transform ***"
                            if abs(skew) > SKEW_THRESHOLD else ""
                        )
                        out.append(
                            f"    range=[{nonnull.min():.4g}, {nonnull.max():.4g}]  "
                            f"mean={nonnull.mean():.4g}  std={nonnull.std():.4g}  "
                            f"skew={skew:.2f}"
                            + skew_flag
                        )
                    except Exception:
                        out.append(f"    range=[{series.min():.4g}, {series.max():.4g}]")
            elif dtype == object or str(dtype) == "string":
                top = series.value_counts(dropna=False).head(5)
                top_str = ", ".join(f"{str(v)!r}:{c}" for v, c in top.items())
                out.append(f"    top_values: {top_str}")

        except Exception as exc:
            out.append(f"  [{col}]  ERROR during profiling: {exc}")
        return out, meta

    # -----------------------------------------------------------------------
    lines: list[str] = []
    csv_files = sorted(f for f in data_dir.iterdir() if f.suffix.lower() == ".csv")
    if not csv_files:
        return "(no CSV files found in data directory)", {}

    # Collect column sets per file for cross-file diff
    file_columns: dict[str, list[str]] = {}
    loaded_dfs: dict[str, "pd.DataFrame"] = {}  # keep refs for post-analysis
    file_meta: dict[str, list[dict]] = {}  # structured metadata per file
    file_nrows: dict[str, int] = {}  # true row counts per file

    for csv_path in csv_files:
        lines.append(f"\n{'='*60}")
        lines.append(f"FILE: {csv_path.name}")
        lines.append(f"{'='*60}")

        df = _read_resilient(csv_path)
        if df is None:
            lines.append("  ERROR: could not read file with any supported encoding.")
            continue

        nrows_full = _count_rows(csv_path)
        if nrows_full < 0:
            nrows_full = len(df)
        sampled = nrows_full > SAMPLE_ROWS

        lines.append(
            f"Shape: ({nrows_full} rows, {len(df.columns)} cols)"
            + (" [stats based on first 100k rows]" if sampled else "")
        )
        lines.append(f"Columns: {list(df.columns)}")
        file_columns[csv_path.name] = list(df.columns)
        loaded_dfs[csv_path.name] = df
        file_meta[csv_path.name] = []
        file_nrows[csv_path.name] = nrows_full
        lines.append("")

        for col in df.columns:
            col_lines, col_meta = _profile_column(col, df[col], nrows_full)
            lines.extend(col_lines)
            file_meta[csv_path.name].append(col_meta)

    # Cross-file column diff summary (train-only columns are target candidates)
    if len(file_columns) >= 2:
        lines.append(f"\n{'='*60}")
        lines.append("CROSS-FILE COLUMN DIFF")
        lines.append(f"{'='*60}")
        all_sets = {name: set(cols) for name, cols in file_columns.items()}
        file_names = list(all_sets.keys())
        for i, fname in enumerate(file_names):
            others = set().union(*(all_sets[n] for n in file_names if n != fname))
            only_here = all_sets[fname] - others
            if only_here:
                lines.append(
                    f"  Columns ONLY in {fname} (not in other files): {sorted(only_here)}"
                )
                if "train" in fname.lower():
                    lines.append(
                        "  ^ These are likely TARGET or LEAKAGE columns — "
                        "drop them from features before fitting."
                    )

    # Feature engineering insights derived from the training file
    train_fname = next((n for n in loaded_dfs if "train" in n.lower()), None)
    # Fallback: if no "train" file, pick the largest non-test/non-submission CSV
    if train_fname is None:
        _test_sub = {n for n in loaded_dfs if "test" in n.lower() or "sample" in n.lower() or "submission" in n.lower()}
        _candidates = [(n, len(loaded_dfs[n])) for n in loaded_dfs if n not in _test_sub]
        if _candidates:
            train_fname = max(_candidates, key=lambda t: t[1])[0]
    test_fname: str | None = None
    target_col: str | None = None
    binary_cols: list[str] = []
    zero_inflated: list[str] = []
    train_only_cols: set[str] = set()
    if train_fname:
        train_df = loaded_dfs[train_fname]
        # Exclude target/leakage: columns present in train but NOT in the test file
        # Compare against test specifically (not sample_submission which also has the target)
        test_fname = next((n for n in file_columns if "test" in n.lower()), None)
        if test_fname:
            test_cols = set(file_columns[test_fname])
        else:
            other_files = [n for n in file_columns if n != train_fname and "submission" not in n.lower()]
            test_cols = set().union(*(set(file_columns[n]) for n in other_files)) if other_files else set()
        train_only_cols = set(file_columns[train_fname]) - test_cols
        insights: list[str] = []

        # Detect binary/boolean columns as interaction candidates
        # Skip train-only columns (they are the target or leakage)
        binary_cols: list[str] = []
        for col in train_df.columns:
            if col in train_only_cols:
                continue
            s = train_df[col]
            nu = s.nunique(dropna=True)
            if pd.api.types.is_bool_dtype(s.dtype) and nu == 2:
                binary_cols.append(col)
            elif s.dtype == object and nu <= 3:
                vals = set(s.dropna().astype(str).str.lower().unique())
                bool_like = {"true", "false", "yes", "no", "t", "f", "0", "1"}
                if vals <= bool_like:
                    binary_cols.append(col)

        if len(binary_cols) >= 2:
            pairs = [(binary_cols[i], binary_cols[j])
                     for i in range(len(binary_cols))
                     for j in range(i + 1, min(i + 3, len(binary_cols)))]
            pair_strs = ", ".join(f"'{a}' x '{b}'" for a, b in pairs[:4])
            insights.append(
                f"  INTERACTION CANDIDATES (binary columns): {pair_strs}\n"
                f"  → Create product features: df['A_x_B'] = df['A'] * df['B']"
            )

        # Detect zero-inflated numeric columns (>30% zeros) as candidates for
        # IsZero binary flags — especially useful for spending/usage columns
        zero_inflated: list[str] = []
        for col in train_df.columns:
            if col in train_only_cols:
                continue
            s = train_df[col]
            if pd.api.types.is_numeric_dtype(s.dtype) and not pd.api.types.is_bool_dtype(s.dtype):
                nonnull = s.dropna()
                if len(nonnull) > 10 and (nonnull == 0).mean() > 0.30:
                    zero_inflated.append(col)

        if zero_inflated:
            cols_str = ", ".join(f"'{c}'" for c in zero_inflated[:6])
            insights.append(
                f"  ZERO-INFLATED NUMERIC COLUMNS (>30% zeros): {cols_str}\n"
                f"  → Add binary flag: df['col_IsZero'] = (df['col'] == 0).astype(int)\n"
                f"  → If multiple spending cols exist, also add: "
                f"df['TotalSpend'] = df[[spending_cols]].fillna(0).sum(axis=1); "
                f"df['IsZeroSpend'] = (df['TotalSpend'] == 0).astype(int)"
            )

        # Target detection: identify the column present in train but NOT in test.
        # Priority: (1) binary/bool column, (2) single train-only column,
        # (3) column matching sample_submission header.
        target_col = next(
            (c for c in train_only_cols
             if train_df[c].dtype == bool
             or (train_df[c].nunique(dropna=True) == 2
                 and pd.api.types.is_numeric_dtype(train_df[c].dtype))),
            None
        )
        if target_col is None and len(train_only_cols) == 1:
            target_col = next(iter(train_only_cols))
        if target_col is None and len(train_only_cols) > 1:
            # Check sample_submission for the target column name
            sample_fname = next(
                (n for n in file_columns if "sample" in n.lower() or "submission" in n.lower()),
                None,
            )
            if sample_fname:
                sample_cols = set(file_columns[sample_fname])
                candidates = train_only_cols & sample_cols
                if len(candidates) == 1:
                    target_col = next(iter(candidates))
                elif len(candidates) > 1:
                    # Multi-target: sample_submission expects multiple predictions
                    # Preserve column order from sample_submission
                    sample_col_order = file_columns[sample_fname]
                    _multi_targets = [c for c in sample_col_order if c in candidates]
                    if _multi_targets:
                        target_col = _multi_targets[0]  # primary for compat
        if target_col is not None:
            try:
                _tgt = train_df[target_col]
                if _tgt.dtype == bool or set(_tgt.dropna().unique()) <= {True, False, 0, 1}:
                    y = _tgt.map({True: 1, False: 0, 1: 1, 0: 0}).fillna(0).astype(float)
                else:
                    y = pd.to_numeric(_tgt, errors='coerce').fillna(0).astype(float)
                corr_rows: list[tuple[float, str]] = []
                for col in train_df.columns:
                    if col in train_only_cols:
                        continue
                    try:
                        x = pd.to_numeric(train_df[col], errors='coerce')
                        if x.notna().sum() > 50:
                            r = float(x.corr(y))
                            if not (r != r):  # not NaN
                                corr_rows.append((abs(r), col, r))
                    except Exception:
                        pass
                corr_rows.sort(reverse=True)
                if corr_rows:
                    top_str = ", ".join(
                        f"'{c}' ({r:+.2f})"
                        for _, c, r in corr_rows[:6]
                    )
                    insights.append(
                        f"  TOP FEATURES BY CORRELATION WITH '{target_col}': {top_str}\n"
                        f"  → Higher |r| = stronger linear signal. "
                        f"Skewed columns benefit from log1p before correlation."
                    )
            except Exception:
                pass

        if insights:
            lines.append(f"\n{'='*60}")
            lines.append("FEATURE ENGINEERING INSIGHTS")
            lines.append(f"{'='*60}")
            for ins in insights:
                lines.append(ins)

    # ---- Build structured metadata for deterministic code generation ----
    # Use true row count (not capped sample) for train size
    _n_train = file_nrows.get(train_fname, 0) if train_fname else 0

    eda_meta: dict = {
        "train_file": train_fname,
        "test_file": test_fname,
        "sample_file": next((n for n in file_columns if "sample" in n.lower() or "submission" in n.lower()), None),
        "target_col": target_col,
        "target_is_bool": (
            loaded_dfs[train_fname][target_col].dtype == bool
            if train_fname and target_col and train_fname in loaded_dfs else False
        ),
        "target_nunique": (
            int(loaded_dfs[train_fname][target_col].nunique(dropna=True))
            if train_fname and target_col and train_fname in loaded_dfs else 0
        ),
        "target_dtype": (
            str(loaded_dfs[train_fname][target_col].dtype)
            if train_fname and target_col and train_fname in loaded_dfs else ""
        ),
        "id_col": None,
        "n_train_rows": _n_train,
        "target_is_proba": False,  # set below if sample_submission target is float
        "target_cols": [],  # multi-target: set below if sample_submission has >1 target
        "columns": {},  # col_name → meta dict (from training file)
        "train_only_cols": sorted(train_only_cols) if train_only_cols else [],
        "spend_cols": zero_inflated,
        "binary_interaction_pairs": (
            [(binary_cols[i], binary_cols[j])
             for i in range(len(binary_cols))
             for j in range(i + 1, min(i + 3, len(binary_cols)))][:4]
            if len(binary_cols) >= 2 else []
        ),
        "numeric_interaction_pairs": [],  # populated below for small numeric datasets
    }

    # Detect top numeric feature pairs for small datasets (≤30K rows, ≤30 features)
    if (train_fname and target_col and train_fname in loaded_dfs
            and _n_train <= 30_000):
        _tdf = loaded_dfs[train_fname]
        # Exclude target and train-only columns; id_cols not populated yet so
        # use the id_col from eda_meta instead
        _skip = set(train_only_cols) | {target_col}
        if eda_meta.get("id_col"):
            _skip.add(eda_meta["id_col"])
        _num_feats = [c for c in _tdf.select_dtypes(include="number").columns
                      if c not in _skip]
        if 4 <= len(_num_feats) <= 30 and target_col in _tdf.columns:
            _target_s = pd.to_numeric(_tdf[target_col], errors="coerce")
            _corrs = {}
            for c in _num_feats:
                _r = _tdf[c].corr(_target_s)
                if pd.notna(_r):
                    _corrs[c] = abs(_r)
            _ranked = sorted(_corrs, key=_corrs.get, reverse=True)[:8]
            _npairs = []
            for _i in range(len(_ranked)):
                for _j in range(_i + 1, len(_ranked)):
                    _npairs.append((_ranked[_i], _ranked[_j]))
            eda_meta["numeric_interaction_pairs"] = _npairs[:10]

    # Populate column metadata from the training file
    if train_fname and train_fname in file_meta:
        for cm in file_meta[train_fname]:
            eda_meta["columns"][cm["name"]] = cm

    # Identify the ID column from sample_submission (first column)
    sample_fname = eda_meta["sample_file"]
    if sample_fname and sample_fname in file_columns:
        _id_candidate = file_columns[sample_fname][0]
        # Don't use the target column as the ID column
        if _id_candidate != target_col:
            eda_meta["id_col"] = _id_candidate
        # Detect multi-target: sample_submission has >1 non-ID columns in train_only
        _sample_non_id = [c for c in file_columns[sample_fname] if c != _id_candidate]
        _multi_in_train = [c for c in _sample_non_id if c in train_only_cols]
        if len(_multi_in_train) > 1:
            eda_meta["target_cols"] = _multi_in_train
        # Detect probability-based scoring: sample_submission target is float in [0,1]
        # (distinguishes AUC-ROC probability from regression float targets)
        _proba_check_cols = _multi_in_train if len(_multi_in_train) > 1 else ([target_col] if target_col else [])
        for _pc in _proba_check_cols:
            if _pc in file_columns[sample_fname] and sample_fname in loaded_dfs:
                _sample_target = loaded_dfs[sample_fname][_pc]
                if (_sample_target.dtype in (float, "float64", "float32")
                        and _sample_target.min() >= 0 and _sample_target.max() <= 1):
                    eda_meta["target_is_proba"] = True
                    break

    return "\n".join(lines), eda_meta


# ---------------------------------------------------------------------------
# Deterministic code generator — Phase 0, zero LLM calls
# ---------------------------------------------------------------------------

_AGE_PATTERNS = re.compile(r'^(age|edad|alter|âge|leeftijd|yaş|wiek|年齢)$', re.IGNORECASE)


def _generate_setup_script(meta: dict, data_dir: str) -> str | None:
    """Generate a minimal data-loading script from EDA metadata.

    Returns a script that:
    - Loads train.csv and test.csv as raw DataFrames
    - Extracts y (target) and ids (submission IDs)
    - Sets DATA_DIR and task-type variables

    Does NOT apply any feature engineering, preprocessing, or modeling.
    The agent uses toolkit functions (fe_*, preprocess, etc.) for that.
    """
    train_file = meta.get("train_file")
    test_file = meta.get("test_file")
    target_col = meta.get("target_col")
    id_col = meta.get("id_col")
    target_is_bool = meta.get("target_is_bool", False)
    target_cols = meta.get("target_cols", [])
    target_nunique = meta.get("target_nunique", 0)
    target_dtype = meta.get("target_dtype", "")

    if not all([train_file, test_file, target_col, id_col]):
        return None

    is_binary = target_is_bool or target_nunique == 2
    is_multiclass = (not is_binary and target_nunique > 2
                     and (target_dtype.startswith("int") or target_dtype == "object"))
    task = "binary" if is_binary else ("multiclass" if is_multiclass else "regression")

    lines: list[str] = []
    w = lines.append

    w("import pandas as pd")
    w("import numpy as np")
    w("import os")
    w("")
    w(f"DATA_DIR = '{data_dir}'")
    w(f"train = pd.read_csv(os.path.join(DATA_DIR, '{train_file}'))")
    w(f"test = pd.read_csv(os.path.join(DATA_DIR, '{test_file}'))")
    w(f"ids = test['{id_col}'].copy()")
    w("")

    # Target extraction
    is_multi_target = len(target_cols) > 1
    if is_multi_target:
        tcols_str = str(target_cols)
        w(f"_target_cols = {tcols_str}")
        w(f"y = train[_target_cols].copy()")
        w(f"train = train.drop(columns=_target_cols)")
    elif target_is_bool:
        w(f"y = train['{target_col}'].map({{True: 1, False: 0, 'True': 1, 'False': 0}}).fillna(0).astype(int)")
        w(f"train = train.drop(columns=['{target_col}'])")
    else:
        w(f"y = train['{target_col}']")
        w(f"train = train.drop(columns=['{target_col}'])")

    w("")
    w(f"TASK = '{task}'")
    w(f"TARGET_COL = '{target_col}'")
    w(f"ID_COL = '{id_col}'")
    w(f"print(f'train: {{train.shape}}, test: {{test.shape}}, y: {{y.shape}}')")
    w(f"print(f'Task: {{TASK}}, Target: {{TARGET_COL}}, ID: {{ID_COL}}')")

    return "\n".join(lines)


def _build_phase0_fe_steps(meta: dict) -> list[dict]:
    """Build deterministic FE pipeline steps from EDA metadata.

    Returns a list of {"fn": "fe_xxx", "args": {...}} dicts suitable for
    _exec_fe_pipeline.  This mirrors what _generate_solution_script does
    in its subprocess, but runs in-process so the results stay in
    exec_globals and don't need to be recomputed by the LLM in Phase B.

    Rules of ML references:
      Rule #7  — Turn heuristics into features (mine Phase 0 logic)
      Rule #16 — Plan to launch and iterate (deterministic baseline)
      Rule #20 — Combine and modify existing features
    """
    columns = meta.get("columns", {})
    target_col = meta.get("target_col", "")
    id_col = meta.get("id_col", "")
    spend_cols = meta.get("spend_cols", [])

    steps: list[dict] = []

    # 1. Bool conversion  (Rule #7: mine heuristic — bool strings → int)
    bool_str_cols = [c for c, m in columns.items()
                     if m.get("role") == "BOOL_STR" and c != target_col]
    if bool_str_cols:
        steps.append({"fn": "fe_bool_convert", "args": {"cols": bool_str_cols}})

    # 2. Structured string split  (Rule #7: mine structured data)
    for col_name, cm in columns.items():
        if cm.get("role") == "STRUCTURED_STR" and "structured_str" in cm and col_name != id_col:
            steps.append({"fn": "fe_split_structured", "args": {
                "col": col_name,
                "delimiter": cm["structured_str"]["delimiter"],
            }})

    # 3. Group size from structured ID  (Rule #7: entity-level heuristic)
    id_meta = columns.get(id_col, {})
    if "structured_id" in id_meta:
        steps.append({"fn": "fe_group_size", "args": {
            "id_col": id_col,
            "delimiter": id_meta["structured_id"]["delimiter"],
        }})

    # 4. Spending aggregates  (Rule #20: combine features)
    _continuous_spend = [c for c in spend_cols
                         if columns.get(c, {}).get("role") == "CONTINUOUS"]
    if len(_continuous_spend) >= 2:
        steps.append({"fn": "fe_spending_aggs", "args": {"spend_cols": _continuous_spend}})

    # 5. Null indicators  (Rule #7: missingness is a heuristic signal)
    null_cols = [c for c, m in columns.items()
                 if m.get("null_pct", 0) > 1.0 and c != target_col
                 and m.get("role") not in ("ID", "FREE_TEXT", "SHORT_FREE_TEXT", "CONSTANT")]
    if null_cols:
        steps.append({"fn": "fe_null_indicators", "args": {"cols": null_cols}})

    # 6. Null row count  (Rule #7: row-level data quality heuristic)
    steps.append({"fn": "fe_null_row_count", "args": {}})

    # 7. Frequency encoding  (Rule #17: directly observed feature)
    cat_cols = [c for c, m in columns.items()
                if m.get("role") in ("CATEGORY", "BOOL_STR", "LOW_CARD_STR")
                and c != target_col and c != id_col]
    if cat_cols:
        steps.append({"fn": "fe_frequency_encode", "args": {"cols": cat_cols}})

    # 8. Log transform for skewed numeric cols  (Rule #20: modify features)
    num_cols = [c for c, m in columns.items()
                if m.get("role") in ("NUMERIC", "FLOAT", "INT", "SPENDING", "CONTINUOUS")
                and c != target_col and c != id_col]
    if num_cols:
        steps.append({"fn": "fe_log_transform", "args": {"cols": num_cols}})

    # 9. Binning for continuous cols  (Rule #20: discretization)
    # Quantile bins capture non-linear thresholds (age brackets, spend tiers)
    # without dataset-specific domain knowledge.
    for c in num_cols[:6]:
        steps.append({"fn": "fe_binning", "args": {"col": c, "bins": 5}})

    # 10. Drop constant columns  (Rule #22: clean up unused features)
    steps.append({"fn": "fe_drop_constant", "args": {"threshold": 0.99}})

    return steps


def _build_kitchen_sink_fe_steps(meta: dict) -> list[dict]:
    """Build ADVANCED FE steps for the kitchen-sink hint.

    These are the transforms that go BEYOND the basic Phase 0 FE.
    They run on top of the already-processed data from _build_phase0_fe_steps.

    Rules of ML references:
      Rule #16 — Launch and iterate (throw many features at the model)
      Rule #20 — Combine and modify existing features
      Rule #22 — Clean up features you are no longer using
    """
    columns = meta.get("columns", {})
    target_col = meta.get("target_col", "")
    id_col = meta.get("id_col", "")

    steps: list[dict] = []

    cat_cols = [c for c, m in columns.items()
                if m.get("role") in ("CATEGORY", "BOOL_STR", "LOW_CARD_STR")
                and c != target_col and c != id_col]
    num_cols = [c for c, m in columns.items()
                if m.get("role") in ("NUMERIC", "FLOAT", "INT", "SPENDING", "CONTINUOUS")
                and c != target_col and c != id_col]

    # 1. Target encoding (highest impact for tabular)
    if cat_cols:
        steps.append({"fn": "fe_target_encode", "args": {"cols": cat_cols[:8]}})

    # 2. Count encoding
    if cat_cols:
        steps.append({"fn": "fe_count_encode", "args": {"cols": cat_cols[:8]}})

    # 3. Rank transform
    if num_cols:
        steps.append({"fn": "fe_rank_transform", "args": {"cols": num_cols[:10]}})

    # 4. Ratios of top numeric pairs
    if len(num_cols) >= 2:
        _pairs = [(num_cols[i], num_cols[j])
                  for i in range(min(len(num_cols), 6))
                  for j in range(i + 1, min(len(num_cols), 6))]
        steps.append({"fn": "fe_ratios", "args": {"col_pairs": _pairs}})

    # 5. Interactions
    if len(num_cols) >= 2:
        steps.append({"fn": "fe_interactions", "args": {"cols": num_cols[:6]}})

    # 6. Row stats
    if len(num_cols) >= 2:
        steps.append({"fn": "fe_row_stats", "args": {"cols": num_cols[:10]}})

    # 7. Categorical cross
    if len(cat_cols) >= 2:
        _cat_pairs = [(cat_cols[i], cat_cols[j])
                      for i in range(min(len(cat_cols), 3))
                      for j in range(i + 1, min(len(cat_cols), 3))]
        steps.append({"fn": "fe_categorical_cross", "args": {"col_pairs": _cat_pairs}})

    # 8. Polynomial (top 3 numeric)  (Rule #20: combine features)
    if len(num_cols) >= 2:
        steps.append({"fn": "fe_polynomial", "args": {"cols": num_cols[:3]}})

    # 9. Power transform
    if num_cols:
        steps.append({"fn": "fe_power_transform", "args": {"cols": num_cols[:8]}})

    # 10. Drop constant at the end
    steps.append({"fn": "fe_drop_constant", "args": {"threshold": 0.99}})

    return steps


def _generate_solution_script(meta: dict, data_dir: str) -> str | None:
    """Generate a complete Python ML script from EDA metadata.

    Returns a runnable Python script string, or None if metadata is
    insufficient (missing train/test/target/id).

    This is the "Phase 0" strategy — no LLM involvement.  The script
    handles: data loading, group/family feature extraction, boolean
    conversion, structured-string splitting, imputation, encoding,
    model training (4-model ensemble for binary classification, LGBM
    for everything else), and submission writing.
    """
    train_file = meta.get("train_file")
    test_file = meta.get("test_file")
    target_col = meta.get("target_col")
    id_col = meta.get("id_col")
    columns = meta.get("columns", {})

    if not all([train_file, test_file, target_col, id_col]):
        return None

    target_is_bool = meta.get("target_is_bool", False)
    target_is_proba = meta.get("target_is_proba", False)
    target_cols = meta.get("target_cols", [])  # multi-target list
    target_nunique = meta.get("target_nunique", 0)
    target_dtype = meta.get("target_dtype", "")
    spend_cols = meta.get("spend_cols", [])
    interaction_pairs = meta.get("binary_interaction_pairs", [])
    numeric_interaction_pairs = meta.get("numeric_interaction_pairs", [])
    n_train = meta.get("n_train_rows", 0)

    # Determine task type
    is_binary = target_is_bool or (
        columns.get(target_col, {}).get("role") in ("BINARY_BOOL", "BINARY_NUMERIC")
    ) or (target_nunique == 2)
    is_multiclass = (not is_binary and target_nunique > 2
                     and (target_dtype.startswith("int") or target_dtype == "object"))

    # Classify columns by role
    id_cols: list[str] = []
    bool_str_cols: list[str] = []
    structured_str_cols: list[dict] = []  # {"name", "delimiter", "n_parts"}
    categorical_cols: list[str] = []
    continuous_cols: list[str] = []
    high_card_cols: list[str] = []
    drop_cols: list[str] = []  # CONSTANT, FREE_TEXT, SHORT_FREE_TEXT
    structured_id_info: dict | None = None
    name_format_col: str | None = None

    for col_name, cm in columns.items():
        role = cm.get("role", "UNKNOWN")
        if col_name == target_col:
            continue  # handled separately
        if role == "ID":
            id_cols.append(col_name)
            if "structured_id" in cm:
                structured_id_info = cm["structured_id"]
                structured_id_info["col"] = col_name
        elif role == "BOOL_STR":
            bool_str_cols.append(col_name)
        elif role == "BINARY_BOOL":
            # Already bool dtype — no conversion needed, but track for features
            pass
        elif role == "BINARY_NUMERIC":
            pass
        elif role == "STRUCTURED_STR" and "structured_str" in cm:
            structured_str_cols.append({
                "name": col_name,
                "delimiter": cm["structured_str"]["delimiter"],
                "n_parts": cm["structured_str"]["n_parts"],
                "part_types": cm["structured_str"].get("part_types", []),
            })
        elif role == "CATEGORICAL" or role == "ORDINAL":
            categorical_cols.append(col_name)
        elif role == "CONTINUOUS":
            continuous_cols.append(col_name)
        elif role == "HIGH_CARD":
            if cm.get("name_format"):
                name_format_col = col_name
            high_card_cols.append(col_name)
        elif role in ("CONSTANT", "FREE_TEXT", "SHORT_FREE_TEXT"):
            drop_cols.append(col_name)
        elif role == "NUMERIC_AS_STR":
            continuous_cols.append(col_name)

    # ---- Detect fixed-length HIGH_CARD string columns for char-split ----
    char_split_cols: list[tuple[str, int]] = []  # (col_name, str_len)
    for hc in list(high_card_cols):
        cm = columns.get(hc, {})
        # Check if EDA sample shows fixed-length strings (all same str len)
        if cm.get("dtype") in ("object", "str", "string"):
            # We need the actual data to verify fixed-length; use the meta hint
            # If sample values are available, check them; otherwise use heuristic:
            # HIGH_CARD with no name_format and no structured_id is likely encoded
            if not cm.get("name_format") and not cm.get("structured_id"):
                char_split_cols.append((hc, 0))  # str_len=0 means detect at runtime

    # ---- Detect lat/lon coordinate pairs for distance features ----
    latlon_pairs: list[tuple[str, str, str, str]] = []  # (lat1, lon1, lat2, lon2)
    _lat_re = re.compile(r'lat', re.IGNORECASE)
    _lon_re = re.compile(r'lon', re.IGNORECASE)
    lat_cols = [c for c in continuous_cols if _lat_re.search(c)]
    lon_cols = [c for c in continuous_cols if _lon_re.search(c)]
    if len(lat_cols) >= 2 and len(lon_cols) >= 2:
        # Try to pair them: pickup/dropoff, start/end, origin/destination
        for prefix_a, prefix_b in [("pickup", "dropoff"), ("start", "end"), ("origin", "dest")]:
            lat_a = [c for c in lat_cols if prefix_a in c.lower()]
            lat_b = [c for c in lat_cols if prefix_b in c.lower()]
            lon_a = [c for c in lon_cols if prefix_a in c.lower()]
            lon_b = [c for c in lon_cols if prefix_b in c.lower()]
            if lat_a and lat_b and lon_a and lon_b:
                latlon_pairs.append((lat_a[0], lon_a[0], lat_b[0], lon_b[0]))
                break

    # ---- Detect datetime string columns for time features ----
    datetime_cols: list[str] = []
    for s_info in structured_str_cols:
        col_n = s_info["name"]
        if any(kw in col_n.lower() for kw in ("datetime", "date", "time", "timestamp")):
            datetime_cols.append(col_n)

    # ---- Guard: skip if too few usable features (e.g. NLP datasets) ----
    usable_features = (
        bool_str_cols + categorical_cols + continuous_cols
        + [s["name"] for s in structured_str_cols]
    )
    if len(usable_features) < 1:
        logger.info("Phase 0: skipping — no usable tabular features (NLP/text dataset?)")
        return None

    # ---- Build the script ----
    lines: list[str] = []
    w = lines.append  # shorthand

    # -- Imports --
    w("import os")
    w("import gc")
    w("import pandas as pd")
    w("import numpy as np")
    w("from sklearn.preprocessing import LabelEncoder")
    w("from sklearn.ensemble import VotingClassifier, ExtraTreesClassifier, ExtraTreesRegressor")
    w("from sklearn.model_selection import StratifiedKFold, cross_val_score")
    w("from lightgbm import LGBMClassifier, LGBMRegressor")
    w("from xgboost import XGBClassifier, XGBRegressor")
    w("from catboost import CatBoostClassifier, CatBoostRegressor")
    w("")
    w("# ---- CPU throttle: use ~75% of available cores, minimum 1 ----")
    w("MAX_JOBS = -1  # use all cores for Phase 0 (deterministic, no memory contention)")
    w("")
    # -- Memory detection for adaptive scaling --
    w("# ---- Detect available memory for adaptive scaling ----")
    w("def _avail_mem_gb():")
    w("    try:")
    w("        with open('/proc/meminfo') as f:")
    w("            for line in f:")
    w("                if line.startswith('MemAvailable:'):")
    w("                    return int(line.split()[1]) / (1024 * 1024)")
    w("    except Exception:")
    w("        pass")
    w("    return 8.0  # safe fallback")
    w("")
    w("_MEM_GB = _avail_mem_gb()")
    w("print(f'Available memory: {_MEM_GB:.1f} GB')")
    w("")

    # -- Data loading (adaptive row cap based on available memory) --
    w(f"DATA_DIR = '{data_dir}'")
    if n_train > 100_000:
        # Adaptive row cap: scale with available memory
        w("# Adaptive row cap based on available memory")
        w("if _MEM_GB >= 32:")
        w(f"    _MAX_ROWS = min({n_train}, 5_000_000)")
        w("elif _MEM_GB >= 16:")
        w(f"    _MAX_ROWS = min({n_train}, 2_000_000)")
        w("elif _MEM_GB >= 8:")
        w(f"    _MAX_ROWS = min({n_train}, 1_000_000)")
        w("else:")
        w(f"    _MAX_ROWS = min({n_train}, 500_000)")
        w("print(f'Row cap: {_MAX_ROWS} (from {" + str(n_train) + "} total)')")
        w(f"train = pd.read_csv(f'{{DATA_DIR}}/{train_file}', nrows=_MAX_ROWS)")
    else:
        w(f"train = pd.read_csv(f'{{DATA_DIR}}/{train_file}')")
    w(f"test  = pd.read_csv(f'{{DATA_DIR}}/{test_file}')")
    w("")

    # -- Target extraction --
    is_multi_target = len(target_cols) > 1
    w(f"# ---- Target extraction ----")
    if is_multi_target:
        tcols_str = str(target_cols)
        w(f"_target_cols = {tcols_str}")
        w(f"y_multi = train[_target_cols].copy()")
        w(f"train = train.drop(columns=_target_cols)")
    elif target_is_bool:
        w(f"y = train['{target_col}'].map({{True: 1, False: 0, 'True': 1, 'False': 0}}).fillna(0).astype(int)")
        w(f"train = train.drop(columns=['{target_col}'])")
    else:
        w(f"y = train['{target_col}']")
        w(f"train = train.drop(columns=['{target_col}'])")
    w("")

    # -- Regression target outlier removal: log-transform for stable RMSE --
    is_regression = not is_binary and not is_multiclass and not is_multi_target
    if is_regression:
        w("# ---- Log-transform regression target for stable RMSE ----")
        w("_y_has_negatives = (y < 0).any()")
        w("if not _y_has_negatives:")
        w("    y = np.log1p(y)")
        w("    print('Target log-transformed (log1p)')")
        w("")

    # -- Drop other train-only columns (targets/leakage not in test) --
    train_only = meta.get("train_only_cols", [])
    if is_multi_target:
        # For multi-target, drop all train-only cols not already extracted as targets
        extra_train_only = [c for c in train_only if c not in target_cols]
    else:
        extra_train_only = [c for c in train_only if c != target_col]
    if extra_train_only:
        drop_list_str = ", ".join(f"'{c}'" for c in extra_train_only)
        w(f"# ---- Drop other train-only columns (not in test) ----")
        w(f"train = train.drop(columns=[{drop_list_str}], errors='ignore')")
        w("")

    # -- Feature engineering: STRUCTURED ID (GroupSize, IsSolo) --
    if structured_id_info:
        sid_col = structured_id_info["col"]
        sid_delim = structured_id_info["delimiter"]
        w(f"# ---- Group features from {sid_col} (BEFORE dropping) ----")
        w(f"train['_g'] = train['{sid_col}'].str.split('{sid_delim}').str[0]")
        w(f"train['GroupSize'] = train.groupby('_g')['{sid_col}'].transform('count')")
        w(f"train['IsSolo'] = (train['GroupSize'] == 1).astype(int)")
        w(f"train = train.drop(columns=['_g'])")
        w(f"test['_g'] = test['{sid_col}'].str.split('{sid_delim}').str[0]")
        w(f"test['GroupSize'] = test.groupby('_g')['{sid_col}'].transform('count')")
        w(f"test['IsSolo'] = (test['GroupSize'] == 1).astype(int)")
        w(f"test = test.drop(columns=['_g'])")
        w("")

    # -- Feature engineering: NAME_FORMAT (FamilySize) --
    if name_format_col:
        w(f"# ---- FamilySize from {name_format_col} (BEFORE dropping) ----")
        w(f"train['_ln'] = train['{name_format_col}'].str.split(' ').str[-1].fillna('Unknown')")
        w(f"train['FamilySize'] = train.groupby('_ln')['{name_format_col}'].transform('count')")
        w(f"train = train.drop(columns=['_ln'])")
        w(f"test['_ln'] = test['{name_format_col}'].str.split(' ').str[-1].fillna('Unknown')")
        w(f"test['FamilySize'] = test.groupby('_ln')['{name_format_col}'].transform('count')")
        w(f"test = test.drop(columns=['_ln'])")
        w("")

    # -- Character-split HIGH_CARD columns (before dropping) --
    _char_split_names = {c for c, _ in char_split_cols}
    if char_split_cols:
        w("# ---- Character-split fixed-length string columns ----")
        for hc_name, _ in char_split_cols:
            w(f"# Split '{hc_name}' into per-character features")
            w(f"_sl = int(train['{hc_name}'].dropna().str.len().mode().iloc[0])")
            w(f"for _i in range(_sl):")
            w(f"    train['{hc_name}_c' + str(_i)] = train['{hc_name}'].str[_i]")
            w(f"    test['{hc_name}_c' + str(_i)]  = test['{hc_name}'].str[_i]")
            w(f"train = train.drop(columns=['{hc_name}'])")
            w(f"test  = test.drop(columns=['{hc_name}'])")
        w("")

    # -- Save IDs and drop ID/HIGH_CARD/free-text columns --
    w(f"# ---- Save IDs and drop non-feature columns ----")
    w(f"train_ids = train['{id_col}']")
    w(f"test_ids  = test['{id_col}']")
    # Exclude char-split cols from drop (already handled above)
    _remaining_hc = [c for c in high_card_cols if c not in _char_split_names]
    all_drop = sorted(set(id_cols + _remaining_hc + drop_cols))
    if all_drop:
        drop_list_str = ", ".join(f"'{c}'" for c in all_drop)
        w(f"train = train.drop(columns=[{drop_list_str}], errors='ignore')")
        w(f"test  = test.drop(columns=[{drop_list_str}], errors='ignore')")
    w("")

    # -- Boolean string conversion (keep NaN → let preprocessing impute) --
    if bool_str_cols:
        w(f"# ---- Boolean string columns ----")
        w(f"_bool_map = {{'True': 1, 'False': 0, True: 1, False: 0, 'true': 1, 'false': 0}}")
        for bc in bool_str_cols:
            w(f"train['{bc}'] = train['{bc}'].map(_bool_map)")
            w(f"test['{bc}']  = test['{bc}'].map(_bool_map)")
        w("")

    # -- Null indicators for columns that will be split/dropped --
    _struct_with_nulls = [ss["name"] for ss in structured_str_cols
                          if columns.get(ss["name"], {}).get("null_pct", 0) > 1.0]
    if _struct_with_nulls:
        w("# ---- Null indicators for structured columns (before split) ----")
        w("for df in [train, test]:")
        for sn in _struct_with_nulls:
            w(f"    df['{sn}_isNull'] = df['{sn}'].isna().astype(int)")
        w("")

    # -- Structured string splitting --
    if structured_str_cols:
        w(f"# ---- Split structured-string columns ----")
        for ss in structured_str_cols:
            sname = ss["name"]
            # Skip datetime columns — handled separately below
            if sname in datetime_cols:
                continue
            sdelim = ss["delimiter"]
            n_parts = ss["n_parts"]
            part_types = ss.get("part_types", ["numeric"] * n_parts)
            part_names = [f"{sname}_p{i}" for i in range(n_parts)]
            part_list_str = str(part_names)
            w(f"_parts = train['{sname}'].str.split('{sdelim}', expand=True)")
            w(f"_parts.columns = {part_list_str}")
            w(f"train = pd.concat([train, _parts], axis=1)")
            w(f"_parts = test['{sname}'].str.split('{sdelim}', expand=True)")
            w(f"_parts.columns = {part_list_str}")
            w(f"test = pd.concat([test, _parts], axis=1)")
            # Only convert parts that are actually numeric; leave categorical as string
            for i, pn in enumerate(part_names):
                if i < len(part_types) and part_types[i] == "numeric":
                    w(f"train['{pn}'] = pd.to_numeric(train['{pn}'], errors='coerce')")
                    w(f"test['{pn}']  = pd.to_numeric(test['{pn}'], errors='coerce')")
            w(f"train = train.drop(columns=['{sname}'])")
            w(f"test  = test.drop(columns=['{sname}'])")
        w("")

    # -- NUMERIC_AS_STR conversion --
    numeric_as_str = [c for c, cm in columns.items()
                      if cm.get("role") == "NUMERIC_AS_STR" and c != target_col]
    if numeric_as_str:
        w(f"# ---- Numeric-as-string columns ----")
        for col_name in numeric_as_str:
            w(f"train['{col_name}'] = pd.to_numeric(train['{col_name}'].astype(str).str.replace(r'[\\$,€£%\\s_]', '', regex=True), errors='coerce')")
            w(f"test['{col_name}']  = pd.to_numeric(test['{col_name}'].astype(str).str.replace(r'[\\$,€£%\\s_]', '', regex=True), errors='coerce')")
        w("")

    # -- Age features (metadata-driven: detect any continuous column with age-like name) --
    age_col_name: str | None = None
    for col_name, cm in columns.items():
        if col_name == target_col:
            continue
        if cm.get("role") in ("CONTINUOUS", "ORDINAL") and _AGE_PATTERNS.match(col_name):
            age_col_name = col_name
            break
    if age_col_name:
        w(f"# ---- Age-derived features ({age_col_name}) ----")
        w(f"for df in [train, test]:")
        w(f"    df['IsChild']  = (df['{age_col_name}'] < 13).astype(int)")
        w(f"    df['IsSenior'] = (df['{age_col_name}'] > 60).astype(int)")
        w("")

    # -- Spending features --
    # Only use truly continuous zero-inflated columns, not binary indicators
    _continuous_spend = [c for c in spend_cols
                         if columns.get(c, {}).get("role") == "CONTINUOUS"]
    if len(_continuous_spend) >= 2:
        cols_str = str(_continuous_spend)
        w(f"# ---- Spending aggregate features ----")
        w(f"_spend_cols = {cols_str}")
        w(f"for df in [train, test]:")
        w(f"    df['TotalSpend']  = df[_spend_cols].fillna(0).sum(axis=1)")
        w(f"    df['IsZeroSpend'] = (df['TotalSpend'] == 0).astype(int)")
        if structured_id_info:
            w(f"    df['SpendPerPerson'] = df['TotalSpend'] / df['GroupSize'].clip(lower=1)")
        w(f"    for sc in _spend_cols:")
        w(f"        df[sc + '_log'] = np.log1p(df[sc].fillna(0))")
        w("")

        # -- NEW: Spending ratios (col / TotalSpend) --
        w(f"# ---- Spending ratio features ----")
        w(f"for df in [train, test]:")
        w(f"    _ts = df['TotalSpend'].clip(lower=1e-9)")
        w(f"    for sc in _spend_cols:")
        w(f"        df[sc + '_ratio'] = (df[sc].fillna(0) / _ts).replace([np.inf, -np.inf], 0).fillna(0)")
        w("")

        # -- NEW: Row stats across spending cols --
        w(f"# ---- Row-wise stats across spending columns ----")
        w(f"_spend_sub = train[_spend_cols].fillna(0).astype(float)")
        w(f"_spend_sub_test = test[_spend_cols].fillna(0).astype(float)")
        w(f"for df, sub in [(train, _spend_sub), (test, _spend_sub_test)]:")
        w(f"    df['spend_mean'] = sub.mean(axis=1)")
        w(f"    df['spend_std']  = sub.std(axis=1).fillna(0)")
        w(f"    df['spend_max']  = sub.max(axis=1)")
        w(f"    df['spend_min']  = sub.min(axis=1)")
        w("")

    # -- Interaction features --
    if interaction_pairs:
        w(f"# ---- Interaction features ----")
        for a, b in interaction_pairs:
            safe_name = f"{a}_x_{b}"
            w(f"for df in [train, test]:")
            w(f"    df['{safe_name}'] = pd.to_numeric(df['{a}'], errors='coerce').fillna(0) * pd.to_numeric(df['{b}'], errors='coerce').fillna(0)")
        w("")

    # -- Numeric pairwise product features (small datasets) --
    if numeric_interaction_pairs:
        w("# ---- Numeric pairwise product features ----")
        w("for df in [train, test]:")
        for a, b in numeric_interaction_pairs:
            safe_name = f"{a}_x_{b}"
            w(f"    df['{safe_name}'] = df['{a}'].fillna(0) * df['{b}'].fillna(0)")
        w("")

    # -- Null indicator features --
    _split_dropped = {ss["name"] for ss in structured_str_cols}
    null_cols = [c for c, cm in columns.items()
                 if cm.get("null_pct", 0) > 1.0 and c != target_col
                 and c not in id_cols and c not in high_card_cols
                 and c not in drop_cols and c not in _split_dropped]
    if null_cols:
        w("# ---- Null indicator features ----")
        w("for df in [train, test]:")
        for nc in null_cols:
            w(f"    df['{nc}_isNull'] = df['{nc}'].isna().astype(int)")
        w("")

    # -- NEW: Null row count (total nulls per row — captures data quality pattern) --
    w("# ---- Null row count ----")
    w("for df in [train, test]:")
    w("    df['null_count'] = df.isnull().sum(axis=1)")
    w("")

    # -- Lat/lon distance features (Haversine) --
    if latlon_pairs:
        # Remove training rows with unreasonable lat/lon (keep valid Earth coords)
        w("# ---- Geographic outlier removal (training only) ----")
        w("_n_before = len(train)")
        all_geo_cols = set()
        for lat1, lon1, lat2, lon2 in latlon_pairs:
            all_geo_cols.update([lat1, lon1, lat2, lon2])
        lat_geo = [c for c in all_geo_cols if _lat_re.search(c)]
        lon_geo = [c for c in all_geo_cols if _lon_re.search(c)]
        for lc in lat_geo:
            w(f"train = train[train['{lc}'].between(-90, 90)]")
        for lc in lon_geo:
            w(f"train = train[train['{lc}'].between(-180, 180)]")
        if is_multi_target:
            w("y_multi = y_multi.loc[train.index]")
            w("y_multi = y_multi.reset_index(drop=True)")
        else:
            w("y = y.loc[train.index]")
            w("y = y.reset_index(drop=True)")
        w("train = train.reset_index(drop=True)")
        w("print(f'Geographic filter: {_n_before} -> {len(train)} rows')")
        w("")

        w("# ---- Haversine distance features ----")
        w("def _haversine(lat1, lon1, lat2, lon2):")
        w("    R = 6371  # Earth radius in km")
        w("    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])")
        w("    dlat = lat2 - lat1")
        w("    dlon = lon2 - lon1")
        w("    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2")
        w("    return 2 * R * np.arcsin(np.sqrt(a))")
        w("")
        for i, (lat1, lon1, lat2, lon2) in enumerate(latlon_pairs):
            dist_col = f"_dist_{i}" if len(latlon_pairs) > 1 else "_distance"
            w("for df in [train, test]:")
            w(f"    df['{dist_col}'] = _haversine(df['{lat1}'], df['{lon1}'], df['{lat2}'], df['{lon2}'])")
            w(f"    df['{dist_col}'] = df['{dist_col}'].replace([np.inf, -np.inf], np.nan).fillna(0)")
        w("")

    # -- Datetime features --
    if datetime_cols:
        w("# ---- Datetime features ----")
        for dtcol in datetime_cols:
            w("for df in [train, test]:")
            w(f"    _dt = pd.to_datetime(df['{dtcol}'], errors='coerce')")
            w(f"    df['{dtcol}_hour'] = _dt.dt.hour")
            w(f"    df['{dtcol}_dayofweek'] = _dt.dt.dayofweek")
            w(f"    df['{dtcol}_month'] = _dt.dt.month")
            w(f"    df['{dtcol}_year'] = _dt.dt.year")
            w(f"train = train.drop(columns=['{dtcol}'], errors='ignore')")
            w(f"test = test.drop(columns=['{dtcol}'], errors='ignore')")
        w("")

    # -- NEW: Cyclical encoding for datetime-derived columns --
    if datetime_cols:
        w("# ---- Cyclical encoding for periodic datetime features ----")
        for dtcol in datetime_cols:
            for feat, period in [("hour", 24), ("dayofweek", 7), ("month", 12)]:
                col_name = f"{dtcol}_{feat}"
                w(f"for df in [train, test]:")
                w(f"    if '{col_name}' in df.columns:")
                w(f"        _v = df['{col_name}'].fillna(0).astype(float)")
                w(f"        df['{col_name}_sin'] = np.sin(2 * np.pi * _v / {period})")
                w(f"        df['{col_name}_cos'] = np.cos(2 * np.pi * _v / {period})")
        w("")

    # -- NEW: Log transform for skewed non-spend continuous columns --
    _non_spend_continuous = [c for c in continuous_cols
                             if c not in spend_cols and c != target_col]
    if _non_spend_continuous:
        w("# ---- Log transform for skewed continuous columns ----")
        w("for df in [train, test]:")
        for cc in _non_spend_continuous:
            w(f"    if '{cc}' in df.columns:")
            w(f"        df['{cc}_log'] = np.log1p(df['{cc}'].clip(lower=0).fillna(0))")
        w("")

    # -- NEW: Rank transform for continuous columns (robust to outliers) --
    _rank_candidates = continuous_cols[:]  # all continuous cols
    if _rank_candidates:
        w("# ---- Rank transform for continuous columns ----")
        for rc in _rank_candidates:
            if rc == target_col:
                continue
            w(f"if '{rc}' in train.columns:")
            w(f"    train['{rc}_rank'] = train['{rc}'].rank(pct=True, method='average').fillna(0.5)")
            w(f"    _sorted_{rc.replace(' ', '_')} = np.sort(train['{rc}'].dropna().values)")
            w(f"    _n_{rc.replace(' ', '_')} = len(_sorted_{rc.replace(' ', '_')})")
            w(f"    if _n_{rc.replace(' ', '_')} > 0 and '{rc}' in test.columns:")
            w(f"        test['{rc}_rank'] = test['{rc}'].apply(lambda x: np.searchsorted(_sorted_{rc.replace(' ', '_')}, x) / _n_{rc.replace(' ', '_')} if pd.notna(x) else 0.5)")
        w("")

    # -- NEW: Categorical cross features (combine top categorical pairs) --
    if len(categorical_cols) >= 2:
        # Use top 2-3 categorical pairs by cardinality (lower is better for crossing)
        _sorted_cats = sorted(categorical_cols,
                              key=lambda c: columns.get(c, {}).get("nunique", 999))
        _cross_pairs = []
        for i in range(min(len(_sorted_cats), 3)):
            for j in range(i + 1, min(len(_sorted_cats), 3)):
                _cross_pairs.append((_sorted_cats[i], _sorted_cats[j]))
        if _cross_pairs:
            w("# ---- Categorical cross features ----")
            for a, b in _cross_pairs:
                xname = f"{a}_X_{b}"
                w(f"for df in [train, test]:")
                w(f"    df['{xname}'] = df['{a}'].fillna('_NA').astype(str) + '_' + df['{b}'].fillna('_NA').astype(str)")
                # Cap cardinality
                w(f"_top_cats_{a}_{b} = set(train['{xname}'].value_counts().head(50).index)")
                w(f"for df in [train, test]:")
                w(f"    df['{xname}'] = df['{xname}'].where(df['{xname}'].isin(_top_cats_{a}_{b}), '_Other')")
            w("")

    # -- NEW: Frequency encoding for categorical columns --
    _freq_encode_cols = categorical_cols + bool_str_cols
    if _freq_encode_cols:
        w("# ---- Frequency encoding for categorical columns ----")
        for fc in _freq_encode_cols:
            w(f"if '{fc}' in train.columns:")
            w(f"    _freq_{fc} = train['{fc}'].value_counts(normalize=True)")
            w(f"    train['{fc}_freq'] = train['{fc}'].map(_freq_{fc}).fillna(0)")
            w(f"    test['{fc}_freq']  = test['{fc}'].map(_freq_{fc}).fillna(0)")
        w("")

    # -- NEW: Target encoding for categorical columns (out-of-fold, smoothed) --
    _te_cols = [c for c in (categorical_cols + bool_str_cols)
                if c in [col_name for col_name in columns]]
    if _te_cols and not is_multi_target:
        w("# ---- Target encoding (out-of-fold, smoothed) ----")
        w("from sklearn.model_selection import KFold as _KFold_TE")
        w(f"_te_global_mean = y.mean()")
        w(f"_te_smoothing = 10.0")
        for tc in _te_cols:
            te_name = f"{tc}_te"
            w(f"if '{tc}' in train.columns:")
            w(f"    train['{te_name}'] = np.nan")
            w(f"    _kf_te = _KFold_TE(n_splits=5, shuffle=True, random_state=42)")
            w(f"    for _tr_idx, _val_idx in _kf_te.split(train):")
            w(f"        _fold = train.iloc[_tr_idx]")
            w(f"        _fold_y = y.iloc[_tr_idx]")
            w(f"        _stats = _fold.assign(_t=_fold_y.values).groupby('{tc}')['_t'].agg(['mean', 'count'])")
            w(f"        _smoothed = (_stats['mean'] * _stats['count'] + _te_global_mean * _te_smoothing) / (_stats['count'] + _te_smoothing)")
            w(f"        train.iloc[_val_idx, train.columns.get_loc('{te_name}')] = train.iloc[_val_idx]['{tc}'].map(_smoothed).values")
            w(f"    _full_stats = train.assign(_t=y.values).groupby('{tc}')['_t'].agg(['mean', 'count'])")
            w(f"    _smoothed_full = (_full_stats['mean'] * _full_stats['count'] + _te_global_mean * _te_smoothing) / (_full_stats['count'] + _te_smoothing)")
            w(f"    test['{te_name}'] = test['{tc}'].map(_smoothed_full).fillna(_te_global_mean)")
            w(f"    train['{te_name}'] = train['{te_name}'].fillna(_te_global_mean)")
        w("")

    # -- Preprocessing function (imputation + encoding) --
    w("# ---- Preprocessing ----")
    w("def preprocess(train_df, test_df):")
    w("    # Impute numeric with median")
    w("    num_cols = list(train_df.select_dtypes(include='number').columns)")
    w("    for c in num_cols:")
    w("        med = train_df[c].median()")
    w("        train_df[c] = train_df[c].fillna(med)")
    w("        test_df[c]  = test_df[c].fillna(med)")
    w("    # Impute object/str with mode, then LabelEncode")
    w("    obj_cols = list(train_df.select_dtypes(include=['object', 'str']).columns)")
    w("    for c in obj_cols:")
    w("        mode_val = train_df[c].mode().iloc[0] if len(train_df[c].mode()) > 0 else 'Unknown'")
    w("        train_df[c] = train_df[c].fillna(mode_val)")
    w("        test_df[c]  = test_df[c].fillna(mode_val)")
    w("        le = LabelEncoder()")
    w("        le.fit(pd.concat([train_df[c], test_df[c]], ignore_index=True))")
    w("        train_df[c] = le.transform(train_df[c])")
    w("        test_df[c]  = le.transform(test_df[c])")
    w("    return train_df, test_df")
    w("")
    w("train, test = preprocess(train, test)")
    w("")

    # -- NEW: Cluster labels (KMeans on numeric features — after imputation) --
    if len(continuous_cols) >= 3:
        _cluster_cols = [c for c in continuous_cols if c != target_col]
        if len(_cluster_cols) >= 3:
            cluster_cols_str = str(_cluster_cols[:15])  # cap at 15 to limit compute
            w("# ---- Cluster labels (KMeans on continuous features) ----")
            w("from sklearn.cluster import KMeans as _KMeans")
            w("from sklearn.preprocessing import StandardScaler as _StdScaler")
            w(f"_cl_cols = [c for c in {cluster_cols_str} if c in train.columns and c in test.columns]")
            w("if len(_cl_cols) >= 2:")
            w("    _cl_train = train[_cl_cols].fillna(0).astype(float)")
            w("    _cl_test  = test[_cl_cols].fillna(0).astype(float)")
            w("    _cl_scaler = _StdScaler()")
            w("    _cl_train_s = _cl_scaler.fit_transform(_cl_train)")
            w("    _cl_test_s  = _cl_scaler.transform(_cl_test)")
            w("    _km = _KMeans(n_clusters=8, random_state=42, n_init=10, max_iter=100)")
            w("    train['cluster_8'] = _km.fit_predict(_cl_train_s)")
            w("    test['cluster_8']  = _km.predict(_cl_test_s)")
            w("    del _cl_train, _cl_test, _cl_train_s, _cl_test_s, _cl_scaler, _km")
            w("    gc.collect()")
            w("")

    # -- Assert no object columns remain --
    w("assert train.select_dtypes(include=['object', 'str']).shape[1] == 0, \\")
    w("    f\"Object cols remain in train: {list(train.select_dtypes(include=['object', 'str']).columns)}\"")
    w("assert test.select_dtypes(include=['object', 'str']).shape[1] == 0, \\")
    w("    f\"Object cols remain in test: {list(test.select_dtypes(include=['object', 'str']).columns)}\"")
    w("")

    # -- Feature importance pruning: keep only features with non-zero importance --
    if not is_multi_target:
        w("# ---- Feature importance pruning ----")
        w("print(f'Features before pruning: {train.shape[1]}')")
        w("_prune_lgbm = LGBMClassifier(n_estimators=100, learning_rate=0.1, num_leaves=31,")
        w("                              random_state=42, verbose=-1, n_jobs=MAX_JOBS)")
        if is_binary or is_multiclass:
            w("_prune_lgbm.fit(train, y)")
        else:
            w("_prune_lgbm = LGBMRegressor(n_estimators=100, learning_rate=0.1, num_leaves=31,")
            w("                             random_state=42, verbose=-1, n_jobs=MAX_JOBS)")
            w("_prune_lgbm.fit(train, y)")
        w("_importances = _prune_lgbm.feature_importances_")
        w("_keep_mask = _importances > 0")
        w("_keep_cols = list(train.columns[_keep_mask])")
        w("_drop_cols = list(train.columns[~_keep_mask])")
        w("if _drop_cols:")
        w("    print(f'Pruning {len(_drop_cols)} zero-importance features: {_drop_cols[:10]}...')")
        w("    train = train[_keep_cols]")
        w("    test  = test[_keep_cols]")
        w("print(f'Features after pruning: {train.shape[1]}')")
        w("del _prune_lgbm, _importances, _keep_mask, _keep_cols, _drop_cols")
        w("gc.collect()")
        w("")

    # -- Model training --
    w("# ---- Model ----")
    w("X_train = train")
    cv_max_rows = 20_000  # subsample cap for CV scoring

    if is_multi_target:
        # Multi-target: train one LGBM per target, predict_proba each
        w("")
        w("# ---- Multi-target: per-target binary classifiers ----")
        if not PHASE0_SKIP_CV:
            w("# ---- Cross-validation scoring (averaged across targets) ----")
            w("try:")
            w(f"    _CV_MAX = {cv_max_rows}")
            w("    if len(X_train) > _CV_MAX:")
            w("        _idx = np.random.RandomState(42).choice(len(X_train), _CV_MAX, replace=False)")
            w("        _X_cv = X_train.iloc[_idx]")
            w("    else:")
            w("        _X_cv = X_train")
            w("        _idx = None")
            w("    _cv_aucs = []")
            w("    for _tc in _target_cols:")
            w("        _y_cv = y_multi[_tc].iloc[_idx] if _idx is not None else y_multi[_tc]")
            w("        _cv_model = LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,")
            w("                                    subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1, n_jobs=MAX_JOBS)")
            w("        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)")
            w("        _scores = cross_val_score(_cv_model, _X_cv, _y_cv, cv=cv, scoring='roc_auc', n_jobs=1)")
            w("        _cv_aucs.append(_scores.mean())")
            w("        print(f'  {_tc} CV roc_auc: {_scores.mean():.6f}')")
            w("    cv_mean = np.mean(_cv_aucs)")
            w("    print(f'CV_METRIC=roc_auc')")
            w("    print(f'CV_SCORE={cv_mean:.6f}')")
            w("except Exception as e:")
            w("    print(f'CV_SCORE=FAILED ({e})')")
        else:
            w("print('CV_SCORE=SKIPPED')")
        w("")
        w("# ---- Train on ALL data for final submission ----")
        w("_models = {}")
        w("for _tc in _target_cols:")
        w("    _lgbm = LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=31,")
        w("                            subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1, n_jobs=MAX_JOBS)")
        w("    _xgb = XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=6,")
        w("                          subsample=0.8, colsample_bytree=0.8, random_state=42,")
        w("                          verbosity=0, eval_metric='logloss', nthread=MAX_JOBS)")
        w("    _et = ExtraTreesClassifier(n_estimators=500, random_state=42, n_jobs=MAX_JOBS)")
        w("    _cat = CatBoostClassifier(iterations=500, learning_rate=0.05, depth=6,")
        w("                               random_seed=42, verbose=0, thread_count=MAX_JOBS)")
        w("    _ens = VotingClassifier(")
        w("        estimators=[('lgbm', _lgbm), ('xgb', _xgb), ('et', _et), ('cat', _cat)],")
        w("        voting='soft')")
        w("    _ens.fit(X_train, y_multi[_tc])")
        w("    _models[_tc] = _ens")
        w("")
        w("# ---- Submission ----")
        w(f"submission = pd.DataFrame({{'{id_col}': test_ids}})")
        w("for _tc in _target_cols:")
        w("    submission[_tc] = _models[_tc].predict_proba(test)[:, 1]")
    elif is_binary and n_train >= 1000:
        # Always use 4-model ensemble for binary classification
        w("lgbm = LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=31,")
        w("                       subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1, n_jobs=MAX_JOBS)")
        w("xgb  = XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=6,")
        w("                      subsample=0.8, colsample_bytree=0.8, random_state=42,")
        w("                      verbosity=0, eval_metric='logloss', nthread=MAX_JOBS)")
        w("et   = ExtraTreesClassifier(n_estimators=300, random_state=42, n_jobs=MAX_JOBS)")
        w("cat  = CatBoostClassifier(iterations=500, learning_rate=0.05, depth=6,")
        w("                           random_seed=42, verbose=0, thread_count=MAX_JOBS)")
        w("model = VotingClassifier(")
        w("    estimators=[('lgbm', lgbm), ('xgb', xgb), ('et', et), ('cat', cat)],")
        w("    voting='soft')")
        w("print(f'Binary model: VotingClassifier (4-model ensemble)')")
    elif is_binary:
        w("model = ExtraTreesClassifier(n_estimators=300, random_state=42, n_jobs=MAX_JOBS)")
    elif is_multiclass:
        # Always use 4-model ensemble for multiclass
        w("lgbm = LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=31,")
        w("                       subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1, n_jobs=MAX_JOBS)")
        w("xgb  = XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=6,")
        w("                      subsample=0.8, colsample_bytree=0.8, random_state=42,")
        w("                      verbosity=0, eval_metric='mlogloss', nthread=MAX_JOBS)")
        w("et   = ExtraTreesClassifier(n_estimators=500, random_state=42, n_jobs=MAX_JOBS)")
        w("cat  = CatBoostClassifier(iterations=500, learning_rate=0.05, depth=6,")
        w("                           random_seed=42, verbose=0, thread_count=MAX_JOBS)")
        w("model = VotingClassifier(")
        w("    estimators=[('lgbm', lgbm), ('xgb', xgb), ('et', et), ('cat', cat)],")
        w("    voting='soft')")
        w("print(f'Multiclass model: VotingClassifier (4-model ensemble)')")
        w("print(f'Multiclass model: {type(model).__name__} (mem={_MEM_GB:.0f}GB)')")
    else:
        # Regression — adaptive sequential ensemble with factory lambdas
        w("# Regression ensemble: factory lambdas for true memory release")
        w("if _MEM_GB >= 32:")
        w("    _reg_factories = [")
        w("        ('lgbm', lambda: LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=31,")
        w("                                        subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1, n_jobs=MAX_JOBS)),")
        w("        ('xgb',  lambda: XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6,")
        w("                                       subsample=0.8, colsample_bytree=0.8, random_state=42,")
        w("                                       verbosity=0, nthread=MAX_JOBS)),")
        w("        ('et',   lambda: ExtraTreesRegressor(n_estimators=500, random_state=42, n_jobs=MAX_JOBS)),")
        w("        ('cat',  lambda: CatBoostRegressor(iterations=500, learning_rate=0.05, depth=6,")
        w("                                            random_seed=42, verbose=0, thread_count=MAX_JOBS)),")
        w("    ]")
        w("elif _MEM_GB >= 16:")
        w("    _reg_factories = [")
        w("        ('lgbm', lambda: LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=31,")
        w("                                        subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1, n_jobs=MAX_JOBS)),")
        w("        ('xgb',  lambda: XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6,")
        w("                                       subsample=0.8, colsample_bytree=0.8, random_state=42,")
        w("                                       verbosity=0, nthread=MAX_JOBS)),")
        w("        ('cat',  lambda: CatBoostRegressor(iterations=500, learning_rate=0.05, depth=6,")
        w("                                            random_seed=42, verbose=0, thread_count=MAX_JOBS)),")
        w("    ]")
        w("elif _MEM_GB >= 8:")
        w("    _reg_factories = [")
        w("        ('lgbm', lambda: LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=31,")
        w("                                        subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1, n_jobs=MAX_JOBS)),")
        w("        ('cat',  lambda: CatBoostRegressor(iterations=500, learning_rate=0.05, depth=6,")
        w("                                            random_seed=42, verbose=0, thread_count=MAX_JOBS)),")
        w("    ]")
        w("else:")
        w("    _reg_factories = [")
        w("        ('lgbm', lambda: LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=31,")
        w("                                        subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1, n_jobs=MAX_JOBS)),")
        w("    ]")
        w("print(f'Regression ensemble: {len(_reg_factories)} models (mem={_MEM_GB:.0f}GB)')")
    w("")
    if not is_multi_target:
        # -- CV scoring (quality measurement only — final model trains on ALL data) --
        # Use the competition metric detected from description.md / EDA.
        comp_scoring = meta.get("competition_scoring", "")
        comp_metric_label = meta.get("competition_metric_label", "")
        if PHASE0_SKIP_CV:
            w("# ---- Phase 0 CV skipped (PURPLE_PHASE0_SKIP_CV=1) ----")
            w("print('CV_SCORE=SKIPPED')")
        elif is_binary or is_multiclass:
            # Use detected metric, fallback to accuracy
            if comp_scoring and comp_scoring not in (
                "neg_root_mean_squared_error", "neg_mean_squared_error",
                "neg_mean_absolute_error", "neg_root_mean_squared_log_error", "r2",
            ):
                cv_scoring = comp_scoring
                cv_metric_name = comp_metric_label or comp_scoring
            elif target_is_proba and is_binary:
                cv_scoring = "roc_auc"
                cv_metric_name = "roc_auc"
            else:
                cv_scoring = "accuracy"
                cv_metric_name = "accuracy"
            w("try:")
            w(f"    _CV_MAX = {cv_max_rows}")
            w("    if len(X_train) > _CV_MAX:")
            w("        _idx = np.random.RandomState(42).choice(len(X_train), _CV_MAX, replace=False)")
            w("        _X_cv, _y_cv = X_train.iloc[_idx], y.iloc[_idx]")
            w("    else:")
            w("        _X_cv, _y_cv = X_train, y")
            w("    _cv_model = LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,")
            w("                                subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1, n_jobs=MAX_JOBS)")
            w("    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)")
            w(f"    cv_scores = cross_val_score(_cv_model, _X_cv, _y_cv, cv=cv, scoring='{cv_scoring}', n_jobs=1)")
            w("    cv_mean = cv_scores.mean()")
            w(f"    print(f'CV_METRIC={cv_metric_name}')")
            w("    print(f'CV_SCORE={cv_mean:.6f}')")
            w("    print(f'CV_STD={cv_scores.std():.6f}')")
            w("except Exception as e:")
            w("    print(f'CV_SCORE=FAILED ({e})')")
        else:
            # Regression — use detected metric or default RMSE
            if comp_scoring and comp_scoring.startswith("neg_"):
                reg_cv_scoring = comp_scoring
                reg_cv_metric = comp_metric_label or comp_scoring
            elif comp_scoring == "r2":
                reg_cv_scoring = "r2"
                reg_cv_metric = "r2"
            else:
                reg_cv_scoring = "neg_root_mean_squared_error"
                reg_cv_metric = "rmse"
            w("try:")
            w("    from sklearn.model_selection import KFold")
            w(f"    _CV_MAX = {cv_max_rows}")
            w("    if len(X_train) > _CV_MAX:")
            w("        _idx = np.random.RandomState(42).choice(len(X_train), _CV_MAX, replace=False)")
            w("        _X_cv, _y_cv = X_train.iloc[_idx], y.iloc[_idx]")
            w("    else:")
            w("        _X_cv, _y_cv = X_train, y")
            w("    _cv_model = LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31,")
            w("                               subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1, n_jobs=MAX_JOBS)")
            w("    cv = KFold(n_splits=5, shuffle=True, random_state=42)")
            w(f"    cv_scores = cross_val_score(_cv_model, _X_cv, _y_cv, cv=cv, scoring='{reg_cv_scoring}', n_jobs=1)")
            w("    cv_mean = cv_scores.mean()")
            w(f"    print(f'CV_METRIC={reg_cv_metric}')")
            w("    print(f'CV_SCORE={cv_mean:.6f}')")
            w("    print(f'CV_STD={cv_scores.std():.6f}')")
            w("except Exception as e:")
            w("    print(f'CV_SCORE=FAILED ({e})')")
        w("")
        if is_regression:
            # Sequential ensemble with factory lambdas: true memory release
            w("# ---- Sequential ensemble: train-predict-delete (memory-safe) ----")
            w("_n_models = len(_reg_factories)")
            w("_preds_sum = np.zeros(len(test))")
            w("for _name, _factory in _reg_factories:")
            w("    print(f'  Training {_name}...')")
            w("    _m = _factory()")
            w("    _m.fit(X_train, y)")
            w("    _preds_sum += _m.predict(test)")
            w("    del _m")
            w("    gc.collect()")
            w("del _reg_factories")
            w("gc.collect()")
            w("preds = _preds_sum / _n_models")
            w("if not _y_has_negatives:")
            w("    preds = np.expm1(preds)")
        else:
            w("# ---- Train on ALL data for final submission ----")
            w("model.fit(X_train, y)")
            w("gc.collect()")
            w("")
            # -- Prediction and submission --
            w("# ---- Submission ----")
            if target_is_proba and is_binary:
                w("preds = model.predict_proba(test)[:, 1]")
            else:
                w("preds = model.predict(test)")
        if target_is_bool:
            w(f"submission = pd.DataFrame({{'{id_col}': test_ids, '{target_col}': preds.astype(bool)}})")
        else:
            w(f"submission = pd.DataFrame({{'{id_col}': test_ids, '{target_col}': preds}})")

    # -- Common submission writing (multi-target already built submission above) --
    w(f"submission.to_csv(f'{{DATA_DIR}}/submission.csv', index=False)")
    w("print(submission.shape)")
    w("print(submission.dtypes)")
    w("print(submission.head())")

    return "\n".join(lines)
# Fail-pattern detector — scan stderr for known crash signatures
# ---------------------------------------------------------------------------

_FAIL_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"ValueError.*cannot convert float NaN to integer", re.I),
     "A column contains NaN and was cast to int without fillna first. "
     "Always call .fillna() BEFORE .astype(int)."),
    (re.compile(r"ValueError.*Input contains NaN", re.I),
     "Features or target contain NaN. Ensure ALL NaN values are filled before "
     "fitting. Check: df.isnull().sum() for features AND y.isnull().sum() for target."),
    (re.compile(r"ValueError.*pandas dtypes must be int, float or bool", re.I),
     "A string/object column was passed to the model. Ensure ALL object columns "
     "are encoded or dropped before fitting."),
    (re.compile(r"ValueError.*could not convert string to float", re.I),
     "A string value was passed where a numeric was expected. Check encoding."),
    (re.compile(r"KeyError: ['\"]?(\w+)", re.I),
     "A column name was referenced that does not exist in the DataFrame."),
    (re.compile(r"LabelEncoder.*unseen label", re.I),
     "LabelEncoder encountered labels in test that were not in training. "
     "Fit on pd.concat([train[col], test[col]]) instead."),
    (re.compile(r"ValueError.*Length of values.*does not match", re.I),
     "Array length mismatch — likely a filtering step changed row count "
     "inconsistently between train and test."),
    (re.compile(r"FileNotFoundError", re.I),
     "A file path was wrong. Check that paths use the data directory variable."),
    (re.compile(r"ArrayMemoryError|Unable to allocate", re.I),
     "OUT OF MEMORY — a transformation created a massive array. "
     "Most likely cause: pd.get_dummies on a high-cardinality column. "
     "Fix: use LabelEncoder instead of pd.get_dummies for ALL categorical columns. "
     "Check column cardinality in the EDA report and NEVER call get_dummies."),
]


def _diagnose_failure(output: str) -> str:
    """Scan execution output for known crash patterns and return structured diagnostics."""
    diagnostics: list[str] = []
    for pattern, explanation in _FAIL_PATTERNS:
        match = pattern.search(output)
        if match:
            diagnostics.append(f"DETECTED: {match.group(0)}\n  FIX: {explanation}")
    if diagnostics:
        header = "AUTOMATED DIAGNOSTICS (fix these specific issues):\n"
        return header + "\n".join(diagnostics) + "\n\n" + f"Raw output tail:\n{output[-4000:]}"
    # No known pattern matched — return truncated raw output
    return output[-8000:]


# ---------------------------------------------------------------------------
# Prompts — plan-first two-turn approach
# ---------------------------------------------------------------------------

def _build_planning_prompt(
    instructions: str, description: str, data_dir: str, eda_report: str = "",
    baseline_script: str = "", baseline_cv: float | None = None,
    baseline_metric: str = "", baseline_output: str = "",
) -> str:
    """Prompt for the planning turn — asks the model to reason before coding."""
    eda_section = (
        f"\n<schema_report>\n{eda_report}\n</schema_report>\n"
    ) if eda_report else ""
    reference_section = ""
    if baseline_script:
        cv_info = f" (CV {baseline_metric}={baseline_cv:.4f})" if baseline_cv is not None else ""
        # Truncate output to key lines (CV scores, shape info, errors)
        output_summary = ""
        if baseline_output:
            key_lines = [
                line for line in baseline_output.splitlines()
                if any(kw in line for kw in (
                    "CV_", "Features", "pruning", "shape", "Binary model",
                    "cluster", "Multiclass model", "Regression",
                ))
            ]
            output_summary = "\n".join(key_lines[:15])
        reference_section = (
            f"\n<reference_solution>\n"
            f"A deterministic reference solution was run on the full data{cv_info}.\n"
            f"Study its approach — what preprocessing, features, and model choices it makes.\n"
            f"Your job is to reason about what it does well, what it misses, and how to\n"
            f"IMPROVE upon it using the toolkit functions available to you.\n\n"
            f"```python\n{baseline_script}\n```\n"
        )
        if output_summary:
            reference_section += (
                f"\nReference solution output:\n{output_summary}\n"
            )
        reference_section += "</reference_solution>\n"
    return textwrap.dedent(f"""
    You are an expert ML engineer competing in a Kaggle competition.

    <context>
    <instructions>{instructions}</instructions>
    <description>{description}</description>
    <data_dir>{data_dir}</data_dir>
    {eda_section}{reference_section}
    </context>

    Produce a concise bullet-point plan (no code). Cover:
    a) TARGET: column name, classification or regression?
    b) ID: submission ID column (from sample_submission.csv). Save and drop.
    c) DROP: list ID/name/free-text columns to drop.
       Do NOT drop structured-string columns — split them into features.
    d) PREPROCESSING: for each remaining column, state type and null handling.
    e) FEATURES: derived features worth adding. Always consider:
       - Structured IDs ("GGGG_PP") → GroupSize, IsSolo
       - Name columns → FamilySize from last-name frequency
       - Age → IsChild, IsSenior bins
       - Spending columns → TotalSpend, IsZeroSpend
       - Structured-string columns → split on delimiter
    f) MODEL: which model and why. Default to LightGBM ensemble for tabular.
    g) SUBMISSION: columns and dtypes for submission.csv.

    <rules>
    - HIGH_CARD columns MUST use LabelEncoder. NEVER pd.get_dummies.
    - Keep plan under 400 words. Use actual column names.
    - Output ONLY the plan — no code, no markdown.
    </rules>
    """).strip()


# ---------------------------------------------------------------------------
# Tool-calling infrastructure — Pipeline orchestrator
# ---------------------------------------------------------------------------

# fe_* functions that DON'T return (train, test) — need special handling
_FE_SPECIAL = {
    "fe_mutual_info",   # returns list[str]
    "fe_haversine",     # returns single df
}

# Functions whose first arg is not (train, test)
_FE_SINGLE_DF = {"fe_haversine"}


def _exec_fe_pipeline(steps: list[dict], exec_globals: dict) -> str:
    """Execute a JSON-specified feature engineering pipeline.

    Each step is {"fn": "fe_xxx", "args": {...}}.
    The orchestrator handles:
      - Threading train/test through each function
      - Special-casing fe_mutual_info (returns list, stored in context)
      - Special-casing fe_haversine (single-df API)
      - Validation after all steps complete
    """
    from purple.ml_toolkit import TOOLKIT_FUNCTIONS

    train = exec_globals.get("train")
    test = exec_globals.get("test")
    y = exec_globals.get("y")

    if train is None or test is None:
        return "ERROR: train/test not loaded. Run setup first."

    original_train_rows = len(train)
    original_test_rows = len(test)
    results: list[str] = []
    created_features: list[str] = []

    for i, step in enumerate(steps):
        fn_name = step.get("fn", "")
        args = step.get("args", {})
        # Support args as JSON string (strict mode schema) or dict
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except (json.JSONDecodeError, TypeError):
                args = {}

        if fn_name not in TOOLKIT_FUNCTIONS:
            results.append(f"Step {i+1}: SKIP — unknown function '{fn_name}'")
            continue

        fn = TOOLKIT_FUNCTIONS[fn_name]

        try:
            if fn_name == "fe_mutual_info":
                # Special: returns list, needs target
                result = fn(train, y, **args)
                exec_globals["_mi_best_features"] = result
                results.append(
                    f"Step {i+1}: {fn_name} → {len(result)} features selected"
                )
            elif fn_name == "fe_target_encode":
                # Special: needs target Series from exec_globals
                # Remove any target/target_col from LLM args to avoid conflicts
                te_args = {k: v for k, v in args.items()
                           if k not in ("target", "target_col")}
                cols_before = set(train.columns)
                train, test = fn(train, test, target=y, **te_args)
                new_cols = set(train.columns) - cols_before
                created_features.extend(new_cols)
                results.append(
                    f"Step {i+1}: {fn_name} → OK"
                    + (f" (+{len(new_cols)} cols)" if new_cols else "")
                )
            elif fn_name == "fe_haversine":
                # Special: single-df API, apply to both
                cols_before = set(train.columns)
                train = fn(train, **args)
                test = fn(test, **args)
                new_cols = set(train.columns) - cols_before
                created_features.extend(new_cols)
                results.append(
                    f"Step {i+1}: {fn_name} → added {list(new_cols)}"
                )
            else:
                # Standard (train, test, ...) → (train, test) pattern
                cols_before = set(train.columns)
                train, test = fn(train, test, **args)
                new_cols = set(train.columns) - cols_before
                created_features.extend(new_cols)
                results.append(
                    f"Step {i+1}: {fn_name} → OK"
                    + (f" (+{len(new_cols)} cols)" if new_cols else "")
                )
        except Exception as exc:
            results.append(f"Step {i+1}: {fn_name} → ERROR: {exc}")

    # Update exec_globals with modified dataframes
    exec_globals["train"] = train
    exec_globals["test"] = test

    # Validation
    validation: list[str] = []
    if len(train) != original_train_rows:
        validation.append(
            f"WARNING: train rows changed {original_train_rows} → {len(train)}"
        )
    if len(test) != original_test_rows:
        validation.append(
            f"WARNING: test rows changed {original_test_rows} → {len(test)}"
        )

    summary = (
        f"=== FE Pipeline Complete ===\n"
        f"Steps executed: {len(results)}\n"
        + "\n".join(results)
        + f"\n\nTrain: {train.shape}, Test: {test.shape}"
        + f"\nNew features: {len(created_features)}"
    )
    if validation:
        summary += "\n\nVALIDATION:\n" + "\n".join(validation)

    return summary[:12_000]


def _exec_model_pipeline(config: dict, exec_globals: dict, data_dir: str,
                         fast_cv: bool = False) -> str:
    """Execute a JSON-specified model training pipeline.

    Config: {
        "model_type": "lgbm"|"ensemble"|"stack"|"tuned",
        "task": "binary"|"multiclass"|"regression",
        "scoring": "accuracy"|"f1"|"roc_auc"|"rmse"|...,
        "target_dtype": "bool"|"int"|"float"|"str",
        "tune": false,
        "tune_timeout": 60,
        "drop_cols": ["Name", ...]
    }

    When fast_cv=True, uses LGBM-only for CV and skips ensemble
    train+predict to save time.  A final ensemble retrain should be
    done after Phase D completes on the best feature set.
    """
    from purple.ml_toolkit import TOOLKIT_FUNCTIONS

    train = exec_globals.get("train")
    test = exec_globals.get("test")
    y = exec_globals.get("y")
    ids = exec_globals.get("ids")

    if train is None or test is None or y is None:
        return "ERROR: train/test/y not loaded."

    task = config.get("task", "binary")
    model_type = config.get("model_type", "ensemble")
    scoring = config.get("scoring", "accuracy")
    target_dtype = config.get("target_dtype", "bool")
    tune = config.get("tune", False)
    tune_timeout = config.get("tune_timeout", 60)
    drop_cols = config.get("drop_cols", [])

    # Cache the config so Phase D can reuse the LLM's choice
    exec_globals["_last_model_config"] = dict(config)

    results: list[str] = []

    try:
        # Step 1: Preprocess (auto)
        preprocess_fn = TOOLKIT_FUNCTIONS["preprocess"]
        id_col = exec_globals.get("ID_COL", "")
        all_drop = list(set(drop_cols + ([id_col] if id_col else [])))

        # Filter to columns that actually exist
        all_drop = [c for c in all_drop if c in train.columns]
        train, test = preprocess_fn(train, test, drop_cols=all_drop)
        exec_globals["train"] = train
        exec_globals["test"] = test
        results.append(f"Preprocess: train={train.shape}, test={test.shape}")

        # Validate no object columns
        obj_cols = train.select_dtypes(include=["object"]).columns.tolist()
        if obj_cols:
            results.append(
                f"WARNING: {len(obj_cols)} object columns remain: {obj_cols[:10]}"
            )

        # Step 2: Build model
        if model_type == "tuned" or tune:
            tune_fn = TOOLKIT_FUNCTIONS["tune_lgbm"]
            best_params = tune_fn(train, y, task=task, timeout=tune_timeout)
            model = TOOLKIT_FUNCTIONS["build_model"](task=task, **best_params)
            results.append(f"Tuned LightGBM: {best_params}")
        elif model_type == "stack":
            stack_fn = TOOLKIT_FUNCTIONS["stack_ensemble"]
            preds, oof_score = stack_fn(train, y, test, task=task)
            results.append(f"Stack ensemble: OOF score={oof_score:.6f}")
            # Write submission directly
            write_fn = TOOLKIT_FUNCTIONS["write_submission"]
            write_fn(ids, preds,
                      exec_globals.get("ID_COL", "id"),
                      exec_globals.get("TARGET_COL", "target"),
                      data_dir, target_dtype=target_dtype)
            results.append(f"CV_METRIC={scoring}")
            results.append(f"CV_SCORE={oof_score:.6f}")
            return "\n".join(results)
        elif fast_cv and model_type == "ensemble":
            # FAST_CV: use single LGBM for CV only, skip ensemble train+predict
            model = TOOLKIT_FUNCTIONS["build_model"](task=task)
            results.append("Fast CV: LGBM-only (ensemble deferred to final retrain)")
        elif model_type == "ensemble":
            model = TOOLKIT_FUNCTIONS["build_ensemble"](task=task)
            results.append("Built 4-model voting ensemble (LGBM+XGB+ET+CatBoost)")
        else:
            model = TOOLKIT_FUNCTIONS["build_model"](task=task)
            results.append("Built LightGBM model")

        # Step 3: Evaluate CV
        cv_fn = TOOLKIT_FUNCTIONS["evaluate_cv"]
        cv_result = cv_fn(model, train, y, scoring=scoring)
        results.append(f"CV_METRIC={scoring}")
        results.append(f"CV_SCORE={cv_result:.6f}")
        results.append(f"CV evaluation complete")

        # Step 4-5: Train, predict, write submission (skip in fast_cv ensemble mode)
        if fast_cv and model_type == "ensemble":
            results.append("Skipped ensemble train+predict (fast_cv mode)")
        else:
            # Step 4: Train and predict
            predict_fn = TOOLKIT_FUNCTIONS["train_and_predict"]
            proba = task == "binary" and scoring in ("roc_auc", "log_loss", "neg_log_loss")
            preds = predict_fn(model, train, y, test, proba=proba)
            results.append(f"Predictions: {len(preds)} rows")

            # Step 5: Write submission
            write_fn = TOOLKIT_FUNCTIONS["write_submission"]
            write_fn(ids, preds,
                      exec_globals.get("ID_COL", "id"),
                      exec_globals.get("TARGET_COL", "target"),
                      data_dir, target_dtype=target_dtype)
            results.append(f"submission.csv written to {data_dir}")

        # Step 6: Feature importance feedback (Rule #26)
        try:
            if hasattr(model, 'estimators_'):
                # VotingClassifier — use first LGBM estimator
                for _name, _est in model.estimators_:
                    if hasattr(_est, 'feature_importances_'):
                        _imp = _est.feature_importances_
                        _cols = list(train.columns)
                        _top_idx = sorted(range(len(_imp)), key=lambda i: -_imp[i])[:10]
                        _top = [((_cols[i] if i < len(_cols) else f"f{i}"), int(_imp[i])) for i in _top_idx]
                        results.append("TOP_FEATURES=" + ", ".join(f"{n}:{v}" for n, v in _top))
                        break
            elif hasattr(model, 'feature_importances_'):
                _imp = model.feature_importances_
                _cols = list(train.columns)
                _top_idx = sorted(range(len(_imp)), key=lambda i: -_imp[i])[:10]
                _top = [((_cols[i] if i < len(_cols) else f"f{i}"), int(_imp[i])) for i in _top_idx]
                results.append("TOP_FEATURES=" + ", ".join(f"{n}:{v}" for n, v in _top))
        except Exception:
            pass  # feature importance is a nice-to-have, not critical

    except Exception as exc:
        import traceback
        tb = traceback.format_exc()
        results.append(f"ERROR: {exc}\n{tb[-2000:]}")

    return "\n".join(results)


_TOOL_SCHEMA_RUN_PYTHON = {
    "type": "function",
    "function": {
        "name": "run_python",
        "strict": True,
        "description": (
            "Execute Python code in a persistent session.  Variables, imports, "
            "and DataFrames persist between calls.  The last expression is "
            "auto-printed, but explicit print() is still recommended for clarity. "
            "Use for EDA, custom logic, and debugging."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python code to execute."},
            },
            "required": ["code"],
            "additionalProperties": False,
        },
    },
}

_TOOL_SCHEMA_FE_PIPELINE = {
    "type": "function",
    "function": {
        "name": "run_fe_pipeline",
        "strict": True,
        "description": (
            "Execute a feature engineering pipeline. Provide an ordered list of "
            "fe_* function calls with arguments. The orchestrator handles "
            "train/test threading, validation, and error recovery. "
            "PREFERRED over run_python for feature engineering. "
            "Example: {\"steps\": ["
            "{\"fn\": \"fe_bool_convert\", \"args\": \"{\\\"cols\\\": [\\\"CryoSleep\\\"]}\"},"
            "{\"fn\": \"fe_spending_aggs\", \"args\": \"{\\\"spend_cols\\\": [\\\"RoomService\\\",\\\"FoodCourt\\\"]}\"},"
            "{\"fn\": \"fe_drop_constant\", \"args\": \"{\\\"threshold\\\": 0.99}\"}"
            "]}"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "steps": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "fn": {
                                "type": "string",
                                "description": "Toolkit function name (e.g. 'fe_ratios')",
                            },
                            "args": {
                                "type": "string",
                                "description": "JSON-encoded keyword arguments (excluding train/test). E.g. '{\"cols\": [\"Age\"]}'",
                            },
                        },
                        "required": ["fn", "args"],
                        "additionalProperties": False,
                    },
                    "description": "Ordered list of FE steps to execute.",
                },
            },
            "required": ["steps"],
            "additionalProperties": False,
        },
    },
}

_TOOL_SCHEMA_RUN_MODEL = {
    "type": "function",
    "function": {
        "name": "run_model",
        "strict": True,
        "description": (
            "Train a model, evaluate CV, and write submission.csv. "
            "Handles preprocessing automatically. "
            "PREFERRED over run_python for model training. "
            "Example: {\"model_type\": \"ensemble\", \"task\": \"binary\", "
            "\"scoring\": \"accuracy\", \"target_dtype\": \"bool\", "
            "\"tune\": false, \"tune_timeout\": 60, \"drop_cols\": [\"Name\"]}"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "model_type": {
                    "type": "string",
                    "enum": ["lgbm", "ensemble", "stack", "tuned"],
                    "description": "Model type: lgbm (single), ensemble (4-model vote), stack (meta-learner), tuned (Optuna).",
                },
                "task": {
                    "type": "string",
                    "enum": ["binary", "multiclass", "regression"],
                    "description": "ML task type.",
                },
                "scoring": {
                    "type": "string",
                    "description": "Scoring metric (accuracy, f1, roc_auc, rmse, etc.).",
                },
                "target_dtype": {
                    "type": "string",
                    "enum": ["bool", "int", "float", "str"],
                    "description": "Expected dtype for target column in submission.",
                },
                "tune": {
                    "type": ["boolean", "null"],
                    "description": "Whether to run Optuna hyperparameter tuning.",
                },
                "tune_timeout": {
                    "type": ["integer", "null"],
                    "description": "Seconds for Optuna tuning (default 60).",
                },
                "drop_cols": {
                    "type": ["array", "null"],
                    "items": {"type": "string"},
                    "description": "Columns to drop before preprocessing (ID, Name, etc.).",
                },
            },
            "required": ["model_type", "task", "scoring", "target_dtype",
                         "tune", "tune_timeout", "drop_cols"],
            "additionalProperties": False,
        },
    },
}

# Phase-specific tool sets — fewer tools = higher accuracy
_TOOLS_PHASE_A = [_TOOL_SCHEMA_RUN_PYTHON]
_TOOLS_PHASE_B = [_TOOL_SCHEMA_FE_PIPELINE, _TOOL_SCHEMA_RUN_PYTHON]
_TOOLS_PHASE_C = [_TOOL_SCHEMA_RUN_MODEL, _TOOL_SCHEMA_RUN_PYTHON]
# Legacy: all tools (for fallback/iteration rounds)
_TOOL_SCHEMAS = [_TOOL_SCHEMA_RUN_PYTHON, _TOOL_SCHEMA_FE_PIPELINE,
                 _TOOL_SCHEMA_RUN_MODEL]


def _auto_print_last_expr(code: str) -> str:
    """If the last statement is a bare expression, wrap it in print(repr(...)).

    This eliminates the 'no stdout' problem where the model forgets print().
    Skips wrapping if the expression is already a print()/assert/raise call.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code  # let exec() report the error
    if not tree.body:
        return code
    last = tree.body[-1]
    if isinstance(last, ast.Expr):
        # Don't wrap if it's already a print() or logging call
        val = last.value
        if (isinstance(val, ast.Call)
                and isinstance(val.func, ast.Name)
                and val.func.id in ("print", "display", "logging")):
            return code
        # Rewrite: replace the last expression with print(repr(expr))
        # Extra parens ensure tuples like  a, b  become repr((a, b))
        # rather than the broken repr(a, b) which passes 2 args.
        source_lines = code.splitlines(keepends=True)
        start = last.lineno - 1
        end = last.end_lineno  # 1-based inclusive → use as exclusive index
        expr_source = "".join(source_lines[start:end]).strip()
        wrapped = f"print(repr(({expr_source})))"
        new_lines = source_lines[:start] + [wrapped + "\n"]
        if end is not None and end < len(source_lines):
            new_lines += source_lines[end:]
        return "".join(new_lines)
    return code


def _exec_run_python(code: str, exec_globals: dict) -> str:
    """Execute *code* in a persistent namespace, capture stdout/stderr.

    Auto-prints the last bare expression to avoid wasted 'no stdout' rounds.
    """
    # Strip leading/trailing whitespace and dedent to avoid IndentationError
    code = textwrap.dedent(code).strip()
    if not code:
        return "ERROR: empty code string"
    # Auto-wrap last bare expression in print()
    code = _auto_print_last_expr(code)
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
            exec(code, exec_globals)
        output = stdout_buf.getvalue()
        err = stderr_buf.getvalue()
        if err:
            output = (output + "\n" + err) if output else err
        if not output:
            # Still no output (e.g. assignments only) — provide namespace summary
            user_vars = sorted(
                k for k in exec_globals
                if not k.startswith("_") and k != "__builtins__"
            )
            return (
                "(executed OK — assignments only)\n"
                f"Namespace variables: {', '.join(user_vars[:30])}"
            )
        return output[:12_000]
    except Exception:
        import traceback
        tb = traceback.format_exc()
        partial = stdout_buf.getvalue()
        result = f"ERROR:\n{tb[-4000:]}"
        if partial:
            result = f"Output before error:\n{partial[-2000:]}\n\n{result}"
        return result[:12_000]


def _exec_list_dir(path: str) -> str:
    """List directory contents."""
    try:
        entries = sorted(os.listdir(path))
        return "\n".join(entries) if entries else "(empty directory)"
    except Exception as exc:
        return f"ERROR: {exc}"


def _exec_read_file(path: str, max_lines: int | None = None) -> str:
    """Read file text, optionally limited to *max_lines*."""
    try:
        text = Path(path).read_text(errors="replace")
        if max_lines is not None:
            lines = text.splitlines(keepends=True)
            text = "".join(lines[:max_lines])
            if len(lines) > max_lines:
                text += f"\n... ({len(lines) - max_lines} more lines)"
        return text[:12_000] if text else "(empty file)"
    except Exception as exc:
        return f"ERROR: {exc}"


def _exec_file_exists(path: str) -> str:
    """Check if a path exists."""
    return "true" if Path(path).exists() else "false"


def _build_tool_system_prompt(
    plan: str, description: str, data_dir: str,
    baseline_script: str = "", baseline_cv: float | None = None,
    baseline_metric: str = "",
) -> str:
    """System prompt for the interactive tool-calling agent.

    This is kept as a fallback for iteration rounds that use all tools.
    Phase-specific prompts below are preferred for Phases A/B/C.
    """
    from purple.ml_toolkit import TOOLKIT_DOCS

    baseline_section = ""
    if baseline_script:
        cv_info = ""
        if baseline_cv is not None and baseline_metric:
            cv_info = f"\nReference solution CV: {baseline_metric}={baseline_cv:.6f}. You MUST beat this."
        baseline_section = (
            f"\n<reference>\n"
            f"A reference solution was run on this data:{cv_info}\n"
            f"Raw data is pre-loaded: train, test, y, ids, DATA_DIR, TASK, TARGET_COL, ID_COL.\n"
            f"</reference>\n"
        )
    return textwrap.dedent(f"""
    You are an expert ML engineer solving a Kaggle competition.
    You have three tools: run_python, run_fe_pipeline, run_model.

    <context>
    <plan>{plan}</plan>
    <description>{description}</description>
    <data_dir>{data_dir}</data_dir>
    {baseline_section}
    </context>

    {TOOLKIT_DOCS}

    RULES:
    - Each run_python call has a 120-second limit. Keep operations focused.
    - Print CV_METRIC=<name> and CV_SCORE=<float> before writing submission.csv.
    - Only write submission.csv if your CV score BEATS the baseline.
    """).strip()


def _build_iteration_hints(
    eda_meta: dict, data_dir: str,
) -> list[str]:
    """Build a progressive 17-level hint ladder from eda_meta.

    Returns up to 17 hints ordered by **Rules of ML** principles:
    - Level 0 is KITCHEN SINK (Rule #16): throw ALL applicable FE at the
      model in one shot and let the ensemble sort out what matters.
    - Levels 1-12 are incremental FE hints, each adding one technique.
    - Levels 13-16 are [MODEL] hints (Phase III optimization), demoted
      to the end because the base 4-model ensemble is already strong.

    Features accumulate via cumulative state: when a round improves CV,
    subsequent rounds build ON TOP of the winning feature set instead of
    restarting from raw data.

    [MODEL]-prefixed hints skip feature engineering and only re-train (~40-60s),
    while FE hints add to the cumulative feature set (~120-140s).

    Ordering rationale (Rules of ML priority):
      Level  0: KITCHEN SINK (all FE)         — ~130s, Phase II Rule #16
      Level  1: Data quality cleanup          — ~130s, foundational
      Level  2: Target encoding               — ~130s, high-cardinality fix
      Level  3: Feature interactions          — ~130s, core tree weakness
      Level  4: Aggregation features          — ~130s, group-level signals
      Level  5: Rank & power transforms       — ~130s, distribution fix
      Level  6: Freq encoding + clipping      — ~130s, incremental
      Level  7: Mutual info selection         — ~130s, noise removal
      Level  8: PCA + K-Means clusters        — ~130s, rotated features
      Level  9: Deviation-from-group          — ~130s, within-group anomalies
      Level 10: Logical imputation + binning  — ~130s, dataset-specific
      Level 11: Permutation pruning           — ~130s, post-hoc selection
      Level 12: Pseudo-labeling               — ~130s, semi-supervised
      Level 13: [MODEL] Low LR + subsampling  — ~40s, Phase III
      Level 14: [MODEL] Stacking              — ~40s, Phase III
      Level 15: [MODEL] Optuna tuning         — ~60s, Phase III
      Level 16: [MODEL] Threshold / clipping  — ~40s, Phase III

    References embedded in hints:
      - Chen & Guestrin (2016) XGBoost paper — learned missing-value splits
      - Prokhorenkova et al. (2018) CatBoost — ordered target statistics
      - Grinsztajn et al. (2022) — tree weakness on rotated features → PCA
      - Jahrer (2017) Porto Seguro 1st — RankGauss transform
      - Friedman (2001/2002) — gradient boosting, stochastic subsampling
      - Wolpert (1992) — stacked generalization
      - Rashmi & Gilad-Bachrach (2015) — DART boosting
      - Breiman (2001) — permutation importance
      - Kursa & Rudnicki (2010) — Boruta feature selection
      - Lee (2013) — pseudo-labeling
      - Akiba et al. (2019) — Optuna framework
    """
    columns = eda_meta.get("columns", {})
    target_col = eda_meta.get("target_col", "")
    id_col = eda_meta.get("id_col", "")
    spend_cols = eda_meta.get("spend_cols", [])
    task = _infer_task_type(eda_meta)
    scoring = eda_meta.get("competition_scoring", "")

    # Collect column-role lists for data-driven hints
    high_null = sorted(
        [(c, m.get("null_pct", 0)) for c, m in columns.items()
         if m.get("null_pct", 0) > 5.0 and c != target_col and c != id_col],
        key=lambda x: -x[1],
    )
    cat_cols = [c for c, m in columns.items()
                if m.get("role") in ("CATEGORY", "BOOL_STR", "LOW_CARD_STR")
                and c != target_col and c != id_col]
    num_cols = [c for c, m in columns.items()
                if m.get("role") in ("NUMERIC", "FLOAT", "INT", "SPENDING")
                and c != target_col and c != id_col]
    bool_cols = [c for c, m in columns.items()
                 if m.get("role") in ("BOOL_STR", "BINARY")]
    structured = [(c, m) for c, m in columns.items()
                  if m.get("role") == "STRUCTURED_STR" and c != id_col]
    id_meta = columns.get(id_col, {})

    # Build all hint strings; final ordering at end of function
    # prioritises impact-per-second (fast [MODEL] hints interleaved early).

    # ================================================================
    # FE HINTS (each ~120-140 s: re-runs FE pipeline + model train)
    # ================================================================

    # ---- FE: Data quality cleanup (Rule #7) ----
    lv0 = (
        "Start with DATA QUALITY clean-up. "
        "(1) Use fe_drop_constant to remove zero-variance and near-constant "
        "features — they add noise without signal "
        "(ref: sklearn VarianceThreshold). "
        "(2) Use fe_null_indicators to create binary is_null flags for every "
        "column with missing values — missingness is often informative when "
        "data is Missing-Not-At-Random "
        "(ref: Chen & Guestrin 2016, XGBoost learned missing-value splits). "
        "(3) Use fe_log_transform on right-skewed numeric columns — this "
        "compresses outlier-heavy tails so histogram-based splits allocate "
        "bins more evenly across the dense region."
    )
    if num_cols:
        lv0 += f" Candidate numeric columns: {', '.join(num_cols[:6])}."

    # ---- FE: Frequency encoding + row null count + clipping (Rule #7, #17) ----
    lv1 = (
        "Apply ENCODING and ROW-LEVEL features. "
        "(1) Use fe_frequency_encode on categorical columns — replacing "
        "categories with their training-set frequency captures "
        "popularity/rarity without target leakage "
        "(ref: standard Kaggle technique, feature-engine library). "
        "(2) Use fe_null_row_count to add a 'number of nulls per row' "
        "feature — rows with many missing values often come from different "
        "data-collection processes and behave differently "
        "(ref: common in Kaggle Tabular Playground top solutions). "
        "(3) Use run_python to clip extreme outliers at the 1st/99th "
        "percentile — outliers waste histogram bins and dominate gradient "
        "computation in boosting "
        "(ref: Tukey 1977, Exploratory Data Analysis — IQR method)."
    )
    if cat_cols:
        lv1 += f" Categorical columns to encode: {', '.join(cat_cols[:6])}."

    # ---- FE: Mutual info feature selection (Rule #22) ----
    lv2 = (
        "FEATURE SELECTION: Use fe_mutual_info to compute the mutual "
        "information between each feature and the target, then DROP features "
        "with MI near zero. MI captures non-linear dependencies unlike "
        "Pearson correlation, so features with zero MI are provably "
        "uninformative and only increase overfitting risk "
        "(ref: Cover & Thomas, Elements of Information Theory; "
        "sklearn mutual_info_classif / mutual_info_regression). "
        "After dropping low-MI features, re-run the pipeline with the "
        "reduced feature set — simpler models often generalize better."
    )

    # ---- FE: Pairwise interactions (Rule #20) ----
    lv3 = (
        "FEATURE INTERACTIONS: Use fe_ratios and fe_differences on the top "
        "correlated numeric column pairs. Tree models cannot natively "
        "represent ratio or additive relationships in a single split — "
        "a ratio like income/debt requires many axis-aligned splits to "
        "approximate, but a pre-computed ratio feature lets the tree find "
        "the pattern in ONE split "
        "(ref: Grinsztajn et al. 2022 showed trees struggle with 'rotated' "
        "features; pre-computing interactions mitigates this weakness). "
        "Also try fe_interactions for multiplicative pairs and "
        "fe_polynomial(degree=2) for quadratic terms of important features."
    )
    if num_cols and len(num_cols) >= 2:
        lv3 += f" Top numeric candidates: {', '.join(num_cols[:5])}."

    # ---- FE: Target encoding (Rule #7, #19) ----
    lv4_parts = (
        "TARGET ENCODING: Use fe_target_encode on categorical columns, "
        "especially high-cardinality ones. This replaces each category with "
        "the K-fold mean of the target for that category — directly "
        "injecting target information while preventing leakage through the "
        "fold-based scheme. For 1000+ level categoricals, one-hot encoding "
        "creates sparse deep trees; target encoding compresses this into a "
        "single informative float "
        "(ref: Prokhorenkova et al. 2018, CatBoost ordered target "
        "statistics; category_encoders library). "
    )
    if cat_cols:
        lv4_parts += (
            f"Apply to: {', '.join(cat_cols[:5])}. "
        )
    lv4_parts += (
        "IMPORTANT: also try fe_count_encode as a leakage-free alternative "
        "when the dataset is small — count encoding encodes category "
        "frequency directly from training counts."
    )

    # ---- FE: GroupBy aggregation features (Rule #20) ----
    lv5 = (
        "AGGREGATION FEATURES: Use fe_agg_features to compute group-level "
        "statistics (mean, std, min, max) of numeric columns grouped by "
        "each categorical column. This captures entity-level context — "
        "'how does this row compare to its group average?' is a powerful "
        "signal because it summarizes local behavior relative to the global "
        "distribution "
        "(ref: featuretools Deep Feature Synthesis, Kanter & "
        "Veeramachaneni 2015; ubiquitous in Kaggle tabular solutions). "
        "Also use fe_group_size to add group-count features: rare groups "
        "behave differently from common ones."
    )
    if cat_cols and num_cols:
        lv5 += (
            f" Group by: {', '.join(cat_cols[:3])}. "
            f"Aggregate: {', '.join(num_cols[:4])}."
        )

    # ---- FE: Rank transforms + power transforms (Rule #20) ----
    lv6 = (
        "RANK & POWER TRANSFORMS: "
        "(1) Use fe_rank_transform to convert numeric features to their "
        "percentile rank — this makes the feature uniformly distributed, "
        "which is optimal for histogram-based splitting and completely "
        "eliminates outlier effects. RankGauss (mapping ranks to Gaussian) "
        "was popularized by Michael Jahrer in his Porto Seguro 1st place "
        "solution "
        "(ref: Jahrer 2017; sklearn QuantileTransformer). "
        "(2) Use fe_power_transform (Yeo-Johnson) on non-normal features "
        "to stabilize variance and make distributions more Gaussian — this "
        "helps tree models by making histogram bins more informative."
    )
    if num_cols:
        lv6 += f" Apply to: {', '.join(num_cols[:5])}."

    # ---- FE: PCA + K-Means cluster labels (Rule #17, demoted) ----
    lv7 = (
        "DIMENSIONALITY REDUCTION as new features: "
        "(1) Use fe_pca_features to run PCA on all numeric features and "
        "add the top-k principal components as NEW columns (keep originals). "
        "PCA captures variance directions spanning multiple correlated "
        "features — tree models struggle with diagonal decision boundaries "
        "but a PCA component rotates data so one split captures the pattern "
        "(ref: Grinsztajn et al. 2022, 'inability to handle rotated "
        "features' is a key tree-model weakness). "
        "(2) Use fe_cluster_labels to run K-Means and add cluster "
        "assignments as a new categorical feature — this captures non-linear "
        "groupings that individual features cannot express "
        "(ref: common in Kaggle Tabular Playground and '30 Days of ML' "
        "competitions)."
    )

    # ---- FE: Deviation-from-group + group-based imputation (Rule #20) ----
    lv8_parts = (
        "DEVIATION-FROM-GROUP features: Use run_python to compute "
        "'value − group_mean' and 'value / group_mean' for numeric features "
        "grouped by categoricals. This normalizes for group-level effects "
        "and exposes within-group anomalies — a salary of 100k is "
        "unremarkable for 'senior engineer' but anomalous for 'intern' "
        "(ref: used extensively in Kaggle credit default, fraud detection, "
        "and insurance competitions). "
    )
    if high_null:
        null_names = ", ".join(c for c, _ in high_null[:4])
        lv8_parts += (
            f"Also apply GROUP-BASED IMPUTATION: instead of filling {null_names} "
            "with global median/mode, group by categoricals and fill with "
            "within-group statistics — global median destroys local structure "
            "(ref: Hastie, Tibshirani & Friedman 2009, Elements of "
            "Statistical Learning, Ch. 9). "
        )
    if structured:
        for c, m in structured[:1]:
            delim = m.get("structured_str", {}).get("delimiter", "/")
            lv8_parts += (
                f"Column '{c}' is a structured string (delimiter='{delim}'). "
                "After fe_split_structured, use the parts for group-based "
                "imputation and cross-column logical null filling."
            )

    # ---- FE: Cross-column logical imputation + binning + polynomial (Rule #20) ----
    lv9_parts = (
        "LOGICAL IMPUTATION + ADVANCED FEATURES: "
    )
    if spend_cols and bool_cols:
        lv9_parts += (
            "Use run_python for CROSS-COLUMN LOGIC: "
            f"spending columns ({', '.join(spend_cols[:3])}) and boolean "
            f"columns ({', '.join(bool_cols[:3])}) may have deterministic "
            "relationships — e.g., a status flag=True always means zero "
            "spending. Exploit this to impute nulls in BOTH directions "
            "(ref: standard in Spaceship Titanic / structured-data "
            "competition top-10 solutions). "
        )
    elif high_null:
        lv9_parts += (
            "Use run_python for cross-column null imputation: examine "
            "correlations between columns with nulls and use deterministic "
            "relationships to fill values instead of statistical estimates "
            "— recovering ground-truth values reduces noise "
            "(ref: consistently used in Kaggle structured-data top solutions). "
        )
    else:
        lv9_parts += (
            "Use run_python for custom preprocessing: examine cross-column "
            "patterns and create domain-specific features from combinations "
            "of existing columns. "
        )
    lv9_parts += (
        "Then try fe_binning on continuous features to capture non-linear "
        "thresholds (e.g., age brackets, income tiers) "
        "(ref: feature-engine EqualFrequencyDiscretiser; common in credit "
        "risk competitions). "
        "And fe_polynomial(degree=2) on the top-3 most important features "
        "to create quadratic interaction terms."
    )

    # ---- FE: Pseudo-labeling (semi-supervised) (Rule #41) ----
    if task in ("binary", "multiclass"):
        lv10 = (
            "PSEUDO-LABELING (semi-supervised learning): Use run_python to: "
            "(1) Train a model on the training set and predict probabilities "
            "on the test set. "
            "(2) Select CONFIDENT test predictions (probability > 0.95 or "
            "< 0.05) as pseudo-labeled samples. "
            "(3) Append these pseudo-labeled rows to the training set and "
            "retrain the model on the augmented dataset. "
            "This leverages the unlabeled test distribution — confident "
            "predictions are likely correct, and including them shifts the "
            "model's decision boundary to better match the test distribution. "
            "Most effective when labeled data is small relative to test "
            "(ref: Lee 2013, 'Pseudo-label: The simple and efficient "
            "semi-supervised learning method'; widely used in Kaggle "
            "competitions with small training sets)."
        )
    else:
        lv10 = (
            "PSEUDO-LABELING (semi-supervised learning): Use run_python to: "
            "(1) Train a model on the training set and predict on test. "
            "(2) Select confident test predictions (where prediction "
            "variance across CV folds is very low) as pseudo-labeled samples. "
            "(3) Append these to training and retrain on the augmented set. "
            "This leverages the unlabeled test distribution to improve the "
            "model's coverage of the feature space "
            "(ref: Lee 2013, 'Pseudo-label'; adapted for regression via "
            "prediction confidence filtering)."
        )

    # ---- FE: Permutation importance pruning ----
    # Rule #22: clean up features you are no longer using
    lv11_fe = (
        "PERMUTATION IMPORTANCE PRUNING (Rule #22): Use run_python to train a quick "
        "LightGBM model, then compute sklearn permutation_importance on a "
        "validation fold. DROP all features with zero or negative "
        "permutation importance — these features are pure noise. "
        "Unlike impurity-based importance (which favors high-cardinality "
        "features), permutation importance measures actual predictive "
        "contribution on held-out data and is model-agnostic "
        "(ref: Breiman 2001; sklearn explicitly recommends permutation "
        "importance over impurity-based importance). "
        "After pruning, re-run the FE pipeline on the reduced set — "
        "removing noise features lets the model focus on real signal."
    )

    # ---- FE: Error analysis ----
    # Rule #26: look for patterns in the measured errors
    lv_error_analysis = (
        "ERROR ANALYSIS (Rule #26): Use run_python to find WHERE the model "
        "fails and create features that fix those errors. "
        "IMPORTANT: The training data is in `train` (DataFrame), labels in `y` "
        "(Series), test data in `test` (DataFrame). Do NOT use `df` or `data`.\n\n"
        "Steps:\n"
        "(1) Train a LightGBM model with 5-fold CV and collect OOF predictions. "
        "Use `train` for features and `y` for labels.\n"
        "(2) Find the MISCLASSIFIED rows (or highest-error rows for regression).\n"
        "(3) Profile the errors: group misclassified rows by each categorical "
        "column and find which categories have the highest error rates. "
        "Also check which numeric ranges have the most errors.\n"
        "(4) Create TARGETED features to fix those patterns:\n"
        "    - Binary flags for high-error categories (e.g., 'is_ErrorProne_Group')\n"
        "    - Binned versions of numeric cols where errors cluster\n"
        "    - Interaction features between the error-prone columns\n"
        "(5) Re-run the model with the new features.\n\n"
        "This is the most powerful technique in production ML: the model "
        "ALREADY KNOWS its mistakes. Creating features that let it fix them "
        "is a direct path to improvement "
        "(ref: Google Rules of ML #26 — 'create new features from error patterns')."
    )

    # ---- FE: Adversarial validation ----
    # Rule #41: look for qualitatively new sources of information
    lv_adversarial = (
        "ADVERSARIAL VALIDATION (Rule #41): Use run_python to detect "
        "train/test distribution shift and create features that help the "
        "model adapt. "
        "IMPORTANT: The training data is in `train`, labels in `y`, "
        "test data in `test`. Do NOT use `df` or `data`.\n\n"
        "Steps:\n"
        "(1) Create a combined dataset: label train rows as 0, test rows as 1.\n"
        "(2) Train a LightGBM classifier to distinguish train from test.\n"
        "(3) If AUC > 0.55, there IS distribution shift. Check which features "
        "the adversarial model uses most (feature_importances_).\n"
        "(4) For high-importance features (the ones that differ between "
        "train and test):\n"
        "    - Add rank transforms (fe_rank_transform) to normalize distributions\n"
        "    - Add binning (fe_binning) to make splits robust to shift\n"
        "    - Consider REMOVING features if they are highly train-specific\n"
        "(5) Use adversarial predictions as sample weights: upweight training "
        "rows that look more like test (higher adversarial score) "
        "to reduce the effective distribution gap.\n\n"
        "This is a qualitatively new source of information — instead of "
        "analyzing feature-target relationships, you analyze feature-split "
        "relationships. Train rows that look unlike test are less useful "
        "(ref: Google Rules of ML #41 — 'qualitatively new sources of information'; "
        "Pan et al. 2010, adversarial validation for domain adaptation)."
    )

    # ================================================================
    # MODEL HINTS (each ~40-60 s: skip FE, re-train only)
    # Rule #40: keep ensembles simple; Rule #25: utilitarian performance
    # ================================================================

    # ---- MODEL: Low LR + stochastic subsampling (Rule #40) ----
    lv12 = (
        "[MODEL] REGULARIZED TRAINING: Try run_model with tune=false but "
        "use run_python to train LightGBM directly with "
        "learning_rate=0.01-0.03, n_estimators=3000, "
        "subsample=0.7, colsample_bytree=0.7, and early_stopping_rounds=100. "
        "Smaller learning rates with more trees produce smoother "
        "approximations with lower test error. Row and column subsampling "
        "decorrelate trees (like Random Forests) and reduce variance "
        "(ref: Friedman 2001 'Greedy function approximation'; "
        "Friedman 2002 'Stochastic gradient boosting'). "
        "This is the single most impactful hyperparameter change."
    )

    # ---- MODEL: Stacking (Rule #40) ----
    lv13 = (
        "[MODEL] MODEL STACKING: Use model_type='stack' in run_model to "
        "train multiple diverse base learners (LightGBM, CatBoost, "
        "XGBoost, Extra Trees) and combine their out-of-fold predictions "
        "via a Ridge/Logistic meta-learner. Stacking exploits the "
        "bias-variance tradeoff across different learners — their errors "
        "are partially decorrelated, so combining them reduces variance "
        "beyond what any single model achieves "
        "(ref: Wolpert 1992 'Stacked generalization'; "
        "used in virtually every winning Kaggle tabular solution). "
        "Make sure to keep the current feature engineering unchanged."
    )

    # ---- MODEL: Optuna tuning (Rule #40) ----
    lv14 = (
        "[MODEL] HYPERPARAMETER OPTIMIZATION: Use run_model with "
        "tune=true, tune_timeout=180 for extended Optuna search. "
        "Optuna uses Tree-structured Parzen Estimators (TPE) to "
        "efficiently explore the hyperparameter space — it prunes "
        "unpromising trials early and focuses on high-performing regions. "
        "Key parameters to tune: max_depth, num_leaves, min_child_samples, "
        "reg_alpha, reg_lambda, learning_rate, subsample, colsample_bytree "
        "(ref: Akiba et al. 2019, 'Optuna: A Next-generation Hyperparameter "
        "Optimization Framework'). "
        "Also consider model_type='tuned' which optimizes a single LightGBM "
        "model specifically."
    )

    # ---- MODEL: Threshold optimization / prediction clipping (Rule #25) ----
    if task in ("binary", "multiclass"):
        lv15 = (
            "[MODEL] THRESHOLD OPTIMIZATION: Use run_python to sweep "
            "classification thresholds on the out-of-fold predictions. "
            "Instead of using 0.5 as the decision boundary, try thresholds "
            "from 0.3 to 0.7 in steps of 0.01 and pick the one that "
            "maximizes the competition metric. The optimal threshold depends "
            "on class balance and the specific metric — for imbalanced "
            "problems 0.5 is almost never optimal. "
            "This is a 'free lunch' that can boost the metric by several "
            "percentage points with zero model changes "
            "(ref: standard competition technique for classification tasks). "
        )
        if scoring and scoring != "accuracy":
            lv15 += (
                f"Competition metric is '{scoring}' — threshold tuning is "
                "especially important when the metric differs from accuracy."
            )
        elif scoring == "accuracy":
            lv15 += (
                "Competition metric is accuracy — threshold optimization "
                "can still help if the class distribution is imbalanced."
            )
    else:
        lv15 = (
            "[MODEL] PREDICTION CLIPPING + BLENDING: Use run_python to: "
            "(1) Clip final predictions to the observed training target "
            "range (min/max) — prevents impossible predictions and reduces "
            "worst-case loss under RMSE/MAE "
            "(ref: standard practice in Kaggle regression competitions). "
            "(2) Try BLENDING: train multiple models separately (e.g., "
            "LightGBM + CatBoost via run_python) and take a weighted "
            "average of predictions, optimizing weights on CV. "
            "When models have comparable performance but different error "
            "patterns, a convex combination achieves lower error than any "
            "individual model "
            "(ref: Netflix Prize 2009; Kaggle competition standard)."
        )
    # ================================================================
    # KITCHEN SINK hint — Rule #16: throw ALL features at the model
    # Rule #16: plan to launch and iterate
    # ================================================================
    # One comprehensive hint that applies EVERY applicable FE technique
    # in a single run_fe_pipeline call.  This replicates what Phase 0 does
    # deterministically: 15+ transforms in one pass, then let the model
    # sort out what matters via feature-importance pruning.

    kitchen_sink_parts = (
        "KITCHEN SINK — add ADVANCED features on top of the existing pipeline.  "
        "Rule #16 of ML: throw many features at the model and let it sort "
        "them out.  The basic transforms (bool convert, structured-string "
        "split, group size, spending aggs, null indicators, null row count, "
        "frequency encoding, log transform) are ALREADY APPLIED.\n\n"
        "DO NOT re-apply those basic transforms.  Instead, ADD these "
        "ADVANCED techniques in a SINGLE run_fe_pipeline call:\n"
        "  1. fe_target_encode — target encoding for categorical columns "
        "(the single highest-impact FE for tabular data)\n"
        "  2. fe_count_encode — count encoding as leakage-free complement\n"
        "  3. fe_rank_transform — percentile rank for all numeric columns\n"
        "  4. fe_ratios — ratios of correlated numeric pairs\n"
        "  5. fe_interactions — multiplicative pairs of top numerics\n"
        "  6. fe_row_stats — row-wise mean/std/min/max of numeric columns\n"
        "  7. fe_categorical_cross — cross top-2 categorical columns\n"
        "  8. fe_polynomial(cols=[top 3 numerics]) — quadratic features\n"
        "  9. fe_power_transform — Yeo-Johnson for non-normal features\n"
        "  10. fe_drop_constant(threshold=0.99) — clean up at the end\n\n"
        "Include ALL 10 steps.  More features >>> fewer features.  "
        "The model ensemble handles noise via regularization and bagging.  "
        "Do NOT be selective or cautious — include everything.\n\n"
        "IMPORTANT: Check column names in your data before calling.  "
        "Use run_python with print(train.columns.tolist()) first if unsure."
    )
    if cat_cols:
        kitchen_sink_parts += f"\nCategorical columns for encoding: {', '.join(cat_cols[:8])}."
    if num_cols:
        kitchen_sink_parts += f"\nNumeric columns for transforms: {', '.join(num_cols[:8])}."

    # ================================================================
    # FINAL ORDERING — Rules of ML priority
    # ================================================================
    # Interleave FE and MODEL hints so we reach MODEL hints within the
    # 600s budget (~10 rounds × 50s/round = 500s available).
    #
    # Ordering rationale (Rules of ML):
    #   - Rule #16: kitchen sink first (applied deterministically in Phase B)
    #   - Rule #26: error analysis early — direct path to improvement
    #   - Rule #7:  heuristic-derived features before exotic transforms
    #   - Rule #40: simple ensemble first, then stacking/tuning
    #   - Rule #41: adversarial validation when standard approaches plateau
    #   - Rule #17: directly observed features before learned features (PCA)
    #   - Rule #22: pruning to clean up noise after feature explosion
    #
    # [MODEL] hints interleaved at positions 5, 8, 10, 12 so they're
    # reachable within the typical 9-12 round budget.
    return [
        kitchen_sink_parts,     #  0  KITCHEN SINK: all FE at once       (Rule #16)
        lv0,                    #  1  Data quality cleanup               (Rule #7)
        lv4_parts,              #  2  Target encoding                    (Rule #7)
        lv_error_analysis,      #  3  Error analysis → targeted features (Rule #26)
        lv3,                    #  4  Feature interactions               (Rule #20)
        lv12,                   #  5  [MODEL] Low LR + subsampling       (Rule #40)
        lv5,                    #  6  Aggregation features               (Rule #20)
        lv9_parts,              #  7  Logical imputation + binning       (Rule #20)
        lv13,                   #  8  [MODEL] Stacking                   (Rule #40)
        lv6,                    #  9  Rank & power transforms            (Rule #20)
        lv14,                   # 10  [MODEL] Optuna tuning              (Rule #40)
        lv1,                    # 11  Freq encoding + clipping           (Rule #7)
        lv15,                   # 12  [MODEL] Threshold / clipping       (Rule #25)
        lv2,                    # 13  Mutual info selection              (Rule #22)
        lv11_fe,                # 14  Permutation pruning                (Rule #22)
        lv_adversarial,         # 15  Adversarial validation             (Rule #41)
        lv8_parts,              # 16  Deviation-from-group               (Rule #20)
        lv10,                   # 17  Pseudo-labeling                    (Rule #41)
        lv7,                    # 18  PCA + K-Means clusters             (Rule #17, demoted)
    ]


def _infer_task_type(eda_meta: dict) -> str:
    """Infer task type from eda_meta fields."""
    is_binary = eda_meta.get("target_is_bool") or eda_meta.get("target_nunique") == 2
    target_dtype = eda_meta.get("target_dtype", "")
    target_nunique = eda_meta.get("target_nunique", 0)
    is_multiclass = (not is_binary and target_nunique > 2
                     and (target_dtype.startswith("int") or target_dtype == "object"))
    return "binary" if is_binary else ("multiclass" if is_multiclass else "regression")


def _build_phase_a_prompt(
    description: str, data_dir: str, eda_meta: dict,
    baseline_cv: float | None = None, baseline_metric: str = "",
) -> str:
    """Phase A system prompt: EDA analysis. Only run_python is available."""
    # Build compact column list from dict keys
    cols = eda_meta.get("columns", {})
    cols_info = ", ".join(list(cols.keys())[:50]) if cols else ""
    n_train = eda_meta.get("n_train_rows", 0)
    n_cols = len(cols)
    task = _infer_task_type(eda_meta)
    target = eda_meta.get("target_col", "unknown")

    baseline_note = ""
    if baseline_cv is not None and baseline_metric:
        baseline_note = f"Reference solution CV: {baseline_metric}={baseline_cv:.6f}. Your goal: beat this."

    return textwrap.dedent(f"""
    # Identity
    You are a data analyst examining a Kaggle competition dataset.
    Your ONLY job in this phase is to run EDA and report column roles.

    # Instructions
    - Call run_python ONCE with ALL THREE EDA functions in a single call
    - Do NOT do feature engineering, preprocessing, or modeling yet
    - Do NOT write submission.csv
    - After EDA completes, respond with text summarizing what you found

    # Example
    <example_call>
    run_python(code=\"\"\"
    eda_profile_columns(train, 'train')
    eda_correlations(train.assign(target=y), 'target')
    eda_null_analysis(train, 'train')
    \"\"\")
    </example_call>

    # Context
    <task>{task}</task>
    <target_col>{target}</target_col>
    <description>{description[:500]}</description>
    <data_dir>{data_dir}</data_dir>
    <train_rows>{n_train}</train_rows>
    <n_columns>{n_cols}</n_columns>
    <columns>{cols_info}</columns>
    <baseline>{baseline_note}</baseline>
    """).strip()


def _build_phase_b_prompt(
    eda_output: str, eda_meta: dict, data_dir: str,
) -> str:
    """Phase B system prompt: Feature Engineering via run_fe_pipeline."""
    from purple.ml_toolkit import TOOLKIT_DOCS
    task = _infer_task_type(eda_meta)
    target = eda_meta.get("target_col", "unknown")

    return textwrap.dedent(f"""
    # Identity
    You are a feature engineer selecting toolkit functions for a Kaggle competition.

    # Instructions
    - Use run_fe_pipeline to specify ALL feature engineering steps as JSON
    - Match column roles from the EDA output to the appropriate fe_* functions
    - Follow this order: data cleaning → encoding → interactions → transforms → cleanup
    - Do NOT write Python code for FE — use the pipeline tool exclusively
    - run_python is available as a fallback for debugging only

    ## Column Role → Function Mapping
    BOOL_STR         → fe_bool_convert
    STRUCTURED_STR   → fe_split_structured
    NUMERIC_AS_STR   → fe_numeric_clean
    ID + structured  → fe_group_size
    datetime         → fe_datetime_features + fe_cyclical_encode
    spending cols    → fe_spending_aggs
    HIGH_CARD        → fe_frequency_encode or fe_target_encode
    CATEGORICAL      → fe_safe_onehot (≤20 values) or fe_frequency_encode
    skewed CONTINUOUS→ fe_log_transform
    correlated pairs → fe_ratios, fe_differences, fe_interactions
    related numerics → fe_row_stats
    category pairs   → fe_categorical_cross
    geo lat/lon      → fe_haversine
    Always include:  → fe_null_row_count, fe_null_indicators, fe_drop_constant

    ## Available fe_* Functions (36 total)
    ### Cleaning: fe_bool_convert, fe_split_structured, fe_numeric_clean, fe_null_indicators, fe_null_row_count, fe_string_features
    ### Numeric Transforms: fe_log_transform, fe_power_transform, fe_rank_transform, fe_math_transforms, fe_cyclical_encode
    ### Interactions: fe_interactions, fe_ratios, fe_differences, fe_row_stats, fe_categorical_cross, fe_polynomial
    ### Encoding: fe_target_encode, fe_frequency_encode, fe_count_encode, fe_safe_onehot
    ### Group/Agg: fe_group_size, fe_agg_features, fe_spending_aggs, fe_percentile_rank_group
    ### Datetime: fe_datetime_features, fe_datetime_elapsed, fe_datetime_flags
    ### Unsupervised: fe_cluster_labels, fe_pca_features
    ### Geo: fe_haversine
    ### Time Series: fe_lag_features, fe_rolling_stats
    ### Selection: fe_mutual_info, fe_drop_constant, fe_binning

    # Example
    <example_call>
    run_fe_pipeline(steps=[
      {{"fn": "fe_bool_convert", "args": "{{\\"cols\\": [\\"CryoSleep\\", \\"VIP\\"]}}" }},
      {{"fn": "fe_split_structured", "args": "{{\\"col\\": \\"Cabin\\", \\"delimiter\\": \\"/\\"}}" }},
      {{"fn": "fe_group_size", "args": "{{\\"id_col\\": \\"PassengerId\\", \\"delimiter\\": \\"_\\"}}" }},
      {{"fn": "fe_spending_aggs", "args": "{{\\"spend_cols\\": [\\"RoomService\\", \\"FoodCourt\\", \\"Spa\\"]}}" }},
      {{"fn": "fe_null_row_count", "args": "{{}}" }},
      {{"fn": "fe_null_indicators", "args": "{{}}" }},
      {{"fn": "fe_ratios", "args": "{{\\"col_pairs\\": [[\\"RoomService\\", \\"TotalSpend\\"]]}}" }},
      {{"fn": "fe_row_stats", "args": "{{\\"cols\\": [\\"RoomService\\", \\"FoodCourt\\", \\"Spa\\", \\"VRDeck\\"]}}" }},
      {{"fn": "fe_log_transform", "args": "{{\\"cols\\": [\\"RoomService\\", \\"FoodCourt\\", \\"Spa\\"]}}" }},
      {{"fn": "fe_frequency_encode", "args": "{{\\"cols\\": [\\"HomePlanet\\", \\"Destination\\"]}}" }},
      {{"fn": "fe_drop_constant", "args": "{{\\"threshold\\": 0.99}}" }}
    ])
    </example_call>

    # Context
    <task>{task}</task>
    <target_col>{target}</target_col>
    <data_dir>{data_dir}</data_dir>
    <eda_output>
    {eda_output[:6000]}
    </eda_output>
    """).strip()


def _build_phase_c_prompt(
    eda_meta: dict, data_dir: str, fe_output: str,
    baseline_cv: float | None = None, baseline_metric: str = "",
    train_shape: str = "", n_features: int = 0,
) -> str:
    """Phase C system prompt: Model training via run_model."""
    task = _infer_task_type(eda_meta)
    target = eda_meta.get("target_col", "target")
    id_col = eda_meta.get("id_col", "")

    baseline_note = ""
    if baseline_cv is not None and baseline_metric:
        baseline_note = f"Reference solution CV: {baseline_metric}={baseline_cv:.6f}. You MUST beat this."

    # Use detected competition metric; fall back to task-based default
    comp_label = eda_meta.get("competition_metric_label", "")
    comp_scoring = eda_meta.get("competition_scoring", "")
    if comp_label:
        scoring_hint = comp_label
    elif task == "regression":
        scoring_hint = "rmse"
    elif task == "multiclass":
        scoring_hint = "accuracy or f1_macro"
    else:
        scoring_hint = "accuracy"

    # Also pass the sklearn scoring string for the example
    example_scoring = comp_scoring if comp_scoring else "accuracy"

    return textwrap.dedent(f"""
    # Identity
    You are an ML engineer selecting model configuration for a Kaggle competition.

    # Instructions
    - Use run_model to train a model, evaluate CV, and write submission.csv
    - Start with model_type="ensemble" unless data is very small (<1000 rows)
    - The run_model tool handles preprocessing automatically
    - Only submit if CV score beats the baseline
    - run_python is available as a fallback for debugging only
    - Include any columns that should be dropped (Name, free-text, ID columns)

    # Example
    <example_call>
    run_model(
      model_type="ensemble",
      task="{task}",
      scoring="{example_scoring}",
      target_dtype="bool",
      tune=false,
      tune_timeout=60,
      drop_cols=["Name"]
    )
    </example_call>

    # Context
    <task>{task}</task>
    <target_col>{target}</target_col>
    <id_col>{id_col}</id_col>
    <scoring_hint>{scoring_hint}</scoring_hint>
    <competition_metric>{comp_scoring}</competition_metric>
    <data_dir>{data_dir}</data_dir>
    <train_shape>{train_shape}</train_shape>
    <n_features>{n_features}</n_features>
    <baseline>{baseline_note}</baseline>
    <fe_summary>
    {fe_output[:2000]}
    </fe_summary>
    """).strip()


MAX_TOOL_ROUNDS = int(os.getenv("PURPLE_MAX_TOOL_ROUNDS", "16"))

EXECUTION_TIMEOUT = int(os.getenv("PURPLE_EXEC_TIMEOUT", "600"))  # seconds
PHASE0_TIMEOUT = int(os.getenv("PURPLE_PHASE0_TIMEOUT", "120"))  # Phase 0 reference cap
PHASE0_SKIP_CV = os.getenv("PURPLE_PHASE0_SKIP_CV", "1").strip() in ("1", "true", "yes")
FAST_CV = os.getenv("PURPLE_FAST_CV", "1").strip() in ("1", "true", "yes")


def _validate_submission(data_dir: Path) -> tuple[bool, str]:
    """Deterministic validation of submission.csv against sample_submission.csv.

    Returns (is_valid, diagnostic_message).  This runs as Python code in
    the agent process — the LLM never generates this logic.
    """
    import csv

    submission_path = data_dir / "submission.csv"
    sample_path = data_dir / "sample_submission.csv"

    if not submission_path.exists():
        return False, "VALIDATION FAILED: submission.csv was not written to disk."

    if not sample_path.exists():
        # No sample to compare against — accept any file that exists.
        return True, "Validation skipped: no sample_submission.csv to compare."

    try:
        import pandas as pd
        sub = pd.read_csv(submission_path)
        samp = pd.read_csv(sample_path)
    except Exception as exc:
        return False, f"VALIDATION FAILED: could not read CSV files: {exc}"

    issues: list[str] = []

    # 1. Column check
    if list(sub.columns) != list(samp.columns):
        issues.append(
            f"Column mismatch: submission has {list(sub.columns)}, "
            f"expected {list(samp.columns)}."
        )

    # 2. Row count
    if len(sub) != len(samp):
        issues.append(
            f"Row count mismatch: submission has {len(sub)} rows, "
            f"expected {len(samp)} rows."
        )

    # 3. ID column match (first column)
    id_col = samp.columns[0]
    if id_col in sub.columns:
        sub_ids = set(sub[id_col].astype(str))
        samp_ids = set(samp[id_col].astype(str))
        if sub_ids != samp_ids:
            missing = samp_ids - sub_ids
            extra = sub_ids - samp_ids
            detail = []
            if missing:
                detail.append(f"{len(missing)} expected IDs missing from submission")
                detail.append(f"  examples: {list(missing)[:5]}")
            if extra:
                detail.append(f"{len(extra)} unexpected IDs in submission")
                detail.append(f"  examples: {list(extra)[:5]}")
            issues.append(
                f"{id_col} values do not match sample_submission.csv:\n"
                + "\n".join(detail)
            )
    else:
        issues.append(f"ID column '{id_col}' not found in submission.")

    # 4. NaN check in non-ID columns
    value_cols = [c for c in sub.columns if c != id_col]
    for col in value_cols:
        nan_count = sub[col].isna().sum()
        if nan_count > 0:
            issues.append(f"Column '{col}' has {nan_count} NaN values.")

    if issues:
        msg = "VALIDATION FAILED — the agent detected these problems:\n"
        for i, issue in enumerate(issues, 1):
            msg += f"  {i}. {issue}\n"
        return False, msg

    return True, (
        f"Validation passed: {len(sub)} rows, "
        f"columns {list(sub.columns)}, ID column '{id_col}' matches."
    )


async def _run_code(code: str, data_dir: Path, timeout: int | None = None) -> tuple[bool, str]:
    """Write *code* to a temp file and execute it.  Returns (success, output)."""
    _timeout = timeout if timeout is not None else EXECUTION_TIMEOUT
    script = data_dir.parent / "_ml_solution.py"
    script.write_text(code)

    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable, str(script),
            cwd=str(data_dir.parent),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        try:
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=_timeout)
        except asyncio.TimeoutError:
            proc.kill()
            return False, f"Execution timed out after {_timeout}s"

        output = stdout.decode(errors="replace")
        logger.info("Execution output:\n%s", output[-2000:])
        return proc.returncode == 0, output

    except Exception as exc:
        return False, str(exc)


# ---------------------------------------------------------------------------
# Agent executor
# ---------------------------------------------------------------------------

class MLPurpleAgent(AgentExecutor):
    """OpenAI-backed ML coding agent for MLE-Bench competitions."""

    def __init__(self, *, debug: bool = False) -> None:
        super().__init__()
        self._debug = debug
        api_key = os.getenv("OPENAI_API_KEY", "")
        self._model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self._client = AsyncOpenAI(api_key=api_key) if api_key else None

    # ------------------------------------------------------------------
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        try:
            await self._execute(context, event_queue)
        except Exception as exc:
            logger.exception("Unhandled exception in execute(): %s", exc)
            await event_queue.enqueue_event(
                TaskStatusUpdateEvent(
                    status=TaskStatus(
                        state=TaskState.failed,
                        message=new_agent_text_message(
                            f"Purple agent internal error: {exc}",
                            context_id=context.context_id,
                        ),
                    ),
                    context_id=context.context_id,
                    task_id=context.task_id,
                    final=True,
                )
            )

    async def _status(
        self,
        event_queue: EventQueue,
        context: RequestContext,
        text: str,
        final: bool = False,
        state: TaskState = TaskState.working,
    ) -> None:
        """Enqueue a TaskStatusUpdateEvent (keeps the stream open when final=False)."""
        await event_queue.enqueue_event(
            TaskStatusUpdateEvent(
                status=TaskStatus(
                    state=state,
                    message=new_agent_text_message(text, context_id=context.context_id),
                ),
                context_id=context.context_id,
                task_id=context.task_id,
                final=final,
            )
        )

    async def _execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        await self._status(event_queue, context, "Purple ML agent starting…")

        # ---- unpack the incoming message ----
        instructions_text = ""
        tar_bytes: bytes | None = None

        message = context.message
        if message:
            for part in message.parts:
                root = part.root
                if isinstance(root, TextPart):
                    instructions_text += "\n" + root.text
                elif isinstance(root, FilePart):
                    file_data = root.file
                    if isinstance(file_data, FileWithBytes):
                        tar_bytes = base64.b64decode(file_data.bytes)

        if not tar_bytes:
            await self._status(event_queue, context, "Error: no competition.tar.gz received.",
                               final=True, state=TaskState.failed)
            return

        if not self._client:
            await self._status(event_queue, context, "Error: OPENAI_API_KEY not set.",
                               final=True, state=TaskState.failed)
            return

        # ---- extract competition data ----
        with tempfile.TemporaryDirectory() as tmp:
            work_dir = Path(tmp)
            _extract_tar(tar_bytes, work_dir)
            data_dir = _find_data_dir(work_dir)
            description = _read_description(work_dir)

            if self._debug:
                logger.info("Data directory: %s", data_dir)
                logger.info("Description snippet: %s", description[:300])

            # ---- iterative code generation + execution ----
            submission_path = data_dir / "submission.csv"

            # ---- Step 0 (deterministic gate): EDA before any LLM calls ----
            await self._status(event_queue, context, "Running EDA on data files…")
            eda_report, eda_meta = _run_eda(data_dir)
            if self._debug:
                logger.info("EDA report:\n%s", eda_report)

            # ---- Detect competition metric from description + EDA ----
            comp_scoring, comp_metric_label = _detect_competition_metric(description, eda_meta)
            eda_meta["competition_scoring"] = comp_scoring
            eda_meta["competition_metric_label"] = comp_metric_label
            logger.info("Detected competition metric: %s (sklearn: %s)", comp_metric_label, comp_scoring)

            # ---- Phase 0 (deterministic): generate few-shot reference ----
            phase0_script = _generate_solution_script(eda_meta, str(data_dir))
            phase0_cv: float | None = None
            phase0_cv_std: float | None = None
            phase0_metric: str = ""
            phase0_output: str = ""
            if phase0_script:
                await self._status(event_queue, context, "Phase 0: running reference solution…")
                logger.info("Phase 0 script (%d chars)", len(phase0_script))
                _p0_t0 = time.monotonic()
                p0_success, p0_output = await _run_code(phase0_script, data_dir, timeout=PHASE0_TIMEOUT)
                _p0_elapsed = time.monotonic() - _p0_t0
                logger.info("⏱ Phase 0: %.1fs (%s)", _p0_elapsed, "OK" if p0_success else "FAILED")
                phase0_output = p0_output or ""
                if p0_success:
                    for line in phase0_output.splitlines():
                        if line.startswith("CV_METRIC="):
                            phase0_metric = line.split("=", 1)[1].strip()
                        elif line.startswith("CV_SCORE="):
                            try:
                                phase0_cv = float(line.split("=", 1)[1])
                            except (ValueError, IndexError):
                                pass
                        elif line.startswith("CV_STD="):
                            try:
                                phase0_cv_std = float(line.split("=", 1)[1])
                            except (ValueError, IndexError):
                                pass
                    logger.info(
                        "Phase 0 reference: CV %s=%.4f (std=%.4f)",
                        phase0_metric, phase0_cv if phase0_cv is not None else -1.0,
                        phase0_cv_std if phase0_cv_std is not None else 0.0,
                    )
                else:
                    logger.warning("Phase 0 execution failed: %s",
                                   phase0_output[-500:])
                # Clear Phase 0 submission — agent must produce its own
                submission_path.unlink(missing_ok=True)
            else:
                logger.info("Phase 0: skipped (insufficient metadata)")

            # ---- Step 1: planning call (always runs, with Phase 0 context if available) ----
            await self._status(event_queue, context, "Generating plan…")
            planning_prompt = _build_planning_prompt(
                instructions_text.strip(), description, str(data_dir),
                eda_report=eda_report,
                baseline_script=phase0_script or "",
                baseline_cv=phase0_cv,
                baseline_metric=phase0_metric,
                baseline_output=phase0_output,
            )
            try:
                logger.info("Planning call to %s…", self._model)
                plan_resp = await self._client.chat.completions.create(
                    model=self._model,
                    messages=[{"role": "user", "content": planning_prompt}],
                    temperature=0.0,
                    max_completion_tokens=1024,
                )
                plan = (plan_resp.choices[0].message.content or "").strip()
                logger.info("Plan:\n%s", plan)
            except Exception as exc:
                logger.warning("Planning call failed: %s", exc)
                await self._status(event_queue, context, f"OpenAI error during planning: {exc}",
                                   final=True, state=TaskState.failed)
                return

            # ================================================================
            # AGENT STRATEGY — Phase-specific prompts and tools
            #   Phase A: EDA (run_python only)
            #   Phase B: Feature Engineering (run_fe_pipeline + run_python)
            #   Phase C: Model (run_model + run_python)
            #   Each phase gets a focused system prompt with only relevant
            #   tools, following OpenAI best practices for GPT models.
            # ================================================================

            # Pre-seed exec globals with setup data + ML toolkit
            from purple.ml_toolkit import TOOLKIT_FUNCTIONS
            exec_globals: dict = {"__builtins__": __builtins__}
            exec_globals.update(TOOLKIT_FUNCTIONS)
            setup_script = _generate_setup_script(eda_meta, str(data_dir))
            if setup_script:
                try:
                    exec(setup_script, exec_globals)
                    logger.info(
                        "Setup script loaded: train=%s, test=%s, y=%s",
                        getattr(exec_globals.get("train"), "shape", "?"),
                        getattr(exec_globals.get("test"), "shape", "?"),
                        getattr(exec_globals.get("y"), "shape", "?"),
                    )
                except Exception as exc:
                    logger.warning("Setup script failed: %s", exc)

            # ---- In-process Phase 0 FE: apply basic transforms directly ----
            # This replaces the need for Phase B to redo basic FE via LLM.
            _p0fe_t0 = time.monotonic()
            _p0_fe_steps = _build_phase0_fe_steps(eda_meta)
            if _p0_fe_steps and exec_globals.get("train") is not None:
                _p0_fe_result = _exec_fe_pipeline(_p0_fe_steps, exec_globals)
                _p0fe_elapsed = time.monotonic() - _p0fe_t0
                logger.info(
                    "⏱ In-process Phase 0 FE: %.1fs, %d steps → %s",
                    _p0fe_elapsed, len(_p0_fe_steps),
                    _p0_fe_result[:200] if _p0_fe_result else "no output",
                )
            else:
                logger.info("In-process Phase 0 FE: skipped (no steps or no data)")

            if phase0_script:
                exec_globals["BASELINE_SCRIPT"] = phase0_script
            # Clear Phase 0 submission so early-exit check doesn't fire
            submission_path.unlink(missing_ok=True)
            start_time = time.monotonic()
            total_rounds = 0

            # Helper: run one phase of the agent loop
            async def _run_phase(
                phase_name: str,
                system_prompt: str,
                user_message: str,
                tools: list[dict],
                max_rounds: int,
            ) -> tuple[str, int]:
                """Run a phase of the agent loop. Returns (last_output, rounds_used)."""
                nonlocal total_rounds
                messages: list[dict] = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ]
                consecutive_errors: dict[str, int] = {}
                last_output = ""
                rounds_used = 0

                for round_num in range(1, max_rounds + 1):
                    total_rounds += 1
                    rounds_used = round_num
                    elapsed = time.monotonic() - start_time
                    if elapsed > EXECUTION_TIMEOUT:
                        logger.warning("%s timed out after %.0fs", phase_name, elapsed)
                        break

                    await self._status(
                        event_queue, context,
                        f"{phase_name} round {round_num}/{max_rounds}…",
                    )

                    try:
                        response = await self._client.chat.completions.create(
                            model=self._model,
                            messages=messages,
                            tools=tools,
                            temperature=0.0,
                            max_completion_tokens=4096,
                        )
                    except Exception as exc:
                        logger.warning("OpenAI request failed on %s round %d: %s",
                                       phase_name, round_num, exc)
                        return f"ERROR: {exc}", rounds_used

                    choice = response.choices[0]

                    # Build assistant message for history
                    assistant_msg: dict = {
                        "role": "assistant",
                        "content": choice.message.content,
                    }
                    if choice.message.tool_calls:
                        assistant_msg["tool_calls"] = [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments,
                                },
                            }
                            for tc in choice.message.tool_calls
                        ]
                    messages.append(assistant_msg)

                    # If no tool calls, model finished with text
                    if not choice.message.tool_calls:
                        last_output = choice.message.content or ""
                        logger.info(
                            "%s finished with text on round %d: %s",
                            phase_name, round_num, last_output[:200],
                        )
                        break

                    # Execute tool calls
                    had_model_run = False
                    for tc in choice.message.tool_calls:
                        fname = tc.function.name
                        try:
                            args = json.loads(tc.function.arguments)
                        except json.JSONDecodeError:
                            args = {}

                        if fname == "run_python":
                            code_snippet = args.get("code", "")
                            logger.info("%s run_python:\n%s", phase_name, code_snippet[:500])
                            try:
                                result = await asyncio.wait_for(
                                    asyncio.to_thread(
                                        _exec_run_python, code_snippet, exec_globals,
                                    ),
                                    timeout=120.0,
                                )
                            except asyncio.TimeoutError:
                                result = (
                                    "ERROR: Code execution timed out after 120 seconds."
                                )
                            if "BASELINE_SCRIPT" in code_snippet:
                                submission_path.unlink(missing_ok=True)
                        elif fname == "run_fe_pipeline":
                            steps = args.get("steps", [])
                            logger.info("%s run_fe_pipeline: %d steps",
                                        phase_name, len(steps))
                            try:
                                result = await asyncio.wait_for(
                                    asyncio.to_thread(
                                        _exec_fe_pipeline, steps, exec_globals,
                                    ),
                                    timeout=120.0,
                                )
                            except asyncio.TimeoutError:
                                result = "ERROR: FE pipeline timed out after 120 seconds."
                        elif fname == "run_model":
                            had_model_run = True
                            logger.info("%s run_model: %s", phase_name, args)
                            try:
                                result = await asyncio.wait_for(
                                    asyncio.to_thread(
                                        _exec_model_pipeline, args, exec_globals,
                                        str(data_dir),
                                    ),
                                    timeout=300.0,
                                )
                            except asyncio.TimeoutError:
                                result = "ERROR: Model pipeline timed out after 300 seconds."
                        else:
                            result = f"Unknown tool: {fname}"

                        last_output = result
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": result,
                        })
                        logger.info("%s tool %s → %s", phase_name, fname, result[:300])

                        # Stuck-loop detection
                        if result.startswith("ERROR:"):
                            err_key = result[:200]
                            consecutive_errors[err_key] = consecutive_errors.get(err_key, 0) + 1
                            if consecutive_errors[err_key] >= 3:
                                messages.append({
                                    "role": "user",
                                    "content": (
                                        "You have repeated the same error 3+ times. "
                                        "STOP and try a completely different approach."
                                    ),
                                })
                                consecutive_errors.clear()
                        else:
                            consecutive_errors.clear()

                    # Check submission after run_model
                    if had_model_run and submission_path.exists():
                        valid, diag = _validate_submission(data_dir)
                        if valid:
                            logger.info("%s submission validated: %s",
                                        phase_name, diag)
                            break
                        else:
                            logger.warning("%s submission failed: %s",
                                           phase_name, diag)
                            submission_path.unlink(missing_ok=True)
                            messages.append({
                                "role": "user",
                                "content": (
                                    f"submission.csv FAILED validation:\n\n"
                                    f"{diag}\n\nPlease fix and rewrite."
                                ),
                            })

                return last_output, rounds_used

            # ================================================================
            # PHASE A: EDA (max 3 rounds, run_python only)
            # ================================================================
            await self._status(event_queue, context,
                               "Phase A: Running EDA analysis…")
            phase_a_prompt = _build_phase_a_prompt(
                description, str(data_dir), eda_meta,
                baseline_cv=phase0_cv, baseline_metric=phase0_metric,
            )
            phase_a_user = (
                "Raw data is pre-loaded: train, test, y, ids, DATA_DIR, TASK, TARGET_COL, ID_COL.\n"
                "Toolkit functions (eda_*, fe_*, preprocess, build_model, etc.) are pre-loaded.\n\n"
                "Run EDA now: call run_python with eda_profile_columns, eda_correlations, "
                "and eda_null_analysis in ONE call."
            )
            _phaseA_t0 = time.monotonic()
            eda_output, _ = await _run_phase(
                "Phase A", phase_a_prompt, phase_a_user,
                _TOOLS_PHASE_A, max_rounds=3,
            )
            _phaseA_elapsed = time.monotonic() - _phaseA_t0
            logger.info("⏱ Phase A complete: %.1fs, EDA output: %d chars", _phaseA_elapsed, len(eda_output))

            # ================================================================
            # PHASE B: Feature Engineering — deterministic kitchen-sink
            # Basic FE already applied in-process above.  Now apply
            # ADVANCED transforms (target encoding, rank, interactions, etc.)
            # deterministically, skipping the LLM entirely.
            # ================================================================
            await self._status(event_queue, context,
                               "Phase B: Applying advanced feature engineering…")
            _phaseB_t0 = time.monotonic()
            _ks_steps = _build_kitchen_sink_fe_steps(eda_meta)
            if _ks_steps and exec_globals.get("train") is not None:
                fe_output = _exec_fe_pipeline(_ks_steps, exec_globals)
            else:
                fe_output = "No kitchen-sink steps applicable."
            _phaseB_elapsed = time.monotonic() - _phaseB_t0
            logger.info("⏱ Phase B (deterministic): %.1fs, %d steps → %s",
                        _phaseB_elapsed, len(_ks_steps),
                        fe_output[:200] if fe_output else "no output")

            # ---- Feature importance pruning (mirrors Phase 0) ----
            # Train a quick LightGBM, drop zero-importance features.
            _prune_train = exec_globals.get("train")
            _prune_y = exec_globals.get("y")
            if _prune_train is not None and _prune_y is not None:
                try:
                    from lightgbm import LGBMClassifier, LGBMRegressor
                    import numpy as _np_prune

                    _n_before = _prune_train.shape[1]
                    _task_type = _infer_task_type(eda_meta)

                    # Preprocess for pruning: fill nulls, encode categoricals
                    _pr_train = _prune_train.copy()
                    for _c in _pr_train.columns:
                        _dk = _pr_train[_c].dtype.kind
                        if _dk in ("O", "U", "S") or str(_pr_train[_c].dtype) in ("category", "string", "str"):
                            # object, unicode, byte-string, category, StringDtype
                            _pr_train[_c] = _pr_train[_c].astype("category").cat.codes
                        elif _dk == "b":
                            _pr_train[_c] = _pr_train[_c].astype(int)
                        elif _dk == "f":
                            _pr_train[_c] = _pr_train[_c].fillna(_pr_train[_c].median())
                        elif _dk == "i":
                            _pr_train[_c] = _pr_train[_c].fillna(0)

                    if _task_type in ("binary", "multiclass"):
                        _prune_model = LGBMClassifier(
                            n_estimators=100, learning_rate=0.1,
                            num_leaves=31, random_state=42, verbose=-1,
                            n_jobs=os.cpu_count() or 4,
                        )
                    else:
                        _prune_model = LGBMRegressor(
                            n_estimators=100, learning_rate=0.1,
                            num_leaves=31, random_state=42, verbose=-1,
                            n_jobs=os.cpu_count() or 4,
                        )
                    _prune_model.fit(_pr_train, _prune_y)
                    _importances = _prune_model.feature_importances_
                    # Conservative pruning: only drop features with strictly
                    # zero importance AND cap drops at 30% of total features.
                    # A single 100-tree fit is noisy — marginal features may
                    # randomly get zero importance due to sampling.
                    _zero_mask = _importances == 0
                    _n_zero = int(_zero_mask.sum())
                    _max_drop = max(1, int(_n_before * 0.20))
                    if _n_zero > _max_drop:
                        # Too many zeros — only drop the lowest-variance ones
                        _zero_idx = _np_prune.where(_zero_mask)[0]
                        _variances = _pr_train.iloc[:, _zero_idx].var().values
                        # Sort by variance ascending; drop only the _max_drop least-variable
                        _sorted_asc = _zero_idx[_np_prune.argsort(_variances)]
                        _to_drop = set(_sorted_asc[:_max_drop])
                        _keep_mask = _np_prune.array([
                            i not in _to_drop for i in range(_n_before)
                        ])
                    else:
                        _keep_mask = ~_zero_mask
                    _keep_cols = list(_prune_train.columns[_keep_mask])
                    _drop_cols = list(_prune_train.columns[~_keep_mask])

                    if _drop_cols:
                        exec_globals["train"] = _prune_train[_keep_cols]
                        exec_globals["test"] = exec_globals["test"][_keep_cols]
                        logger.info(
                            "⏱ Phase B pruning: %d → %d features (dropped %d: %s)",
                            _n_before, len(_keep_cols), len(_drop_cols),
                            _drop_cols[:10],
                        )
                    else:
                        logger.info("⏱ Phase B pruning: %d features, none dropped", _n_before)

                    del _prune_model, _pr_train, _importances, _keep_mask
                except Exception as _prune_err:
                    logger.warning("Phase B pruning failed: %s", _prune_err)

            # Get current train shape for Phase C context
            train_obj = exec_globals.get("train")
            train_shape_str = str(getattr(train_obj, "shape", "unknown"))
            n_features = train_obj.shape[1] if train_obj is not None and hasattr(train_obj, "shape") else 0

            # ================================================================
            # PHASE C: Model (LLM chooses config — adapts to the dataset)
            # The LLM's config is cached and reused for Phase D direct eval.
            # ================================================================
            await self._status(event_queue, context,
                               "Phase C: Model training and submission…")
            phase_c_prompt = _build_phase_c_prompt(
                eda_meta, str(data_dir), fe_output,
                baseline_cv=phase0_cv, baseline_metric=phase0_metric,
                train_shape=train_shape_str, n_features=n_features,
            )
            phase_c_user = (
                "Feature engineering is complete. "
                "Use run_model to train, evaluate CV, and write submission.csv."
            )
            _phaseC_t0 = time.monotonic()
            model_output, _ = await _run_phase(
                "Phase C", phase_c_prompt, phase_c_user,
                _TOOLS_PHASE_C, max_rounds=4,
            )
            _phaseC_elapsed = time.monotonic() - _phaseC_t0

            # Cache the model config the LLM chose for Phase D direct eval
            _cached_model_config = exec_globals.get("_last_model_config")

            elapsed = time.monotonic() - start_time
            logger.info(
                "⏱ Phase C complete: %.1fs | Agent loop A+B+C: %d rounds, %.1fs elapsed",
                _phaseC_elapsed, total_rounds, elapsed,
            )

            # ================================================================
            # PHASE D: Progressive iteration with hint ladder
            # Keep iterating FE+Model with escalating hints until agent
            # beats Phase 0 CV, time runs out, or hints are exhausted.
            # ================================================================
            def _parse_cv(output: str) -> tuple[float | None, float | None]:
                """Extract CV_SCORE and CV_STD from model output."""
                score, std = None, None
                for line in (output or "").splitlines():
                    if line.startswith("CV_SCORE="):
                        try:
                            score = float(line.split("=", 1)[1])
                        except (ValueError, IndexError):
                            pass
                    elif line.startswith("CV_STD="):
                        try:
                            std = float(line.split("=", 1)[1])
                        except (ValueError, IndexError):
                            pass
                return score, std

            agent_cv, agent_cv_std = _parse_cv(model_output)
            best_cv = agent_cv
            best_cv_std = agent_cv_std
            best_output = model_output
            _fast_cv_improved = False  # Track if Phase D improved in fast_cv mode
            _best_is_stacking = False  # Only stacking submissions beat the ensemble
            # Track whether agent ever produced a valid submission
            if submission_path.exists():
                best_submission_bytes = submission_path.read_bytes()
            else:
                best_submission_bytes = None

            # ---- FAST_CV: re-baseline with LGBM-only CV ----
            # Phase C used ensemble CV which is ~0.003 higher than LGBM-only.
            # Re-measure baseline with LGBM so Phase D comparisons are fair.
            if FAST_CV and agent_cv is not None:
                from purple.ml_toolkit import TOOLKIT_FUNCTIONS as _TK
                try:
                    _lgbm_model = _TK["build_model"](task=_infer_task_type(eda_meta))
                    _scoring = comp_scoring or "accuracy"
                    _lgbm_cv = _TK["evaluate_cv"](
                        _lgbm_model, exec_globals["train"], exec_globals["y"],
                        scoring=_scoring,
                    )
                    logger.info(
                        "Fast CV re-baseline: ensemble CV=%.4f → LGBM CV=%.4f (delta=%.4f)",
                        agent_cv, _lgbm_cv, agent_cv - _lgbm_cv,
                    )
                    best_cv = _lgbm_cv
                    agent_cv = _lgbm_cv
                except Exception as exc:
                    logger.warning("Fast CV re-baseline failed: %s", exc)

            hints = _build_iteration_hints(eda_meta, str(data_dir))
            # Skip hint 0 (kitchen sink) — already applied in Phase B deterministically
            hints = hints[1:]
            hint_round = 0

            # ---- Cached model config for direct eval in FE rounds ----
            # Reuse whatever the LLM chose in Phase C.  Falls back to a
            # sensible default derived from eda_meta if Phase C didn't cache.
            if _cached_model_config:
                _std_model_config = _cached_model_config
                logger.info("Phase D: using LLM's cached model config: %s", _std_model_config)
            else:
                _task = _infer_task_type(eda_meta)
                _target_dtype = "bool" if eda_meta.get("target_is_bool") else (
                    "int" if eda_meta.get("target_dtype", "").startswith("int") else "float"
                )
                _std_model_config = {
                    "model_type": "ensemble",
                    "task": _task,
                    "scoring": comp_scoring,
                    "target_dtype": _target_dtype,
                    "tune": False,
                    "tune_timeout": 60,
                    "drop_cols": [eda_meta.get("id_col", "")],
                }
                logger.info("Phase D: using fallback model config: %s", _std_model_config)

            # ---- Cumulative state: snapshot best feature-engineered data ----
            # When a round improves CV, we save (train, test) copies.
            # Subsequent rounds build ON TOP of the best state instead of
            # restarting from raw data every time.  This implements
            # Rule #16: features compound — more features >>> fewer features.
            _snap_train = None
            _snap_test = None
            # Save initial Phase B/C state as the starting snapshot
            _init_train = exec_globals.get("train")
            _init_test = exec_globals.get("test")
            if _init_train is not None and _init_test is not None:
                _snap_train = _init_train.copy()
                _snap_test = _init_test.copy()
                logger.info(
                    "Phase D: initial snapshot saved (train=%s, test=%s)",
                    _snap_train.shape, _snap_test.shape,
                )

            # Always iterate through ALL hint levels to maximize CV.
            # Don't stop early when agent beats Phase 0 — keep improving.
            # If Phase 0 failed/timed out, use 1.0 as target so hints still run.
            if phase0_cv is None:
                phase0_cv = 1.0
                phase0_metric = phase0_metric or "accuracy"
                logger.info("Phase D: Phase 0 unavailable, using target CV=1.0 so hint ladder runs")

            # ---- Timing metrics for Phase D ----
            _phase_d_metrics: list[dict] = []

            while (agent_cv is not None
                    and hint_round < len(hints)):
                remaining = EXECUTION_TIMEOUT - (time.monotonic() - start_time)
                if remaining < 90:
                    logger.info("Phase D: time budget exhausted (%.0fs left), stopping", remaining)
                    break

                hint = hints[hint_round]
                hint_round += 1
                logger.info(
                    "Phase D round %d/%d: best CV %.4f < target %.4f — hint level %d",
                    hint_round, len(hints), best_cv, phase0_cv, hint_round - 1,
                )
                await self._status(
                    event_queue, context,
                    f"Phase D round {hint_round}/{len(hints)}: iterating with hint level {hint_round - 1}…",
                )

                try:
                    _round_t0 = time.monotonic()
                    submission_path.unlink(missing_ok=True)

                    # Determine if this is FE-focused or model-focused hint
                    is_model_hint = hint.startswith("[MODEL] ")

                    if not is_model_hint:
                        # ---- FE hint: cumulative state ----
                        # Reload base variables (y, ids, constants) then
                        # restore the best feature-engineered snapshot so
                        # new FE builds ON TOP of prior winning features.
                        setup_script = _generate_setup_script(eda_meta, str(data_dir))
                        if setup_script:
                            try:
                                exec(setup_script, exec_globals)
                            except Exception:
                                pass

                        # Restore cumulative snapshot if available
                        has_snapshot = _snap_train is not None and _snap_test is not None
                        if has_snapshot:
                            exec_globals["train"] = _snap_train.copy()
                            exec_globals["test"] = _snap_test.copy()

                        phase_b_prompt_d = _build_phase_b_prompt(
                            eda_output, eda_meta, str(data_dir),
                        )
                        if has_snapshot:
                            snap_ncols = _snap_train.shape[1]
                            phase_b_user_d = (
                                f"Your best CV so far: {phase0_metric}={best_cv:.6f}. "
                                f"Target to beat: {phase0_cv:.6f}.\n\n"
                                f"Your best feature set ({snap_ncols} columns) is "
                                "ALREADY LOADED from prior rounds.  "
                                "DO NOT re-apply transforms that are already in the data.  "
                                f"ADD ONLY the new technique on top:\n\n"
                                f"HINT: {hint}\n\n"
                                "Use run_fe_pipeline with ONLY the new steps."
                            )
                        else:
                            phase_b_user_d = (
                                f"Your best CV so far: {phase0_metric}={best_cv:.6f}. "
                                f"Target to beat: {phase0_cv:.6f}.\n\n"
                                f"HINT: {hint}\n\n"
                                "Apply the hint and use run_fe_pipeline with improved steps."
                            )
                        _fe_t0 = time.monotonic()
                        fe_output_d, _ = await _run_phase(
                            f"Phase D{hint_round}-FE", phase_b_prompt_d, phase_b_user_d,
                            _TOOLS_PHASE_B, max_rounds=3,
                        )
                        _fe_elapsed = time.monotonic() - _fe_t0

                        train_obj = exec_globals.get("train")
                        ts_d = str(getattr(train_obj, "shape", "unknown"))
                        nf_d = train_obj.shape[1] if train_obj is not None and hasattr(train_obj, "shape") else 0

                        # Direct model eval — skip LLM, use standard ensemble config
                        fe_output_d = fe_output
                    else:
                        # ---- MODEL hint: use best feature set ----
                        # Restore the best snapshot so MODEL hints train on
                        # feature-engineered data, not raw data.
                        setup_script = _generate_setup_script(eda_meta, str(data_dir))
                        if setup_script:
                            try:
                                exec(setup_script, exec_globals)
                            except Exception:
                                pass
                        if _snap_train is not None and _snap_test is not None:
                            exec_globals["train"] = _snap_train.copy()
                            exec_globals["test"] = _snap_test.copy()

                        fe_output_d = fe_output
                        train_obj = exec_globals.get("train")
                        ts_d = str(getattr(train_obj, "shape", "unknown"))
                        nf_d = train_obj.shape[1] if train_obj is not None and hasattr(train_obj, "shape") else 0

                        phase_c_prompt_d = _build_phase_c_prompt(
                            eda_meta, str(data_dir), fe_output_d,
                            baseline_cv=phase0_cv, baseline_metric=phase0_metric,
                            train_shape=ts_d, n_features=nf_d,
                        )
                        phase_c_user_d = (
                            f"Your best CV: {phase0_metric}={best_cv:.6f}. "
                            f"Target: {phase0_cv:.6f}.\n\n"
                            f"HINT: {hint}"
                        )

                    _model_t0 = time.monotonic()
                    if not is_model_hint:
                        # Direct model eval — no LLM, just run the pipeline
                        submission_path.unlink(missing_ok=True)
                        # After stacking wins, FE rounds must re-stack to get
                        # a comparable OOF score (LGBM-only can never beat it).
                        if _best_is_stacking:
                            _restack_config = dict(_std_model_config, model_type="stack")
                            _use_fast_cv = False
                        else:
                            _restack_config = _std_model_config
                            _use_fast_cv = FAST_CV
                        try:
                            model_output_d = await asyncio.wait_for(
                                asyncio.to_thread(
                                    _exec_model_pipeline, _restack_config,
                                    exec_globals, str(data_dir),
                                    _use_fast_cv,
                                ),
                                timeout=300.0,
                            )
                        except asyncio.TimeoutError:
                            model_output_d = "ERROR: Model pipeline timed out after 300 seconds."
                        logger.info(
                            "Phase D%d-Model (direct): %s",
                            hint_round, model_output_d[:300],
                        )
                    else:
                        # MODEL hints use LLM for creative model config
                        model_output_d, _ = await _run_phase(
                            f"Phase D{hint_round}-Model", phase_c_prompt_d, phase_c_user_d,
                            _TOOLS_PHASE_C, max_rounds=3,
                        )
                    _model_elapsed = time.monotonic() - _model_t0
                    new_cv, new_cv_std = _parse_cv(model_output_d)

                    _round_elapsed = time.monotonic() - _round_t0
                    _round_type = "MODEL" if is_model_hint else "FE"
                    _fe_time = 0.0 if is_model_hint else _fe_elapsed
                    _metric_entry = {
                        "round": hint_round, "type": _round_type,
                        "fe_s": round(_fe_time, 1),
                        "model_s": round(_model_elapsed, 1),
                        "total_s": round(_round_elapsed, 1),
                        "cv": new_cv, "improved": False,
                    }
                    logger.info(
                        "⏱ D%d [%s]: fe=%.1fs model=%.1fs total=%.1fs → CV=%s",
                        hint_round, _round_type, _fe_time, _model_elapsed,
                        _round_elapsed, new_cv,
                    )

                    if new_cv is not None and (best_cv is None or new_cv > best_cv):
                        _metric_entry["improved"] = True
                        logger.info("Phase D%d: improved CV %.4f → %.4f", hint_round, best_cv or 0, new_cv)
                        best_cv = new_cv
                        best_cv_std = new_cv_std
                        best_output = model_output_d
                        _fast_cv_improved = True
                        _best_is_stacking = "Stack ensemble" in (model_output_d or "")
                        if submission_path.exists():
                            best_submission_bytes = submission_path.read_bytes()
                        # Save cumulative snapshot of the winning state
                        _t = exec_globals.get("train")
                        _te = exec_globals.get("test")
                        if _t is not None and _te is not None:
                            _snap_train = _t.copy()
                            _snap_test = _te.copy()
                            logger.info(
                                "Phase D%d: snapshot updated (train=%s, test=%s)",
                                hint_round, _snap_train.shape, _snap_test.shape,
                            )
                    else:
                        logger.info("Phase D%d: no improvement (%.4f → %s)", hint_round, best_cv or 0, new_cv)
                        # Safety: always keep at least ONE agent submission so
                        # we never fall back to Phase 0 or submit nothing.
                        if best_submission_bytes is None and submission_path.exists():
                            best_submission_bytes = submission_path.read_bytes()
                            logger.info("Phase D%d: captured floor submission (agent's first valid output)", hint_round)

                    _phase_d_metrics.append(_metric_entry)

                except Exception as exc:
                    logger.warning("Phase D%d failed: %s", hint_round, exc)

            # ---- Open-ended continuation rounds ----
            # After all structured hints are exhausted, keep iterating
            # with free-form prompts until the time budget runs out.
            freeform_round = 0
            while (agent_cv is not None):
                remaining = EXECUTION_TIMEOUT - (time.monotonic() - start_time)
                if remaining < 90:
                    logger.info("Phase D freeform: time budget exhausted (%.0fs left), stopping", remaining)
                    break

                freeform_round += 1
                total_d_round = hint_round + freeform_round
                logger.info(
                    "Phase D freeform round %d: best CV %.4f, target %.4f, %.0fs left",
                    freeform_round, best_cv or 0, phase0_cv, remaining,
                )
                await self._status(
                    event_queue, context,
                    f"Phase D freeform round {freeform_round}: exploring novel approaches…",
                )

                try:
                    submission_path.unlink(missing_ok=True)

                    # Restore cumulative snapshot for freeform rounds too
                    setup_script = _generate_setup_script(eda_meta, str(data_dir))
                    if setup_script:
                        try:
                            exec(setup_script, exec_globals)
                        except Exception:
                            pass
                    if _snap_train is not None and _snap_test is not None:
                        exec_globals["train"] = _snap_train.copy()
                        exec_globals["test"] = _snap_test.copy()

                    # Alternate between FE-focused and model-focused rounds
                    if freeform_round % 2 == 1:
                        # FE-focused: ask agent to try something new
                        phase_b_prompt_d = _build_phase_b_prompt(
                            eda_output, eda_meta, str(data_dir),
                        )
                        snap_info = ""
                        if _snap_train is not None:
                            snap_info = (
                                f"Your best feature set ({_snap_train.shape[1]} columns) is "
                                "ALREADY LOADED.  DO NOT re-apply previous transforms.  "
                                "ADD ONLY new techniques on top.\n\n"
                            )
                        phase_b_user_d = (
                            f"Your best CV so far: {phase0_metric}={best_cv:.6f}. "
                            f"Target to beat: {phase0_cv:.6f}.\n\n"
                            f"{snap_info}"
                            "You have already tried the standard approaches. "
                            "Think creatively: try a COMPLETELY DIFFERENT feature engineering "
                            "strategy than anything you've done before. "
                            "Consider: unusual feature combinations, domain-specific transforms, "
                            "or removing features that might be adding noise. "
                            "The goal is to find something the previous approaches missed."
                        )
                        fe_out_ff, _ = await _run_phase(
                            f"Phase D-free{freeform_round}-FE",
                            phase_b_prompt_d, phase_b_user_d,
                            _TOOLS_PHASE_B, max_rounds=3,
                        )
                        train_obj = exec_globals.get("train")
                        ts_d = str(getattr(train_obj, "shape", "unknown"))
                        nf_d = train_obj.shape[1] if train_obj is not None and hasattr(train_obj, "shape") else 0
                        phase_c_prompt_d = _build_phase_c_prompt(
                            eda_meta, str(data_dir), fe_out_ff,
                            baseline_cv=phase0_cv, baseline_metric=phase0_metric,
                            train_shape=ts_d, n_features=nf_d,
                        )
                        phase_c_user_d = (
                            "New feature engineering complete. "
                            "Use run_model to train, evaluate CV, and write submission.csv."
                        )
                    else:
                        # Model-focused: try different model config
                        train_obj = exec_globals.get("train")
                        ts_d = str(getattr(train_obj, "shape", "unknown"))
                        nf_d = train_obj.shape[1] if train_obj is not None and hasattr(train_obj, "shape") else 0
                        phase_c_prompt_d = _build_phase_c_prompt(
                            eda_meta, str(data_dir), fe_output,
                            baseline_cv=phase0_cv, baseline_metric=phase0_metric,
                            train_shape=ts_d, n_features=nf_d,
                        )
                        phase_c_user_d = (
                            f"Your best CV: {phase0_metric}={best_cv:.6f}. "
                            f"Target: {phase0_cv:.6f}.\n\n"
                            "Try a DIFFERENT model configuration than anything tried before. "
                            "Options: different model_type, different scoring metric, "
                            "different drop_cols, or use run_python for a custom model "
                            "(e.g., CatBoost with native categoricals, HistGBM, "
                            "or a weighted blend of multiple models)."
                        )

                    model_out_ff, _ = await _run_phase(
                        f"Phase D-free{freeform_round}-Model",
                        phase_c_prompt_d, phase_c_user_d,
                        _TOOLS_PHASE_C, max_rounds=3,
                    )
                    new_cv, new_cv_std = _parse_cv(model_out_ff)

                    if new_cv is not None and (best_cv is None or new_cv > best_cv):
                        logger.info("Phase D freeform %d: improved CV %.4f → %.4f",
                                    freeform_round, best_cv or 0, new_cv)
                        best_cv = new_cv
                        best_cv_std = new_cv_std
                        best_output = model_out_ff
                        _fast_cv_improved = True
                        _best_is_stacking = "Stack ensemble" in (model_out_ff or "")
                        if submission_path.exists():
                            best_submission_bytes = submission_path.read_bytes()
                        # Update cumulative snapshot
                        _t = exec_globals.get("train")
                        _te = exec_globals.get("test")
                        if _t is not None and _te is not None:
                            _snap_train = _t.copy()
                            _snap_test = _te.copy()
                            logger.info(
                                "Phase D freeform %d: snapshot updated (train=%s)",
                                freeform_round, _snap_train.shape,
                            )
                    else:
                        logger.info("Phase D freeform %d: no improvement (%.4f → %s)",
                                    freeform_round, best_cv or 0, new_cv)

                except Exception as exc:
                    logger.warning("Phase D freeform %d failed: %s", freeform_round, exc)

            total_d_rounds = hint_round + freeform_round
            if total_d_rounds > 0:
                elapsed = time.monotonic() - start_time
                logger.info(
                    "Phase D complete: %d hint rounds + %d freeform rounds = %d total, "
                    "best CV=%.4f, %.1fs elapsed",
                    hint_round, freeform_round, total_d_rounds,
                    best_cv or 0, elapsed,
                )
                # ---- Timing summary table ----
                if _phase_d_metrics:
                    _total_fe = sum(m["fe_s"] for m in _phase_d_metrics)
                    _total_model = sum(m["model_s"] for m in _phase_d_metrics)
                    _total_round = sum(m["total_s"] for m in _phase_d_metrics)
                    logger.info(
                        "⏱ Phase D timing summary: %d rounds, "
                        "FE=%.1fs Model=%.1fs Total=%.1fs | "
                        "avg per round: FE=%.1fs Model=%.1fs Total=%.1fs",
                        len(_phase_d_metrics),
                        _total_fe, _total_model, _total_round,
                        _total_fe / len(_phase_d_metrics),
                        _total_model / len(_phase_d_metrics),
                        _total_round / len(_phase_d_metrics),
                    )
                    for m in _phase_d_metrics:
                        logger.info(
                            "⏱   D%d [%s]: fe=%5.1fs model=%5.1fs total=%5.1fs cv=%s %s",
                            m["round"], m["type"],
                            m["fe_s"], m["model_s"], m["total_s"],
                            m["cv"], "✓" if m["improved"] else "",
                        )

            # ---- Final ensemble retrain (FAST_CV mode) ----
            # During fast_cv rounds, only LGBM CV was run — no ensemble
            # train+predict, so no submission.csv was written.  Now retrain
            # the full ensemble on the best feature snapshot.
            # SKIP retrain when the best round was a MODEL hint or freeform
            # round — those already wrote a real submission (stacking, tuning, etc.)
            # and the ensemble retrain would overwrite it with a weaker model.
            if FAST_CV and _fast_cv_improved and _snap_train is not None and not _best_is_stacking:
                logger.info(
                    "Final ensemble retrain on best snapshot (train=%s, test=%s)",
                    _snap_train.shape, _snap_test.shape if _snap_test is not None else "?",
                )
                _retrain_t0 = time.monotonic()
                try:
                    # Restore best snapshot
                    exec_globals["train"] = _snap_train.copy()
                    exec_globals["test"] = _snap_test.copy()
                    # Run full pipeline (fast_cv=False) for final submission
                    _retrain_output = await asyncio.wait_for(
                        asyncio.to_thread(
                            _exec_model_pipeline, _std_model_config,
                            exec_globals, str(data_dir), False,
                        ),
                        timeout=300.0,
                    )
                    _retrain_elapsed = time.monotonic() - _retrain_t0
                    logger.info(
                        "⏱ Final ensemble retrain: %.1fs — %s",
                        _retrain_elapsed, _retrain_output[:300],
                    )
                    if submission_path.exists():
                        best_submission_bytes = submission_path.read_bytes()
                except Exception as exc:
                    logger.warning("Final ensemble retrain failed: %s", exc)
            elif FAST_CV and _best_is_stacking:
                logger.info(
                    "Skipping final ensemble retrain — best round was stacking (already multi-model)"
                )

            # ---- Restore agent's best submission ----
            # Phase 0 is ONLY a CV reference target.  We NEVER submit
            # Phase 0's output — the agent must produce its own submission.
            # Phase 0's submission.csv was deleted immediately after
            # extracting its CV score.
            if best_submission_bytes is not None:
                submission_path.write_bytes(best_submission_bytes)
                logger.info(
                    "Submitting agent's best (CV %.4f). "
                    "Phase 0 reference was %.4f (not submitted).",
                    best_cv or 0, phase0_cv or 0,
                )
            elif submission_path.exists():
                # Agent produced submissions in Phase D rounds but never
                # improved over Phase C.  Submit the last round's output
                # rather than nothing — this is still the agent's work.
                logger.info(
                    "Submitting agent's last-round output (CV %.4f, "
                    "no improvement over initial). Phase 0 NOT submitted.",
                    best_cv or 0,
                )
            else:
                logger.warning(
                    "Agent produced no valid submission. "
                    "Phase 0 will NOT be submitted as fallback."
                )

            # ---- Final validation ----
            if submission_path.exists():
                valid, diag = _validate_submission(data_dir)
                if not valid:
                    logger.warning("Final validation failed: %s", diag)
                    submission_path.unlink(missing_ok=True)
                else:
                    logger.info("Final validation: %s", diag)

            total_api_calls = 1 + total_rounds  # planning + phase rounds
            logger.info("Total API calls: %d (planning: 1, agent rounds: %d)",
                        total_api_calls, total_rounds)

            # ---- return submission or error ----
            if submission_path.exists():
                csv_bytes = submission_path.read_bytes()
                encoded = base64.b64encode(csv_bytes).decode("ascii")
                logger.info("Submitting artifact (%d bytes, %d API calls)…",
                            len(csv_bytes), total_api_calls)

                artifact = Artifact(
                    artifact_id="submission",
                    name="submission.csv",
                    parts=[
                        Part(root=FilePart(
                            file=FileWithBytes(
                                bytes=encoded,
                                name="submission.csv",
                                mime_type="text/csv",
                            )
                        ))
                    ],
                )
                await event_queue.enqueue_event(
                    TaskArtifactUpdateEvent(
                        artifact=artifact,
                        context_id=context.context_id,
                        task_id=context.task_id,
                    )
                )
                await self._status(event_queue, context,
                                   f"Submission ready ({len(csv_bytes):,} bytes, "
                                   f"{total_api_calls} API calls).",
                                   final=True, state=TaskState.completed)
            else:
                await self._status(
                    event_queue, context,
                    f"Failed to produce submission.csv "
                    f"({total_api_calls} API calls, {total_rounds} rounds).",
                    final=True,
                    state=TaskState.failed,
                )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

def prepare_agent_card(url: str) -> AgentCard:
    skill = AgentSkill(
        id="kaggle-ml",
        name="Kaggle ML Competition Solver",
        description="Solves Kaggle-style ML competitions by generating and executing Python ML code.",
        tags=["kaggle", "machine-learning", "openai"],
        examples=[],
    )
    return AgentCard(
        name="AgentWhetter_MLE",
        description="OpenAI-powered ML coding agent for MLE-Bench evaluations.",
        url=url,
        version="1.0.1",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the OpenAI purple ML agent.")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9022)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--card-url", default="")
    args = parser.parse_args()

    debug_env = os.getenv("AGENT_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}
    debug = args.debug or debug_env
    logging.basicConfig(level=logging.INFO if debug else logging.WARNING)

    card_host = "127.0.0.1" if args.host == "0.0.0.0" else args.host
    card_url = args.card_url or f"http://{card_host}:{args.port}"

    card = prepare_agent_card(card_url)
    request_handler = DefaultRequestHandler(
        agent_executor=MLPurpleAgent(debug=debug),
        task_store=InMemoryTaskStore(),
    )
    app = A2AStarletteApplication(
        agent_card=card,
        http_handler=request_handler,
        max_content_length=None,
    )
    logger.info("Starting purple ML agent on %s:%d", args.host, args.port)
    uvicorn.run(app.build(), host=args.host, port=args.port, timeout_keep_alive=600)


if __name__ == "__main__":
    main()
