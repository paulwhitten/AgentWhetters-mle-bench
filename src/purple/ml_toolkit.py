"""ML Toolkit — building blocks for EDA, feature engineering, and modeling.

These functions are injected into the agent's exec_globals so the LLM agent
can call them by name.  The agent's workflow: explore → engineer → model → evaluate.

Extracted from Phase 0 deterministic pipeline + new building blocks.
"""
from __future__ import annotations

import os
import re
import warnings
import gc
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

_MAX_JOBS = os.cpu_count() or 4  # use all available CPU cores


# =========================================================================
# SECTION 1: EDA — Exploratory Data Analysis
# =========================================================================

def eda_summary(df: pd.DataFrame, name: str = "df") -> str:
    """Comprehensive EDA summary: shape, dtypes, nulls, stats, categoricals.

    Call this on train and test to understand the data before engineering features.
    """
    lines = [f"=== EDA: {name} ({df.shape[0]} rows × {df.shape[1]} cols) ===\n"]

    dtype_counts = df.dtypes.value_counts()
    lines.append(f"Dtypes: {dict(dtype_counts)}")

    nulls = df.isnull().sum()
    has_nulls = nulls[nulls > 0].sort_values(ascending=False)
    if len(has_nulls):
        lines.append(f"\nNull columns ({len(has_nulls)}):")
        for col, cnt in has_nulls.items():
            lines.append(f"  {col}: {cnt} ({cnt/len(df)*100:.1f}%)")
    else:
        lines.append("\nNo null values.")

    num_cols = df.select_dtypes(include="number").columns.tolist()
    if num_cols:
        lines.append(f"\nNumeric columns ({len(num_cols)}):")
        for col in num_cols[:20]:
            s = df[col]
            lines.append(f"  {col}: min={s.min():.3g} max={s.max():.3g} "
                         f"mean={s.mean():.3g} std={s.std():.3g} skew={s.skew():.2f}")

    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    if cat_cols:
        lines.append(f"\nCategorical columns ({len(cat_cols)}):")
        for col in cat_cols[:15]:
            nuniq = df[col].nunique()
            top = df[col].value_counts().head(3)
            top_str = ", ".join(f"{v}:{c}" for v, c in top.items())
            lines.append(f"  {col}: {nuniq} unique — top: {top_str}")

    result = "\n".join(lines)
    print(result)
    return result


def eda_correlations(df: pd.DataFrame, target_col: str) -> pd.Series:
    """Show correlations of all numeric columns with the target.

    Returns sorted Series of correlations. Helps identify the most predictive features.
    """
    num_df = df.select_dtypes(include="number")
    if target_col not in num_df.columns:
        print(f"Target '{target_col}' is not numeric — try encoding it first.")
        return pd.Series(dtype=float)

    corrs = num_df.corr()[target_col].drop(target_col, errors="ignore").sort_values(
        key=abs, ascending=False
    )
    print(f"=== Correlations with '{target_col}' (top 20) ===")
    for col, r in corrs.head(20).items():
        bar = "+" * int(abs(r) * 40) if r > 0 else "-" * int(abs(r) * 40)
        print(f"  {col:30s} {r:+.4f} {bar}")
    return corrs


def eda_target(df: pd.DataFrame, target_col: str) -> dict:
    """Analyze the target variable: distribution, class balance, type.

    Returns dict with keys: type, n_classes, balance, stats.
    """
    y = df[target_col]
    info: dict = {}

    if y.dtype == bool or y.nunique() == 2:
        info["type"] = "binary"
        vc = y.value_counts(normalize=True)
        info["n_classes"] = 2
        info["balance"] = dict(vc)
        print(f"Target '{target_col}': BINARY — {dict(y.value_counts())}")
        print(f"  Balance: {dict(vc)}")
    elif y.dtype in ("object", "category") or (y.dtype.kind == "i" and y.nunique() <= 20):
        info["type"] = "multiclass"
        info["n_classes"] = y.nunique()
        info["balance"] = dict(y.value_counts(normalize=True))
        print(f"Target '{target_col}': MULTICLASS — {y.nunique()} classes")
        print(f"  Distribution: {dict(y.value_counts().head(10))}")
    else:
        info["type"] = "regression"
        info["stats"] = {"mean": float(y.mean()), "std": float(y.std()),
                         "min": float(y.min()), "max": float(y.max()),
                         "skew": float(y.skew())}
        print(f"Target '{target_col}': REGRESSION")
        print(f"  mean={y.mean():.4g} std={y.std():.4g} skew={y.skew():.2f}")

    return info


def eda_profile_columns(df: pd.DataFrame, name: str = "df") -> dict:
    """Deep column profiling with role classification (from Phase 0 engine).

    Returns dict mapping column name → {role, dtype, nunique, null_pct, ...}.
    Roles: ID, CONSTANT, BINARY, BOOL_STR, NUMERIC_AS_STR, STRUCTURED_STR,
           HIGH_CARD, CATEGORICAL, CONTINUOUS, FREE_TEXT, SHORT_FREE_TEXT.
    """
    DELIMITERS = ["/", "-", "_", "|", ":", ";", "."]

    def _sniff_delimiter(series):
        sample = series.dropna().astype(str).head(200)
        if len(sample) < 10:
            return None
        for delim in DELIMITERS:
            counts = sample.str.count(re.escape(delim))
            mode_count = int(counts.mode().iloc[0]) if len(counts.mode()) > 0 else 0
            if mode_count > 0 and (counts == mode_count).mean() > 0.80:
                return delim
        return None

    _FLOAT_RE = re.compile(r'^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$')
    _NOISE_RE = re.compile(r'[\$,€£%\s_]')

    def _is_numeric_as_str(series):
        nonnull = series.dropna().astype(str)
        if len(nonnull) == 0:
            return False
        n = min(250, len(nonnull))
        sample = nonnull.sample(n=n, random_state=42) if len(nonnull) > n else nonnull
        cleaned = sample.apply(lambda v: _NOISE_RE.sub("", v).strip())
        m1 = sum(1 for val in cleaned if val and _FLOAT_RE.match(val))
        m2_ratio = pd.to_numeric(cleaned, errors="coerce").notna().mean()
        return (m1 / len(sample)) >= 0.90 and m2_ratio >= 0.90

    result = {}
    nrows = len(df)
    lines = [f"=== Column Profile: {name} ({nrows} rows × {len(df.columns)} cols) ==="]

    for col in df.columns:
        s = df[col]
        meta = {"dtype": str(s.dtype), "nunique": int(s.nunique(dropna=False)),
                "null_count": int(s.isnull().sum()),
                "null_pct": round(100.0 * s.isnull().sum() / max(nrows, 1), 1)}
        nunique = meta["nunique"]
        dtype = s.dtype

        if nunique <= 1:
            meta["role"] = "CONSTANT"
        elif nrows > 10 and nunique == nrows and not pd.api.types.is_float_dtype(dtype):
            meta["role"] = "ID"
            delim = _sniff_delimiter(s)
            if delim:
                ex = str(s.dropna().iloc[0]) if s.notna().any() else "?"
                meta["structured_id"] = {"delimiter": delim, "example": ex,
                                         "n_parts": len(ex.split(delim))}
        elif pd.api.types.is_bool_dtype(dtype):
            meta["role"] = "BINARY"
        elif pd.api.types.is_numeric_dtype(dtype):
            meta["role"] = "BINARY" if nunique == 2 else (
                "ORDINAL" if nunique <= 10 else "CONTINUOUS")
            if meta["role"] == "CONTINUOUS":
                sk = float(s.dropna().skew()) if s.notna().sum() > 1 else 0.0
                meta["skew"] = sk
                if abs(sk) > 5.0:
                    meta["skewed"] = True
                # Zero-inflated check
                nonnull = s.dropna()
                if len(nonnull) > 10 and (nonnull == 0).mean() > 0.30:
                    meta["zero_inflated"] = True
        elif pd.api.types.is_string_dtype(dtype) or dtype == object:
            str_vals = s.dropna().astype(str)
            bool_like = {"true", "false", "yes", "no", "t", "f", "0", "1"}
            if nunique <= 4 and set(str_vals.str.lower().unique()) <= bool_like:
                meta["role"] = "BOOL_STR"
            elif _is_numeric_as_str(s):
                meta["role"] = "NUMERIC_AS_STR"
            else:
                avg_len = float(str_vals.str.len().mean()) if len(str_vals) > 0 else 0
                delim = _sniff_delimiter(s)
                if delim and avg_len < 30 and nunique > 20:
                    n_parts = str_vals.head(200).str.split(re.escape(delim), expand=True).shape[1]
                    meta["role"] = "STRUCTURED_STR"
                    meta["delimiter"] = delim
                    meta["n_parts"] = n_parts
                elif avg_len > 50:
                    meta["role"] = "FREE_TEXT"
                elif nunique > 50:
                    meta["role"] = "HIGH_CARD"
                elif nunique <= 50:
                    meta["role"] = "CATEGORICAL"
                else:
                    meta["role"] = "UNKNOWN"
        else:
            meta["role"] = "UNKNOWN"

        result[col] = meta
        lines.append(f"  {col:25s} {meta['role']:18s} dtype={meta['dtype']}  "
                      f"unique={nunique}  null={meta['null_pct']:.1f}%")

    print("\n".join(lines))
    return result


def eda_find_target(
    train: pd.DataFrame, test: pd.DataFrame,
    sample_submission: pd.DataFrame | None = None,
) -> dict:
    """Detect target column, ID column, and task type from train/test column diff.

    Returns dict with keys: target_col, id_col, task, train_only_cols, test_cols.
    """
    train_cols = set(train.columns)
    test_cols = set(test.columns)
    train_only = train_cols - test_cols
    info = {"train_only_cols": sorted(train_only), "test_cols": sorted(test_cols)}

    # ID column: first column of sample_submission, or first shared column
    if sample_submission is not None:
        info["id_col"] = sample_submission.columns[0]
        # Target from sample_submission non-ID cols in train_only
        sample_targets = [c for c in sample_submission.columns[1:] if c in train_only]
        if len(sample_targets) == 1:
            info["target_col"] = sample_targets[0]
        elif len(sample_targets) > 1:
            info["target_col"] = sample_targets[0]
            info["target_cols"] = sample_targets
    else:
        info["id_col"] = None

    # Fallback target detection
    if "target_col" not in info and len(train_only) == 1:
        info["target_col"] = next(iter(train_only))
    elif "target_col" not in info:
        # Pick binary bool column from train-only
        for c in train_only:
            if train[c].nunique(dropna=True) == 2:
                info["target_col"] = c
                break

    # Task type
    tc = info.get("target_col")
    if tc and tc in train.columns:
        y = train[tc]
        if y.dtype == bool or y.nunique() == 2:
            info["task"] = "binary"
        elif y.dtype in ("object", "category") or (y.dtype.kind == "i" and y.nunique() <= 20):
            info["task"] = "multiclass"
        else:
            info["task"] = "regression"

    print(f"eda_find_target: {info}")
    return info


def eda_null_analysis(df: pd.DataFrame, name: str = "df") -> dict:
    """Analyze null patterns, zero-inflated columns, and suggest imputation.

    Returns dict: {null_cols: [...], zero_inflated: [...], strategies: {...}}.
    """
    result: dict = {"null_cols": [], "zero_inflated": [], "strategies": {}}

    for col in df.columns:
        null_pct = df[col].isnull().mean() * 100
        if null_pct > 0:
            result["null_cols"].append({"col": col, "pct": round(null_pct, 1)})
            if pd.api.types.is_numeric_dtype(df[col]):
                result["strategies"][col] = "fillna(median)"
            else:
                result["strategies"][col] = "fillna(mode)"

        if pd.api.types.is_numeric_dtype(df[col]) and not pd.api.types.is_bool_dtype(df[col]):
            nonnull = df[col].dropna()
            if len(nonnull) > 10 and (nonnull == 0).mean() > 0.30:
                result["zero_inflated"].append(col)

    print(f"=== Null Analysis: {name} ===")
    print(f"Columns with nulls: {len(result['null_cols'])}")
    for nc in result["null_cols"][:10]:
        print(f"  {nc['col']}: {nc['pct']:.1f}% null → {result['strategies'].get(nc['col'], '?')}")
    if result["zero_inflated"]:
        print(f"Zero-inflated (>30% zeros): {result['zero_inflated']}")
    return result


# =========================================================================
# SECTION 2: Feature Engineering
# =========================================================================

def fe_target_encode(
    train: pd.DataFrame, test: pd.DataFrame, cols: list[str],
    target: pd.Series, n_folds: int = 5, smoothing: float = 10.0,
    **kwargs,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Target-encode categorical columns with out-of-fold leakage prevention.

    Adds '{col}_te' columns. Does NOT drop the original columns.
    """
    # Alias: LLM often hallucinates k_folds instead of n_folds
    if "k_folds" in kwargs:
        n_folds = kwargs.pop("k_folds")
    from sklearn.model_selection import KFold

    global_mean = target.mean()

    for col in cols:
        if col not in train.columns:
            continue
        te_col = f"{col}_te"
        train[te_col] = np.nan
        test[te_col] = global_mean

        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        for tr_idx, val_idx in kf.split(train):
            fold_df = train.iloc[tr_idx]
            fold_y = target.iloc[tr_idx]
            stats = fold_df.assign(_t=fold_y.values).groupby(col)["_t"].agg(["mean", "count"])
            smoothed = (stats["mean"] * stats["count"] + global_mean * smoothing) / (stats["count"] + smoothing)
            train.iloc[val_idx, train.columns.get_loc(te_col)] = train.iloc[val_idx][col].map(smoothed).values

        full_stats = train.assign(_t=target.values).groupby(col)["_t"].agg(["mean", "count"])
        smoothed_full = (full_stats["mean"] * full_stats["count"] + global_mean * smoothing) / (full_stats["count"] + smoothing)
        test[te_col] = test[col].map(smoothed_full).fillna(global_mean)
        train[te_col] = train[te_col].fillna(global_mean)

    print(f"fe_target_encode: added {[f'{c}_te' for c in cols if c in train.columns]}")
    return train, test


def fe_frequency_encode(
    train: pd.DataFrame, test: pd.DataFrame, cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Frequency-encode categorical columns. Adds '{col}_freq' columns."""
    for col in cols:
        if col not in train.columns:
            continue
        freq = train[col].value_counts(normalize=True)
        train[f"{col}_freq"] = train[col].map(freq).fillna(0)
        test[f"{col}_freq"] = test[col].map(freq).fillna(0)
    print(f"fe_frequency_encode: added {[f'{c}_freq' for c in cols if c in train.columns]}")
    return train, test


def fe_interactions(
    train: pd.DataFrame, test: pd.DataFrame, col_pairs: list[tuple[str, str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create interaction features (products) for specified column pairs.

    Example: train, test = fe_interactions(train, test, [('Age', 'RoomService')])
    """
    created = []
    for a, b in col_pairs:
        name = f"{a}_x_{b}"
        for df in [train, test]:
            if a in df.columns and b in df.columns:
                df[name] = df[a].fillna(0).astype(float) * df[b].fillna(0).astype(float)
        created.append(name)
    print(f"fe_interactions: created {created}")
    return train, test


def fe_agg_features(
    train: pd.DataFrame, test: pd.DataFrame, group_col: str = "", agg_cols: list[str] | None = None,
    aggs: list[str] = ["mean", "std"],
    **kwargs,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create group-aggregation features (e.g., mean spend per HomePlanet).

    Example: train, test = fe_agg_features(train, test, 'HomePlanet', ['RoomService'], ['mean'])
    """
    # Aliases: LLM hallucinates group_cols, agg_funcs
    if "group_cols" in kwargs:
        group_col = group_col or kwargs.pop("group_cols")
    if "agg_funcs" in kwargs:
        aggs = kwargs.pop("agg_funcs")
    # Unwrap single-element list: LLM passes ["HomePlanet"] instead of "HomePlanet"
    if isinstance(group_col, list):
        group_col = group_col[0] if group_col else ""
    if agg_cols is None:
        agg_cols = []
    created = []
    for agg_col in agg_cols:
        if agg_col not in train.columns or group_col not in train.columns:
            continue
        for agg_fn in aggs:
            name = f"{group_col}_{agg_col}_{agg_fn}"
            mapping = train.groupby(group_col)[agg_col].agg(agg_fn)
            for df in [train, test]:
                if group_col in df.columns:
                    df[name] = df[group_col].map(mapping).fillna(0)
            created.append(name)
    print(f"fe_agg_features: created {created}")
    return train, test


def fe_binning(
    train: pd.DataFrame, test: pd.DataFrame, col: str = "", bins: int = 5, labels: bool = False,
    **kwargs,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Bin a numeric column into quantile-based buckets. Adds '{col}_bin'."""
    # Alias: LLM hallucinates cols (plural) instead of col (singular)
    if "cols" in kwargs and not col:
        col = kwargs.pop("cols")
        if isinstance(col, list):
            col = col[0]  # take first if list passed
    # Ignore unknown params like 'strategy'
    kwargs.pop("strategy", None)
    # Fit bins on train, apply to both
    try:
        _, bin_edges = pd.qcut(train[col].dropna(), q=bins, retbins=True, duplicates="drop")
    except ValueError:
        # Fallback: too few unique values for requested bins
        bins = min(bins, train[col].nunique())
        if bins < 2:
            print(f"fe_binning: skipped {col} (< 2 unique values)")
            return train, test
        _, bin_edges = pd.qcut(train[col].dropna(), q=bins, retbins=True, duplicates="drop")
    for df in [train, test]:
        df[f"{col}_bin"] = pd.cut(df[col], bins=bin_edges, labels=False, include_lowest=True)
    print(f"fe_binning: added {col}_bin ({bins} bins)")
    return train, test


def fe_log_transform(
    train: pd.DataFrame, test: pd.DataFrame, cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply log1p transform to skewed numeric columns. Adds '{col}_log' columns."""
    for df in [train, test]:
        for col in cols:
            if col in df.columns:
                df[f"{col}_log"] = np.log1p(df[col].clip(lower=0).fillna(0))
    print(f"fe_log_transform: added {[f'{c}_log' for c in cols if c in train.columns]}")
    return train, test


def fe_bool_convert(
    train: pd.DataFrame, test: pd.DataFrame, cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convert boolean-string columns (True/False/Yes/No) to int 0/1.

    From Phase 0: handles multiple string formats safely.
    """
    _bmap = {"True": 1, "False": 0, True: 1, False: 0,
             "true": 1, "false": 0, "Yes": 1, "No": 0,
             "yes": 1, "no": 0, "T": 1, "F": 0, "1": 1, "0": 0}
    for col in cols:
        if col in train.columns:
            train[col] = train[col].map(_bmap)
        if col in test.columns:
            test[col] = test[col].map(_bmap)
    print(f"fe_bool_convert: converted {cols}")
    return train, test


def fe_split_structured(
    train: pd.DataFrame, test: pd.DataFrame, col: str, delimiter: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a structured-string column by delimiter into separate features.

    From Phase 0: splits 'Cabin' by '/' into Cabin_p0, Cabin_p1, ...
    Numeric parts auto-converted; categorical parts kept as strings.
    """
    for df in [train, test]:
        parts = df[col].str.split(delimiter, expand=True)
        for i in range(parts.shape[1]):
            pname = f"{col}_p{i}"
            df[pname] = parts[i]
            num = pd.to_numeric(df[pname], errors="coerce")
            if num.notna().mean() > 0.5:
                df[pname] = num
    train = train.drop(columns=[col])
    test = test.drop(columns=[col])
    print(f"fe_split_structured: split '{col}' by '{delimiter}', dropped original")
    return train, test


def fe_numeric_clean(
    train: pd.DataFrame, test: pd.DataFrame, cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Clean numeric-as-string columns: strip $, commas, %, convert to float.

    From Phase 0: handles currency/thousands/percent noise.
    """
    _noise = re.compile(r'[\$,€£%\s_]')
    for col in cols:
        for df in [train, test]:
            if col in df.columns:
                df[col] = pd.to_numeric(
                    df[col].astype(str).str.replace(_noise, '', regex=True),
                    errors="coerce",
                )
    print(f"fe_numeric_clean: cleaned {cols}")
    return train, test


def fe_datetime_features(
    train: pd.DataFrame, test: pd.DataFrame, col: str,
    drop_original: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extract hour, dayofweek, month, year from a datetime column.

    From Phase 0: creates 4 new features, optionally drops original.
    """
    for df in [train, test]:
        dt = pd.to_datetime(df[col], errors="coerce")
        df[f"{col}_hour"] = dt.dt.hour
        df[f"{col}_dayofweek"] = dt.dt.dayofweek
        df[f"{col}_month"] = dt.dt.month
        df[f"{col}_year"] = dt.dt.year
    if drop_original:
        train = train.drop(columns=[col], errors="ignore")
        test = test.drop(columns=[col], errors="ignore")
    print(f"fe_datetime_features: extracted hour/dow/month/year from '{col}'")
    return train, test


def fe_group_size(
    train: pd.DataFrame, test: pd.DataFrame, id_col: str = "", delimiter: str = "_",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extract GroupSize + IsSolo from a structured ID column (e.g., 'GGGG_PP').

    From Phase 0: splits ID, groups by first segment, counts members.
    """
    for df in [train, test]:
        df["_g"] = df[id_col].str.split(delimiter).str[0]
        df["GroupSize"] = df.groupby("_g")[id_col].transform("count")
        df["IsSolo"] = (df["GroupSize"] == 1).astype(int)
        df.drop(columns=["_g"], inplace=True)
    print(f"fe_group_size: added GroupSize, IsSolo from '{id_col}'")
    return train, test


def fe_null_indicators(
    train: pd.DataFrame, test: pd.DataFrame, threshold_pct: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add binary _isNull indicator features for columns above null threshold.

    From Phase 0: creates {col}_isNull for any column with > threshold_pct% nulls.
    """
    created = []
    for col in train.columns:
        null_pct = train[col].isnull().mean() * 100
        if null_pct > threshold_pct:
            name = f"{col}_isNull"
            for df in [train, test]:
                if col in df.columns:
                    df[name] = df[col].isna().astype(int)
            created.append(name)
    print(f"fe_null_indicators: added {created}")
    return train, test


def fe_spending_aggs(
    train: pd.DataFrame, test: pd.DataFrame, spend_cols: list[str],
    group_col: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate spending columns: TotalSpend, IsZeroSpend, plus log transforms.

    From Phase 0: for zero-inflated spending features. Optionally adds
    SpendPerPerson if group_col (like GroupSize) exists.
    """
    valid = [c for c in spend_cols if c in train.columns]
    if not valid:
        print("fe_spending_aggs: no valid spend columns found")
        return train, test

    for df in [train, test]:
        df["TotalSpend"] = df[valid].fillna(0).sum(axis=1)
        df["IsZeroSpend"] = (df["TotalSpend"] == 0).astype(int)
        if group_col and group_col in df.columns:
            df["SpendPerPerson"] = df["TotalSpend"] / df[group_col].clip(lower=1)
        for sc in valid:
            df[f"{sc}_log"] = np.log1p(df[sc].fillna(0))

    print(f"fe_spending_aggs: TotalSpend, IsZeroSpend, logs from {valid}")
    return train, test


def fe_haversine(
    df: pd.DataFrame, lat1: str, lon1: str, lat2: str, lon2: str,
    name: str = "distance",
) -> pd.DataFrame:
    """Compute Haversine distance between two lat/lon pairs.

    From Phase 0: for geo-based competitions with pickup/dropoff or origin/dest.
    """
    R = 6371  # Earth radius in km
    la1, lo1, la2, lo2 = map(np.radians,
                              [df[lat1], df[lon1], df[lat2], df[lon2]])
    dlat = la2 - la1
    dlon = lo2 - lo1
    a = np.sin(dlat / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin(dlon / 2) ** 2
    df[name] = 2 * R * np.arcsin(np.sqrt(a))
    df[name] = df[name].replace([np.inf, -np.inf], np.nan).fillna(0)
    print(f"fe_haversine: added '{name}' from ({lat1},{lon1}) → ({lat2},{lon2})")
    return df


# =========================================================================
# SECTION 2b: Feature Engineering — Tier 1 (High impact, low cost)
# =========================================================================

def fe_ratios(
    train: pd.DataFrame, test: pd.DataFrame,
    col_pairs: list[tuple[str, str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create ratio features (col_a / col_b) for specified column pairs.

    Guards against division by zero by clipping denominator away from zero.
    Replaces inf/NaN results with 0.

    Example: train, test = fe_ratios(train, test, [('Income', 'Debt')])
    """
    created = []
    for a, b in col_pairs:
        name = f"{a}_div_{b}"
        for df in [train, test]:
            if a in df.columns and b in df.columns:
                denom = df[b].fillna(0).astype(float)
                denom = denom.where(denom.abs() > 1e-9, 1e-9)
                df[name] = (df[a].fillna(0).astype(float) / denom).replace(
                    [np.inf, -np.inf], np.nan
                ).fillna(0)
        created.append(name)
    print(f"fe_ratios: created {created}")
    return train, test


def fe_differences(
    train: pd.DataFrame, test: pd.DataFrame,
    col_pairs: list[tuple[str, str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create difference features (col_a - col_b) for specified column pairs.

    Example: train, test = fe_differences(train, test, [('Revenue', 'Cost')])
    """
    created = []
    for a, b in col_pairs:
        name = f"{a}_minus_{b}"
        for df in [train, test]:
            if a in df.columns and b in df.columns:
                df[name] = df[a].fillna(0).astype(float) - df[b].fillna(0).astype(float)
        created.append(name)
    print(f"fe_differences: created {created}")
    return train, test


def fe_row_stats(
    train: pd.DataFrame, test: pd.DataFrame,
    cols: list[str],
    stats: list[str] | None = None,
    prefix: str = "row",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute row-wise statistics across a group of related numeric columns.

    Defaults to mean, std, min, max. Useful for groups of spending columns,
    sensor readings, multiple test scores, etc.

    Example: train, test = fe_row_stats(train, test, ['RoomService', 'FoodCourt', 'Spa'])
    """
    if stats is None:
        stats = ["mean", "std", "min", "max"]
    created = []
    for df in [train, test]:
        valid = [c for c in cols if c in df.columns]
        if not valid:
            continue
        sub = df[valid].fillna(0).astype(float)
        for stat in stats:
            name = f"{prefix}_{stat}"
            if stat == "mean":
                df[name] = sub.mean(axis=1)
            elif stat == "std":
                df[name] = sub.std(axis=1).fillna(0)
            elif stat == "min":
                df[name] = sub.min(axis=1)
            elif stat == "max":
                df[name] = sub.max(axis=1)
            elif stat == "sum":
                df[name] = sub.sum(axis=1)
            elif stat == "range":
                df[name] = sub.max(axis=1) - sub.min(axis=1)
            created.append(name)
    print(f"fe_row_stats: created {list(dict.fromkeys(created))}")
    return train, test


def fe_categorical_cross(
    train: pd.DataFrame, test: pd.DataFrame,
    col_pairs: list[tuple[str, str]] | None = None,
    max_categories: int = 50,
    **kwargs,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create categorical cross features by combining two categoricals.

    Concatenates values: cat_A × cat_B → "valueA_valueB". Caps cardinality by
    grouping rare combinations (below top max_categories) as "_Other".

    Example: train, test = fe_categorical_cross(train, test, [('HomePlanet', 'Destination')])
    """
    # Alias: LLM passes cols (flat list) instead of col_pairs (list of tuples)
    if col_pairs is None and "cols" in kwargs:
        cols = kwargs.pop("cols")
        if isinstance(cols, list) and cols and isinstance(cols[0], str):
            # Convert flat list to all pairwise tuples
            col_pairs = [(cols[i], cols[j])
                         for i in range(len(cols))
                         for j in range(i + 1, len(cols))]
    if col_pairs is None:
        col_pairs = []
    created = []
    for a, b in col_pairs:
        name = f"{a}_X_{b}"
        if a not in train.columns or b not in train.columns:
            continue
        # Build combined column
        for df in [train, test]:
            if a in df.columns and b in df.columns:
                df[name] = (
                    df[a].fillna("_NA").astype(str) + "_" +
                    df[b].fillna("_NA").astype(str)
                )
        # Cap cardinality by keeping only top categories from train
        top_cats = set(train[name].value_counts().head(max_categories).index)
        for df in [train, test]:
            df[name] = df[name].where(df[name].isin(top_cats), "_Other")
        created.append(name)
    print(f"fe_categorical_cross: created {created}")
    return train, test


def fe_null_row_count(
    train: pd.DataFrame, test: pd.DataFrame,
    cols: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Count total null values per row. Adds 'null_count' feature.

    Rows with many nulls often indicate data quality patterns that correlate
    with the target (e.g., CryoSleep passengers in Spaceship Titanic).

    Example: train, test = fe_null_row_count(train, test)
    """
    for df in [train, test]:
        subset = df[cols] if cols else df
        df["null_count"] = subset.isnull().sum(axis=1)
    print(f"fe_null_row_count: added null_count (cols={'all' if cols is None else len(cols)})")
    return train, test


def fe_cyclical_encode(
    train: pd.DataFrame, test: pd.DataFrame,
    col: str,
    max_value: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Encode periodic/cyclical features via sin/cos transformation.

    Captures that hour 23 and hour 0 are adjacent, month 12 and month 1
    are adjacent, etc. Keeps the original column.

    Example: train, test = fe_cyclical_encode(train, test, 'hour', max_value=24)
    """
    for df in [train, test]:
        if col in df.columns:
            vals = df[col].fillna(0).astype(float)
            df[f"{col}_sin"] = np.sin(2 * np.pi * vals / max_value)
            df[f"{col}_cos"] = np.cos(2 * np.pi * vals / max_value)
    print(f"fe_cyclical_encode: added {col}_sin, {col}_cos (period={max_value})")
    return train, test


# =========================================================================
# SECTION 2c: Feature Engineering — Tier 2 (Medium-High impact)
# =========================================================================

def fe_percentile_rank_group(
    train: pd.DataFrame, test: pd.DataFrame,
    group_col: str,
    value_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute percentile rank of value_col within each group.

    Answers: "Where does this row rank within its group?" (0.0 to 1.0).
    Fit group statistics on train, apply mapping to test.

    Example: train, test = fe_percentile_rank_group(train, test, 'HomePlanet', 'RoomService')
    """
    name = f"{value_col}_pctrank_by_{group_col}"
    # Compute rank within group on train
    train[name] = train.groupby(group_col)[value_col].rank(pct=True, method="average")
    train[name] = train[name].fillna(0.5)

    # For test: compute group quantile boundaries from train, then map
    # Use a simpler approach: merge group stats from train
    group_stats = train.groupby(group_col)[value_col].agg(["mean", "std"]).reset_index()
    group_stats.columns = [group_col, "_g_mean", "_g_std"]
    group_stats["_g_std"] = group_stats["_g_std"].clip(lower=1e-9)

    test = test.merge(group_stats, on=group_col, how="left")
    test[name] = ((test[value_col] - test["_g_mean"]) / test["_g_std"]).fillna(0)
    # Convert z-score to approximate percentile via sigmoid
    test[name] = 1 / (1 + np.exp(-test[name]))
    test.drop(columns=["_g_mean", "_g_std"], inplace=True)

    print(f"fe_percentile_rank_group: added {name}")
    return train, test


def fe_cluster_labels(
    train: pd.DataFrame, test: pd.DataFrame,
    cols: list[str],
    n_clusters: int = 8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit KMeans on specified numeric columns and assign cluster ID as a feature.

    Captures non-linear groupings that individual features miss.
    Fit on train only; predict on test.

    Example: train, test = fe_cluster_labels(train, test, ['Age', 'RoomService', 'Spa'], n_clusters=5)
    """
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    valid = [c for c in cols if c in train.columns and c in test.columns]
    if len(valid) < 2:
        print(f"fe_cluster_labels: need >=2 valid cols, got {len(valid)} — skipped")
        return train, test

    name = f"cluster_{n_clusters}"
    # Prepare: impute and scale
    train_sub = train[valid].fillna(0).astype(float)
    test_sub = test[valid].fillna(0).astype(float)

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_sub)
    test_scaled = scaler.transform(test_sub)

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, max_iter=100)
    train[name] = km.fit_predict(train_scaled)
    test[name] = km.predict(test_scaled)

    print(f"fe_cluster_labels: added {name} from {valid}")
    return train, test


def fe_pca_features(
    train: pd.DataFrame, test: pd.DataFrame,
    cols: list[str],
    n_components: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit PCA on specified numeric columns and add principal components as features.

    Captures variance axes; reduces noise among correlated features.
    Fit on train only.

    Example: train, test = fe_pca_features(train, test, ['Spa', 'VRDeck', 'RoomService'], n_components=2)
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    valid = [c for c in cols if c in train.columns and c in test.columns]
    if len(valid) < 2:
        print(f"fe_pca_features: need >=2 valid cols, got {len(valid)} — skipped")
        return train, test

    n_comp = min(n_components, len(valid))
    train_sub = train[valid].fillna(0).astype(float)
    test_sub = test[valid].fillna(0).astype(float)

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_sub)
    test_scaled = scaler.transform(test_sub)

    pca = PCA(n_components=n_comp, random_state=42)
    train_pca = pca.fit_transform(train_scaled)
    test_pca = pca.transform(test_scaled)

    for i in range(n_comp):
        train[f"pca_{i}"] = train_pca[:, i]
        test[f"pca_{i}"] = test_pca[:, i]

    explained = pca.explained_variance_ratio_
    print(f"fe_pca_features: added pca_0..pca_{n_comp - 1} "
          f"(explained variance: {[f'{v:.2%}' for v in explained]})")
    return train, test


def fe_safe_onehot(
    train: pd.DataFrame, test: pd.DataFrame,
    cols: list[str],
    max_categories: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Safe one-hot encoding with max_categories guard.

    Groups infrequent categories into '_Other' before encoding.
    Aligns train/test columns. Drops the original columns.

    Example: train, test = fe_safe_onehot(train, test, ['HomePlanet', 'Destination'])
    """
    for col in cols:
        if col not in train.columns:
            continue
        # Keep only top categories from train
        vc = train[col].value_counts()
        top_cats = set(vc.head(max_categories).index)
        for df in [train, test]:
            if col in df.columns:
                df[col] = df[col].where(df[col].isin(top_cats), "_Other").fillna("_NA")

        # One-hot encode using train categories as the universe
        all_cats = sorted(train[col].unique())
        for cat in all_cats:
            ohe_name = f"{col}_{cat}"
            train[ohe_name] = (train[col] == cat).astype(int)
            if col in test.columns:
                test[ohe_name] = (test[col] == cat).astype(int)

        # Drop original
        train.drop(columns=[col], inplace=True)
        if col in test.columns:
            test.drop(columns=[col], inplace=True)

    print(f"fe_safe_onehot: one-hot encoded {cols} (max_categories={max_categories})")
    return train, test


def fe_mutual_info(
    train: pd.DataFrame,
    target: pd.Series,
    n_best: int = 20,
    task: str = "auto",
    **kwargs,
) -> list[str]:
    """Score features by mutual information with target. Returns top-n feature names.

    Helps the agent prioritize which features to keep after feature explosion.

    Does NOT modify DataFrames -- returns a ranked list for feature selection.

    Example: best_features = fe_mutual_info(train, y, n_best=15)
    """
    # Alias: LLM hallucinates n_features instead of n_best
    if "n_features" in kwargs:
        n_best = kwargs.pop("n_features")
    # Ignore target_col — the pipeline dispatcher passes the Series directly
    kwargs.pop("target_col", None)
    from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

    num_df = train.select_dtypes(include="number").fillna(0)
    if num_df.shape[1] == 0:
        print("fe_mutual_info: no numeric features found")
        return []

    if task == "auto":
        task = "classification" if target.nunique() <= 20 else "regression"

    if task == "classification":
        mi = mutual_info_classif(num_df, target, random_state=42, n_neighbors=5)
    else:
        mi = mutual_info_regression(num_df, target, random_state=42, n_neighbors=5)

    mi_series = pd.Series(mi, index=num_df.columns).sort_values(ascending=False)
    top = mi_series.head(n_best)

    print(f"=== Mutual Information (top {n_best}, task={task}) ===")
    for feat, score in top.items():
        bar = "#" * int(score / top.max() * 30) if top.max() > 0 else ""
        print(f"  {feat:30s} {score:.4f} {bar}")

    return list(top.index)


def fe_datetime_elapsed(
    train: pd.DataFrame, test: pd.DataFrame,
    col_a: str,
    col_b: str,
    unit: str = "days",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute elapsed time between two datetime columns.

    Creates '{col_a}_minus_{col_b}_{unit}' feature.

    Example: train, test = fe_datetime_elapsed(train, test, 'dropoff_dt', 'pickup_dt', 'seconds')
    """
    name = f"{col_a}_minus_{col_b}_{unit}"
    for df in [train, test]:
        if col_a in df.columns and col_b in df.columns:
            dt_a = pd.to_datetime(df[col_a], errors="coerce")
            dt_b = pd.to_datetime(df[col_b], errors="coerce")
            delta = dt_a - dt_b
            if unit == "seconds":
                df[name] = delta.dt.total_seconds()
            elif unit == "hours":
                df[name] = delta.dt.total_seconds() / 3600
            else:  # days
                df[name] = delta.dt.days.astype(float)
            df[name] = df[name].fillna(0)
    print(f"fe_datetime_elapsed: added {name}")
    return train, test


def fe_datetime_flags(
    train: pd.DataFrame, test: pd.DataFrame,
    col: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add calendar flag features from a datetime column.

    Adds is_weekend, is_month_start, is_month_end, is_quarter_end.

    Example: train, test = fe_datetime_flags(train, test, 'purchase_date')
    """
    created = []
    for df in [train, test]:
        if col not in df.columns:
            continue
        dt = pd.to_datetime(df[col], errors="coerce")
        df[f"{col}_is_weekend"] = dt.dt.dayofweek.ge(5).astype(int)
        df[f"{col}_is_month_start"] = dt.dt.is_month_start.astype(int)
        df[f"{col}_is_month_end"] = dt.dt.is_month_end.astype(int)
        df[f"{col}_is_quarter_end"] = dt.dt.is_quarter_end.astype(int)
        created = [f"{col}_is_weekend", f"{col}_is_month_start",
                   f"{col}_is_month_end", f"{col}_is_quarter_end"]
    print(f"fe_datetime_flags: added {created}")
    return train, test


# =========================================================================
# SECTION 2d: Feature Engineering — Tier 3 (Situational)
# =========================================================================

def fe_power_transform(
    train: pd.DataFrame, test: pd.DataFrame,
    cols: list[str],
    method: str = "yeo-johnson",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply sklearn PowerTransformer to Gaussianize numeric columns.

    Fit on train, transform both. Adds '{col}_pwr' columns.
    Yeo-Johnson handles negatives; Box-Cox requires positive data.

    Example: train, test = fe_power_transform(train, test, ['Income', 'Debt'])
    """
    from sklearn.preprocessing import PowerTransformer

    valid = [c for c in cols if c in train.columns]
    if not valid:
        print("fe_power_transform: no valid columns")
        return train, test

    pt = PowerTransformer(method=method)
    train_vals = train[valid].fillna(0).astype(float)
    test_vals = test[valid].fillna(0).astype(float) if all(c in test.columns for c in valid) else None

    try:
        fitted = pt.fit_transform(train_vals)
        for i, col in enumerate(valid):
            train[f"{col}_pwr"] = fitted[:, i]
        if test_vals is not None:
            test_fitted = pt.transform(test_vals)
            for i, col in enumerate(valid):
                test[f"{col}_pwr"] = test_fitted[:, i]
        print(f"fe_power_transform: added {[f'{c}_pwr' for c in valid]} (method={method})")
    except Exception as e:
        print(f"fe_power_transform: failed — {e}")

    return train, test


def fe_rank_transform(
    train: pd.DataFrame, test: pd.DataFrame,
    cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convert numeric columns to percentile rank (0-1). Robust to outliers.

    Fit rank mapping on train; test values mapped by interpolation into
    train's percentile distribution.

    Example: train, test = fe_rank_transform(train, test, ['Age', 'Fare'])
    """
    for col in cols:
        if col not in train.columns:
            continue
        name = f"{col}_rank"
        # Train: simple percentile rank
        train[name] = train[col].rank(pct=True, method="average").fillna(0.5)
        # Test: map via train's empirical CDF
        if col in test.columns:
            sorted_train = np.sort(train[col].dropna().values)
            n = len(sorted_train)
            if n > 0:
                test[name] = test[col].apply(
                    lambda x: np.searchsorted(sorted_train, x) / n
                    if pd.notna(x) else 0.5
                )
            else:
                test[name] = 0.5
    print(f"fe_rank_transform: added {[f'{c}_rank' for c in cols if c in train.columns]}")
    return train, test


def fe_math_transforms(
    train: pd.DataFrame, test: pd.DataFrame,
    cols: list[str],
    transforms: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply sqrt, square, and/or reciprocal transforms to numeric columns.

    Example: train, test = fe_math_transforms(train, test, ['distance'], ['sqrt', 'square'])
    """
    if transforms is None:
        transforms = ["sqrt", "square"]  # reciprocal excluded by default (risky)
    created = []
    for col in cols:
        for df in [train, test]:
            if col not in df.columns:
                continue
            vals = df[col].fillna(0).astype(float)
            if "sqrt" in transforms:
                df[f"{col}_sqrt"] = np.sqrt(vals.clip(lower=0))
                created.append(f"{col}_sqrt")
            if "square" in transforms:
                df[f"{col}_sq"] = vals ** 2
                created.append(f"{col}_sq")
            if "reciprocal" in transforms:
                safe = vals.where(vals.abs() > 1e-9, 1e-9)
                df[f"{col}_inv"] = 1.0 / safe
                created.append(f"{col}_inv")
    print(f"fe_math_transforms: added {list(dict.fromkeys(created))}")
    return train, test


def fe_lag_features(
    train: pd.DataFrame, test: pd.DataFrame,
    col: str,
    lags: list[int] | None = None,
    sort_col: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create lag features for time-ordered data.

    IMPORTANT: Data must be sorted by time before calling this.
    If sort_col is provided, sorts both DataFrames first.

    Example: train, test = fe_lag_features(train, test, 'sales', lags=[1, 7], sort_col='date')
    """
    if lags is None:
        lags = [1, 2, 3]
    created = []
    for df in [train, test]:
        if col not in df.columns:
            continue
        if sort_col and sort_col in df.columns:
            df.sort_values(sort_col, inplace=True)
        for lag in lags:
            name = f"{col}_lag_{lag}"
            df[name] = df[col].shift(lag)
            created.append(name)
    print(f"fe_lag_features: added {list(dict.fromkeys(created))}")
    return train, test


def fe_rolling_stats(
    train: pd.DataFrame, test: pd.DataFrame,
    col: str,
    windows: list[int] | None = None,
    stats: list[str] | None = None,
    sort_col: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute rolling window statistics for time-ordered data.

    IMPORTANT: Data must be sorted by time before calling this.

    Example: train, test = fe_rolling_stats(train, test, 'sales', windows=[7, 14], stats=['mean', 'std'])
    """
    if windows is None:
        windows = [3, 7]
    if stats is None:
        stats = ["mean", "std"]
    created = []
    for df in [train, test]:
        if col not in df.columns:
            continue
        if sort_col and sort_col in df.columns:
            df.sort_values(sort_col, inplace=True)
        for w in windows:
            roller = df[col].rolling(window=w, min_periods=1)
            for stat in stats:
                name = f"{col}_roll{w}_{stat}"
                df[name] = getattr(roller, stat)()
                created.append(name)
    print(f"fe_rolling_stats: added {list(dict.fromkeys(created))}")
    return train, test


def fe_string_features(
    train: pd.DataFrame, test: pd.DataFrame,
    cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extract string length and word count from text/string columns.

    Adds '{col}_len' and '{col}_wordcount' features.

    Example: train, test = fe_string_features(train, test, ['Name', 'Title'])
    """
    created = []
    for col in cols:
        for df in [train, test]:
            if col not in df.columns:
                continue
            s = df[col].fillna("").astype(str)
            df[f"{col}_len"] = s.str.len()
            df[f"{col}_wordcount"] = s.str.split().str.len().fillna(0).astype(int)
            created.extend([f"{col}_len", f"{col}_wordcount"])
    print(f"fe_string_features: added {list(dict.fromkeys(created))}")
    return train, test


def fe_count_encode(
    train: pd.DataFrame, test: pd.DataFrame,
    cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Replace each category with its raw count in the training set.

    Adds '{col}_count' columns. Fit on train only; unseen test categories → 1.

    Example: train, test = fe_count_encode(train, test, ['HomePlanet'])
    """
    for col in cols:
        if col not in train.columns:
            continue
        counts = train[col].value_counts()
        train[f"{col}_count"] = train[col].map(counts).fillna(1).astype(int)
        if col in test.columns:
            test[f"{col}_count"] = test[col].map(counts).fillna(1).astype(int)
    print(f"fe_count_encode: added {[f'{c}_count' for c in cols if c in train.columns]}")
    return train, test


def fe_polynomial(
    train: pd.DataFrame, test: pd.DataFrame,
    cols: list[str],
    interaction_only: bool = False,
    **kwargs,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create degree-2 polynomial features (squared terms + pairwise products).

    WARNING: n columns → n*(n+1)/2 new features. Limit input to <10 columns.
    Use interaction_only=True to skip squared terms.

    Example: train, test = fe_polynomial(train, test, ['Age', 'Fare'], interaction_only=True)
    """
    # Alias: LLM hallucinates degree param (always degree 2, not configurable)
    kwargs.pop("degree", None)
    from sklearn.preprocessing import PolynomialFeatures

    valid = [c for c in cols if c in train.columns and c in test.columns]
    if len(valid) < 2:
        print(f"fe_polynomial: need >=2 cols, got {len(valid)} — skipped")
        return train, test
    if len(valid) > 10:
        print(f"fe_polynomial: too many cols ({len(valid)}) — limiting to first 10")
        valid = valid[:10]

    pf = PolynomialFeatures(degree=2, interaction_only=interaction_only, include_bias=False)
    train_sub = train[valid].fillna(0).astype(float)
    test_sub = test[valid].fillna(0).astype(float)

    train_poly = pf.fit_transform(train_sub)
    test_poly = pf.transform(test_sub)

    feature_names = pf.get_feature_names_out(valid)
    # Skip the original features (already in the DataFrame)
    new_mask = [i for i, name in enumerate(feature_names) if name not in valid]
    new_names = [f"poly_{feature_names[i]}" for i in new_mask]

    for i, nm in zip(new_mask, new_names):
        train[nm] = train_poly[:, i]
        test[nm] = test_poly[:, i]

    print(f"fe_polynomial: added {len(new_names)} features from {valid}")
    return train, test


def fe_drop_constant(
    train: pd.DataFrame, test: pd.DataFrame,
    threshold: float = 0.99,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drop columns where a single value dominates >= threshold fraction of rows.

    Cleanup pass after feature engineering to remove noise features.

    Example: train, test = fe_drop_constant(train, test, threshold=0.99)
    """
    drop_cols = []
    for col in train.columns:
        vc = train[col].value_counts(normalize=True, dropna=False)
        if vc.iloc[0] >= threshold:
            drop_cols.append(col)

    if drop_cols:
        train.drop(columns=drop_cols, inplace=True)
        test.drop(columns=[c for c in drop_cols if c in test.columns], inplace=True)
    print(f"fe_drop_constant: dropped {len(drop_cols)} cols (threshold={threshold}): {drop_cols}")
    return train, test


# =========================================================================
# SECTION 3: Preprocessing
# =========================================================================

def preprocess(
    train: pd.DataFrame, test: pd.DataFrame,
    drop_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Full preprocessing pipeline: impute, encode, downcast.

    From Phase 0: numeric→median, categorical→mode+LabelEncoder, float64→float32.
    Optionally drops specified columns first.
    NEVER uses pd.get_dummies (OOM risk on high-cardinality columns).
    """
    from sklearn.preprocessing import LabelEncoder


    if drop_cols:
        train = train.drop(columns=drop_cols, errors="ignore")
        test = test.drop(columns=drop_cols, errors="ignore")

    # Impute numeric with median
    num_cols = list(train.select_dtypes(include="number").columns)
    for c in num_cols:
        med = train[c].median()
        train[c] = train[c].fillna(med)
        if c in test.columns:
            test[c] = test[c].fillna(med)

    # Impute object/str with mode, then LabelEncode
    obj_cols = list(train.select_dtypes(include=["object", "str"]).columns)
    for c in obj_cols:
        mode_val = train[c].mode().iloc[0] if len(train[c].mode()) > 0 else "Unknown"
        train[c] = train[c].fillna(mode_val)
        if c in test.columns:
            test[c] = test[c].fillna(mode_val)
        le = LabelEncoder()
        combined = pd.concat([train[c], test[c]], ignore_index=True) if c in test.columns else train[c]
        le.fit(combined)
        train[c] = le.transform(train[c])
        if c in test.columns:
            test[c] = le.transform(test[c])

    # Float32 downcast
    f64 = train.select_dtypes(include=["float64"]).columns
    if len(f64):
        train[f64] = train[f64].astype(np.float32)
        f64t = [c for c in f64 if c in test.columns]
        if f64t:
            test[f64t] = test[f64t].astype(np.float32)

    n_obj = train.select_dtypes(include=["object", "str"]).shape[1]
    print(f"preprocess: imputed + encoded {len(obj_cols)} cat cols, "
          f"{len(num_cols)} num cols → {n_obj} object cols remain")
    return train, test


# =========================================================================
# SECTION 4: Model Building & Evaluation
# =========================================================================

def build_model(task: str = "binary", **kwargs):
    """Create a LightGBM model with good defaults. Override any param via kwargs.

    task: 'binary', 'multiclass', or 'regression'.
    Returns an unfitted model instance.
    """
    from lightgbm import LGBMClassifier, LGBMRegressor

    defaults = dict(
        n_estimators=500, learning_rate=0.05, num_leaves=31,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        verbose=-1, n_jobs=_MAX_JOBS,
    )
    defaults.update(kwargs)

    if task == "regression":
        model = LGBMRegressor(**defaults)
    else:
        model = LGBMClassifier(**defaults)

    print(f"build_model: {type(model).__name__} with {defaults}")
    return model


def build_ensemble(task: str = "binary"):
    """Create a 4-model VotingClassifier/Regressor ensemble.

    From Phase 0: LGBM + XGBoost + ExtraTrees + CatBoost with soft voting.
    This is the ensemble that gets Gold on Kaggle competitions.
    """
    from lightgbm import LGBMClassifier, LGBMRegressor
    from xgboost import XGBClassifier, XGBRegressor
    from sklearn.ensemble import (VotingClassifier, VotingRegressor,
                                  ExtraTreesClassifier, ExtraTreesRegressor)
    from catboost import CatBoostClassifier, CatBoostRegressor

    if task == "regression":
        lgbm = LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=31,
                              subsample=0.8, colsample_bytree=0.8, random_state=42,
                              verbose=-1, n_jobs=_MAX_JOBS)
        xgb = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6,
                            subsample=0.8, colsample_bytree=0.8, random_state=42,
                            verbosity=0, nthread=_MAX_JOBS)
        et = ExtraTreesRegressor(n_estimators=300, random_state=42, n_jobs=_MAX_JOBS)
        cat = CatBoostRegressor(iterations=500, learning_rate=0.05, depth=6,
                                 random_seed=42, verbose=0, thread_count=_MAX_JOBS)
        model = VotingRegressor(estimators=[("lgbm", lgbm), ("xgb", xgb),
                                            ("et", et), ("cat", cat)])
    else:
        lgbm = LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=31,
                               subsample=0.8, colsample_bytree=0.8, random_state=42,
                               verbose=-1, n_jobs=_MAX_JOBS)
        xgb = XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=6,
                             subsample=0.8, colsample_bytree=0.8, random_state=42,
                             verbosity=0, eval_metric="logloss", nthread=_MAX_JOBS)
        et = ExtraTreesClassifier(n_estimators=300, random_state=42, n_jobs=_MAX_JOBS)
        cat = CatBoostClassifier(iterations=500, learning_rate=0.05, depth=6,
                                  random_seed=42, verbose=0, thread_count=_MAX_JOBS)
        model = VotingClassifier(estimators=[("lgbm", lgbm), ("xgb", xgb),
                                             ("et", et), ("cat", cat)],
                                 voting="soft")

    print(f"build_ensemble: {type(model).__name__} (4 models, task={task})")
    return model


def evaluate_cv(
    model, X: pd.DataFrame, y: pd.Series,
    scoring: str = "accuracy", n_folds: int = 5,
    max_rows: int = 20_000,
) -> float:
    """Cross-validate a model. Prints CV_METRIC/CV_SCORE for the keep-best system.

    Subsamples to max_rows for speed on large datasets.
    Returns mean CV score.
    """
    from sklearn.model_selection import cross_val_score

    if len(X) > max_rows:
        idx = np.random.RandomState(42).choice(len(X), max_rows, replace=False)
        X_cv, y_cv = X.iloc[idx], y.iloc[idx]
        print(f"evaluate_cv: subsampled {len(X)} → {max_rows} rows for speed")
    else:
        X_cv, y_cv = X, y

    cv = cross_val_score(model, X_cv, y_cv, cv=n_folds, scoring=scoring, n_jobs=1)
    print(f"CV_METRIC={scoring}")
    print(f"CV_SCORE={cv.mean():.6f}")
    print(f"CV_STD={cv.std():.6f}")
    return cv.mean()


def train_and_predict(
    model, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame,
    proba: bool = False,
) -> np.ndarray:
    """Fit model on full training data and predict on test.

    Set proba=True for predict_proba (probability-based scoring like AUC).
    """
    model.fit(X_train, y_train)
    if proba and hasattr(model, "predict_proba"):
        preds = model.predict_proba(X_test)[:, 1]
    else:
        preds = model.predict(X_test)
    print(f"train_and_predict: fitted on {X_train.shape}, predicted {len(preds)} samples")
    return preds


def stack_ensemble(
    X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame,
    task: str = "binary", n_folds: int = 5,
) -> tuple[np.ndarray, float]:
    """Stacking ensemble: LGBM + XGB + ExtraTrees → LogisticRegression meta-learner.

    Returns (test_predictions, oof_cv_score).
    """
    from lightgbm import LGBMClassifier, LGBMRegressor
    from xgboost import XGBClassifier, XGBRegressor
    from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.model_selection import StratifiedKFold, KFold
    from sklearn.base import clone

    X_tr = X_train.values if hasattr(X_train, "values") else np.array(X_train)
    X_te = X_test.values if hasattr(X_test, "values") else np.array(X_test)
    y_tr = y_train.values if hasattr(y_train, "values") else np.array(y_train)
    n_train, n_test = len(X_tr), len(X_te)

    if task == "regression":
        base_models = [
            LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=31,
                          subsample=0.8, colsample_bytree=0.8, random_state=42,
                          verbose=-1, n_jobs=_MAX_JOBS),
            XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6,
                         subsample=0.8, colsample_bytree=0.8, random_state=42,
                         verbosity=0, nthread=_MAX_JOBS),
            ExtraTreesRegressor(n_estimators=300, random_state=42, n_jobs=_MAX_JOBS),
        ]
        meta = Ridge(alpha=1.0)
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    else:
        base_models = [
            LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=31,
                           subsample=0.8, colsample_bytree=0.8, random_state=42,
                           verbose=-1, n_jobs=_MAX_JOBS),
            XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=6,
                          subsample=0.8, colsample_bytree=0.8, random_state=42,
                          verbosity=0, eval_metric="logloss", nthread=_MAX_JOBS),
            ExtraTreesClassifier(n_estimators=300, random_state=42, n_jobs=_MAX_JOBS),
        ]
        meta = LogisticRegression(max_iter=1000, random_state=42)
        kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    oof = np.zeros((n_train, len(base_models)))
    test_preds = np.zeros((n_test, len(base_models)))

    for i, base in enumerate(base_models):
        name = type(base).__name__
        print(f"  stacking: {name}...")
        fold_test = np.zeros((n_test, n_folds))
        for fi, (tr_idx, val_idx) in enumerate(kf.split(X_tr, y_tr)):
            m = clone(base)
            m.fit(X_tr[tr_idx], y_tr[tr_idx])
            if task != "regression" and hasattr(m, "predict_proba"):
                oof[val_idx, i] = m.predict_proba(X_tr[val_idx])[:, 1]
                fold_test[:, fi] = m.predict_proba(X_te)[:, 1]
            else:
                oof[val_idx, i] = m.predict(X_tr[val_idx])
                fold_test[:, fi] = m.predict(X_te)
            del m
        test_preds[:, i] = fold_test.mean(axis=1)

    print("  stacking: meta-learner...")
    meta.fit(oof, y_tr)
    if task != "regression":
        final = meta.predict(test_preds)
        oof_score = float((meta.predict(oof) == y_tr).mean())
    else:
        final = meta.predict(test_preds)
        from sklearn.metrics import r2_score
        oof_score = float(r2_score(y_tr, meta.predict(oof)))

    print(f"stack_ensemble: OOF score = {oof_score:.6f}")
    return final, oof_score


def tune_lgbm(
    X_train: pd.DataFrame, y_train: pd.Series,
    task: str = "binary", n_trials: int = 20, timeout: int = 60,
) -> dict:
    """Quick Optuna hyperparameter search for LightGBM (time-boxed).

    Returns best params dict ready for build_model().
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    from lightgbm import LGBMClassifier, LGBMRegressor
    from sklearn.model_selection import cross_val_score

    scoring = "neg_mean_squared_error" if task == "regression" else "accuracy"

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1000, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "random_state": 42, "verbose": -1,
        }
        model = LGBMRegressor(**params) if task == "regression" else LGBMClassifier(**params)
        return cross_val_score(model, X_train, y_train, cv=3, scoring=scoring, n_jobs=1).mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    best = study.best_params
    best.update({"random_state": 42, "verbose": -1})
    print(f"tune_lgbm: best score={study.best_value:.6f} in {len(study.trials)} trials")
    print(f"tune_lgbm: params={best}")
    return best


def write_submission(
    ids: pd.Series, predictions, id_col: str, target_col: str,
    data_dir: str, target_dtype: str = "bool",
) -> None:
    """Write submission.csv matching the expected sample format.

    Handles bool conversion for binary classification.
    """
    sub = pd.DataFrame({id_col: ids, target_col: predictions})

    if target_dtype == "bool":
        sub[target_col] = sub[target_col].astype(bool)
    elif target_dtype == "int":
        sub[target_col] = sub[target_col].astype(int)

    path = os.path.join(data_dir, "submission.csv")
    sub.to_csv(path, index=False)
    print(f"write_submission: wrote {path}")
    print(f"  shape: {sub.shape}")
    print(f"  dtypes:\n{sub.dtypes.to_string()}")
    print(f"  head:\n{sub.head().to_string()}")


def feature_importance(model, feature_names: list[str], top_n: int = 20) -> pd.Series:
    """Show feature importances from a fitted model. Returns sorted Series."""
    if hasattr(model, "feature_importances_"):
        imp = pd.Series(model.feature_importances_, index=feature_names)
    elif hasattr(model, "coef_"):
        imp = pd.Series(np.abs(model.coef_).flatten()[:len(feature_names)], index=feature_names)
    else:
        print("Model has no feature_importances_ or coef_ attribute")
        return pd.Series(dtype=float)

    imp = imp.sort_values(ascending=False)
    print(f"=== Feature Importance (top {top_n}) ===")
    for feat, val in imp.head(top_n).items():
        bar = "#" * int(val / imp.max() * 30) if imp.max() > 0 else ""
        print(f"  {feat:30s} {val:8.1f} {bar}")
    return imp


# =========================================================================
# Registry
# =========================================================================

TOOLKIT_FUNCTIONS = {
    # EDA
    "eda_summary": eda_summary,
    "eda_correlations": eda_correlations,
    "eda_target": eda_target,
    "eda_profile_columns": eda_profile_columns,
    "eda_find_target": eda_find_target,
    "eda_null_analysis": eda_null_analysis,
    # Feature Engineering — Existing (14)
    "fe_target_encode": fe_target_encode,
    "fe_frequency_encode": fe_frequency_encode,
    "fe_interactions": fe_interactions,
    "fe_agg_features": fe_agg_features,
    "fe_binning": fe_binning,
    "fe_log_transform": fe_log_transform,
    "fe_bool_convert": fe_bool_convert,
    "fe_split_structured": fe_split_structured,
    "fe_numeric_clean": fe_numeric_clean,
    "fe_datetime_features": fe_datetime_features,
    "fe_group_size": fe_group_size,
    "fe_null_indicators": fe_null_indicators,
    "fe_spending_aggs": fe_spending_aggs,
    "fe_haversine": fe_haversine,
    # Feature Engineering — Tier 1 (6)
    "fe_ratios": fe_ratios,
    "fe_differences": fe_differences,
    "fe_row_stats": fe_row_stats,
    "fe_categorical_cross": fe_categorical_cross,
    "fe_null_row_count": fe_null_row_count,
    "fe_cyclical_encode": fe_cyclical_encode,
    # Feature Engineering — Tier 2 (7)
    "fe_percentile_rank_group": fe_percentile_rank_group,
    "fe_cluster_labels": fe_cluster_labels,
    "fe_pca_features": fe_pca_features,
    "fe_safe_onehot": fe_safe_onehot,
    "fe_mutual_info": fe_mutual_info,
    "fe_datetime_elapsed": fe_datetime_elapsed,
    "fe_datetime_flags": fe_datetime_flags,
    # Feature Engineering — Tier 3 (9)
    "fe_power_transform": fe_power_transform,
    "fe_rank_transform": fe_rank_transform,
    "fe_math_transforms": fe_math_transforms,
    "fe_lag_features": fe_lag_features,
    "fe_rolling_stats": fe_rolling_stats,
    "fe_string_features": fe_string_features,
    "fe_count_encode": fe_count_encode,
    "fe_polynomial": fe_polynomial,
    "fe_drop_constant": fe_drop_constant,
    # Preprocessing
    "preprocess": preprocess,
    # Modeling
    "build_model": build_model,
    "build_ensemble": build_ensemble,
    "evaluate_cv": evaluate_cv,
    "train_and_predict": train_and_predict,
    "stack_ensemble": stack_ensemble,
    "tune_lgbm": tune_lgbm,
    "write_submission": write_submission,
    "feature_importance": feature_importance,
}

TOOLKIT_DOCS = """
<toolkit>
## Pre-loaded ML Toolkit (call directly — no imports needed)
## ALL fe_* and preprocess functions return (train, test). ALWAYS assign:
##   train, test = fe_*(train, test, ...)
##   train, test = preprocess(train, test, ...)

### EDA — Explore first, then engineer features based on findings
- eda_summary(df, name) → shape, dtypes, nulls, numeric stats, categoricals
- eda_correlations(df, target_col) → sorted correlations with target
- eda_target(df, target_col) → target type (binary/multiclass/regression), balance
- eda_profile_columns(df, name) → deep role classification per column:
    ID, CONSTANT, BINARY, BOOL_STR, NUMERIC_AS_STR, STRUCTURED_STR,
    HIGH_CARD, CATEGORICAL, CONTINUOUS, FREE_TEXT
- eda_find_target(train, test, sample_sub) → detect target, id_col, task type
- eda_null_analysis(df, name) → null patterns, zero-inflated cols, imputation strategies

### Feature Engineering — Data Cleaning (return (train, test))
- fe_bool_convert(train, test, cols) → boolean strings → int 0/1
- fe_split_structured(train, test, col, delimiter) → split delimited strings into columns
- fe_numeric_clean(train, test, cols) → strip $,€,% noise from numeric-as-string
- fe_null_indicators(train, test, threshold_pct=1.0) → binary null flag columns
- fe_null_row_count(train, test, cols=None) → total nulls per row
- fe_string_features(train, test, cols) → string length + word count

### Feature Engineering — Numeric Transforms (return (train, test))
- fe_log_transform(train, test, cols) → log1p for skewed numerics
- fe_power_transform(train, test, cols, method='yeo-johnson') → Gaussianize via Box-Cox/Yeo-Johnson
- fe_rank_transform(train, test, cols) → percentile rank (0-1), robust to outliers
- fe_math_transforms(train, test, cols, transforms=['sqrt','square']) → sqrt/square/reciprocal
- fe_cyclical_encode(train, test, col, max_value) → sin/cos for periodic features (hour/month)

### Feature Engineering — Interactions & Combinations (return (train, test))
- fe_interactions(train, test, [(colA, colB)]) → pairwise products (col_a * col_b)
- fe_ratios(train, test, [(colA, colB)]) → ratio features (col_a / col_b)
- fe_differences(train, test, [(colA, colB)]) → difference features (col_a - col_b)
- fe_row_stats(train, test, cols, stats=['mean','std','min','max']) → row-wise stats
- fe_categorical_cross(train, test, [(catA, catB)], max_categories=50) → combined categories
- fe_polynomial(train, test, cols, interaction_only=False) → degree-2 polynomial features

### Feature Engineering — Encoding (return (train, test))
- fe_target_encode(train, test, cols, target, n_folds=5) → out-of-fold target encoding
- fe_frequency_encode(train, test, cols) → value frequency as feature
- fe_count_encode(train, test, cols) → raw category count from training set
- fe_safe_onehot(train, test, cols, max_categories=20) → safe one-hot with cardinality guard

### Feature Engineering — Group & Aggregation (return (train, test))
- fe_group_size(train, test, id_col, delimiter) → GroupSize + IsSolo from structured IDs
- fe_agg_features(train, test, group_col, agg_cols, ['mean','std']) → group aggregations
- fe_spending_aggs(train, test, spend_cols, group_col=None) → TotalSpend/IsZeroSpend/logs
- fe_percentile_rank_group(train, test, group_col, value_col) → rank within group (0-1)

### Feature Engineering — Datetime (return (train, test))
- fe_datetime_features(train, test, col) → hour/dayofweek/month/year extraction
- fe_datetime_elapsed(train, test, col_a, col_b, unit='days') → time between two dates
- fe_datetime_flags(train, test, col) → is_weekend/month_start/month_end/quarter_end
- fe_cyclical_encode(train, test, col, max_value) → sin/cos for hour(24)/dow(7)/month(12)

### Feature Engineering — Unsupervised (return (train, test))
- fe_cluster_labels(train, test, cols, n_clusters=8) → KMeans cluster ID as feature
- fe_pca_features(train, test, cols, n_components=3) → PCA principal components

### Feature Engineering — Geo & Distance
- fe_haversine(df, lat1, lon1, lat2, lon2) → geodesic distance feature

### Feature Engineering — Time Series (return (train, test))
- fe_lag_features(train, test, col, lags=[1,2,3], sort_col=None) → lagged values
- fe_rolling_stats(train, test, col, windows=[3,7], stats=['mean','std']) → rolling window

### Feature Engineering — Selection & Cleanup
- fe_mutual_info(train, target, n_best=20) → top features by mutual information (returns list)
- fe_drop_constant(train, test, threshold=0.99) → drop quasi-constant columns
- fe_binning(train, test, col, bins=5) → quantile-based bins

### Preprocessing — MUST assign: train, test = preprocess(...)
- preprocess(train, test, drop_cols) → impute + LabelEncode + float32 downcast. NEVER pd.get_dummies!

### Modeling — Build, evaluate, predict
- build_model(task='binary', **kwargs) → LightGBM with good defaults
- build_ensemble(task='binary') → 4-model LGBM+XGB+ET+CatBoost voting ensemble
- evaluate_cv(model, X, y, scoring='accuracy', max_rows=20000) → CV_METRIC/CV_SCORE
- train_and_predict(model, X_train, y, X_test, proba=False) → fit & predict
- stack_ensemble(X_train, y, X_test, task) → 3-model stacking (preds, oof_score)
- tune_lgbm(X_train, y, task, n_trials=20, timeout=60) → Optuna search → best params
- write_submission(ids, preds, id_col, target_col, data_dir, target_dtype) → submission.csv
- feature_importance(model, feature_names) → ranked importance chart
</toolkit>
"""
