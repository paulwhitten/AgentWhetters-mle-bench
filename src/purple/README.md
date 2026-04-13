---
title: Purple Agent
description: Hybrid deterministic and LLM-guided agent for MLE-Bench Green tabular ML competitions
ms.date: 2026-04-12
---

## Overview

Purple is a tabular ML competition agent that combines deterministic
feature engineering with LLM-guided iteration. It runs as an A2A
server inside the MLE-Bench Green harness and targets Kaggle-style
binary classification, multiclass, and regression tasks within a
600-second budget.

The agent produces a `submission.csv` without any human intervention.

## Architecture

Execution follows five sequential phases:

| Phase | Name | LLM | Time |
|-------|------|-----|------|
| 0 | Deterministic baseline | No | ~34 s |
| A | EDA analysis | Yes | ~10 s |
| B | Feature engineering | Yes | ~15 s |
| C | Model training | Yes | ~30 s |
| D | Iteration loop | Hybrid | remainder |

**Phase 0** generates and executes a complete Python script with no LLM
calls. It performs EDA, feature engineering (36 `fe_*` functions from
`ml_toolkit.py`), and cross-validation to establish a CV reference
target for later phases.

**Phases A–C** use gpt-4o-mini with function calling (strict mode) to
analyze the data, refine features, and train an initial ensemble.

**Phase D** iterates through a 19-level hint ladder, alternating
between feature engineering rounds and model rounds until the time
budget runs out.

## Files

- [server.py](server.py) — A2A server, phase orchestration, prompt
  engineering, tool implementations, deterministic codegen
- [ml_toolkit.py](ml_toolkit.py) — 36 feature engineering functions,
  model builders, CV evaluation, stacking, and Optuna tuning
- [\_\_init\_\_.py](__init__.py) — package marker

## Rules of ML

The agent's design draws heavily from Google's
[Rules of Machine Learning](https://developers.google.com/machine-learning/guides/rules-of-ml).
Key rules and where they appear:

| Rule | Principle | How it applies |
|------|-----------|----------------|
| #7 | Turn heuristics into features | Phase 0 mines bool strings, structured IDs, null indicators, and spending aggregates as features |
| #16 | Plan to launch and iterate | Phase 0 ships a working baseline immediately; Phase D iterates on top of it |
| #17 | Use directly observed features | Frequency encoding of categorical columns |
| #20 | Combine and modify existing features | Polynomial interactions, log transforms, ratio features, binning |
| #22 | Clean up unused features | Phase B zero-importance pruning (capped at 20% to prevent over-pruning) |
| #26 | Look for patterns in measured errors | Feature importance feedback after every CV round |

## Hint Ladder

Phase D uses a progressive 19-level hint ladder that feeds the LLM
structured improvement suggestions. MODEL hints are interleaved early
so they are reachable within the typical 9–12 round budget:

| Level | Type | Technique |
|-------|-------|-----------|
| 0 | FE | Kitchen sink — all applicable transforms at once |
| 1 | FE | Data quality cleanup |
| 2 | FE | Target encoding |
| 3 | FE | Error analysis — targeted features |
| 4 | FE | Feature interactions |
| 5 | MODEL | Low learning rate + stochastic subsampling |
| 6 | FE | Aggregation features |
| 7 | FE | Logical imputation + binning |
| 8 | MODEL | Stacking (LGBM + XGB + ExtraTrees meta-learner) |
| 9 | FE | Rank and power transforms |
| 10 | MODEL | Optuna hyperparameter tuning |
| 11 | FE | Frequency encoding + outlier clipping |
| 12 | MODEL | Threshold optimization / prediction clipping |
| 13 | FE | Mutual info selection |
| 14 | FE | Permutation pruning |
| 15 | FE | Adversarial validation |
| 16 | FE | Deviation-from-group |
| 17 | FE | Pseudo-labeling |
| 18 | FE | PCA + K-Means clusters |

FE hints add features to a cumulative snapshot — each round builds on
the best prior state rather than restarting from raw data. MODEL hints
skip feature engineering and only retrain, so they complete in roughly
half the time.

## Ensemble

The default model is a soft-voting ensemble of four classifiers (or
regressors for regression tasks):

1. LightGBM
2. XGBoost
3. ExtraTrees
4. CatBoost

Phase D MODEL hints can upgrade to a **stacking ensemble** (LGBM +
XGB + ExtraTrees base learners with a logistic regression
meta-learner) or trigger **Optuna tuning** on the LightGBM
estimator.

## FAST\_CV Mode

When `PURPLE_FAST_CV=1` (default), Phase D FE rounds evaluate with a
single LightGBM model (~11 s) instead of the full ensemble (~56 s).
This allows roughly 3× more iteration rounds within the same budget.

Once stacking wins a round, subsequent FE rounds automatically switch
from LGBM-only to re-stacking (~24 s) so the comparison is
apples-to-apples. Without this, LGBM-only CV can never beat a
stacking OOF score and every post-stacking FE round would be wasted.

A final ensemble retrain runs after Phase D completes, but only when
the best round was a fast-CV round — stacking submissions are
preserved as-is.
