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

**Phase D** iterates through a 17-level hint ladder, alternating
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

Phase D uses a progressive hint ladder that feeds the LLM structured
improvement suggestions. Hints are ordered by expected impact per
second of compute time:

| Level | Hint type | Technique |
|-------|-----------|-----------|
| 0 | FE | Kitchen sink — all applicable transforms at once |
| 1–2 | FE | Data quality cleanup, target encoding |
| 3–5 | FE | Feature interactions, aggregation, rank transforms |
| 6–8 | FE | Frequency encoding, mutual info selection, PCA + clusters |
| 9–12 | FE | Deviation-from-group, imputation, permutation pruning, pseudo-labeling |
| 13 | MODEL | Low learning rate + stochastic subsampling |
| 14 | MODEL | Stacking (LGBM + XGB + ExtraTrees with meta-learner) |
| 15 | MODEL | Optuna hyperparameter tuning |
| 16 | MODEL | Threshold optimization / prediction clipping |

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
A final ensemble retrain runs after Phase D completes, but only when
the best round was a fast-CV round — MODEL hint rounds that produce
stacking or tuned submissions are preserved as-is.
