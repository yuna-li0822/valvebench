# Heart Valve Project

This workspace contains a compact benchmark system for synthetic heart-valve scaffold prediction. The canonical implementation now lives in the ValveBench package and is designed for reproducible model comparison.

## Structure

- `data/` — synthetic and processed datasets
- `valvebench/` — canonical benchmark package for data, splits, models, and evaluation
- `notebooks/` — analysis notebooks
- `experiments/` — training and evaluation scripts
- `leaderboard.md` — first generated benchmark leaderboard

## Quick start

1. Install dependencies:

```bash
py -3 -m pip install -r requirements.txt
```

2. Run the smoke test:

```bash
py -3 aipv/heart_valve_project/smoke_test.py
```

3. Run the full benchmark pipeline:

```bash
py -3 aipv/heart_valve_project/run_pipeline.py
```

4. Regenerate the leaderboard:

```bash
py -3 aipv/heart_valve_project/valvebench/evaluation/generate_leaderboard.py
```

## Research definition

ValveBench is a reproducible machine learning benchmark for predicting pediatric heart-valve scaffold behavior from synthetic design parameters. The benchmark focuses on a single, frozen synthetic dataset version, `v0.1`, and evaluates multiple baseline models under identical experimental conditions.

The current task is to predict valve diameter outcomes from scaffold design features such as initial diameter, fold geometry, material properties, and suture dissolution behavior. The benchmark is intended to support fair comparison between linear, regularized, neural, and physics-informed approaches.

## Benchmark protocol

- Dataset version: `v0.1`
- Splits: deterministic, leakage-safe patient-level train/validation/test splits
- Age handling: age-stratified splits using age bins so each split preserves age-group coverage
- Evaluation: MAE and MSE are reported for each model

## Baseline models

- Linear regression
- Ridge regression
- Multi-layer perceptron (MLP)
- PINN-style placeholder
