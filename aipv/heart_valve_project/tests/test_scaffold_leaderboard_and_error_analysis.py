import pytest


def torch_available():
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not torch_available(), reason="torch not installed")


def test_generate_scaffold_leaderboard_writes_a_valid_table(tmp_path):
    from valvebench.evaluation.generate_scaffold_leaderboard import generate_scaffold_leaderboard

    output_path = tmp_path / "scaffold_leaderboard.md"
    result_path = generate_scaffold_leaderboard(
        output_path=str(output_path), seeds=(0,), n_train_samples=20
    )
    assert result_path == output_path
    text = output_path.read_text(encoding="utf-8")
    assert "mlp_only" in text
    assert "PINNVALVE" in text
    assert "OOD" in text


def test_per_corner_errors_covers_every_corner_design():
    from valvebench.datasets.synthetic import interior_train_val_test_split_sma, ood_corner_dataset_sma
    from valvebench.models.pinn_scaffold import TrainConfig, train_scaffold_pinn

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
    import error_analysis

    train_df, val_df, id_test_df = interior_train_val_test_split_sma(n_samples=20, seed=1)
    ood_df = ood_corner_dataset_sma(seed=123)

    config = TrainConfig(epochs=40, hidden_dim=8, physics_weight=1.0, physics_ramp_epochs=15, log_every=0)
    result = train_scaffold_pinn(train_df, config)

    per_design = error_analysis.per_corner_errors(result.model, ood_df)
    assert per_design["sample_id"].nunique() == ood_df["sample_id"].nunique() == 16
    for col in error_analysis.PARAM_COLS:
        assert f"{col}_side" in per_design.columns
        assert set(per_design[f"{col}_side"].unique()) <= {"low", "high"}


def test_summarize_by_parameter_side_covers_all_parameters_and_sides():
    from valvebench.datasets.synthetic import ood_corner_dataset_sma
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
    import error_analysis
    import numpy as np

    ood_df = ood_corner_dataset_sma(seed=123)
    per_design = ood_df.groupby("sample_id").agg(
        abs_error=("Diameter_mm", lambda x: float(np.random.RandomState(0).rand())),
        **{col: (col, "first") for col in error_analysis.PARAM_COLS},
    ).reset_index()
    midpoints = {
        "Suture_Count": 7, "Material_Stiffness": 1.5, "Dissolution_Rate": 0.175, "Fold_Angle": 37.5,
    }
    for col, mid in midpoints.items():
        per_design[f"{col}_side"] = np.where(per_design[col] < mid, "low", "high")

    summary = error_analysis.summarize_by_parameter_side(per_design)
    assert len(summary) == len(error_analysis.PARAM_COLS) * 2
    assert set(summary["parameter"]) == set(error_analysis.PARAM_COLS)
