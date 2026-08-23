import pytest

from valvebench.tasks import AGE_BANDS, run_task
from valvebench.tasks.annular_growth_12mo import _age_band


def test_age_band_boundaries():
    assert _age_band(0.0) == "infant"
    assert _age_band(1.9) == "infant"
    assert _age_band(2.0) == "child"
    assert _age_band(9.9) == "child"
    assert _age_band(10.0) == "adolescent"
    assert _age_band(17.9) == "adolescent"


def test_run_task_returns_overall_and_per_stratum_metrics():
    result = run_task("linear", n_samples=150, seed=1)
    assert "MAE" in result.overall and "MSE" in result.overall
    assert set(result.per_stratum.keys()) == set(AGE_BANDS.keys())
    for band, metrics in result.per_stratum.items():
        assert "MAE" in metrics and "n" in metrics
    assert result.n_train + result.n_test <= 150
    assert result.n_train > 0 and result.n_test > 0


def test_run_task_unknown_model_raises():
    with pytest.raises(KeyError):
        run_task("not_a_real_model")


def test_monotone_gbm_has_fewer_violations_than_unconstrained_gbm():
    unconstrained = run_task("gbm", n_samples=200, seed=42)
    constrained = run_task("gbm_monotone", n_samples=200, seed=42)
    assert constrained.monotonicity_violation_rate <= unconstrained.monotonicity_violation_rate
    assert constrained.monotonicity_violation_rate == pytest.approx(0.0, abs=1e-9)
