from ..pipeline import run_pipeline


def smoke_test():
    print("Generating dataset...")
    metrics = run_pipeline(model_name="linear", n_samples=50, seed=42)

    print("Running baseline model...")
    assert "MAE" in metrics
    assert metrics["MAE"] >= 0
    assert "MSE" in metrics
    assert metrics["MSE"] >= 0

    print("Smoke test passed. Metrics:", metrics)
    return metrics


if __name__ == '__main__':
    smoke_test()
