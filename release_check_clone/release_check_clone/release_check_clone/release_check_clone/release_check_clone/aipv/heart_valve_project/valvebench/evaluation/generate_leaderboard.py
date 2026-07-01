from pathlib import Path

try:
    from ...pipeline import run_pipeline
except ImportError:  # pragma: no cover - direct script execution fallback
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from valvebench.pipeline import run_pipeline


def generate_leaderboard(output_path: str | None = None, n_samples: int = 80, seed: int = 7):
    models = ["linear", "ridge", "mlp", "pinn"]
    rows = []
    for model_name in models:
        metrics = run_pipeline(model_name=model_name, n_samples=n_samples, seed=seed)
        rows.append({
            "model": model_name,
            "MAE": float(metrics["MAE"]),
            "MSE": float(metrics["MSE"]),
            "n_samples": int(metrics["n_samples"]),
        })

    lines = [
        "# Leaderboard",
        "",
        "This leaderboard is generated from the canonical ValveBench pipeline for the frozen synthetic dataset version v0.1.",
        "",
        "| model | MAE | MSE | n_samples |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(f"| {row['model']} | {row['MAE']:.4f} | {row['MSE']:.4f} | {row['n_samples']} |")

    output = Path(output_path or Path(__file__).resolve().parents[2] / "leaderboard.md")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


if __name__ == "__main__":
    generate_leaderboard()
