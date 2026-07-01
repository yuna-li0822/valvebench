import argparse
from pathlib import Path

from .evaluation.generate_leaderboard import generate_leaderboard
from .pipeline import run_pipeline


def build_parser():
    parser = argparse.ArgumentParser(prog="valvebench")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--all", action="store_true", help="Run the full benchmark and regenerate the leaderboard")
    run_parser.add_argument("--model", default="linear", help="Model name to run")
    run_parser.add_argument("--n-samples", type=int, default=80, help="Number of synthetic samples")
    run_parser.add_argument("--seed", type=int, default=7, help="Random seed")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "run" and args.all:
        for model_name in ["linear", "ridge", "mlp", "pinn"]:
            metrics = run_pipeline(model_name=model_name, n_samples=args.n_samples, seed=args.seed)
            print(f"{model_name}: MAE={metrics['MAE']:.4f}, MSE={metrics['MSE']:.4f}")
        generate_leaderboard(n_samples=args.n_samples, seed=args.seed)
        print("Leaderboard regenerated.")
        return 0

    if args.command == "run":
        metrics = run_pipeline(model_name=args.model, n_samples=args.n_samples, seed=args.seed)
        print(metrics)
        return 0

    parser.error("Unsupported command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
