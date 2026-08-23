import argparse

from .evaluation.generate_leaderboard import generate_leaderboard
from .pipeline import run_pipeline


def build_parser():
    parser = argparse.ArgumentParser(prog="valvebench")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--all", action="store_true", help="Run every registered model (ID+OOD, 5 seeds) and regenerate leaderboard.md")
    run_parser.add_argument("--model", default="linear", help="Model name to run (used unless --all is given)")
    run_parser.add_argument("--n-samples", type=int, default=80, help="Number of synthetic samples (single-model run only)")
    run_parser.add_argument("--seed", type=int, default=7, help="Random seed (single-model run only)")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "run" and args.all:
        print("Running every registered model across 5 seeds (ID + OOD)...")
        output = generate_leaderboard()
        print(output.read_text(encoding="utf-8"))
        print(f"Leaderboard regenerated: {output}")
        return 0

    if args.command == "run":
        metrics = run_pipeline(model_name=args.model, n_samples=args.n_samples, seed=args.seed)
        print(metrics)
        return 0

    parser.error("Unsupported command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
