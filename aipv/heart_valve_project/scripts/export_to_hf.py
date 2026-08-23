"""Optional: export VALVEBENCH's synthetic dataset to a Hugging Face
Datasets repo for wider accessibility (Idea 1 spec: "Hugging Face Datasets
mirror").

This script is NOT run automatically by anything in this project, and
nothing here has ever pushed data anywhere. Publishing to a third-party
hosting service is an action with real external consequences -- it
publishes data under somebody's account and is visible to others -- so it
requires a human decision each time, plus credentials this environment
doesn't have (there is no Hugging Face token configured here). The push
step (`push_to_hub`) is real, working code that a maintainer with their own
HF account can run; it was written but deliberately never executed.

Usage (after `pip install huggingface_hub` and either running
`huggingface-cli login` or setting the HF_TOKEN environment variable):

    py -3 aipv/heart_valve_project/scripts/export_to_hf.py \\
        --repo-id <your-username>/valvebench-v0.1 --n-samples 500

Add --dry-run to generate and save the CSV locally without pushing
anything, e.g. to sanity-check the export before deciding to publish.
"""
import argparse
import os
import sys
from pathlib import Path

try:
    from huggingface_hub import HfApi
    _HF_AVAILABLE = True
except Exception:
    HfApi = None
    _HF_AVAILABLE = False

_PROJECT_DIR = Path(__file__).resolve().parents[1]
_AIPV_DIR = _PROJECT_DIR.parent
if str(_AIPV_DIR) not in sys.path:
    sys.path.insert(0, str(_AIPV_DIR))

from valvebench.datasets.synthetic import generate_synthetic_dataset, get_dataset_version  # noqa: E402


def export_csv(n_samples: int, seed: int, output_path: Path) -> Path:
    df = generate_synthetic_dataset(n_samples=n_samples, seed=seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path


def push_to_hub(csv_path: Path, repo_id: str, token: str = None) -> str:
    if not _HF_AVAILABLE:
        raise RuntimeError("huggingface_hub is required for this step: pip install huggingface_hub")
    token = token or os.environ.get("HF_TOKEN")
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
    api.upload_file(
        path_or_fileobj=str(csv_path),
        path_in_repo=csv_path.name,
        repo_id=repo_id,
        repo_type="dataset",
    )
    return f"https://huggingface.co/datasets/{repo_id}"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo-id", required=True, help="e.g. your-username/valvebench-v0.1")
    parser.add_argument("--n-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true", help="Export the CSV locally without pushing to the Hub")
    args = parser.parse_args()

    version = get_dataset_version()
    output_path = _PROJECT_DIR / "data" / "synthetic" / f"valvebench_{version}_hf_export.csv"
    export_csv(args.n_samples, args.seed, output_path)
    print(f"Wrote {output_path}")

    if args.dry_run:
        print("--dry-run set: not pushing to Hugging Face Hub.")
        return

    url = push_to_hub(output_path, args.repo_id)
    print(f"Pushed to {url}")


if __name__ == "__main__":
    main()
