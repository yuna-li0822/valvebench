"""Run the canonical ValveBench pipeline end-to-end."""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from valvebench.pipeline import run_pipeline


def main(n_samples=500):
    print('Running ValveBench canonical pipeline...')
    metrics = run_pipeline(model_name='linear', n_samples=n_samples, seed=42)
    print('Pipeline metrics:', metrics)


if __name__ == '__main__':
    main()
