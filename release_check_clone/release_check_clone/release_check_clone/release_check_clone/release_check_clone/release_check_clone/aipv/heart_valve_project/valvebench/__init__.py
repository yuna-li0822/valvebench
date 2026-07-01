from .pipeline import run_pipeline
from .core.registry import MODEL_REGISTRY

__all__ = ["run_pipeline", "MODEL_REGISTRY"]


def main():
    from .cli import main as cli_main
    return cli_main()

