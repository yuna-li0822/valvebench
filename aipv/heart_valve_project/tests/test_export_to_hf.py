import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import export_to_hf  # noqa: E402


def test_export_csv_writes_expected_columns(tmp_path):
    output_path = tmp_path / "export.csv"
    result_path = export_to_hf.export_csv(n_samples=10, seed=1, output_path=output_path)
    assert result_path == output_path
    assert output_path.exists()

    import pandas as pd
    df = pd.read_csv(output_path)
    assert len(df) == 10
    assert "Valve_t0" in df.columns
    assert "Valve_t12" in df.columns


def test_push_to_hub_raises_clearly_without_huggingface_hub(tmp_path, monkeypatch):
    monkeypatch.setattr(export_to_hf, "_HF_AVAILABLE", False)
    csv_path = tmp_path / "export.csv"
    csv_path.write_text("a,b\n1,2\n")
    with pytest.raises(RuntimeError, match="huggingface_hub"):
        export_to_hf.push_to_hub(csv_path, repo_id="someone/somewhere")
