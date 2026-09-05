from pathlib import Path
from src.generate_stream import generate


def test_generate_returns_dataframe():
    df = generate()
    assert len(df) > 0
    assert "timestamp" in df.columns
    assert "label" in df.columns


def test_generate_has_fraud_and_benign():
    df = generate()
    assert df["label"].sum() > 0
    assert (df["label"] == 0).sum() > 0


def test_generate_parquet_output(tmp_path: Path):
    df = generate()
    out = tmp_path / "transactions.parquet"
    df.to_parquet(out, index=False)
    assert out.exists()
