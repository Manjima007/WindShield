import pandas as pd

from src.explain import explain_row, narrative


def test_explain_row_and_narrative():
    row = pd.Series({"transaction_count": 14, "amount_sum": 530.0, "velocity_delta": 3})
    explanation = explain_row(type("Model", (), {"model": None})(), row, top_n=2)
    text = narrative(explanation, 0.84)
    assert explanation
    assert "Risk score" in text
