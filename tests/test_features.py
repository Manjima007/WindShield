import pandas as pd

from src.features import make_features


def test_make_features_returns_feature_matrix_and_labels():
    frame = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=8, freq="min"),
        "transaction_id": range(8),
        "merchant_id": ["m001"] * 8,
        "customer_id": ["c00001", "c00002", "c00003", "c00004", "c00005", "c00006", "c00007", "c00008"],
        "device_id": ["d0001"] * 8,
        "ip_id": ["ip-0001"] * 8,
        "amount": [10, 12, 11, 50, 13, 14, 12, 80],
        "is_fraud": [0, 0, 0, 1, 0, 0, 0, 1],
    })
    X, y = make_features(frame, freq="15min")
    assert not X.empty
    assert len(X) == len(y)
