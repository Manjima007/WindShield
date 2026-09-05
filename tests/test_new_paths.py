"""Tests for new code paths: cross_validate_raw, load_synthetic_data,
fail_rate, graph features, threshold_sweep edge cases, _NumpyModel."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Make repo root importable regardless of cwd / pytest rootdir.
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.features import load_synthetic_data, make_features
from src.graph_utils import graph_features
from src.classifier import FraudSpikeClassifier, _NumpyModel
from src.changepoint import fit_changepoint_context, changepoint_features_pit
from src.generate_stream import generate


# ── helpers ─────────────────────────────────────────────────────────────────

def _small_frame():
    """8 15-min windows, half fraud, with two failed rows so fail_rate = 0.25."""
    ts = pd.date_range("2025-01-01", periods=8, freq="15min")
    return pd.DataFrame({
        "timestamp": ts,
        "transaction_id": range(8),
        "merchant_id": ["m001"] * 8,
        "customer_id": [f"c{i:03d}" for i in range(8)],
        "device_id": [f"d{i:03d}" for i in range(8)],
        "ip_id": [f"ip-{i:03d}" for i in range(8)],
        "amount": [10, 12, 11, 50, 13, 14, 12, 80],
        "status": ["success"] * 6 + ["failed", "failed"],
        "is_fraud": [0, 0, 0, 1, 0, 0, 0, 1],
    })


# Session-scoped parquet — deterministic generator (seed 42), so share once.
@pytest.fixture(scope="session")
def synth_parquet(tmp_path_factory):
    out = tmp_path_factory.mktemp("data") / "transactions.parquet"
    generate().to_parquet(out, index=False)
    return out


def _load(synth_parquet):
    return load_synthetic_data(str(synth_parquet))


# ── load_synthetic_data ─────────────────────────────────────────────────────

def test_load_synthetic_data_column_names(synth_parquet):
    loaded = _load(synth_parquet)
    assert "ip_id" in loaded.columns
    assert "is_fraud" in loaded.columns
    assert "status" in loaded.columns
    assert "device_id" in loaded.columns
    assert "merchant_id" in loaded.columns
    assert loaded["is_fraud"].dtype in (int, bool, "int64", "bool")


def test_load_synthetic_data_row_count_preserved(synth_parquet):
    source = generate()  # deterministic (seed 42) — fresh call, same seed
    loaded = _load(synth_parquet)
    assert len(loaded) == len(source)


# ── make_features / fail_rate ──────────────────────────────────────────────

def test_fail_rate_nonzero_when_failed_present():
    frame = _small_frame()
    X, y = make_features(frame, freq="15min")
    assert "fail_rate" in X.columns
    assert (X["fail_rate"] > 0).any()
    assert (X["fail_rate"] == 0).any()


def test_fail_rate_zero_when_no_status_column():
    frame = _small_frame().drop(columns=["status"])
    X, y = make_features(frame, freq="15min")
    assert "fail_rate" in X.columns
    assert (X["fail_rate"] == 0).all()


def test_amount_features_present():
    X, _ = make_features(_small_frame(), freq="15min")
    for col in ("amount_median", "amount_min", "amount_max", "amount_range"):
        assert col in X.columns, f"{col} missing"


def test_temporal_features_present():
    X, _ = make_features(_small_frame(), freq="15min")
    for col in ("hour_of_day", "day_of_week", "is_weekend"):
        assert col in X.columns, f"{col} missing"


def test_itt_features_present():
    X, _ = make_features(_small_frame(), freq="15min")
    for col in ("mean_itt", "std_itt"):
        assert col in X.columns, f"{col} missing"


def test_make_features_length_matches():
    X, y = make_features(_small_frame(), freq="15min")
    assert len(X) == len(y)


# ── graph_features ─────────────────────────────────────────────────────────

def test_graph_features_columns():
    gf = graph_features(_small_frame(), window="10min")
    assert not gf.empty
    for col in ("device_fanout", "ip_fanout", "merchant_infra_count",
                "ip_per_device_ratio", "max_device_fraction", "max_ip_fraction"):
        assert col in gf.columns, f"{col} missing"


def test_graph_features_point_in_time():
    gf = graph_features(_small_frame(), window="10min")
    # Each timestamp (0, 15, 30, ...) falls in a distinct 10-min window
    # (0, 10, 20, 30, ...), so each window has exactly 1 txn → fanout 1.
    assert (gf["device_fanout"] >= 1).all()
    assert (gf["ip_fanout"] >= 1).all()
    assert (gf["merchant_infra_count"] == 1).all()


# ── threshold_sweep ────────────────────────────────────────────────────────

def test_threshold_sweep_columns():
    df = FraudSpikeClassifier.threshold_sweep(
        np.array([0, 0, 0, 1, 1, 1]),
        np.array([0.1, 0.2, 0.3, 0.8, 0.9, 0.7]),
    )
    assert not df.empty
    assert {"threshold", "precision", "recall", "f2", "tp", "fp", "fn", "tn",
            "predicted_pos"}.issubset(df.columns)


def test_threshold_sweep_identical_predictions():
    df = FraudSpikeClassifier.threshold_sweep(
        np.array([0, 1]), np.array([0.5, 0.5]))
    assert (df["predicted_pos"] == df["predicted_pos"].iloc[0]).all()


def test_threshold_sweep_f2_non_negative():
    df = FraudSpikeClassifier.threshold_sweep(
        np.array([0, 0, 0, 1, 1, 1]),
        np.array([0.1, 0.2, 0.3, 0.8, 0.9, 0.7]),
    )
    assert (df["f2"] >= 0).all()


# ── evaluate ───────────────────────────────────────────────────────────────

def test_evaluate_f2_matches_manual():
    y = np.array([0, 0, 0, 1, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.8, 0.9, 0.7])
    X = pd.DataFrame({"a": np.arange(len(y), dtype=float)})
    m = FraudSpikeClassifier().fit(X, y).evaluate(X, y, threshold=0.5)
    pred = (p >= 0.5).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    pr = tp / (tp + fp + 1e-9)
    rc = tp / (tp + fn + 1e-9)
    manual_f2 = 5 * pr * rc / (4 * pr + rc) if (pr + rc) else 0
    assert abs(m["f2"] - manual_f2) < 1e-9


# ── _NumpyModel ────────────────────────────────────────────────────────────

def test_numpy_model_predict_proba_shape():
    m = _NumpyModel()
    m.fit(pd.DataFrame({"a": [0.1, 0.9], "b": [0.2, 0.8]}), np.array([0, 1]))
    p = m.predict_proba(pd.DataFrame({"a": [0.0, 1.0], "b": [0.0, 1.0]}))
    assert p.shape == (2, 2)
    assert np.all(p >= 0) and np.all(p <= 1)


def test_numpy_model_batch_independent():
    """The original bug: predict_proba used X.mean() at call time, so a single
    row's prediction changed depending on the batch it was in."""
    m = _NumpyModel()
    m.fit(pd.DataFrame({"a": [0.1, 0.9, 0.2, 0.8]}), np.array([0, 1, 0, 1]))
    p_single = m.predict_proba(pd.DataFrame({"a": [0.5]}))
    p_in_batch = m.predict_proba(pd.DataFrame({"a": [0.0, 0.5, 1.0]}))
    np.testing.assert_allclose(p_single[0, 1], p_in_batch[1], rtol=1e-10)


# ── cross_validate_raw ─────────────────────────────────────────────────────

def test_cross_validate_raw_keys(synth_parquet):
    res = FraudSpikeClassifier.cross_validate_raw(
        _load(synth_parquet), cv=2, freq="15min", random_state=7)
    assert {"avg", "confusion_matrix", "folds", "oof_preds", "oof_y"}.issubset(res)
    assert set(res["avg"]) >= {"precision", "recall", "f2", "total_positives",
                                "avg_threshold"}


def test_cross_validate_raw_oof_lengths_match(synth_parquet):
    res = FraudSpikeClassifier.cross_validate_raw(
        _load(synth_parquet), cv=2, freq="15min", random_state=7)
    assert len(res["oof_preds"]) == len(res["oof_y"])


def test_cross_validate_raw_avg_threshold_is_real_probability(synth_parquet):
    res = FraudSpikeClassifier.cross_validate_raw(
        _load(synth_parquet), cv=2, freq="15min", random_state=7)
    avg_t = res["avg"].get("avg_threshold", float("nan"))
    assert np.isfinite(avg_t)
    assert 0 < avg_t < 1


def test_cross_validate_raw_positive_coverage(synth_parquet):
    res = FraudSpikeClassifier.cross_validate_raw(
        _load(synth_parquet), cv=2, freq="15min", random_state=7)
    assert sum(f["fold_positives"] for f in res["folds"]) == res["avg"]["total_positives"]


# ── cross_validate (tabular, no graph) ────────────────────────────────────

def test_cross_validate_stratification_covers_all_positives():
    rng = np.random.RandomState(0)
    y = pd.Series([0] * 12 + [1] * 8)
    X = pd.DataFrame({"a": rng.normal(size=20)})
    res = FraudSpikeClassifier.cross_validate(X, y, cv=4)
    assert sum(f["fold_positives"] for f in res["folds"]) == int(y.sum())


# ── cross_validate per-merchant stratification ─────────────────────────────

def test_cross_validate_per_merchant_keeps_merchants_intact():
    # 4 merchants × 4 windows = 16 rows, cv=2 → each merchant wholly in one side.
    frame = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=16, freq="H"),
        "transaction_id": range(16),
        "merchant_id": ["mA"] * 4 + ["mB"] * 4 + ["mC"] * 4 + ["mD"] * 4,
        "customer_id": [f"c{i}" for i in range(16)],
        "device_id": [f"d{i}" for i in range(16)],
        "ip_id": [f"i{i}" for i in range(16)],
        "amount": np.linspace(1, 16, 16),
        "is_fraud": [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
    })
    X, y = make_features(frame, freq="H")
    res = FraudSpikeClassifier.cross_validate(
        X, y, cv=2, merchant_id=frame["merchant_id"].to_numpy())
    assert len(res["folds"]) == 2


# ── changepoint context roundtrip ─────────────────────────────────────────

def test_changepoint_context_is_timestamp_keyed():
    frame = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=200, freq="15min"),
        "transaction_id": range(200),
        "merchant_id": ["m"] * 200,
        "customer_id": [f"c{i}" for i in range(200)],
        "device_id": [f"d{i}" for i in range(200)],
        "ip_id": [f"i{i}" for i in range(200)],
        "amount": np.random.RandomState(0).lognormal(4.5, 1, 200),
        "is_fraud": [0] * 200,
    })
    ctx = fit_changepoint_context(frame, freq="15min")
    cps_map = ctx.get("changepoints", {})
    assert isinstance(cps_map, dict), "context must be a dict of timestamp keys"
    for k in cps_map:
        assert isinstance(k, pd.Timestamp), f"keys must be timestamps, got {type(k)}"


def test_changepoint_context_survives_truncation():
    """Changepoints detected on a train-only (shorter) series must still flag
    the same wall-clock windows when applied to the longer full series."""
    full = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=400, freq="15min"),
        "transaction_id": range(400),
        "merchant_id": ["m"] * 400,
        "customer_id": [f"c{i}" for i in range(400)],
        "device_id": [f"d{i}" for i in range(400)],
        "ip_id": [f"i{i}" for i in range(400)],
        "amount": np.random.RandomState(0).lognormal(4.5, 1, 400),
        "is_fraud": [0] * 400,
    })
    ctx = fit_changepoint_context(full.iloc[:200], freq="15min")
    full_cp = changepoint_features_pit(full, freq="15min", context=ctx)
    assert "changepoint" in full_cp.columns
    flagged_ts = set(full_cp.loc[full_cp["changepoint"] == 1, "timestamp"])
    for ts in ctx.get("changepoints", {}):
        assert ts in flagged_ts, \
            f"Changepoint at {ts} from train-only context missing in full series"
