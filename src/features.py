"""Leakage-safe, point-in-time feature construction."""
from __future__ import annotations
import pandas as pd
from .changepoint import changepoint_features_pit, fit_changepoint_context
from .graph_utils import graph_features, fit_graph_context

def load_ulb_data(path: str) -> pd.DataFrame:
    """Load Kaggle creditcard.csv when supplied, without making it mandatory."""
    raw = pd.read_csv(path)
    label = "Class" if "Class" in raw else "is_fraud"
    out = pd.DataFrame({"timestamp": pd.Timestamp("2025-01-01") +
                        pd.to_timedelta(raw.get("Time", range(len(raw))), unit="s"),
                        "transaction_id": range(len(raw)), "merchant_id": "ulb",
                        "customer_id": raw.index.astype(str), "device_id": "ulb",
                        "ip_id": "ulb", "amount": raw.get("Amount", 0),
                        "is_fraud": raw[label].astype(int)})
    return out

def load_synthetic_data(path: str) -> pd.DataFrame:
    """Load generate_stream.py's parquet and adapt it to make_features' schema.

    Maps the generator's real columns (ip, label) to what the feature builder
    expects (ip_id, is_fraud) and adds constant merchant_id plus a row-index
    transaction_id. Device/IP columns are real here (unlike ULB, where they are
    placeholders), so graph features carry genuine signal on this data.
    """
    raw = pd.read_parquet(path)
    out = pd.DataFrame({
        "timestamp": pd.to_datetime(raw["timestamp"]),
        "transaction_id": range(len(raw)),
        "merchant_id": "synth",
        "customer_id": raw["customer_id"],
        "device_id": raw["device_id"],
        "ip_id": raw["ip"],
        "amount": raw["amount"],
        "status": raw["status"],
        "is_fraud": raw["label"].astype(int),
    })
    return out

def make_features(frame: pd.DataFrame, freq="15min",
                  graph_context: dict | None = None,
                  changepoint_context: dict | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """Build windowed features. Tabular features are point-in-time by
    construction. Graph columns require a ``graph_context`` fit on TRAIN rows
    only (see src.graph_utils.fit_graph_context). Changepoint flags require a
    ``changepoint_context`` fit on TRAIN rows only (see
    src.changepoint.fit_changepoint_context). When either context is omitted,
    it is fit on the full frame (convenience/demo path, not for evaluation)."""
    x=frame.copy(); x["timestamp"]=pd.to_datetime(x["timestamp"]); x=x.sort_values("timestamp")
    grouped=x.groupby(["merchant_id", pd.Grouper(key="timestamp",freq=freq)])
    f=grouped.agg(transaction_count=("transaction_id","count"), amount_sum=("amount","sum"),
                  amount_mean=("amount","mean"), unique_customers=("customer_id","nunique"),
                  unique_devices=("device_id","nunique"), unique_ips=("ip_id","nunique")).reset_index()
    f["velocity_delta"]=f.groupby("merchant_id")["transaction_count"].diff().fillna(0)
    stats=grouped["amount"].std().reset_index(name="amount_std")
    f=f.merge(stats,on=["merchant_id","timestamp"],how="left")
    # Failure rate
    if "status" in x.columns:
        fail_count = grouped.apply(lambda g: (g["status"] == "failed").sum()).reset_index(name="fail_count")
        f = f.merge(fail_count, on=["merchant_id", "timestamp"], how="left")
        f["fail_rate"] = f["fail_count"] / f["transaction_count"]
        f.drop(columns=["fail_count"], inplace=True)
    else:
        f["fail_rate"] = 0.0
    # Amount distribution
    amount_stats = grouped["amount"].agg(
        amount_median="median", amount_min="min", amount_max="max"
    ).reset_index()
    f = f.merge(amount_stats, on=["merchant_id", "timestamp"], how="left")
    f["amount_range"] = f["amount_max"] - f["amount_min"]
    # Concentration ratios
    f["txn_per_customer"] = f["transaction_count"] / f["unique_customers"]
    f["amount_per_customer"] = f["amount_sum"] / f["unique_customers"]
    f["ip_concentration"] = f["unique_ips"] / f["transaction_count"]
    f["device_concentration"] = f["unique_devices"] / f["transaction_count"]
    # Temporal features
    f["hour_of_day"] = f["timestamp"].dt.hour
    f["day_of_week"] = f["timestamp"].dt.dayofweek
    f["is_weekend"] = (f["timestamp"].dt.dayofweek >= 5).astype(int)
    # Inter-transaction time (within each merchant-window, gaps between consecutive txns)
    x2 = x.sort_values(["merchant_id", "timestamp"])
    x2["_gap"] = x2.groupby(["merchant_id", pd.Grouper(key="timestamp", freq=freq)])["timestamp"].diff().dt.total_seconds()
    itt = x2.groupby(["merchant_id", pd.Grouper(key="timestamp", freq=freq)])["_gap"].agg(
        mean_itt="mean", std_itt="std").reset_index()
    itt["mean_itt"] = itt["mean_itt"].fillna(0)
    itt["std_itt"] = itt["std_itt"].fillna(0)
    f = f.merge(itt, on=["merchant_id", "timestamp"], how="left")
    if graph_context is None:
        graph_context=fit_graph_context(x)
    gf=graph_features(x, freq, context=graph_context)
    f=f.merge(gf,on=["merchant_id","timestamp"],how="left")
    cp=changepoint_features_pit(x, freq, context=changepoint_context)
    f=f.merge(cp,on="timestamp",how="left")
    labels=grouped["is_fraud"].max().reset_index(name="is_fraud")
    f=f.sort_values(["merchant_id","timestamp"]).reset_index(drop=True)
    labels=labels.merge(f[["merchant_id","timestamp"]],on=["merchant_id","timestamp"],how="right")["is_fraud"]
    return f.drop(columns=["merchant_id","timestamp"]), labels.astype(int).reset_index(drop=True)
