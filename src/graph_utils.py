"""Graph features for transaction infrastructure sharing.

These features capture the *narrow-device/narrow-IP-pool* pattern that
distinguishes fraud rings (structuring / card-testing) from benign spikes
(flash sales). They are computed FROM THE WINDOW'S OWN TRANSACTIONS ONLY, so
they are point-in-time by construction and can never leak the label:

* ``device_fanout``  = max distinct customers sharing a single device in window
* ``ip_fanout``      = max distinct customers sharing a single IP in window
* ``merchant_infra_count`` = distinct devices seen in the window
* ``ip_per_device_ratio``  = ratio of max IP fanout to max device fanout
* ``max_device_fraction``  = fraction of window txns using the most popular device
* ``max_ip_fraction``      = fraction of window txns using the most popular IP

A benign flash sale uses roughly one device per customer (fanout ~ 1). A fraud
ring concentrates many distinct customers on a handful of devices/IPs (fanout
well above 1), regardless of whether those devices were ever seen before.
Because the signal lives entirely inside the current window, no training
context is needed and no test-period connectivity can leak in.
"""
from __future__ import annotations
import pandas as pd

def fit_graph_context(frame: pd.DataFrame) -> dict:
    """Retained for API compatibility.

    Within-window graph features need no training context, so this returns an
    empty dict. It exists so callers that pass a context still work.
    """
    return {}

def _window_graph_features(sub: pd.DataFrame) -> pd.Series:
    dev = sub.groupby("device_id")["customer_id"].nunique()
    ip = sub.groupby("ip_id")["customer_id"].nunique()
    total = len(sub)
    return pd.Series({
        "device_fanout": float(dev.max()) if len(dev) else 0.0,
        "ip_fanout": float(ip.max()) if len(ip) else 0.0,
        "merchant_infra_count": int(dev.shape[0]),
        "ip_per_device_ratio": float(ip.max() / (dev.max() + 1e-9)) if len(dev) and len(ip) else 0.0,
        "max_device_fraction": float(dev.max() / (total + 1e-9)) if len(dev) else 0.0,
        "max_ip_fraction": float(ip.max() / (total + 1e-9)) if len(ip) else 0.0,
    })

def graph_features(frame: pd.DataFrame, window: str = "60min",
                   context: dict | None = None) -> pd.DataFrame:
    """Windowed, within-window graph features (point-in-time by construction)."""
    x = frame.copy(); x["timestamp"] = pd.to_datetime(x["timestamp"])
    g = x.groupby(["merchant_id", pd.Grouper(key="timestamp", freq=window)])
    # Apply per (merchant, window); aggregator sees that window's transactions only.
    r = g.apply(_window_graph_features).reset_index()
    # g.apply with a Series-returning function yields columns for grouping keys + result
    return r[["merchant_id", "timestamp", "device_fanout", "ip_fanout",
              "merchant_infra_count", "ip_per_device_ratio",
              "max_device_fraction", "max_ip_fraction"]]
