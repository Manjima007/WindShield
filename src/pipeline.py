"""Shared pipeline helpers — single source of truth for the dashboard, the
precompute script, and the config test matrix.

Both ``prepare_results`` (precompute to disk) and the in-app "Recompute" flow
build the honest point-in-time CV through :func:`build_pipeline`, so every
``(freq, cv)`` config follows the exact same code path. :func:`select_windows`
picks the "most confident fraud window" / "one we missed" only from windows that
genuinely exist in the raw stream and carry a definite fraud/benign label — the
guard that keeps every config, not just the validated default, from crashing on
an empty or unlabeled window.
"""
from __future__ import annotations
import time
import numpy as np
import pandas as pd


def build_pipeline(data: pd.DataFrame, freq: str = "15min", cv: int = 5,
                   random_state: int = 7) -> tuple:
    """Run the honest point-in-time CV plus features, model, and conformal router.

    Returns ``(res, X, y, win_ts, model, conformal, meta)`` — the exact tuple
    the dashboard renders. This is the slow path (~1-2 min) that precomputed
    artifacts on disk shortcut. Every manual Recompute and the prep script go
    through here so results are reproducible and identical in shape.
    """
    from .features import make_features
    from .classifier import FraudSpikeClassifier
    from .uncertainty import ConformalRisk

    t0 = time.time()
    res = FraudSpikeClassifier.cross_validate_raw(data, cv=cv, freq=freq,
                                                  random_state=random_state)
    X, y = make_features(data, freq=freq)
    win_ts = pd.DatetimeIndex(
        sorted(pd.to_datetime(data["timestamp"]).dt.floor(freq).unique())[: len(X)])
    model = FraudSpikeClassifier(random_state=random_state).fit(X, y)
    conformal = ConformalRisk(alpha=0.1)
    conformal.fit(res["oof_preds"], res["oof_y"])
    meta = {
        "freq": freq, "cv": cv, "random_state": random_state,
        "n_windows": int(len(X)), "n_fraud_windows": int(np.asarray(y).sum()),
        "precision": float(res["avg"]["precision"]),
        "recall": float(res["avg"]["recall"]),
        "f2": float(res["avg"]["f2"]),
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    print(f"[pipeline] {freq} x {cv} folds: CV+features {time.time()-t0:.0f}s")
    return res, X, y, win_ts, model, conformal, meta


def select_windows(aligned: pd.DataFrame, data: pd.DataFrame,
                   freq: str = "15min"):
    """Pick (topfraud_win, missed_win) restricted to REAL, labeled windows.

    ``aligned`` maps oof risk/label to a window timestamp. Before trusting any
    timestamp we verify the window actually exists in the raw stream (has at
    least one transaction) AND carries a definite is_fraud label (1 or 0, not
    NaN). This works for every (freq, cv) configuration, not just the default
    15min/5-fold one.

    Returns ``(topfraud_win, missed_win)``; both ``None`` when no labeled fraud
    window exists in this config (caller renders a graceful notice).
    """
    f = data.assign(_w=pd.to_datetime(data["timestamp"]).dt.floor(freq))
    grp = f.groupby("_w")
    has_rows = grp.size()
    labeled = grp["is_fraud"].max().dropna()
    ok = aligned["_win"].isin(has_rows.index) & aligned["_win"].isin(labeled.index)
    cand = aligned[ok]
    pos = cand.loc[cand["_y"] == 1]
    if not len(pos):
        return None, None
    topfraud_win = pos.sort_values("_risk", ascending=False).iloc[0]["_win"]
    missed_win   = pos.sort_values("_risk", ascending=True).iloc[0]["_win"]
    return topfraud_win, missed_win