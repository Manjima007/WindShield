"""Volume changepoint detection with an optional ruptures acceleration."""
from __future__ import annotations
import numpy as np
import pandas as pd

def volume_series(frame: pd.DataFrame, freq="15min") -> pd.DataFrame:
    x=frame.copy(); x["timestamp"]=pd.to_datetime(x["timestamp"])
    return x.set_index("timestamp").resample(freq).size().rename("volume").to_frame()

def detect_changepoints(series: pd.Series, penalty: float = 3.0) -> list[int]:
    values=np.asarray(series, dtype=float)
    if len(values)<4: return []
    try:
        import ruptures as rpt
        return [int(i) for i in rpt.Pelt(model="rbf").fit(values).predict(pen=penalty)[:-1]]
    except ImportError:
        delta=np.abs(np.diff(values)); threshold=delta.mean()+penalty*delta.std()
        return (np.where(delta > threshold)[0]+1).tolist()

def changepoint_features(frame: pd.DataFrame, freq="15min") -> pd.DataFrame:
    s=volume_series(frame,freq); s["rolling_mean"]=s.volume.shift(1).rolling(8,min_periods=2).mean()
    s["rolling_std"]=s.volume.shift(1).rolling(8,min_periods=2).std().fillna(0)
    s["volume_zscore"]=((s.volume-s.rolling_mean)/(s.rolling_std+1)).clip(-20,20)
    cps=set(detect_changepoints(s.volume)); s["changepoint"]=0
    for i in cps:
        if i < len(s): s.iloc[i, s.columns.get_loc("changepoint")]=1
    return s.reset_index()

def fit_changepoint_context(frame: pd.DataFrame, freq="15min",
                            penalty: float = 3.0) -> dict:
    """Detect changepoints on the supplied (training) frame only.

    Returns a context dict mapping each detected changepoint *timestamp* to 1 so
    the flag survives series truncation: a context built from a train-only series
    can be applied to the full series by timestamp lookup, avoiding the index-
    mismatch bug where integer positions point at different wall-clock windows in
    different-length series. Pass the result to :func:`changepoint_features_pit`.
    """
    s = volume_series(frame, freq)
    cps = detect_changepoints(s.volume, penalty)
    cps_map: dict = {}
    for i in cps:
        if i < len(s):
            cps_map[s.index[i]] = 1
    return {"changepoints": cps_map}


def changepoint_features_pit(frame: pd.DataFrame, freq="15min",
                           context: dict | None = None) -> pd.DataFrame:
    """Changepoint features with point-in-time changepoint flags.

    Rolling stats (rolling_mean, rolling_std, volume_zscore) are backward-looking
    by construction (shift + rolling window) so computing them on the full series
    is safe. The changepoint *flag* is the leak-prone part: when ``context`` is
    provided (from :func:`fit_changepoint_context` fit on train data), the
    context's timestamp-keyed flags are applied to the series by timestamp
    lookup, so they correctly map to the same wall-clock windows regardless of
    series length. When ``context`` is None, changepoints are detected on the
    full series (convenience / demo path).
    """
    s = volume_series(frame, freq)
    s["rolling_mean"] = s.volume.shift(1).rolling(8, min_periods=2).mean()
    s["rolling_std"] = s.volume.shift(1).rolling(8, min_periods=2).std().fillna(0)
    s["volume_zscore"] = ((s.volume - s.rolling_mean) / (s.rolling_std + 1)).clip(-20, 20)
    s["changepoint"] = 0
    if context is not None:
        cps_map = context.get("changepoints", {})
        for ts, val in cps_map.items():
            if ts in s.index:
                s.loc[ts, "changepoint"] = val
    else:
        cps = detect_changepoints(s.volume)
        for i in cps:
            if i < len(s):
                s.iloc[i, s.columns.get_loc("changepoint")] = 1
    return s.reset_index()
