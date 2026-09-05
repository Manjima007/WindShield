"""Precompute honest CV results to disk so the dashboard loads in ~2s.

Runs the real leak-safe point-in-time 5-fold CV (cross_validate_raw) once per
window frequency, plus the fitted model and conformal router, and persists them
to ``data/precomputed/``. The dashboard reads these artifacts at startup
instead of re-running the multi-minute CV on every cold load.

This is transparent and reproducible, not a hidden/slideshow result: you can
re-run this script at any time to recompute every number the dashboard shows.
Usage:
    python -m src.prepare_results
    python -m src.prepare_results --freq 15min 30min 60min
    python -m src.prepare_results --cv 3
"""
from __future__ import annotations
import os
import argparse
import json
import time
import numpy as np
import pandas as pd

OUT = os.path.join("data", "precomputed")
DEFAULT_FREQS = ["15min", "30min", "60min"]


def serialize(base: str, res: dict, X: pd.DataFrame, y, win_ts,
              model, conformal, meta: dict) -> None:
    """Write one config's artifacts to disk (the loader's exact schema).

    Used both by ``prepare`` and by the dashboard's in-app "Recompute" flow, so
    a config recomputed in the app becomes instant on later loads and survives
    restarts.
    """
    import joblib

    res_flat = {
        "avg": res["avg"],
        "confusion_matrix": res["confusion_matrix"],
        "folds": res["folds"],
        "oof_preds": [float(x) for x in np.asarray(res["oof_preds"])],
        "oof_y": [int(x) for x in np.asarray(res["oof_y"])],
    }
    with open(base + "_res.json", "w") as fh:
        json.dump(res_flat, fh)

    X.to_parquet(base + "_X.parquet", index=False)
    pd.Series(np.asarray(y), name="y").to_frame().to_parquet(
        base + "_y.parquet", index=False)
    pd.Series(pd.to_datetime(win_ts), name="win_ts").to_frame().to_parquet(
        base + "_win_ts.parquet", index=False)

    if getattr(model, "model", None) is not None:
        joblib.dump(model, base + "_model.joblib")

    json.dump({"alpha": conformal.alpha, "q": conformal.q},
              open(base + "_conformal.json", "w"))
    json.dump(meta, open(base + "_meta.json", "w"))


def prepare(freq: str, cv: int = 5, random_state: int = 7) -> dict:
    """Run CV + features + model + conformal for one window size and save files.

    Returns a dict of the artifact paths written.
    """
    from .features import load_synthetic_data
    from .pipeline import build_pipeline

    data = load_synthetic_data("data/synthetic/transactions.parquet")
    res, X, y, win_ts, model, conformal, meta = build_pipeline(
        data, freq=freq, cv=cv, random_state=random_state)

    tag = freq.replace("min", "") + "min_cv" + str(cv)
    base = os.path.join(OUT, tag)
    os.makedirs(OUT, exist_ok=True)
    serialize(base, res, X, y, win_ts, model, conformal, meta)

    print(f"[{freq}] precomputed -> {base}")
    return {
        "res": base + "_res.json", "X": base + "_X.parquet",
        "y": base + "_y.parquet", "win_ts": base + "_win_ts.parquet",
        "model": base + "_model.joblib", "conformal": base + "_conformal.json",
        "meta": base + "_meta.json",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--freq", nargs="*", default=DEFAULT_FREQS)
    ap.add_argument("--cv", type=int, default=5)
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    for freq in args.freq:
        prepare(freq, cv=args.cv)
    print("\nDone. Dashboard will load these instantly.")


if __name__ == "__main__":
    main()
