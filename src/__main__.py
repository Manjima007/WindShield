"""CLI entry point: runs generator → features → honest CV → prints metrics table.

Usage:
    python -m src                    # end-to-end run, prints metrics
    python -m src --no-generate      # skip generation if parquet exists
"""
from __future__ import annotations

import argparse
import sys
import time

import pandas as pd

from src.generate_stream import generate
from src.features import load_synthetic_data, make_features
from src.classifier import FraudSpikeClassifier
from src.changepoint import fit_changepoint_context
from src.graph_utils import fit_graph_context


def run(path: str = "data/synthetic/transactions.parquet",
        freq: str = "15min", cv: int = 5, random_state: int = 7,
        no_generate: bool = False) -> dict:
    """Run the honest pipeline and return the CV result dict."""
    import os
    if not no_generate and not os.path.exists(path):
        print("[windshild] parquet not found — running generator ...")
        t0 = time.perf_counter()
        df = generate()
        import pathlib
        pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        print(f"[windshild] wrote {len(df):,} txns ({time.perf_counter() - t0:.1f}s)")

    print(f"[windshild] loading {path} ...")
    t0 = time.perf_counter()
    frame = load_synthetic_data(path)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    print(f"[windshild] loaded {len(frame):,} txns in {time.perf_counter() - t0:.2f}s")

    print(f"[windshild] running honest {cv}-fold CV (freq={freq}) ...")
    t0 = time.perf_counter()
    res = FraudSpikeClassifier.cross_validate_raw(
        frame, cv=cv, freq=freq, random_state=random_state,
    )
    print(f"[windshild] CV done in {time.perf_counter() - t0:.1f}s")
    return res


def _fmt(res: dict) -> str:
    avg = res["avg"]
    cm = res.get("confusion_matrix", [[0, 0], [0, 0]])
    lines = [
        "5-fold stratified CV, auto-tuned threshold (max F2 per fold)",
        "",
        f"               precision    recall       F2     avg_threshold",
        (f"overall        {avg['precision']:.3f}       {avg['recall']:.3f}"
         f"      {avg['f2']:.3f}   {avg.get('avg_threshold', float('nan')):.3f}"),
        "",
        "Confusion matrix (pooled across folds):",
        "              pred_F  pred_T",
        f"  actual_F    {cm[0][0]:5d}  {cm[0][1]:5d}  -> false-positive cost",
        f"  actual_T    {cm[1][0]:5d}  {cm[1][1]:5d}",
        "",
    ]
    if avg.get("auroc") == avg.get("auroc"):  # not-nan
        lines.append(f"AUROC (per-fold mean): {avg['auroc']:.3f}")
    else:
        lines.append("AUROC (per-fold mean): n/a (unreliable at this prevalence)")
    if avg.get("auprc") == avg.get("auprc"):
        lines.append(f"AUPRC (per-fold mean): {avg['auprc']:.3f}")
    else:
        lines.append("AUPRC (per-fold mean): n/a (unreliable at this prevalence)")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m src",
                                description="WindShield honest fraud-spike CV")
    p.add_argument("--path", default="data/synthetic/transactions.parquet")
    p.add_argument("--freq", default="15min")
    p.add_argument("--cv", type=int, default=5)
    p.add_argument("--random-state", type=int, default=7)
    p.add_argument("--no-generate", action="store_true",
                   help="fail instead of generating if parquet is missing")
    p.add_argument("--json", action="store_true",
                   help="print the full result dict as JSON to stdout")
    args = p.parse_args(argv)

    res = run(path=args.path, freq=args.freq, cv=args.cv,
              random_state=args.random_state, no_generate=args.no_generate)

    if args.json:
        import json
        print(json.dumps(res, indent=2, default=str))
    else:
        print(_fmt(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
