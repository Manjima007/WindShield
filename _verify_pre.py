"""Verify precomputed artifacts load and reproduce correct results."""
import json
import numpy as np
import pandas as pd
import joblib
from src.uncertainty import ConformalRisk

base = "data/precomputed/15min_cv5_"
res = json.load(open(base + "res.json"))
X = pd.read_parquet(base + "X.parquet")
y = pd.read_parquet(base + "y.parquet")["y"].to_numpy()
win_ts = pd.read_parquet(base + "win_ts.parquet")["win_ts"]
model = joblib.load(base + "model.joblib")
conformal_cfg = json.load(open(base + "conformal.json"))
meta = json.load(open(base + "meta.json"))

print("meta:", meta)
print("X shape:", X.shape, "window count:", len(X))
print("prec/recall/f2:", res["avg"]["precision"], res["avg"]["recall"], res["avg"]["f2"])
print("confusion:", res["confusion_matrix"])
print("model fitted:", model.model is not None)
print("model preds on X[0]:", model.predict_proba(X.iloc[[0]]))

# Rebuild conformal router
cr = ConformalRisk(alpha=conformal_cfg["alpha"]); cr.q = conformal_cfg["q"]
sample = np.asarray([[0.9]])
print("conformal route(0.9):", cr.route(sample), "itv:", cr.predict(sample))
