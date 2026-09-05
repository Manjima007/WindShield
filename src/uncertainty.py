"""Dependency-light split conformal prediction for binary alerts."""
from __future__ import annotations
import numpy as np

class ConformalRisk:
    def __init__(self, alpha=.1): self.alpha=alpha; self.q=0.5
    def fit(self, probabilities, y):
        p=np.asarray(probabilities); y=np.asarray(y)
        scores=np.where(y==1,1-p,p); self.q=float(np.quantile(scores,1-self.alpha,method="higher")); return self
    def predict(self, probabilities):
        p=np.asarray(probabilities); return np.c_[np.maximum(0,p-self.q),np.minimum(1,p+self.q)]
    def route(self, probabilities):
        p=np.asarray(probabilities); intervals=self.predict(p)
        return np.where(intervals[:,0]>.5,"block",np.where(intervals[:,1]<.5,"allow","escalate"))
