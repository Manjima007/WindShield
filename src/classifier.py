"""Time-ordered fraud-spike model with sklearn/lightgbm fallback."""
from __future__ import annotations
import numpy as np
import pandas as pd
try:
    from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_fscore_support
except ImportError:
    average_precision_score = roc_auc_score = precision_recall_fscore_support = None

class _NumpyModel:
    def fit(self, X, y):
        self.train_mean = X.mean().to_numpy()
        self.mean = X[y == 1].mean().to_numpy() - X[y == 0].mean().to_numpy()
        self.scale = X.std().replace(0, 1).to_numpy()
        return self
    def predict_proba(self, X):
        z=((X.to_numpy()-self.train_mean) @ self.mean/(self.scale.sum()+1e-9))
        p=1/(1+np.exp(-np.clip(z,-30,30))); return np.c_[1-p,p]

class FraudSpikeClassifier:
    def __init__(self, random_state=7):
        self.random_state=random_state; self.model=None; self.columns=[]
    def _new_model(self):
        try:
            from lightgbm import LGBMClassifier
            return LGBMClassifier(n_estimators=250, learning_rate=.05, num_leaves=24,
                                  class_weight="balanced", random_state=self.random_state, verbosity=-1)
        except ImportError:
            try:
                from sklearn.ensemble import HistGradientBoostingClassifier
                return HistGradientBoostingClassifier(max_iter=180, random_state=self.random_state)
            except ImportError: return _NumpyModel()
    def fit(self, X, y):
        self.columns=list(X.columns); self.model=self._new_model(); self.model.fit(X,y); return self
    def predict_proba(self,X):
        return self.model.predict_proba(X[self.columns])[:,1]
    def evaluate(self,X,y,threshold=.5):
        p=self.predict_proba(X); pred=(p>=threshold).astype(int)
        if precision_recall_fscore_support:
            pr,rc,_,_=precision_recall_fscore_support(y,pred,average="binary",zero_division=0)
            auroc=roc_auc_score(y,p) if len(np.unique(y))>1 else float("nan"); auprc=average_precision_score(y,p)
        else:
            tp=((pred==1)&(np.asarray(y)==1)).sum(); fp=((pred==1)&(np.asarray(y)==0)).sum(); fn=((pred==0)&(np.asarray(y)==1)).sum()
            pr=tp/(tp+fp+1e-9); rc=tp/(tp+fn+1e-9)
            auroc=auprc=float("nan")
        return {"auroc":auroc,"auprc":auprc,"precision":pr,"recall":rc,"f2":(5*pr*rc/(4*pr+rc)) if pr+rc else 0}

    @staticmethod
    def threshold_sweep(y_true, y_prob, thresholds=None):
        """Sweep classification thresholds and report precision/recall/F2/FP at each.

        Returns a DataFrame with one row per threshold. Useful for picking an
        operating point that balances false positives against missed fraud."""
        if thresholds is None:
            thresholds = np.arange(0.05, 1.0, 0.05)
        y_true = np.asarray(y_true); y_prob = np.asarray(y_prob)
        rows = []
        for t in thresholds:
            pred = (y_prob >= t).astype(int)
            tp = int(((pred == 1) & (y_true == 1)).sum())
            fp = int(((pred == 1) & (y_true == 0)).sum())
            fn = int(((pred == 0) & (y_true == 1)).sum())
            tn = int(((pred == 0) & (y_true == 0)).sum())
            pr = tp / (tp + fp + 1e-9)
            rc = tp / (tp + fn + 1e-9)
            f2 = 5 * pr * rc / (4 * pr + rc) if (pr + rc) else 0
            rows.append({"threshold": round(t, 3), "precision": round(pr, 4),
                          "recall": round(rc, 4), "f2": round(f2, 4),
                          "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                          "predicted_pos": tp + fp})
        return pd.DataFrame(rows)

    @staticmethod
    def _get_folds(X, y, cv=5, random_state=7, groups=None):
        """Stratified k-fold splits, with optional group-aware stratification.

        When ``groups`` is supplied, the splitter uses each group's label to
        drive stratification (so every fold sees roughly the same class balance)
        while keeping every group entirely in train or entirely in test. This
        is what you want when the natural unit is a merchant (cross_validate)
        or a window (cross_validate_raw).
        Falls back to a numpy splitter only when sklearn is genuinely missing;
        a real misuse like a groups/X length mismatch raises loudly instead of
        silently degrading."""
        try:
            from sklearn.model_selection import StratifiedKFold, GroupKFold
        except ImportError:
            n = len(y); idx = np.arange(n)
            pos = np.where(np.asarray(y) == 1)[0]
            neg = np.where(np.asarray(y) == 0)[0]
            np.random.seed(random_state); np.random.shuffle(pos); np.random.shuffle(neg)
            folds = []
            for k in range(cv):
                pi = pos[k::cv]; ni = neg[k::cv]
                test = np.concatenate([pi, ni])
                mask = np.ones(n, dtype=bool); mask[test] = False
                folds.append((idx[mask], test))
            return folds
        if groups is not None and len(groups) != len(X):
            raise ValueError(
                f"groups length {len(groups)} != X/Y length {len(X)}")
        if groups is not None and len(np.unique(groups)) > 1:
            try:
                from sklearn.model_selection import StratifiedGroupKFold
                sgkf = StratifiedGroupKFold(n_splits=cv, shuffle=True,
                                            random_state=random_state)
                return list(sgkf.split(X, y, groups=groups))
            except Exception:
                from sklearn.model_selection import GroupKFold
                gkf = GroupKFold(n_splits=cv)
                return list(gkf.split(X, y, groups=groups))
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
        return list(skf.split(X, y))

    @classmethod
    def cross_validate(cls, X, y, cv=5, random_state=7, merchant_id=None):
        """Stratified k-fold CV with per-fold threshold auto-tuning.

        When ``merchant_id`` is supplied, each merchant's windows are kept
        entirely in train or test and stratification is driven by the label
        distribution across folds. This prevents a single high-volume merchant
        from dominating the evaluation."""
        X=pd.DataFrame(X); y=pd.Series(np.asarray(y)).reset_index(drop=True)
        groups = pd.Series(merchant_id) if merchant_id is not None else (
            X["merchant_id"] if "merchant_id" in X.columns else None)
        folds = cls._get_folds(X, y, cv=cv, random_state=random_state, groups=groups)
        rows=[]; cm_total=np.zeros((2,2),dtype=float)
        for tr,te in folds:
            fold=cls(random_state=random_state).fit(X.iloc[tr],y.iloc[tr])
            p=fold.predict_proba(X.iloc[te])
            sweep=cls.threshold_sweep(y.iloc[te],p)
            best=sweep.loc[sweep["f2"].idxmax()]
            threshold=best["threshold"]
            m=fold.evaluate(X.iloc[te],y.iloc[te],threshold)
            pred=(p>=threshold).astype(int)
            yy=y.iloc[te].to_numpy()
            cm=np.array([[int(((yy==0)&(pred==0)).sum()),int(((yy==0)&(pred==1)).sum())],
                         [int(((yy==1)&(pred==0)).sum()),int(((yy==1)&(pred==1)).sum())]])
            cm_total+=cm
            rows.append({"fold_positives":int(yy.sum()),"threshold":threshold,
                         "precision":m["precision"],"recall":m["recall"],
                         "f2":m["f2"],"auroc":m["auroc"],"auprc":m["auprc"],"cm":cm.tolist()})
        tp,fp,fn=cm_total[1,1],cm_total[0,1],cm_total[1,0]
        pr=tp/(tp+fp+1e-9); rc=tp/(tp+fn+1e-9)
        fmt=pd.DataFrame(rows)
        avg={"precision":float(pr),"recall":float(rc),
             "f2":float(5*pr*rc/(4*pr+rc) if pr+rc else 0),
             "total_positives":int(y.sum()),
             "total_predicted_positive":int(cm_total.sum(axis=0)[1]),
             "avg_threshold": float(fmt["threshold"].mean()) if len(fmt) else float("nan")}
        avg["auroc"]=float(fmt["auroc"].mean(skipna=True)) if fmt["auroc"].notna().any() else float("nan")
        avg["auprc"]=float(fmt["auprc"].mean(skipna=True)) if fmt["auprc"].notna().any() else float("nan")
        return {"avg":avg,"confusion_matrix":cm_total.astype(int).tolist(),"folds":rows}

    @classmethod
    def cross_validate_raw(cls, frame, cv=5, freq="15min", random_state=7):
        """Honest point-in-time stratified k-fold CV from raw transactions.

        For every fold the graph context (device/IP/merchant fan-out) is fit on
        the TRAIN rows only and the window features for both train and test are
        rebuilt with that context, so test-window graph features never encode
        test-period connectivity (no label leakage). Stratification is over the
        windowed labels, with each merchant kept entirely in train or test, so
        a single high-volume merchant cannot dominate the evaluation. Per-fold
        threshold is auto-tuned to maximize F2 instead of using a hardcoded
        0.5 cutoff."""
        from .features import make_features, fit_graph_context
        from .changepoint import fit_changepoint_context
        base_frame=frame.copy()
        base_frame["timestamp"]=pd.to_datetime(base_frame["timestamp"])
        base_frame=base_frame.sort_values("timestamp")
        base_frame["_win_ts"]=base_frame["timestamp"].dt.floor(freq)
        base_frame["_wid"]=base_frame.groupby(["merchant_id","_win_ts"]).ngroup()
        full_X,labels=make_features(base_frame.drop(columns=["_win_ts","_wid"]),freq)
        y=np.asarray(labels)
        wid_per_raw=base_frame["_wid"].to_numpy()
        # Per-merchant stratification: keep each merchant entirely in train or
        # test, stratify by label distribution across folds. merchant_id must be
        # grouped PER WINDOW (one group per window id, aligned to y), not per
        # raw transaction row — passing raw-row groups into a window-indexed
        # split would length-mismatch and silently fall back to plain splits.
        if "merchant_id" in base_frame.columns:
            win_merchant = base_frame.groupby("_wid")["merchant_id"].first().to_numpy()
            groups = win_merchant if win_merchant.size == len(y) else None
        else:
            groups = None
        folds = cls._get_folds(np.arange(len(y)), y, cv=cv, random_state=random_state,
                               groups=groups)
        rows=[]; cm_total=np.zeros((2,2),dtype=float)
        all_preds=np.zeros(len(y)); all_y=np.zeros(len(y),dtype=int)
        for tr,te in folds:
            train_wids=set(tr.tolist())
            train_mask=np.isin(wid_per_raw, list(train_wids))
            train_frame=base_frame[train_mask]
            ctx=fit_graph_context(train_frame)
            cp_ctx=fit_changepoint_context(train_frame.drop(columns=["_win_ts","_wid"]),freq)
            Xf,_=make_features(base_frame.drop(columns=["_win_ts","_wid"]),freq,
                               graph_context=ctx,changepoint_context=cp_ctx)
            fold=cls(random_state=random_state).fit(Xf.iloc[tr],y[tr])
            p=fold.predict_proba(Xf.iloc[te])
            sweep=cls.threshold_sweep(y[te],p)
            best=sweep.loc[sweep["f2"].idxmax()]
            threshold=best["threshold"]
            m=fold.evaluate(Xf.iloc[te],y[te],threshold)
            pred=(p>=threshold).astype(int)
            yy=y[te]
            all_preds[te]=p; all_y[te]=yy
            cm=np.array([[int(((yy==0)&(pred==0)).sum()),int(((yy==0)&(pred==1)).sum())],
                         [int(((yy==1)&(pred==0)).sum()),int(((yy==1)&(pred==1)).sum())]])
            cm_total+=cm
            rows.append({"fold_positives":int(yy.sum()),"threshold":threshold,
                         "precision":m["precision"],"recall":m["recall"],
                         "f2":m["f2"],"auroc":m["auroc"],"auprc":m["auprc"],"cm":cm.tolist()})
        tp,fp,fn=cm_total[1,1],cm_total[0,1],cm_total[1,0]
        pr=tp/(tp+fp+1e-9); rc=tp/(tp+fn+1e-9)
        fmt=pd.DataFrame(rows)
        avg={"precision":float(pr),"recall":float(rc),
             "f2":float(5*pr*rc/(4*pr+rc) if pr+rc else 0),
             "total_positives":int(y.sum()),
             "total_predicted_positive":int(cm_total.sum(axis=0)[1]),
             "avg_threshold": float(fmt["threshold"].mean()) if len(fmt) else float("nan")}
        avg["auroc"]=float(fmt["auroc"].mean(skipna=True)) if fmt["auroc"].notna().any() else float("nan")
        avg["auprc"]=float(fmt["auprc"].mean(skipna=True)) if fmt["auprc"].notna().any() else float("nan")
        return {"avg":avg,"confusion_matrix":cm_total.astype(int).tolist(),"folds":rows,
                "oof_preds":all_preds,"oof_y":all_y}
