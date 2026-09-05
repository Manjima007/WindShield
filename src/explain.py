"""Human-readable explanations with optional SHAP support."""
from __future__ import annotations
import numpy as np

def explain_row(model, row, top_n=5):
    values=np.asarray(row, dtype=float); names=list(row.index)
    try:
        import shap
        explainer=shap.TreeExplainer(model.model); impacts=np.asarray(explainer.shap_values(row.to_frame().T))
        if impacts.ndim>1: impacts=impacts[-1]
        impacts=impacts.ravel()
    except Exception:
        # Rank by standardized signed value; avoid dividing by near-zero std which
        # turns constant features into spurious top drivers.
        vals=np.asarray(values, dtype=float)
        s = np.nanstd(vals)
        impacts = vals / (s + 1e-9) if s > 1e-12 else np.sign(vals)
    order=np.argsort(np.abs(impacts))[::-1][:top_n]
    return [{"feature":names[i],"value":float(values[i]),"impact":float(impacts[i])} for i in order]

def narrative(explanation, risk):
    drivers=", ".join(f"{e['feature']} ({e['value']:.2g})" for e in explanation[:3])
    return f"Risk score {risk:.1%}. Main drivers: {drivers or 'no material drivers identified'}."
