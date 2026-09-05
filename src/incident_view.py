"""Incident view: drill into a single flagged window.

Provides three building blocks the dashboard composes:
  * ``window_transactions`` — pull a single window's transactions from the raw
    stream so the graph and metrics are scoped to that one incident.
  * ``build_window_graph``  — a pyvis (vis.js) interactive network of that
    window's devices/IPs/customers, color-coded by fan-out so a narrow
    device/IP pool (the fraud-ring signature) is visually obvious.
  * ``incident_card``       — the summary stats (risk, interval, routing,
    top flagged IPs/devices, fan-out, transaction count) for one window.

All of these operate on the window's OWN transactions only, so the values are
point-in-time and match what the classifier actually saw.
"""
from __future__ import annotations
import pandas as pd
import numpy as np


def window_transactions(frame: pd.DataFrame, timestamp, freq: str = "15min") -> pd.DataFrame:
    """Return the transactions whose window (rolling ``freq`` bin) contains
    ``timestamp``. ``timestamp`` may be a pandas Timestamp or str."""
    ts = pd.to_datetime(timestamp)
    x = frame.copy()
    x["_win"] = pd.to_datetime(x["timestamp"]).dt.floor(freq)
    return x[x["_win"] == ts].drop(columns="_win").copy()


def _fanout(sub: pd.DataFrame) -> dict:
    """Per-device / per-IP fan-out: number of distinct customers sharing each."""
    dev = sub.groupby("device_id")["customer_id"].nunique().sort_values(ascending=False)
    ip = sub.groupby("ip_id")["customer_id"].nunique().sort_values(ascending=False)
    return {"devices": dev, "ips": ip}


def build_window_graph(sub: pd.DataFrame, freq: str = "15min", height: int = 460):
    """Build a Plotly network figure for one window, colored by IP/device fan-out.

    Customers are neutral nodes, devices are blue, IPs are amber; fan-out (how
    many distinct customers share a single device/IP, the fraud-ring signature)
    is encoded by node size and by shifting the node to a hot red. Edges are
    drawn in medium grays that stay visible against a light background.

    Returns a ``plotly.graph_objects.Figure`` for ``st.plotly_chart`` — the
    previous pyvis (vis.js) iframe rendered a blank black canvas in real
    browsers (verified pixel-level), so the graph is built with Plotly, which
    draws reliably inside Streamlit and needs no CDN or iframe.
    """
    import networkx as nx
    import plotly.graph_objects as go

    fan = _fanout(sub)
    dev_fan, ip_fan = fan["devices"], fan["ips"]

    customers = list(sub["customer_id"].unique())
    devices   = list(dev_fan.index)
    ips       = list(ip_fan.index)

    g = nx.Graph()
    for c in customers:
        g.add_node(str(c), kind="customer")
    for d in devices:
        g.add_node(str(d), kind="device", fanout=int(dev_fan[d]))
    for ip in ips:
        g.add_node(str(ip), kind="ip", fanout=int(ip_fan[ip]))
    for _, r in sub.iterrows():
        g.add_edge(str(r["customer_id"]), str(r["device_id"]), kind="dev")
        g.add_edge(str(r["customer_id"]), str(r["ip_id"]), kind="ip")

    # Deterministic spring layout (cached by the caller via st.cache_data).
    pos = nx.spring_layout(g, seed=7, weight=None,
                           iterations=60, k=0.9 / max(1.0, len(g) ** 0.35))

    def node_style(node):
        kind = g.nodes[node].get("kind")
        fanout = g.nodes[node].get("fanout", 1)
        if kind == "customer":
            return dict(color="#dbe2ea", line="#5a6b80", size=6,
                        label="", title=f"customer {node}")
        if kind == "device":
            if fanout == 1:
                return dict(color="#2563eb", line="#1d4ed8", size=8,
                            label=str(node)[:10], title=f"{fanout} customer shares device {node}")
            return dict(color="#dc2626", line="#991b1b", size=10 + 3 * min(fanout, 8),
                        label=f"{str(node)[:10]} x{fanout}", title=f"{fanout} customers share device {node}")
        # IP
        if fanout == 1:
            return dict(color="#f59e0b", line="#b45309", size=8,
                        label=str(node)[:14], title=f"{fanout} customer shares IP {node}")
        return dict(color="#eb4432", line="#991b1b", size=10 + 3 * min(fanout, 9),
                    label=f"{str(node)[:14]} x{fanout}", title=f"{fanout} customers share IP {node}")

    # Edge segments (x, y with None separators so each edge is its own line)
    edge_dev_x, edge_dev_y = [], []
    edge_ip_x, edge_ip_y = [], []
    for a, b, data in g.edges(data=True):
        x0, y0 = pos[a]; x1, y1 = pos[b]
        if data.get("kind") == "dev":
            edge_dev_x += [x0, x1, None]; edge_dev_y += [y0, y1, None]
        else:
            edge_ip_x += [x0, x1, None]; edge_ip_y += [y0, y1, None]

    node_x, node_y, node_txt, node_lbl, node_st = [], [], [], [], []
    for n in g.nodes():
        x, y = pos[n]
        st_ = node_style(n)
        node_x.append(x); node_y.append(y)
        node_txt.append(st_["title"])
        node_lbl.append(st_["label"])
        node_st.append(st_)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=edge_dev_x, y=edge_dev_y, mode="lines", hoverinfo="skip",
        line=dict(color="#94a3b8", width=1), showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=edge_ip_x, y=edge_ip_y, mode="lines", hoverinfo="skip",
        line=dict(color="#c7a15d", width=1.2), showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y, mode="markers+text",
        text=[l[:26] for l in node_lbl], textposition="top center",
        hovertext=node_txt, hoverinfo="text",
        marker=dict(
            size=[s["size"] for s in node_st],
            color=[s["color"] for s in node_st],
            line=dict(color=[s["line"] for s in node_st], width=1.4),
        ),
        textfont=dict(size=9),
        hovertemplate="%{hovertext}<extra></extra>",
        showlegend=False,
    ))
    fig.update_layout(
        height=height,
        margin=dict(l=6, r=6, t=6, b=6),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="system-ui, sans-serif"),
        dragmode="pan",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return fig


def _risk_interval(row, conformal=None, fail_rate=0.0):
    """Compute risk + [lo, hi] interval + routing.

    When ``row`` (a feature/score row) is None, risk is derived from the
    window's own failure rate so the card still shows a sensible value.
    """
    if row is not None:
        try:
            risk = float(row.get("risk", 0.5))
        except AttributeError:
            risk = 0.5
        lo = hi = risk
    else:
        risk = float(np.clip(0.35 * fail_rate * 4.0, 0.0, 0.99))
        lo = hi = risk
    route = "escalate"
    if conformal is not None:
        try:
            itv = conformal.predict(np.asarray([[risk]]))[0]
            lo, hi = float(itv[0]), float(itv[1])
            route = str(conformal.route(np.asarray([[risk]]))[0])
        except Exception:
            pass
    else:
        route = "block" if risk > 0.5 else "allow"
    return risk, lo, hi, route


def incident_card(sub: pd.DataFrame, row: pd.Series = None, conformal=None,
                  freq: str = "15min") -> dict:
    """Summary stats for one incident window.

    ``row`` is an optional feature row (indexed by feature name) carrying risk /
    interval / routing; if absent, routing derives from transaction failure rate.
    Returns a dict of display-ready strings and numbers.
    """
    fan = _fanout(sub)
    dev_fan, ip_fan = fan["devices"], fan["ips"]
    total = len(sub)
    # Ground truth must degrade gracefully: an empty or all-NaN is_fraud column
    # (a window with no transactions, or an unlabeled window under some config)
    # must render as "unknown" instead of `int(NaN)` crashing the whole page.
    has_label = "is_fraud" in sub and bool(sub["is_fraud"].notna().any())
    fraud = 0
    if has_label:
        v = float(sub["is_fraud"].max())
        fraud = int(v) if v == v else 0
    elif len(sub):
        fraud = 0
    failed = int((sub["status"] == "failed").sum()) if "status" in sub else 0
    fail_rate = failed / total if total else 0.0

    risk, lo, hi, route = _risk_interval(row, conformal, fail_rate)

    top_ips = ip_fan.head(5)
    top_devs = dev_fan.head(5)
    return {
        "total_txns": total,
        "unique_customers": int(sub["customer_id"].nunique()),
        "unique_devices": int(sub["device_id"].nunique()),
        "unique_ips": int(sub["ip_id"].nunique()),
        "device_fanout": float(dev_fan.max()) if len(dev_fan) else 0.0,
        "ip_fanout": float(ip_fan.max()) if len(ip_fan) else 0.0,
        "top_ip_fraction": float(ip_fan.max() / total) if len(ip_fan) and total else 0.0,
        "top_device_fraction": float(dev_fan.max() / total) if len(dev_fan) and total else 0.0,
        "failed": failed,
        "fail_rate": fail_rate,
        "is_fraud": fraud,
        "label_known": has_label,
        "risk": risk,
        "interval_lo": lo,
        "interval_hi": hi,
        "routing": route,
        "top_ips": [(str(i), int(f)) for i, f in top_ips.items()],
        "top_devices": [(str(d), int(f)) for d, f in top_devs.items()],
    }
