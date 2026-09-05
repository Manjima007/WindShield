"""
Synthetic transaction stream generator for fraud-spike-sentinel.

Produces a merchant transaction stream with:
- a baseline daily volume pattern (seasonality + noise)
- injected BENIGN spikes (viral/flash-sale: diverse customers, diverse IPs/devices, normal failure rate)
- injected HARD-NEGATIVE benign spikes (flash-sale traffic with a payment-processor hiccup:
  elevated failure rate but genuinely benign — forces the model past fail_rate alone)
- injected FRAUD spikes (structuring/card-testing: many small txns, narrow IP/device pool,
  moderately elevated failure rate that overlaps the benign range)

Output: data/synthetic/transactions.parquet
Columns: timestamp, customer_id, ip, device_id, bin, amount, status, window_id, label
  - label is ONLY for evaluation later. Never feed it to the model as a feature.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import uuid

RNG = np.random.default_rng(42)

START = datetime(2026, 8, 1)
DAYS = 90
WINDOW_MINUTES = 10


def random_ip(pool=None):
    if pool is not None:
        return RNG.choice(pool)
    return f"{RNG.integers(1,255)}.{RNG.integers(0,255)}.{RNG.integers(0,255)}.{RNG.integers(1,255)}"


def random_device(pool=None):
    if pool is not None:
        return RNG.choice(pool)
    return f"dev-{uuid.uuid4().hex[:10]}"


def random_bin():
    return str(RNG.choice([411111, 424242, 510510, 601100, 340000, 555555]))


def baseline_volume(t: datetime) -> float:
    """Transactions per minute, with daily + weekly seasonality."""
    hour = t.hour + t.minute / 60
    daily = 8 + 6 * np.sin((hour - 9) / 24 * 2 * np.pi)  # peak midday
    weekly = 1.2 if t.weekday() < 5 else 0.8
    noise = RNG.normal(0, 1.0)
    return max(0.5, daily * weekly + noise)


def gen_normal_txn(t):
    return dict(
        timestamp=t,
        customer_id=f"cust-{RNG.integers(0, 50000)}",
        ip=random_ip(),
        device_id=random_device(),
        bin=random_bin(),
        amount=round(float(RNG.lognormal(mean=4.5, sigma=1.0)), 2),
        status="success" if RNG.random() > 0.03 else "failed",
        label=0,
    )


def gen_benign_spike(t, n):
    """Flash sale / viral moment: many txns, diverse everyone, normal failure rate."""
    rows = []
    for _ in range(n):
        row = gen_normal_txn(t + timedelta(seconds=int(RNG.integers(0, 600))))
        row["amount"] = round(float(RNG.lognormal(mean=4.0, sigma=0.6)), 2)
        row["label"] = 0
        rows.append(row)
    return rows


def gen_fraud_spike(t, n):
    """Structuring / card-testing: narrow IP+device pool, many small txns.

    Failure rate is deliberately softened and given per-spike variance so its
    distribution overlaps the benign range instead of being a fixed 55% that
    trivially separates classes."""
    ip_pool = [random_ip() for _ in range(RNG.integers(2, 4))]
    device_pool = [random_device() for _ in range(RNG.integers(2, 5))]
    fail_prob = RNG.uniform(0.10, 0.20)
    rows = []
    for _ in range(n):
        rows.append(dict(
            timestamp=t + timedelta(seconds=int(RNG.integers(0, 300))),
            customer_id=f"cust-{uuid.uuid4().hex[:8]}",
            ip=random_ip(ip_pool),
            device_id=random_device(device_pool),
            bin=random_bin(),
            amount=round(float(RNG.uniform(5, 45)), 2),
            status="success" if RNG.random() > fail_prob else "failed",
            label=1,
        ))
    return rows


def gen_benign_spike_with_failures(t, n):
    """Hard negative: legitimate flash-sale traffic on a bad day.

    Takes a benign spike (diverse customers/devices/IPs, normal amounts) and
    raises the failure rate to mimic a payment-processor hiccup. The elevated
    fail_rate is NOT evidence of fraud — the model must fall back on velocity /
    graph signals to disambiguate a high-failure window that is actually benign."""
    rows = []
    fail_prob = RNG.uniform(0.15, 0.35)
    for _ in range(n):
        row = gen_normal_txn(t + timedelta(seconds=int(RNG.integers(0, 600))))
        row["amount"] = round(float(RNG.lognormal(mean=4.0, sigma=0.6)), 2)
        row["status"] = "success" if RNG.random() > fail_prob else "failed"
        row["label"] = 0
        rows.append(row)
    return rows


def generate():
    all_rows = []
    t = START
    end = START + timedelta(days=DAYS)
    spike_schedule = []

    n_spikes = int(DAYS / 1.5)
    spike_times = sorted(RNG.uniform(0, DAYS * 24 * 60, n_spikes))
    for i, minute_offset in enumerate(spike_times):
        if i % 2 == 0:
            kind = "fraud"
        else:
            # ~1 in 3 benign spikes is a hard negative (flash sale + processor hiccup).
            kind = "hard_benign" if RNG.random() < 0.33 else "benign"
        spike_schedule.append((
            START + timedelta(minutes=float(minute_offset)),
            kind,
        ))

    while t < end:
        vol = baseline_volume(t)
        n_txns = RNG.poisson(max(vol, 0.1))
        for _ in range(n_txns):
            all_rows.append(gen_normal_txn(t + timedelta(seconds=int(RNG.integers(0, 60)))))
        t += timedelta(minutes=1)

    for spike_time, kind in spike_schedule:
        n = int(RNG.integers(60, 150))
        if kind == "fraud":
            all_rows.extend(gen_fraud_spike(spike_time, n))
        elif kind == "hard_benign":
            all_rows.extend(gen_benign_spike_with_failures(spike_time, n))
        else:
            all_rows.extend(gen_benign_spike(spike_time, n))

    df = pd.DataFrame(all_rows).sort_values("timestamp").reset_index(drop=True)
    df["window_id"] = df["timestamp"].dt.floor(f"{WINDOW_MINUTES}min")
    return df


if __name__ == "__main__":
    df = generate()
    print(f"Generated {len(df)} transactions, {df['label'].sum()} labeled as part of a fraud spike")
    print(df.groupby("label").size())
    df.to_parquet("data/synthetic/transactions.parquet", index=False)
    print("Saved to data/synthetic/transactions.parquet")