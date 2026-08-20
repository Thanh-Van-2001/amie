"""DIAGNOSTIC (not a strategy tune): is the H1b 72h flow-continuation effect
just instrument drift x yes_sign?

For every H1b event (|netflow_all_z| >= thr on expansion markets), the
follow-flow return is decomposed against a DRIFT BASELINE: direction x the
instrument's mean 72h forward return over the same window. If the excess
over drift is ~0, continuation is a drift artifact and dies here. Also
reports per-instrument attribution and a direction-shuffled null.
"""
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, r"D:\amie")
from amie.common import DATA_DIR
from amie.validation.event_study import (
    NS_H, assign_instruments, boot_se, forward_return, load_external,
)
from amie.validation.expansion_test import detect_abs_events

RNG = np.random.default_rng(11)
THR = 2.0
H = 72


def main():
    feats = pd.read_parquet(DATA_DIR / "features_all.parquet")
    uni = pd.read_parquet(DATA_DIR / "universe.parquet")
    v1 = set((DATA_DIR / "v1_trade_markets.txt").read_text(encoding="utf-8-sig").split())
    feats = feats[~feats["market"].isin(v1)]
    mapping = assign_instruments(uni)
    ext = load_external()
    lo, hi = feats["ts"].min().value, feats["ts"].max().value

    # per-instrument mean 72h forward return over the expansion window (drift)
    drift = {}
    for tick, p in ext.items():
        ts_ns = p["ts_ns"].to_numpy()
        px = p["px"].to_numpy()
        idx = np.where((ts_ns >= lo) & (ts_ns <= hi))[0]
        j = np.searchsorted(ts_ns, ts_ns[idx] + H * NS_H)
        ok = j < len(ts_ns)
        if ok.sum() > 100:
            drift[tick] = float(np.mean(px[j[ok]] / px[idx[ok]] - 1))

    rows, days = [], []
    for cid, g in feats.groupby("market"):
        if cid not in mapping:
            continue
        g = g.sort_values("ts")
        for _, e in detect_abs_events(g, "netflow_all_z", THR).iterrows():
            ticker, ysign = mapping[cid]["instruments"][0]
            if ticker not in ext or ticker not in drift:
                continue
            r = forward_return(ext[ticker], e["ts"].value, H)
            if r is None or e["netflow_all_z"] == 0:
                continue
            d = np.sign(e["netflow_all_z"]) * ysign  # FOLLOW flow
            rows.append({"ticker": ticker, "d": d, "follow": d * r,
                         "excess": d * r - d * drift[ticker]})
            days.append(e["ts"].floor("D").value)

    df = pd.DataFrame(rows)
    days = np.array(days)
    print(f"events n={len(df)}  (thr={THR}, h={H}h, expansion markets only)")

    fol, exc = df["follow"].to_numpy(), df["excess"].to_numpy()
    se_f, se_e = boot_se(fol, days, H), boot_se(exc, days, H)
    print(f"follow-flow  gross: mean={fol.mean()*1e4:+.1f} bps  t={fol.mean()/se_f:+.2f}")
    print(f"excess-over-drift:  mean={exc.mean()*1e4:+.1f} bps  t={exc.mean()/se_e:+.2f}")

    # direction-shuffled null: same events/instruments, random directions
    nulls = []
    for _ in range(500):
        ds = RNG.choice([-1, 1], size=len(df))
        nulls.append(np.mean(ds * (df["follow"] / df["d"]).to_numpy()))
    nulls = np.array(nulls)
    z = (fol.mean() - nulls.mean()) / nulls.std()
    print(f"shuffled-direction null: mean={nulls.mean()*1e4:+.1f} bps  real z vs null={z:+.2f}")

    print("\nper-instrument attribution (follow-flow gross):")
    att = df.groupby("ticker").agg(n=("follow", "size"), mean_bps=("follow", lambda s: s.mean() * 1e4),
                                   excess_bps=("excess", lambda s: s.mean() * 1e4),
                                   net_long_frac=("d", lambda s: (s > 0).mean()))
    print(att.sort_values("n", ascending=False).to_string(float_format=lambda v: f"{v: .1f}"))


if __name__ == "__main__":
    main()
