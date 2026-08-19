"""LENS B — dissonance RESOLUTION events + directional variants. EXPLORATORY.

Event: dissonance_z crosses DOWN through thr_lo after having spiked >= thr_hi
within the past K hours ("the argument ends"). Direction = flow sign in the
resolution hour(s) x yes_sign of the mapped instrument. Lag-1 entry, all-in
round-trip costs (5 bps ETF/stock, 10 bps crypto).

Explore split: first 60% of feature period (before ~2026-05-20).
Holdout: after boundary + 3d embargo — run ONCE with --holdout on best config.

Usage:  python explore/lens_resolution.py            # in-sample sweep
        python explore/lens_resolution.py --holdout  # single best config, once
"""
import argparse
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, r"D:\amie")
from amie.validation.event_study import (  # noqa: E402
    NS_H, assign_instruments, forward_return, load_external)

DATA = r"D:\amie\data"
TRAIN_FRAC = 0.6
EMBARGO_D = 3
DEDUP_H = 24
CRYPTO = {"BTCUSDT", "ETHUSDT"}
COST = {t: (0.0010 if t in CRYPTO else 0.0005) for t in
        ["BTCUSDT", "ETHUSDT", "AAPL", "MSFT", "NVDA", "TSLA", "SPY", "GLD",
         "TLT", "UNG", "USO", "UUP"]}


def load_all():
    feats = pd.read_parquet(f"{DATA}\\features_all.parquet")
    t_lo, t_hi = feats["ts"].min(), feats["ts"].max()
    boundary = t_lo + (t_hi - t_lo) * TRAIN_FRAC
    uni = pd.read_parquet(f"{DATA}\\universe.parquet")
    mapping = assign_instruments(uni)
    ext = load_external()
    return feats, boundary, mapping, ext


def hourly_flow(markets):
    """Per-market hourly signed USDC flow (toward YES), all + smart wallets.

    Trade at time t is bucketed to ceil(t, 1h): bucket ts holds trades in
    (ts-1h, ts] — strictly no future info relative to a feature bar at ts.
    """
    smart = set(pd.read_parquet(f"{DATA}\\wallets_scored.parquet")
                .query("smart_flag")["wallet"])
    out = {}
    for m in markets:
        try:
            d = pd.read_parquet(f"{DATA}\\trades\\{m}.parquet")
        except (FileNotFoundError, OSError):
            continue
        sgn = np.where(((d["side"] == "BUY") & (d["outcome"] == "Yes")) |
                       ((d["side"] == "SELL") & (d["outcome"] == "No")), 1.0, -1.0)
        d = d.assign(signed=sgn * d["size_usdc"],
                     bucket=d["ts"].dt.ceil("h"))
        allf = d.groupby("bucket")["signed"].sum()
        smf = d[d["wallet"].isin(smart)].groupby("bucket")["signed"].sum()
        out[m] = (allf, smf)
    return out


def resolution_events(g, thr_hi, thr_lo, k_h):
    """Down-cross of thr_lo with a >=thr_hi spike within past k_h hours."""
    g = g.sort_values("ts").reset_index(drop=True)
    z = g["dissonance_z"]
    cross_dn = (z < thr_lo) & (z.shift(1) >= thr_lo)
    spike_ts = g.loc[z >= thr_hi, "ts"].to_numpy()
    kept, last = [], None
    for i in np.where(cross_dn.fillna(False))[0]:
        t = g.at[i, "ts"]
        lo = t - pd.Timedelta(hours=k_h)
        if not ((spike_ts >= lo) & (spike_ts < t)).any():
            continue
        if last is not None and (t - last) < pd.Timedelta(hours=DEDUP_H):
            continue
        kept.append(g.iloc[i])
        last = t
    return pd.DataFrame(kept)


def direction(e, flows, m, src, dirw):
    """Signed direction BEFORE yes_sign. 0 -> skip event."""
    if src == "nf_sign":
        v = float(e.get("netflow_sign", 0) or 0)
        return v if np.isfinite(v) else 0.0
    if src == "centroid":
        v = float(e.get("centroid_signed_z", np.nan))
        return np.sign(v) if np.isfinite(v) and v != 0 else 0.0
    if m not in flows:
        return 0.0
    f = flows[m][0] if src == "flow_all" else flows[m][1]
    t = e["ts"]
    idx = [t - pd.Timedelta(hours=i) for i in range(dirw)]
    s = float(f.reindex(idx).fillna(0).sum())
    return np.sign(s) if s != 0 else 0.0


def run_config(feats, mapping, ext, flows, cfg, years):
    rows = []
    for m, g in feats.groupby("market"):
        if m not in mapping:
            continue
        ev = resolution_events(g, cfg["thr_hi"], cfg["thr_lo"], cfg["k"])
        for _, e in ev.iterrows():
            for ticker, ysign in mapping[m]["instruments"][:1]:
                if ticker not in ext:
                    continue
                d = direction(e, flows, m, cfg["src"], cfg["dirw"]) * ysign
                if d == 0:
                    continue
                r = forward_return(ext[ticker], e["ts"].value, cfg["h"], lag_bars=1)
                if r is None:
                    continue
                rows.append((e["ts"].value, np.sign(d) * r - COST[ticker]))
    if not rows:
        return dict(n=0)
    rows.sort()
    x = np.array([r for _, r in rows])
    n, mu, sd = len(x), x.mean(), x.std(ddof=1) if len(x) > 1 else np.nan
    cum = np.cumsum(x)
    dd = float(np.max(np.maximum.accumulate(cum) - cum)) if n else np.nan
    return dict(
        n=n, mean_bps=mu * 1e4, hit=float((x > 0).mean()),
        t=float(mu / (sd / np.sqrt(n))) if sd and sd > 0 else np.nan,
        sharpe=float(mu / sd * np.sqrt(n / years)) if sd and sd > 0 else np.nan,
        maxdd=dd)


def sweep(configs, holdout=False):
    feats, boundary, mapping, ext = load_all()
    if holdout:
        sub = feats[feats["ts"] >= boundary + pd.Timedelta(days=EMBARGO_D)]
    else:
        sub = feats[feats["ts"] < boundary]
    years = (sub["ts"].max() - sub["ts"].min()).total_seconds() / (365.25 * 86400)
    flows = hourly_flow(sorted(sub["market"].unique()))
    print(f"split={'HOLDOUT' if holdout else 'IN-SAMPLE'}  rows={len(sub):,}  "
          f"{sub['ts'].min():%Y-%m-%d} -> {sub['ts'].max():%Y-%m-%d}  years={years:.3f}")
    out = []
    for cfg in configs:
        res = run_config(sub, mapping, ext, flows, cfg, years)
        out.append({**cfg, **res})
        print("  ", {**cfg, **{k: (round(v, 3) if isinstance(v, float) else v)
                               for k, v in res.items()}})
    df = pd.DataFrame(out)
    print(df.to_string(index=False, float_format=lambda v: f"{v: .3f}"))
    return df


# ---- config lists ----------------------------------------------------------
def stage1():
    cfgs = []
    for src in ["nf_sign", "flow_all", "flow_smart", "centroid"]:
        for h in [4, 24, 72]:
            cfgs.append(dict(src=src, thr_hi=2.0, thr_lo=1.0, k=12, dirw=1, h=h))
    return cfgs


def stage2(src, h_list):
    cfgs = []
    for k in [6, 24]:
        for h in h_list:
            cfgs.append(dict(src=src, thr_hi=2.0, thr_lo=1.0, k=k, dirw=1, h=h))
    for h in h_list:
        cfgs.append(dict(src=src, thr_hi=2.0, thr_lo=0.5, k=12, dirw=1, h=h))
        cfgs.append(dict(src=src, thr_hi=2.0, thr_lo=1.0, k=12, dirw=3, h=h))
        cfgs.append(dict(src=src, thr_hi=1.5, thr_lo=1.0, k=12, dirw=1, h=h))
    return cfgs


def stage3():
    return [
        dict(src="flow_all", thr_hi=2.0, thr_lo=1.0, k=12, dirw=3, h=4),
        dict(src="flow_smart", thr_hi=2.0, thr_lo=1.0, k=12, dirw=3, h=4),
        dict(src="flow_smart", thr_hi=2.0, thr_lo=1.0, k=12, dirw=3, h=24),
        dict(src="flow_smart", thr_hi=2.0, thr_lo=1.0, k=12, dirw=3, h=72),
        dict(src="flow_smart", thr_hi=1.5, thr_lo=1.0, k=12, dirw=3, h=24),
    ]


BEST = dict(src="flow_all", thr_hi=2.0, thr_lo=1.0, k=12, dirw=3, h=24)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", action="store_true")
    ap.add_argument("--stage", default="1")
    ap.add_argument("--src", default="flow_all")
    ap.add_argument("--hs", default="4,24,72")
    args = ap.parse_args()
    if args.holdout:
        sweep([BEST], holdout=True)
    elif args.stage == "1":
        sweep(stage1())
    elif args.stage == "3":
        sweep(stage3())
    else:
        sweep(stage2(args.src, [int(v) for v in args.hs.split(",")]))
