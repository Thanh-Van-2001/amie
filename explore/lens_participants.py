"""LENS C — participant-flow signals (exploratory, post-sprint).

Smart-wallet flow from the raw tape (148 PIT-clean smart wallets) -> direction
on the mapped liquid instrument. Three signal families:
  A  smart-net-flow z crossing (signed)
  B  first-mover: smart z high while crowd volume_z low
  C  cross-market breadth: >=k markets agree on (ticker, direction) within 6h

Discipline: explore/tune ONLY in-sample (ts < 2026-05-20). Holdout
(ts >= 2026-05-23) is run ONCE on the single best config via --holdout.
Execution lag-1 (entry at first external bar strictly after the signal hour,
same convention as the pre-registered harness). Costs all-in round trip:
5 bps ETF/stock, 10 bps crypto.

Usage:
  python explore/lens_participants.py            # in-sample sweep (<=30 configs)
  python explore/lens_participants.py --holdout A:2.0:24   # single final run
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, r"D:\amie")
from amie.validation.event_study import (  # noqa: E402
    NS_H, assign_instruments, forward_return, load_external,
)

DATA = r"D:\amie\data"
IS_END = pd.Timestamp("2026-05-20", tz="UTC")       # explore strictly before
HOLD_START = pd.Timestamp("2026-05-23", tz="UTC")   # 3-day embargo
CRYPTO = {"BTCUSDT", "ETHUSDT"}
COST = {t: (0.0010 if t in CRYPTO else 0.0005) for t in
        ["BTCUSDT", "ETHUSDT", "AAPL", "MSFT", "NVDA", "TSLA", "SPY",
         "GLD", "TLT", "UNG", "USO", "UUP"]}
ZWIN, ZMIN = 168, 48      # trailing z window (hours), min periods
DEDUP_H = 24
BREADTH_WIN_H = 6
BREADTH_BASE_THR = 1.5


def build_hourly_flow():
    """Per market: hourly grid over its own tape window with smart/total flow.

    Signed flow in yes-direction units: BUY Yes = +, SELL Yes = -,
    BUY No = -, SELL No = + (USD). Hour labeled by floor (trades in [t, t+1h)
    -> label t), so lag-1 entry at the first external bar > t never sees a
    price older than the last trade used.
    """
    w = pd.read_parquet(os.path.join(DATA, "wallets_scored.parquet"))
    smart = set(w.loc[w["smart_flag"], "wallet"])
    out = {}
    for p in glob.glob(os.path.join(DATA, "trades", "*.parquet")):
        cid = os.path.basename(p)[:-8]
        t = pd.read_parquet(p)
        sgn = np.where(t["side"] == "BUY", 1.0, -1.0) * np.where(
            t["outcome"] == "Yes", 1.0, -1.0)
        t["yflow"] = sgn * t["size_usdc"]
        t["hr"] = t["ts"].dt.floor("h")
        t["is_smart"] = t["wallet"].isin(smart)
        g = t.groupby("hr").agg(
            smart_net=("yflow", lambda s: s[t.loc[s.index, "is_smart"]].sum()),
            total_net=("yflow", "sum"),
            total_vol=("size_usdc", "sum"),
        )
        # dense hourly grid over the tape's own coverage only (tape is
        # row-capped; outside [lo,hi] zero-flow would be an artifact)
        idx = pd.date_range(t["hr"].min(), t["hr"].max(), freq="h", tz="UTC")
        g = g.reindex(idx, fill_value=0.0)
        # causal z: trailing ZWIN hours EXCLUDING current
        for col in ("smart_net", "total_net"):
            mu = g[col].shift(1).rolling(ZWIN, min_periods=ZMIN).mean()
            sd = g[col].shift(1).rolling(ZWIN, min_periods=ZMIN).std()
            g[col + "_z"] = (g[col] - mu) / sd.replace(0.0, np.nan)
        g.index.name = "ts"
        out[cid] = g.reset_index()
    return out


def _dedup(events, key=None):
    """24h dedup, optionally per key (e.g. per ticker+dir for breadth)."""
    last = {}
    kept = []
    for e in sorted(events, key=lambda e: e["ts"]):
        k = key(e) if key else "all"
        if k not in last or (e["ts"] - last[k]) >= pd.Timedelta(hours=DEDUP_H):
            kept.append(e)
            last[k] = e["ts"]
    return kept


def detect_smart_events(flow, mapping, thr, vol_cap=None, feats=None):
    """Family A/B raw events: |smart_net_z| crosses thr upward, nonzero flow,
    direction = sign(actual smart flow) x yes_sign. vol_cap (family B): keep
    only hours where the market's crowd volume_z < vol_cap (features join)."""
    events = []
    for cid, g in flow.items():
        if cid not in mapping:
            continue
        ticker, ysign = mapping[cid]["instruments"][0]
        z = g["smart_net_z"]
        cross = (z.abs() >= thr) & (z.abs().shift(1) < thr) & (g["smart_net"] != 0)
        ev = g[cross.fillna(False)]
        if vol_cap is not None:
            f = feats[feats["market"] == cid][["ts", "volume_z"]]
            ev = ev.merge(f, on="ts", how="left")
            ev = ev[ev["volume_z"].fillna(np.inf) < vol_cap]
        for _, r in ev.iterrows():
            events.append({"ts": r["ts"], "market": cid, "ticker": ticker,
                           "dir": float(np.sign(r["smart_net"]) * ysign)})
    # dedup 24h per market
    return _dedup(events, key=lambda e: e["market"])


def breadth_trades(base_events, k):
    """Family C: trade when >=k distinct markets agree on (ticker, dir)
    within the trailing BREADTH_WIN_H hours. Trade at the k-th event's ts,
    dedup 24h per (ticker, dir)."""
    evs = sorted(base_events, key=lambda e: e["ts"])
    trades = []
    for i, e in enumerate(evs):
        mkts = {e["market"]}
        for j in range(i - 1, -1, -1):
            if (e["ts"] - evs[j]["ts"]) > pd.Timedelta(hours=BREADTH_WIN_H):
                break
            if evs[j]["ticker"] == e["ticker"] and evs[j]["dir"] == e["dir"]:
                mkts.add(evs[j]["market"])
        if len(mkts) >= k:
            trades.append(e)
    return _dedup(trades, key=lambda e: (e["ticker"], e["dir"]))


def score(events, ext, horizon_h, lo, hi):
    """Execute events in [lo, hi): lag-1 forward return, all-in costs."""
    rows = []
    for e in events:
        if not (lo <= e["ts"] < hi):
            continue
        if e["ticker"] not in ext:
            continue
        r = forward_return(ext[e["ticker"]], e["ts"].value, horizon_h, lag_bars=1)
        if r is None:
            continue
        rows.append({"ts": e["ts"], "ticker": e["ticker"],
                     "net": e["dir"] * r - COST[e["ticker"]]})
    return pd.DataFrame(rows)


def metrics(tr, lo, hi):
    if len(tr) == 0:
        return dict(n=0)
    x = tr.sort_values("ts")["net"].to_numpy()
    n = len(x)
    yrs = (hi - lo).days / 365.25
    mean, sd = x.mean(), x.std(ddof=1) if n > 1 else np.nan
    sharpe = mean / sd * np.sqrt(n / yrs) if sd and sd > 0 else np.nan
    cum = np.cumsum(x)
    dd = float(np.max(np.maximum.accumulate(cum) - cum)) if n else np.nan
    t = mean / (sd / np.sqrt(n)) if sd and sd > 0 else np.nan
    return dict(n=n, mean_bps=mean * 1e4, hit=float((x > 0).mean()),
                sharpe=sharpe, maxdd_bps=dd * 1e4, t=t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", default=None,
                    help="FAMILY:THR:HORIZON (e.g. A:2.0:24) — run ONCE")
    args = ap.parse_args()

    uni = pd.read_parquet(os.path.join(DATA, "universe.parquet"))
    mapping = assign_instruments(uni)
    ext = load_external()
    feats = pd.read_parquet(os.path.join(DATA, "features_all.parquet"))
    flow = build_hourly_flow()
    print(f"markets with tape: {len(flow)}, mapped: {len(set(flow) & set(mapping))}")

    is_lo = feats["ts"].min()
    hold_hi = feats["ts"].max()

    def run_cfg(fam, thr, h, lo, hi, vol_cap=None, k=None):
        if fam == "A":
            ev = detect_smart_events(flow, mapping, thr)
        elif fam == "B":
            ev = detect_smart_events(flow, mapping, thr, vol_cap=vol_cap, feats=feats)
        else:  # C
            base = detect_smart_events(flow, mapping, BREADTH_BASE_THR)
            ev = breadth_trades(base, k)
        tr = score(ev, ext, h, lo, hi)
        return metrics(tr, lo, hi)

    if args.holdout:
        fam, mid, h = args.holdout.split(":")
        h = int(h)
        kw = {"k": int(mid)} if fam == "C" else {}
        thr = None if fam == "C" else float(mid)
        print(f"=== HOLDOUT (single run) {fam} {'k' if fam == 'C' else 'thr'}={mid} h={h} "
              f"[{HOLD_START:%Y-%m-%d} .. {hold_hi:%Y-%m-%d}] ===")
        m = run_cfg(fam, thr, h, HOLD_START, hold_hi, **kw)
        print(m)
        return

    # ---------------- in-sample sweep (budget <= 30) ----------------
    rows = []
    for thr in (1.5, 2.0, 3.0):
        for h in (4, 24, 72):
            m = run_cfg("A", thr, h, is_lo, IS_END)
            rows.append({"cfg": f"A thr={thr} h={h}", **m})
    for thr in (1.5, 2.0):
        for vcap in (0.0, 0.5):
            for h in (4, 24):
                m = run_cfg("B", thr, h, is_lo, IS_END, vol_cap=vcap)
                rows.append({"cfg": f"B thr={thr} vcap={vcap} h={h}", **m})
    for k in (2, 3):
        for h in (4, 24, 72):
            m = run_cfg("C", None, h, is_lo, IS_END, k=k)
            rows.append({"cfg": f"C k={k} h={h}", **m})
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 160)
    print(df.to_string(index=False, float_format=lambda v: f"{v: .3f}"))


if __name__ == "__main__":
    main()
