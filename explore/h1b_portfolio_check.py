"""DIAGNOSTIC: portfolio-level economics of the single H1b-follow config
(thr=2.0, h=72h, follow all-wallet flow). ONE config, no sweeping.

Simulates an hourly equity curve: each event opens a position in the mapped
instrument (direction = sign(flow) x yes_sign), capital split equally across
open positions (max 10 concurrent, first-come), held exactly 72h, costs
5 bps ETF / 10 bps crypto round trip. Reports annualized Sharpe, maxDD,
exposure stats. Expansion markets only.
"""
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, r"D:\amie")
from amie.common import DATA_DIR
from amie.validation.event_study import NS_H, assign_instruments, load_external
from amie.validation.expansion_test import detect_abs_events

THR, H, MAXPOS = 2.0, 72, 10
COST = {"BTCUSDT": 0.0010, "ETHUSDT": 0.0010}
DEFAULT_COST = 0.0005


def main():
    feats = pd.read_parquet(DATA_DIR / "features_all.parquet")
    uni = pd.read_parquet(DATA_DIR / "universe.parquet")
    v1 = set((DATA_DIR / "v1_trade_markets.txt").read_text(encoding="utf-8-sig").split())
    feats = feats[~feats["market"].isin(v1)]
    mapping = assign_instruments(uni)
    ext = load_external()

    # collect signals
    sigs = []
    for cid, g in feats.groupby("market"):
        if cid not in mapping:
            continue
        g = g.sort_values("ts")
        for _, e in detect_abs_events(g, "netflow_all_z", THR).iterrows():
            ticker, ysign = mapping[cid]["instruments"][0]
            if ticker not in ext or e["netflow_all_z"] == 0:
                continue
            sigs.append((e["ts"].value, ticker, float(np.sign(e["netflow_all_z"]) * ysign)))
    sigs.sort()
    print(f"signals: {len(sigs)}")

    # hourly price grid per ticker (px at ts)
    px = {t: p.set_index("ts_ns")["px"] for t, p in ext.items()}
    tsn = {t: p["ts_ns"].to_numpy() for t, p in ext.items()}

    # build trades with entry/exit prices
    trades = []
    for t_ns, tick, d in sigs:
        arr = tsn[tick]
        i = np.searchsorted(arr, t_ns, side="right")
        if i >= len(arr):
            continue
        j = np.searchsorted(arr, arr[i] + H * NS_H)
        if j >= len(arr):
            continue
        trades.append({"entry_ns": arr[i], "exit_ns": arr[j], "tick": tick, "d": d,
                       "ret": d * (px[tick].iloc[j] / px[tick].iloc[i] - 1) - COST.get(tick, DEFAULT_COST)})
    tr = pd.DataFrame(trades).sort_values("entry_ns").reset_index(drop=True)

    # first-come max-10 concurrency filter
    open_exits = []
    keep = []
    for _, r in tr.iterrows():
        open_exits = [e for e in open_exits if e > r["entry_ns"]]
        if len(open_exits) < MAXPOS:
            keep.append(True)
            open_exits.append(r["exit_ns"])
        else:
            keep.append(False)
    tr = tr[np.array(keep)]
    print(f"trades taken (max {MAXPOS} concurrent): {len(tr)}")

    # daily P&L series: each trade contributes ret/MAXPOS spread at exit day
    tr["exit_day"] = pd.to_datetime(tr["exit_ns"], unit="ns", utc=True).dt.floor("D")
    daily = tr.groupby("exit_day")["ret"].sum() / MAXPOS
    span_days = (daily.index.max() - daily.index.min()).days + 1
    idx = pd.date_range(daily.index.min(), daily.index.max(), freq="D", tz="UTC")
    daily = daily.reindex(idx).fillna(0.0)
    sh = daily.mean() / daily.std() * np.sqrt(365) if daily.std() > 0 else np.nan
    eq = daily.cumsum()
    dd = float((eq.cummax() - eq).max())
    tot = float(eq.iloc[-1])
    print(f"span: {span_days}d | total return {tot*100:+.2f}% (on gross 1.0) | "
          f"ann Sharpe {sh:.2f} | maxDD {dd*100:.2f}% | avg trade net {tr['ret'].mean()*1e4:+.1f} bps | hit {(tr['ret']>0).mean():.3f}")
    print("\nby year:")
    for y, grp in daily.groupby(daily.index.year):
        s = grp.mean() / grp.std() * np.sqrt(365) if grp.std() > 0 else np.nan
        print(f"  {y}: ret {grp.sum()*100:+.2f}%  Sharpe {s:.2f}")
    print("\nby instrument (trades taken):")
    print(tr.groupby("tick").agg(n=("ret", "size"), mean_bps=("ret", lambda s: s.mean() * 1e4),
                                 hit=("ret", lambda s: (s > 0).mean())).to_string(float_format=lambda v: f"{v: .2f}"))


if __name__ == "__main__":
    main()
