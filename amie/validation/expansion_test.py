"""Expansion study runner — implements spec_expansion.md exactly.

Test set: markets with trades ingested in the expansion whose condition_id is
NOT in data/v1_trade_markets.txt (the 60 sprint markets). One run, 8 cells.

H1  fade smart flow @72h: event |netflow_z| crossing thr, direction
    = -sign(z) x yes_sign, lag-1, hold 72h, costs 5/10 bps.
H1b same on netflow_all_z.
H2  quiet replication: dissonance_z crossing, |4h r| minus hour-matched
    unconditional mean, expect negative.
"""
import numpy as np
import pandas as pd

from amie.common import DATA_DIR
from amie.validation.event_study import (
    NS_H, assign_instruments, boot_se, detect_events, forward_return,
    load_external,
)

THRS_H1 = [2.0, 1.5, 3.0]
THRS_H2 = [2.0, 3.0]
H1_HOLD = 72
H2_HOLD = 4
COST = {"BTCUSDT": 0.0010, "ETHUSDT": 0.0010}
DEFAULT_COST = 0.0005


def detect_abs_events(g, col, thr):
    z = g[col].abs()
    cross = (z >= thr) & (z.shift(1) < thr)
    ev = g[cross.fillna(False)]
    kept, last = [], None
    for _, r in ev.iterrows():
        if last is None or (r["ts"] - last) >= pd.Timedelta(hours=24):
            kept.append(r)
            last = r["ts"]
    return pd.DataFrame(kept)


def hour_baseline(prices, h, lo_ns, hi_ns):
    ts_ns = prices["ts_ns"].to_numpy()
    px = prices["px"].to_numpy()
    m = (ts_ns >= lo_ns) & (ts_ns <= hi_ns)
    idx = np.where(m)[0]
    j = np.searchsorted(ts_ns, ts_ns[idx] + h * NS_H)
    ok = j < len(ts_ns)
    absr = np.abs(px[j[ok]] / px[idx[ok]] - 1)
    hours = pd.to_datetime(ts_ns[idx[ok]], unit="ns", utc=True).hour
    return pd.Series(absr).groupby(np.asarray(hours)).mean()


def sharpe_dd(net_rets, ts_list, years):
    if len(net_rets) < 2 or years <= 0:
        return np.nan, np.nan
    order = np.argsort(ts_list)
    r = np.array(net_rets)[order]
    per_trade_sh = r.mean() / r.std() if r.std() > 0 else np.nan
    ann = per_trade_sh * np.sqrt(len(r) / years) if np.isfinite(per_trade_sh) else np.nan
    eq = np.cumsum(r)
    dd = float(np.max(np.maximum.accumulate(eq) - eq)) if len(eq) else np.nan
    return ann, dd


def main():
    feats = pd.read_parquet(DATA_DIR / "features_all.parquet")
    uni = pd.read_parquet(DATA_DIR / "universe.parquet")
    v1 = set((DATA_DIR / "v1_trade_markets.txt").read_text(encoding="utf-8-sig").split())
    mapping = assign_instruments(uni)
    ext = load_external()

    feats = feats[~feats["market"].isin(v1)]
    n_mkts = feats["market"].nunique()
    span_y = (feats["ts"].max() - feats["ts"].min()).days / 365.25
    print(f"expansion set: {n_mkts} markets, {len(feats):,} hourly rows, span {span_y:.2f}y")

    rows = []
    # ---- H1 / H1b ----
    for label, col in (("H1_smart", "netflow_z"), ("H1b_all", "netflow_all_z")):
        if col not in feats.columns:
            print(f"{label}: column {col} missing — features not regenerated?")
            continue
        for thr in THRS_H1:
            x, days, ts_l = [], [], []
            for cid, g in feats.groupby("market"):
                if cid not in mapping:
                    continue
                g = g.sort_values("ts")
                for _, e in detect_abs_events(g, col, thr).iterrows():
                    ticker, ysign = mapping[cid]["instruments"][0]
                    if ticker not in ext:
                        continue
                    r = forward_return(ext[ticker], e["ts"].value, H1_HOLD)
                    if r is None or not np.isfinite(e[col]) or e[col] == 0:
                        continue
                    d = -np.sign(e[col]) * ysign
                    x.append(d * r - COST.get(ticker, DEFAULT_COST))
                    days.append(e["ts"].floor("D").value)
                    ts_l.append(e["ts"].value)
            x, days = np.array(x), np.array(days)
            se = boot_se(x, days, H1_HOLD)
            t = x.mean() / se if len(x) and se and se > 0 else np.nan
            sh, dd = sharpe_dd(x, ts_l, span_y)
            rows.append({"cell": f"{label} thr={thr}", "n": len(x), "testable": len(x) >= 40,
                         "mean_net_bps": x.mean() * 1e4 if len(x) else np.nan,
                         "hit": (x > 0).mean() if len(x) else np.nan,
                         "boot_t": t, "ann_sharpe": sh, "maxDD_bps": dd * 1e4 if np.isfinite(dd) else np.nan})

    # ---- H2 ----
    lo, hi = feats["ts"].min().value, feats["ts"].max().value
    hb_cache = {}
    for thr in THRS_H2:
        x, days = [], []
        for cid, g in feats.groupby("market"):
            if cid not in mapping:
                continue
            g = g.sort_values("ts")
            for _, e in detect_events(g, "dissonance_z", thr).iterrows():
                ticker = mapping[cid]["instruments"][0][0]
                if ticker not in ext:
                    continue
                p = ext[ticker]
                ts_ns = p["ts_ns"].to_numpy()
                i = np.searchsorted(ts_ns, e["ts"].value, side="right")
                if i >= len(p):
                    continue
                j = np.searchsorted(ts_ns, ts_ns[i] + H2_HOLD * NS_H)
                if j >= len(p):
                    continue
                absr = abs(p["px"].iloc[j] / p["px"].iloc[i] - 1)
                if ticker not in hb_cache:
                    hb_cache[ticker] = hour_baseline(p, H2_HOLD, lo, hi)
                hb = hb_cache[ticker]
                hour = pd.Timestamp(ts_ns[i], unit="ns", tz="UTC").hour
                if hour not in hb.index:
                    continue
                x.append(absr - hb.loc[hour])
                days.append(e["ts"].floor("D").value)
        x, days = np.array(x), np.array(days)
        se = boot_se(x, days, H2_HOLD)
        t = x.mean() / se if len(x) and se and se > 0 else np.nan
        rows.append({"cell": f"H2_quiet thr={thr}", "n": len(x), "testable": len(x) >= 40,
                     "mean_net_bps": x.mean() * 1e4 if len(x) else np.nan,
                     "hit": (x < 0).mean() if len(x) else np.nan,
                     "boot_t": t, "ann_sharpe": np.nan, "maxDD_bps": np.nan})

    res = pd.DataFrame(rows)
    res.to_csv(DATA_DIR / "results_expansion.csv", index=False)
    print(res.to_string(index=False, float_format=lambda v: f"{v: .3f}"))
    print(f"-> {DATA_DIR / 'results_expansion.csv'}")


if __name__ == "__main__":
    main()
