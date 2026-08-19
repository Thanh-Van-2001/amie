"""EXPLORATORY (post-sprint, not pre-registered): is the quiet-after-dissonance
effect a session-timing artifact?

For each unsigned cell, compare event |r| against an unconditional baseline
STRATIFIED BY ENTRY HOUR-OF-DAY (and instrument), instead of the flat split
mean. If the negative effect vanishes under hour matching, it was clustering
of events at low-|r| hours; if it survives, the quiet effect is real.
Uses the full feature period for power — exploratory label applies.
"""
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, r"D:\amie")
from amie.common import DATA_DIR
from amie.validation.event_study import (
    NS_H, assign_instruments, detect_events, load_external,
)

HORIZONS = [4, 24]
CELLS = [("dissonance_z", 2.0), ("dissonance_z", 3.0), ("rhythm_entropy_z", 2.0)]


def entry_exit(prices, t_ns, h):
    ts_ns = prices["ts_ns"].to_numpy()
    i = np.searchsorted(ts_ns, t_ns, side="right")
    if i >= len(prices):
        return None
    j = np.searchsorted(ts_ns, ts_ns[i] + h * NS_H)
    if j >= len(prices):
        return None
    px = prices["px"].to_numpy()
    hour = pd.Timestamp(ts_ns[i], unit="ns", tz="UTC").hour
    return abs(px[j] / px[i] - 1), hour


def hour_baseline(prices, h):
    """mean |r| per entry hour-of-day over the feature period."""
    ts_ns = prices["ts_ns"].to_numpy()
    px = prices["px"].to_numpy()
    j = np.searchsorted(ts_ns, ts_ns + h * NS_H)
    ok = j < len(ts_ns)
    absr = np.abs(px[j[ok]] / px[ok.nonzero()[0]] - 1)
    hours = pd.to_datetime(ts_ns[ok], unit="ns", utc=True).hour
    return pd.Series(absr).groupby(np.asarray(hours)).mean()


def main():
    feats = pd.read_parquet(DATA_DIR / "features_all.parquet")
    uni = pd.read_parquet(DATA_DIR / "universe.parquet")
    mapping = assign_instruments(uni)
    ext = load_external()
    lo, hi = feats["ts"].min().value, feats["ts"].max().value
    ext_w = {}
    for k, p in ext.items():
        m = (p["ts_ns"] >= lo) & (p["ts_ns"] <= hi)
        ext_w[k] = p[m].reset_index(drop=True)

    for col, thr in CELLS:
        for h in HORIZONS:
            flat_x, matched_x, hours_used = [], [], []
            hb_cache = {}
            for cid, g in feats.groupby("market"):
                if cid not in mapping:
                    continue
                ev = detect_events(g.sort_values("ts"), col, thr)
                for _, e in ev.iterrows():
                    ticker = mapping[cid]["instruments"][0][0]
                    if ticker not in ext_w or len(ext_w[ticker]) < 100:
                        continue
                    r = entry_exit(ext_w[ticker], e["ts"].value, h)
                    if r is None:
                        continue
                    absr, hour = r
                    if ticker not in hb_cache:
                        hb_cache[ticker] = hour_baseline(ext_w[ticker], h)
                    hb = hb_cache[ticker]
                    flat = hb.mean()
                    if hour not in hb.index:
                        continue
                    flat_x.append(absr - flat)
                    matched_x.append(absr - hb.loc[hour])
                    hours_used.append(hour)
            fx, mx = np.array(flat_x), np.array(matched_x)
            if len(fx) < 20:
                print(f"{col} z{thr:.0f} h{h}: n={len(fx)} too small")
                continue
            tf = fx.mean() / (fx.std() / np.sqrt(len(fx)))
            tm = mx.mean() / (mx.std() / np.sqrt(len(mx)))
            top_hours = pd.Series(hours_used).value_counts().head(4).to_dict()
            print(f"{col} z{thr:.0f} h{h}: n={len(fx)}  flat mean={fx.mean():+.4f} (naive t={tf:+.2f})  "
                  f"hour-matched mean={mx.mean():+.4f} (naive t={tm:+.2f})  event hours(UTC)={top_hours}")


if __name__ == "__main__":
    main()
