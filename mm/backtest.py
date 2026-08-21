"""Tape-replay backtest of passive market making on the full-depth tape.

Model. For every taker print we assume we are the resting maker on the other
side with probability `share` (our slice of maker volume in that market),
capped at `clip` USD. The position is held `hold` minutes and closed at the
next print's price -- the same mid proxy used in the markout study. Costs:
geopolitics/macro pays zero fees on both sides, so the only cost is the
half-tick of slippage on exit, charged explicitly.

What this is: an honest replay of realised trades under a stated fill
assumption. What it is not: a queue simulation. We cannot know whether our
order would actually have been at the front of the book, so `share` is the
single parameter that carries that uncertainty -- and we sweep it.
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

DATA = r"D:\amie\data\trades_full"
HOLD_MIN = 5
CLIP = 50.0          # max USD per fill
SLIP_C = 0.10        # exit slippage charged, in cents of price
MAXPOS = 3000.0      # inventory cap, USD net
PMIN, PMAX = 0.15, 0.85  # only quote mid-range prices (literature: makers lose on longshots)


def market_pnl(f, share, hold_min, rng):
    d = pd.read_parquet(f, columns=["ts", "outcome", "side", "price", "size_usdc"])
    if len(d) < 2000:
        return None
    d = d.sort_values("ts").reset_index(drop=True)
    isy = d.outcome.str.strip().str.lower().isin(["yes", "up"]).to_numpy()
    ypx = np.where(isy, d.price.to_numpy(), 1 - d.price.to_numpy())
    t = d.ts.astype("int64").to_numpy() / 1e9
    tb = (d.side.str.upper() == "BUY").to_numpy()
    maker_long = np.where(isy, ~tb, tb)          # taker buys -> maker is short yes
    notional = np.minimum(d.size_usdc.to_numpy(), CLIP)

    band = (ypx >= PMIN) & (ypx <= PMAX)         # longshot filter
    take = (rng.random(len(d)) < share) & band   # did we get this fill
    j = np.searchsorted(t, t + hold_min * 60)
    ok = (j < len(t)) & take
    if ok.sum() < 20:
        return None
    idx = np.where(ok)[0]
    entry, exit_ = ypx[idx], ypx[j[idx]]
    drift = np.where(maker_long[idx], exit_ - entry, entry - exit_)
    shares = notional[idx] / np.maximum(entry, 0.01)
    gross = drift * shares
    cost = (SLIP_C / 100.0) * shares             # exit slippage
    pnl = gross - cost

    # inventory cap: drop fills that would push net exposure past the cap
    signed = np.where(maker_long[idx], notional[idx], -notional[idx])
    net, keep = 0.0, np.ones(len(idx), bool)
    for i, s in enumerate(signed):
        if abs(net + s) > MAXPOS:
            keep[i] = False
            continue
        net += s
        if i % 40 == 39:
            net *= 0.5                            # periodic merge/flatten
    return pd.DataFrame({"ts": d.ts.to_numpy()[idx][keep], "pnl": pnl[keep],
                         "notional": notional[idx][keep]})


def run(share, hold_min=HOLD_MIN, seed=7, label=""):
    rng = np.random.default_rng(seed)
    parts = []
    for f in glob.glob(os.path.join(DATA, "*.parquet")):
        r = market_pnl(f, share, hold_min, rng)
        if r is not None:
            parts.append(r)
    tr = pd.concat(parts).sort_values("ts")
    day = tr.set_index("ts").pnl.resample("1D").sum()
    day = day.reindex(pd.date_range(day.index.min(), day.index.max(), freq="D", tz="UTC")).fillna(0)
    sh = day.mean() / day.std() * np.sqrt(365) if day.std() > 0 else np.nan
    eq = day.cumsum()
    dd = float((eq.cummax() - eq).max())
    win = (day > 0).sum() / max((day != 0).sum(), 1)
    print(f"{label:22} fills {len(tr):>7,}  total ${eq.iloc[-1]:>9,.0f}  "
          f"day ${day.mean():>7,.0f}  Sharpe {sh:>5.2f}  maxDD ${dd:>8,.0f}  "
          f"win-days {win:.0%}  bps/fill {1e4*tr.pnl.sum()/tr.notional.sum():>6.1f}")
    return day


if __name__ == "__main__":
    print(f"Passive MM tape replay | hold {HOLD_MIN}m | clip ${CLIP:.0f} | "
          f"slippage {SLIP_C}c | inventory cap ${MAXPOS:.0f}\n")
    print("--- capture-share sweep (our slice of maker volume) ---")
    for s in (0.01, 0.02, 0.05, 0.10):
        run(s, label=f"share {s:.0%}")
    print("\n--- hold-horizon sweep at 2% share ---")
    for h in (2, 5, 15, 60):
        run(0.02, hold_min=h, label=f"hold {h}m")
    print("\n--- seed stability at 2% share, 5m ---")
    for sd in (1, 2, 3):
        run(0.02, seed=sd, label=f"seed {sd}")
    print("\n--- yearly split at 2% share, 5m ---")
    d = run(0.02, label="all")
    for y, g in d.groupby(d.index.year):
        s = g.mean() / g.std() * np.sqrt(365) if g.std() > 0 else np.nan
        print(f"  {y}: total ${g.sum():>8,.0f}  day ${g.mean():>6,.0f}  Sharpe {s:5.2f}")
