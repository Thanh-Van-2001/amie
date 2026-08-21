"""Re-pull sports/crypto tapes WITH timestamps, then run the same markout and
backtest measurements used on the geopolitics/macro universe.

The earlier sports/crypto pull dropped the timestamp column, which made the
maker-markout and tape-replay analyses impossible there. This fixes that, so
the two universes can be compared on identical measurements.
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor

D = "https://data-api.polymarket.com"
S = requests.Session()
S.headers["User-Agent"] = "amie/1"
OUT = r"D:\amie\data\tape_sc_ts"
META = r"D:\amie\data\tape_sc\meta.json"
os.makedirs(OUT, exist_ok=True)
CLIP = 50.0
SLIP_C = 0.10
PMIN, PMAX = 0.15, 0.85


def pull(item):
    cid, mm = item
    p = os.path.join(OUT, cid + ".parquet")
    if os.path.exists(p):
        return 1
    rows = []
    for pg in range(21):
        try:
            b = S.get(D + "/trades", params={"market": cid, "limit": 500,
                                             "offset": pg * 500, "takerOnly": "true"},
                      timeout=40).json()
        except Exception:
            break
        if not b:
            break
        rows += b
        if len(b) < 500:
            break
    if len(rows) < 400:
        return 0
    pd.DataFrame({
        "ts": pd.to_datetime([t.get("timestamp") for t in rows], unit="s", utc=True),
        "outcome": [t.get("outcome") for t in rows],
        "side": [t.get("side") for t in rows],
        "price": [float(t.get("price") or 0) for t in rows],
        "size_usdc": [float(t.get("size") or 0) * float(t.get("price") or 0) for t in rows],
    }).sort_values("ts").to_parquet(p, index=False)
    return 1


def measure(files, meta, label, share=0.02, hold=5, seed=7, band=True):
    """Markout and tape-replay P&L on one group of markets."""
    rng = np.random.default_rng(seed)
    mo_per_market, parts = [], []
    for f in files:
        d = pd.read_parquet(f)
        if len(d) < 400:
            continue
        isy = d.outcome.str.strip().str.lower().isin(["yes", "up"]).to_numpy()
        ypx = np.where(isy, d.price.to_numpy(), 1 - d.price.to_numpy())
        t = d.ts.astype("int64").to_numpy() / 1e9
        tb = (d.side.str.upper() == "BUY").to_numpy()
        mlong = np.where(isy, ~tb, tb)
        notional = np.minimum(d.size_usdc.to_numpy(), CLIP)
        j = np.searchsorted(t, t + hold * 60)
        ok = j < len(t)
        if ok.sum() < 60:
            continue
        drift_all = np.where(mlong, ypx[np.minimum(j, len(t) - 1)] - ypx,
                             ypx - ypx[np.minimum(j, len(t) - 1)])
        mo_per_market.append(float(np.mean(drift_all[ok])))

        sel = ok & (rng.random(len(d)) < share)
        if band:
            sel &= (ypx >= PMIN) & (ypx <= PMAX)
        idx = np.where(sel)[0]
        if len(idx) < 8:
            continue
        entry = ypx[idx]
        sh = notional[idx] / np.maximum(entry, 0.01)
        pnl = drift_all[idx] * sh - (SLIP_C / 100.0) * sh
        parts.append(pd.DataFrame({"ts": d.ts.to_numpy()[idx], "pnl": pnl,
                                   "notional": notional[idx]}))
    if not parts:
        print(f"{label}: insufficient data")
        return
    mo = np.array(mo_per_market)
    t_mo = mo.mean() / (mo.std() / np.sqrt(len(mo)))
    tr = pd.concat(parts).sort_values("ts")
    day = tr.set_index("ts").pnl.resample("1D").sum()
    day = day.reindex(pd.date_range(day.index.min(), day.index.max(), freq="D", tz="UTC")).fillna(0)
    sr = day.mean() / day.std() * np.sqrt(365) if day.std() > 0 else np.nan
    eq = day.cumsum()
    dd = float((eq.cummax() - eq).max())
    print(f"{label:26} mkts {len(mo):>4}  markout {mo.mean()*100:+.3f}c (t={t_mo:+.1f})  "
          f"| fills {len(tr):>7,}  bps/fill {1e4*tr.pnl.sum()/tr.notional.sum():>6.1f}  "
          f"Sharpe {sr:>5.2f}  maxDD ${dd:>7,.0f}  total ${eq.iloc[-1]:>8,.0f}")


if __name__ == "__main__":
    meta = json.load(open(META))
    have = {os.path.basename(f)[:-8] for f in glob.glob(r"D:\amie\data\tape_sc\*.parquet")}
    todo = [(c, m) for c, m in meta.items() if c in have]
    todo = todo[:400]
    print(f"re-pulling {len(todo)} sports/crypto markets with timestamps", flush=True)
    with ThreadPoolExecutor(12) as ex:
        got = sum(ex.map(pull, todo))
    print(f"tapes with ts: {got}\n", flush=True)

    cry = {"crypto", "bitcoin", "ethereum"}
    files = glob.glob(os.path.join(OUT, "*.parquet"))
    gc = [f for f in files if meta.get(os.path.basename(f)[:-8], {}).get("tag") in cry]
    gs = [f for f in files if meta.get(os.path.basename(f)[:-8], {}).get("tag") not in cry]
    print("--- maker markout + tape-replay backtest, 15-85c band, 2% share, 5m hold ---")
    measure(gc, meta, "CRYPTO")
    measure(gs, meta, "SPORTS")
    print("\n--- reference: geopolitics / macro (mandated universe) ---")
    measure(glob.glob(r"D:\amie\data\trades_full\*.parquet"), meta, "GEOPOLITICS/MACRO")
