"""Forward signal logger — implements spec_forward.md's real-time protocol.

Daily cron job: refresh the trade tape of LIVE markets, rebuild their fields
and features, detect |netflow_all_z| >= 2.0 upward crossings in the last 48h,
and append them to data/forward_signals.csv with the wall-clock log
timestamp. A signal only counts for the forward study if logged_at precedes
the entry bar — structural lookahead immunity.
"""
import csv
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from amie.common import DATA_DIR
from amie.features.extract import extract_market
from amie.features.field import synthesize
from amie.ingest.trades import fetch_trades
from amie.validation.event_study import assign_instruments

THR = 2.0
LOOKBACK_H = 48
OUT = DATA_DIR / "forward_signals.csv"


def refresh_live_market(cid: str) -> bool:
    """Re-pull tape and re-synthesize field for one live market."""
    try:
        trades = fetch_trades(cid)
    except Exception as e:  # network hiccup: keep yesterday's tape
        print(f"  refresh failed {cid[:14]}: {e}")
        return False
    if len(trades) < 100:
        return False
    trades.to_parquet(DATA_DIR / "trades" / f"{cid}.parquet", index=False)
    smart = None
    ws = DATA_DIR / "wallets_scored.parquet"
    if ws.exists():
        w = pd.read_parquet(ws)
        smart = set(w.loc[w["smart_flag"], "wallet"])
    field = synthesize(trades, smart)
    if not field.empty:
        field.to_parquet(DATA_DIR / "fields" / f"{cid}.parquet", index=False)
    return True


def main():
    now = datetime.now(timezone.utc)
    uni = pd.read_parquet(DATA_DIR / "universe.parquet")
    live = uni[~uni["closed"]].drop_duplicates("condition_id")
    mapping = assign_instruments(uni)
    print(f"[{now:%Y-%m-%d %H:%M}Z] forward logger: {len(live)} live markets")

    seen = set()
    if OUT.exists():
        prev = pd.read_csv(OUT)
        seen = set(zip(prev["market"], prev["signal_ts"]))

    new_rows = []
    for cid in live["condition_id"]:
        if cid not in mapping:
            continue
        if not refresh_live_market(cid):
            continue
        feats = extract_market(cid)
        if feats is None or "netflow_all_z" not in feats.columns:
            continue
        g = feats.sort_values("ts")
        z = g["netflow_all_z"].abs()
        cross = (z >= THR) & (z.shift(1) < THR)
        recent = g[cross.fillna(False) & (g["ts"] >= now - pd.Timedelta(hours=LOOKBACK_H))]
        for _, e in recent.iterrows():
            key = (cid, str(e["ts"]))
            if key in seen:
                continue
            ticker, ysign = mapping[cid]["instruments"][0]
            direction = int(np.sign(e["netflow_all_z"]) * ysign)
            new_rows.append({
                "logged_at": now.isoformat(timespec="seconds"),
                "signal_ts": str(e["ts"]),
                "market": cid,
                "netflow_all_z": round(float(e["netflow_all_z"]), 3),
                "ticker": ticker,
                "direction": direction,
            })

    if new_rows:
        write_header = not OUT.exists()
        with open(OUT, "a", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(new_rows[0].keys()))
            if write_header:
                w.writeheader()
            w.writerows(new_rows)
    print(f"[{now:%Y-%m-%d %H:%M}Z] logged {len(new_rows)} new signal(s) -> {OUT}")


if __name__ == "__main__":
    main()
