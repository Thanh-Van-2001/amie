"""Days 9-10 (downgraded per spec risk #3) — forward whale-event collector.

Without BigQuery credentials the 12-month backfill is Phase 2; this collector
starts building the forward dataset now. Polls blockchain.info unconfirmed
transactions, keeps BTC txs above the USD threshold, and appends to
data/whale_events.parquet. Cron-able; dedupes by tx hash.
"""
import time
from datetime import datetime, timezone

import pandas as pd
import requests

from amie.common import DATA_DIR

THRESHOLD_USD = 1_000_000
OUT = DATA_DIR / "whale_events.parquet"


def btc_price() -> float:
    r = requests.get("https://api.binance.com/api/v3/ticker/price",
                     params={"symbol": "BTCUSDT"}, timeout=15)
    r.raise_for_status()
    return float(r.json()["price"])


def poll_once() -> pd.DataFrame:
    px = btc_price()
    r = requests.get("https://blockchain.info/unconfirmed-transactions?format=json", timeout=30)
    r.raise_for_status()
    txs = r.json().get("txs", [])
    rows = []
    now = datetime.now(timezone.utc)
    for t in txs:
        btc = sum(o.get("value", 0) for o in t.get("out", [])) / 1e8
        usd = btc * px
        if usd >= THRESHOLD_USD:
            rows.append({"ts": now, "hash": t.get("hash"), "btc": btc, "usd": usd,
                         "n_out": len(t.get("out", []))})
    return pd.DataFrame(rows)


def main(cycles: int = 1, sleep_s: int = 60):
    for i in range(cycles):
        try:
            df = poll_once()
        except requests.RequestException as e:
            print(f"  poll failed: {e}")
            df = pd.DataFrame()
        if not df.empty:
            if OUT.exists():
                prev = pd.read_parquet(OUT)
                df = pd.concat([prev, df], ignore_index=True).drop_duplicates("hash")
            df.to_parquet(OUT, index=False)
            print(f"  +{len(df):,} whale txs on file (>= ${THRESHOLD_USD:,})")
        if i < cycles - 1:
            time.sleep(sleep_s)


if __name__ == "__main__":
    import sys

    main(cycles=int(sys.argv[1]) if len(sys.argv) > 1 else 1)
