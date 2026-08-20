"""Deribit DVOL hourly history + Binance 1m klines for the vol study."""
import sys
import time

import pandas as pd
import requests

from amie.common import DATA_DIR

OUT = DATA_DIR / "vol"
OUT.mkdir(exist_ok=True)
DAYS = 600


def pull_dvol(cur: str) -> pd.DataFrame:
    end = int(time.time() * 1000)
    start = end - DAYS * 86400_000
    rows = []
    t0 = start
    while t0 < end:
        t1 = min(t0 + 40 * 86400_000, end)
        r = requests.get("https://www.deribit.com/api/v2/public/get_volatility_index_data",
                         params={"currency": cur, "resolution": 3600,
                                 "start_timestamp": t0, "end_timestamp": t1}, timeout=30)
        r.raise_for_status()
        rows.extend(r.json().get("result", {}).get("data") or [])
        t0 = t1 + 1
        time.sleep(0.3)
    df = pd.DataFrame(rows, columns=["ts_ms", "o", "h", "l", "dvol"]).drop_duplicates("ts_ms")
    df["ts"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True)
    return df[["ts", "dvol"]].sort_values("ts").reset_index(drop=True)


def pull_1m(symbol: str) -> pd.DataFrame:
    end = int(time.time() * 1000)
    start = end - DAYS * 86400_000
    rows = []
    while start < end:
        r = requests.get("https://api.binance.com/api/v3/klines",
                         params={"symbol": symbol, "interval": "1m",
                                 "startTime": start, "limit": 1000}, timeout=30)
        r.raise_for_status()
        b = r.json()
        if not b:
            break
        rows.extend((x[0], float(x[4])) for x in b)
        start = b[-1][6] + 1
        time.sleep(0.06)
    df = pd.DataFrame(rows, columns=["ts_ms", "close"])
    df["ts"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True)
    return df[["ts", "close"]]


def main():
    for cur, sym in (("BTC", "BTCUSDT"), ("ETH", "ETHUSDT")):
        p = OUT / f"dvol_{cur}.parquet"
        if not p.exists():
            df = pull_dvol(cur)
            df.to_parquet(p, index=False)
            print(f"DVOL {cur}: {len(df):,} hourly pts {df['ts'].min():%Y-%m-%d} -> {df['ts'].max():%Y-%m-%d}")
        q = OUT / f"m1_{sym}.parquet"
        if not q.exists():
            df = pull_1m(sym)
            df.to_parquet(q, index=False)
            print(f"1m {sym}: {len(df):,} bars")


if __name__ == "__main__":
    main()
