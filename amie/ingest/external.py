"""Day 1 — hourly price history for the mapped LIQUID instruments.

These are the outcome variables of every validation test (per the vision:
the signal must lead futures/equities/commodities, not the Polymarket token).
ETF proxies via yfinance (1h bars, ~2 years max); BTC/ETH via Binance klines.
Writes data/external/<ticker>.parquet with columns ts, close.
"""
import time

import pandas as pd
import requests

from amie.common import DATA_DIR

OUT = DATA_DIR / "external"
OUT.mkdir(exist_ok=True)

ETF_TICKERS = ["SPY", "TLT", "GLD", "USO", "UNG", "UUP", "NVDA", "TSLA", "AAPL", "MSFT"]
CRYPTO_TICKERS = ["BTCUSDT", "ETHUSDT"]


# Uniform convention (audit fix): every row is (ts, px) where px is the bar's
# OPEN price — executable AT ts. Binance klines are close-labeled by default;
# we use open time b[0] + open price b[1] instead, so "first bar strictly
# after event t" is a genuinely future, tradeable price on every instrument.
def pull_yf(ticker: str) -> pd.DataFrame:
    import yfinance as yf

    df = yf.download(ticker, period="730d", interval="1h", progress=False, auto_adjust=True)
    if df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    out = df.reset_index()[["Datetime", "Open"]].rename(columns={"Datetime": "ts", "Open": "px"})
    out["ts"] = pd.to_datetime(out["ts"], utc=True)
    return out


def pull_binance(symbol: str, days: int = 730) -> pd.DataFrame:
    """Hourly klines, paginated 1000 bars per call. Open-labeled, open price."""
    end = int(time.time() * 1000)
    start = end - days * 86400_000
    rows = []
    while start < end:
        r = requests.get(
            "https://api.binance.com/api/v3/klines",
            params={"symbol": symbol, "interval": "1h", "startTime": start, "limit": 1000},
            timeout=30,
        )
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        rows.extend(batch)
        start = batch[-1][6] + 1
        time.sleep(0.15)
    df = pd.DataFrame({"ts": pd.to_datetime([b[0] for b in rows], unit="ms", utc=True),
                       "px": [float(b[1]) for b in rows]})
    return df


def main():
    for t in ETF_TICKERS:
        path = OUT / f"{t}.parquet"
        if path.exists():
            continue
        df = pull_yf(t)
        if df.empty:
            print(f"  EMPTY  {t}")
            continue
        df.to_parquet(path, index=False)
        print(f"  ok  {t}: {len(df):,} hourly bars  {df['ts'].min():%Y-%m-%d} -> {df['ts'].max():%Y-%m-%d}")
    for s in CRYPTO_TICKERS:
        path = OUT / f"{s}.parquet"
        if path.exists():
            continue
        df = pull_binance(s)
        df.to_parquet(path, index=False)
        print(f"  ok  {s}: {len(df):,} hourly bars")
    print(f"external prices -> {OUT}")


if __name__ == "__main__":
    main()
