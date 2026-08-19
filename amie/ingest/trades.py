"""Day 2 — trades ingestion, resume-safe.

Data-API /trades per market (condition id), newest-first pagination.
One schema: ts, market, wallet, side, outcome, price, size_usdc.
Writes data/trades/<condition_id>.parquet + data/ingest_log.json.
"""
import json

import pandas as pd

from amie.common import CONFIG, DATA_DIR, get_json

DATA_API = CONFIG["api"]["data"]
OUT = DATA_DIR / "trades"
OUT.mkdir(exist_ok=True)
LOG_PATH = DATA_DIR / "ingest_log.json"


def fetch_trades(condition_id: str) -> pd.DataFrame:
    cfg = CONFIG["ingest"]
    rows = []
    for page in range(cfg["max_trades_pages_per_market"]):
        batch = get_json(
            f"{DATA_API}/trades",
            {
                "market": condition_id,
                "limit": cfg["trades_page_size"],
                "offset": page * cfg["trades_page_size"],
                "takerOnly": "true",
            },
        )
        if not batch:
            break
        rows.extend(batch)
        if len(batch) < cfg["trades_page_size"]:
            break
    df = pd.DataFrame(
        {
            "ts": pd.to_datetime([t.get("timestamp") for t in rows], unit="s", utc=True),
            "market": condition_id,
            "wallet": [t.get("proxyWallet") for t in rows],
            "side": [t.get("side") for t in rows],
            "outcome": [t.get("outcome") for t in rows],
            "price": [float(t.get("price") or 0) for t in rows],
            "size_usdc": [float(t.get("size") or 0) * float(t.get("price") or 0) for t in rows],
        }
    )
    return df.sort_values("ts").reset_index(drop=True)


def main(limit: int | None = None):
    uni = pd.read_parquet(DATA_DIR / "universe.parquet")
    if limit:
        uni = uni.nlargest(limit, "volume_usd")
    log = json.loads(LOG_PATH.read_text()) if LOG_PATH.exists() else {}
    for _, r in uni.iterrows():
        cid = r["condition_id"]
        path = OUT / f"{cid}.parquet"
        if path.exists():
            continue
        df = fetch_trades(cid)
        df.to_parquet(path, index=False)
        log[cid] = {
            "question": r["question"],
            "n_trades": len(df),
            "sum_usdc": round(float(df["size_usdc"].sum()), 2),
            "gamma_volume": r["volume_usd"],
            "n_wallets": int(df["wallet"].nunique()),
        }
        LOG_PATH.write_text(json.dumps(log, indent=2))
        cover = log[cid]["sum_usdc"] / max(r["volume_usd"], 1)
        print(f"  {len(df):>7,} trades | {log[cid]['n_wallets']:>6,} wallets | cover {cover:5.0%} | {r['question'][:60]}")
    print(f"trades: {len(log)} markets logged -> {OUT}")


if __name__ == "__main__":
    import sys

    main(limit=int(sys.argv[1]) if len(sys.argv) > 1 else None)
