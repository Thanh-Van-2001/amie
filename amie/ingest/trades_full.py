"""Full-depth trade tape via time-window bisection (scout finding 2026-08-20:
the 10k-offset cap is PER TIME WINDOW — /trades accepts start/end epoch
seconds with a fresh budget per window, and limit=10000 works in one call).

Recursive bisection: query a window with limit=10000; if it comes back full,
split the window and recurse. Requests ~= 2 x total_trades / 10k. Windows
overlap by one second at the boundary; rows are deduped on
(tx, ts, wallet, side, outcome, price, size). Writes data/trades_full/<cid>.parquet.
"""
import sys
import time

import pandas as pd

from amie.common import CONFIG, DATA_DIR, get_json

DATA_API = CONFIG["api"]["data"]
OUT = DATA_DIR / "trades_full"
OUT.mkdir(exist_ok=True)
PAGE = 10_000


def fetch_window(cid: str, t0: int, t1: int) -> list[dict]:
    """Cursor-walk backwards: every call yields up to 10k usable rows
    (requests ~= total/10k, no bisection waste). Rows come newest-first;
    the batch's min timestamp becomes the next window end (1s overlap,
    deduped later)."""
    rows, end = [], t1
    while end > t0:
        try:
            batch = get_json(
                f"{DATA_API}/trades",
                {"market": cid, "limit": PAGE, "takerOnly": "true", "start": t0, "end": end},
            ) or []
        except RuntimeError as e:
            print(f"    window {t0}-{end} failed: {e}")
            break
        rows.extend(batch)
        if len(batch) < PAGE:
            break
        ts_min = min(int(b.get("timestamp") or 0) for b in batch)
        end = ts_min if ts_min < end else end - 1
    return rows


def to_frame(rows: list[dict], cid: str) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "ts": pd.to_datetime([t.get("timestamp") for t in rows], unit="s", utc=True),
            "market": cid,
            "wallet": [t.get("proxyWallet") for t in rows],
            "side": [t.get("side") for t in rows],
            "outcome": [t.get("outcome") for t in rows],
            "price": [float(t.get("price") or 0) for t in rows],
            "size_usdc": [float(t.get("size") or 0) * float(t.get("price") or 0) for t in rows],
            "tx": [t.get("transactionHash") for t in rows],
        }
    )
    df = df.drop_duplicates(["tx", "ts", "wallet", "side", "outcome", "price", "size_usdc"])
    return df.sort_values("ts").reset_index(drop=True)


def main(limit: int | None = None, stride: int = 1, rem: int = 0):
    uni = pd.read_parquet(DATA_DIR / "universe.parquet").drop_duplicates("condition_id")
    have_tape = {f.stem for f in (DATA_DIR / "trades").glob("*.parquet")}
    uni = uni[uni["condition_id"].isin(have_tape)].sort_values("volume_usd", ascending=False)
    if limit:
        uni = uni.head(limit)
    uni = uni.iloc[rem::stride]  # striping for parallel workers
    now = int(time.time())
    done = 0
    for _, r in uni.iterrows():
        cid = r["condition_id"]
        path = OUT / f"{cid}.parquet"
        if path.exists():
            continue
        t0 = int(pd.to_datetime(r["start_date"], utc=True, errors="coerce").timestamp()) - 7 * 86400 \
            if pd.notna(r["start_date"]) else now - 730 * 86400
        t1 = min(now, int(pd.to_datetime(r["end_date"], utc=True, errors="coerce").timestamp()) + 7 * 86400) \
            if pd.notna(r["end_date"]) else now
        rows = fetch_window(cid, t0, t1)
        if not rows:
            print(f"  EMPTY {r['question'][:60]}")
            continue
        df = to_frame(rows, cid)
        df.to_parquet(path, index=False)
        done += 1
        capped = pd.read_parquet(DATA_DIR / "trades" / f"{cid}.parquet")
        print(f"  {len(df):>8,} full vs {len(capped):>7,} capped | {r['question'][:58]}")
    print(f"trades_full: {done} markets -> {OUT}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None,
         int(sys.argv[2]) if len(sys.argv) > 2 else 1,
         int(sys.argv[3]) if len(sys.argv) > 3 else 0)
