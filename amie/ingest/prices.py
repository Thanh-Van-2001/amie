"""Day 1 — pull price history for every market in the universe.

CLOB prices-history endpoint, hourly fidelity, both YES and NO tokens.
Writes data/prices/<condition_id>.parquet.
"""
import pandas as pd

from amie.common import CONFIG, DATA_DIR, get_json

CLOB = CONFIG["api"]["clob"]
OUT = DATA_DIR / "prices"
OUT.mkdir(exist_ok=True)


def fetch_history(token_id: str) -> pd.DataFrame:
    js = get_json(
        f"{CLOB}/prices-history",
        {"market": token_id, "interval": "max", "fidelity": CONFIG["ingest"]["price_fidelity_minutes"]},
    )
    hist = js.get("history") or []
    df = pd.DataFrame(hist)
    if df.empty:
        return df
    df["ts"] = pd.to_datetime(df["t"], unit="s", utc=True)
    return df[["ts", "p"]]


def main(limit: int | None = None):
    uni = pd.read_parquet(DATA_DIR / "universe.parquet")
    if limit:
        uni = uni.nlargest(limit, "volume_usd")
    ok = skip = 0
    for _, r in uni.iterrows():
        path = OUT / f"{r['condition_id']}.parquet"
        if path.exists():
            skip += 1
            continue
        frames = []
        for side, token in (("yes", r["token_yes"]), ("no", r["token_no"])):
            if not token:
                continue
            df = fetch_history(token)
            if df.empty:
                continue
            df["side"] = side
            frames.append(df)
        if frames:
            pd.concat(frames, ignore_index=True).to_parquet(path, index=False)
            ok += 1
            print(f"  ok  {r['question'][:70]}")
        else:
            print(f"  EMPTY  {r['question'][:70]}")
    print(f"prices: {ok} written, {skip} already present -> {OUT}")


if __name__ == "__main__":
    import sys

    main(limit=int(sys.argv[1]) if len(sys.argv) > 1 else None)
