"""Day 1 — build the market universe.

Pulls Polymarket events (live + resolved) from the Gamma API, keeps only
markets that can plausibly LEAD liquid external markets (geopolitics, macro,
weather, earnings — per the vision filter in config.yaml), caps by volume,
and writes data/universe.parquet with one row per market.
"""
import json
from datetime import datetime, timedelta, timezone

import pandas as pd

from amie.common import CONFIG, DATA_DIR, get_json

GAMMA = CONFIG["api"]["gamma"]
U = CONFIG["universe"]


# Gamma event payloads are heavy (~300 KB each), so query by relevant tag with
# small pages instead of paging the whole exchange.
TAG_SLUGS = ["geopolitics", "economy", "crypto", "business", "politics", "world",
             "science", "weather", "finance", "earnings", "ai"]


def fetch_events(closed: bool, max_pages_per_tag: int = 10) -> list[dict]:
    events, seen, limit = [], set(), 25
    cutoff = datetime.now(timezone.utc) - timedelta(days=U["lookback_days"])
    for tag in TAG_SLUGS:
        for page in range(max_pages_per_tag):
            params = {
                "limit": limit,
                "offset": page * limit,
                "closed": str(closed).lower(),
                "order": "volume",
                "ascending": "false",
                "tag_slug": tag,
            }
            if closed:
                # server-side filter: only events that ended inside the lookback
                params["end_date_min"] = cutoff.strftime("%Y-%m-%dT%H:%M:%SZ")
            batch = get_json(f"{GAMMA}/events", params)
            if not batch:
                break
            fresh = [e for e in batch if e.get("id") not in seen]
            seen.update(e.get("id") for e in fresh)
            events.extend(fresh)
            print(f"  tag={tag} page={page + 1}: +{len(fresh)} events (closed={closed})", flush=True)
            # volume-sorted: once a full page is below threshold, next tag
            if all(float(e.get("volume") or 0) < U["min_volume_usd"] for e in batch):
                break
    return events


def relevant(event: dict) -> bool:
    tags = " ".join(t.get("label", "") + " " + t.get("slug", "") for t in event.get("tags") or [])
    text = f"{event.get('title', '')} {tags}".lower()
    if any(k.lower() in text for k in U["exclude_keywords"]):
        return False
    return any(k.lower() in text for k in U["include_keywords"])


def flatten(event: dict, closed: bool) -> list[dict]:
    rows = []
    tags = "|".join(t.get("slug", "") for t in event.get("tags") or [])
    for m in event.get("markets") or []:
        try:
            token_ids = json.loads(m.get("clobTokenIds") or "[]")
        except json.JSONDecodeError:
            token_ids = []
        rows.append(
            {
                "event_id": event.get("id"),
                "event_title": event.get("title"),
                "tags": tags,
                "market_id": m.get("id"),
                "condition_id": m.get("conditionId"),
                "question": m.get("question"),
                "slug": m.get("slug"),
                "token_yes": token_ids[0] if len(token_ids) > 0 else None,
                "token_no": token_ids[1] if len(token_ids) > 1 else None,
                "volume_usd": float(m.get("volumeNum") or m.get("volume") or 0),
                "liquidity_usd": float(m.get("liquidityNum") or m.get("liquidity") or 0),
                "start_date": m.get("startDate"),
                "end_date": m.get("endDate"),
                "closed": closed,
                "resolved_outcome": m.get("outcomePrices"),
            }
        )
    return rows


def build() -> pd.DataFrame:
    cutoff = datetime.now(timezone.utc) - timedelta(days=U["lookback_days"])
    frames = []
    for closed, cap in ((False, U["max_live_markets"]), (True, U["max_resolved_markets"])):
        events = fetch_events(closed)
        kept = [e for e in events if relevant(e) and float(e.get("volume") or 0) >= U["min_volume_usd"]]
        rows = [r for e in kept for r in flatten(e, closed)]
        df = pd.DataFrame(rows)
        if df.empty:
            continue
        df = df[df["volume_usd"] >= U["min_volume_usd"]]
        df = df.dropna(subset=["condition_id", "token_yes"])
        if closed:
            end = pd.to_datetime(df["end_date"], errors="coerce", utc=True)
            df = df[end >= cutoff]
        # one market per event first (dedupe multi-outcome events by top volume),
        # then cap the universe by volume
        df = df.sort_values("volume_usd", ascending=False)
        df = df.groupby("event_id", as_index=False).head(3)
        frames.append(df.head(cap * 2).reset_index(drop=True))  # keep 2x cap as buffer
    out = pd.concat(frames, ignore_index=True)
    return out


def main():
    df = build()
    path = DATA_DIR / "universe.parquet"
    df.to_parquet(path, index=False)
    live, res = (~df["closed"]).sum(), df["closed"].sum()
    print(f"universe: {len(df)} markets ({live} live / {res} resolved) -> {path}")
    print(f"total volume: ${df['volume_usd'].sum():,.0f}")
    print("\ntop 15 by volume:")
    for _, r in df.nlargest(15, "volume_usd").iterrows():
        state = "LIVE" if not r["closed"] else "RES "
        print(f"  [{state}] ${r['volume_usd']:>12,.0f}  {r['question'][:80]}")


if __name__ == "__main__":
    main()
