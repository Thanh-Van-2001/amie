"""Shadow-book collector: snapshots the CLOB book for target markets and the
recent tape. Fill simulation and markout are computed OFFLINE from this log,
so data collection is never contaminated by modelling choices.
Writes book_snaps.jsonl and tape.jsonl (append-only).
"""
import json, time
from datetime import datetime, timezone
import requests

CLOB = "https://clob.polymarket.com"
DATA = "https://data-api.polymarket.com"
S = requests.Session(); S.headers["User-Agent"] = "amie-mm/0.1"
INTERVAL = 20


def snap(tid):
    r = S.get(f"{CLOB}/book", params={"token_id": tid}, timeout=15).json()
    bids, asks = r.get("bids") or [], r.get("asks") or []
    if not bids or not asks:
        return None
    bids = sorted(((float(b["price"]), float(b["size"])) for b in bids), reverse=True)[:5]
    asks = sorted((float(a["price"]), float(a["size"])) for a in asks)[:5]
    return {"bids": bids, "asks": asks}


def main(cycles=100000):
    tg = json.load(open("targets.json"))
    print(f"collecting {len(tg)} markets every {INTERVAL}s", flush=True)
    seen = set()
    for i in range(cycles):
        ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with open("book_snaps.jsonl", "a", encoding="utf-8") as fb, \
             open("tape.jsonl", "a", encoding="utf-8") as ft:
            for m in tg:
                try:
                    y, n = snap(m["yes"]), snap(m["no"])
                    if y and n:
                        fb.write(json.dumps({"ts": ts, "cid": m["cid"], "yes": y, "no": n}) + "\n")
                    tr = S.get(f"{DATA}/trades", params={"market": m["cid"], "limit": 60,
                                                        "takerOnly": "true"}, timeout=20).json() or []
                    for t in tr:
                        k = t.get("transactionHash", "") + str(t.get("timestamp"))
                        if k in seen:
                            continue
                        seen.add(k)
                        ft.write(json.dumps({"cid": m["cid"], "ts": t.get("timestamp"),
                                             "side": t.get("side"), "outcome": t.get("outcome"),
                                             "price": t.get("price"), "size": t.get("size")}) + "\n")
                except Exception:
                    continue
        if i % 15 == 0:
            print(f"[{ts}] cycle {i} tape={len(seen)}", flush=True)
        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()
