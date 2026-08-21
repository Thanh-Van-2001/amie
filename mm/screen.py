"""Target selection for paper market-making. Inverts the naive APY screen:
rank by SAFETY, not by reward yield. Empty reward bands are empty because
they are toxic (verified: median implied APY 1561% on unquoted markets).

Gates: someone else already quotes (bookQ>0), touch <= max_touch,
volume24h >= min_vol, resolution >= min_hours away, mid in [0.15,0.85].
"""
import json, sys, time
import requests

CLOB = "https://clob.polymarket.com"
GAMMA = "https://gamma-api.polymarket.com"
S = requests.Session(); S.headers["User-Agent"] = "amie-mm/0.1"

MAX_TOUCH = 0.02
MIN_VOL24 = 100_000
MIN_HOURS = 24
MID_LO, MID_HI = 0.15, 0.85


def gamma_markets(limit=500):
    out, off = [], 0
    while len(out) < limit:
        b = S.get(f"{GAMMA}/markets", params={"limit": 100, "offset": off, "closed": "false",
                                              "order": "volume24hr", "ascending": "false"}, timeout=30).json()
        if not b:
            break
        out += b; off += 100
    return out


def book(tid):
    try:
        r = S.get(f"{CLOB}/book", params={"token_id": tid}, timeout=15).json()
        bids, asks = r.get("bids") or [], r.get("asks") or []
        if not bids or not asks:
            return None
        bb = max(float(x["price"]) for x in bids)
        ba = min(float(x["price"]) for x in asks)
        qb = sum(float(x["size"]) for x in bids if abs(float(x["price"]) - bb) < 1e-9)
        qa = sum(float(x["size"]) for x in asks if abs(float(x["price"]) - ba) < 1e-9)
        return bb, ba, qb, qa
    except Exception:
        return None


def main(n_out=10):
    now = time.time()
    rows = []
    for m in gamma_markets():
        try:
            v24 = float(m.get("volume24hr") or 0)
            if v24 < MIN_VOL24:
                continue
            end = m.get("endDate")
            if not end:
                continue
            hrs = (time.mktime(time.strptime(end[:19], "%Y-%m-%dT%H:%M:%S")) - now) / 3600
            if hrs < MIN_HOURS:
                continue
            tids = json.loads(m.get("clobTokenIds") or "[]")
            if len(tids) < 2:
                continue
            b = book(tids[0])
            if not b:
                continue
            bb, ba, qb, qa = b
            touch, mid = ba - bb, (ba + bb) / 2
            if touch > MAX_TOUCH or not (MID_LO <= mid <= MID_HI) or min(qb, qa) <= 0:
                continue
            rows.append({"q": (m.get("question") or "")[:60], "cid": m.get("conditionId"),
                         "yes": tids[0], "no": tids[1], "v24": round(v24),
                         "touch_c": round(touch * 100, 2), "mid": round(mid, 3),
                         "bookQ": round(min(qb, qa)), "hrs": round(hrs),
                         "fee": m.get("feeType") or m.get("category") or "?"})
        except Exception:
            continue
    rows.sort(key=lambda r: (r["touch_c"], -r["v24"]))
    json.dump(rows[:n_out], open("targets.json", "w"), indent=1)
    print(f"{len(rows)} passed gates; top {min(n_out,len(rows))}:")
    for r in rows[:n_out]:
        print(f"  {r['touch_c']:.2f}c  ${r['v24']:>9,}  mid {r['mid']:.2f}  Q{r['bookQ']:>5}  {r['hrs']:>4}h  {r['fee'][:14]:14} {r['q']}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 10)
