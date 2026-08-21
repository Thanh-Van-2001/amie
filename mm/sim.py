"""Offline paper-quote simulator with markout — the decisive test.

We quote post-only BUY-YES and BUY-NO (RN1's structure: both sides are bids
in a binary market; a filled pair merges to USDC at locked edge 1-p-q).
Quotes sit one tick behind the touch, so queue-ahead = all size at that price.
A fill happens only when the tape trades THROUGH our price. Every fill is
then marked out at +60s and +300s against the book mid.

P&L decomposition (the part the 5 failed rounds lacked):
    net = spread_capture - markout_loss   (rewards/rebates reported separately)
Kill rule: if markout_loss > spread_capture, the market is toxic. Drop it.
"""
import json
from collections import defaultdict
from datetime import datetime
import numpy as np

TICK = 0.01
CLIP = 50.0        # USD per quote, RN1's median clip size
MARKOUTS = [60, 300]


def load():
    snaps = defaultdict(list)
    for ln in open("book_snaps.jsonl", encoding="utf-8"):
        d = json.loads(ln)
        t = datetime.fromisoformat(d["ts"]).timestamp()
        snaps[d["cid"]].append((t, d["yes"], d["no"]))
    tape = defaultdict(list)
    for ln in open("tape.jsonl", encoding="utf-8"):
        d = json.loads(ln)
        try:
            tape[d["cid"]].append((float(d["ts"]), str(d["outcome"]).lower(),
                                   str(d["side"]).upper(), float(d["price"]), float(d["size"])))
        except (TypeError, ValueError):
            continue
    for k in snaps: snaps[k].sort()
    for k in tape: tape[k].sort()
    return snaps, tape


def mid_at(rows, t):
    """Book mid of the YES token at or before time t."""
    best = None
    for ts, y, _ in rows:
        if ts > t:
            break
        best = (y["bids"][0][0] + y["asks"][0][0]) / 2
    return best


def main():
    snaps, tape = load()
    out = []
    for cid, rows in snaps.items():
        trades = tape.get(cid, [])
        if len(rows) < 3 or not trades:
            continue
        fills = []
        for i in range(len(rows) - 1):
            t0, y, n = rows[i]
            t1 = rows[i + 1][0]
            # our post-only bids, one tick behind each touch
            qy = round(y["bids"][0][0] - TICK, 3)
            qn = round(n["bids"][0][0] - TICK, 3)
            for ts, outc, side, px, sz in trades:
                if not (t0 <= ts < t1) or side != "SELL":
                    continue  # a taker SELL is what lifts a resting bid
                is_yes = outc in ("yes", "up")
                q = qy if is_yes else qn
                if px <= q:  # traded through our price
                    fills.append((ts, "YES" if is_yes else "NO", q, min(CLIP, sz * px)))
        if not fills:
            continue
        rec = {"cid": cid[:12], "n_fills": len(fills), "notional": round(sum(f[3] for f in fills), 1)}
        for h in MARKOUTS:
            mos = []
            for ts, sidey, q, notion in fills:
                m0 = mid_at(rows, ts)
                m1 = mid_at(rows, ts + h)
                if m0 is None or m1 is None or m1 == m0:
                    continue
                # long YES gains if mid rises; long NO gains if mid falls
                d = (m1 - m0) if sidey == "YES" else (m0 - m1)
                mos.append(d / max(q, 1e-6))
            rec[f"mo{h}_bps"] = round(float(np.mean(mos)) * 1e4, 1) if mos else None
            rec[f"mo{h}_n"] = len(mos)
        # spread capture: half-touch earned per fill, as bps of price paid
        caps = []
        for ts, sidey, q, notion in fills:
            m0 = mid_at(rows, ts)
            if m0 is None:
                continue
            edge = (m0 - q) if sidey == "YES" else ((1 - m0) - q)
            caps.append(edge / max(q, 1e-6))
        rec["spread_bps"] = round(float(np.mean(caps)) * 1e4, 1) if caps else None
        out.append(rec)
    if not out:
        print("no fills yet — collector needs more runtime")
        return
    print(f"{'market':14}{'fills':>6}{'notional':>10}{'spread':>9}{'mo60':>9}{'mo300':>9}{'verdict':>10}")
    for r in out:
        s, m = r.get("spread_bps"), r.get("mo300_bps")
        v = "?" if s is None or m is None else ("TOXIC" if -m > s else "OK")
        print(f"{r['cid']:14}{r['n_fills']:>6}{r['notional']:>10.0f}"
              f"{(s if s is not None else 0):>9.1f}{(r.get('mo60_bps') or 0):>9.1f}"
              f"{(m if m is not None else 0):>9.1f}{v:>10}")
    json.dump(out, open("sim_results.json", "w"), indent=1)


if __name__ == "__main__":
    main()
