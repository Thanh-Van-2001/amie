"""Day 3 — wallet scoring: who in the room has a track record.

Computes realized PnL per wallet from our own trade tape on RESOLVED markets
(payout from universe.resolved_outcome minus cost basis), plus hit rate.
smart_flag = top decile by PnL among wallets with >= 3 resolved-market trades
and positive hit rate.

Point-in-time discipline for the sprint: smart flags are only derived from
markets that resolved BEFORE the train/test boundary (set in spec.md), so the
test set never leaks outcome knowledge. Writes data/wallets_scored.parquet.
"""
import json

import numpy as np
import pandas as pd

from amie.common import DATA_DIR

TRADES_DIR = DATA_DIR / "trades"


def market_payouts(uni: pd.DataFrame, boundary: pd.Timestamp | None) -> dict[str, tuple[float, float]]:
    """condition_id -> (yes_payout, no_payout) for markets RESOLVED before boundary.

    Audit fix: filtering trades alone is not point-in-time — a market that
    resolves inside the test window still leaks its outcome into the flag.
    Only markets whose end_date precedes the boundary may contribute.
    """
    out = {}
    for _, r in uni[uni["closed"]].iterrows():
        if boundary is not None:
            end = pd.to_datetime(r["end_date"], errors="coerce", utc=True)
            if pd.isna(end) or end > boundary:
                continue
        try:
            prices = json.loads(r["resolved_outcome"]) if r["resolved_outcome"] else None
        except (json.JSONDecodeError, TypeError):
            prices = None
        if prices and len(prices) >= 2:
            out[r["condition_id"]] = (float(prices[0]), float(prices[1]))
    return out


def wallet_pnl_for_market(trades: pd.DataFrame, yes_pay: float, no_pay: float,
                          boundary: pd.Timestamp | None) -> pd.DataFrame:
    if boundary is not None:
        trades = trades[trades["ts"] <= boundary]
    if trades.empty:
        return pd.DataFrame()
    t = trades.copy()
    is_yes = t["outcome"].str.strip().str.lower().isin(["yes", "up"])
    sign = np.where(t["side"].str.upper() == "BUY", 1.0, -1.0)
    shares = np.where(t["price"] > 0, t["size_usdc"] / t["price"], 0.0)
    payout = np.where(is_yes, yes_pay, no_pay)
    # PnL = shares held to resolution * payout - net cost
    t["pnl"] = sign * shares * payout - sign * t["size_usdc"]
    g = t.groupby("wallet").agg(pnl=("pnl", "sum"), notional=("size_usdc", "sum"), n=("pnl", "size"))
    g["won"] = g["pnl"] > 0
    return g.reset_index()


def main(boundary_iso: str):
    if not boundary_iso:
        raise SystemExit("boundary date is required: python -m amie.ingest.score_wallets 2026-05-20")
    uni = pd.read_parquet(DATA_DIR / "universe.parquet")
    boundary = pd.Timestamp(boundary_iso, tz="UTC")
    payouts = market_payouts(uni, boundary)
    frames = []
    for cid, (yp, np_) in payouts.items():
        f = TRADES_DIR / f"{cid}.parquet"
        if not f.exists():
            continue
        pnl = wallet_pnl_for_market(pd.read_parquet(f), yp, np_, boundary)
        if not pnl.empty:
            pnl["market"] = cid
            frames.append(pnl)
    if not frames:
        print("no resolved markets with trades yet — run trades ingestion first")
        return
    allp = pd.concat(frames, ignore_index=True)
    reg = allp.groupby("wallet").agg(
        pnl=("pnl", "sum"),
        notional=("notional", "sum"),
        n_resolved_trades=("n", "sum"),
        n_resolved_markets=("market", "nunique"),
        hit_rate=("won", "mean"),
    ).reset_index()
    eligible = reg[(reg["n_resolved_trades"] >= 3) & (reg["hit_rate"] > 0.5)]
    cut = eligible["pnl"].quantile(0.9) if len(eligible) >= 10 else np.inf
    reg["smart_flag"] = reg["wallet"].isin(eligible[eligible["pnl"] >= cut]["wallet"])
    reg.to_parquet(DATA_DIR / "wallets_scored.parquet", index=False)
    print(f"scored {len(reg):,} wallets on {allp['market'].nunique()} resolved markets"
          + (f" (data <= {boundary_iso})" if boundary_iso else ""))
    print(f"smart set: {reg['smart_flag'].sum():,} wallets "
          f"(top-decile PnL of {len(eligible):,} eligible), "
          f"total smart PnL ${reg.loc[reg['smart_flag'], 'pnl'].sum():,.0f}")


if __name__ == "__main__":
    import sys

    main(sys.argv[1] if len(sys.argv) > 1 else "")
