"""Day 3 groundwork — the participant ecosystem (the crowd in the room).

For every wallet seen trading the universe, build a registry row:
markets touched, notional traded, first/last seen. Top holders per market
add position snapshots. This is the boss's "ecosystem of market
participants" — who is in the room, and who screams first.

Writes data/participants.parquet (registry) and data/holders.parquet.
"""
import pandas as pd

from amie.common import CONFIG, DATA_DIR, get_json

DATA_API = CONFIG["api"]["data"]
TRADES_DIR = DATA_DIR / "trades"


def registry_from_trades() -> pd.DataFrame:
    frames = []
    for f in TRADES_DIR.glob("*.parquet"):
        df = pd.read_parquet(f)
        if df.empty:
            continue
        g = df.groupby("wallet").agg(
            n_trades=("ts", "size"),
            notional_usdc=("size_usdc", "sum"),
            first_seen=("ts", "min"),
            last_seen=("ts", "max"),
        )
        g["market"] = f.stem
        frames.append(g.reset_index())
    per_market = pd.concat(frames, ignore_index=True)
    reg = per_market.groupby("wallet").agg(
        n_markets=("market", "nunique"),
        n_trades=("n_trades", "sum"),
        notional_usdc=("notional_usdc", "sum"),
        first_seen=("first_seen", "min"),
        last_seen=("last_seen", "max"),
    )
    return reg.sort_values("notional_usdc", ascending=False).reset_index()


def fetch_holders(condition_id: str, cap: int = 100) -> pd.DataFrame:
    js = get_json(f"{DATA_API}/holders", {"market": condition_id, "limit": cap})
    rows = []
    for tok in js if isinstance(js, list) else [js]:
        for h in tok.get("holders") or []:
            rows.append(
                {
                    "market": condition_id,
                    "token": tok.get("token"),
                    "wallet": h.get("proxyWallet"),
                    "shares": float(h.get("amount") or 0),
                    "outcome": h.get("outcomeIndex"),
                }
            )
    return pd.DataFrame(rows)


def main(holders_limit: int | None = 10):
    reg = registry_from_trades()
    reg.to_parquet(DATA_DIR / "participants.parquet", index=False)
    print(f"participants: {len(reg):,} unique wallets across {TRADES_DIR.glob('*.parquet').__sizeof__() and len(list(TRADES_DIR.glob('*.parquet')))} markets")
    print(f"  total notional: ${reg['notional_usdc'].sum():,.0f}")
    print(f"  wallets in 2+ markets: {(reg['n_markets'] >= 2).sum():,}")
    print("\ntop 10 by notional:")
    for _, r in reg.head(10).iterrows():
        print(f"  {r['wallet'][:12]}...  ${r['notional_usdc']:>12,.0f}  {r['n_trades']:>6,} trades  {r['n_markets']:>3} markets")

    if holders_limit:
        uni = pd.read_parquet(DATA_DIR / "universe.parquet").nlargest(holders_limit, "volume_usd")
        frames = [fetch_holders(cid) for cid in uni["condition_id"]]
        hold = pd.concat([f for f in frames if not f.empty], ignore_index=True)
        hold.to_parquet(DATA_DIR / "holders.parquet", index=False)
        print(f"\nholders: {len(hold):,} position snapshots on top {holders_limit} markets")


if __name__ == "__main__":
    main()
