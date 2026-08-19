"""Day 12 — daily collector: ingest -> field -> features -> alert log.

Designed for cron (once daily). Each stage is idempotent and resume-safe;
new markets and new trades extend the parquet stores, features regenerate,
and any feature whose latest causal z >= 2 lands in data/alerts.log.
"""
from datetime import datetime, timezone

import pandas as pd

from amie.common import DATA_DIR


def main():
    from amie.ingest import universe, prices, trades, participants
    from amie.features import field, extract

    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print(f"[{stamp}] AMIE daily cycle start")
    universe.main()
    prices.main()
    trades.main()
    participants.main(holders_limit=10)
    field.main()
    extract.main()

    feats = pd.read_parquet(DATA_DIR / "features_all.parquet")
    latest = feats.sort_values("ts").groupby("market").tail(1)
    cols = ["loudness_z", "dissonance_z", "rhythm_entropy_z", "centroid_shift_z"]
    hot = latest[(latest[cols] >= 2.0).any(axis=1)]
    with open(DATA_DIR / "alerts.log", "a", encoding="utf-8") as fh:
        for _, r in hot.iterrows():
            vals = ", ".join(f"{c}={r[c]:.2f}" for c in cols if pd.notna(r[c]) and r[c] >= 2.0)
            fh.write(f"{stamp} {r['market']} {vals}\n")
    print(f"[{stamp}] cycle done — {len(hot)} market(s) above z=2 -> alerts.log")


if __name__ == "__main__":
    main()
