"""Day 4 — oscillator encoding and composite field synthesis.

Every wallet is an oscillator. Per trade:
  amplitude  = log1p(trade notional USD)
  frequency  = YES band (8 cycles/h) or NO band (2 cycles/h); SELL flips the band
  phase      = trade timestamp
  envelope   = exponential decay, half-life 2 h

The market's field s(t) = sum over trades of A * exp(-lambda*(t-t0)) * exp(i*2pi*f*(t-t0))
sampled on a 1-minute grid. A second field uses smart wallets only.
Writes data/fields/<condition_id>.parquet: ts, re, im, re_smart, im_smart, n_active.
"""
import numpy as np
import pandas as pd

from amie.common import DATA_DIR

TRADES_DIR = DATA_DIR / "trades"
OUT = DATA_DIR / "fields"
OUT.mkdir(exist_ok=True)

F_YES = 8.0 / 3600.0   # cycles per second
F_NO = 2.0 / 3600.0
HALF_LIFE_S = 2 * 3600.0
LAMBDA = np.log(2) / HALF_LIFE_S
GRID_S = 60
ENVELOPE_CUTOFF_S = 6 * HALF_LIFE_S   # ignore contribution beyond ~1.6% amplitude


def trade_frequency(outcome: str, side: str) -> float:
    """YES exposure -> high band, NO exposure -> low band. Selling YES ~ buying NO."""
    is_yes = str(outcome).strip().lower() in ("yes", "up", "0")
    if str(side).upper() == "SELL":
        is_yes = not is_yes
    return F_YES if is_yes else F_NO


def synthesize(trades: pd.DataFrame, smart: set[str] | None = None) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    t0 = trades["ts"].min().floor("1min")
    t1 = trades["ts"].max().ceil("1min")
    grid = pd.date_range(t0, t1, freq=f"{GRID_S}s", tz="UTC")
    gsec = grid.astype("int64").to_numpy() / 1e9

    re = np.zeros(len(grid))
    im = np.zeros(len(grid))
    re_s = np.zeros(len(grid))
    im_s = np.zeros(len(grid))
    n_active = np.zeros(len(grid), dtype=np.int32)

    tsec = trades["ts"].astype("int64").to_numpy() / 1e9
    amp = np.log1p(trades["size_usdc"].to_numpy())
    freq = np.array([trade_frequency(o, s) for o, s in zip(trades["outcome"], trades["side"])])
    wallets = trades["wallet"].to_numpy()

    for i in range(len(trades)):
        start = np.searchsorted(gsec, tsec[i])
        stop = np.searchsorted(gsec, tsec[i] + ENVELOPE_CUTOFF_S)
        if start >= len(grid):
            continue
        dt = gsec[start:stop] - tsec[i]
        contrib = amp[i] * np.exp(-LAMBDA * dt) * np.exp(2j * np.pi * freq[i] * dt)
        re[start:stop] += contrib.real
        im[start:stop] += contrib.imag
        n_active[start:stop] += 1
        if smart is not None and wallets[i] in smart:
            re_s[start:stop] += contrib.real
            im_s[start:stop] += contrib.imag

    return pd.DataFrame(
        {"ts": grid, "re": re, "im": im, "re_smart": re_s, "im_smart": im_s, "n_active": n_active}
    )


def main():
    smart = None
    scored = DATA_DIR / "wallets_scored.parquet"
    if scored.exists():
        ws = pd.read_parquet(scored)
        smart = set(ws.loc[ws["smart_flag"], "wallet"])
        print(f"smart set: {len(smart):,} wallets")
    for f in sorted(TRADES_DIR.glob("*.parquet")):
        path = OUT / f.name
        if path.exists():
            continue
        trades = pd.read_parquet(f)
        field = synthesize(trades, smart)
        if field.empty:
            continue
        field.to_parquet(path, index=False)
        print(f"  ok  {f.stem[:16]}...  {len(field):,} grid pts, {len(trades):,} trades")
    print(f"fields -> {OUT}")


if __name__ == "__main__":
    main()
