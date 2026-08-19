"""Days 7-8 + 11 — the pre-registered event-study harness.

For every (feature, threshold, horizon) in the frozen test budget:
  event  = causal z-score crossing the threshold upward (dedup 24 h/market)
  outcome = forward return of the MAPPED liquid instrument (lag-1 entry:
            first external bar strictly after the event hour)
  stats  = mean, hit rate, day-block bootstrap t-stat
  twin   = the same pipeline on the plain-flow baseline feature
Gauntlet per feature: shuffle test, lag sweep 0/1/2, top-3 drop, half-split.

Directional convention (pre-registered in spec.md): direction of an event =
netflow_sign (is the crowd pushing YES or NO) x yes_sign of the instrument
for the market's theme (e.g. war YES -> USO +1, SPY -1). Unsigned features
(dissonance, rhythm entropy) test |return| vs the instrument's unconditional
|return| at matched horizon.

Usage: python -m amie.validation.event_study [--split train|test|all]
Writes data/results_<split>.csv and prints the summary table.
"""
import argparse
import json

import numpy as np
import pandas as pd
import yaml

from amie.common import DATA_DIR, ROOT

RNG = np.random.default_rng(7)

FEATURES = {
    "loudness_z": {"twin": "volume_z", "signed": True},
    "dissonance_z": {"twin": "imbalance_z", "signed": False},
    "rhythm_entropy_z": {"twin": "gapvar_z", "signed": False},
    "centroid_shift_z": {"twin": "netflow_z", "signed": True},
}
THRESHOLDS = [2.0, 3.0]
HORIZONS_H = [4, 24, 72]
DEDUP_H = 24
EMBARGO_D = 3
TRAIN_FRAC = 0.6
N_BOOT = 2000


def load_mapping():
    m = yaml.safe_load((ROOT / "amie" / "mapping.yaml").read_text(encoding="utf-8"))
    return m["themes"], m["company_tickers"]


def assign_instruments(uni: pd.DataFrame):
    """market -> list of (ticker, yes_sign). First matching theme wins."""
    themes, companies = load_mapping()
    out = {}
    for _, r in uni.iterrows():
        text = f"{r['question']} {r['event_title']} {r['tags']}".lower()
        for theme, spec in themes.items():
            if any(k.lower() in text for k in spec["match"]):
                instruments = []
                for inst in spec["instruments"]:
                    if inst == "BY_COMPANY":
                        for name, ticker in companies.items():
                            if name in text:
                                instruments.append((ticker, 1))
                                break
                    else:
                        ticker = inst.split(":")[-1].strip()
                        sign = spec.get("yes_sign", {}).get(ticker, 1)
                        instruments.append((ticker, sign))
                if instruments:
                    out[r["condition_id"]] = {"theme": theme, "instruments": instruments}
                break
    return out


def load_external():
    prices = {}
    for f in (DATA_DIR / "external").glob("*.parquet"):
        df = pd.read_parquet(f).sort_values("ts").reset_index(drop=True)
        prices[f.stem] = df
    return prices


def forward_return(prices: pd.DataFrame, t: pd.Timestamp, horizon_h: int, lag_bars: int = 1):
    """Entry at the lag_bars-th bar strictly after t; exit at first bar >= entry+h."""
    ts = prices["ts"].to_numpy()
    i = np.searchsorted(ts, np.datetime64(t.tz_convert("UTC").tz_localize(None)), side="right")
    i += lag_bars - 1
    if i >= len(prices):
        return None
    entry_t, entry_p = prices["ts"].iloc[i], prices["close"].iloc[i]
    j = np.searchsorted(ts, np.datetime64((entry_t + pd.Timedelta(hours=horizon_h)).tz_localize(None)))
    if j >= len(prices):
        return None
    return float(prices["close"].iloc[j] / entry_p - 1)


def detect_events(g: pd.DataFrame, col: str, thr: float) -> pd.DataFrame:
    z = g[col]
    cross = (z >= thr) & (z.shift(1) < thr)
    ev = g[cross.fillna(False)]
    kept, last = [], None
    for _, r in ev.iterrows():
        if last is None or (r["ts"] - last) >= pd.Timedelta(hours=DEDUP_H):
            kept.append(r)
            last = r["ts"]
    return pd.DataFrame(kept)


def boot_t(x: np.ndarray, days: np.ndarray) -> float:
    """Day-block bootstrap t-stat for the mean (handles clustered events)."""
    if len(x) < 5:
        return np.nan
    uniq = np.unique(days)
    means = []
    for _ in range(N_BOOT):
        sample = RNG.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([np.where(days == d)[0] for d in sample])
        means.append(x[idx].mean())
    se = np.std(means)
    return float(x.mean() / se) if se > 0 else np.nan


def run(split: str = "test"):
    feats = pd.read_parquet(DATA_DIR / "features_all.parquet")
    uni = pd.read_parquet(DATA_DIR / "universe.parquet")
    mapping = assign_instruments(uni)
    ext = load_external()

    t_lo, t_hi = feats["ts"].min(), feats["ts"].max()
    boundary = t_lo + (t_hi - t_lo) * TRAIN_FRAC
    if split == "train":
        feats = feats[feats["ts"] < boundary]
    elif split == "test":
        feats = feats[feats["ts"] >= boundary + pd.Timedelta(days=EMBARGO_D)]
    print(f"split={split}  rows={len(feats):,}  boundary={boundary:%Y-%m-%d}")

    rows = []
    for col, spec in FEATURES.items():
        for twin_pass, use_col in ((False, col), (True, spec["twin"])):
            for thr in THRESHOLDS:
                for h in HORIZONS_H:
                    rets, absrets, base_abs, days = [], [], [], []
                    n_ev = 0
                    for cid, g in feats.groupby("market"):
                        if cid not in mapping:
                            continue
                        g = g.sort_values("ts")
                        ev = detect_events(g, use_col, thr)
                        for _, e in ev.iterrows():
                            for ticker, ysign in mapping[cid]["instruments"][:1]:
                                if ticker not in ext:
                                    continue
                                r = forward_return(ext[ticker], e["ts"], h)
                                if r is None:
                                    continue
                                n_ev += 1
                                days.append(e["ts"].floor("D").value)
                                if spec["signed"]:
                                    direction = np.sign(e.get("netflow_sign", 1) or 1) * ysign
                                    rets.append(direction * r)
                                else:
                                    absrets.append(abs(r))
                                    p = ext[ticker]
                                    k = RNG.integers(0, max(len(p) - h - 1, 1))
                                    rb = forward_return(p, p["ts"].iloc[k], h, lag_bars=0)
                                    if rb is not None:
                                        base_abs.append(abs(rb))
                    x = np.array(rets if spec["signed"] else absrets)
                    d = np.array(days[: len(x)])
                    if spec["signed"]:
                        mean, hit = (x.mean(), (x > 0).mean()) if len(x) else (np.nan, np.nan)
                        t = boot_t(x, d) if len(x) else np.nan
                    else:
                        ub = np.mean(base_abs) if base_abs else np.nan
                        mean = x.mean() - ub if len(x) else np.nan
                        hit = (x > ub).mean() if len(x) else np.nan
                        t = boot_t(x - ub, d) if len(x) and base_abs else np.nan
                    rows.append(
                        {
                            "feature": use_col,
                            "kind": "baseline" if twin_pass else "acoustic",
                            "acoustic_of": col,
                            "thr": thr,
                            "horizon_h": h,
                            "n": int(n_ev),
                            "mean_ret": mean,
                            "hit_rate": hit,
                            "boot_t": t,
                        }
                    )
    res = pd.DataFrame(rows)
    out = DATA_DIR / f"results_{split}.csv"
    res.to_csv(out, index=False)
    print(res.to_string(index=False, float_format=lambda v: f"{v: .4f}"))
    print(f"-> {out}")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["train", "test", "all"])
    args = ap.parse_args()
    run(args.split)
