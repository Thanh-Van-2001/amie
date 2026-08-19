"""Days 7-8 + 11 — the pre-registered event-study harness.

For every (feature, threshold, horizon) in the frozen test budget:
  event  = causal z-score crossing the threshold upward (dedup 24 h/market)
  outcome = forward return of the MAPPED liquid instrument (lag-1 entry:
            first external bar strictly after the event hour)
  stats  = mean, hit rate, day-block bootstrap t-stat
  twin   = the same pipeline on the plain-flow baseline feature
  diff_t = two-sample day-block bootstrap t of (acoustic mean - twin mean);
           spec requires diff_t >= 2.0 for GO
Gauntlet (--gauntlet feature thr horizon): shuffle test, lag sweep 0/1/2,
top-3 drop, half-split — any failure kills the feature.

Directional convention (pre-registered in spec.md): direction of an event =
netflow_sign x yes_sign of the instrument for the market's theme. Unsigned
features (dissonance, rhythm entropy) test |return| minus the instrument's
unconditional |return| at matched horizon.

Usage: python -m amie.validation.event_study --split train|test
       python -m amie.validation.event_study --split test --gauntlet dissonance_z 3 24
Writes data/results_<split>.csv and prints the summary table.
"""
import argparse

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
N_SHUFFLE = 200


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
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        df["ts_ns"] = df["ts"].astype("int64")
        prices[f.stem] = df
    return prices


NS_H = 3_600_000_000_000
NS_D = 24 * NS_H


def forward_return(prices: pd.DataFrame, t_ns: int, horizon_h: int, lag_bars: int = 1):
    """Entry at the lag_bars-th bar strictly after t; exit at first bar >= entry+h.

    External rows are (ts, px) with px executable AT ts (bar-open convention).
    """
    ts_ns = prices["ts_ns"].to_numpy()
    i = np.searchsorted(ts_ns, t_ns, side="right")
    i += lag_bars - 1
    if i < 0 or i >= len(prices):
        return None
    entry_ns, entry_p = ts_ns[i], prices["px"].iloc[i]
    j = np.searchsorted(ts_ns, entry_ns + horizon_h * NS_H)
    if j >= len(prices):
        return None
    return float(prices["px"].iloc[j] / entry_p - 1)


def unconditional_abs(prices: pd.DataFrame, horizon_h: int, lo_ns: int, hi_ns: int) -> float:
    """Mean |forward return| over every bar inside [lo, hi] — audit fix: the
    unsigned baseline is deterministic and drawn from the SPLIT's own window,
    never the full multi-year history."""
    ts_ns = prices["ts_ns"].to_numpy()
    px = prices["px"].to_numpy()
    m = (ts_ns >= lo_ns) & (ts_ns <= hi_ns)
    idx = np.where(m)[0]
    if len(idx) < 10:
        return np.nan
    j = np.searchsorted(ts_ns, ts_ns[idx] + horizon_h * NS_H)
    ok = j < len(ts_ns)
    if ok.sum() < 10:
        return np.nan
    return float(np.mean(np.abs(px[j[ok]] / px[idx[ok]] - 1)))


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


def cell_events(feats, mapping, ext, col, thr, horizon_h, signed, lag_bars=1,
                shuffle=False, uncond=None):
    """All events for one (feature, thr, horizon) cell -> arrays of returns/days.

    Unsigned cells return excess |r| over the split-window unconditional mean
    |r| of the same instrument/horizon (`uncond` dict, precomputed). Signed
    events with direction 0 (no net flow either smart or all-wallet) are
    dropped — pre-registered amendment A. shuffle=True replaces each market's
    event times with uniform random hours from its own span (the null pipeline).
    """
    x, days = [], []
    for cid, g in feats.groupby("market"):
        if cid not in mapping:
            continue
        g = g.sort_values("ts")
        ev = detect_events(g, col, thr)
        if shuffle and len(ev):
            picks = RNG.choice(len(g), size=len(ev), replace=False)
            ev = g.iloc[sorted(picks)]
        for _, e in ev.iterrows():
            for ticker, ysign in mapping[cid]["instruments"][:1]:
                if ticker not in ext:
                    continue
                r = forward_return(ext[ticker], e["ts"].value, horizon_h, lag_bars)
                if r is None:
                    continue
                if signed:
                    direction = float(e.get("netflow_sign", 0) or 0) * ysign
                    if not np.isfinite(direction) or direction == 0:
                        continue
                    x.append(np.sign(direction) * r)
                else:
                    ub = (uncond or {}).get((ticker, horizon_h), np.nan)
                    if not np.isfinite(ub):
                        continue
                    x.append(abs(r) - ub)
                days.append(e["ts"].floor("D").value)
    return np.array(x), np.array(days)


def _blocks(days_ns: np.ndarray, horizon_h: int) -> np.ndarray:
    """Audit fix: block length scales with horizon so overlapping forward
    windows on adjacent days share a block (24h -> 2d, 72h -> 4d blocks)."""
    block_len = int(np.ceil(horizon_h / 24)) + 1
    return (days_ns // NS_D) // block_len


def boot_se(x: np.ndarray, days: np.ndarray, horizon_h: int) -> float:
    """Horizon-block bootstrap standard error of the mean."""
    if len(x) < 5:
        return np.nan
    b = _blocks(days, horizon_h)
    uniq = np.unique(b)
    means = []
    for _ in range(N_BOOT):
        sample = RNG.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([np.where(b == d)[0] for d in sample])
        means.append(x[idx].mean())
    return float(np.std(means))


def boot_diff_t(xa, da, xb, db, horizon_h: int) -> float:
    """Audit fix: JOINT block bootstrap of (mean_a - mean_b). Acoustic and twin
    events overlap in time, so independent SEs overstate Var(diff); resampling
    the union of blocks once per iteration keeps the covariance."""
    if len(xa) < 5 or len(xb) < 5:
        return np.nan
    ba, bb = _blocks(da, horizon_h), _blocks(db, horizon_h)
    uniq = np.unique(np.concatenate([ba, bb]))
    diffs = []
    for _ in range(N_BOOT):
        sample = RNG.choice(uniq, size=len(uniq), replace=True)
        ia = np.concatenate([np.where(ba == d)[0] for d in sample])
        ib = np.concatenate([np.where(bb == d)[0] for d in sample])
        if len(ia) == 0 or len(ib) == 0:
            continue
        diffs.append(xa[ia].mean() - xb[ib].mean())
    if len(diffs) < 100:
        return np.nan
    se = np.std(diffs)
    return float((xa.mean() - xb.mean()) / se) if se > 0 else np.nan


def split_feats(split: str):
    feats = pd.read_parquet(DATA_DIR / "features_all.parquet")
    t_lo, t_hi = feats["ts"].min(), feats["ts"].max()
    boundary = t_lo + (t_hi - t_lo) * TRAIN_FRAC
    if split == "train":
        feats = feats[feats["ts"] < boundary]
    elif split == "test":
        feats = feats[feats["ts"] >= boundary + pd.Timedelta(days=EMBARGO_D)]
    return feats, boundary


def build_uncond(feats, ext):
    """Split-window unconditional mean |r| per (ticker, horizon)."""
    lo, hi = feats["ts"].min().value, feats["ts"].max().value
    return {(t, h): unconditional_abs(p, h, lo, hi)
            for t, p in ext.items() for h in HORIZONS_H}


def run(split: str = "test"):
    feats, boundary = split_feats(split)
    uni = pd.read_parquet(DATA_DIR / "universe.parquet")
    mapping = assign_instruments(uni)
    ext = load_external()
    uncond = build_uncond(feats, ext)
    print(f"split={split}  rows={len(feats):,}  boundary={boundary:%Y-%m-%d}")

    cells = {}
    rows = []
    for col, spec in FEATURES.items():
        for kind, use_col in (("acoustic", col), ("baseline", spec["twin"])):
            for thr in THRESHOLDS:
                for h in HORIZONS_H:
                    x, d = cell_events(feats, mapping, ext, use_col, thr, h,
                                       spec["signed"], uncond=uncond)
                    cells[(use_col, thr, h)] = (x, d)
                    se = boot_se(x, d, h)
                    rows.append(
                        {
                            "feature": use_col,
                            "kind": kind,
                            "acoustic_of": col,
                            "thr": thr,
                            "horizon_h": h,
                            "n": len(x),
                            "testable": len(x) >= 40,
                            "mean_ret": x.mean() if len(x) else np.nan,
                            "hit_rate": (x > 0).mean() if len(x) else np.nan,
                            "boot_t": x.mean() / se if len(x) and se and se > 0 else np.nan,
                        }
                    )
    res = pd.DataFrame(rows)

    # acoustic-vs-twin difference t (joint block bootstrap, audit fix)
    diff = []
    for col, spec in FEATURES.items():
        for thr in THRESHOLDS:
            for h in HORIZONS_H:
                xa, da = cells[(col, thr, h)]
                xb, db = cells[(spec["twin"], thr, h)]
                diff.append(boot_diff_t(xa, da, xb, db, h))
    res.loc[res["kind"] == "acoustic", "diff_t_vs_twin"] = diff

    out = DATA_DIR / f"results_{split}.csv"
    res.to_csv(out, index=False)
    print(res.to_string(index=False, float_format=lambda v: f"{v: .4f}"))
    print(f"-> {out}")
    return res


def gauntlet(split: str, col: str, thr: float, h: int):
    """Pre-registered kill tests for one surviving cell."""
    feats, _ = split_feats(split)
    uni = pd.read_parquet(DATA_DIR / "universe.parquet")
    mapping = assign_instruments(uni)
    ext = load_external()
    uncond = build_uncond(feats, ext)
    signed = FEATURES[col]["signed"]

    print(f"GAUNTLET {col} thr={thr} h={h} split={split}")
    x, d = cell_events(feats, mapping, ext, col, thr, h, signed, uncond=uncond)
    se = boot_se(x, d, h)
    t_real = x.mean() / se if se and se > 0 else np.nan
    print(f"  real:      n={len(x)}  mean={x.mean():+.4f}  t={t_real:+.2f}")

    # 1. lag sweep — edge must survive lag 1 and 2
    for lag in (0, 1, 2):
        xl, dl = cell_events(feats, mapping, ext, col, thr, h, signed, lag_bars=lag, uncond=uncond)
        sel = boot_se(xl, dl, h)
        tl = xl.mean() / sel if sel and sel > 0 else np.nan
        print(f"  lag {lag}:     n={len(xl)}  mean={xl.mean():+.4f}  t={tl:+.2f}")

    # 2. shuffle — effect must vanish on random event times
    sh_means = []
    for _ in range(N_SHUFFLE):
        xs, _ = cell_events(feats, mapping, ext, col, thr, h, signed, shuffle=True, uncond=uncond)
        if len(xs):
            sh_means.append(xs.mean())
    sh = np.array(sh_means)
    z = (x.mean() - sh.mean()) / sh.std() if len(sh) and sh.std() > 0 else np.nan
    p = float((sh >= x.mean()).mean()) if len(sh) else np.nan
    print(f"  shuffle:   null mean={sh.mean():+.4f}  real z vs null={z:+.2f}  p={p:.3f}")

    # 3. top-3 drop — sign must survive
    if len(x) > 3:
        keep = np.argsort(-np.abs(x))[3:]
        print(f"  top3 drop: mean={x[keep].mean():+.4f}  (sign {'KEPT' if np.sign(x[keep].mean()) == np.sign(x.mean()) else 'FLIPPED'})")

    # 4. half-split — both halves same sign
    order = np.argsort(d)
    half = len(x) // 2
    m1, m2 = x[order[:half]].mean(), x[order[half:]].mean()
    print(f"  halves:    first={m1:+.4f}  second={m2:+.4f}  ({'SAME SIGN' if np.sign(m1) == np.sign(m2) else 'DIFFER'})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["train", "test", "all"])
    ap.add_argument("--gauntlet", nargs=3, metavar=("FEATURE", "THR", "HORIZON"))
    args = ap.parse_args()
    if args.gauntlet:
        gauntlet(args.split, args.gauntlet[0], float(args.gauntlet[1]), int(args.gauntlet[2]))
    else:
        run(args.split)
