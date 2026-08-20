"""Secondary replication per spec_forward.md: the IDENTICAL H-F config
(|netflow_all_z| >= 2.0 crossing, follow direction, 72h hold, same costs)
run on PRE-CAP tape windows — the part of each market's full-depth tape
strictly BEFORE the earliest timestamp of its capped tape. No analysis has
ever touched those windows.

Stages (idempotent):
  1. slice pre-cap trades  -> data/trades_precap/
  2. synthesize fields     -> data/fields_precap/
  3. extract features      -> data/features_precap.parquet
  4. run the frozen config -> event stats + portfolio sim
"""
import numpy as np
import pandas as pd

import amie.features.extract as fx
from amie.common import DATA_DIR
from amie.features.field import synthesize
from amie.validation.event_study import NS_H, assign_instruments, boot_se, load_external
from amie.validation.expansion_test import detect_abs_events, sharpe_dd

THR, H, MAXPOS = 2.0, 72, 10
COST = {"BTCUSDT": 0.0010, "ETHUSDT": 0.0010}
DEFAULT_COST = 0.0005
MAX_ENTRY_GAP_NS = 4 * 24 * NS_H  # skip events whose next external bar is >4d away

TP = DATA_DIR / "trades_precap"
FP = DATA_DIR / "fields_precap"
TP.mkdir(exist_ok=True)
FP.mkdir(exist_ok=True)


def build_precap():
    smart = None
    ws = DATA_DIR / "wallets_scored.parquet"
    if ws.exists():
        w = pd.read_parquet(ws)
        smart = set(w.loc[w["smart_flag"], "wallet"])
    n = 0
    for f in sorted((DATA_DIR / "trades_full").glob("*.parquet")):
        cid = f.stem
        capped = DATA_DIR / "trades" / f"{cid}.parquet"
        out_t, out_f = TP / f"{cid}.parquet", FP / f"{cid}.parquet"
        if out_f.exists() or not capped.exists():
            continue
        cutoff = pd.read_parquet(capped)["ts"].min()
        full = pd.read_parquet(f)
        pre = full[full["ts"] < cutoff].drop(columns=["tx"], errors="ignore")
        if len(pre) < 500 or (pre["ts"].max() - pre["ts"].min()) < pd.Timedelta(days=14):
            continue
        pre = pre.reset_index(drop=True)
        pre.to_parquet(out_t, index=False)
        field = synthesize(pre, smart)
        if field.empty:
            continue
        field.to_parquet(out_f, index=False)
        n += 1
        print(f"  precap {len(pre):>8,} trades ({pre['ts'].min():%Y-%m-%d} -> {pre['ts'].max():%Y-%m-%d}) | {cid[:14]}")
    print(f"precap built for {n} markets")


def extract_precap():
    fx.TRADES_DIR, fx.FIELDS_DIR = TP, FP  # point the extractor at pre-cap data
    frames = []
    for f in sorted(TP.glob("*.parquet")):
        df = fx.extract_market(f.stem)
        if df is not None:
            frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out.to_parquet(DATA_DIR / "features_precap.parquet", index=False)
    print(f"features_precap: {len(out):,} rows, {out['market'].nunique()} markets")
    return out


def run_config(feats):
    uni = pd.read_parquet(DATA_DIR / "universe.parquet")
    mapping = assign_instruments(uni)
    ext = load_external()

    x, days, ts_l, tick_l = [], [], [], []
    for cid, g in feats.groupby("market"):
        if cid not in mapping:
            continue
        g = g.sort_values("ts")
        for _, e in detect_abs_events(g, "netflow_all_z", THR).iterrows():
            ticker, ysign = mapping[cid]["instruments"][0]
            if ticker not in ext or e["netflow_all_z"] == 0:
                continue
            p = ext[ticker]
            arr = p["ts_ns"].to_numpy()
            i = np.searchsorted(arr, e["ts"].value, side="right")
            if i >= len(p) or arr[i] - e["ts"].value > MAX_ENTRY_GAP_NS:
                continue
            j = np.searchsorted(arr, arr[i] + H * NS_H)
            if j >= len(p):
                continue
            d = np.sign(e["netflow_all_z"]) * ysign
            r = d * (p["px"].iloc[j] / p["px"].iloc[i] - 1) - COST.get(ticker, DEFAULT_COST)
            x.append(r)
            days.append(e["ts"].floor("D").value)
            ts_l.append(e["ts"].value)
            tick_l.append(ticker)
    x, days = np.array(x), np.array(days)
    span_y = (max(ts_l) - min(ts_l)) / (365.25 * 24 * NS_H)
    se = boot_se(x, days, H)
    t = x.mean() / se if len(x) and se and se > 0 else np.nan
    sh, dd = sharpe_dd(list(x), ts_l, span_y)
    print(f"\nPRE-CAP REPLICATION (frozen config: |netflow_all_z|>=2, follow, 72h, net of costs)")
    print(f"  n={len(x)}  span={span_y:.2f}y  mean={x.mean()*1e4:+.1f} bps  hit={(x>0).mean():.3f}  "
          f"boot_t={t:+.2f}  per-event Sharpe={sh:.2f}  eventDD={dd*1e4:,.0f} bps")
    att = pd.DataFrame({"tick": tick_l, "r": x}).groupby("tick").agg(
        n=("r", "size"), mean_bps=("r", lambda s: s.mean() * 1e4), hit=("r", lambda s: (s > 0).mean()))
    print(att.to_string(float_format=lambda v: f"{v: .2f}"))
    by_year = pd.DataFrame({"y": pd.to_datetime(ts_l, utc=True).year, "r": x}).groupby("y")["r"]
    print("\nby event year:")
    for y, s in by_year:
        print(f"  {y}: n={len(s)}  mean={s.mean()*1e4:+.1f} bps  hit={(s>0).mean():.3f}")
    return x, days


if __name__ == "__main__":
    build_precap()
    feats = extract_precap()
    run_config(feats)
