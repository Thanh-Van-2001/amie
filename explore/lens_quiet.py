"""EXPLORATORY (post-sprint, NOT pre-registered) — LENS A: quiet-window exploitation.

Premise (from completed analysis): after dissonance_z / rhythm_entropy_z upward
crossings, the mapped instrument's |4h move| is 20-40 bps below its
hour-of-day-matched norm.  Monetization tried here: intraday mean-reversion
inside the quiet window — observe the first 1h move after (lag-1) entry becomes
possible, trade AGAINST it, exit at quiet-window end.  Variants: fade the bar
ending at entry (pre), min-move filters, inverse-vol sizing, combined feature.

HOLDOUT DISCIPLINE: all sweeping on the first 60% of the feature period
(< 2026-05-20).  The last 40% (>= boundary + 3d embargo) is touched exactly
once, by the single best in-sample config, via --mode holdout.

Execution: lag-1 (entry strictly after the signal bar; the fade reference uses
only prices at-or-before the entry print).  Costs all-in round trip:
5 bps stocks/ETF, 10 bps crypto.  Per-ticker no-overlap rule: a new trade on a
ticker is skipped while a previous trade on that ticker is still open (also
kills duplicate trades from several markets mapping to one instrument).

Usage:
  python explore/lens_quiet.py --mode sweep            # in-sample, all configs
  python explore/lens_quiet.py --mode holdout --config <id>   # ONCE
"""
import argparse
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, r"D:\amie")
from amie.common import DATA_DIR
from amie.validation.event_study import (
    NS_H, assign_instruments, detect_events, load_external,
)

CRYPTO = {"BTCUSDT", "ETHUSDT"}
COST = {t: 0.0010 for t in CRYPTO}          # 10 bps round trip crypto
COST_DEFAULT = 0.0005                        # 5 bps round trip stock/ETF
VOL_LOOKBACK = 72                            # bars for realized 1h sigma
TRAIN_FRAC = 0.6
EMBARGO_D = 3

# ---------------------------------------------------------------- configs ----
# entry_ref: "post" = observe first in-window 1h move (i -> i+1), enter at i+1
#            "pre"  = fade the move over the bar ending at the entry print
# minmove_k: skip unless |ref move| >= k * rolling sigma;  vol_size: w ~ 1/sigma
CONFIGS = {}


def _add(cid, feature, thr, window_h, entry_ref="post", minmove_k=0.0,
         vol_size=False, direction=-1):
    CONFIGS[cid] = dict(feature=feature, thr=thr, window_h=window_h,
                        entry_ref=entry_ref, minmove_k=minmove_k,
                        vol_size=vol_size, direction=direction)


_i = 1
for _f in ("dissonance_z", "rhythm_entropy_z"):
    for _t in (1.5, 2.0, 3.0):
        for _w in (2, 4, 6):
            _add(_i, _f, _t, _w); _i += 1                      # configs 1-18
_add(19, "dissonance_z", 2.0, 4, entry_ref="pre")
_add(20, "rhythm_entropy_z", 2.0, 4, entry_ref="pre")
# refinement slots 21+ are appended after inspecting the base sweep (see
# REFINEMENTS below) — total stays <= 30 and every config run is listed.
REFINEMENTS = [
    # (cid, kwargs) filled in code below once BASE_BEST is chosen; kept static
    # in-file so the full list is auditable.
]
# Chosen AFTER the base sweep ran (configs 1-20): best in-sample cell was
# cfg 3 = dissonance_z thr 1.5 W6 (mean +30.0 bps, t 1.20, Sharpe 1.96).
# Refinements 21-25 target that cell.  Total configs run: 25 (<= 30 budget).
_add(21, "dissonance_z", 1.5, 6, minmove_k=0.5)
_add(22, "dissonance_z", 1.5, 6, minmove_k=1.0)
_add(23, "dissonance_z", 1.5, 6, vol_size=True)
_add(24, "COMBO", 1.5, 6)                       # dissonance OR rhythm event
_add(25, "dissonance_z", 1.5, 6, direction=+1)  # momentum sanity (anti-fade)


# ----------------------------------------------------------------- engine ----
def load_all():
    feats = pd.read_parquet(DATA_DIR / "features_all.parquet")
    uni = pd.read_parquet(DATA_DIR / "universe.parquet")
    mapping = assign_instruments(uni)
    ext = load_external()
    for t, p in ext.items():
        px = p["px"].to_numpy()
        ret1 = np.full(len(px), np.nan)
        ret1[1:] = px[1:] / px[:-1] - 1
        sig = pd.Series(ret1).rolling(VOL_LOOKBACK, min_periods=24).std().to_numpy()
        p["ret1"] = ret1
        p["sigma"] = sig
    t_lo, t_hi = feats["ts"].min(), feats["ts"].max()
    boundary = t_lo + (t_hi - t_lo) * TRAIN_FRAC
    return feats, mapping, ext, boundary


def detect_combo(g, thr):
    """Union of dissonance and rhythm crossings, dedup 24h via detect_events
    on a synthetic max-column."""
    g = g.copy()
    g["_combo"] = g[["dissonance_z", "rhythm_entropy_z"]].max(axis=1)
    return detect_events(g, "_combo", thr)


def gen_trades(feats, mapping, ext, cfg):
    """Candidate trades for one config on the given (pre-filtered) feats."""
    rows = []
    for cid, g in feats.groupby("market"):
        if cid not in mapping:
            continue
        g = g.sort_values("ts")
        if cfg["feature"] == "COMBO":
            ev = detect_combo(g, cfg["thr"])
        else:
            ev = detect_events(g, cfg["feature"], cfg["thr"])
        if not len(ev):
            continue
        ticker, _ = mapping[cid]["instruments"][0]
        if ticker not in ext:
            continue
        p = ext[ticker]
        ts_ns = p["ts_ns"].to_numpy()
        px = p["px"].to_numpy()
        sig = p["sigma"].to_numpy()
        for _, e in ev.iterrows():
            t_ns = e["ts"].value
            i = np.searchsorted(ts_ns, t_ns, side="right")   # lag-1 entry bar
            if cfg["entry_ref"] == "post":
                ent = i + 1                                   # after observing i->i+1
                if ent >= len(px) or i < 0:
                    continue
                ref = px[ent] / px[i] - 1                     # first in-window 1h move
            else:                                             # "pre"
                ent = i
                if ent < 1 or ent >= len(px):
                    continue
                ref = px[ent] / px[ent - 1] - 1               # bar ending at entry
            if not np.isfinite(ref) or ref == 0:
                continue
            s = sig[ent] if np.isfinite(sig[ent]) else np.nan
            if cfg["minmove_k"] > 0:
                if not np.isfinite(s) or abs(ref) < cfg["minmove_k"] * s:
                    continue
            j = np.searchsorted(ts_ns, ts_ns[i] + cfg["window_h"] * NS_H)
            if j >= len(px) or j <= ent:
                continue
            pos = cfg["direction"] * np.sign(ref)             # -1 = fade
            gross = pos * (px[j] / px[ent] - 1)
            w = 1.0
            if cfg["vol_size"] and np.isfinite(s) and s > 0:
                w = min(3.0, 0.005 / s)                       # target ~50bps/h
            cost = COST.get(ticker, COST_DEFAULT)
            rows.append(dict(ticker=ticker, entry_ns=ts_ns[ent], exit_ns=ts_ns[j],
                             net=w * (gross - cost), gross=gross, w=w))
    tr = pd.DataFrame(rows)
    if not len(tr):
        return tr
    tr = tr.sort_values("entry_ns").reset_index(drop=True)
    keep, open_until = [], {}
    for k, r in tr.iterrows():                                # no-overlap per ticker
        if r["entry_ns"] < open_until.get(r["ticker"], -1):
            continue
        keep.append(k)
        open_until[r["ticker"]] = r["exit_ns"]
    return tr.loc[keep].reset_index(drop=True)


def metrics(tr, span_days):
    if len(tr) < 5:
        return dict(n=len(tr), mean_bps=np.nan, hit=np.nan, t=np.nan,
                    sharpe=np.nan, maxdd_bps=np.nan)
    x = tr["net"].to_numpy()
    n = len(x)
    mean, std = x.mean(), x.std(ddof=1)
    t = mean / (std / np.sqrt(n)) if std > 0 else np.nan
    tpy = n / (span_days / 365.25)
    sharpe = (mean / std) * np.sqrt(tpy) if std > 0 else np.nan
    eq = np.cumsum(x)
    dd = np.maximum.accumulate(np.concatenate([[0], eq])) [1:] - eq
    return dict(n=n, mean_bps=mean * 1e4, hit=float((x > 0).mean()), t=t,
                sharpe=sharpe, maxdd_bps=dd.max() * 1e4)


def run(mode, config_id=None):
    feats, mapping, ext, boundary = load_all()
    if mode == "sweep":
        sub = feats[feats["ts"] < boundary]
        label = "IN-SAMPLE"
    else:
        sub = feats[feats["ts"] >= boundary + pd.Timedelta(days=EMBARGO_D)]
        label = "HOLDOUT (single shot)"
    span_days = (sub["ts"].max() - sub["ts"].min()).days
    print(f"{label}: {sub['ts'].min():%Y-%m-%d} .. {sub['ts'].max():%Y-%m-%d} "
          f"({span_days}d), boundary={boundary:%Y-%m-%d %H:%M}")
    ids = [config_id] if config_id else sorted(CONFIGS)
    out = []
    for cid in ids:
        cfg = CONFIGS[cid]
        tr = gen_trades(sub, mapping, ext, cfg)
        m = metrics(tr, span_days)
        out.append(dict(cfg=cid, feature=cfg["feature"], thr=cfg["thr"],
                        W=cfg["window_h"], ref=cfg["entry_ref"],
                        mmk=cfg["minmove_k"], vs=int(cfg["vol_size"]),
                        dir=cfg["direction"], **m))
    df = pd.DataFrame(out)
    print(df.to_string(index=False, float_format=lambda v: f"{v: .3f}"))
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="sweep", choices=["sweep", "holdout"])
    ap.add_argument("--config", type=int, default=None)
    args = ap.parse_args()
    if args.mode == "holdout" and args.config is None:
        sys.exit("holdout requires --config <id> (run it ONCE)")
    run(args.mode, args.config)
