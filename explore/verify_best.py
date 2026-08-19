"""INDEPENDENT VERIFIER — re-implementation of the three lenses' claimed BEST
configs, run on HOLDOUT ONLY (>= boundary + 3d embargo = 2026-05-23 19:36 UTC).

Written from the reports' config specs, NOT from lens code. Only the
pre-registered harness helpers are reused (assign_instruments, detect_events,
forward_return, load_external).

Per lens: base holdout run, lag sweep 0/1/2, top-3 |net| drop (sign check),
first/second half of holdout (sign check).

Configs verified:
  A: dissonance_z >= 1.5 upcross, dedup 24h/mkt; fade first 1h move at lag-1
     bar; exit first bar >= obs_bar + 6h; per-ticker no-overlap; costs 5/10bps.
  B: dissonance_z down-cross 1.0 with >=2.0 spike in past 12h, dedup 24h/mkt;
     direction = sign(all-wallet net YES flow in (t-3h, t]) x yes_sign;
     hold 24h via forward_return; costs 5/10bps.
  C: smart-wallet hourly yes-flow z (trailing 168h excl current, minp 48,
     dense grid over each tape's own span); |z|>=1.5 upcross & flow!=0,
     dir = sign(smart_net) x yes_sign, dedup 24h/mkt; breadth: >=2 distinct
     markets same (ticker,dir) within 6h, dedup 24h/(ticker,dir); hold 4h.
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, r"D:\amie")
from amie.validation.event_study import (  # noqa: E402
    NS_H, assign_instruments, detect_events, forward_return, load_external,
)

DATA = r"D:\amie\data"
CRYPTO = {"BTCUSDT", "ETHUSDT"}


def cost_of(t):
    return 0.0010 if t in CRYPTO else 0.0005


def load_base():
    feats = pd.read_parquet(os.path.join(DATA, "features_all.parquet"))
    t_lo, t_hi = feats["ts"].min(), feats["ts"].max()
    boundary = t_lo + (t_hi - t_lo) * 0.6
    hold_start = boundary + pd.Timedelta(days=3)
    uni = pd.read_parquet(os.path.join(DATA, "universe.parquet"))
    mapping = assign_instruments(uni)
    ext = load_external()
    return feats, hold_start, mapping, ext


def stats(trades, span_years, label):
    """trades: DataFrame[ts, net]. Prints one metrics line, returns dict."""
    if len(trades) < 3:
        print(f"  {label:<18} n={len(trades)} (too few)")
        return None
    tr = trades.sort_values("ts").reset_index(drop=True)
    x = tr["net"].to_numpy()
    n = len(x)
    mu, sd = x.mean(), x.std(ddof=1)
    t = mu / (sd / np.sqrt(n)) if sd > 0 else np.nan
    sharpe = mu / sd * np.sqrt(n / span_years) if sd > 0 else np.nan
    cum = np.cumsum(x)
    dd = float(np.max(np.maximum.accumulate(cum) - cum))
    print(f"  {label:<18} n={n:<4d} mean={mu*1e4:+7.1f}bps hit={np.mean(x>0):.3f} "
          f"t={t:+5.2f} annSharpe={sharpe:+5.2f} maxDD={dd*1e4:.0f}bps")
    return dict(n=n, mean_bps=mu * 1e4, hit=float(np.mean(x > 0)), t=t,
                sharpe=sharpe, maxdd_bps=dd * 1e4, x=x, tr=tr)


def robustness(tr, span_years):
    """top-3 |net| drop + time-half split on a base trades frame."""
    x = tr.sort_values("ts")["net"].to_numpy()
    full = x.mean()
    keep = np.argsort(-np.abs(x))[3:]
    m_drop = x[keep].mean() if len(keep) else np.nan
    print(f"  top3|net| drop:    mean={m_drop*1e4:+7.1f}bps  "
          f"(sign {'KEPT' if np.sign(m_drop) == np.sign(full) else 'FLIPPED'} vs full {full*1e4:+.1f})")
    # also drop the 3 most POSITIVE trades (does any positive edge rely on outlier wins)
    keep_p = np.argsort(-x)[3:]
    print(f"  top3 winners drop: mean={x[keep_p].mean()*1e4:+7.1f}bps")
    half = len(x) // 2
    m1, m2 = x[:half].mean(), x[half:].mean()
    print(f"  halves (time):     first={m1*1e4:+7.1f}bps  second={m2*1e4:+7.1f}bps  "
          f"({'SAME SIGN' if np.sign(m1) == np.sign(m2) else 'DIFFER'})")


# ---------------------------------------------------------------- LENS A ----
def lens_a_trades(feats_hold, mapping, ext, lag=1):
    """Fade first 1h move after dissonance_z>=1.5 crossing; exit obs_bar+6h.
    lag shifts the observation bar: base lag=1 -> obs bar = first bar
    strictly after event ts; lag=0 -> one bar earlier (contains lookahead);
    lag=2 -> one bar later."""
    rows = []
    for cid, g in feats_hold.groupby("market"):
        if cid not in mapping:
            continue
        g = g.sort_values("ts")
        ev = detect_events(g, "dissonance_z", 1.5)
        if not len(ev):
            continue
        ticker, _ = mapping[cid]["instruments"][0]
        if ticker not in ext:
            continue
        p = ext[ticker]
        ts_ns = p["ts_ns"].to_numpy()
        px = p["px"].to_numpy()
        for _, e in ev.iterrows():
            i = np.searchsorted(ts_ns, e["ts"].value, side="right") + (lag - 1)
            ent = i + 1                       # enter after observing i -> i+1
            if i < 1 or ent >= len(px):
                continue
            ref = px[ent] / px[i] - 1
            if not np.isfinite(ref) or ref == 0:
                continue
            j = np.searchsorted(ts_ns, ts_ns[i] + 6 * NS_H)
            if j >= len(px) or j <= ent:
                continue
            pos = -np.sign(ref)               # fade
            net = pos * (px[j] / px[ent] - 1) - cost_of(ticker)
            rows.append(dict(ts=pd.Timestamp(ts_ns[ent], tz="UTC"),
                             ticker=ticker, entry_ns=ts_ns[ent],
                             exit_ns=ts_ns[j], net=net))
    tr = pd.DataFrame(rows)
    if not len(tr):
        return tr
    tr = tr.sort_values("entry_ns").reset_index(drop=True)
    keep, open_until = [], {}
    for k, r in tr.iterrows():                # per-ticker no-overlap
        if r["entry_ns"] < open_until.get(r["ticker"], -1):
            continue
        keep.append(k)
        open_until[r["ticker"]] = r["exit_ns"]
    return tr.loc[keep].reset_index(drop=True)


# ---------------------------------------------------------------- LENS B ----
def load_tape():
    tapes = {}
    for p in glob.glob(os.path.join(DATA, "trades", "*.parquet")):
        cid = os.path.basename(p)[:-8]
        d = pd.read_parquet(p)
        sgn = np.where(((d["side"] == "BUY") & (d["outcome"] == "Yes")) |
                       ((d["side"] == "SELL") & (d["outcome"] == "No")), 1.0, -1.0)
        tapes[cid] = pd.DataFrame({"ts": d["ts"], "signed": sgn * d["size_usdc"].to_numpy()}
                                  ).sort_values("ts").reset_index(drop=True)
    return tapes


def resolution_events(g, thr_hi=2.0, thr_lo=1.0, k_h=12):
    g = g.sort_values("ts").reset_index(drop=True)
    z = g["dissonance_z"]
    cross_dn = ((z < thr_lo) & (z.shift(1) >= thr_lo)).fillna(False)
    spike_ts = g.loc[z >= thr_hi, "ts"].to_numpy()
    kept, last = [], None
    for i in np.where(cross_dn.to_numpy())[0]:
        t = g.at[i, "ts"]
        if not ((spike_ts >= t - pd.Timedelta(hours=k_h)) & (spike_ts < t)).any():
            continue
        if last is not None and (t - last) < pd.Timedelta(hours=24):
            continue
        kept.append(g.iloc[i])
        last = t
    return pd.DataFrame(kept)


def lens_b_trades(feats_hold, mapping, ext, tapes, lag=1):
    rows = []
    for cid, g in feats_hold.groupby("market"):
        if cid not in mapping:
            continue
        ev = resolution_events(g)
        if not len(ev):
            continue
        ticker, ysign = mapping[cid]["instruments"][0]
        if ticker not in ext or cid not in tapes:
            continue
        tape = tapes[cid]
        for _, e in ev.iterrows():
            t = e["ts"]
            # all-wallet net YES flow over trades in (t-3h, t] — strictly causal
            m = (tape["ts"] > t - pd.Timedelta(hours=3)) & (tape["ts"] <= t)
            s = float(tape.loc[m, "signed"].sum())
            if s == 0:
                continue
            d = np.sign(s) * ysign
            r = forward_return(ext[ticker], t.value, 24, lag_bars=lag)
            if r is None:
                continue
            rows.append(dict(ts=t, ticker=ticker, net=d * r - cost_of(ticker)))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- LENS C ----
def smart_flow_events(mapping, thr=1.5):
    """Base events for breadth: |smart-flow z| >= thr upcross, per market."""
    w = pd.read_parquet(os.path.join(DATA, "wallets_scored.parquet"))
    smart = set(w.loc[w["smart_flag"], "wallet"])
    events = []
    for p in glob.glob(os.path.join(DATA, "trades", "*.parquet")):
        cid = os.path.basename(p)[:-8]
        if cid not in mapping:
            continue
        ticker, ysign = mapping[cid]["instruments"][0]
        d = pd.read_parquet(p)
        d = d[d["wallet"].isin(smart)]
        if not len(d):
            continue
        sgn = np.where(((d["side"] == "BUY") & (d["outcome"] == "Yes")) |
                       ((d["side"] == "SELL") & (d["outcome"] == "No")), 1.0, -1.0)
        d = d.assign(yflow=sgn * d["size_usdc"], hr=d["ts"].dt.floor("h"))
        # dense hourly grid over the FULL tape's coverage (all wallets), so
        # zero-smart hours inside coverage are real zeros
        full = pd.read_parquet(p)
        lo, hi = full["ts"].dt.floor("h").min(), full["ts"].dt.floor("h").max()
        idx = pd.date_range(lo, hi, freq="h", tz="UTC")
        s = d.groupby("hr")["yflow"].sum().reindex(idx, fill_value=0.0)
        mu = s.shift(1).rolling(168, min_periods=48).mean()
        sd = s.shift(1).rolling(168, min_periods=48).std().replace(0.0, np.nan)
        z = (s - mu) / sd
        az = z.abs()
        cross = ((az >= thr) & (az.shift(1) < thr) & (s != 0)).fillna(False)
        for t in idx[cross.to_numpy()]:
            events.append(dict(ts=t, market=cid, ticker=ticker,
                               dir=float(np.sign(s.loc[t]) * ysign)))
    # dedup 24h per market
    events.sort(key=lambda e: e["ts"])
    kept, last = [], {}
    for e in events:
        k = e["market"]
        if k not in last or (e["ts"] - last[k]) >= pd.Timedelta(hours=24):
            kept.append(e)
            last[k] = e["ts"]
    return kept


def lens_c_trades(base_events, ext, hold_start, hold_end, lag=1, k=2):
    evs = sorted(base_events, key=lambda e: e["ts"])
    raw = []
    for i, e in enumerate(evs):
        mkts = {e["market"]}
        for j in range(i - 1, -1, -1):
            if (e["ts"] - evs[j]["ts"]) > pd.Timedelta(hours=6):
                break
            if evs[j]["ticker"] == e["ticker"] and evs[j]["dir"] == e["dir"]:
                mkts.add(evs[j]["market"])
        if len(mkts) >= k:
            raw.append(e)
    # dedup 24h per (ticker, dir)
    kept, last = [], {}
    for e in raw:
        key = (e["ticker"], e["dir"])
        if key not in last or (e["ts"] - last[key]) >= pd.Timedelta(hours=24):
            kept.append(e)
            last[key] = e["ts"]
    rows = []
    for e in kept:
        if not (hold_start <= e["ts"] <= hold_end):
            continue
        if e["ticker"] not in ext:
            continue
        r = forward_return(ext[e["ticker"]], e["ts"].value, 4, lag_bars=lag)
        if r is None:
            continue
        rows.append(dict(ts=e["ts"], ticker=e["ticker"],
                         net=e["dir"] * r - cost_of(e["ticker"])))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ main ----
def main():
    feats, hold_start, mapping, ext = load_base()
    hold = feats[feats["ts"] >= hold_start]
    hold_end = feats["ts"].max()
    span_years = (hold_end - hold["ts"].min()).total_seconds() / (365.25 * 86400)
    print(f"HOLDOUT: {hold['ts'].min()} -> {hold_end}  ({span_years:.3f} yr)")

    print("\n=== LENS A: fade-1h-move after dissonance>=1.5, exit +6h ===")
    base_a = None
    for lag in (0, 1, 2):
        tr = lens_a_trades(hold, mapping, ext, lag=lag)
        s = stats(tr, span_years, f"lag={lag}" + (" (BASE)" if lag == 1 else ""))
        if lag == 1:
            base_a = s
    if base_a:
        robustness(base_a["tr"], span_years)
        print("  per-ticker:", base_a["tr"].groupby("ticker")["net"]
              .agg(["count", "mean"]).mul([1, 1e4]).round(1).to_dict("index"))

    print("\n=== LENS B: dissonance resolution + 3h all-flow direction, hold 24h ===")
    tapes = load_tape()
    base_b = None
    for lag in (0, 1, 2):
        tr = lens_b_trades(hold, mapping, ext, tapes, lag=lag)
        s = stats(tr, span_years, f"lag={lag}" + (" (BASE)" if lag == 1 else ""))
        if lag == 1:
            base_b = s
    if base_b:
        robustness(base_b["tr"], span_years)
        print("  per-ticker:", base_b["tr"].groupby("ticker")["net"]
              .agg(["count", "mean"]).mul([1, 1e4]).round(1).to_dict("index"))

    print("\n=== LENS C: smart-flow breadth k=2 within 6h, hold 4h ===")
    base_events = smart_flow_events(mapping)
    print(f"  base smart-flow events (full period): {len(base_events)}")
    base_c = None
    for lag in (0, 1, 2):
        tr = lens_c_trades(base_events, ext, hold_start, hold_end, lag=lag)
        s = stats(tr, span_years, f"lag={lag}" + (" (BASE)" if lag == 1 else ""))
        if lag == 1:
            base_c = s
    if base_c:
        robustness(base_c["tr"], span_years)
        print("  per-ticker:", base_c["tr"].groupby("ticker")["net"]
              .agg(["count", "mean"]).mul([1, 1e4]).round(1).to_dict("index"))


if __name__ == "__main__":
    main()
