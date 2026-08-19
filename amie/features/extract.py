"""Days 5-6 — hourly feature extraction per market, all causal.

Acoustic features (from the oscillator field + trade tape):
  F1 loudness_z          rolling RMS of |s(t)|, z vs trailing 14d
  F2 dissonance_z        min(A_yes,A_no)*A_yes*A_no/(A_yes+A_no)^2 on band envelopes
  F3 rhythm_entropy_z    Shannon entropy of 5-min trade-count histogram, 6h window
  F4 centroid_shift_z    spectral centroid delta of the smart-only field (falls back
                         to the full field when no smart wallets are flagged)

Plain baseline twins (same information, no acoustic transform):
  B1 volume_z            hourly USDC volume
  B2 imbalance_z         same dissonance formula on raw hourly YES/NO buy volumes
  B3 gapvar_z            variance of inter-trade gaps, 6h window (sign-flipped)
  B4 netflow_z           net signed smart-money flow (falls back to all wallets)

F5 (cross-market resonance) and its twin are computed in the harness, where
all markets are visible. Writes data/features_all.parquet.
"""
import numpy as np
import pandas as pd

from amie.common import DATA_DIR
from amie.features.field import ENVELOPE_CUTOFF_S, LAMBDA

TRADES_DIR = DATA_DIR / "trades"
FIELDS_DIR = DATA_DIR / "fields"

Z_WINDOW_H = 14 * 24
Z_MIN_PERIODS = 48


def causal_z(s: pd.Series) -> pd.Series:
    m = s.rolling(Z_WINDOW_H, min_periods=Z_MIN_PERIODS).mean().shift(1)
    sd = s.rolling(Z_WINDOW_H, min_periods=Z_MIN_PERIODS).std().shift(1)
    return (s - m) / sd.replace(0, np.nan)


def band_envelope(tsec: np.ndarray, amp: np.ndarray, grid_sec: np.ndarray) -> np.ndarray:
    """Sum of decaying envelopes at each grid point (no oscillation)."""
    out = np.zeros(len(grid_sec))
    for i in range(len(tsec)):
        start = np.searchsorted(grid_sec, tsec[i])
        stop = np.searchsorted(grid_sec, tsec[i] + ENVELOPE_CUTOFF_S)
        if start >= len(grid_sec):
            continue
        out[start:stop] += amp[i] * np.exp(-LAMBDA * (grid_sec[start:stop] - tsec[i]))
    return out


def spectral_centroid(x: np.ndarray) -> float:
    """Amplitude-weighted mean frequency (cycles/hour) of a 1-min complex window."""
    if len(x) < 16 or not np.any(x):
        return np.nan
    spec = np.abs(np.fft.fft(x - x.mean()))
    freqs = np.abs(np.fft.fftfreq(len(x), d=1 / 60.0))  # cycles per hour
    tot = spec.sum()
    return float((spec * freqs).sum() / tot) if tot > 0 else np.nan


def extract_market(cid: str) -> pd.DataFrame | None:
    tf, ff = TRADES_DIR / f"{cid}.parquet", FIELDS_DIR / f"{cid}.parquet"
    if not (tf.exists() and ff.exists()):
        return None
    trades, field = pd.read_parquet(tf), pd.read_parquet(ff)
    if len(trades) < 100 or field.empty:
        return None

    field = field.set_index("ts")
    mag = np.hypot(field["re"], field["im"])
    hours = pd.date_range(field.index.min().ceil("1h"), field.index.max().floor("1h"), freq="1h", tz="UTC")
    if len(hours) < Z_MIN_PERIODS:
        return None

    # --- F1 loudness: hourly RMS of |s| ---
    # audit fix: label/closed='right' so the value stamped t covers (t-1h, t]
    rms = (mag.resample("1h", closed="right", label="right")
           .apply(lambda w: np.sqrt(np.mean(w**2)) if len(w) else np.nan).reindex(hours))

    # --- F2 dissonance: band envelopes at hourly points ---
    is_yes = trades["outcome"].str.strip().str.lower().isin(["yes", "up"])
    is_buy = trades["side"].str.upper() == "BUY"
    yes_exp = (is_yes & is_buy) | (~is_yes & ~is_buy)
    tsec = trades["ts"].astype("int64").to_numpy() / 1e9
    amp = np.log1p(trades["size_usdc"].to_numpy())
    hsec = hours.astype("int64").to_numpy() / 1e9
    a_yes = band_envelope(tsec[yes_exp.to_numpy()], amp[yes_exp.to_numpy()], hsec)
    a_no = band_envelope(tsec[~yes_exp.to_numpy()], amp[~yes_exp.to_numpy()], hsec)
    tot = a_yes + a_no
    dis = np.where(tot > 0, np.minimum(a_yes, a_no) * a_yes * a_no / np.maximum(tot, 1e-9) ** 2, np.nan)

    # --- F3 rhythmic entropy + B3 gap variance (6h trailing) ---
    counts_5m = trades.set_index("ts")["size_usdc"].resample("5min", closed="right", label="right").count()
    ent, gapvar = [], []
    ts_sorted = trades["ts"].sort_values().astype("int64").to_numpy() / 1e9
    for h in hours:
        w = counts_5m[(counts_5m.index > h - pd.Timedelta(hours=6)) & (counts_5m.index <= h)]
        n = w.sum()
        if n < 10:
            ent.append(np.nan)
        else:
            p = w[w > 0] / n
            ent.append(float(-(p * np.log(p)).sum() / np.log(len(w))))
        h0 = (h - pd.Timedelta(hours=6)).timestamp()
        gaps = np.diff(ts_sorted[(ts_sorted > h0) & (ts_sorted <= h.timestamp())])
        gapvar.append(float(np.var(gaps)) if len(gaps) >= 10 else np.nan)

    # --- F4 spectral centroid of smart field (fallback: full field) ---
    smart_mag = np.hypot(field["re_smart"], field["im_smart"])
    use = field[["re_smart", "im_smart"]].to_numpy() if smart_mag.sum() > 0 else field[["re", "im"]].to_numpy()
    z = use[:, 0] + 1j * use[:, 1]
    fidx = field.index
    cent = []
    for h in hours:
        m = (fidx > h - pd.Timedelta(hours=2)) & (fidx <= h)
        cent.append(spectral_centroid(z[m]))
    cent = pd.Series(cent, index=hours)

    # --- baselines --- (all right-labeled: stamp t covers (t-1h, t])
    RS = {"closed": "right", "label": "right"}
    vol = trades.set_index("ts")["size_usdc"].resample("1h", **RS).sum().reindex(hours)
    t2 = trades.copy()
    t2["signed"] = np.where(yes_exp, 1.0, -1.0) * t2["size_usdc"]
    smart_wallets = set()
    ws = DATA_DIR / "wallets_scored.parquet"
    if ws.exists():
        w = pd.read_parquet(ws)
        smart_wallets = set(w.loc[w["smart_flag"], "wallet"])
    t_sm = t2[t2["wallet"].isin(smart_wallets)] if smart_wallets else t2
    netflow = t_sm.set_index("ts")["signed"].resample("1h", **RS).sum().reindex(hours).fillna(0)
    netflow_all = t2.set_index("ts")["signed"].resample("1h", **RS).sum().reindex(hours).fillna(0)
    v_yes = t2[yes_exp.to_numpy()].set_index("ts")["size_usdc"].resample("1h", **RS).sum().reindex(hours).fillna(0)
    v_no = t2[~yes_exp.to_numpy()].set_index("ts")["size_usdc"].resample("1h", **RS).sum().reindex(hours).fillna(0)
    vt = v_yes + v_no
    imb = np.where(vt > 0, np.minimum(v_yes, v_no) * v_yes * v_no / np.maximum(vt, 1e-9) ** 2, np.nan)
    # direction: smart 6h flow sign, falling back to all-wallet flow when the
    # smart flow is exactly zero (69% of hours) — pre-registered amendment A
    sign_sm = np.sign(netflow.rolling(6, min_periods=1).sum())
    sign_all = np.sign(netflow_all.rolling(6, min_periods=1).sum())
    direction = np.where(sign_sm != 0, sign_sm, sign_all)

    df = pd.DataFrame(
        {
            "ts": hours,
            "market": cid,
            "loudness_z": causal_z(rms),
            "dissonance_z": causal_z(pd.Series(dis, index=hours)),
            "rhythm_entropy_z": causal_z(pd.Series(ent, index=hours)),
            "centroid_shift_z": causal_z(cent.diff().abs()),
            "centroid_signed_z": causal_z(cent.diff()),
            "volume_z": causal_z(vol),
            "imbalance_z": causal_z(pd.Series(imb, index=hours)),
            "gapvar_z": causal_z(pd.Series(gapvar, index=hours)),
            "netflow_z": causal_z(netflow),
            "netflow_sign": direction,
        }
    ).reset_index(drop=True)
    return df


def main():
    frames = []
    for f in sorted(TRADES_DIR.glob("*.parquet")):
        df = extract_market(f.stem)
        if df is not None:
            frames.append(df)
            print(f"  ok  {f.stem[:16]}...  {len(df):,} hourly rows")
    out = pd.concat(frames, ignore_index=True)
    out.to_parquet(DATA_DIR / "features_all.parquet", index=False)
    print(f"features_all: {len(out):,} rows, {out['market'].nunique()} markets")


if __name__ == "__main__":
    main()
