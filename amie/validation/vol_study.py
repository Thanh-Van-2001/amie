"""Vol study runner — implements spec_vol.md exactly. One pass, no sweeps.

H-V1: median 4h RV below hour-matched norm after dissonance events (crypto).
H-V2: variance risk premium (DVOL-implied 4h var minus realized 4h var)
      wider after events than hour-matched norm; boot-t >= 2.5.
H-V3: short 4h ATM straddle held to expiry: P&L = premium - |dS| - fees
      (premium = 0.7979 * S * sigma * sqrt(T), sigma = DVOL at entry,
      fees 3% of premium RT). boot-t >= 2.0 and worst loss <= 10x mean win.
"""
import numpy as np
import pandas as pd

from amie.common import DATA_DIR
from amie.validation.event_study import assign_instruments, boot_se, detect_events

H = 4
THR = 2.0
FEE = 0.03
ANN_H = 8766.0


def load_events():
    uni = pd.read_parquet(DATA_DIR / "universe.parquet")
    mapping = assign_instruments(uni)
    evs = []
    for fname in ("features_all.parquet", "features_precap.parquet"):
        p = DATA_DIR / fname
        if not p.exists():
            continue
        feats = pd.read_parquet(p)
        for cid, g in feats.groupby("market"):
            m = mapping.get(cid)
            if not m or m["instruments"][0][0] not in ("BTCUSDT", "ETHUSDT"):
                continue
            for _, e in detect_events(g.sort_values("ts"), "dissonance_z", THR).iterrows():
                evs.append((e["ts"], m["instruments"][0][0]))
    ev = pd.DataFrame(evs, columns=["ts", "sym"]).drop_duplicates().sort_values("ts")
    return ev


def main():
    ev = load_events()
    print(f"events: {len(ev)} (BTC {sum(ev.sym=='BTCUSDT')} / ETH {sum(ev.sym=='ETHUSDT')})")
    out = []
    for sym, cur in (("BTCUSDT", "BTC"), ("ETHUSDT", "ETH")):
        m1 = pd.read_parquet(DATA_DIR / "vol" / f"m1_{sym}.parquet").set_index("ts")["close"]
        dvol = pd.read_parquet(DATA_DIR / "vol" / f"dvol_{cur}.parquet").set_index("ts")["dvol"]
        lr = np.log(m1).diff()
        # realized 4h variance anchored at each full hour
        rv = (lr**2).resample("1h", closed="right", label="right").sum()
        rv4 = rv.rolling(H).sum().shift(-H)  # variance over (t, t+4h], anchored t
        hours = rv4.index
        dv = dvol.reindex(hours).ffill()
        s0 = m1.resample("1h", closed="right", label="right").last().reindex(hours)
        s4 = s0.shift(-H)
        iv_var4 = (dv / 100.0) ** 2 * (H / ANN_H)
        vrp = iv_var4 - rv4
        prem = 0.7979 * s0 * (dv / 100.0) * np.sqrt(H / ANN_H)
        pnl = (prem - (s4 - s0).abs() - FEE * prem) / s0  # as fraction of spot
        base = pd.DataFrame({"rv4": rv4, "vrp": vrp, "pnl": pnl, "hour": hours.hour}).dropna()
        norms = base.groupby("hour").agg(rv_n=("rv4", "mean"), vrp_n=("vrp", "mean"),
                                         pnl_n=("pnl", "mean"), rv_med=("rv4", "median"))
        for _, r in ev[ev["sym"] == sym].iterrows():
            t = r["ts"].ceil("1h")
            if t not in base.index:
                continue
            b = base.loc[t]
            n = norms.loc[t.hour]
            out.append({"ts": t, "sym": sym,
                        "rv_ex": b["rv4"] - n["rv_n"], "rv_below_med": b["rv4"] < n["rv_med"],
                        "vrp_ex": b["vrp"] - n["vrp_n"], "pnl": b["pnl"], "pnl_ex": b["pnl"] - n["pnl_n"]})
    df = pd.DataFrame(out).dropna()
    days = df["ts"].dt.floor("D").astype("int64").to_numpy()
    print(f"usable events: {len(df)}")

    med_frac = df["rv_below_med"].mean()
    print(f"H-V1 median-quiet: {med_frac:.3f} of events below hour-matched median RV (bar >0.5)")
    for col, bar, name in (("vrp_ex", 2.5, "H-V2 VRP widening"), ("pnl_ex", 2.0, "H-V3 straddle excess P&L")):
        x = df[col].to_numpy()
        se = boot_se(x, days, H)
        t = x.mean() / se if se and se > 0 else np.nan
        print(f"{name}: mean={x.mean():+.6f}  boot_t={t:+.2f}  (bar {bar})")
    pnl = df["pnl"].to_numpy()
    wins = pnl[pnl > 0]
    print(f"H-V3 raw: mean={pnl.mean()*1e4:+.1f} bps of spot  hit={(pnl>0).mean():.3f}  "
          f"worst={pnl.min()*1e4:,.0f} bps  meanwin={wins.mean()*1e4:.1f} bps  "
          f"tail ratio={abs(pnl.min())/wins.mean() if len(wins) else np.nan:.1f} (bar <=10)")
    df.to_csv(DATA_DIR / "results_vol.csv", index=False)


if __name__ == "__main__":
    main()
