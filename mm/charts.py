"""Charts for the AMIE final report.

Palette: the validated reference instance from the dataviz skill.
Categorical slots are used in fixed order; the price-band chart is the only
one with genuine polarity, so it uses the diverging blue/red pair with a
neutral gray midpoint. One measure per axis; no dual axes anywhere.
"""
import base64
import glob
import io
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, os.path.dirname(__file__))
from backtest import run, market_pnl, CLIP, PMIN, PMAX  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e3e2df"
S1 = "#2a78d6"   # categorical slot 1 - blue
S2 = "#eb6834"   # slot 2 - orange
POS = "#2a78d6"  # diverging cool pole
NEG = "#e34948"  # diverging warm pole
OUT = r"D:\AMIE_Reports\charts"
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 9,
    "text.color": INK, "axes.labelcolor": INK2, "axes.edgecolor": GRID,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
})


def finish(fig, ax, title, sub, fname, ylab=""):
    ax.set_title(title, color=INK, fontsize=11, fontweight="bold", loc="left", pad=14)
    ax.text(0, 1.02, sub, transform=ax.transAxes, color=INK2, fontsize=8.5, va="bottom")
    if ylab:
        ax.set_ylabel(ylab, fontsize=8.5)
    ax.set_axisbelow(True)
    fig.tight_layout()
    p = os.path.join(OUT, fname)
    fig.savefig(p, dpi=170, facecolor=SURFACE)
    plt.close(fig)
    print("wrote", fname)
    return p


def b64(p):
    return "data:image/png;base64," + base64.b64encode(open(p, "rb").read()).decode()


# ---- 1. equity curve, base case -------------------------------------------
day = run(0.02, label="chart-base")
eq = day.cumsum()
fig, ax = plt.subplots(figsize=(7.2, 3.0))
ax.plot(eq.index, eq.values, color=S1, linewidth=2)
ax.fill_between(eq.index, 0, eq.values, color=S1, alpha=0.10, linewidth=0)
peak = eq.cummax()
dd_i = (peak - eq).idxmax()
ax.annotate(f"max drawdown ${(peak - eq).max():,.0f}", xy=(dd_i, eq.loc[dd_i]),
            xytext=(10, -26), textcoords="offset points", color=INK2, fontsize=8,
            arrowprops=dict(arrowstyle="-", color=INK2, linewidth=0.8))
ax.annotate(f"${eq.iloc[-1]:,.0f}", xy=(eq.index[-1], eq.iloc[-1]),
            xytext=(-6, 8), textcoords="offset points", color=INK,
            fontsize=9.5, fontweight="bold", ha="right")
ax.set_ylim(bottom=min(0, eq.min() * 1.1))
p1 = finish(fig, ax, "Cumulative profit - passive market making",
            "Base case: 2% of maker volume, $50 clips, 5-minute holds, costs charged",
            "equity.png", "USD")

# ---- 2. bps per fill by price band (the longshot trap) ---------------------
bands = [(0.00, 0.05), (0.05, 0.10), (0.10, 0.15), (0.15, 0.30),
         (0.30, 0.50), (0.50, 0.70), (0.70, 0.85), (0.85, 1.00)]
tot = np.zeros(len(bands))
notl = np.zeros(len(bands))
for f in glob.glob(r"D:\amie\data\trades_full\*.parquet"):
    d = pd.read_parquet(f, columns=["ts", "outcome", "side", "price", "size_usdc"])
    if len(d) < 2000:
        continue
    d = d.sort_values("ts").reset_index(drop=True)
    isy = d.outcome.str.strip().str.lower().isin(["yes", "up"]).to_numpy()
    ypx = np.where(isy, d.price.to_numpy(), 1 - d.price.to_numpy())
    t = d.ts.astype("int64").to_numpy() / 1e9
    tb = (d.side.str.upper() == "BUY").to_numpy()
    mlong = np.where(isy, ~tb, tb)
    notional = np.minimum(d.size_usdc.to_numpy(), CLIP)
    j = np.searchsorted(t, t + 300)
    ok = j < len(t)
    idx = np.where(ok)[0]
    entry, ex = ypx[idx], ypx[j[idx]]
    drift = np.where(mlong[idx], ex - entry, entry - ex)
    sh = notional[idx] / np.maximum(entry, 0.01)
    pnl = drift * sh - 0.001 * sh
    for k, (lo, hi) in enumerate(bands):
        m = (entry >= lo) & (entry < hi)
        tot[k] += pnl[m].sum()
        notl[k] += notional[idx][m].sum()
bps = 1e4 * tot / np.maximum(notl, 1)
labels = [f"{int(lo*100)}-{int(hi*100)}c" for lo, hi in bands]
cols = [NEG if v < 0 else POS for v in bps]
fig, ax = plt.subplots(figsize=(7.2, 3.2))
ax.bar(labels, bps, color=cols, width=0.62, linewidth=0)
ax.axhline(0, color=INK2, linewidth=1)
ax.axvspan(2.5, 6.5, color="#f0efec", alpha=0.55, zorder=0)
ax.set_ylim(bottom=min(bps)*1.28)
ax.text(4.5, ax.get_ylim()[1] * 0.72, "quoted band (15-85c)", ha="center",
        color=INK2, fontsize=8.5)
for i, v in enumerate(bps):
    ax.annotate(f"{v:+,.0f}", (i, v), textcoords="offset points",
                xytext=(0, 5 if v >= 0 else -16), ha="center", va="top" if v<0 else "bottom",
                color=INK, fontsize=8)
p2 = finish(fig, ax, "Why longshot quotes destroy a market maker",
            "Profit per fill by entry price - the same strategy, split by where it quotes",
            "bands.png", "basis points per fill")

# ---- 3. capture-share sweep ------------------------------------------------
shares = [0.01, 0.02, 0.05, 0.10]
res = [run(s, label=f"chart-share {s:.0%}") for s in shares]
daily = [r.mean() for r in res]
sharpe = [r.mean() / r.std() * np.sqrt(365) for r in res]
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
xl = [f"{int(s*100)}%" for s in shares]
axes[0].bar(xl, daily, color=S1, width=0.6, linewidth=0)
for i, v in enumerate(daily):
    axes[0].annotate(f"${v:,.0f}", (i, v), textcoords="offset points",
                     xytext=(0, 4), ha="center", color=INK, fontsize=8.5)
axes[0].set_ylabel("USD per day", fontsize=8.5)
axes[0].set_title("Profit per day", color=INK, fontsize=9.5, loc="left", pad=8)
axes[1].bar(xl, sharpe, color=S2, width=0.6, linewidth=0)
for i, v in enumerate(sharpe):
    axes[1].annotate(f"{v:.1f}", (i, v), textcoords="offset points",
                     xytext=(0, 4), ha="center", color=INK, fontsize=8.5)
axes[1].set_title("Annualised Sharpe", color=INK, fontsize=9.5, loc="left", pad=8)
for a in axes:
    a.set_xlabel("our share of maker volume", fontsize=8.5)
    a.set_axisbelow(True)
fig.suptitle("Scaling with participation", color=INK, fontsize=11,
             fontweight="bold", x=0.008, ha="left", y=1.0)
fig.text(0.008, 0.93, "Profit per fill stays flat (62-67 bps); Sharpe rises because more fills means more independent bets",
         color=INK2, fontsize=8.5, ha="left")
fig.tight_layout(rect=[0, 0, 1, 0.90])
p3 = os.path.join(OUT, "sweep.png")
fig.savefig(p3, dpi=170, facecolor=SURFACE)
plt.close(fig)
print("wrote sweep.png")

# ---- 4. daily P&L distribution --------------------------------------------
d2 = day[day != 0]
fig, ax = plt.subplots(figsize=(7.2, 2.6))
ax.hist(d2.values, bins=45, color=S1, alpha=0.85, linewidth=0)
ax.axvline(0, color=INK2, linewidth=1)
ax.axvline(d2.mean(), color=S2, linewidth=2)
ax.annotate(f"mean ${d2.mean():,.0f}/day", xy=(d2.mean(), ax.get_ylim()[1] * 0.85),
            xytext=(8, 0), textcoords="offset points", color=INK, fontsize=8.5)
ax.text(0.99, 0.85, f"{(d2 > 0).mean():.0%} of active days profitable",
        transform=ax.transAxes, ha="right", color=INK2, fontsize=8.5)
p4 = finish(fig, ax, "Daily profit distribution",
            "Active trading days, base case - a spread-capture book, not a directional one",
            "dist.png", "days")

open(os.path.join(OUT, "embed.txt"), "w").write("\n".join(
    [b64(p1), b64(p2), b64(p3), b64(p4)]))
print("charts done")
