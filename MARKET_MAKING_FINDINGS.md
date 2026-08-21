# Market Making on Polymarket — Measured Findings (2026-08-21)

Round 6 of the AMIE research programme. After five nulls on signal-prediction,
this tests the one activity the literature documents as profitable: liquidity
provision. All numbers measured on our own tapes.

## 1. Hold-to-resolution: NO maker edge anywhere

Taker ROI by effective price band, market-clustered (equal weight per market):

| Universe | markets | notional | FAV>=50c taker ROI | implied maker edge |
|---|---|---|---|---|
| Geopolitics/macro (vision universe) | 126 | $2,031M | -0.07% (median +1.86%) | +0.07% / -1.86% |
| Crypto | 32 | $234M | +5.12% (t=2.15) | **-5.12%** |
| Sports | 368 | $1,108M | +0.60% (t=0.36) | -0.60% |

Whelan et al.'s "+2.6% for makers on contracts >=50c" does NOT replicate on any
of the three universes. Holding the position to settlement is a coin flip.

## 2. Short-horizon markout: the maker edge IS real

The maker does not hold to resolution — they capture spread and exit. Measuring
the price drift AGAINST the maker after each taker fill (147 markets, 6.6M
trades, full-depth tape):

| Horizon | mean | median | t | markets positive |
|---|---|---|---|---|
| +5 min | **+0.172c** | +0.118c | **+13.2** | 96% |
| +60 min | +0.178c | +0.118c | +9.2 | 97% |

Positive means the price moves IN FAVOUR of the maker: no adverse selection at
this horizon; the drift is the bid-ask bounce, i.e. the spread the maker earns.
This reconciles the two measurements — the business is spread capture, not
directional betting.

Caveat: computed from subsequent trade prints, so it assumes the maker is on the
other side of every fill. It is an UPPER BOUND on gross spread capture, before
queue competition and pickoff.

## 3. Capacity — the binding constraint

Total maker gross P&L available across 147 markets' lifetimes: **$1.92M**
(0.086% of notional), split among ALL makers.

- Per market-day: median **$84**, mean **-$10** (heavy left tail: on some
  markets makers are run over by news).
- Top 10 markets = **56%** of the entire pie.
- Capturing 5% of the top-20 pie would be ~$1,005/day against a $20,095/day
  pool shared by every maker present.

## 4. Decay — the edge is halving year over year

| Year | markets | pie per market-day (median) | pie as % of notional |
|---|---|---|---|
| 2025 | 41 | $119 | 0.156% |
| 2026 | 106 | $76 | **0.056%** |

Independent corroboration: the poly-maker author ran $10k -> $200/day (peak
$700-800/day) and quit when rewards fell; kachence/polymm's fill rate fell
37% -> 1% in four months and now nets ~$650/month.

## Verdict

The edge is real, small, concentrated, and shrinking fast. It is an engineering
and inventory business with a capacity ceiling in the low hundreds of dollars
per day for a new entrant, competing against incumbents quoting hundreds of
fills per hour. It is not the research programme AMIE was built as, and the
vision's own categories (geopolitics, fee-free, hence no maker rebate) are the
worst place to run it.

## 5. Cross-universe comparison (2026-08-21) — sports/crypto NOT trustworthy

Same measurements, three universes, identical settings (15-85c band, 2% share,
5-minute holds):

| Universe | markets | maker markout | backtest bps/fill | Sharpe | agree? |
|---|---|---|---|---|---|
| Geopolitics/macro | 148 | +0.176c (t=+13.0) | +68.6 | 7.46 | YES |
| Crypto | 32 | +0.164c (t=+5.0) | -11.1 | -0.25 | NO |
| Sports | 223 | -0.125c (t=-1.6) | +520.5 | 2.95 | NO |

Both alternative universes are internally contradictory: two independent
measurements of the same edge on the same data disagree in sign. Additional
red flags on the sports result:

- Top 10 markets = 55% of P&L; only 54% of markets positive.
- Implied 4.3% per fill vs the 1.17% return-on-volume of the verified live
  operator who earned $12.7M over 409 days. An order-of-magnitude claim.
- Median sports tape spans 2.6 hours (in-play markets) and is capped to the
  busiest window, so a 5-minute hold is not a realistic model there.

**Conclusion: no change to the recommendation.** Geopolitics/macro remains the
only universe where the two measurements agree, and it is the universe the
mandate points at anyway. Testing sports/crypto properly would require
order-book data and a fill model built for in-play markets - not a tape replay.
