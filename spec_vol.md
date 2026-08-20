# AMIE Volatility Study — Pre-Registration (frozen 2026-08-20)

New DATA family (implied vol), not a new config on old outcomes. Committed
before any Deribit data is analyzed.

## Motivation (what survived four dead rounds)

The one confirmed physical fact: after dissonance_z >= 2 crossings, the
mapped instrument is MEDIAN-quieter for ~4h (hit 64-66% below hour-matched
norm on 2,900+ expansion events) but the MEAN is killed by a heavy right
tail. That asymmetry is untradeable directionally — but it is exactly the
shape of a volatility-selling question: is post-event implied vol rich
relative to subsequent realized vol, net of the tail?

## Hypotheses (crypto only — the sole free IV history is Deribit DVOL)

- H-V1 (sanity): after dissonance events on BTC/ETH-mapped markets, 4h
  realized vol (1-min bars) is below its hour-of-day-matched norm in the
  MEDIAN (replication of the known fact on finer data).
- H-V2 (the edge): the variance risk premium widens — (DVOL_t^2 x 4h/8760
  vs realized 4h variance) is larger after events than its hour-matched
  unconditional norm. Bar: boot-t >= 2.5 on the event-vs-norm spread.
- H-V3 (economics): a simulated 4h short-straddle (ATM, IV = DVOL at entry,
  Black-Scholes reprice at exit with realized underlying move and DVOL at
  exit, fees 3% of premium round trip) has positive mean P&L with
  boot-t >= 2.0 AND a worst-event loss that does not exceed 10x the mean
  win (tail sanity). All three must pass for a GO.

## Protocol

- Events: dissonance_z >= 2.0 upward crossings (dedup 24h) on markets whose
  FIRST mapped instrument is BTCUSDT or ETHUSDT, full feature period, both
  the capped and pre-cap feature stores (declared: these events overlap
  prior studies; the OUTCOME variables — DVOL, realized variance, straddle
  P&L — have never been examined).
- Entry: first DVOL hour strictly after the event hour (lag-1). One config.
  No sweeps. n >= 40 else UNTESTABLE.
- Stats: horizon-block bootstrap (4h -> 2-day blocks), as in the main harness.
- Outcomes: GO -> add 30-day forward leg to the existing logger and evaluate
  2026-10-01 alongside H-F's formality. FAIL on any bar -> the vol route
  closes too, and AMIE's research phase ends with the ecosystem asset.
