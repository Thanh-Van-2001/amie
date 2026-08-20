# AMIE Forward Study — Pre-Registration (frozen 2026-08-20)

The final arbiter for the one surviving lead. Committed BEFORE any forward
data exists; the commit hash is the timestamp.

## The hypothesis (one, final)

**H-F: 72h all-wallet flow continuation.** When |netflow_all_z| of a relevant
Polymarket market crosses >= 2.0 (dedup 24h/market), the mapped liquid
instrument moves in the direction sign(netflow_all_z) x yes_sign over the
next 72h, net of costs.

Provenance (disclosed): third-generation post-hoc lead. Expansion study
(spec_expansion.md) pre-registered the FADE and failed with t = -3.2, i.e.
continuation; drift-neutralization (+14.4 bps excess, t = +2.14) and a
direction-shuffled null (z = +2.0) support it in-sample; single-config
portfolio simulation shows ann Sharpe 1.28 (2025: 2.5, 2026: 0.47 — decay
risk is real and is exactly what this forward test measures).

## Protocol

- Signals are computed and LOGGED IN REAL TIME by the daily collector
  (data/forward_signals.csv: log timestamp, market, z, ticker, direction).
  A signal is valid only if its log timestamp precedes the entry bar —
  structural immunity to every form of lookahead.
- Execution: lag-1 (first hourly bar after the signal), hold 72h, costs
  5 bps ETF / 10 bps crypto all-in, max 10 concurrent, equal weight.
- Evaluation date: 2026-10-01 (~6 weeks). One evaluation. Success bar:
  net mean > 0 with horizon-block boot-t >= 2.0 AND portfolio Sharpe >= 1.0
  over the logging period. n < 40 signals by then extends the window,
  it does not relax the bar.
- Secondary replication, same bar, run at the same date: the identical
  config on any FULL-DEPTH historical tape acquired in the meantime
  (pre-cap windows never seen by any analysis to date).

## Outcomes

- PASS both bars -> 30-day paper trade, then small live capital.
- FAIL -> the flow-continuation family dies with it; AMIE reverts to the
  ecosystem-dataset asset and collectors only. No further post-hoc flips:
  this is the last test of this family, pass or fail.
