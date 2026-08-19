# AMIE Expansion Study — Pre-Registration

**Frozen 2026-08-19, BEFORE any expansion-market data is analyzed.**
Commit hash of this file predates the first expansion test run.

## Motivation

The original sprint (spec.md) killed the pre-registered acoustic claim, and
the exploration phase burned the temporal holdout for three signal families
(explore/, all DEAD out-of-sample). The only legitimate continuation is NEW
data. This study expands the universe from ~120 markets / 120 days to the
full 365-day lookback (~250+ resolved), and tests ONLY on markets whose
trades were never ingested before this study (i.e., not among the 60 markets
in data/universe_v1_60mkts.parquet's trade set). Those markets were never
touched by any sweep, tune, or holdout run.

## Frozen hypotheses (chosen from post-hoc observations, tested on new data)

**H1 — Fade smart-money flow at 3 days (primary).**
Observation source: lens C in-sample showed smart-flow direction consistently
anti-predictive at 72h (t ~ -2, monotone in threshold; never selected, never
tested OOS). Test: event = |netflow_z| crossing >= thr (dedup 24h/market);
trade direction = MINUS sign(netflow_z) x yes_sign; lag-1 entry; hold 72h.
Thresholds: 2.0 (primary), 1.5 and 3.0 (secondary). Costs 5 bps ETF / 10 bps
crypto all-in.

**H1b — same, all-wallet flow** (netflow_all_z, new feature column): immune to
the smart-flag PIT caveat below.

**H2 — Quiet-after-dissonance replication (second moment).**
Event = dissonance_z crossing >= 2.0 / 3.0; metric = |4h forward return| of
the mapped instrument MINUS the hour-of-day-matched unconditional mean |r|
(computed on the expansion window). Prediction: negative (quieter), t <= -2.5.

## Evaluation rules

- Test set: expansion markets only (condition_id NOT in the v1 trade set),
  >= 100 trades. Same mapping.yaml, same audited pipeline (Amendment A).
- Statistics: horizon-block bootstrap (event_study.boot_se), n >= 40 per cell
  else UNTESTABLE. H1 economic bar: net Sharpe >= 1.0 AND boot-t >= 2.5.
  H2 bar: boot-t <= -2.5.
- Test budget: 3 (H1) + 3 (H1b) + 2 (H2) = 8 cells. One run. No iteration.
- If H1/H1b passes: gauntlet (lag sweep, top-3 drop, half-split by time,
  shuffle) before any claim; then 30-day forward paper validation before
  any capital.

## Disclosed caveats

- Smart flags (148 wallets) derive from the ORIGINAL 60-market set with
  resolutions <= 2026-05-20. For expansion markets active before that date
  there is a residual cross-market PIT concern (a wallet flagged partly on
  outcomes resolving after an early expansion market's window). H1b is the
  clean control: if H1 passes and H1b fails badly, suspect the flags.
- Trade tape per market capped at most recent ~10k trades (API): expansion
  events are recency-weighted within each market's life.
- The instrument mapping and yes_sign conventions are unchanged from
  mapping.yaml (frozen in the sprint).
