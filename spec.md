# AMIE Sprint — Pre-Registration (frozen before any test-set run)

This file freezes every degree of freedom of the validation BEFORE results
exist. The commit hash of this file predates the first test-set run; any
change after that invalidates the run.

## Question

Do psychoacoustic features of the Polymarket relevant-universe sound field
carry EARLY information about forward moves of mapped liquid instruments,
beyond the plain flow statistics they are built from?

## Universe

Polymarket markets, volume >= $50k, themes: geopolitics, macro/economics,
weather, earnings, crypto (config.yaml keyword filter). Sports, esports,
mentions, culture excluded. Lookback ~120 days for resolved markets.

## Feature definitions (see amie/features/)

- Oscillator: amplitude = log1p(trade USDC), YES band 8 cyc/h, NO band 2 cyc/h,
  SELL flips band, exponential envelope half-life 2 h, 1-minute grid.
- F1 loudness_z: hourly RMS of |field|, causal z (trailing 14 d, min 48 h, shift 1).
- F2 dissonance_z: min(A_yes,A_no)*A_yes*A_no/(A_yes+A_no)^2 on band envelopes.
- F3 rhythm_entropy_z: Shannon entropy of 5-min trade counts, 6 h window.
- F4 centroid_shift_z: |delta| of 2 h spectral centroid of smart-only field
  (fallback: full field when no smart wallets flagged).
- Baseline twins: volume_z, imbalance_z (same formula on raw hourly volumes),
  gapvar_z, netflow_z.

## Test budget (hard cap)

4 features x 2 thresholds (z >= 2, z >= 3) x 3 horizons (4 h, 24 h, 72 h)
= 24 acoustic tests + 24 twin tests. Cross-market resonance (F5) and the
whale study run only if data supports them; same thresholds.

## Protocol

- Event: causal z crosses threshold upward; dedup 24 h per market per feature.
- Outcome: forward return of the FIRST mapped instrument (mapping.yaml),
  entry at first external bar strictly after the event hour (lag-1),
  exit at first bar >= entry + horizon.
- Direction (signed features F1, F4): netflow_sign x yes_sign(theme, instrument).
- Unsigned features (F2, F3): |return| vs unconditional |return| of the same
  instrument at matched horizon.
- Split: train = first 60% of feature timestamps, embargo 3 days, test = rest.
  One locked test-set run; results are final.
- Stats: day-block bootstrap t (2000 resamples). n >= 40 events else UNTESTABLE.
- Smart flags: derived only from markets resolved before the train/test boundary.

## Gauntlet (any failure kills the feature)

1. Shuffle: random event timestamps, same count -> effect must vanish.
2. Lag sweep 0/1/2: edge at lag 0 that dies at lag 1 = lookahead artifact.
3. Top-3 drop: sign of mean must survive removing 3 largest |events|.
4. Half-split: both halves of the test set same sign.

## GO / PIVOT / KILL (Day 12, read mechanically from the table)

- GO: >= 1 acoustic feature with boot-t >= 3.0, n >= 40, hit >= 55%
  (unsigned: |move| >= 1.3x unconditional), beats twin (diff-t >= 2.0),
  toy strategy net Sharpe >= 1.0 after costs (5 bps ETF, 10 bps perp),
  survives full gauntlet.
- PIVOT: exactly one feature at t in [2,3) beating twin (diff-t >= 1.5) +
  gauntlet pass -> 2-week extension on that feature only.
- KILL: nothing beats its twin, or edges die at lag-1, or effect
  concentrates in < 5 events.

## Amendment A — pre-test audit fixes (2026-08-19, test set still untouched)

An independent 3-lens audit (lookahead / statistics / code) of the harness ran
BEFORE the locked test run. All fixes below were applied and frozen before any
test-set execution; train-set results were regenerated:

1. Smart flags: markets must RESOLVE before the boundary (not just trades
   truncated); boundary argument now required.
2. All hourly/5-min resamples right-labeled — a feature stamped t uses only
   trades <= t (was label-left: up to 60 min of future trades).
3. External bars unified to (ts = bar open, px = open price, executable AT ts);
   Binance klines were close-labeled, making crypto "lag-1" a same-hour fill.
4. Signed direction = smart 6h flow sign, falling back to all-wallet flow when
   smart flow is exactly zero; direction-0 events dropped (was: default +1,
   69% of rows — a disguised drift bet).
5. Unsigned baseline = deterministic unconditional mean |r| over the SPLIT's
   own window (was: one random draw per event from the full 2-year history).
6. Bootstrap blocks scale with horizon (ceil(h/24)+1 days); diff-t uses a
   JOINT block bootstrap preserving acoustic-twin covariance.
7. n >= 40 now enforced as a `testable` column in the results table.

## Known limitations (declared up front)

- Trades ingestion capped at max_trades_pages_per_market (config) — the cap
  and its fill rate are logged per market in ingest_log.json, never silent.
- ETF proxies trade ~7 h/day; events outside RTH map to the next session
  open by the lag-1 rule. This widens, never shrinks, the claimed lead time.
- Sprint dataset is one ~120-day window; a GO here is a license to spend
  30 more days, not a live-trading verdict.
