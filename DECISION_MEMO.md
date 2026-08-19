# AMIE Sprint — Decision Memo

**Date:** 2026-08-19 · **Verdict: KILL** (read mechanically from the frozen criteria table)

## The question we asked

Do psychoacoustic features of the Polymarket relevant-universe sound field
carry early information about forward moves of mapped liquid instruments,
beyond the plain flow statistics they are built from? (spec.md, frozen before
any test-set run; Amendment A audit fixes applied and frozen pre-test.)

## The criteria table, filled with actual test-set numbers

| Gate (frozen in spec.md)                        | Required | Best actual (test set)                  | Pass? |
|--------------------------------------------------|----------|-----------------------------------------|-------|
| Any acoustic cell boot-t ≥ 3.0, n ≥ 40           | ≥ 3.0    | +1.47 (rhythm_entropy z2/72h, n=41)     | NO    |
| Hit rate ≥ 55% on that cell                      | ≥ 55%    | 58.5% on a t=1.47 cell (not significant)| NO    |
| Beats plain-flow twin, diff-t ≥ 2.0              | ≥ 2.0    | +1.69 (centroid z2/72h, itself t=1.02)  | NO    |
| PIVOT: exactly one cell t ∈ [2,3), diff-t ≥ 1.5  | —        | no positive cell reaches t = 2          | NO    |
| KILL trigger: nothing beats its twin             | —        | triggered                               | KILL  |

Toy strategy and gauntlet were not run: no cell qualified to enter them.

## What actually happened, in one paragraph

The pre-audit pipeline showed dissonance boot-t up to 5.0 on the training
window. An independent three-lens audit (lookahead, statistics, code) then
found seven real defects — non-point-in-time smart flags, left-labeled
resamples leaking up to 60 minutes of future trades into feature stamps,
close-labeled crypto bars collapsing lag-1 to a same-hour fill, a direction
default that turned 69% of signed events into a disguised drift bet, a
volatility-regime-mismatched unsigned baseline, bootstrap blocks too short
for overlapping horizons, and an unenforced n≥40 gate. With all seven fixed
and refrozen, the training signal collapsed, and the single locked test run
confirmed it: no acoustic feature carries positive predictive information
beyond its plain-flow twin on this dataset. The apparent edge was the bugs.

## Exploratory observation (NOT a claim; sign was not pre-registered)

Dissonance and rhythm-entropy spikes are followed by *quieter*-than-
unconditional 4h windows in the mapped instruments (t = −5.0 to −6.3, and
more negative than their twins, diff-t −2.0 to −3.9). Candidate explanations
include a session-timing confound (events clustering outside US market hours)
or a real calm-before-resolution effect. If pursued, it must be a NEW
pre-registered study with a timing-matched baseline — not a resurrection of
this one.

## What survives the kill

- **The participant-ecosystem dataset** (the boss's goal #1): 114,466 wallets,
  $342M notional, 28,219 multi-market wallets, PIT-clean scored smart set —
  regenerable daily by `pipeline_daily.py`. This asset is independent of the
  acoustic hypothesis.
- The ingestion infrastructure (universe, tape, prices, external bars) and the
  audited event-study harness — reusable for any future signal study.
- Forward collectors (daily cron + whale watcher) keep accumulating data as a
  cheap option, per the KILL row of the plan.

## Next 30 days (per the frozen KILL row)

1. Archive the repo in its current state (this memo is the closing document).
2. Keep the daily collector and whale watcher running 30 days.
3. No patent filing.
4. Return attention to the trading P&L.
