# Expansion Study — Results (single locked run, 2026-08-20)

Per spec_expansion.md (frozen at commit 1c4ef3e, before any expansion-market
data was analyzed). Test set: 288 markets never touched by any prior sweep,
353,623 hourly rows, 1.57 years of span. Full table: data/results_expansion.csv.

| cell | n | mean net bps | hit | boot-t | bar | verdict |
|---|---|---|---|---|---|---|
| H1 fade smart flow, thr 2.0 (primary) | 1,616 | -8.5 | .483 | -0.66 | t >= +2.5 | FAIL |
| H1 thr 1.5 / 3.0 | 1,748 / 1,424 | -7.0 / -3.7 | .48 | -0.60 / -0.29 | " | FAIL |
| H1b fade all-wallet flow, thr 2.0 | 4,464 | -18.6 | .474 | -2.81 | t >= +2.5 | FAIL (opposite sign) |
| H1b thr 1.5 / 3.0 | 5,162 / 3,475 | -19.8 / -17.3 | .47 | -3.16 / -1.67 | " | FAIL |
| H2 quiet replication, thr 2.0 / 3.0 | 2,907 / 1,909 | +0.9 / +2.9 | .66 / .64 quieter | +0.30 / +0.82 | t <= -2.5 | FAIL — does not replicate |

## Readings (honest)

1. **H1 dead.** The fade-smart-flow lead from the exploration phase was noise;
   at n ~ 1,600 on fresh markets it is indistinguishable from zero.
2. **H1b failed with the opposite sign** — all-wallet Polymarket flow shows
   72h CONTINUATION into mapped instruments (fading it loses ~-20 bps net,
   t -3.2 at n=5,162). This is the third consecutive post-hoc sign flip in
   this research line (dissonance->quiet, smart-fade, now flow-momentum).
   A flipped-sign result on a pre-registered test is NOT a discovery; it is
   at best the next hypothesis. Caveats stacked against it: overlapping 72h
   windows, many markets mapping to few instruments (cross-event correlation
   beyond the day-block correction), and instrument drift over a 1.57y span
   with yes_sign conventions that correlate with that drift.
3. **H2's mean-based quiet effect does not replicate** on new markets
   (mean +1 to +3 bps vs required negative). The hit-rate asymmetry (64-66%
   of events ARE quieter than the hour-matched norm, yet the mean is
   positive) says the original effect was median-quiet with a heavy right
   tail — the sprint-window mean result was period-specific.

## Decision

Three rounds (pre-registered sprint -> exploration with holdout -> expansion
with fresh markets) have now each killed the prior round's best lead. This
dataset is exhausted for this family of hypotheses. The only legitimate
remaining test is the FORWARD data the collectors are accumulating; the only
candidate worth pre-registering on it, if any, is 72h all-wallet flow
continuation (H1b sign-flip) with drift-neutralized controls — one test,
in ~4-6 weeks, and whatever it says is final.

## Pre-cap replication (2026-08-20, per spec_forward.md secondary bar)

Frozen config on 117 markets' pre-cap windows (full-depth tape via
time-window cursor-walk; windows never touched by any analysis):
n=5,542, span 1.61y, mean net -8.8 bps, hit 49.3%, boot-t -1.33.
Negative in both 2025 and 2026; USO (the in-sample driver) -18.4 bps.

**The flow-continuation family is dead.** Fourth consecutive lead to fail on
fresh data. Per spec_forward.md: no further post-hoc flips. The forward
logger keeps running for the 2026-10-01 formality at zero cost; the research
line is closed. What stands: the participant-ecosystem dataset, the audited
pipeline, and a full-depth tape ingester that removes the 10k cap for any
future study.

## Vol study (2026-08-20, spec_vol.md — new data family, one locked run)

n=1,417 dissonance events on BTC-mapped markets (ETH: 0 — no market's first
mapped instrument is ETH). All three pre-registered bars FAIL:

- H-V1 median-quiet: 0.452 of events below hour-matched median RV (bar >0.50)
  — the quiet effect does NOT replicate on 1-minute realized vol.
- H-V2 VRP widening: boot-t -1.50 (bar >= +2.50) — variance risk premium
  NARROWS after events; DVOL falls at least as fast as realized vol.
- H-V3 straddle: excess-over-norm boot-t -1.50 (bar >= +2.0). Raw P&L is
  +7.4 bps/event with 68% hit, but it is BELOW the unconditional norm (i.e.
  it is the ordinary short-vol premium, not an event edge), and the tail
  ratio is 11.1 (bar <= 10): one -471 bps event vs +42 bps mean win.

**The volatility route closes too.** The 4h quiet effect measured on hourly
ETF/crypto bars does not survive on finer data, and where vol IS sellable the
event adds nothing over always-selling. AMIE's research phase ends here; the
ecosystem dataset, the audited pipeline, and the full-depth tape ingester are
what the project keeps.
