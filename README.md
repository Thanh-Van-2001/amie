# AMIE — Acoustic Market Intelligence

Research codebase exploring whether the behaviour of prediction-market
participants contains **early signals for liquid financial markets**
(index futures, commodities, rates, FX, single stocks, crypto).

The working metaphor: a crowded room with one exit. Before the crowd rushes
the door, someone screams. Prediction-market wallets — especially informed
ones — are the scream; the move in liquid markets is the rush. This project
builds the instruments to hear the scream first.

## Approach

1. **Universe** — Polymarket markets filtered to themes that can plausibly
   *lead* liquid markets: geopolitics, macro/economics, weather, earnings,
   crypto. Sports, culture, and mentions markets are excluded by design.
2. **Participant ecosystem** — a registry of every wallet active in that
   universe: notional, breadth, timing, track record. Who is in the room,
   and who reacts first.
3. **Acoustic encoding** — wallets become oscillators (size → amplitude,
   direction → frequency band, trade timing → phase); the composite field
   per market yields psychoacoustic features (loudness, dissonance,
   rhythmic entropy, spectral centroid, cross-market resonance).
4. **Validation** — event studies of feature threshold-crossings against
   *forward returns of mapped external instruments* (see
   `amie/mapping.yaml`), with strict lag-1 execution, train/test split,
   pre-registered thresholds, and plain-flow baseline comparisons.

No live trading. This is signal research on public data.

## Layout

```
config.yaml            universe filters, API endpoints, ingest caps
amie/mapping.yaml      Polymarket theme -> liquid instrument proxies
amie/ingest/
  universe.py          Day 1: filtered market universe -> universe.parquet
  prices.py            Day 1: CLOB hourly price history per market
  trades.py            Day 2: full trade tape per market, resume-safe
  participants.py      Day 3: wallet registry + top-holder snapshots
data/                  parquet outputs (gitignored)
```

## Run

```bash
pip install -r requirements.txt
python -m amie.ingest.universe
python -m amie.ingest.prices
python -m amie.ingest.trades
python -m amie.ingest.participants
```

All data sources are free public APIs (Polymarket Gamma / Data-API / CLOB).
