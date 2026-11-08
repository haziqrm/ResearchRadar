# Topic radar (Person A)

What should a quant researcher look into next? Built on B's map: 10,232 OpenAlex papers (a random 380 per year, 2000-2026), SPECTER2 vectors and 25 groups. The radar reads B's files and never changes them. The definitions and pass rules are in [HYPOTHESES.md](HYPOTHESES.md), written before the real run, including the amendments found by synthetic tests.

Run: `python -m src.topic_radar.run` (or `python main.py --only r`), using `.venv` (Python 3.12). It takes about 5 minutes and needs about 2 GB of free RAM.

## Outputs
| File | What |
|---|---|
| `data/area_forecasts.csv` | Contract file. Backtest rows per group per origin year (B's frozen group names), plus live rows for year 2025 with `actual_boom` blank. `year` is the origin year; the forecast covers the 3 years after it. |
| `data/topic_forecast_eval.csv` | H11 metrics: refit groups (the honest test) and frozen groups (optimistic). |
| `data/subtopics.csv` | 181 sub-topics: label, paper counts 2020-22 vs 2023-25, growth, Holm p, `rising`. |
| `data/seed_papers.csv` | The top 10 papers from 2023-2025 per group (reading list), with sub-topic label. |
| `data/seed_eval.csv` | H12 metrics. |

## Results (real run, 2026-11-08)

### H11: can we rank which groups grow next? **Fails, narrowly.**
Groups are refit each year on past papers only. Evaluated over 13 origins, 2010-2022:

| | Ridge | Momentum | Size | Random |
|---|---|---|---|---|
| Rank correlation with actual growth | **0.30** | 0.15 | 0.26 | 0 |
| Of the 5 actual booms, how many in the top 5 | **1.6** | 1.4 | 1.5 | 1.0 |

Pass tests against 200 simulated noise worlds (Holm-adjusted):
- Rank correlation vs random: noise worlds' 95th percentile is 0.09; **p = 0.015, passes**.
- Precision@5 vs random: **p = 0.050, fails** (needs < 0.05).
- Beating momentum: **p = 0.075, fails**.

Reading:
- There is real, if weak, predictable structure in topic growth: it is clearly beyond noise.
- Most of it is "big groups keep growing": size alone gets 0.26.
- The model does not reliably beat simple rules.
- B's frozen groups do worse (ridge -0.07), so the refit test matters. Momentum on frozen groups looks better (0.28), probably because those groups were fitted on later papers.

**Live 2025 forecast (top 5):**
1. Deep learning and LLM forecasting
2. Crypto, COVID and spillovers
3. Oil and energy prices
4. Behavioral finance
5. IPOs

Show it as **exploratory, not validated**.

### H13: the same forecast at sub-topic level (75 units)? **Fails, same pattern as H11.**
Units are refit each year on past papers only. Evaluated over 13 origins:

| | Ridge | Momentum | Size | Random |
|---|---|---|---|---|
| Rank correlation with actual growth | **0.13** | 0.08 | 0.18 | 0 |
| Of the 15 actual booms, how many in the top 15 | 3.5 | 2.9 | 3.5 | 3.0 |

Pass tests against 200 noise worlds (Holm):
- Rank correlation vs random: noise 95th percentile 0.06, **p = 0.015, passes**.
- Hits vs random: **p = 0.24, fails**.
- Beating momentum: **p = 0.24, fails**.

Reading:
- The signal is again real but weak, and "bigger keeps growing" (size) matches the model.
- A synthetic power check at the real sample size explains it: even a planted +8%/year trend gives about 2.9 hits vs 3 random. At about 5 papers per sub-topic per year, moderate trends are invisible.
- This needs more papers per year (a larger recent OpenAlex sample, or D's arXiv feed placed into B's groups), not a different model.

**Live 2025 sub-topic ranking (top 6, exploratory),** labelled by title terms. Paper counts are 2020-22 → 2023-25.

| # | Sub-topic | Parent topic | Papers |
|---|---|---|---|
| 1 | LSTM / deep forecasting | Deep learning and LLM forecasting | 36 → 94 |
| 2 | COVID connectedness / volatility | Crypto, COVID and spillovers | 55 → 81 |
| 3 | Bitcoin / crypto | Crypto, COVID and spillovers | 35 → 48 |
| 4 | Deep RL trading | Deep learning and LLM forecasting | 22 → 54 |
| 5 | Neural stock price prediction | Classical stock price prediction | 37 → 43 |
| 6 | Oil price shocks / inflation | Oil and energy prices | 32 → 27 |

Files: `data/subtopic_forecasts.csv` (all 75, with label, parent topic, counts and scores) and `data/subtopic_forecast_eval.csv`. The secondary ("optimistic") fit uses papers up to 2025; it excludes the partial year 2026, a small deviation from "all papers" in HYPOTHESES.md.

### Sub-topics: descriptive, **0 of 181 pass the strict "rising" test**
- Sub-topics have 10-40 papers each, and the test asks whether a sub-topic's share of *its own group* rose, with Holm across 181 tests. That is too strict for this sample size.
- The raw counts are still informative. For example, inside "Deep learning and LLM forecasting":
  - "deep forecasting frameworks": 0 → 22 papers;
  - "AI trading": 6 → 17;
  - "deep RL portfolios": 11 → 10, flat.
- Show the counts, labelled "papers 2020-22 vs 2023-25"; don't claim significance.

### H12: can we pick papers that later work clusters around? **Fails against density.**
Precision of the top 20 picks, 2013-2022 (random = 10%):

| Ridge | Density alone | Newest first | Random |
|---|---|---|---|
| 91% | **93%** | 15% | 10% |

- On the 4 non-overlapping origins, ridge vs random has p < 0.001, but ridge vs density has p = 0.63.
- The seed score as defined (the number of later papers that are very similar) is almost entirely "this paper sits in a busy area".
- So the reading list is honestly **"the most typical recent papers of a busy area"**, not "the papers that will be influential". A better target would need citations over time (OpenAlex `counts_by_year`), not just similarity.

## Lessons recorded
1. **The first H11 test could be passed by pure noise.** Growth divided by a noisy current share makes lucky groups "shrink back". It is fixed by the split-half design: features from half of each year's papers, the target's base from the other half.
2. **The block bootstrap gave 17-27% false alarms** at a nominal 5% with about 15 overlapping origins. It was replaced by simulated noise worlds (H11) and permutation tests on non-overlapping origins (H12).
3. **The first H12 density test was unfair:** shuffling inside deciles also destroyed density's own fine ordering. It was replaced by shuffling only the part of the score that density does not explain.

## Caveats
- SPECTER2 was pretrained on citation links, so older papers' vectors may carry slight hindsight.
- Only titles are available locally (no abstracts), so sub-topic labels come from titles.
- 2026 is a partial year and is never a target.
