# Signal decay (Person A)

How much do published trading signals stop working, is the drop caused by
publication, and can we tell in advance which signals will fade?

All tests, thresholds and exclusion rules were written down before results were
computed: see [HYPOTHESES.md](HYPOTHESES.md).

## Run it

```bash
python -m src.signal_decay.run_decay --charts     # 7 Ken French signals -> contract files
python -m src.signal_decay.run_osap --charts      # 212 OSAP signals -> data/osap_*.csv
python -m src.signal_decay.decay_model --charts   # decay predictor (H7) -> data/osap_model_*.csv
python -m src.signal_decay.decay_model_h8 --charts  # second attempt (H8) -> data/osap_h8_*.csv
python -m src.signal_decay.forecasts --charts       # month-ahead forecasts (H9) -> data/osap_h9_summary.csv, data/signal_forecasts.csv
```

Citations need `OPENALEX_API_KEY` and `OPENALEX_EMAIL` in `.env`. Every OpenAlex and
OSAP download is cached in `data/cache/`, so reruns are fast and offline.

## Results

| # | Question | Data | Result | Verdict |
|---|---|---|---|---|
| H1 | Do returns fall after publication? | 7 Ken French | All 7 fall, about 4 points a year on average; only short-term reversal is individually clear | Supported, weakly |
| H2 | Is the drop tied to the publication date? | 7 Ken French | Placebo p = 0.07 | Weakly supported |
| H3 | More early attention, more decay? | 7 Ken French | rho = 0.44, p = 0.16 | Not supported |
| H4 | Returns fall after the sample, and again after publication? | 209 OSAP | 100% in-sample, 74% after sample, 44% after publication; extra drop at publication p = 0.03 | Supported (replicates McLean & Pontiff 2016) |
| H5 | Is the drop tied to the publication date? | 196 OSAP | Real drop about twice the fake-date drop, p < 0.0001 | Strongly supported |
| H6 | More early attention, more decay? | 111 OSAP papers | rho = 0.11, p = 0.12 | Not supported |
| H7 | Can decay be predicted at publication? | 127 train / 81 test | Ridge and random forest both fail to beat "keep 37%" on 2007-2016 papers | Not supported |
| H8 | Do trading-cost, novelty and crowding inputs help? | 181 test signals over 3 walk-forward folds | Pooled error 0.447 vs baseline 0.435; ridge is slightly *worse* (CI +0.002 to +0.023) | Not supported |
| H9a | Do recent winners among published signals keep winning next month? | 195+ OSAP, 479 months | Past-year winners earn 0.64%/month vs about 0% for the rest, t = 4.1 | Supported |
| H9b | Does recent volatility predict next month's better than long-run volatility? | 624 months | Lower forecast error, t = 4.7; volatility-managed Sharpe 1.55 vs 1.01 (before costs) | Supported |

**In one line:** published signals keep roughly 40-45% of their in-sample edge, part
of the drop is caused by publication itself, and neither attention nor other features
known at publication predict which signals fade more, even with trading-cost, novelty
and crowding inputs (H8). What *is* predictable is month-ahead: published signals that
won over the past year keep winning, and recent volatility forecasts next month's (H9).

**Evaluation audit.** H7 and H8 were checked for evaluation bugs (same target and test
rows for every model, no test data in the baseline, time split by paper, a synthetic
signal recovered exactly). None was found. Afterwards the early-citation feature was
dropped (it counts citations after publication), and squared error, a training-mean
constant and a median regression were added. No verdict changed: on 2007-2016 papers,
ridge MAE is 0.368 and the median regression 0.437, against 0.366 for the baseline. See
the correction note at the end of [HYPOTHESES.md](HYPOTHESES.md).

Exploratory, not pre-registered: across the 181 H8 test signals, signals whose edge
holds up in big stocks kept *more* of it (Spearman 0.15, p = 0.04, one of three
uncorrected checks), the opposite of the H8 expectation. Treat it as a lead, not a result.

## Output files

| File | Contents |
|---|---|
| `data/factors.csv`, `data/papers.csv`, `data/decay_results.csv` | Shared contract files (7 Ken French signals) |
| `data/decay_details.csv`, `data/signal_decay_summary.csv` | Extra Ken French stats, H2 and H3 |
| `data/paper_ids.csv` | Hand-checked OpenAlex IDs for the 7 papers |
| `data/osap_decay_results.csv`, `data/osap_papers.csv`, `data/osap_summary.csv` | OSAP per signal, per paper, H4-H6 |
| `data/osap_model_summary.csv`, `_predictions.csv`, `_importance.csv` | H7 decay predictor |
| `data/osap_h8_summary.csv`, `_predictions.csv`, `_features.csv`, `_coefficients.csv` | H8 walk-forward predictor and its new inputs |
| `data/osap_h9_summary.csv` | H9 tests and descriptive Sharpe ratios |
| `data/signal_forecasts.csv` | Latest month-ahead forecast per signal (OSAP and the 7 Ken French signals, with D's signal names in `name`) |

## Limits

- **Market eras are mixed up with decay.** Trading got cheaper after the 1990s, value
  had a long bad run after 2007 and momentum crashed in 2009, mostly after the papers
  came out. The placebo tests (H2, H5) check that the drop is tied to publication
  dates rather than to time alone, but cannot rule out every era effect.
- **Ken French factors are built with hindsight.** They use today's definitions, not
  each paper's original recipe, so "before publication" is not exactly what authors saw.
  OSAP's original-paper portfolios are closer to the papers.
- **Publication dates are approximate.** Ken French rows use the journal issue month;
  OSAP gives the year only, so post-publication starts in January of the next year.
  Working-paper versions often circulated earlier.
- **Half-lives are too noisy to report.** None of the 7 passes the bootstrap and
  no-decay checks, so `half_life_months` is left empty rather than filled with a guess.
- **Attention is a proxy.** Citations measure academic attention, not money traded.
  29 of 141 OSAP papers could not be matched to OpenAlex automatically and are left
  out of H6 rather than guessed.
- **Selection.** OSAP covers signals that were published and later replicated; signals
  that never got published, or got dropped, are missing.
- **No trading costs.** Returns are before costs. H9's momentum and volatility-managed
  portfolios trade more and change leverage, so their edge after costs would be smaller. High-turnover signals such as
  short-term reversal would earn less in practice.
- **Small samples.** 7 signals for H1-H3; about 130 training signals for H7, which is
  little for any model. H8's first walk-forward fold trains on only 27 signals.
- **H8 reuses the H7 test years.** The 2007-2016 fold was already seen in H7; the H8
  verdict uses all three folds pooled so it does not rest on that fold alone. Nulls here mean "not detectable with this data", not "proven absent".
