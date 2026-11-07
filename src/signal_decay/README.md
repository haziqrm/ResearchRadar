# Signal decay (Person A)

How much do published trading signals stop working, is the drop caused by
publication, and can we tell in advance which signals will fade?

All tests, thresholds and exclusion rules were written down before results were
computed: see [HYPOTHESES.md](HYPOTHESES.md).

## Run it

```bash
python -m src.signal_decay.run_decay --charts     # 7 Ken French signals -> contract files
python -m src.signal_decay.run_osap --charts      # 212 OSAP signals -> data/osap_*.csv
python -m src.signal_decay.decay_model --charts   # decay predictor -> data/osap_model_*.csv
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

**In one line:** published signals keep roughly 40-45% of their in-sample edge, part
of the drop is caused by publication itself, and neither attention nor other features
known at publication predict which signals fade more.

## Output files

| File | Contents |
|---|---|
| `data/factors.csv`, `data/papers.csv`, `data/decay_results.csv` | Shared contract files (7 Ken French signals) |
| `data/decay_details.csv`, `data/signal_decay_summary.csv` | Extra Ken French stats, H2 and H3 |
| `data/paper_ids.csv` | Hand-checked OpenAlex IDs for the 7 papers |
| `data/osap_decay_results.csv`, `data/osap_papers.csv`, `data/osap_summary.csv` | OSAP per signal, per paper, H4-H6 |
| `data/osap_model_summary.csv`, `_predictions.csv`, `_importance.csv` | H7 decay predictor |

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
- **No trading costs.** Returns are before costs. High-turnover signals such as
  short-term reversal would earn less in practice.
- **Small samples.** 7 signals for H1-H3; about 130 training signals for H7, which is
  little for any model. Nulls here mean "not detectable with this data", not "proven absent".
