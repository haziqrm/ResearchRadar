# Live layer (Person D): how it works and what differs from the plan

This file records every place where D's outputs differ from what the plan, the
contracts or the sample files suggest. **Read "Differences from the plan" before using
D's files.** Kept up to date as the work changes.

## Pipeline

```bash
python -m src.live.run_live          # every stage, in order (also run by python main.py)
python -m src.live.keys              # check the Gemini and ElevenLabs keys
```

| Stage | Module | Writes | Committed? |
|---|---|---|---|
| fetch | `src/live/arxiv_feed.py` | `data/cache/arxiv/papers.csv` (+ raw XML) | No (cache) |
| tag | `src/live/tagging.py` | `data/arxiv_tagged.csv`, `data/tag_audit.csv` | Yes |
| crowding | `src/live/crowding.py` | `data/crowding.csv` | Yes |
| backtest | `src/live/backtest.py` | `data/backtest_results.csv`, `data/backtest_summary.csv`, `data/backtest_weights.csv` | Yes |
| briefing | `src/live/briefing.py` | `data/briefing.txt`, `.mp3` | Not written yet |

## Differences from the plan

### Backtest: built on real data, without decay_results.csv
- The plan said to write the backtest against the **sample** `decay_results.csv`, then
  rerun it on A's real file. A's real `factors.csv` and `papers.csv` were already in,
  so it went straight to real data.
- It does **not read `decay_results.csv`**. That file's `post_mean` averages every month
  after publication up to today, so trading on it would be look-ahead. Each month the
  backtest rebuilds everything from `factors.csv` and `papers.csv` using only past returns.
  (This is what the roadmap review suggested.)
- **Universe:** only signals whose paper was out by that month, and only once they have
  10 years of returns before publication. So it starts in **1981-04** (Size alone until
  1985) and runs to **2026-09**: 546 months.
- **Strategies** (`strategy` column). Each factor strategy appears twice: after costs
  (plain name) and before costs (`_gross`):
  - `decay_tilt`: expects each signal to keep **50%** of its pre-publication edge
    (`KEEP`, set in advance), then lets the signal's own post-publication record take
    over (weighted as if the prior were worth 60 months). Weights are expected edge
    after costs divided by variance.
    **Abstention rule:** skip any signal expected to earn under **0.1% a month after
    costs**. If every signal is skipped, hold cash (return 0; this happened in 79 of 546
    months, mostly 1981 to 1985).
  - `naive_insample`: believes the paper's numbers, with no decay.
  - `equal_weight`: 1/N over published signals. This is the main baseline.
  - `market`: Mkt-RF, with no costs.
- **Costs are a rough haircut.** Each signal has a fixed monthly cost per unit held
  (0.05% for size, value, profitability and investment; 0.10% for long-term reversal;
  0.20% for momentum; 0.60% for short-term reversal), roughly in line with
  Novy-Marx and Velikov (2016). Costs of moving between signals are not charged.
- All returns are **excess returns** (long-short factors, and the market minus the
  risk-free rate), so `market` is not directly comparable as an investable alternative.
  It is there for scale.

**Two extra D files** (not in the contracts):

| File | Columns | Use |
|---|---|---|
| `data/backtest_summary.csv` | strategy, sharpe, sharpe_ci_low, sharpe_ci_high, ann_return, ann_vol, max_drawdown, n_months, sharpe_minus_equal_weight, diff_ci_low, diff_ci_high | Results with intervals |
| `data/backtest_weights.csv` | date, signal, weight, expected_edge_net | `decay_tilt` weights per month (for the decision card) |

These are the review's proposed summary columns plus `ann_vol`, `n_months` and a paired
difference against `equal_weight` (each `_gross` strategy is compared with
`equal_weight_gross`). Sharpe intervals come from a 95% circular block bootstrap with
12-month blocks and 2,000 draws. `ann_return` is compound (CAGR).

**Results (2026-11-07):**

| Strategy | Sharpe [95% CI] | vs equal weight [95% CI] |
|---|---|---|
| decay_tilt | 0.03 [-0.29, 0.31] | +0.11 [-0.18, +0.40] |
| naive_insample | -0.21 [-0.56, 0.13] | **-0.13 [-0.23, -0.04]** |
| equal_weight | -0.08 [-0.43, 0.27] | baseline |
| decay_tilt_gross | 0.15 [-0.16, 0.43] | -0.12 [-0.40, +0.17] |
| equal_weight_gross | 0.27 [-0.08, 0.61] | baseline |
| market | 0.56 [0.23, 0.91] | |

How to read it:
- After 1981, these 7 published signals earn little, even before costs.
- Taking papers at face value is **reliably worse** than spreading evenly once costs
  are counted. The naive strategy leans on short-term reversal, whose costs eat its edge.
- The decay tilt beats equal weight after costs, but the interval includes zero, so
  this is suggestive only. Before costs it trails equal weight (also not significant).
- The tilt's result barely depends on `KEEP`: with 0.25, 0.5, 0.75 and 1.0 it beats
  equal weight by +0.11, +0.11, +0.06 and +0.01. Most of the gain comes from abstaining
  from high-cost signals, not from the size of the haircut.
- Latest weights (for 2026-09): Profitability 54%, Value 25%, Momentum 21%. Size,
  short-term reversal, long-term reversal and investment are skipped.

### arXiv volume is lower than estimated
- The plan and guide expected 5,000 to 10,000 q-fin papers over 12 to 24 months.
  The real number is **3,687 papers for Dec 2024 to 7 Nov 2026** (about 100 a month
  in late 2024, about 240 a month by late 2026), cross-lists included.
- Checked: arXiv's own reported total matches what we downloaded for every month,
  and no month needed a second page, so nothing is missing.
- `q-fin.EC` returns nothing; that category's papers now go to arXiv's `econ` categories.
- Consequence: about 35 papers a week, about 4 per area per week. That is why the
  crowding score uses a 4-week window (below).

### Crowding score is a z-score of share, not a ratio
- Plan: "this week versus own yearly average", i.e. a ratio around 1.
- Built: a **Poisson z-score** comparing the area's papers over the **last 4 weeks**
  with what its **share over the 52 weeks before** would predict, given all tagged
  papers in those 4 weeks.
- Why: q-fin as a whole more than doubled over the window, so a raw count or a ratio
  to the yearly average would make every area look crowded. With 4 papers a week,
  a 1-week ratio is mostly noise.
- **Reading it (for C):** 0 = the area's usual share; +2 = unusually crowded; -2 =
  unusually quiet. **Values can be negative.** Centre any colour scale on 0.
- `paper_count` is the area's count in **that one week**. The score uses 4 weeks, so a
  week with 2 papers can still score high. Label this in the dashboard.
- `date` is the **Monday** that starts the week. The latest week may be partial
  (the method handles that).
- The first 29 weeks have too little history and are dropped: `crowding.csv` starts
  on **2025-06-16**, not Dec 2024 (73 weeks to 2026-11-02).
- Settings live at the top of `crowding.py`: `WINDOW_WEEKS = 4`,
  `BASELINE_WEEKS = 52`, `MIN_BASELINE_WEEKS = 26`.
- The sample `data/sample/crowding.csv` has been updated to z-score values to match.

### Tags are keyword-based until the Gemini key is in
- Current `arxiv_tagged.csv` and `crowding.csv` come from the **keyword tagger**
  (`tagger` column = `keyword`). They will be regenerated with Gemini.
- The keyword tagger stays as a fallback for any paper Gemini hasn't tagged (quota,
  errors) and as a baseline to compare Gemini against. Each row says which tagger made it.
- `area` and `signal` are **empty strings** when nothing fits (631 of 3,687 papers have
  no area; most papers have no signal). These rows are left out of crowding.
- "Matching signal" uses only the 7 signals in `papers.csv` (Size, Value, Momentum,
  Short-term reversal, Long-term reversal, Profitability, Investment). If A adds
  signals, rerun tagging.
- Gemini model defaults to `gemini-flash-latest`; set `GEMINI_MODEL` in `.env` to change it.

### Two new D files that aren't in the contracts yet
| File | Columns | Use |
|---|---|---|
| `data/arxiv_tagged.csv` | arxiv_id, published, title, area, signal, confidence, tagger | Live feed of tagged papers for C |
| `data/tag_audit.csv` | arxiv_id, title, abstract, tagger, area, signal, correct_area, correct_signal | 50 papers checked by eye (D4) |

`arxiv_tagged.csv` should be added to `src/common/contracts.py` once C agrees the columns.

### Repo-wide changes D made
- `requirements.txt` is pinned. **pandas is 2.2.3**, not 3.x: `openassetpricing`'s
  dependencies cap it, so anyone on pandas 3 is moved down when they reinstall.
- `main.py` expects runner modules at `src.signal_decay.run_decay` (A),
  `src.research_map.run_map` (B) and `src.live.run_live` (D). B's name was chosen by D;
  change it in `main.py` if B prefers another.

## Tag accuracy (D4)

Fill `correct_area` / `correct_signal` in `data/tag_audit.csv` with `y` or `n`, then:

```bash
python -m src.live.tagging --score-audit
```

Result: _not measured yet_ (redo the sheet after the Gemini run).
