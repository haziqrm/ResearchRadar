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
| map_tags | `src/live/map_tags.py` | `data/arxiv_map_tags.csv` (Gemini cache in `data/cache/arxiv/`) | Not yet (partial, see below) |
| crowding | `src/live/crowding.py` | `data/crowding.csv`, `data/crowding_meta.json` | Yes |
| backtest | `src/live/backtest.py` | `data/backtest_results.csv`, `data/backtest_summary.csv`, `data/backtest_weights.csv` | Yes |
| briefing | `src/live/briefing.py` | `data/briefing.txt`, `data/briefing.mp3`, `data/briefing_card.json` | Yes |

`src/live/tagging.py` (D's original 8 areas plus the paper -> signal match) is no longer a
stage. Its committed outputs `arxiv_tagged.csv` and `tag_audit.csv` are kept as they were.

On the website, D's panels are `live_data.py` (standard library only), `static/js/live.js`
and `static/css/live.css`: **Crowded in <year>**, **Does trading on decay help?** and the
**Briefing**, served at `/api/crowding`, `/api/backtest`, `/api/briefing` and
`/media/briefing.mp3`.

## Differences from the plan

### Crowding now uses B's areas and OpenAlex, by year (2026-11-08)
- **Why it changed:** the shared area list in `contracts.py` is now B's 25 groups
  (`groups.csv`). D's old crowding used D's own 8 areas ("Factor and anomaly research"
  and so on), so `crowding.csv` no longer passed `contracts.validate()`.
- **Areas:** the 8 B areas the team chose for the site (`src.live.map_tags.MAP_AREAS`,
  checked by a test to equal `research_data.AREAS`). Shares are of **all** papers: the
  other 17 B areas count towards the total only.
- **Source, for now: OpenAlex.** `python -m src.live.crowding` reads B's
  `network_nodes.csv` (380 sampled papers a year). One period is a **calendar year**:
  the area's share this year vs its share over the **5 years before** (at least 3), same
  Poisson z-score as before. `date` is `YYYY-01-01`; `crowding.csv` covers 2003-2026.
  2026 is the year so far (353 of 380 papers sampled).
- **It is not weekly any more.** 380 papers a year is about 0.3 per area per week, too
  few for a weekly score. The weekly view comes back with arXiv (below).
- `data/crowding_meta.json` (new, not a contract) says which source and period
  `crowding.csv` came from (`source`, `unit`, `window`, `baseline`). The briefing and
  the site read it to say "2026 so far" or "this week".
- **arXiv later:** `python -m src.live.crowding --source arxiv` scores weekly from
  `data/arxiv_map_tags.csv` (4-week window vs 52 weeks, as before). It needs every
  arXiv paper tagged into the 8 map areas first. `map_tags` keeps `area = "none"` for
  "none of the 8" and `""` for not tagged yet; untagged papers are left out.
- **Gemini quota:** the key is on the **free tier, 20 requests a day** (50 papers per
  request). On 2026-11-08, 308 of 3,687 papers were tagged before the daily limit. At
  1,000 papers a day the rest takes about 4 days, or minutes with billing on.
  `map_tags` now stops cleanly at the daily limit and the next run carries on from the
  cache. `arxiv_map_tags.csv` is not committed until the tagging is complete.
- **Briefing:** the card now has `period_start`, `unit`, `window`, `source`, and
  `papers` (was `week` and `papers_4w`); `taggers` is gone. The example paper is the
  area's **most cited paper this year** (OpenAlex) or its newest one (arXiv). The text
  says "for 2026 so far, from a sample of OpenAlex quant-finance papers". Regenerated on
  2026-11-08 with the template text (Gemini over quota) and new ElevenLabs audio.
- **Backtest:** rerun on 2026-11-08; inputs (A's `factors.csv`, `papers.csv`) hadn't
  changed, and neither did the results.
- The tag accuracy (92% area, 50 papers) was measured on D's old 8 areas, **not** on
  the map areas; it hasn't been rechecked.
- The sections below describe the earlier weekly arXiv version where they mention
  weeks.

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

### Tags come from Gemini (all 3,687 papers)
- `arxiv_tagged.csv` and `crowding.csv` now come from **Gemini** for every paper
  (`tagger` = `gemini`; finished 2026-11-07). Mean confidence 0.91.
- The keyword tagger stays as a fallback for any paper Gemini hasn't tagged (quota,
  errors) and as a baseline. It agrees with Gemini on the area for only **2,301 of
  3,687 papers (62%)**, so keyword-tagged rows should be treated as rough.
- `area` and `signal` are **empty strings** when nothing fits: **302 papers have no
  area**, and **3,619 have no signal**. These rows are left out of crowding.
- Areas: Portfolio and risk 1,101; Derivatives and volatility 653; Market microstructure
  400; Crypto and DeFi 346; Machine learning for returns 341; LLMs in finance 317;
  Factor and anomaly research 124; Alternative data and text 103.
- Signals: Momentum 45; Size 9; Short-term reversal 8; Value 4; Profitability 1;
  Investment 1; Long-term reversal 0. **Signal counts are far too small for crowding by
  signal** (the "Improvements" idea): only momentum has more than a handful.
- Crowding from the Gemini tags: scores run from -2.53 to +3.54, and 7.4% of area-weeks
  are beyond ±2. Week of 2026-11-02: Market microstructure +3.03 (highest), Machine
  learning for returns -1.57 (lowest).
- "Matching signal" uses only the 7 signals in `papers.csv` (Size, Value, Momentum,
  Short-term reversal, Long-term reversal, Profitability, Investment). If A adds
  signals, rerun tagging.
- Gemini model is **`gemini-3.5-flash`**. Set `GEMINI_MODEL` in `.env` to change it.
  A wrong model name or bad key stops the run instead of silently falling back to keywords.
  A **403 "permission denied"** is retried instead: it appeared once mid-run on a
  working key and cleared on its own.
- Each cached answer records its model, and only the current model's answers are reused,
  so all Gemini tags come from one model (no mix).

### Why gemini-3.5-flash (model comparison, 2026-11-07)
`gemini-2.5-flash` is closed to new accounts and the `gemini-flash-latest` alias isn't
offered. `gemini-3.8-flash` (the newest) was overloaded: most requests returned
"503 high demand", which would have made the run take about 4.5 hours.

The same 50 papers (oldest month) went through each model; D judged each paper by eye:

| Model | Clear errors in 50 | Time per 25 papers |
|---|---|---|
| gemini-3.5-flash | 0 | ~21 s |
| gemini-3.8-flash | 2 | 156 s (overloaded) |
| gemini-3.5-flash-lite | 3 | ~3 s |
| gemini-3.1-flash-lite | 4 | ~3 s |
| keyword rules | ~16 | instant |

The four Gemini models gave the same area on 42 of 50. The lite models forced off-topic
papers (for example a smart-meter energy market) into an area instead of leaving them
empty, which would inflate crowding. This is a quick check on 50 papers, not the D4
accuracy figure, and none of the 50 had a signal, so it says nothing about signal tags.

### Free-tier limit and batch size
- The Gemini free tier allows only **20 requests a day per model** for this model (the
  plan assumed limits were not a problem). The run stops cleanly with "gemini quota
  reached"; tags so far stay cached, and untagged papers get keyword tags until a rerun.
  The team switched free keys when a key ran out.
- To need fewer requests, the batch went from 25 to **50 abstracts per request**
  (`--batch-size`, default 50). Tested against papers already tagged in batches of 25:

  | Batch size | Same area as batch of 25 | Same signal | Clearly worse tags |
  |---|---|---|---|
  | 50 | 46/50 (92%) | 50/50 | 1 (an off-topic paper put into an area) |
  | 75 (run twice) | 66/75 and 67/75 | 74-75/75 | 4 (more off-topic papers put into areas) |

  Rerunning the same batch also changes a few borderline tags, so some of this is noise.
- The first 474 papers were tagged in batches of 25, the rest in batches of 50.

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

### Briefing: numbers come from code, Gemini only rewords
- Plan: "Gemini text, ElevenLabs audio". Built: **code computes every fact**, a fixed
  template turns them into text, and Gemini (optional) only rewords it to sound natural
  aloud. A rewrite is **thrown away unless it has exactly the same numbers** as the
  template, so the model can't invent or change a figure. `text_source` in the card
  says which text was used.
- Content (the review's "decision card", written to `data/briefing_card.json` for C):
  the 3 busiest areas this week and the quietest one, the newest tagged paper in the
  busiest area, **next month's allocation**, the skipped signals, and one caveat
  (decay_tilt vs equal weight, with its interval).
- The allocation applies the backtest's `decay_tilt` rule to **every return in
  `factors.csv`**, so it is one month beyond the last backtest month. Ken French data
  ends 2026-09, so the briefing gives the allocation for **October 2026**, not November.
  It can differ from the last row of `backtest_weights.csv` (Profitability falls just
  under the 0.1% hurdle once September is included).
- The template is written to be **heard**, in a conversational style ("So, what
  would we hold for October?", "One honest caveat, though."), so it sounds natural
  even without Gemini.
- Audio: ElevenLabs voice "Brian" (`nPczCjzI2devNBz1zQrb`), model `eleven_v4`. These were
  picked by ear on 2026-11-07 from Daniel and Brian on `eleven_multilingual_v2`,
  `eleven_v3` and `eleven_v4`; `eleven_v4` with the spoken-style text sounded most like
  a real person. Change them with `ELEVENLABS_VOICE_ID` / `ELEVENLABS_MODEL`
  in `.env`. Audio is only regenerated when the text changes. With no key, or if the call
  fails, the **committed `briefing.mp3` is the pre-recorded fallback**, so the demo never
  needs a live API. Use `--no-gemini` / `--no-audio` to skip either service.
- **Current files (2026-11-07, 23:24):** made from the Gemini-tagged `crowding.csv`,
  reworded by `gemini-3.5-flash` (the rewrite kept every number), audio regenerated.
  Week of 2 November: Market microstructure most crowded (+3.0, 34 papers in 4 weeks);
  October 2026 allocation Value 52%, Momentum 48%.
- **Wording fixed (2026-11-07):**
  - The briefing says "most crowded **compared with its usual share**", never
    "busiest": the area with the most papers can score low (Portfolio and risk had
    49 papers in 4 weeks but scores -1.3). When the area with the most papers isn't
    the most crowded one, the briefing names it "for scale". The card has a new
    `most_papers` key; `busiest` / `quietest` keep their names (ranked by score).
  - Spoken numbers round half up (0.35 -> 0.4, 52.5% -> 53%), not with Python's
    `round` (which gave 0.3).
  - A Gemini rewrite containing "busiest" is rejected, like one with changed numbers.
- The text is now about 200 words (~80 seconds), over the prompt's 60-second target.

## Tag accuracy (D4)

Fill `correct_area` / `correct_signal` in `data/tag_audit.csv` with `y` or `n`, then:

```bash
python -m src.live.tagging --score-audit
```

Result (checked 2026-11-07, 50 random Gemini-tagged papers):

```
area: 46 of 50 correct (92%)
signal: 50 of 50 correct (100%)
```

- **Area: 92%.** This is the figure to quote. The four misses (2607.02695, 2511.05030,
  2508.03790, 2602.00383) are theory or topology papers that don't fit any of the eight
  areas well, so Gemini forced one in. 2601.11348 (carbon abatement) has no area, which
  counts as correct.
- **Signal: don't quote the 100% on its own.** All 50 sampled papers had no signal, because
  only 68 of 3,687 papers have one. So the figure only shows that Gemini doesn't add a
  factor signal where there isn't one. Whether those 68 signal tags are right hasn't been
  checked.
