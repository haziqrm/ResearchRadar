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
| map_tags | `src/live/map_tags.py` | `data/arxiv_map_tags.csv` (Gemini cache in `data/cache/arxiv/`) | Yes (complete, see below) |
| crowding | `src/live/crowding.py` | `data/crowding.csv`, `data/crowding_meta.json` | Yes |
| backtest | `src/live/backtest.py` | `data/backtest_results.csv`, `data/backtest_summary.csv`, `data/backtest_weights.csv` | Yes |
| briefing | `src/live/briefing.py` | `data/briefing.txt`, `data/briefing.mp3`, `data/briefing_card.json` | Yes |

`src/live/tagging.py` (D's original 8 areas plus the paper -> signal match) is no longer a
stage. Its committed outputs `arxiv_tagged.csv` and `tag_audit.csv` are kept as they were.

On the website, D's code is `live_data.py` (standard library only), `static/js/live.js`
and `static/css/live.css`. It draws the **headline** above the map, **Where research is
heading**, the one-line lead on the decay panel, the **?** drawer and the **Weekly
briefing** modal. It is served at `/api/heading`, `/api/crowding/history`,
`/api/crowding`, `/api/backtest`, `/api/briefing` and `/media/briefing.mp3`. The page
layout is described in "The page is built around the map" below.

## Differences from the plan

### The page is built around the map (2026-11-08)
The plan had a dashboard of side-by-side panels: D's crowding table, the backtest and
the briefing next to C's map and A's radar. The page now leads with the map and keeps
everything else one click away.

**Layout, top to bottom** (`html_main.py`):
- **Header.** The title, a one-line **headline** from the briefing card ("2026 so far:
  Deep learning and LLM forecasting is the most crowded area (+4.8); Classical stock
  price prediction is the quietest (−2.5)."), a **Weekly briefing** button and a **?**
  button.
- **The map** (C's code, inline in `html_main.py`) fills the first screen at full width
  and redraws through a `ResizeObserver`. Circles are coloured by crowding for the
  slider's year, using `crowdColour()` and `/api/crowding/history`. **Inside a topic**
  (A's `radar.js`) moves into `#dock` beside the map only while a topic is open
  (`body.topic-open`); otherwise the map has the full width.
- **Where research is heading** (D, `/api/heading`, drawn by `live.js`). It has one
  row per map topic, sorted by crowding:
  - **Crowding now:** the latest score, shaded with the map's colours.
  - **Forecast:** A's 25-topic growth forecast re-ranked among the 8 map topics ("#2 of
    8"). "likely to grow" marks A's predicted top 5. The column is badged
    **exploratory** because test H11 did not pass. The overall rank is in the tooltip.
  - **Read first:** the topic's most cited paper in the latest crowding year.
  - Rows carry `data-topic`. `radar.js` opens any `[data-topic]` element on the map
    when clicked; Enter and Space work too.
- **What happens to a published edge** (A's decay chart and haircut calculator, drawn
  by `radar.js`). D's `live.js` adds the one-line lead, computed from `/api/signals`:
  about half kept on average, and the typical signal's share.
- **What could be next** (A's topic and sub-topic forecasts) shows only with `?full` in
  the URL. It is an internal view, not part of the demo.

**Numbers made to agree, names made readable (2026-11-08):**
- **One rank denominator, always labelled.** Three places ranked the same forecast
  against different, unlabelled totals: the drill-down said "rank N of" the number of
  topics in its dropdown (25), the heading table re-ranked among the 8 map topics, and
  the guide hard-coded "top 5 of 25". Now:
  - **Topic ranks are among the 8 topics on the map**, in the heading table ("#2 of 8")
    and the drill-down ("Forecast rank 2 of 8 on the map"). Both read
    `radar_data.map_rank()`, which orders the map's topics by the forecast's own rank.
    `get_forecasts()` adds `map_rank` per topic and `n_map`.
  - A topic **off the map** (opened from the dropdown or a "builds on" link) has no map
    rank, so the drill-down says "Forecast rank N of all 25 (not on the map)".
  - **"Predicted to grow"** is still the forecast's own top 5 of all 25 topics (that is
    how the model picks them), and every place says so: the legend, the guide and the
    table tooltip, which also gives the overall rank ("#1 of the 8 topics on the map
    (#1 of all 25 forecast topics)").
  - Every total comes from `forecasts.json` (`n_topics`, `n_map`, `top_topics`,
    `n_subtopics`, `top_subtopics`), kept in `fcCount` in `radar.js`, which also fills
    the guide's `.fc-top` / `.fc-n` spans. Nothing is hard-coded.
  - `/api/heading`'s `forecast` is `{rank, of, rank_all, of_all, top, likely}`. "likely
    to grow" now reads "predicted to grow", as everywhere else.
- **Crowding has a unit:** scores show as "+4.8 SD" (standard deviations vs the topic's
  usual share) in the headline, the table, the briefing chips and the map tooltip.
- **"Kept" over 100%** is explained under the signals table: over 100% means the signal
  earned more after publication than the paper reported, below 0% that it lost money.
  The decay chart labels its 100% line "the return the paper reported".
- **Plurals:** "1 citation", "1 paper", "1 article", "1 paper pair … cites".
- **Readable sub-topic names.** The export labels sub-topics with keyword lists
  ("prediction, lstm, deep, forecasting"). `data/subtopic_names.csv` gives each of the
  75 a name (e.g. "Deep learning stock forecasting (LSTM)"). **How they were checked:**
  each name was written after reading 9 of the sub-topic's paper titles (its suggested
  reading first, then a random draw, seed 1), not from the keywords alone. Where the
  papers are mixed the name says so or stays broad: 53 is "Fractal time-series methods
  (mostly outside finance)", 0 is "Investing strategies and wealth management". The
  keywords are still in the data (`label`) and show as the row's tooltip.
  `radar_data.py` adds `name` on the way out, so the export doesn't need rerunning.
  **If A re-clusters, the ids change and this file must be rewritten.**
- **Readable signal names.** Codes like `MomOffSeason11YrPlus` now show OSAP's own
  description from its signal documentation ("Off season reversal years 11 to 15",
  Heston and Sadka 2008), written to `data/signal_names.csv` from the cached
  `signal_doc.parquet` (195 OSAP codes; the 7 Ken French factors already had names).
  Nothing was named by hand. The code shows on hover.
- **"Read first" picks a cited paper.** It used to be the topic's most cited paper in
  2026, which picked papers with 0 to 2 citations (for example a 0-citation paper on AI
  trading in Indonesia). It is now the **most cited per year since publication among
  papers from the last five years** (2022–2026, `READ_FIRST_YEARS`). Dividing by age
  stops both brand-new and decades-old papers from winning on age alone. Examples:
  Deep learning and LLM forecasting went from a 12-citation 2026 paper to a 2022 review
  with 246 citations; Information and stock reactions from 2 citations to
  "Information in Financial Markets and Its Real Effects" (2022, 249). The briefing's
  example paper is unchanged (still this year's most cited).

**Map changes after the restructure (2026-11-08):**
- A's dashed **"predicted to grow" rings** used to show on every year of the slider, but
  the forecast is made from 2025 data (`forecasts.json` `year`) about the next three
  years. They now show only when the slider is on 2025 or later (`syncRings` in
  `radar.js`). They are **navy** (the site's brand colour), not oxblood, and are in the
  map legend as "predicted to grow".
- **Inside a topic was too cramped** with the dock open: `scales()` in `radar.js` fitted
  the papers into a fixed square in the middle of the box, so islands and labels piled
  up with empty space either side. It now fits the papers' own extent to the box, with
  80px at each side for labels, and lets one axis stretch up to 1.6 times the other. At
  1440px wide the 14 island labels for "Information and stock reactions" no longer
  overlap. The dock is wider on big screens (`clamp(340px, 27vw, 440px)`, was 340px)
  and its sub-topic rows have more room.
- The header buttons were restyled: **Weekly briefing** has a headphones icon and a
  pulsing "new" dot (off under `prefers-reduced-motion`), and **?** is now an outlined
  **Guide** button with a question-mark icon (icon only under 560px).

**Dialogs:**
- **? drawer:** how to read the map (size, lines, colour, dashed ring, islands), A's
  test cards ("How we tested it") and "Data and caveats".
- **Weekly briefing modal:** the audio, the date, the 3 most crowded areas and the
  quietest, next month's holdings and skipped signals, and the caveat. Collapsed under
  it are **Does trading on decay help?** (backtest chart and table), **Signals right
  now** (A's `/api/signals`) and the briefing text.
- A **"new" dot** on the briefing button shows when the card's `generated` date is newer
  than `localStorage["radar-briefing-seen"]`. Opening the modal records the date. If
  storage is blocked, the dot simply shows.

**New endpoint `/api/heading`** (`live_data.get_heading`) returns
`{period_start, unit, forecast_passed, topics: [{topic, score, papers, forecast: {rank,
of, likely, rank_all, of_all} | null, read_first}]}`. It joins `get_crowding()` with
A's `data/app/forecasts.json`; `forecast_passed` is H11's `passed` flag.

**`briefing_card.json` has a new `generated` field** (ISO date the card was written, set
by `src/live/briefing.py`). The "new" dot and the modal's "Week of …" label read it.

**Cut from the page:**
- The side **Research topics** table, which repeated what the map and the heading
  table show.
- **Show affected path** on the map.
- D's separate **Crowded in <year>** panel. Its content is now the heading table and
  the map colours. `/api/crowding` is still served but the page no longer fetches it
  directly.
- The backtest and signals are no longer panels on the page; they are collapsed inside
  the briefing modal.

**Why the headline says "2026 so far", not "This week":** crowding is now scored by
calendar year from OpenAlex (see the next section), and 2026 is a partial year. `live.js`
says "<year> so far" when `crowding_meta.json` has `unit = "year"` and "This week"
when it has `"week"`. Once the arXiv tagging is complete and crowding switches to
`--source arxiv`, the headline goes back to "This week" without code changes.

**Known mismatch:** the button and modal are still called **Weekly briefing**, and the
modal date reads "Week of <Monday of `generated`>", while the crowding inside it is for
the year so far. The modal also says "crowding for 2026 so far" next to the date. This
is left as it is because the briefing becomes weekly again with arXiv.

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
  cache. **All 3,687 papers were tagged later on 2026-11-08** (three keys, with
  gemini-3.5-flash, 3.7-flash and 3.8-flash tagging in parallel). The CSV is committed.
  3.7 and 3.8 tagged 1,338 of them and weren't in the 50-paper accuracy check.
  `crowding.csv` still uses the OpenAlex source. Switching to `--source arxiv` is a
  separate decision.
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

### Monthly map at /monthly (2026-11-08)
- **What it is.** `/monthly` is the main page with the map switched to arXiv papers by
  calendar month: the papers Gemini tagged into the 8 map areas
  (`data/arxiv_map_tags.csv`), Dec 2024 to now. `monthly_data.py` serves
  `/api/network/monthly` and `/api/crowding/history/monthly`. The main page (`/`) is
  unchanged. It's an alias for now, so the team can compare the two views before
  deciding which one the main page uses.
- **What changes and what doesn't.** On `/monthly` the slider steps through months, and
  the newest month is marked partial, e.g. "Nov 2026 (to 5 Nov)". The slider starts on
  the last complete month. Circle size is the area's papers that month, on one scale
  across months. Colour is crowding with `src.live.crowding`'s method, using a 1-month
  window against the 6 months before (`monthly_data.BASELINE`). There's no score for
  the first 3 months. Everything else on the page (sub-topics, rings, "Where research
  is heading", briefing) stays OpenAlex by year and follows the month's year.
- **Links are still OpenAlex's, over all years.** arXiv gives no reference lists, so the
  links are the same in every month, and the tooltip says so.
- **Deviation: "none" vs not-yet-tagged.** The tags CSV merged in PR #6 had a blank area
  for papers Gemini put in "none". Main's `map_tags` reads a blank as "not tagged yet",
  so crowding and this view would have left out 2,175 papers. The CSV now says `none`
  for them. No other column changed.
- **Why the map's JS changed.** The slider used to step through years by adding 1. It
  now steps through the positions in `slider_years`, so the same code handles
  "YYYY-MM". `MAP_MODE` in the page picks the endpoints. On `/` it behaves as before.

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
