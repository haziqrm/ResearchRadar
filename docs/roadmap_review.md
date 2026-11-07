# Proposed roadmap changes

These are proposals; nothing is agreed yet. Items marked **(team)** touch shared files or other people's work and need agreement first. P0 means decide before anyone pulls data.

## Before starting
- **(team, P0)** Confirm the deadline, the judging criteria for the G-Research and HRT tracks, whether the demo is live or recorded, and each person's skill level.

## Shared section (edit identically in all four plans)
- **(team)** Add `backtest_summary.csv`: `strategy, sharpe, sharpe_ci_low, sharpe_ci_high, ann_return, max_drawdown`. Backtest intervals currently have nowhere to go.
- **(team)** Add `arxiv_tagged.csv`: `arxiv_id, title, submitted_date, area, signal, confidence`, committed to git. Without it, a fresh clone has no live feed.
- **(team)** Define the crowding `date` as the week-ending date.
- **(team, P0)** Add `sample_end_date` to `papers.csv` (see A1).
- **(team)** Add a handoff: A → D, verified `papers.csv`, about 20:30. D's backtest no longer waits on `decay_results.csv`, so the critical path gets shorter.
- **(team)** Use one OpenAlex fetch and cache for both A and B (for example in `src/common/`).
- **(team)** One area-tagging rule for B and D. Either Gemini tags B's papers too, or both taggers run on a shared overlap set and B and D report how often they agree.
- **(team)** Add two risks: "the key result comes out null", and "the project looks like two projects stapled together".
- Fix the sleep plan: the risk table says 3 to 4 hours, but the schedule leaves about 7.5.

## Brief
- **(team)** Replace "200+ signals" with "8 Ken French signals, OSAP if time allows", unless OSAP becomes core (see Improvements).
- **(team)** Replace "daily briefing" with "weekly crowding, refreshed daily".
- **(team)** Give "where research attention should go next" to D's briefing.
- **(team)** Present the decay result as a replication of McLean & Pontiff (2016). The new parts are the attention link and the live crowding.
- **(team)** Either evaluate crowding and tagging, or soften "all results are checked".

## Person A
- **P0:** Split each signal's history into three periods: the original sample, after the sample but before publication, and after publication. This is the McLean & Pontiff method. A plain before/after-publication split mixes overfitting with decay.
- **P0:** Count early citations by querying the papers that cite each signal paper, grouped by year. OpenAlex's per-year citation counts only cover about the last 10 years (verify this), so they miss the first years of papers from 1981 to 2013. Normalise by publication year.
- Hand the verified `papers.csv` to D by about 20:30.
- Call the attention-versus-decay result suggestive. With 6 to 8 signals it will probably be null, so plan how to present that.
- Add three limits to the README: market regimes are confounded with decay, the Ken French factors were built with hindsight, and half-lives are noisy (show their intervals prominently).

## Person B
- **P0:** Build the area forecasts from field-wide OpenAlex paper counts by area and year. Keep the one-hop citation pull for the network only. The one-hop neighbourhood is mostly factor research, so it can't forecast crypto or LLM areas.
- Decide how to handle "LLMs in finance", which has almost no history before about 2020. The minimum-size rule would otherwise drop it.
- Use logistic regression and drop gradient boosting. There are only about 120 training rows.
- Apply the agreed tagging rule, and use the shared OpenAlex cache.

## Person C
- Add views for the backtest summary (with intervals), the live feed of tagged papers, and the weekly crowding heatmap.
- Add an "honest testing" panel: B's hit rate and D's backtest, each against baselines.
- Build the demo around one through-line: attention, then decay, then allocation, then crowding today. Cut it to fit 2 minutes and keep the other views as backup.
- Run the demo only from committed files. That includes `briefing.mp3` and `arxiv_tagged.csv`.
- Add a smoke test that loads every view from the sample files.
- **(team)** Take the README and DEV post, or share them with D.

## Person D
- Pull 12 to 24 months of arXiv q-fin papers (about 5 to 10k), not a few hundred.
- Send 20 to 50 abstracts per Gemini call and cache the results by arXiv ID. For "matching signal", use the `signal` column of `papers.csv`.
- Make the crowding score weekly and smoothed (for example a Poisson z-score), not a raw ratio to the yearly average.
- Report the agreement rate from the 50 papers checked by eye.
- **P0:** Make the backtest walk-forward. At each month, re-estimate from `factors.csv` and `papers.csv` using only data up to that month, never the full-sample `decay_results.csv`.
  - Tilt: after publication, underweight signals with high early citations, starting 36 months after publication.
  - Abstain: hold the baseline weight if a signal is under 36 months past publication, or its edge's interval includes zero.
  - Baselines: equal weight, inverse volatility, and a flat post-publication haircut taken from McLean & Pontiff (2016), who found returns about 58% below in-sample after publication (about 32% of that due to publication itself). Pick one figure and state it.
  - Intervals: block-bootstrap the Sharpe ratios.
- Pre-record a fallback briefing early, not at 09:30.
- **(team)** Pin versions in `requirements.txt`.
- **(team)** Replace or delete the PyCharm boilerplate `main.py`.

## Improvements
- **(team)** Consider promoting OSAP to core for A. It has 200+ signals with publication years and sample ends, which fixes A1, the null-result risk and the regime problem.
- Placebo test: rerun the decay test with random fake publication dates.
- Commit the hypotheses and tests before seeing results.
- Track crowding by signal as well as by area, to connect the live layer to decay.
- Add one decision-card panel: weights for each signal with intervals, the top 3 areas to watch, and one caveat. The briefing reads it aloud.
- Add a rough transaction-cost haircut to the backtest.
- Run the whole pipeline end to end on the sample files by about 21:00.
- Add one command that rebuilds every output, and a GitHub Action that runs `pytest` on every push.
- Precompute the network layout, with a static image as fallback.
- Record a backup demo video after the 11:00 dry run.
