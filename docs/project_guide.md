# Research Lifecycle Radar: a beginner's guide

This guide teaches the whole project from scratch. It assumes you know basic programming (variables, loops, functions, files) but no finance and not much statistics. Part 1 explains the project. Part 2 explains each proposed change in [roadmap_review.md](roadmap_review.md): what it says and why.

Read it in order the first time. The [glossary](#glossary) at the end is there to look things up later.

---

# Part 1: The project

## 1. The idea in one story

Imagine a professor discovers that stocks that went up a lot over the past year tend to keep going up next month. She publishes a paper. Before the paper, few people knew this, so following the pattern made money. After the paper, thousands of traders read it and start buying those stocks. Their buying pushes prices up early, so the pattern gets weaker. The money in the idea leaks away.

The project asks two questions:

1. **How fast do published trading ideas stop working?** This is the **decay test**.
2. **Can we see which research areas are heating up, and does early attention to a paper predict how fast its idea fades?** This is the **research map** and the **live crowding** view.

Then it turns the answers into a **decision**: which trading signals still deserve money, and where research attention is piling in today.

Every piece of data is free and public. Every result is tested on data the model never saw, and compared with simple baselines, so we know whether it beats doing something dumb.

## 2. Finance basics

### Stocks and returns
A **stock** is a small share of a company. Its price changes every day.

A **return** is how much the price changed, as a fraction:

```
return = (price_end - price_start) / price_start
```

If a stock goes from $100 to $103 in a month, its monthly return is 0.03, or 3%. The project works with **monthly** returns throughout.

### Signals
A **signal** (also called a **factor**, **predictor** or **anomaly**) is a rule for ranking stocks, in the hope that the top-ranked ones will do better than the bottom-ranked ones. The signals in this project:

| Signal | Ranking rule | Paper |
|---|---|---|
| Momentum | Past 12-month return. Recent winners rank high. | Jegadeesh and Titman (1993) |
| Value | Company's book value relative to its price. "Cheap" stocks rank high. | Fama and French (1992) |
| Size | Company's market value. Small companies rank high. | Banz (1981) |
| Profitability | Profits relative to assets. Profitable firms rank high. | Novy-Marx (2013) |
| Investment | How fast the firm grows its assets. Slow growers rank high. | Titman, Wei and Xie (2004) |
| Short-term reversal | Last month's return. Last month's losers rank high. | Jegadeesh (1990) |
| Long-term reversal | Return over the past 3 to 5 years. Long-run losers rank high. | De Bondt and Thaler (1985) |

### Long-short portfolios: how a signal becomes a return series
To measure a signal, researchers build a **long-short portfolio**:

1. Each month, rank all stocks by the signal.
2. **Buy** (go "long") the top group, for example the top 30%.
3. **Sell short** the bottom group. Short selling means borrowing a stock and selling it, then buying it back later. You make money if its price falls.
4. The portfolio's return that month is roughly *top group's return minus bottom group's return*.

Do this every month and you get one number per month: the signal's **return series**. If the signal works, the numbers are positive on average. Because you buy and sell at the same time, the general direction of the market mostly cancels out. What's left is the signal's own effect.

### Edge, decay and half-life
- **Edge:** the signal's average monthly return. If momentum averaged 1.1% a month before its paper, that's its edge before publication.
- **Decay:** the edge shrinking after publication. If momentum averages 0.6% a month afterwards, it decayed by about 45%.
- **Half-life:** how long until the edge falls to half its starting size. The idea comes from radioactive decay. A half-life of 96 months means that after 8 years the edge is half of what it was.

### Why decay happens
Two reasons, and the project needs to tell them apart (see change A1 in Part 2):

1. **The original result was partly luck.** Researchers try many ideas and publish the ones that worked on their data. Some of that "edge" was a fluke, so it disappears on new data whether or not anyone trades it. This is called **overfitting** or **data mining**.
2. **Traders copy it.** After publication, money flows into the signal and competes the edge away. This is the effect the project actually wants to measure.

### Crowding
**Crowding** means many people pursuing the same idea. In this project it means many new research papers on the same topic. If 19 papers on "LLMs in finance" appear this week and the usual number is 8, that area is crowded.

## 3. The data sources

| Source | What it is | Used by | How we get it |
|---|---|---|---|
| **Ken French Data Library** | Free monthly return series for the classic signals, kept by Professor Kenneth French. Some go back to 1926. | A | Download zip files of CSVs from his website. |
| **Open Source Asset Pricing (OSAP)** | A free project by Andrew Chen and Tom Zimmermann that rebuilt 200+ published signals from their original papers, with documentation for each. | A (stretch goal) | The `openassetpricing` Python package. |
| **OpenAlex** | A free catalogue of almost every academic paper: titles, dates, authors, topics, and who cites whom. | A, B | A web API (see below). |
| **arXiv** | A free website where researchers post papers before (or instead of) journals. Its finance section is called **q-fin**. | D | The arXiv API. |
| **Gemini** | Google's large language model (LLM). | D | An API with a free tier. |
| **ElevenLabs** | Turns text into realistic speech. | D | An API with a free tier. |

### APIs, rate limits and caching
An **API** (application programming interface) is a website meant for programs instead of people. Your code sends a request to a web address, and gets data back as JSON or XML instead of a web page. For example:

```python
import requests
r = requests.get("https://api.openalex.org/works", params={"search": "momentum"})
data = r.json()
```

Free APIs protect themselves with **rate limits**: you may only make so many requests per second or per day. arXiv asks for at most one request every 3 seconds. Gemini's free tier has a daily cap. Break the limit and you get errors or a temporary ban.

The defence is **caching**. Save every response to disk the first time, and next time read the file instead of calling the API again. In this repo, caches live in `data/cache/` and raw downloads in `data/raw/`. Git ignores both, so they never get uploaded. Caching also makes the demo safe: if the internet fails on stage, everything still loads.

### API keys
Some APIs need a **key**: a secret string that identifies you, like a password. Keys go in a file called `.env`, which git ignores. **Never commit a key.** If a key gets into a public repo, bots will find and abuse it within minutes. `.env.example` lists which keys exist, but has no values.

## 4. How the project fits together

### The four people

| Person | Job | Produces |
|---|---|---|
| **A**: Signal decay | Gets signal returns and paper dates, measures decay, tests whether early citations predict decay | `factors.csv`, `papers.csv`, `decay_results.csv` |
| **B**: Research map | Gets papers and their citations from OpenAlex, sorts them into 8 areas, draws the network, forecasts which areas will boom | `network_nodes.csv`, `network_edges.csv`, `area_forecasts.csv` |
| **C**: Frontend and pitch | Builds the dashboard (Streamlit), the demo and the slides | The app |
| **D** (you): Live layer and decision | Pulls new arXiv papers, tags them with Gemini, computes crowding, runs the allocation backtest, makes the spoken briefing | `crowding.csv`, `backtest_results.csv`, `briefing.txt` and `.mp3` |

### Data flow

```
Ken French ──► factors.csv ─────────────┐
                                        ├──► decay test (A) ──► decay_results.csv ──┐
papers (by hand) ──► papers.csv ────────┤                                           │
OpenAlex citations ─────────────────────┘                                           │
                                                                                    ▼
OpenAlex papers ──► areas + network (B) ──► network_*.csv, area_forecasts.csv ──► DASHBOARD (C)
                                                                                    ▲
arXiv ──► Gemini tags (D) ──► crowding (D) ──► crowding.csv ────────────────────────┤
factors + papers ──► backtest (D) ──► backtest_results.csv ─────────────────────────┤
crowding + tags ──► briefing (D) ──► briefing.txt / .mp3 ───────────────────────────┘
```

### Contracts: how four people work without breaking each other's code
Each output is a **CSV** file: a plain-text table where each line is a row and commas separate the columns. Everyone agreed in advance on each file's exact column names. That agreement is the **contract**.

The contracts live in code, in `src/common/contracts.py`:

- `AREAS`: the 8 research areas. B and D must spell them exactly the same way.
- `CONTRACTS`: each file name, its owner, and its columns.
- `save(name, df)`: checks that a table matches its contract, then writes it. Use it for every output.
- `load(name)`: reads the real file if it exists, otherwise the made-up **sample** file in `data/sample/`.

Sample files are why nobody waits on anyone. C builds the whole dashboard from fake data on day one, and the real files simply replace the fakes as they arrive. `tests/test_contracts.py` checks that every file matches its contract. Run it with `pytest`.

**`df`** is short for a pandas **DataFrame**: Python's standard table object. Think of it as a spreadsheet in memory.

### Repo rules in plain words
- **Git branches:** each person works on their own copy (`a/...`, `b/...` and so on) and merges into `main` often. `main` must always work.
- **Own folders:** A works in `src/signal_decay/`, B in `src/research_map/`, C in `src/app/`, D in `src/live/`. If two people never edit the same file, git can always merge their work automatically.
- **Shared files:** `requirements.txt` (the list of Python packages), `contracts.py` and the README. Announce changes to these before pushing.

## 5. The statistics you need

You don't need formulas to follow the project, but you do need these ideas.

**Mean.** The average. A signal's edge is the mean of its monthly returns.

**Noise.** Monthly returns jump around a lot. A signal averaging 0.5% a month might return +4% one month and −3% the next. So any average computed from returns is uncertain.

**Confidence interval (CI).** A range that says how sure we are of a number. "Post-publication edge is 0.3%, 95% CI from −0.1% to 0.7%" means the true value is plausibly anywhere in that range. **If the interval includes zero, we can't rule out that the edge is gone.** Wide intervals mean "not enough data to say".

**Statistical significance.** A result is "significant" when it is unlikely to be pure chance. With few data points, real effects often fail to reach significance. That doesn't prove there is no effect; it means we couldn't detect one.

**Bootstrap.** A way to get a confidence interval when there is no neat formula:

1. Resample your data with replacement many times, for example 1,000 times.
2. Recompute the number you care about each time.
3. The spread of those 1,000 answers is your uncertainty.

A **block bootstrap** resamples chunks of consecutive months instead of single months. That keeps the fact that one month's return is related to the next.

**Overfitting.** A model that memorises its training data instead of learning a general pattern. It looks great on old data and fails on new data. Fewer data rows plus a more flexible model means more overfitting.

**Train/test split.** Fit the model on one period (train), and judge it only on a later period it never saw (test). In finance the split must be **by time**: train on the past, test on the future, just like real life.

**Look-ahead bias (leakage).** Accidentally using information that would not have been available at that moment. Example: deciding in 1995 how much to invest in momentum using a decay estimate computed from 1995–2024 data. Look-ahead makes a strategy look much better than it could ever be in reality. It's the most common and most damaging mistake in finance research.

**Walk-forward test.** The cure for look-ahead. Step through time one month at a time. At each month, use only the data up to that month to decide, then record what happened the next month. It simulates trading for real.

**Baseline.** A simple, dumb strategy to beat. "Split money equally between all signals" is a baseline. If your clever model doesn't beat it, the cleverness isn't worth anything. Judges care a lot about baselines.

**Sharpe ratio.** Return per unit of risk: the average return divided by how much returns swing (their standard deviation). Monthly figures are usually scaled to a year by multiplying by √12. Higher is better. It's the standard way to compare strategies.

**Max drawdown.** The worst peak-to-bottom loss. If a strategy grew to 150, fell to 105, then recovered, its max drawdown is 30%.

**Regression.** Fitting a line through data to see how one thing relates to another. Here: do signals with more early citations decay faster?

**Logistic regression.** A regression that predicts a probability between 0 and 1. B uses it for "will this area boom in the next 2 years?"

**Gradient boosting.** A more powerful prediction method that combines many small decision trees. It needs much more data than logistic regression to avoid overfitting.

**Multiple testing.** Run 20 tests and on average one will look "significant" by chance alone. A multiple-testing correction raises the bar to account for this.

**Confounding.** When something else causes both things you're looking at. If every signal's post-publication period happens to be a bad decade for all signals, decay and "bad decade" get mixed up.

**Placebo test.** Rerun your test where the effect should *not* exist, for example with fake dates. If you still find an "effect", your method is picking up something else.

**z-score.** How many standard deviations a value sits from normal. A z-score of 2 means unusually high.

**Poisson.** The standard model for counts of random events, like papers per week. Under it, a count's typical random swing is about √(expected count). That gives a simple crowding z-score:

```
z = (this_week_count - expected_count) / sqrt(expected_count)
```

## 6. Each part in detail

### A: Signal decay
1. Download Ken French monthly returns and save `factors.csv` (date, signal, return).
2. Build `papers.csv` by hand: for each signal, the paper, its publication date, and which Ken French column holds its returns. Pick one **date rule** (for example "the journal issue date") and check every row against the real paper.
3. **Before/after test:** compare the mean return before and after publication, with confidence intervals. Draw a chart per signal with a vertical line at the publication date.
4. Pull citation counts from OpenAlex. `early_citations` is the number of citations in years 1 to 3 after publication, a measure of how much attention the paper got.
5. Fit half-lives, and run the attention-versus-decay regression.
6. Write `decay_results.csv`, one row per signal.

### B: Research map
1. Pull papers from OpenAlex: A's signal papers plus one hop. One hop means the papers they cite and the papers that cite them.
2. Give each paper one of the 8 areas, using OpenAlex's topic labels.
3. Write the network: **nodes** are papers, **edges** are citations ("paper X cites paper Y"). Each paper also gets a **lifecycle stage**: emerging, peaking or saturated.
4. Compute features for each area and year: paper count, growth, acceleration, new-author share, citation momentum.
5. Define a **boom**: top-10% growth in paper count, plus an above-median share of citations, over the next 2 years. Train a model on data up to 2016, test it on 2017 onwards, and compare it with baselines.
6. Write `area_forecasts.csv`.

### C: Frontend and pitch
**Streamlit** is a Python library that turns a script into a web app: you write Python, it draws buttons, charts and tables. C builds:

- signal timelines with publication markers;
- the paper network;
- the crowding heatmap (areas by weeks, coloured by crowding);
- the backtest chart;
- a "play briefing" button.

Then C writes the 2-minute demo and the slides.

### D (you): Live layer and decision
1. **arXiv pull:** fetch recent q-fin papers. arXiv returns **Atom XML**, a feed format; the `feedparser` library turns it into Python dictionaries. Wait 3 seconds between requests, and cache everything.
2. **Gemini tagging:** send each paper's title and abstract to Gemini with a prompt like *"Which of these 8 areas is this paper in? Does it study one of these signals? How confident are you?"* Ask for JSON output so the code can read the answer. Check 50 tags by eye before running all of them.
3. **Crowding:** count papers per area per week, compare with that area's usual level, and write `crowding.csv`.
4. **Backtest:** simulate a fund that splits money across the signals, giving less to signals expected to decay faster. Compare it with baselines. Write `backtest_results.csv`.
5. **Briefing:** Gemini writes a short "this week on the radar" script from the crowding and the tags, and ElevenLabs reads it aloud as an `.mp3`. Keep a pre-recorded copy in case the API fails.
6. **Repo hygiene:** README, DEV post (a blog post for the MLH track), and a fresh-clone test: download the repo to an empty folder and check that it runs.

---

# Part 2: Each roadmap change explained

Each heading matches [roadmap_review.md](roadmap_review.md). **(team)** means it touches shared files or other people's work. **P0** means it changes what someone builds, so decide it before anyone pulls data.

## Before starting

**Confirm the deadline, judging criteria, demo format and skill levels. (team, P0)**
Every time in the plans is a guess ("Sat 18:00 to Sun 12:00"). The "why it fits" section of the brief guesses what G-Research and HRT reward. If HRT mainly judges on prediction accuracy and you've spent the night on visuals, you've built the wrong thing. Live versus recorded demo decides whether the briefing has to work live. Skill levels decide who can take over if someone gets stuck. It costs 10 minutes and protects everything else.

## Shared section

**Add `backtest_summary.csv`. (team)**
Your plan says the backtest is reported "with intervals". The existing `backtest_results.csv` has one row per month per strategy (date, return, running total), with nowhere to put a confidence interval. A summary file with one row per strategy (Sharpe, its interval, annual return, max drawdown) gives C a ready-made "honest results" table.

**Add `arxiv_tagged.csv`, committed to git. (team)**
Your tagged papers currently live only in `data/cache/`, which git ignores. Anyone who downloads the repo, including a judge, gets no papers, and can't regenerate them without a Gemini key. Committing a small tagged-papers file means the live feed always shows something.

**Define the crowding `date` as the week-ending date. (team)**
`crowding.csv` has a `date` column, but nobody said whether it's daily or weekly. C needs to know to build the heatmap's time axis, and the sample file has only one date, so C can't even test a heatmap. Weekly is the right choice because daily counts per area are mostly 0 or 1 (see the D changes).

**Add `sample_end_date` to `papers.csv`. (team, P0)**
Needed for change A1. Every paper studied data up to some date, which is usually a few years before it was published. A needs that date to split each signal's history into three periods.

**Add a handoff: A gives D the verified `papers.csv` around 20:30. (team)**
After change D5, your backtest needs the publication dates, not A's final decay results. The dates are ready much earlier than the results, so you can start the real backtest hours sooner. And the project's critical path, the chain where any delay delays everything, gets shorter.

**One shared OpenAlex fetch and cache. (team)**
A and B both download the same signal papers from OpenAlex. Doing it twice wastes rate limit and time, and risks two slightly different copies of the data. One shared module in `src/common/` that both import avoids all three problems.

**One area-tagging rule for B and D. (team)**
B gives papers areas by mapping OpenAlex's topic labels. You use Gemini. Both write the same 8 labels, but the two methods could disagree. Then the dashboard's "LLMs in finance" means one thing on the map and another on the heatmap, and a sharp judge will ask about it. The two options:

- Gemini also tags B's papers: consistent, but costs more API calls.
- Both methods run on a shared set of papers, and you report how often they agree, for example "they agree on 82%". Cheaper, and the agreement rate is itself honest evidence.

**Add two risks: a null result, and "two projects stapled together". (team)**
These are the two most likely ways the project disappoints. A *null result* means the key test finds no effect (see A4). "Stapled together" means the map and the decay test feel unrelated (see C3). Writing them in the risk table makes the team plan for them, rather than discovering them at 09:00.

**Fix the sleep plan.**
The risk table says everyone sleeps 3 to 4 hours, but the schedule stops at 00:30 and starts at 08:00, which is 7.5 hours. Agree which one is real, so nobody turns up late or plans work for hours that don't exist.

## Brief

**Replace "200+ signals" with "8 Ken French signals, OSAP if time allows". (team)**
OSAP is only a stretch goal for A, and may not happen. Promising judges 200+ and showing 8 hurts credibility. Either make OSAP core (see Improvements) or say 8.

**Replace "daily briefing" with "weekly crowding, refreshed daily". (team)**
Crowding is measured weekly, so a "daily" briefing would repeat itself most days. The honest wording still sounds live.

**Give "where research attention should go next" to D's briefing. (team)**
The brief promises this decision, but no roadmap owns it. It needs B's forecasts plus your crowding numbers, and your briefing already reads both.

**Present the decay result as a replication of McLean & Pontiff (2016). (team)**
In 2016, David McLean and Jeffrey Pontiff studied about 97 published signals and found returns are about 26% lower after each paper's sample ends, and about 58% lower after publication. Quant judges will know that paper. Claiming "we discovered signals decay" sounds naive. Saying "we reproduce the known result, then add something new: the attention link and live crowding" sounds expert.

**Either evaluate crowding and tagging, or soften "all results are checked". (team)**
Crowding is a description, not a prediction, so there's nothing to test it against yet. The Gemini tags could be wrong. Either measure the tag accuracy (see D4), or change the wording so it doesn't promise more than you deliver.

## Person A

**A1. Split each signal's history into three periods. (P0)**
The plan compares "before publication" with "after publication". But "before" includes the paper's own data, which the author chose because it worked. Take momentum as an example:

| Period | Example dates | What it tells you |
|---|---|---|
| Original sample | 1965–1989 | Edge as the author found it, flukes included |
| After the sample, before publication | 1990–1993 | Edge on new data nobody was trading yet. A drop here is overfitting. |
| After publication | 1993 onwards | Edge once everyone knew. The extra drop here is traders copying. |

With only two periods you can't tell overfitting from copying, and the copying effect is exactly what the project claims to measure. This is the McLean & Pontiff method, and it needs `sample_end_date` in `papers.csv`.

**A2. Count early citations by year from the citing papers. (P0)**
OpenAlex gives each paper a convenient list of citations per year, but (according to its documentation; A should verify) only for about the last 10 years. A paper from 1993 got its "early" citations in 1993–1996, so they aren't in that list. Instead, ask OpenAlex for the papers that cite each signal paper, grouped by publication year. That's one request per paper, and still cheap.

Also **normalise by publication year**: far more papers are published now than in 1985, so citation counts have inflated. Fifty early citations in 1985 might be as impressive as 300 in 2013. Dividing by a typical count for that year makes them comparable.

**A3. Hand the verified `papers.csv` to D by about 20:30.**
This is the other side of the new handoff in the shared section.

**A4. Call the attention-versus-decay result "suggestive".**
With 6 to 8 signals, the regression has 6 to 8 data points. Even a real effect will rarely be statistically significant with that few. And "multiple-testing correction" is meaningless with one test on 8 points. Plan the honest framing now, for example: "the direction is consistent with attention speeding up decay, but 8 signals can't confirm it; here's how OSAP would". A null result explained well beats an overclaim that judges take apart.

**A5. Add three limits to the README.**
- *Regimes:* trading got much cheaper after the 1990s, value stocks had a long bad stretch after 2007, and momentum crashed in 2009. All of this happened after most of the papers came out, so "decay" may partly just be a bad era for everything (**confounding**). With 8 signals you can't separate the two; with 200+ signals published in different years, you can.
- *Hindsight factors:* Ken French's value and profitability series were built later using today's definitions, not exactly the papers' original recipes. So the "before publication" returns aren't exactly what the authors saw.
- *Noisy half-lives:* fitting a decay curve to noisy monthly returns gives very uncertain answers, and decay may not even follow a neat curve. Show the confidence intervals prominently, and consider also reporting the simpler "edge after publication as a share of edge before".

## Person B

**B1. Build the area forecasts from field-wide paper counts. (P0)**
B's corpus is A's 7–8 signal papers plus their one-hop neighbours. Those are mostly classic factor papers from 1981–2013, and the papers around them are mostly factor research too. Crypto and LLM papers will barely appear. So a growth model trained on that corpus measures "growth of papers near famous factor papers", not growth of research areas.

The fix: keep the one-hop pull for the network picture, but get each area's yearly paper count across the whole field. OpenAlex can count papers grouped by year for a topic in a single request (a `group_by` query), so this takes a handful of calls.

**B2. Decide how to handle "LLMs in finance".**
LLMs barely existed as a finance topic before about 2020. The boom definition has a minimum area size, so this area drops out of training and testing. That means the model can't forecast the area that's hottest in the demo. Either say so openly ("too new to forecast; here is its crowding instead"), or handle it as a special case.

**B3. Use logistic regression and drop gradient boosting.**
Training rows are one per area per year: about 8 × 15 = 120, minus the last 2 years because their "boom in the next 2 years" outcome isn't known yet. Gradient boosting needs far more rows than that, and with 120 it will memorise noise (overfit). Logistic regression is simpler and more honest at this size.

**B4. Apply the agreed tagging rule, and use the shared OpenAlex cache.**
These are the B side of the shared-section changes.

## Person C

**C1. Add views for the new files.**
A table for `backtest_summary.csv`, a list of this week's tagged arXiv papers, and a heatmap with weeks along one axis and areas along the other.

**C2. Add an "honest testing" panel.**
The brief promises honest evaluation, but no view shows it. One panel with B's forecast hit rate next to its baselines, and D's backtest next to its baselines, each with intervals. This is where the HRT judges will look.

**C3. Build the demo around one through-line, and cut to fit 2 minutes.**
The risk is two separate projects: a paper map, and a signal decay test. The thread that ties them together: *papers get attention → attention predicts faster decay → so we give less weight to crowded signals → and here's where attention is piling up today.* Every view in the demo should be a step in that story. Six views in 2 minutes is 20 seconds each, which is too rushed. Pick the 3 or 4 that tell the story, and keep the rest for questions.

**C4. Run the demo only from committed files.**
If the demo depends on an API call and the Wi-Fi drops, the demo dies. Committed files always load.

**C5. Add a smoke test.**
A **smoke test** is a quick check that the app starts and each page loads, without checking details. `pytest` currently tests only the contracts, so `main` could "pass" while the app crashes.

**C6. Take the README and DEV post, or share them. (team)**
D's last hour holds the briefing, the README, the DEV post and the fresh-clone test. That's too much. C is already writing the pitch, which overlaps heavily with the README and the post.

## Person D

**D1. Pull 12–24 months of arXiv q-fin, not a few hundred papers.**
Crowding compares this week with each area's normal level. "Normal" needs a long history: 12 to 24 months is about 5,000 to 10,000 q-fin papers. A few hundred papers covers only a few weeks. The pull itself is fast (about 1,000 results per request at one request every 3 seconds), so it's minutes, not hours.

**D2. Batch the Gemini calls and cache them. Use `papers.csv` for signal names.**
Ten thousand papers at one request each would blow through the free tier's daily limit. Putting 20 to 50 abstracts in one prompt cuts the number of requests by 20 to 50 times. Cache each result by its arXiv ID so a crash never means re-tagging. For "matching signal", give Gemini a fixed list (the `signal` column of `papers.csv`), so it can't invent new names.

**D3. Make the crowding score weekly and smoothed.**
Spread across 8 areas, q-fin gives about 2 to 15 papers per area per week. With counts that small, the raw ratio "this week / usual" swings wildly: going from 2 papers to 4 looks like a doubling but is just chance. A Poisson z-score (see section 5) scales the jump by how big a random swing normally is. Weekly counts are big enough to mean something; daily counts aren't.

**D4. Report the agreement rate from the 50 papers checked by eye.**
You already plan to check 50 tags. Write down how many Gemini got right, for example "44 of 50, 88%". That one number turns "we used an LLM" into "we measured our LLM", and backs up the brief's claim that results are checked.

**D5. Make the backtest walk-forward. (P0)**
This is the most important change for you. The plan uses `decay_results.csv`, but its half-lives come from each signal's whole history, future included. A backtest deciding the 1998 weights with them would be using information from 2024. That's look-ahead, and it makes the results meaningless. The fix is a walk-forward loop:

```
for each month t:
    data_so_far = returns up to month t        # nothing after t
    estimate each signal's edge from data_so_far
    choose weights using those estimates and the rules below
    record the portfolio's return in month t+1
```

The rules:
- **Tilt:** after publication, give less weight to signals with high early citations. This turns the project's key question into a trading rule you can test. `early_citations` counts years 1 to 3, so it only exists from 36 months after publication. Using it earlier would be look-ahead.
- **Abstain:** if there isn't enough evidence (the signal is under 36 months past publication, or its edge's confidence interval includes zero), don't tilt; keep the baseline weight. A model that knows when to stay quiet is more believable.
- **Baselines:**
  - equal weight: the same money in every signal;
  - inverse volatility: more money in calmer signals;
  - a flat haircut: cut every signal's expected edge by the same amount after publication. McLean & Pontiff found post-publication returns about 58% below in-sample, and about 32% of that is the effect of publication itself. The team should pick one figure and state it. If your attention tilt can't beat "assume everything decays by the textbook amount", the attention data adds nothing.
- **Intervals:** block-bootstrap the Sharpe ratios, so the result reads "tilt Sharpe 0.45 (CI 0.20 to 0.70) vs equal weight 0.40 (CI 0.15 to 0.65)". In that example the intervals overlap heavily, so you'd honestly report no clear win.

**D6. Pre-record a fallback briefing early.**
At 09:30 on Sunday, ElevenLabs may be down, or you may be out of free credits, with no time left to fix it. Record a version on Saturday night, and replace it with a newer one if things go well.

**D7. Pin versions in `requirements.txt`. (team)**
`requirements.txt` lists packages without versions, for example just `pandas`. A judge installing next week might get a newer pandas that changed something, and the app breaks. **Pinning** means writing exact versions (`pandas==2.2.3`). `pip freeze` prints the versions you currently have installed.

**D8. Replace or delete `main.py`. (team)**
It's PyCharm's auto-generated "Hi, PyCharm" example. It looks sloppy to a judge reading the repo. Better: make it the one command that rebuilds every output (see Improvements).

## Improvements

**Promote OSAP to core for A. (team)**
OSAP gives 200+ signals, each with its paper's publication year and sample end already documented. That fixes three problems at once:

- A1: the sample end dates come with the data.
- A4: 200 data points can actually detect an effect.
- A5: signals published in different decades let you separate decay from bad eras.

It's the biggest single upgrade available. A should test the package early in the evening to see if it's feasible.

**Placebo test.**
Rerun the decay test with each signal's publication date moved to a random fake date. If the fake dates "decay" as much as the real ones, the drop is a general time trend, not caused by publication. If only the real dates show the drop, that's strong evidence. It's cheap: same code, different dates.

**Commit the hypotheses before seeing results.**
Write down what you'll test, how, and what would count as success, and commit it before running anything. Git timestamps every commit, so this proves you didn't choose tests after seeing which ones worked. That's a practice called **pre-registration**, and it's exactly the honest-evaluation story HRT wants.

**Crowding by signal as well as by area.**
You already tag which signal each new paper studies. Counting those gives "momentum got 12 new papers this month, versus 4 usually". That directly links the live layer to A's result: if attention predicts decay, a newly crowded signal deserves less weight. This is the through-line of C3, in data form.

**One decision-card panel.**
The brief ends with "the decision", but no single place shows it. One panel with three parts:
- suggested weights for each signal, with intervals;
- the top 3 areas to watch;
- one honest caveat.

The briefing reads this card aloud, which gives the audio a clear purpose.

**A rough transaction-cost haircut.**
Real trading costs money: fees, and the gap between buy and sell prices. Subtracting even an assumed figure, for example 0.1% per month, shows judges you know that backtests without costs flatter themselves.

**Run the pipeline end to end on samples by 21:00.**
Connect every step with fake data before any real data exists: download → tag → crowding → backtest → briefing → dashboard. This is called a **walking skeleton**. It finds plumbing bugs while there's still time, instead of at 09:00 when the real files land.

**One rebuild command and CI.**
One command (for example `python main.py`) rebuilds every output from the caches, so nobody has to remember the order. **CI** (continuous integration) is a GitHub Action that runs `pytest` automatically on every push. It catches a broken `main` immediately instead of at demo time.

**Precompute the network layout, with a static fallback.**
Laying out a graph of thousands of papers can take a long time on each page load. Compute the positions once, save them, and keep a picture as a backup.

**Record a backup demo video.**
After the 11:00 dry run, screen-record the demo. If anything fails on stage, play the video.

---

# Glossary

| Term | Meaning |
|---|---|
| Abstain | Make no tilt when the evidence is weak; keep the baseline weight |
| API | A website for programs: send a request, get data back |
| arXiv | Free site where researchers post papers; finance is "q-fin" |
| Backtest | Simulating a strategy on past data |
| Baseline | A simple strategy any clever model must beat |
| Block bootstrap | Bootstrap that resamples chunks of consecutive months |
| Bootstrap | Resampling the data many times to measure uncertainty |
| Cache | Saved API responses, so you don't call the API again |
| CI (statistics) | Confidence interval: the plausible range for a number |
| CI (engineering) | Continuous integration: tests that run automatically on every push |
| Confounding | Something else driving both things you're comparing |
| Contract | The agreed column names for each shared file |
| Crowding | Unusually many papers on one topic right now |
| CSV | Plain-text table, commas between columns |
| DataFrame (`df`) | pandas' in-memory table |
| Decay | A signal's edge shrinking after publication |
| Edge | A signal's average return |
| Factor / signal / predictor | A rule for ranking stocks |
| Half-life | Time for the edge to fall to half |
| Look-ahead bias | Using information that wasn't available yet |
| Long-short | Buy the top-ranked stocks, short the bottom-ranked |
| Max drawdown | Worst peak-to-bottom loss |
| Null result | A test that finds no detectable effect |
| OpenAlex | Free catalogue of academic papers and citations |
| OSAP | Open Source Asset Pricing: 200+ rebuilt signals |
| Overfitting | Memorising noise instead of learning a pattern |
| Pinning | Fixing exact package versions |
| Placebo test | A test where the effect shouldn't exist, as a sanity check |
| Pre-registration | Committing your tests before seeing results |
| Rate limit | Maximum requests an API allows |
| Sharpe ratio | Average return divided by its variability |
| Short selling | Borrowing a stock and selling it, betting it falls |
| Smoke test | Quick check that the app starts and loads |
| Walk-forward | Deciding each month using only data up to that month |
| Walking skeleton | Every step connected end to end, on fake data |
| z-score | How many standard deviations from normal |
