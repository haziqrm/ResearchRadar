# Research Lifecycle Radar: Plan for Person A (Signal decay)

You are **Person A**. Read "Your plan" first, then the shared context below it. The shared context is the same in all four files, so everyone is working from the same agreements.

One line: we map where quant-finance research is heading and measure how fast published trading signals stop working, using only free public data.

Assumed window: Sat 7 Nov 18:00 to Sun 8 Nov 12:00. Confirm the real deadline and shift all times.

**Read this first:** the file formats and the area list in the shared section are fixed. If you need to change a column name or an area name, tell the team before you do it.

---
## Your plan: Person A: signal decay

**Goal:** show how fast published signals fade, and whether early attention predicts it.

| When | Task | Done when |
|---|---|---|
| 18:00 to 18:45 | Shared agreements, scaffold of `src/signal_decay/` | Tests pass, folder committed |
| 18:45 to 19:30 | Download Ken French monthly factors, write `factors.csv`, plot cumulative returns | Plot shows 8 signals, date ranges printed |
| 19:30 to 20:30 | Build `papers.csv` for 8 signals. Pick one date rule. Verify each date against the paper | Every row verified = yes |
| 20:30 to 22:00 | Before/after test with confidence intervals, plus before/after charts with a publication marker | One chart per signal |
| 22:00 to 23:00 | Get OpenAlex key, pull citations for each paper, compute `early_citations` (years 1 to 3) | Column filled for every signal |
| 23:00 to 00:30 | Fit half-lives, attention-versus-decay regression, multiple-testing correction. **Commit a first `decay_results.csv` by 00:30** | File matches the contract |
| 08:00 to 09:30 Sun | OSAP scale-up toward 50+ signals if time allows | Larger-sample result, or a note it was cut |
| 09:30 to 10:30 | README section on limits: small sample, fuzzy dates, associated not causes | Section committed |

**Needs from others:** nothing to start.
**Needed by others:** `decay_results.csv` for D and C; signal paper list for B.
**Fallback:** if OSAP slips, ship with Ken French only and say so.
**Watch for:** wrong publication dates, mixing date rules, claiming more than the small sample supports.

---

---

# Shared context (same in every file)

## Shared: Who does what

| | Role | Owns | Main output files |
|---|---|---|---|
| **A** | Signal decay | Ken French and OSAP returns, paper table, before/after tests, half-lives, attention test | `factors.csv`, `papers.csv`, `decay_results.csv` |
| **B** | Research map | OpenAlex papers, areas, network, area-growth model | `network_nodes.csv`, `network_edges.csv`, `area_forecasts.csv` |
| **C** | Frontend and pitch | Streamlit dashboard, visuals, demo, slides, pitch | The app and the demo |
| **D** | Live layer and decision | arXiv feed, Gemini tagging, crowding, backtest, briefing, repo hygiene | `crowding.csv`, `backtest_results.csv`, `briefing.txt/.mp3` |

---

## Shared: Shared agreements (all four, 18:00 to 18:45)

### 1 Repo rules
- One GitHub repo, one branch per person, merge to main often, **main must always run**.
- Each person works in their own folder: A `src/signal_decay/`, B `src/research_map/`, C `src/app/`, D `src/live/`.
- Shared Python version: 3.12. One `requirements.txt` (add, don't rewrite).
- Never commit API keys. Keys live in each person's own `.env`.

### 2 The area list (B and D must use identical spelling)
Fix this list now and write it in the README. Rename nothing without telling the others.

1. Factor and anomaly research
2. Machine learning for returns
3. Market microstructure
4. Portfolio and risk
5. Derivatives and volatility
6. Alternative data and text
7. Crypto and DeFi
8. LLMs in finance

### 3 File contracts (column names are fixed once agreed)

| File | Owner | Columns |
|---|---|---|
| `data/factors.csv` | A | date, signal, return |
| `data/papers.csv` | A | signal, paper, publication_date, date_rule, ken_french_column, source_verified |
| `data/decay_results.csv` | A | signal, paper, publication_date, pre_mean, post_mean, half_life_months, ci_low, ci_high, early_citations |
| `data/network_nodes.csv` | B | paper_id, title, year, area, citations, lifecycle_stage |
| `data/network_edges.csv` | B | source_id, target_id |
| `data/area_forecasts.csv` | B | area, year, growth_score, boom_probability, predicted_boom, actual_boom |
| `data/crowding.csv` | D | date, area, paper_count, crowding_score |
| `data/backtest_results.csv` | D | date, strategy, monthly_return, cumulative_return |
| `data/briefing.txt`, `data/briefing.mp3` | D | text and audio |

### 4 Sample files first
Each owner commits a **sample file with 5 to 10 made-up rows** for each of their outputs before 19:00. C builds the whole frontend from these, then swaps in real files.

### 5 Definitions everyone should use the same way
- **Signal (factor, predictor):** something you sort stocks by, such as momentum or value.
- **Edge:** the extra return a signal earned.
- **Decay:** the edge shrinking after the paper is published.
- **Half-life:** time for the edge to fall to half.
- **Published trading edge:** an edge reported in a named paper with a date, and with a return series we can measure.
- **Next big area:** top-decile growth in paper count and above-median citation share over the next 2 years, with a minimum area size.

### 6 Time windows (fixed in advance, same for everyone)

| Module | Window |
|---|---|
| A: decay | Full available series, at least 10 years before and after each paper |
| B: map and growth | About 2000 to now; train up to about 2016, test 2017 onward |
| D: crowding baseline | Last 12 to 24 months of arXiv, plus daily refresh |
| D: backtest | Starts after 10 years of history, runs to the present |

---

## Shared: Data dependencies

### 1 Raw sources

| Source | Used by | For |
|---|---|---|
| Ken French library | A | Signal returns |
| Open Source Asset Pricing (OSAP) | A | 200+ extra signals (stretch) |
| OpenAlex | A, B | A: citations per signal paper. B: papers, topics, links |
| arXiv | D | New papers for the live feed |

### 2 Derived files: what needs what

| File | Built from | Needed by |
|---|---|---|
| `papers.csv` | The papers (by hand) | A, B, D, C |
| `factors.csv` | Ken French | A, D |
| `decay_results.csv` | factors, papers, OpenAlex citations | D (backtest), C |
| network files | OpenAlex papers, area list | C |
| `area_forecasts.csv` | OpenAlex papers, area list | C |
| tagged arXiv papers | arXiv, area list | D |
| `crowding.csv` | tagged arXiv papers | C |
| `backtest_results.csv` | decay_results, factors | C |
| `briefing.*` | crowding, tagged papers | C |

### 3 The critical path
`Ken French download` → `papers.csv (verified)` → `before/after test` → `decay_results.csv` → `backtest` → `dashboard`

This chain runs through A. If A's early steps slip, D's backtest and C's real charts slip too. Everything else runs in parallel.

### 4 Handoffs (who gives what to whom, and by when)

| By | From | To | What |
|---|---|---|---|
| 18:30 | A | B, D | Draft list of signal paper names (no dates needed yet) |
| 18:45 | B and D | each other | Agreed area list |
| 19:00 | all | C | Sample files for every output |
| 20:30 | B | C | First real network files |
| 22:00 | D | C | First real `crowding.csv` |
| 00:30 | A | D, C | First real `decay_results.csv` (even if only 4 signals) |
| 09:00 Sun | B, D | C | Final forecasts, backtest results, briefing |

---

## Shared: Checkpoints (10-minute huddles)

| When | Question for everyone |
|---|---|
| 19:00 | Do the sample files exist and does the app load them? |
| 21:00 | Has each person got a first real output? Is anyone blocked? |
| **00:30** | **Decision point: does every half have a real output file? If one has stalled, cut it back to a simple version and keep the other as the main project.** |
| 08:00 | What's left? Divide the remaining work |
| 10:00 | Freeze new features; only fixes and polish |
| 11:00 | Dry-run the demo end to end |

---

## Shared: Packages and accounts

**Core (everyone):** pandas, numpy, scipy, statsmodels, scikit-learn, matplotlib, plotly, requests, python-dotenv, pyarrow, pytest.
**By role:** A: openassetpricing. B: networkx. C: streamlit. D: feedparser, google-genai, elevenlabs.
**Optional (B stretch only):** umap-learn, hdbscan, bertopic or sentence-transformers.
**Free keys (no card):** OpenAlex (A and B, each get their own), Gemini (D, maybe B), ElevenLabs (D).

---

## Shared: Risks and fallbacks

| Risk | Plan |
|---|---|
| A's decay file is late | Commit a rough version with 3 to 4 signals by 00:30 |
| File formats drift | Contract table is the source of truth; announce any change |
| Wrong publication dates | One date rule; verify every row |
| Free-tier limits (Gemini, ElevenLabs, OpenAlex) | Cache, batch, pre-record fallbacks |
| Look-ahead leakage | Time-based splits; features use only data up to the split |
| Merge conflicts | Own folders, small frequent merges |
| Live demo fails | Demo from cached data; keep recordings as backup |
| Fatigue | Everyone sleeps 3 to 4 hours; no new features after 10:00 |

**Cut order if time runs short:** OSAP scale-up, embeddings and extra areas, live-feed polish, ElevenLabs audio (keep the text briefing).
**Never cut:** the before/after result, honest evaluation, the paper network, and demo rehearsal.

---

## Shared: Definition of done

- Every output file exists, matches its column contract, and loads in the app.
- The dashboard runs from a fresh clone with no keys committed.
- Results are shown with confidence intervals, baselines, and a plain statement of limits.
- README explains the idea, the data, the method, and how to run it.
- DEV post published (for the MLH track).
- The 2-minute demo has been rehearsed at least twice.

---

## Shared: Open questions to answer tonight

- Exact submission deadline for the HRT and G-Research tracks.
- Whether finance is required or only preferred.
- Live demo or recorded demo.
- Each person's skill level, in case work needs to move between roles.
