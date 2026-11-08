# Research Lifecycle Radar

We map where quant-finance research is heading and measure how fast published trading
signals stop working, using only free public data.

## Team and folders

| | Role | Folder | Outputs |
|---|---|---|---|
| A | Signal decay | `src/signal_decay/` | `factors.csv`, `papers.csv`, `decay_results.csv` |
| B | Research map | `src/research_map/` | `network_nodes.csv`, `network_edges.csv`, `area_forecasts.csv` |
| C | Frontend and pitch | `src/app/` | Streamlit app, demo |
| D | Live layer and decision | `src/live/` | `crowding.csv`, `backtest_results.csv`, `briefing.txt/.mp3` |

Full per-person plans: [docs/roadmap/](docs/roadmap/). Where D's outputs differ from the plan (crowding is a z-score, new files, arXiv volume): [docs/live_layer.md](docs/live_layer.md). Challenge briefs we're judged against: [docs/challenges/](docs/challenges/).

## Measured results

- **Tag accuracy (D):** Gemini's area tags on arXiv papers were 92% correct (46 of 50) in a
  random sample checked by hand. Signal tags haven't been checked: none of the 50 sampled
  papers had one. Details: [docs/live_layer.md](docs/live_layer.md#tag-accuracy-d4).

## Repo rules (to avoid merge conflicts)

- One branch per person (`a/...`, `b/...`, `c/...`, `d/...`); merge to `main` small and often. **`main` must always run** (`pytest` passes).
- Only edit files in your own folder and your own `data/` outputs.
- Shared files: `requirements.txt` (append to your section, don't reorder), `src/common/contracts.py`, `main.py` (stage list) and this README. Announce changes to these before pushing.
- Never commit keys. Copy `.env.example` to `.env`.
- Raw downloads and API caches go in `data/raw/` or `data/cache/` (git-ignored).
- IDE settings (`.idea/`, `.vscode/`) are git-ignored because they hold per-machine paths.

## The area list (discovered research groups)

Areas are not a hand-written list. B discovers research groups from the papers'
SPECTER2 embeddings, freezes them, and names them by hand. Every `area` column
(B's network and forecast files, D's crowding) must use the `name` values in
`data/groups.csv` (`data/sample/groups.csv` until real output exists).

```python
from src.common.contracts import area_names
area_names()          # current group names; AREAS still works and returns the same list
```

To place new papers (for example D's arXiv papers) into the groups:

```python
from src.research_map.encoding.groups import assign_groups
assign_groups(titles, abstracts)   # columns: area, area_score, group_ids, group_scores
```

This needs the B packages (`torch`, `transformers`, `adapters`) and downloads
SPECTER2 on first use. The frozen model is `data/group_model.npz`. B only refits
the groups (`--refit-groups`) when the corpus changes a lot; that renumbers and
renames them, so B announces it first.

## File contracts (fixed)

| File | Owner | Columns |
|---|---|---|
| `data/factors.csv` | A | date, signal, return |
| `data/papers.csv` | A | signal, paper, publication_date, date_rule, ken_french_column, source_verified |
| `data/decay_results.csv` | A | signal, paper, publication_date, source, pre_mean, post_mean, share_kept, half_life_months, ci_low, ci_high, early_citations (key: source + signal) |
| `data/groups.csv` | B | group_id, name, keywords |
| `data/network_nodes.csv` | B | paper_id, title, year, area, citations, lifecycle_stage |
| `data/network_edges.csv` | B | source_id, target_id, shared, strength |
| `data/area_forecasts.csv` | B | area, year, growth_score, boom_probability, predicted_boom, actual_boom |
| `data/paper_encodings.csv` | B | paper_id, top_topic, topic_ids, topic_scores, topic_labels, model, text_quality |
| `data/crowding.csv` | D | date, area, paper_count, crowding_score |
| `data/backtest_results.csv` | D | date, strategy, monthly_return, cumulative_return |
| `data/briefing.txt`, `data/briefing.mp3` | D | text and audio |

The code version is `src/common/contracts.py`:

```python
from src.common.contracts import load, save

df = load("decay_results")   # real data/decay_results.csv if present, else data/sample/
save("crowding", my_df)      # validates columns and area names (from groups.csv), then writes data/crowding.csv
```

Made-up sample files for every contract are in `data/sample/`, so the app can be built before real data lands.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
pytest
```

Python 3.12. Versions in `requirements.txt` are pinned; pin anything you add.

## Rebuild every output

```bash
python main.py            # runs each stage in order: A, then B, then D
python main.py --list     # shows which stages exist yet
python main.py --only d   # one stage
```

Each stage is a runner module: A `src.signal_decay.run_decay` then `src.signal_decay.combined`, B `src.research_map.run_map`, D `src.live.run_live`. Stages that aren't written yet are skipped. Some stages call live APIs and need keys in `.env`.

GitHub Actions runs `pytest` on every push and pull request (`.github/workflows/tests.yml`). Don't merge to `main` while it's red.
