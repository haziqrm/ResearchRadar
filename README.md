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

## Repo rules (to avoid merge conflicts)

- One branch per person (`a/...`, `b/...`, `c/...`, `d/...`); merge to `main` small and often. **`main` must always run** (`pytest` passes).
- Only edit files in your own folder and your own `data/` outputs.
- Shared files: `requirements.txt` (append to your section, don't reorder), `src/common/contracts.py`, `main.py` (stage list) and this README. Announce changes to these before pushing.
- Never commit keys. Copy `.env.example` to `.env`.
- Raw downloads and API caches go in `data/raw/` or `data/cache/` (git-ignored).
- IDE settings (`.idea/`, `.vscode/`) are git-ignored because they hold per-machine paths.

## The area list (fixed)

B and D must use these exact strings. They live in code as `src.common.contracts.AREAS`.

1. Factor and anomaly research
2. Machine learning for returns
3. Market microstructure
4. Portfolio and risk
5. Derivatives and volatility
6. Alternative data and text
7. Crypto and DeFi
8. LLMs in finance

## File contracts (fixed)

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

The code version is `src/common/contracts.py`:

```python
from src.common.contracts import load, save

df = load("decay_results")   # real data/decay_results.csv if present, else data/sample/
save("crowding", my_df)      # validates columns and area names, then writes data/crowding.csv
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

Each stage is a runner module: A `src.signal_decay.run_decay`, B `src.research_map.run_map`, D `src.live.run_live`. Stages that aren't written yet are skipped. Some stages call live APIs and need keys in `.env`.

GitHub Actions runs `pytest` on every push and pull request (`.github/workflows/tests.yml`). Don't merge to `main` while it's red.
