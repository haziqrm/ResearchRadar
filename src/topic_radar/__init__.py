"""Topic radar (Person A): which research groups grow next, what drives them, which papers to read.

Reads B's committed files (network nodes and edges, paper encodings, paper vectors, group
model) without changing them. Three levels:

* ``forecast``  - H11, ranks B's 25 groups by expected growth in share of new papers.
* ``subtopics`` - descriptive sub-topics inside each group, with which ones are rising.
* ``seeds``     - H12, recent papers that later work is expected to cluster around.

See HYPOTHESES.md for the definitions and pass rules, written before any real run.
"""

from src.common.contracts import DATA_DIR

FORECAST_EVAL_CSV = DATA_DIR / "topic_forecast_eval.csv"
SUBTOPICS_CSV = DATA_DIR / "subtopics.csv"
SEED_PAPERS_CSV = DATA_DIR / "seed_papers.csv"
SEED_EVAL_CSV = DATA_DIR / "seed_eval.csv"
