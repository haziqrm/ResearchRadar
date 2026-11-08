"""H13: the H11 forecast at sub-topic level (75 units instead of B's 25 groups).

Same split-half design, features, walk-forward, refit-per-origin test and noise worlds as
H11 (see HYPOTHESES.md, H13). Boom = top 15 of 75. Live output names each sub-topic by
its title terms and its parent B group.

Run: python -m src.topic_radar.subtopic_forecast
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.research_map.encoding.labels import topic_labels
from src.topic_radar import DATA_DIR
from src.topic_radar.forecast import (
    LIVE_ORIGIN, ORIGINS, build_panel, evaluate, feature_half, live_forecast, null_worlds,
    per_origin_metrics, walk_forward,
)
from src.topic_radar.panel import Corpus, group_view, load_corpus, refit_model

K_SUB = 75
TOP_SUB = 15
BASE_RATE_SUB = TOP_SUB / K_SUB
PRIOR = range(2020, 2023)
RECENT = range(2023, 2026)
SUBTOPIC_FORECASTS_CSV = DATA_DIR / "subtopic_forecasts.csv"
SUBTOPIC_EVAL_CSV = DATA_DIR / "subtopic_forecast_eval.csv"
SUBTOPIC_NULL_CSV = DATA_DIR / "subtopic_forecast_null.csv"  # one row per noise world


def describe_units(corpus: Corpus, labels: np.ndarray, k: int = K_SUB) -> pd.DataFrame:
    """Per unit: title-term label, parent B group (majority), paper counts 2020-22 / 2023-25."""
    names = topic_labels(list(corpus.title), labels, n_terms=4, n_topics=k)
    rows = []
    for u in range(k):
        mine = labels == u
        parents = pd.Series(corpus.frozen_group[mine]).map(corpus.group_names)
        rows.append({"group": u, "label": names[u],
                     "parent_group": parents.mode().iloc[0] if len(parents) else "",
                     "parent_share": float(parents.value_counts(normalize=True).iloc[0]) if len(parents) else np.nan,
                     "n_papers": int(mine.sum()),
                     "papers_2020_22": int((mine & np.isin(corpus.year, list(PRIOR))).sum()),
                     "papers_2023_25": int((mine & np.isin(corpus.year, list(RECENT))).sum())})
    return pd.DataFrame(rows)


def run(corpus: Corpus | None = None, n_worlds: int | None = None) -> dict:
    corpus = corpus or load_corpus()
    models = {t: refit_model(corpus, t, k=K_SUB) for t in ORIGINS}
    final = refit_model(corpus, LIVE_ORIGIN, k=K_SUB)  # fitted on papers <= 2025 (secondary + live)
    view = group_view(final, corpus.vectors)
    shares = np.bincount(view["group"], minlength=K_SUB).astype(float) + 1.0
    kw = {} if n_worlds is None else {"n_worlds": n_worlds}
    null = null_worlds(corpus.year, feature_half(corpus.year), view["top_prob"].to_numpy(),
                       view["dist"].to_numpy(), corpus.edges, shares, top=TOP_SUB, **kw)
    results, evals = {"null": null}, []
    for mode in ["refit", "frozen"]:
        panel, labels = build_panel(corpus, mode, models=models if mode == "refit" else None,
                                    k=K_SUB, top=TOP_SUB, frozen=final if mode == "frozen" else None)
        pred = walk_forward(panel, base_rate=BASE_RATE_SUB)
        metrics = per_origin_metrics(pred, top=TOP_SUB, base_rate=BASE_RATE_SUB)
        evals.append(evaluate(metrics, mode, null=null, top=TOP_SUB, k=K_SUB, pass_column="h13_pass"))
        results[mode] = {"panel": panel, "pred": pred, "metrics": metrics}
    evaluation = pd.concat(evals, ignore_index=True)
    evaluation.to_csv(SUBTOPIC_EVAL_CSV, index=False)
    null.to_csv(SUBTOPIC_NULL_CSV, index=False)

    live = live_forecast(results["refit"]["panel"], corpus, frozen=final, k=K_SUB, top=TOP_SUB)
    units = describe_units(corpus, view["group"].to_numpy())
    out = units.merge(live[["group", "growth_score", "boom_probability"]], on="group")
    out = out.sort_values("growth_score", ascending=False).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    out["predicted_boom"] = (out["rank"] <= TOP_SUB).astype(int)
    out = out.rename(columns={"group": "subtopic_id"}).round(
        {"growth_score": 4, "boom_probability": 4, "parent_share": 3})
    out.to_csv(SUBTOPIC_FORECASTS_CSV, index=False)
    results.update({"eval": evaluation, "live": out})
    return results


def main() -> None:
    res = run()
    ev = res["eval"]
    show = ["metric", "model", "groups_mode", "value", "noise_world_p95", "holm_p"]
    print(ev[ev["metric"].isin(["spearman", "pass_test", f"precision_at_{TOP_SUB}"])][show].round(3).to_string(index=False))
    print("H13 pass:", bool(ev["h13_pass"].dropna().iloc[0]))
    cols = ["rank", "label", "parent_group", "papers_2020_22", "papers_2023_25", "growth_score"]
    print(res["live"].head(15)[cols].to_string(index=False))


if __name__ == "__main__":
    main()
