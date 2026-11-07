"""H7: predict how much of its in-sample edge a published signal keeps.

Rules are fixed in HYPOTHESES.md (H7): time split (train <= 2006, test 2007+),
features known at publication, a constant baseline, ridge and a small random
forest, and a paper-level bootstrap on the test-set error.

Usage:
    python -m src.signal_decay.decay_model [--charts]

Needs data/osap_papers.csv (from run_osap) for the attention feature, and the
cached OSAP downloads. Writes A-owned files only:
    data/osap_model_summary.csv      one row per model: test MAE, CI vs baseline, verdict
    data/osap_model_predictions.csv  test-set predictions per signal
    data/osap_model_importance.csv   permutation importance on the test set
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.signal_decay import CHARTS_DIR, DATA_DIR
from src.signal_decay.osap import load_ls_returns, load_signal_doc, period_panel, predictor_table

TRAIN_LAST_YEAR = 2006
TARGET_CLIP = (-1.0, 2.0)
SEED = 0
BOOTSTRAP_DRAWS = 2000

SUMMARY_CSV = DATA_DIR / "osap_model_summary.csv"
PREDICTIONS_CSV = DATA_DIR / "osap_model_predictions.csv"
IMPORTANCE_CSV = DATA_DIR / "osap_model_importance.csv"
OSAP_PAPERS_CSV = DATA_DIR / "osap_papers.csv"

DATA_TYPES = ["Accounting", "Price", "Analyst", "Trading", "Other"]
TOP3 = {"JF", "JFE", "RFS"}
ACCOUNTING_JOURNALS = {"AR", "JAE", "JAR", "RAS", "CAR"}
NUMERIC = ["op_tstat", "is_mean", "is_tstat", "sample_years", "pub_lag", "pub_year", "citation_pct"]
IMPUTED = ["op_tstat", "citation_pct"]


def build_dataset(doc: pd.DataFrame, returns: pd.DataFrame, osap_papers: pd.DataFrame | None) -> pd.DataFrame:
    """One row per usable signal: raw features, target and grouping keys."""
    pt = predictor_table(doc)
    panel = period_panel(returns[returns["signal"].isin(pt["signal"])], pt)
    g = panel.groupby(["signal", "period"])["return"].agg(["mean", "std", "count"]).unstack("period")
    s = pd.DataFrame({
        "is_mean": g[("mean", "IS")], "is_std": g[("std", "IS")], "n_is": g[("count", "IS")],
        "post_mean": g.get(("mean", "POST")), "n_post": g.get(("count", "POST")),
    }).reset_index()
    s["is_tstat"] = s["is_mean"] / (s["is_std"] / np.sqrt(s["n_is"]))

    d = doc[doc["Cat.Signal"] == "Predictor"].rename(columns={"Acronym": "signal"})
    df = pt.merge(s, on="signal").merge(
        d[["signal", "Cat.Data", "Cat.Form", "Predictability in OP", "Stock Weight", "T-Stat"]], on="signal")
    df["paper_key"] = df["authors"] + "|" + df["year"].astype(str) + "|" + df["journal"]

    if osap_papers is not None and "citation_percentile" in osap_papers:
        op = osap_papers.assign(paper_key=osap_papers["authors"] + "|" + osap_papers["year"].astype(str)
                                + "|" + osap_papers["journal"])
        df = df.merge(op[["paper_key", "citation_percentile"]], on="paper_key", how="left")
    else:
        df["citation_percentile"] = np.nan

    df = df[(df["is_mean"] > 0) & (df["n_post"].fillna(0) > 0)].copy()
    df["target"] = (df["post_mean"] / df["is_mean"]).clip(*TARGET_CLIP)
    df["data_type"] = df["Cat.Data"].where(df["Cat.Data"].isin(DATA_TYPES[:-1]), "Other")
    df["discrete"] = (df["Cat.Form"] == "discrete").astype(float)
    df["clear"] = (df["Predictability in OP"] == "1_clear").astype(float)
    df["equal_weighted"] = (df["Stock Weight"] == "EW").astype(float)
    df["op_tstat"] = pd.to_numeric(df["T-Stat"], errors="coerce")
    df["sample_years"] = df["sample_end"] - df["sample_start"] + 1
    df["pub_lag"] = df["year"] - df["sample_end"]
    df["pub_year"] = df["year"].astype(float)
    df["journal_group"] = np.where(df["journal"].isin(TOP3), "top3",
                                   np.where(df["journal"].isin(ACCOUNTING_JOURNALS), "accounting", "other"))
    df["citation_pct"] = pd.to_numeric(df["citation_percentile"], errors="coerce")
    return df.reset_index(drop=True)


def feature_matrix(df: pd.DataFrame, train_medians: dict[str, float]) -> pd.DataFrame:
    """Model-ready features. Missing values filled with *training* medians, plus flags."""
    X = pd.DataFrame(index=df.index)
    for col in NUMERIC:
        X[col] = df[col]
    for col in IMPUTED:
        X[f"{col}_missing"] = X[col].isna().astype(float)
        X[col] = X[col].fillna(train_medians[col])
    for t in DATA_TYPES:
        X[f"data_{t.lower()}"] = (df["data_type"] == t).astype(float)
    for gname in ["top3", "accounting"]:
        X[f"journal_{gname}"] = (df["journal_group"] == gname).astype(float)
    for col in ["discrete", "clear", "equal_weighted"]:
        X[col] = df[col]
    return X


def paper_bootstrap_ci(err_model: np.ndarray, err_base: np.ndarray, groups: np.ndarray,
                       draws: int = BOOTSTRAP_DRAWS, seed: int = SEED) -> tuple[float, float]:
    """95% CI of MAE(model) - MAE(baseline), resampling test papers with replacement."""
    rng = np.random.default_rng(seed)
    papers = np.unique(groups)
    idx = {p: np.flatnonzero(groups == p) for p in papers}
    diffs = []
    for _ in range(draws):
        rows = np.concatenate([idx[p] for p in rng.choice(papers, size=len(papers), replace=True)])
        diffs.append(err_model[rows].mean() - err_base[rows].mean())
    return tuple(np.percentile(diffs, [2.5, 97.5]))


def fit_and_evaluate(df: pd.DataFrame, train_last_year: int = TRAIN_LAST_YEAR) -> dict:
    """Train on papers up to train_last_year, evaluate on later papers (H7)."""
    train, test = df[df["year"] <= train_last_year], df[df["year"] > train_last_year]
    medians = {c: float(train[c].median()) for c in IMPUTED}
    Xtr, Xte = feature_matrix(train, medians), feature_matrix(test, medians)
    ytr, yte = train["target"].to_numpy(), test["target"].to_numpy()
    groups_tr, groups_te = train["paper_key"].to_numpy(), test["paper_key"].to_numpy()

    preds = {"baseline": np.full(len(test), np.median(ytr))}
    ridge = GridSearchCV(
        make_pipeline(StandardScaler(), Ridge()),
        {"ridge__alpha": np.logspace(-2, 3, 30)},
        cv=GroupKFold(n_splits=5), scoring="neg_mean_absolute_error",
    ).fit(Xtr, ytr, groups=groups_tr)
    preds["ridge"] = ridge.predict(Xte)
    forest = RandomForestRegressor(n_estimators=500, max_depth=3, min_samples_leaf=5,
                                   random_state=SEED).fit(Xtr, ytr)
    preds["random_forest"] = forest.predict(Xte)

    err_base = np.abs(preds["baseline"] - yte)
    rows = []
    for name, p in preds.items():
        err = np.abs(p - yte)
        row = {"model": name, "test_mae": err.mean(), "n_train": len(train), "n_test": len(test),
               "n_test_papers": len(np.unique(groups_te)),
               "spearman_pred_vs_actual": stats.spearmanr(p, yte).statistic if np.std(p) > 0 else np.nan}
        if name == "baseline":
            row.update(mae_diff_vs_baseline=0.0, diff_ci_low=np.nan, diff_ci_high=np.nan,
                       verdict=f"predicts {np.median(ytr):.2f} for every signal")
        else:
            lo, hi = paper_bootstrap_ci(err, err_base, groups_te)
            row.update(mae_diff_vs_baseline=err.mean() - err_base.mean(), diff_ci_low=lo, diff_ci_high=hi,
                       verdict="beats baseline" if hi < 0 else "does not beat baseline")
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary["ridge_alpha"] = [None, ridge.best_params_["ridge__alpha"], None]

    imp = []
    for name, model in [("ridge", ridge.best_estimator_), ("random_forest", forest)]:
        r = permutation_importance(model, Xte, yte, n_repeats=50, random_state=SEED,
                                   scoring="neg_mean_absolute_error")
        imp += [{"model": name, "feature": f, "importance_mae": m, "std": s}
                for f, m, s in zip(Xte.columns, r.importances_mean, r.importances_std)]

    predictions = test[["signal", "authors", "year", "journal", "target"]].rename(
        columns={"target": "actual_retained"})
    for name, p in preds.items():
        predictions[f"pred_{name}"] = p
    return {"summary": summary, "predictions": predictions.reset_index(drop=True),
            "importance": pd.DataFrame(imp).sort_values(["model", "importance_mae"], ascending=[True, False])}


def plot_predictions(predictions: pd.DataFrame, summary: pd.DataFrame, path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    for ax, name in zip(axes, ["ridge", "random_forest"]):
        row = summary.set_index("model").loc[name]
        ax.scatter(predictions[f"pred_{name}"], predictions["actual_retained"], s=18, alpha=0.7)
        ax.axhline(predictions["pred_baseline"].iloc[0], color="grey", ls="--", lw=1, label="baseline")
        ax.plot([-1, 2], [-1, 2], color="black", lw=0.8, label="perfect")
        ax.set_xlim(-0.2, 1.5)
        ax.set_title(f"{name}: test MAE {row['test_mae']:.2f} vs baseline "
                     f"{summary.set_index('model').loc['baseline', 'test_mae']:.2f}\n{row['verdict']}")
        ax.set_xlabel("Predicted share of edge kept")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Actual share of edge kept (clipped -1 to 2)")
    fig.suptitle(f"Decay predictor, trained on papers to {TRAIN_LAST_YEAR}, "
                 f"tested on {int(predictions['year'].min())}-{int(predictions['year'].max())}")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="H7: train and test the decay predictor.")
    parser.add_argument("--charts", action="store_true")
    args = parser.parse_args(argv)

    osap_papers = pd.read_csv(OSAP_PAPERS_CSV) if OSAP_PAPERS_CSV.exists() else None
    if osap_papers is None:
        print("data/osap_papers.csv not found; run run_osap first. Attention feature will be missing.")
    df = build_dataset(load_signal_doc(), load_ls_returns(), osap_papers)
    res = fit_and_evaluate(df)
    res["summary"].to_csv(SUMMARY_CSV, index=False)
    res["predictions"].to_csv(PREDICTIONS_CSV, index=False)
    res["importance"].to_csv(IMPORTANCE_CSV, index=False)
    print(f"Wrote {SUMMARY_CSV}, {PREDICTIONS_CSV}, {IMPORTANCE_CSV}")
    if args.charts:
        CHARTS_DIR.mkdir(parents=True, exist_ok=True)
        plot_predictions(res["predictions"], res["summary"], CHARTS_DIR / "osap_decay_model.png")
        print(f"Saved chart to {CHARTS_DIR / 'osap_decay_model.png'}")


if __name__ == "__main__":
    main()
