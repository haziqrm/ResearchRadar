"""H8: second attempt at predicting decay, with trading-cost, novelty and crowding inputs.

Rules are fixed in HYPOTHESES.md (H8). Adds three features to the H7 set:
    liquid_share    IS edge in big stocks / IS edge in the original portfolio
    prior_max_corr  largest |correlation| with any earlier-published predictor
    category_heat   same-category predictors published in the 5 years before
and scores models with an expanding-window walk-forward over three test folds.

Usage:
    python -m src.signal_decay.decay_model_h8 [--charts]

Writes A-owned files only:
    data/osap_h8_summary.csv       per fold and pooled: MAE, CI vs baseline, verdict
    data/osap_h8_predictions.csv   out-of-sample predictions per signal and fold
    data/osap_h8_features.csv      the three new feature values per signal
    data/osap_h8_coefficients.csv  ridge coefficients (last fold) and permutation importance
"""

from __future__ import annotations

import argparse
from pathlib import Path

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
from src.signal_decay.decay_model import (
    IMPUTED, OSAP_PAPERS_CSV, SEED, TARGET_CLIP, build_dataset, feature_matrix, paper_bootstrap_ci,
)
from src.signal_decay.osap import OSAP_CACHE_DIR, _client, load_ls_returns, load_signal_doc, predictor_table

LIQUID_PORTFOLIO = "ex_nyse_p20_me"  # stocks above the NYSE 20th size percentile
MIN_MONTHS = 60
HEAT_YEARS = 5
FOLDS = [(1996, 2000), (2001, 2006), (2007, 2016)]
H8_FEATURES = ["liquid_share", "prior_max_corr", "category_heat"]
H8_IMPUTED = ["liquid_share", "prior_max_corr"]

SUMMARY_CSV = DATA_DIR / "osap_h8_summary.csv"
PREDICTIONS_CSV = DATA_DIR / "osap_h8_predictions.csv"
FEATURES_CSV = DATA_DIR / "osap_h8_features.csv"
COEFFICIENTS_CSV = DATA_DIR / "osap_h8_coefficients.csv"


def load_liquid_ls_returns(cache_dir: Path = OSAP_CACHE_DIR, refresh: bool = False) -> pd.DataFrame:
    """Liquidity-screened long-short returns, tidy (date, signal, return), same format as load_ls_returns."""
    path = cache_dir / f"{LIQUID_PORTFOLIO}_ports.parquet"
    if path.exists() and not refresh:
        ports = pd.read_parquet(path)
    else:
        cache_dir.mkdir(parents=True, exist_ok=True)
        ports = _client().dl_port(LIQUID_PORTFOLIO, "pandas")
        ports.to_parquet(path)
    ls = ports[ports["port"].astype(str) == "LS"]
    out = pd.DataFrame({
        "date": pd.to_datetime(ls["date"]) + pd.offsets.MonthEnd(0),
        "signal": ls["signalname"].astype(str),
        "return": ls["ret"].astype(float) / 100.0,
    }).dropna()
    return out.sort_values(["signal", "date"]).reset_index(drop=True)


# ---------- new features ----------

def compute_h8_features(doc: pd.DataFrame, returns: pd.DataFrame, liquid: pd.DataFrame) -> pd.DataFrame:
    """liquid_share, prior_max_corr and category_heat for every predictor (pre-publication data only)."""
    pt = predictor_table(doc)
    cats = doc[doc["Cat.Signal"] == "Predictor"].set_index("Acronym")["Cat.Economic"].astype(str)
    pt["category"] = pt["signal"].map(cats)
    wide = returns.pivot(index="date", columns="signal", values="return").sort_index()
    wide_liq = liquid.pivot(index="date", columns="signal", values="return").reindex(wide.index)
    years = wide.index.year

    rows = []
    for r in pt.itertuples(index=False):
        in_is = (years >= r.sample_start) & (years <= r.sample_end)
        op = wide[r.signal][in_is] if r.signal in wide else pd.Series(dtype=float)

        liquid_share = np.nan
        if r.signal in wide_liq:
            both = pd.concat([op, wide_liq[r.signal][in_is]], axis=1).dropna()
            if len(both) >= MIN_MONTHS and both.iloc[:, 0].mean() > 0:
                liquid_share = float(np.clip(both.iloc[:, 1].mean() / both.iloc[:, 0].mean(), *TARGET_CLIP))

        prior = [s for s in pt.loc[pt["year"] < r.year, "signal"] if s in wide and s != r.signal]
        prior_max_corr = np.nan
        if prior and op.notna().sum() >= MIN_MONTHS:
            sub = wide.loc[in_is, prior]
            overlap = sub.notna().mul(op.notna(), axis=0).sum()
            ok = overlap[overlap >= MIN_MONTHS].index
            if len(ok):
                prior_max_corr = float(sub[ok].corrwith(op).abs().max())

        heat = int(((pt["category"] == r.category) & (pt["year"] >= r.year - HEAT_YEARS)
                    & (pt["year"] <= r.year - 1)).sum())
        rows.append({"signal": r.signal, "liquid_share": liquid_share,
                     "prior_max_corr": prior_max_corr, "category_heat": heat})
    return pd.DataFrame(rows)


def feature_matrix_h8(df: pd.DataFrame, medians: dict[str, float], include_h8: bool = True) -> pd.DataFrame:
    """H7 features, plus the H8 ones (training-fold medians for missing values, plus flags)."""
    X = feature_matrix(df, medians)
    if include_h8:
        for col in H8_FEATURES:
            X[col] = df[col].astype(float)
        for col in H8_IMPUTED:
            X[f"{col}_missing"] = X[col].isna().astype(float)
            X[col] = X[col].fillna(medians[col])
    return X


# ---------- walk-forward ----------

def _ridge(Xtr, ytr, groups):
    n_splits = min(5, len(np.unique(groups)))
    return GridSearchCV(make_pipeline(StandardScaler(), Ridge()), {"ridge__alpha": np.logspace(-2, 3, 30)},
                        cv=GroupKFold(n_splits=n_splits), scoring="neg_mean_absolute_error",
                        ).fit(Xtr, ytr, groups=groups)


def fit_fold(train: pd.DataFrame, test: pd.DataFrame) -> tuple[dict[str, np.ndarray], dict]:
    """Fit every model on one training fold and predict its test fold."""
    medians = {c: float(train[c].median()) for c in IMPUTED + H8_IMPUTED}
    ytr, groups = train["target"].to_numpy(), train["paper_key"].to_numpy()
    Xtr, Xte = feature_matrix_h8(train, medians), feature_matrix_h8(test, medians)
    Xtr7, Xte7 = feature_matrix_h8(train, medians, False), feature_matrix_h8(test, medians, False)

    ridge = _ridge(Xtr, ytr, groups)
    ridge7 = _ridge(Xtr7, ytr, groups)
    forest = RandomForestRegressor(n_estimators=500, max_depth=3, min_samples_leaf=5,
                                   random_state=SEED).fit(Xtr, ytr)
    preds = {
        "baseline": np.full(len(test), np.median(ytr)),
        "ridge_h8": ridge.predict(Xte),
        "random_forest_h8": forest.predict(Xte),
        "ridge_h7_only": ridge7.predict(Xte7),
    }
    fitted = {"ridge": ridge, "forest": forest, "Xte": Xte, "yte": test["target"].to_numpy(),
              "alpha_h8": ridge.best_params_["ridge__alpha"], "alpha_h7": ridge7.best_params_["ridge__alpha"]}
    return preds, fitted


def _score(name: str, pred: np.ndarray, y: np.ndarray, base: np.ndarray, groups: np.ndarray) -> dict:
    err, err_base = np.abs(pred - y), np.abs(base - y)
    row = {"model": name, "test_mae": err.mean(), "n_test": len(y), "n_test_papers": len(np.unique(groups)),
           "spearman_pred_vs_actual": stats.spearmanr(pred, y).statistic if np.std(pred) > 0 else np.nan}
    if name == "baseline":
        row.update(mae_diff_vs_baseline=0.0, diff_ci_low=np.nan, diff_ci_high=np.nan, verdict="")
    else:
        lo, hi = paper_bootstrap_ci(err, err_base, groups)
        row.update(mae_diff_vs_baseline=err.mean() - err_base.mean(), diff_ci_low=lo, diff_ci_high=hi,
                   verdict="beats baseline" if hi < 0 else "does not beat baseline")
    return row


def walk_forward(df: pd.DataFrame, folds: list[tuple[int, int]] = FOLDS) -> dict:
    """Expanding-window evaluation (H8). Returns summary, predictions and last-fold coefficients."""
    pred_frames, rows, fitted = [], [], None
    for start, end in folds:
        train = df[df["year"] < start]
        test = df[(df["year"] >= start) & (df["year"] <= end)]
        preds, fitted = fit_fold(train, test)
        label = f"{start}-{end}"
        frame = test[["signal", "authors", "year", "journal", "paper_key", "target"]].copy()
        frame.insert(0, "fold", label)
        for name, p in preds.items():
            frame[f"pred_{name}"] = p
        pred_frames.append(frame)
        for name, p in preds.items():
            row = _score(name, p, frame["target"].to_numpy(), preds["baseline"], frame["paper_key"].to_numpy())
            rows.append({"fold": label, "n_train": len(train), **row})
        rows[-4]["verdict"] = f"predicts {preds['baseline'][0]:.2f} for every signal"

    allp = pd.concat(pred_frames, ignore_index=True)
    y, groups = allp["target"].to_numpy(), allp["paper_key"].to_numpy()
    for name in ["baseline", "ridge_h8", "random_forest_h8", "ridge_h7_only"]:
        rows.append({"fold": "pooled", "n_train": np.nan,
                     **_score(name, allp[f"pred_{name}"].to_numpy(), y, allp["pred_baseline"].to_numpy(), groups)})
    summary = pd.DataFrame(rows)

    coef_rows = []
    pipe = fitted["ridge"].best_estimator_
    for f, c in zip(fitted["Xte"].columns, pipe.named_steps["ridge"].coef_):
        coef_rows.append({"model": "ridge_h8", "feature": f, "kind": "std_coefficient_last_fold", "value": c})
    for name, model in [("ridge_h8", pipe), ("random_forest_h8", fitted["forest"])]:
        imp = permutation_importance(model, fitted["Xte"], fitted["yte"], n_repeats=50, random_state=SEED,
                                     scoring="neg_mean_absolute_error")
        coef_rows += [{"model": name, "feature": f, "kind": "permutation_importance_last_fold", "value": m}
                      for f, m in zip(fitted["Xte"].columns, imp.importances_mean)]
    coefs = pd.DataFrame(coef_rows)
    coefs.attrs["alphas"] = (fitted["alpha_h8"], fitted["alpha_h7"])
    return {"summary": summary, "predictions": allp.rename(columns={"target": "actual_retained"}),
            "coefficients": coefs}


def plot_walk_forward(summary: pd.DataFrame, path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = ["baseline", "ridge_h7_only", "ridge_h8", "random_forest_h8"]
    folds = list(dict.fromkeys(summary["fold"]))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    w = 0.2
    for i, m in enumerate(models):
        s = summary[summary["model"] == m].set_index("fold").reindex(folds)
        ax.bar(np.arange(len(folds)) + (i - 1.5) * w, s["test_mae"], w, label=m)
    ax.set_xticks(np.arange(len(folds)), folds)
    ax.set_ylabel("Mean absolute error (lower is better)")
    pooled = summary[(summary["fold"] == "pooled") & (summary["model"] == "ridge_h8")].iloc[0]
    ax.set_title(f"H8 walk-forward. Pooled ridge H8 vs baseline: {pooled['mae_diff_vs_baseline']:+.3f} "
                 f"[{pooled['diff_ci_low']:+.3f}, {pooled['diff_ci_high']:+.3f}], {pooled['verdict']}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="H8: walk-forward decay predictor with new inputs.")
    parser.add_argument("--charts", action="store_true")
    args = parser.parse_args(argv)

    doc, returns = load_signal_doc(), load_ls_returns()
    feats = compute_h8_features(doc, returns, load_liquid_ls_returns())
    feats.to_csv(FEATURES_CSV, index=False)
    osap_papers = pd.read_csv(OSAP_PAPERS_CSV) if OSAP_PAPERS_CSV.exists() else None
    df = build_dataset(doc, returns, osap_papers).merge(feats, on="signal", how="left")
    res = walk_forward(df)
    res["summary"].to_csv(SUMMARY_CSV, index=False)
    res["predictions"].to_csv(PREDICTIONS_CSV, index=False)
    res["coefficients"].to_csv(COEFFICIENTS_CSV, index=False)
    print(f"Wrote {SUMMARY_CSV}, {PREDICTIONS_CSV}, {FEATURES_CSV}, {COEFFICIENTS_CSV}")
    if args.charts:
        CHARTS_DIR.mkdir(parents=True, exist_ok=True)
        plot_walk_forward(res["summary"], CHARTS_DIR / "osap_h8_walk_forward.png")
        print(f"Saved chart to {CHARTS_DIR / 'osap_h8_walk_forward.png'}")


if __name__ == "__main__":
    main()
