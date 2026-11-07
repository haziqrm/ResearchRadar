# Signal decay: hypotheses and test rules

Written before the era-adjusted citations, the attention test, the placebo test or
the bootstrap intervals were computed. Already known when this was written: the
plain before/after means for the 7 signals, and their raw early-citation counts.

## Data
- Returns: Ken French monthly long-short series, to the latest month available.
- Signals: the verified rows of `data/papers.csv` (7 signals).
- Split: a month is "post" if its month-end is on or after `publication_date`.

## H1. Publication lowers returns (replication of McLean and Pontiff 2016)
- Statistic: post_mean - pre_mean per signal, 95% Newey-West interval (12 lags).
- Expectation: negative for most signals. This is a replication, not a new finding.

## H2. The drop is tied to the publication date, not just to time passing (placebo)
- For each signal, compute the same difference at every admissible fake split month
  (at least 120 months on each side).
- Per-signal placebo p = share of fake splits whose difference is at least as
  negative as the real one.
- Pooled test: mean real difference across signals, against the mean of one random
  fake split per signal, 10,000 draws, seed 0. One-sided p.
- Expectation: pooled p < 0.10. Per-signal p values are descriptive only.

## H3. More early attention goes with faster decay (the new part)
- Attention: early citations (citing works published in years 1 to 3 after the
  paper's publication year), as a **percentile among peers**: articles in the same
  journal and same OpenAlex publication year, with at least one citation ever
  (to drop front matter), random sample of up to 50, seed 0, same citation window.
- Decay: relative decay = 1 - post_mean / pre_mean (share of the edge lost).
- Test: Spearman rank correlation, exact one-sided permutation p over all 5,040
  orderings of 7 signals.
- Expectation: positive correlation. With 7 signals this is **suggestive at best**;
  a null result is the likely outcome and will be reported as such.

## Half-life reporting rule
- Point estimate: exponential fit to the 36-month rolling post-publication mean.
- Interval: residual block bootstrap (12-month blocks, 1,000 draws, seed 0):
  resample the residuals around the fitted monthly path, add them back, refit.
  2.5th to 97.5th percentile of successful fits. If the point estimate finds no
  decay, no interval is computed.
  - Correction made before any real-data run: the first version resampled the
    raw post-publication returns, which scrambles their order and erases the
    trend. A synthetic test with a known 60-month half-life found decay in only
    26% of draws, so the method was switched to residuals.
- Null check: resample residuals around a flat line (no decay), 1,000 draws, and
  record how often the fit finds a decay rate at least as steep as observed
  (`half_life_null_p`). Added before any real-data run, after a synthetic test showed
  the residual bootstrap alone accepts a curve fitted to pure noise.
- `half_life_months` in `decay_results.csv` is filled only if half_life_null_p < 0.10,
  at least 80% of bootstrap fits find decay, and the upper bound is at most 600
  months. Otherwise it is left empty and the reason is written to `decay_details.csv`.
- Also reported: retained share = post_mean / pre_mean, with a block-bootstrap
  interval (pre and post resampled separately).

## What would change the conclusions
- H2 pooled p >= 0.10: the drop may just be returns falling over time.
- H3 correlation <= 0 or p >= 0.10: no evidence that attention speeds up decay.

---

# OSAP extension (written before any OSAP decay result was computed)

Known when this was written: the OSAP signal documentation (212 predictors from
141 papers, published 1973-2016, every sample end year before the publication
year) and that long-short returns run 1926-01 to 2025-12. No window means, decay
estimates or citation counts had been computed.

## Data
- Signals: `Cat.Signal == "Predictor"` in the OSAP 2026.10 release (212 signals).
- Returns: original-paper long-short portfolios (`op`, port `LS`), percent / 100,
  dated to month-end. No trimming or winsorizing.
- OSAP gives the publication **year** only, so the date rule is "end of
  publication year": post-publication starts in January of Year + 1.

## Three periods (McLean and Pontiff 2016)
- In-sample (IS): January of SampleStartYear to December of SampleEndYear.
- Out-of-sample, pre-publication (OOS): January of SampleEndYear + 1 to December of Year.
- Post-publication (POST): January of Year + 1 onward.
- Months before SampleStartYear are dropped.

## H4. Returns fall after the sample ends, and fall further after publication
- Scale each signal's monthly returns by its IS mean. Signals with IS mean <= 0
  are excluded (count reported).
- Pooled OLS: scaled return = const + b_oos * OOS + b_post * POST, standard errors
  clustered by month.
- Expectation: b_oos < 0 and b_post < b_oos (an extra drop at publication beyond
  overfitting). McLean and Pontiff found about -0.26 and -0.58.
- Key test: b_post - b_oos < 0, one-sided, 5% level.

## H5. The drop is tied to the publication date (placebo)
- Same pooled placebo as H2: full available LS series per signal, real split at
  January of Year + 1, at least 120 months each side, 10,000 draws, seed 0.

## H6. More early attention goes with more decay
- Unit: the **paper** (signals from the same paper share one attention value, so
  the decay measure is averaged over a paper's signals).
- Decay: relative decay = 1 - POST mean / IS mean.
- Attention (primary): early citations in years Year + 1 to Year + 3, percentile among
  30 sampled same-journal, same-year peers (seed 0), as in H3.
- Matching papers to OpenAlex: automatic. Same journal, OpenAlex publication year
  in [Year - 1, Year], and every listed author surname present. Exactly one candidate
  = matched; zero or several = excluded and counted. Checked against the 4 papers
  already verified by hand.
- Test: Spearman, one-sided permutation p (20,000 shuffles, seed 0).
- Secondary attention measure: OSAP's Google Scholar citation count (Sept 2025),
  as a percentile among OSAP papers published in the same year. This is total, not
  early, attention, so it can partly reflect fame gained after publication.
- Expectation: positive correlation. A null is a real possibility and will be reported.

## What would change the conclusions
- b_post - b_oos not below zero: no evidence of a publication effect beyond overfitting.
- H5 pooled p >= 0.10: the drop may just be returns falling over time.
- H6 correlation <= 0 or p >= 0.10: no evidence that attention speeds up decay.

---

# H7. Decay predictor (written before any model was trained)

Known when this was written: H4-H6 results, and how many signals fall in each
publication-year range. No model had been fit and no test-set error computed.

## Question
Can information available at publication predict how much of its in-sample edge
a signal keeps afterwards, better than assuming every signal keeps the same share?

## Data
- OSAP predictors with IS mean > 0 and POST data.
- Target: retained share = POST mean / IS mean, clipped to [-1, 2] (tiny IS means
  otherwise produce ratios in the hundreds).
- Split by publication year: train <= 2006, test 2007-2016. Signals from one paper
  always fall on the same side. The test set is never used for any choice.

## Features (all known at publication)
- Data type (OSAP Cat.Data; Options, 13F, Event and Other merged into Other).
- Discrete vs continuous signal; "clear" vs "likely" predictability in the paper;
  equal- vs value-weighted portfolio.
- Original paper's t-stat (missing filled with the training median, plus a flag).
- In-sample mean and t-stat of the long-short returns; sample length in years;
  years from sample end to publication; publication year.
- Journal group: top-3 finance (JF, JFE, RFS), accounting (AR, JAE, JAR, RAS, CAR), other.
- Early-citation peer percentile (missing filled with the training median, plus a flag).

## Models
- Baseline: the training-set median retained share, the same for every signal.
- Ridge regression on standardised features, penalty chosen by 5-fold cross-validation
  grouped by paper, on the training set only.
- Random forest: 500 trees, max depth 3, at least 5 signals per leaf, seed 0.

## Pass mark
- Metric: mean absolute error (MAE) on the test set.
- A model "beats the baseline" only if the 95% interval of MAE(model) - MAE(baseline),
  from a bootstrap that resamples test **papers** (2,000 draws, seed 0), lies entirely
  below zero.
- Also reported, descriptive only: Spearman correlation of predicted vs actual on the
  test set, and permutation feature importance.
- Expectation: with about 130 training signals, a clear win is unlikely. A tie with the
  baseline means "decay is not predictable from these features beyond the average".

---

# H8. Decay predictor, second attempt with new inputs (written before any H8 feature was computed)

Known when this was written: the H7 result (no H7 model beats the constant baseline on
2007-2016 papers), the count of predictors per publication-year range, and that OSAP
release 2026.10 has a liquidity-screened portfolio file (`ex_nyse_p20_me`: stocks above
the NYSE 20th size percentile) covering all 212 predictors. No H8 feature value, model
fit or error had been computed.

## Why a second attempt
H7 used descriptive paper details. Published work (McLean and Pontiff 2016) finds decay
is larger where arbitrage is cheap, so H8 adds inputs about trading cost, novelty and
how crowded the idea was. This is a second attempt and is reported as one.

## Data and target
Same as H7: OSAP predictors with IS mean > 0 and POST data; target = POST mean / IS mean,
clipped to [-1, 2]; signals from one paper always on the same side of a split.

## New features (all computed only from data before publication)
- `liquid_share`: IS-period mean of the liquidity-screened long-short return divided by
  the IS-period mean of the original-paper long-short return, same months, clipped to
  [-1, 2]. Missing if fewer than 60 IS months in the screened series.
  Expectation: higher (edge survives in big stocks, cheap to trade) -> lower retained share.
- `prior_max_corr`: largest absolute correlation between the signal's IS-period monthly
  long-short returns and those of every predictor published in an earlier year, over
  overlapping months (at least 60). Missing if no earlier predictor qualifies.
  No directional expectation.
- `category_heat`: number of OSAP predictors with the same `Cat.Economic` published in the
  5 years before (Year - 5 to Year - 1). Expectation: higher -> lower retained share.
- Missing values filled with the training-fold median, plus a missing flag. All H7
  features are kept.

## Evaluation: expanding-window walk-forward
- Three test folds by publication year: 1996-2000, 2001-2006, 2007-2016. Each trains on
  all papers published before the fold starts. Medians, scaling and the ridge penalty are
  set on the training fold only.
- Models: baseline (training-fold median retained share), ridge with H7 + H8 features
  (primary), random forest with H7 + H8 features (500 trees, depth 3, min leaf 5, seed 0),
  and ridge with H7 features only (to show what the new inputs add; descriptive).
- Ridge penalty: grid logspace(-2, 3, 30), 5-fold GroupKFold by paper on the training fold.
- Disclosure: the 2007-2016 fold is the H7 test set, already seen once. The primary test
  pools all three folds so it does not rest on that fold alone.

## Pass mark
- Primary: pooled over all test folds, the 95% interval of MAE(ridge H7+H8) - MAE(baseline),
  bootstrap over test papers (2,000 draws, seed 0), lies entirely below zero.
- Secondary: the same for the random forest.
- Descriptive only: per-fold MAE, Spearman of predicted vs actual, ridge coefficients of
  the new features on the last training fold, permutation importance.
- Expectation: a clear win is still unlikely with about 200 signals; `liquid_share` is the
  input most likely to help.
