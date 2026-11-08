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

---

# Correction after an evaluation audit (2026-11-07, after H7 and H8 results were known)

An audit checked H7 and H8 for evaluation bugs: same target and test rows for every
model, no test data in the baseline, time split by paper, a synthetic series with a
known 56% retained share recovered exactly. It found no bug behind the null results,
but led to these changes. They were made after seeing results, so they are corrections,
not new tests, and they do not change any verdict.
- `citation_pct` dropped as a feature: it counts citations in years Y+1 to Y+3, which
  are not known at publication. (Model results were essentially unchanged.)
- Squared error (MSE) reported alongside MAE. Ridge minimises squared error and so
  shrinks toward the training *mean*, while MAE rewards the *median*: most of ridge's
  small H7 loss on MAE comes from that mismatch.
- Added, descriptive only: a constant at the training mean, and a median (quantile 0.5)
  regression with an L1 penalty chosen the same way as ridge, so the estimator matches
  the metric. Each row records the chosen penalty and whether it sits at the grid edge.
- Guards: a zero-variance in-sample series gives a missing t-stat instead of infinity,
  and an all-missing feature column no longer crashes the fit.

---

# H9. Short-horizon forecasts for published signals (written before any H9 number was computed)

Known when this was written: H1-H8 results and the published literature on factor
momentum (Ehsani and Linnainmaa 2022) and volatility-managed portfolios (Moreira and
Muir 2017). No H9 forecast, portfolio or loss had been computed on this data.

## Question
H7 and H8 found that *how much* edge a signal keeps after publication cannot be predicted
at publication. H9 asks two month-ahead questions a trader of published signals can act on:
(a) does a published signal's recent return predict its next month's return, and
(b) does its recent volatility predict next month's volatility better than its long-run level?

## Data
- OSAP original-paper long-short returns (decimals), 212 predictors.
- Only **post-publication** months are scored (date year > publication year), so every
  signal is public when it is traded. Trailing windows may reach back before
  publication: those returns were already observable at the forecast date.
- Every forecast for month t uses returns up to month t-1 only.

## H9a. Factor momentum
- Signal: trailing 12-month mean return (months t-12 to t-1; all 12 required).
- Each month with at least 10 eligible signals and at least one in each group:
  spread_t = mean return of signals with trailing mean > 0 minus mean of the rest.
- Test: mean spread > 0, Newey-West t (12 lags), one-sided p < 0.05.
- Descriptive: the share of months with a positive spread; mean returns and annualised
  Sharpe ratios of equal-weighting all published signals vs only the past winners.

## H9b. Volatility forecast
- Forecast of month-t volatility for each signal: model = square root of the mean
  squared return over t-12 to t-1; baseline = the same over all months before t
  (at least 36 required).
- Loss: |r_t| against sqrt(2/pi) x forecast (the expected |r| under a normal), absolute error.
- Test: monthly mean of (model loss - baseline loss) across signals; its time-series mean
  < 0, Newey-West t (12 lags), one-sided p < 0.05.
- Descriptive: a volatility-managed equal-weight portfolio of published signals, each
  weighted by 1 / forecast variance (scaled to the full-sample volatility of the plain
  version, which changes no Sharpe ratio). Sharpe ratios before trading costs.

## What would change the conclusions
- H9a p >= 0.05: no evidence that recent winners among published signals keep winning.
- H9b p >= 0.05: recent volatility is no better a guide than long-run volatility.

---

# H10. Trained volatility model (written before any H10 model was fitted)

Known when this was written: the H9b result (the 12-month rule beats the long-run rule,
t = 4.7). A test download of FRED's daily VIX (`VIXCLS`) succeeded; it starts 1990-01-02.
No H10 feature, model fit, loss or portfolio had been computed. H1-H9 files, rules and
results are not changed.

## Question
Does a trained model forecast a published signal's next-month volatility better than
the H9b 12-month rule?

## Data
- The H9 panel: OSAP original-paper long-short returns, post-publication months only.
- Sample: months from January 1990 onward (so the VIX is available), with every feature
  and both H9b forecasts available. All models and baselines are scored on the same rows.
- Every feature for month t uses data up to the end of month t-1.

## Target and loss
- Target: |r_t|, the absolute monthly return.
- Loss, as in H9b: | |r_t| - sqrt(2/pi) x forecast |. Each model outputs a volatility forecast.

## Features (all logs of volatilities, HAR-style, except age)
- Root mean squared return over the last 1, 3, 6 and 12 months, and over all prior months.
- Volatility of volatility: standard deviation of |r| over t-12 to t-1.
- Signal age: months since January of the year after publication.
- Cross-signal average volatility: mean of |r_{t-1}| across all OSAP predictors in t-1.
- Market volatility: root mean squared Ken French Mkt-RF over the last 1 and 12 months.
- VIX: month-end close of month t-1, divided by 100 and by sqrt(12) (monthly decimal scale).
- Logs use log(x + 1e-4).

## Models
- Baseline A: the H9b 12-month rule. Baseline B: the H9b long-run rule.
- Model 1 (primary): ridge on standardised features, target log(|r_t| + 1e-4); forecast =
  exp(prediction) x c. c is chosen on the validation block to minimise the H9 loss
  (never on the rows the ridge was fitted on).
- Model 2 (secondary): HistGradientBoostingRegressor, loss = absolute error, target
  |r_t| / sqrt(2/pi), so it predicts the volatility forecast directly.
- No neural networks.

## Walk-forward
- One fold per test year Y, 1996-2025. Training months: January 1990 to December Y-2. Year
  Y-1 is a 12-month gap and is never used to train the fold that tests Y.
- Folds with at least 8 training years (test years 1999 onward) tune settings on time-based
  validation inside the training years: validation = the last 3 training years, a 12-month
  gap before it, and the inner model trained on the years before that gap.
  - Ridge penalty grid: logspace(-3, 3, 13); c from the validation block.
  - Boosting grid: learning_rate {0.05, 0.1} x max_leaf_nodes {15, 31} x
    min_samples_leaf {50, 200}, max_iter 300.
- Folds with fewer than 8 training years (test years 1996-1998) use fixed defaults:
  ridge alpha = 1, c = exp((Euler's gamma + ln 2) / 2) = 1.887 (the exact correction for
  log|r| under a normal); boosting learning_rate 0.1, max_leaf_nodes 31,
  min_samples_leaf 200, max_iter 300.
- After choosing settings, each model is fitted once on all training months of the fold.

## Test
- Primary period: test years 2000-2025. Secondary: 1996-1999, reported separately.
- Statistic: monthly mean across signals of (model loss - baseline A loss); its time-series
  mean < 0, Newey-West t with 12 lags, one-sided p.
- Ridge is primary and boosting secondary; the two p-values are Holm-adjusted, and a model
  passes only if its adjusted p < 0.05.
- Also reported: the same against baseline B, mean loss (MAE), Spearman correlation of
  forecast with |r|, and a 95% bootstrap interval for the mean loss difference that
  resamples signals (2,000 draws, seed 0).

## Decision check (descriptive)
- A volatility-managed equal-weight portfolio of published signals for each forecast
  (baseline A, ridge, boosting), on test months: raw weight 1 / forecast^2, capped at 5 times
  the average raw weight, then scaled so average gross exposure is 1. Same cap and scaling
  for all three; plain equal weight is shown for reference.
- Costs, illustrative only: 10 and 25 bps per unit of one-way weight change per month.
  Only the extra trading from rescaling is charged; the long-short portfolios' own
  trading costs are not included.
- Reported: annualised Sharpe before and after costs, average monthly turnover, maximum
  drawdown, and the break-even cost at which each model's net Sharpe equals the H9b rule's.

## What would change the conclusions
- Neither Holm-adjusted p < 0.05: a trained model adds nothing over the 12-month rule; the
  dashboard keeps the H9b rule.
- A model wins on loss but its net Sharpe is below the H9b rule's at 10 bps: statistically
  better, not worth trading.

## H10 addition before any real-data run: scale-corrected baselines
A synthetic test with constant volatility showed that a trained model beats the long-run
rule under the H9 loss without any timing skill: the loss rewards forecasts scaled to the
*median* of |r| (0.674 sigma), while sqrt(2/pi) x sigma is its *mean*, and the ridge's
validation scale c learns that shift. To separate scale from timing, two descriptive
baselines are added: baseline A and baseline B each multiplied by one constant, chosen to
minimise the H9 loss on that fold's training rows (no model is fitted on them). The
pre-registered primary and secondary tests (against unscaled baseline A) are unchanged;
comparisons against the scaled baselines are reported alongside and labelled as added.
