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
