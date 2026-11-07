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
