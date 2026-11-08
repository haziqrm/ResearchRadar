# Topic radar hypotheses (written before any real run)

The data is B's corpus: a seeded random sample of 380 quant-finance papers per year, 2000-2026, from OpenAlex. Each paper has a SPECTER2 vector (`data/paper_vectors.parquet`).

2026 is a partial year, so it is never used as a target year.

## Definitions
- **Share** of group g in year y: papers from year y in g, divided by all sampled papers from year y.
- **Research growth** at origin year t: `G = share(t+1..t+3) / share(t-2..t) - 1`, where each share is pooled over its 3 years.
  - Counts get +0.5 smoothing, so a group that is absent in one window still gets a finite G.
  - +0.5 means the group's slice of new papers grows by half.
- **Boom:** one of the 5 groups with the highest G at that origin (base rate 5/25 = 20%).
- **Seed score** of a paper at origin t: the number of papers from years t+1..t+3 whose cosine similarity to it is at or above tau.
  - tau is the 99th percentile of cosine similarity between random paper pairs. It is fixed once from all vectors; it is a scale, not a fitted parameter.

## No look-ahead rules
- Every feature at origin t uses papers with year <= t only.
- No citation counts: OpenAlex `cited_by_count` is today's total.
- A model scored at origin t is trained only on origins s <= t-3, so every training target has ended by t.
- **Primary group definition:** the PCA + Gaussian mixture (k = 25, as B) is refit at each origin on papers <= t, and later papers are placed with that model.
- **Secondary ("optimistic"):** B's frozen groups, which were fitted on all years and so may know the future.
- **Known residual leak:** SPECTER2 was pretrained on citation links, so vectors of older papers may carry a little hindsight.

## H11: group growth can be ranked ahead of time
- **Origins:** 2005-2022. The model is scored on origins with at least 3 training origins, i.e. 2008-2022. Baselines are scored on the same origins.
- **Features**, each z-scored across groups within an origin:
  - past growth (t-2..t vs t-5..t-3);
  - 1-year acceleration;
  - current share;
  - share volatility over 5 years;
  - bridging: the share of a group's shared-reference edges that go to other groups;
  - fuzziness: mean top membership probability;
  - spread: mean distance of year-t papers to the group centre.
- **Models:**
  - ridge on demeaned log(1+G) gives `growth_score`;
  - logistic regression on boom gives `boom_probability`;
  - defaults are alpha 10 and C 1, re-chosen on the last 3 training origins once 8 or more training origins exist.
- **Baselines:** momentum (past growth), mean reversion (minus past growth), size (share), random.
- **Metrics per origin:** Spearman rank correlation of score vs G; precision@5 (overlap of the predicted and actual top 5; random = 1); Brier score.
- **Inference:** 95% CIs and one-sided p-values from a moving-block bootstrap over origins (block 3, 5,000 draws).
- **Pass rule (refit groups):** both of the following must hold.
  1. The mean Spearman of ridge minus momentum, and of ridge minus mean reversion, are each above 0 with Holm-adjusted p < 0.05 (2 tests).
  2. The mean precision@5 minus 1 is above 0, with the 95% CI lower bound above 0.

## H12: seed papers can be picked ahead of time
- **Origins:** 2008-2022.
- **Candidates:** papers from years t-2..t.
- **True seeds:** the top 10% of candidates by seed score at that origin.
- **Features**, z-scored within an origin:
  - age;
  - the level-1 ridge `growth_score` of the paper's refit group;
  - local growth: of its neighbours (similarity >= tau, year <= t), those from t-2..t vs those from t-5..t-3, smoothed;
  - local density: neighbours with year <= t;
  - novelty: 1 minus the highest similarity to any older paper;
  - bridging: the share of its shared-reference edges (to papers <= t) that go to other refit groups.
- **Model:** ridge on log(1 + seed score), using the same walk-forward rule and alpha choice as H11.
- **Baselines:**
  - random (precision = 10%);
  - newest first (ties broken by a fixed random order);
  - local density alone.
- **Metric:** precision@20 per origin (the share of the model's top 20 that are true seeds).
- **Pass rule:** ridge precision@20 minus density-alone, and minus random, are each above 0 with Holm-adjusted p < 0.05 (2 tests, same bootstrap as H11).

## Level 2 (descriptive, no hypothesis)
- Sub-topics within each frozen group come from k-means on normalised vectors, with k = clip(n/45, 2, 8) and seed 0. They are labelled with B's c-TF-IDF (`topic_labels`) on titles.
- `rising`: the sub-topic's share of its group went up from 2020-22 to 2023-25, by a one-sided two-proportion z-test, with Holm-adjusted p < 0.05 across all sub-topics, **and** the recent count is at least 10.
- This describes what is driving a group. It is not a forecast.

## Amendment to H11, made before any real run (found by the synthetic tests)
On **pure noise** (every group's true share constant, so the only variation is sampling noise from 380 papers a year), the H11 rule above was passed easily:
- ridge Spearman 0.64 and precision@5 2.8 (random = 1);
- ridge also beat mean reversion (0.47).

The reason: G divides by the current share, which is itself noisy, so a group that looks big by luck "shrinks" next. Any model that uses current share learns this. Beating momentum, mean reversion or random therefore shows nothing about research trends.

**First fix tried and dropped:** a shrinkage baseline, which expects each group to drift back to its 10-year (or full-history) share. On noise, ridge still beat it, simply by learning better shrinkage. Any baseline of this kind can be out-shrunk, so the cause is fixed instead.

**Fix 1: split-half design.** Each year's papers are a random sample, so each year is split at random (seed 0) into halves A and B.
- Every H11 feature uses half A only: shares, growth, volatility, bridging, fuzziness and spread.
- The target's current share, share(t-2..t), uses half B.
- The future share, share(t+1..t+3), uses all papers.

Sampling noise in the features is now independent of the target. On synthetic noise, ridge Spearman fell to between -0.01 and 0.10, against 0.64 without the split.

**Fix 2: inference.** On 30 synthetic noise worlds, the moving-block bootstrap gave false alarms on 17-27% of individual tests at a nominal 5%. With about 15 origins, overlapping windows make neighbouring origins strongly dependent. H11 is therefore judged against **simulated noise worlds**:
- B = 200 worlds. Each paper's group is drawn at random with each group's overall share, constant over time. Years, halves, edges and per-paper fuzziness and spread are kept.
- The same pipeline (features, walk-forward, scoring) runs on each world.
- p = (1 + number of worlds whose statistic is at least the real one) / (B + 1).
- Bootstrap CIs are still reported, but only as descriptions.

**H11 pass rule (replaces the one above).** On refit groups, all three must have Holm p < 0.05 against the noise worlds:
1. mean ridge Spearman;
2. mean ridge precision@5;
3. mean (ridge Spearman - momentum Spearman).

Mean reversion and size are reported for context.

## Amendment to H12, made before any real run
The H12 bootstrap has the same dependence problem, so inference uses only the **non-overlapping origins** 2013, 2016, 2019 and 2022. Their candidate sets (t-2..t) and future windows (t+1..t+3) do not overlap. Two tests, with Holm across them:
1. **vs random:** the null draws 20 random candidates per origin, 5,000 times. p is the share of draws whose mean precision is at least ridge's.
2. **vs density:** a conditional permutation, 2,000 times. Within each origin, ridge scores are shuffled inside density deciles, which keeps what density knows and destroys anything extra. p is the share of shuffles whose mean precision is at least ridge's.

**Pass:** both have Holm p < 0.05. All origins (2013-2022) are still reported as descriptions.

**Correction to test 2, made before any real run.** Shuffling inside density deciles is not a fair null. When the score is density itself, the shuffle also destroys the finer density ordering inside each decile, so plain density "beat" its own shuffle (p = 0.0005 on synthetic data).

The replacement, still before any real run: within each origin, regress the ridge score on density (linear), keep the fitted part, and shuffle only the leftover (residual), 2,000 times. If the leftover says nothing about seeds, the real precision is not unusual. On synthetic data, density-only scores now get p ≈ 1.

## H13: the same forecast, at sub-topic level (written before any H13 run)
H11 failed narrowly at the level of 25 groups. H13 asks whether finer units forecast better, now that there are more units per origin.

- **Units:** **k = 75** sub-topics, from the same PCA + Gaussian mixture (`fit_topics`, k fixed at 75, 50 PCA components, seed 0). That is about 5 papers per sub-topic per year. 75 is chosen before running, as a middle ground between detail and noise, and is not tuned.
- **Everything else is identical to H11 as amended:**
  - the split-half design, the features and the walk-forward;
  - refit at each origin on papers <= t (primary);
  - one fit on all papers (secondary, "optimistic");
  - 200 noise worlds with k = 75.
- **Boom:** the top **15** of 75 sub-topics by G (base rate 20%, as in H11). Precision@15 under random picking is 15 × 15/75 = 3 hits.
- **Pass rule (refit sub-topics):** all three must have Holm p < 0.05 against the noise worlds:
  1. mean ridge Spearman;
  2. mean ridge hits in the top 15;
  3. mean (ridge Spearman - momentum Spearman).
- **Outputs:**
  - `data/subtopic_forecasts.csv`: the live forecast from 2025 for each sub-topic, with its title label (B's c-TF-IDF), parent group (the B group most of its papers belong to) and paper counts for 2020-22 and 2023-25;
  - `data/subtopic_forecast_eval.csv`.

## Deviation recorded before running
The plan named "the sub-topic's past growth" as a seed feature, from a per-origin sub-topic refit. It is replaced by **local growth** (above), the same idea measured on each paper's own neighbours. This avoids one k-means per group per origin and keeps every feature to papers <= t.
