# Viva Preparation Guide

Short, defensible answers you can give in a faculty evaluation. The numbers come from the
committed pipeline run (`artifacts/`). If you re-run the pipeline, check the Model Evaluation page.

## 1. The problem in one sentence
Using only what is known at the end of a reporting window, can we estimate a hospital's
risk-adjusted CABG 30-day readmission rate for the next window better than assuming "it stays the
same"? And how do readmission patterns differ across conditions, counties and population groups?

## 2. What is new (the contribution)
We do not claim a new algorithm. The contribution is a **healthcare-specific evaluation workflow**:
1. **Temporal alignment.** Inputs come from window *t* and the target from window *t+1*. Pairs are
   formed only when the target starts exactly one year later, so a missing window is never bridged.
2. **Baseline-first evaluation.** Every model is compared with **persistence** on the same hospitals,
   with paired bootstrap confidence intervals.
3. **Found a design artefact.** HCAI's two-year windows overlap (2018-2019 and 2019-2020 share 2019).
   This inflates persistence; we quantified it with a non-overlapping design.
4. **Three levels side by side, never merged:** hospital CABG, hospital stroke and county/population
   all-cause readmissions.
5. **Deployable and reproducible:** one-command pipeline, tests, CI workflow, and a dashboard that
   verifies it runs the *same* evaluated models.

## 3. Datasets (and what was wrong with them)
| Dataset | Key facts | Issues handled |
|---|---|---|
| A CABG (1,937 rows) | 127 hospitals, 4 measures, windows 2018-2019 ... 2022-2023 | 168 name spellings per 127 IDs; `'.'` values; 2020-2021 has only 2 hospitals; overlapping windows |
| B Stroke (2,188 rows) | 288 hospitals, readmission and mortality, 2011-2012 ... 2014-2015 | 2013-2014 count columns transposed in all 545 rows (fixed and logged); statewide rows |
| C All-cause (1,057 rows) | 58 counties, 2011-2024, Age/Sex/Race/Payer strata (state only) | ICD-9 vs ICD-10 break; 2015 partial year; rates as text; a suppressed cell |
| D Synthetic (120 rows) | flagged synthetic, `SYNTH_` IDs | used **only** for schema, smoke and scenario tests |

## 4. Algorithms
* **Persistence baseline:** next = previous. A standard naive forecast.
* **Training-mean baseline:** predicts the average for every hospital.
* **Linear Regression:** interpretable. Its slope below 1 shows regression to the mean.
* **Random Forest:** non-linear. Tuned by hospital-grouped CV on training data only
  (selected max_depth 5, min_samples_leaf 10).

## 5. Evaluation design (say this clearly)
* **Primary:** forward temporal hold-out. Training uses 124 transitions with targets 2019-2020 and
  2020-2021; the test is the 2022-2023 window (118 hospitals), used once.
* **Model selection:** the recommended model was picked by grouped 5-fold CV *inside the training
  set*, before the test set was looked at.
* **Leakage controls:** features come from the previous window only. There are no IDs or names as
  features. Imputer, log transform and scaler are fitted on training rows inside a Pipeline. Train
  and test target windows are disjoint. Unit tests check all of this.
* **Sensitivity analyses:** pooled grouped CV (n = 242); a non-overlapping horizon *t -> t+2*
  (n = 120); an extended feature set.

## 6. Key results
* **CABG test (MAE, pp):** persistence 2.487, Linear Regression 2.543, Random Forest 2.741,
  mean 3.919. LR − persistence = **+0.055, 95% CI [-0.229, +0.335]**, so the evidence is
  **insufficient to name a winner**.
* **Why:** overlapping windows (r = 0.72 consecutive vs 0.47 non-overlapping). On the
  non-overlapping horizon, persistence gets **3.96** while LR gets **3.52** (better in 88% of folds).
* **Explainability:** the previous rate dominates (shuffling it adds +1.76 pp MAE). Volume adds a
  little: larger programmes are predicted slightly lower, an association only. Post-op stroke adds
  nothing. Errors concentrate in small hospitals: MAE 3.35 for the lowest-volume third vs 1.57 for
  the highest.
* **Stroke (more data, 259 test hospitals):** LR 1.853 vs persistence 1.981 (−6.4%). The CI
  [−0.268, +0.007] just touches 0, but LR wins in 100% of grouped-CV folds. Hospital readmission and
  mortality are nearly uncorrelated (r between −0.12 and +0.05), consistent with Krumholz et al.,
  JAMA 2013.
* **Population:** 15.95% statewide in 2024, up from 14.48% in 2016 (ICD-10). Medicare 17.02% is the
  highest payer group and Self-Pay 6.98% the lowest. Among counties with at least 1,000 admissions:
  Siskiyou 10.19% to Butte 18.78%.

## 7. Limitations (volunteer these)
Only two usable one-step CABG transitions; overlapping windows; small hospitals give noisy rates;
the risk-adjusted outcome is itself an estimate; the ICD coding change; aggregate data only (no
patient-level prediction and no causes); not clinically validated; provenance and licence should
be confirmed.

## 8. Likely questions and answers
1. **Why not use the stroke data to test the CABG model?** It covers a different condition,
   different years (2011-2015 vs 2018-2023) and a different cohort. It would not be a valid
   external test.
2. **Your Linear Regression did not beat the baseline. Is the project a failure?** No. Reporting it
   honestly is the scientific result. We also explained *why* (overlapping windows) and showed that
   learned models do help when the future window is genuinely new (non-overlapping CABG; stroke).
3. **Why is R² 0.54 for persistence?** Half of the next window's patients are already in the
   previous window, so the two rates are mechanically correlated.
4. **Why not include 2019-2020 -> 2021-2022 as training data?** That pair skips the 2020-2021
   window. Treating it as "next period" would bridge a gap and change the horizon, so it is used
   only in the explicitly labelled non-overlapping design.
5. **How do you prevent leakage?** Previous-window features only, no identifiers, preprocessing
   inside Pipelines fitted on training rows, disjoint target windows, and selection/tuning without
   the test set. There are tests for each.
6. **Why grouped CV?** The same hospital appears in several rows. Random K-fold would let the model
   see a hospital in training and be tested on the same hospital, which gives optimistic error.
7. **What does an estimate of 8.8% mean?** About 8.8% of that hospital's eligible isolated CABG
   patients are expected to be readmitted within 30 days, after risk adjustment. It is not any
   patient's probability.
8. **Why is Random Forest worse?** With about 120 rows and one dominant, roughly linear input, extra
   flexibility fits noise. The grid search itself chose shallow, heavily regularised trees.
9. **What is permutation importance?** You shuffle one input on the test set and measure how much
   the error rises. A large rise means the model relies on that input.
10. **What does the bootstrap CI tell you?** We resample test hospitals 2,000 times to see how much
    the MAE difference could vary. If the interval contains 0, the difference is not reliable.
11. **Why not compare ICD-9 and ICD-10 rates directly?** The coding change (1 Oct 2015) altered how
    diagnoses are recorded. We plot the eras separately, and 2015 is partial.
12. **Why do some counties not appear on the ranking?** Fewer than 1,000 admissions makes rates
    unstable (Alpine has about 10 a year). They are greyed out rather than ranked.
13. **How does the dashboard know it uses the evaluated model?** It loads `models/cabg_models.joblib`
    and re-predicts the stored test set. The integrity check confirms an exact match.
14. **How do GitHub Actions and Streamlit Cloud work together?** The workflow runs tests, the
    pipeline and the report, and can commit refreshed artifacts. Streamlit Cloud redeploys from the
    repository when main changes. They are separate services.
15. **What is the synthetic data for?** Checking the schema mapping, smoke-testing inference and
    demonstrating hypothetical scenarios. It is never used for training or reported as accuracy.
16. **What would you do next?** Add new CABG windows as they are published. Use empirical-Bayes or
    hierarchical shrinkage with case counts. Calibrate prediction intervals.

## 9. Two-minute demo script
1. **Overview:** explain the three data levels, the KPIs (each labelled with its own period), the
   coverage timeline (point at the faded 2020-2021 bar) and the findings.
2. **CABG Prediction:** pick a hospital, show its history with CIs, the 2023-2024 estimate vs
   persistence vs benchmark and the contribution chart. Switch the window to 2021-2022 to show an
   out-of-sample back-test with the published actual (star). Then open the scenario tab.
3. **Model Evaluation:** show the comparison table, the three-design chart and the bootstrap CI
   crossing 0, then the learned curve (regression to the mean).
4. **Population:** show the map, the payer bars and the ICD-era note.
