# Data Dictionary and Dataset Audit

_Generated automatically by `run_pipeline.py` from the files in `data/raw/`. Do not edit by hand._

The four datasets are analysed in **separate workflows**; they are not merged into one master table because they differ in clinical condition, geography, population and reporting period.

## Dataset A - California CABG hospital performance

- **File:** `cabg_hospital_performance.csv` (354,767 bytes, encoding `cp1252`)
- **Role:** Primary supervised ML experiment (next-period isolated CABG readmission rate)
- **Granularity:** Hospital x reporting period x performance measure
- **Rows x columns:** 1,937 x 20; exact duplicate rows: 0
- **SHA-256:** `e42406d31c93317a7a5baff436052ecd97b911188188ed20cca62b0cde7b147c`
- **Reporting periods:** 2018-2019, 2019, 2019-2020, 2020, 2020-2021, 2021-2022, 2022, 2022-2023, 2023
- **Unique hospital IDs:** 127 (168 name variants)
- **Blank / '.' values by column:** Total Hospital CABG Cases for Performance Measure Reporting: 11, Hospital Adverse Events: 11, Hospital Observed Adverse Events Rate (%): 11, Hospital Expected Adverse Events Rate (%): 11, Hospital Risk-Adjusted Adverse Events Rate (%,RAAER): 11, Hospital Lower 95% CI for RAAER: 11, Hospital Upper 95% CI for RAAER: 11

| Column | Meaning |
|---|---|
| `Year` | Reporting period: two-year rolling window ('2018-2019') or single year ('2019', operative mortality only) |
| `County` | County of the hospital |
| `Hospital` | Hospital name as published in that period (names change over time) |
| `Hospital ID` | 9-digit OSHPD/HCAI facility identifier - stable panel key |
| `Performance Measure` | One of 4 CABG measures (30-day readmission, post-op stroke, isolated CABG mortality, CABG+valve mortality) |
| `Total Statewide CABG Cases` | All CABG cases statewide in the period |
| `Total Statewide CABG Cases for Performance Measure Reporting` | Statewide cases eligible for this measure |
| `Statewide Adverse Events` | Statewide count of the adverse event |
| `Statewide Adverse Events Rate (%)` | Statewide observed rate for the measure (%) |
| `Total Hospital CABG Cases` | Hospital CABG volume in the period (used as case-volume feature) |
| `Total Hospital CABG Cases for Performance Measure Reporting` | Hospital cases eligible for the measure (denominator) |
| `Hospital Adverse Events` | Hospital adverse-event count ('.' = not reported) |
| `Hospital Observed Adverse Events Rate (%)` | Events / eligible cases (%) |
| `Hospital Expected Adverse Events Rate (%)` | Rate expected from the hospital's patient case mix (%) |
| `Hospital Risk-Adjusted Adverse Events Rate (%,RAAER)` | Risk-adjusted rate (%) - the modelling target/feature |
| `Hospital Lower 95% CI for RAAER` | Lower 95% confidence limit of RAAER |
| `Hospital Upper 95% CI for RAAER` | Upper 95% confidence limit of RAAER |
| `Performance Rating` | Better / Average / Worse than state average, or Not Applicable |
| `Longitude` | Hospital longitude |
| `Latitude` | Hospital latitude |

## Dataset B - California ischemic stroke 30-day outcomes

- **File:** `ischemic_stroke_outcomes.csv` (288,116 bytes, encoding `cp1252`)
- **Role:** Separate stroke-specific analytics and temporal model
- **Granularity:** Hospital x reporting period x measure (readmission / mortality)
- **Rows x columns:** 2,188 x 10; exact duplicate rows: 0
- **SHA-256:** `2327c6adb2f6608d334cc23cc04f5fdf6f8738d7cb2079ac6c06d303a858e2a2`
- **Reporting periods:** 2011-2012, 2012-2013, 2013-2014, 2014-2015
- **Unique hospital IDs:** 288
- **Blank / '.' values by column:** County: 2, OSHPDID: 8, Risk Adjusted Rate: 10, # of Deaths/Readmissions: 10, # of Cases: 10, Hospital Ratings: 18, Location 1: 8

| Column | Meaning |
|---|---|
| `Year` | Two-year rolling reporting window (2011-2012 ... 2014-2015) |
| `County` | County (blank / 'AAAA' on statewide rows) |
| `Hospital` | Hospital name or 'Statewide' |
| `OSHPDID` | 9-digit OSHPD facility identifier ('.' on statewide rows) |
| `Measure` | 30-day Readmission or 30-day Mortality |
| `Risk Adjusted Rate` | Risk-adjusted rate (%) |
| `# of Deaths/Readmissions` | Event count (columns transposed in source for 2013-2014; corrected) |
| `# of Cases` | Eligible ischemic stroke cases (denominator) |
| `Hospital Ratings` | Better / As Expected / Worse |
| `Location 1` | '(lat, lon)' text |

## Dataset C - California all-cause unplanned 30-day readmissions

- **File:** `california_allcause_readmissions.csv` (57,752 bytes, encoding `utf-8`)
- **Role:** Population-level trends, county comparison and strata breakdowns
- **Granularity:** Year x stratum x county (strata other than 'Overall' are statewide only)
- **Rows x columns:** 1,057 x 10; exact duplicate rows: 0
- **SHA-256:** `17e8610b80ed451ecf3484e19a83685249ce48bb1fffa425bed660a4ae745729`
- **Years:** 2011-2024; coding systems: ICD-10 2016-2024; ICD-9 2011-2015
- **Counties:** 58 (plus 'State' and, in 2019 only, 'Unknown County')
- **Blank / '.' values by column:** Total Admits (ICD-9): 682, 30-day Readmits (ICD-9): 682, 30-day Readmission Rate (ICD-9): 682, Total Admits (ICD-10): 375, 30-day Readmits (ICD-10): 376, 30-day Readmission Rate (ICD-10): 376

| Column | Meaning |
|---|---|
| `Year` | Calendar year 2011-2024 |
| `Strata` | Overall, Age, Sex, Race-Ethnicity or Expected Payer |
| `Strata Name` | Stratum value (e.g. '65 years and above', 'Medi-Cal') |
| `County` | County name, 'State' or 'Unknown County' (non-Overall strata are State only) |
| `Total Admits (ICD-9)` | Index admissions coded in ICD-9 (2011-2015) |
| `30-day Readmits (ICD-9)` | Unplanned 30-day readmissions (ICD-9 era) |
| `30-day Readmission Rate (ICD-9)` | Readmission rate text, e.g. '13.80%' (ICD-9 era) |
| `Total Admits (ICD-10)` | Index admissions coded in ICD-10 (2016-2024), text with thousands separators |
| `30-day Readmits (ICD-10)` | Unplanned 30-day readmissions (ICD-10 era) |
| `30-day Readmission Rate (ICD-10)` | Readmission rate text (ICD-10 era) |

## Dataset D - Synthetic CABG scenario data

- **File:** `synthetic_cabg_test.csv` (13,762 bytes, encoding `utf-8`)
- **Role:** Schema validation, smoke tests and hypothetical scenarios ONLY (never training/evaluation)
- **Granularity:** One synthetic hospital per row
- **Rows x columns:** 120 x 10; exact duplicate rows: 0
- **SHA-256:** `8585469a7fe21353059686bcb9fdb78c8c24297aa5c46eb19b6a1eca9931db1e`
- **Synthetic flag:** 120/120 rows flagged True. **Never used for training or real-data evaluation.**

| Column | Meaning |
|---|---|
| `Hospital_ID` | Synthetic ID (SYNTH_HOSP_###) |
| `Hospital_Name` | Synthetic name |
| `Input_Period` | Always 'Synthetic prior period' |
| `Target_Period` | Always 'Synthetic next period' |
| `Previous_CABG_Readmission_Rate_Pct` | -> prev_readmit_rate |
| `CABG_Case_Volume` | -> prev_cabg_volume |
| `Operative_Mortality_Rate_Pct` | Not used by the primary model (no aligned real-data feature) |
| `Postoperative_Stroke_Rate_Pct` | -> prev_stroke_rate |
| `Next_Period_CABG_Readmission_Rate_Pct` | Synthetic label (agreement check only, never performance evidence) |
| `Synthetic_Record_Flag` | Must be True for every row |

## Processed tables (`data/processed/`)

| File | Contents |
|---|---|
| `cabg_clean.csv` | Tidy CABG table: one row per hospital x period x measure; typed numerics; measure_code; period_start/end/years; hospital_name = latest name for the ID. |
| `cabg_panel.csv` | One row per hospital x two-year window with readmission, stroke, CABG+valve mortality, final-year isolated mortality and volume side by side. |
| `stroke_clean.csv` | Tidy stroke table; is_statewide flag; counts_corrected flag for the 2013-2014 swap fix. |
| `stroke_panel.csv` | One row per hospital x window with readmission and mortality side by side. |
| `allcause_clean.csv` | Tidy all-cause table with one rate column plus coding_system (ICD-9 / ICD-10), rate_reported, rate_computed, geography_level, coding_transition_year. |
| `synthetic_clean.csv` | Synthetic rows with model-schema columns added; is_synthetic=True. |

## Preprocessing decisions (log)

| Dataset | Step | Rows | Detail |
|---|---|---|---|
| cabg | duplicates | 0 | Removed 0 exact duplicate rows |
| cabg | missing_markers | 11 | Converted '.' (not reported) to NaN in rate/count fields; 11 rows had no RAAER (all 'CABG+Valve Operative Mortality', rating 'Not Applicable'). Missing values were NOT set to 0. |
| cabg | numeric_parsing |  | Removed thousands separators (e.g. '31,086') and cast counts/rates to float. |
| cabg | text_normalisation |  | Collapsed embedded line breaks/double spaces in County, Hospital and Rating (e.g. 'San\nBernardino', 'Contra  Costa'); mapped Windows-1252 en dashes to '-'; fixed rating typo 'Not Appplicable' -> 'Not Applicable'. |
| cabg | hospital_identity |  | 168 reported name variants map to 127 stable Hospital IDs; the most recent name per ID is used for display. Hospital ID (not name) is the panel key. |
| cabg | range_check | 0 | RAAER outside [0, 100] set to NaN: 0 rows |
| cabg | events_le_cases | 0 | Rows with adverse events > eligible cases: 0 |
| stroke | duplicates | 0 | Removed 0 exact duplicate rows |
| stroke | statewide_rows | 8 | Identified 8 'Statewide' summary rows; their county placeholder 'AAAA' / blank and ID '.' were set to NaN. |
| stroke | missing_rates | 10 | 10 hospital-measure rows have no published rate/counts; kept as NaN (not 0). |
| stroke | column_swap_fix | 545 | Period 2013-2014: 100% of rows had '# of Deaths/Readmissions' > '# of Cases' (impossible). The two count columns are systematically transposed in this period; they were swapped back. Risk-adjusted rates were unaffected. Verified: after the swap the observed rate correlates with the risk-adjusted rate at the same level as in other periods. |
| stroke | hospital_identity |  | 396 reported names map to 288 OSHPD IDs; most recent name used for display. |
| stroke | location_parsing |  | Parsed 'Location 1' '(lat, lon)' text into numeric latitude/longitude. |
| allcause | column_names |  | Stripped trailing spaces from column names (e.g. '30-day Readmits (ICD-9) '). |
| allcause | duplicates | 0 | Removed 0 exact duplicate rows |
| allcause | icd_eras |  | ICD-9 fields populated for 375 rows (years 2011-2015); ICD-10 fields for 682 rows (2016-2024). Combined into one tidy table with an explicit 'coding_system' column; the two eras are never pooled without that label. |
| allcause | rate_parsing |  | Parsed rates such as '14%', '13.80%' into percentage points; removed thousands separators from ICD-10 counts (e.g. '1,877,198'). |
| allcause | suppressed_values | 1 | 1 row(s) report admissions but no readmissions/rate (small-cell suppression, e.g. Alpine 2017); left as NaN. |
| allcause | transition_year |  | 2015 is the final ICD-9 year. Its statewide admissions are ~25% below the prior year, consistent with ICD-9 coding ending on 1 Oct 2015 (no ICD-10 rows exist for that year in this file). It is flagged 'coding_transition_year' and treated as a partial year. |
| synthetic | schema_mapping |  | Mapped synthetic columns onto the CABG model feature schema: Previous_CABG_Readmission_Rate_Pct -> prev_readmit_rate, Postoperative_Stroke_Rate_Pct -> prev_stroke_rate, CABG_Case_Volume -> prev_cabg_volume. 'Operative_Mortality_Rate_Pct' has no aligned real-data counterpart in the primary model and is reported but not used. |
| synthetic | isolation |  | All synthetic rows are tagged is_synthetic=True and are excluded from training, testing and every real-data statistic. |

## Validation checks

| Status | Check | Detail |
|---|---|---|
| PASS | All required columns present | 20 required columns found |
| PASS | Dataset is non-empty | 1,937 rows; encoding cp1252 |
| PASS | No exact duplicate rows | 0 duplicate rows |
| PASS | All required columns present | 10 required columns found |
| PASS | Dataset is non-empty | 2,188 rows; encoding cp1252 |
| PASS | No exact duplicate rows | 0 duplicate rows |
| PASS | All required columns present | 10 required columns found |
| PASS | Dataset is non-empty | 1,057 rows; encoding utf-8 |
| PASS | No exact duplicate rows | 0 duplicate rows |
| PASS | All required columns present | 10 required columns found |
| PASS | Dataset is non-empty | 120 rows; encoding utf-8 |
| PASS | No exact duplicate rows | 0 duplicate rows |
| PASS | One row per hospital x period x measure | 0 duplicate keys |
| PASS | Every row has a valid 9-digit Hospital ID | 0 missing IDs; 127 unique IDs |
| PASS | All measures recognised | CABG+Valve Operative Mortality, Isolated CABG 30-Day Readmission, Isolated CABG Operative Mortality, Isolated CABG Post-Operative Stroke |
| PASS | Risk-adjusted rates within [0, 100] % | min 0.00, max 97.55 |
| PASS | Lower CI <= RAAER <= Upper CI | 0 rows violate CI ordering |
| PASS | Adverse events <= eligible cases | 0 violations |
| WARN | Readmission periods have comparable hospital coverage | 2018-2019: 123 hospitals; 2019-2020: 123 hospitals; 2020-2021: 2 hospitals; 2021-2022: 120 hospitals; 2022-2023: 119 hospitals -> sparse: 2020-2021 |
| PASS | Two-year periods are overlapping rolling windows (statewide cases = sum of single years) | 2019-2020: 29,877 vs 2019+2020 = 29,877; 2022-2023: 31,086 vs 2022+2023 = 31,094. Consecutive two-year periods share one calendar year of patients. |
| PASS | One row per hospital x period x measure | 0 duplicate keys |
| PASS | Hospital rows have valid OSHPD IDs | 0 missing; 288 unique |
| PASS | Risk-adjusted rates within [0, 100] % | min 0.00, max 29.72 |
| PASS | Deaths/readmissions <= cases (after correction) | 0 violations remain; 545 rows corrected by column swap |
| WARN | Source count columns consistent across periods | 545 rows in 2013-2014 had transposed count columns in the source file (corrected in memory). |
| PASS | Missing hospital rates are kept as NaN | 10 hospital-measure rows without a published rate (not imputed) |
| PASS | One row per year x stratum x county | 0 duplicate keys |
| PASS | Each row uses exactly one coding system |                 min   max; coding_system            ; ICD-10         2016  2024; ICD-9          2011  2015 |
| PASS | Published rate agrees with readmits / admits (rounding tolerance 0.5 pp) | max abs difference 0.050 pp; mean 0.009 pp |
| PASS | Rates within [0, 100] % | min 0.00, max 40.00 |
| WARN | County rates based on >= 1,000 admissions | 117 county-years below threshold (e.g. Alpine, Colusa, Del Norte, Inyo, Lassen, Mariposa); flagged in dashboard rankings |
| PASS | Every synthetic row has Synthetic_Record_Flag = True | 120/120 flagged |
| PASS | Synthetic IDs use the SYNTH_ prefix | all IDs start with SYNTH_ |
| PASS | No synthetic ID collides with a real hospital ID | 0 collisions |
| PASS | Synthetic model inputs are numeric and finite | prev_readmit_rate, prev_stroke_rate, prev_cabg_volume, synthetic_label_next_rate |

## Joins performed

- **CABG measures -> CABG panel:** joined on `hospital_id` + identical two-year `period` (readmission, post-operative stroke, CABG+valve mortality). Isolated CABG operative mortality (single-year) is joined on `hospital_id` + the input window's final calendar year and is used only in the labelled 'extended feature' sensitivity analysis.
- **Stroke measures -> stroke panel:** joined on `hospital_id` + identical `period`.
- **All-cause counties -> county boundaries:** county name to Census county `NAME` (58/58 matched exactly).
- **Not joined:** CABG with stroke (no overlapping years), hospital datasets with all-cause population data (different units of analysis), synthetic with any real data.
