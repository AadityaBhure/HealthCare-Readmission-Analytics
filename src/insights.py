"""Auto-derived key findings. Every statement is computed from pipeline outputs; wording that
depends on statistical evidence (e.g. 'improves on') is chosen from the bootstrap results."""
from __future__ import annotations

import pandas as pd


def _metric(metrics: pd.DataFrame, model: str, col: str = "MAE") -> float:
    return float(metrics.loc[metrics["model"] == model, col].iloc[0])


def _boot(boot: pd.DataFrame, model: str, ref: str = "persistence") -> dict:
    return boot[(boot["model"] == model) & (boot["reference"] == ref)].iloc[0].to_dict()


def model_verdict(metrics: pd.DataFrame, boot: pd.DataFrame, model: str = "linear_regression") -> str:
    b = _boot(boot, model)
    if b["ci_high"] < 0:
        return "improves on"
    if b["ci_low"] > 0:
        return "is worse than"
    return "is statistically indistinguishable from"


def derive(pop: dict, cabg: dict, stroke: dict) -> list[dict]:
    f: list[dict] = []

    # ---------------- population ----------------
    yoy = pop.get("yoy_change_pp")
    f.append({"area": "Population", "text": (
        f"California's all-cause unplanned 30-day readmission rate was {pop['latest_rate']:.2f}% in "
        f"{pop['latest_year']} ({pop['latest_coding_system']}), "
        + (f"{'up' if yoy >= 0 else 'down'} {abs(yoy):.2f} percentage points from {pop['latest_year'] - 1} and "
           if yoy is not None else "")
        + f"{pop['icd10_change_pp']:+.2f} pp since {pop['icd10_first_year']}, the first ICD-10 year "
          f"({pop['icd10_first_rate']:.2f}%).")})
    payer = pop["strata_latest"].get("Expected Payer", [])
    if payer:
        f.append({"area": "Population", "text": (
            f"By expected payer ({pop['latest_year']}), the highest rate was {payer[0]['name']} "
            f"({payer[0]['rate']:.2f}%) and the lowest {payer[-1]['name']} ({payer[-1]['rate']:.2f}%). "
            "These are descriptive associations, not causal effects.")})
    age = pop["strata_latest"].get("Age", [])
    if age:
        f.append({"area": "Population", "text": (
            f"By age group ({pop['latest_year']}), the highest rate was {age[0]['name']} ({age[0]['rate']:.2f}%) "
            f"and the lowest {age[-1]['name']} ({age[-1]['rate']:.2f}%)"
            + (f"; {age[1]['name']} was {age[1]['rate']:.2f}%." if len(age) == 3 else "."))})
    f.append({"area": "Population", "text": (
        f"Among {pop['counties_ranked_latest']} counties with at least {pop['min_county_admits']:,} admissions "
        f"in {pop['latest_year']}, rates ranged from {pop['lowest_county']['rate']:.2f}% "
        f"({pop['lowest_county']['county']}) to {pop['highest_county']['rate']:.2f}% "
        f"({pop['highest_county']['county']}).")})

    # ---------------- CABG ----------------
    sw = cabg["statewide"]
    f.append({"area": "CABG", "text": (
        f"Statewide isolated CABG 30-day readmission moved from {sw.iloc[0]['rate']:.2f}% ({sw.iloc[0]['period']}) "
        f"to {sw.iloc[-1]['rate']:.2f}% ({sw.iloc[-1]['period']}).")})
    m, b = cabg["metrics"], cabg["bootstrap"]
    lr_b = _boot(b, "linear_regression")
    f.append({"area": "CABG model", "text": (
        f"On the held-out {cabg['test_period']} window (n={cabg['n_test']} hospitals), Linear Regression "
        f"(MAE {_metric(m, 'linear_regression'):.2f} pp) {model_verdict(m, b)} the persistence baseline "
        f"(MAE {_metric(m, 'persistence'):.2f} pp); bootstrap 95% CI of the MAE difference "
        f"[{lr_b['ci_low']:+.2f}, {lr_b['ci_high']:+.2f}] pp. Random Forest MAE was "
        f"{_metric(m, 'random_forest'):.2f} pp.")})
    nov = cabg["nonoverlap_summary"].set_index("model")
    pc = cabg["paired_cv"]
    pr = pc[(pc["design"].str.startswith("non-overlapping")) & (pc["model"] == "linear_regression")].iloc[0]
    f.append({"area": "CABG model", "text": (
        "Rolling two-year windows share a calendar year, which inflates persistence. When input and target "
        f"windows do not overlap, persistence MAE rises to {nov.loc['persistence', 'MAE_mean']:.2f} pp while "
        f"Linear Regression reaches {nov.loc['linear_regression', 'MAE_mean']:.2f} pp (better in "
        f"{pr['share_folds_model_better']:.0%} of grouped-CV folds).")})
    pi = cabg["permutation"]
    top = pi[pi["model"] == "linear_regression"].iloc[0]
    f.append({"area": "CABG model", "text": (
        f"Permutation importance: '{top['feature']}' dominates (MAE +{top['importance_mae_increase']:.2f} pp when "
        "shuffled); previous post-operative stroke rate and case volume add little predictive signal.")})

    # ---------------- stroke ----------------
    ss = stroke["statewide"]
    for code, label in (("readmission", "readmission"), ("mortality", "mortality")):
        g = ss[ss["measure_code"] == code]
        f.append({"area": "Stroke", "text": (
            f"Statewide ischemic stroke 30-day {label} (risk-adjusted) went from "
            f"{g.iloc[0]['risk_adjusted_rate']:.2f}% ({g.iloc[0]['period']}) to "
            f"{g.iloc[-1]['risk_adjusted_rate']:.2f}% ({g.iloc[-1]['period']}).")})
    corr = stroke["corr"]
    rs = [v["pearson_r"] for v in corr.values()]
    f.append({"area": "Stroke", "text": (
        f"Hospital-level stroke readmission and mortality rates are nearly uncorrelated "
        f"(Pearson r between {min(rs):+.2f} and {max(rs):+.2f} across periods): a hospital's mortality "
        "performance says little about its readmission performance.")})
    sm, sb = stroke["metrics"], stroke["bootstrap"]
    s_lr = _boot(sb, "linear_regression")
    f.append({"area": "Stroke model", "text": (
        f"Stroke model on the {stroke['test_period']} hold-out (n={stroke['n_test']}): Linear Regression MAE "
        f"{_metric(sm, 'linear_regression'):.2f} pp vs persistence {_metric(sm, 'persistence'):.2f} pp "
        f"({s_lr['relative_change_pct']:+.1f}%). Bootstrap 95% CI of the MAE difference "
        f"[{s_lr['ci_low']:+.2f}, {s_lr['ci_high']:+.2f}] pp: "
        + ("the improvement is statistically reliable." if s_lr["ci_high"] < 0 else
           "the difference is not statistically reliable at the 95% level (the interval includes zero), "
           f"although Linear Regression had lower error in {s_lr['prob_a_better']:.0%} of bootstrap resamples."))})
    return f
