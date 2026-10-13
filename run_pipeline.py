"""Backend analysis pipeline.

    python run_pipeline.py            # full run (about 1 minute)
    python run_pipeline.py --fast     # quick smoke run (fewer trees / repeats) - used by tests
    python run_pipeline.py --report   # full run, then regenerate reports/Project_Report.docx + .pdf

Steps: load raw CSVs -> validate -> clean -> audit & data dictionary -> CABG experiment ->
stroke experiment -> population analytics -> synthetic scenario tests -> key findings.
Outputs go to data/processed/, models/, artifacts/ and DATA_DICTIONARY.md. Raw files are
never modified. The process exits with status 1 and a clear message if a check fails.
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src import (cabg_model, config, data_dictionary, data_loader, data_validation, insights,
                 model_store, population_analysis, preprocessing, stroke_analysis, synthetic_testing)


def _clean_json(obj):
    if isinstance(obj, float):
        return None if math.isnan(obj) or math.isinf(obj) else obj
    if isinstance(obj, dict):
        return {str(k): _clean_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean_json(v) for v in obj]
    if hasattr(obj, "item"):  # numpy scalar
        return _clean_json(obj.item())
    return obj


def write_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_clean_json(obj), indent=2, default=str), encoding="utf-8")


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def save_experiment(out: dict, folder: Path) -> None:
    r = out["result"]
    write_csv(out["transitions"], folder / "transitions.csv")
    write_csv(out["excluded"], folder / "excluded_transitions.csv")
    write_csv(r.test_predictions, folder / "test_predictions.csv")
    write_csv(r.test_metrics, folder / "metrics_test.csv")
    write_csv(r.train_cv_selection, folder / "train_cv_selection.csv")
    write_csv(r.bootstrap, folder / "bootstrap_mae_diff.csv")
    write_csv(r.permutation_importance, folder / "permutation_importance.csv")
    write_csv(r.model_native_importance, folder / "native_importance.csv")
    write_csv(r.error_by_volume, folder / "error_by_volume.csv")
    write_csv(r.learned_curve, folder / "learned_curve.csv")
    write_csv(out["cv_rows"], folder / "cv_pooled_folds.csv")
    write_csv(out["cv_summary"], folder / "cv_pooled_summary.csv")
    write_csv(out["nonoverlap_rows"], folder / "cv_nonoverlap_folds.csv")
    write_csv(out["nonoverlap_summary"], folder / "cv_nonoverlap_summary.csv")
    write_csv(out["paired_cv"], folder / "paired_cv_vs_persistence.csv")
    write_json({
        "recommended_model": r.recommended_model,
        "rf_best_params": r.rf_best_params,
        "train_summary": r.train_summary,
        "n_excluded": int(len(out["excluded"])),
        "n_pooled": int(out["n_pooled"]),
        "n_nonoverlap": int(out["n_nonoverlap"]),
        "lag_correlation": out["lag_correlation"],
    }, folder / "summary.json")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fast", action="store_true", help="quick smoke run (fewer trees/repeats)")
    ap.add_argument("--report", action="store_true", help="regenerate the DOCX/PDF report afterwards")
    ap.add_argument("--out-dir", default=None, help="write outputs under this directory instead of the project root")
    args = ap.parse_args(argv)

    root = Path(args.out_dir).resolve() if args.out_dir else config.PROJECT_ROOT
    processed, models_dir, art = root / "data" / "processed", root / "models", root / "artifacts"
    for d in (processed, models_dir, art):
        d.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    def step(msg: str) -> None:
        print(f"[{time.time() - t0:6.1f}s] {msg}", flush=True)

    step("1/8 Loading raw datasets")
    try:
        raw = data_loader.load_all_raw()
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    hashes = {k: v.attrs["sha256"] for k, v in raw.items()}

    step("2/8 Validating raw schemas")
    checks = data_validation.validate_raw(raw)
    if any(c.status == "fail" for c in checks):
        for c in checks:
            if c.status == "fail":
                print(f"FAILED CHECK {c.check_id}: {c.description} -> {c.detail}", file=sys.stderr)
        return 1

    step("3/8 Cleaning and preprocessing")
    log = preprocessing.PreprocessingLog()
    cabg = preprocessing.clean_cabg(raw["cabg"], log)
    stroke = preprocessing.clean_stroke(raw["stroke"], log)
    allcause = preprocessing.clean_allcause(raw["allcause"], log)
    synthetic = preprocessing.prepare_synthetic(raw["synthetic"], log)
    checks += data_validation.validate_processed(cabg, stroke, allcause, synthetic)
    checks_df = data_validation.checks_frame(checks)
    write_csv(checks_df, art / "validation_checks.csv")
    summary = data_validation.summarize(checks)
    step(f"    validation: {summary['pass']} pass, {summary['warn']} warn, {summary['fail']} fail")
    if summary["fail"]:
        for c in checks:
            if c.status == "fail":
                print(f"FAILED CHECK {c.check_id}: {c.description} -> {c.detail}", file=sys.stderr)
        return 1
    for name, df in (("cabg_clean", cabg), ("stroke_clean", stroke), ("allcause_clean", allcause),
                     ("synthetic_clean", synthetic)):
        write_csv(df, processed / f"{name}.csv")
    write_csv(log.to_frame(), art / "preprocessing_log.csv")

    step("4/8 Dataset audit and data dictionary")
    clean = {"cabg": cabg, "stroke": stroke, "allcause": allcause, "synthetic": synthetic}
    audit = data_dictionary.audit(raw, clean)
    write_json(audit, art / "dataset_audit.json")
    data_dictionary.write_markdown(audit, log.to_frame(), checks_df, root / "DATA_DICTIONARY.md")

    step("5/8 CABG experiment (temporal hold-out, grouped CV, sensitivity analyses)")
    cabg_out = cabg_model.run(cabg, fast=args.fast)
    write_csv(cabg_out["panel"], processed / "cabg_panel.csv")
    cabg_sw = cabg_model.statewide_series(cabg)
    write_csv(cabg_sw, art / "cabg" / "statewide.csv")
    save_experiment(cabg_out, art / "cabg")
    write_csv(cabg_out["feature_set_comparison"], art / "cabg" / "feature_set_comparison.csv")
    write_csv(cabg_out["hospital_predictions"], art / "cabg" / "hospital_predictions.csv")
    cabg_schema = cabg_model.save_bundle(cabg_out, models_dir, hashes)
    m = cabg_out["result"].test_metrics.set_index("model")
    step(f"    CABG test MAE  persistence {m.loc['persistence', 'MAE']:.3f} | LR {m.loc['linear_regression', 'MAE']:.3f}"
         f" | RF {m.loc['random_forest', 'MAE']:.3f}  (recommended: {cabg_out['result'].recommended_model})")

    step("6/8 Ischemic stroke analytics and model")
    stroke_out = stroke_analysis.run(stroke, fast=args.fast)
    write_csv(stroke_out["panel"], processed / "stroke_panel.csv")
    write_csv(stroke_out["statewide"], art / "stroke" / "statewide.csv")
    write_csv(stroke_out["county_summary"], art / "stroke" / "county_summary.csv")
    write_json(stroke_out["descriptive"], art / "stroke" / "descriptive.json")
    save_experiment(stroke_out, art / "stroke")
    stroke_analysis.save_bundle(stroke_out, models_dir, hashes)
    m = stroke_out["result"].test_metrics.set_index("model")
    step(f"    Stroke test MAE persistence {m.loc['persistence', 'MAE']:.3f} | LR {m.loc['linear_regression', 'MAE']:.3f}"
         f" | RF {m.loc['random_forest', 'MAE']:.3f}")

    step("7/8 Population and county trends")
    pop = population_analysis.summarize(allcause)
    write_csv(population_analysis.statewide_overall(allcause), art / "population" / "statewide.csv")
    write_csv(population_analysis.county_overall(allcause), art / "population" / "county.csv")
    write_csv(population_analysis.strata_statewide(allcause), art / "population" / "strata.csv")
    write_json(pop, art / "population" / "summary.json")

    step("8/8 Synthetic scenario tests (using the models re-loaded from disk)")
    models, schema = cabg_model.load_bundle(models_dir)
    integrity = {
        "cabg": model_store.verify_bundle(models, schema, cabg_out["result"].test_predictions),
        "stroke": model_store.verify_bundle(*model_store.load_model_bundle("stroke", models_dir),
                                            stroke_out["result"].test_predictions),
    }
    write_json(integrity, art / "model_integrity.json")
    if not all(v["match"] for exp in integrity.values() for v in exp.values()):
        print("ERROR: reloaded models do not reproduce evaluated predictions", file=sys.stderr)
        return 1
    syn_preds, syn_summary = synthetic_testing.run_scenarios(synthetic, models, schema)
    write_csv(syn_preds, art / "synthetic" / "scenario_predictions.csv")
    write_json(syn_summary, art / "synthetic" / "summary.json")

    findings = insights.derive(pop, {
        "statewide": cabg_sw[cabg_sw["measure_code"] == "readmission"].reset_index(drop=True),
        "metrics": cabg_out["result"].test_metrics, "bootstrap": cabg_out["result"].bootstrap,
        "nonoverlap_summary": cabg_out["nonoverlap_summary"], "paired_cv": cabg_out["paired_cv"],
        "permutation": cabg_out["result"].permutation_importance,
        "test_period": cabg_out["result"].train_summary["test_target_periods"][0],
        "n_test": cabg_out["result"].train_summary["n_test"],
    }, {
        "statewide": stroke_out["statewide"], "corr": stroke_out["descriptive"]["readmission_vs_mortality_correlation"],
        "metrics": stroke_out["result"].test_metrics, "bootstrap": stroke_out["result"].bootstrap,
        "test_period": stroke_out["result"].train_summary["test_target_periods"][0],
        "n_test": stroke_out["result"].train_summary["n_test"],
    })
    write_json(findings, art / "key_findings.json")

    import numpy, sklearn, streamlit  # noqa: E401 - versions recorded for reproducibility
    write_json({
        "run_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "mode": "fast" if args.fast else "full",
        "duration_seconds": round(time.time() - t0, 1),
        "python": platform.python_version(), "platform": platform.platform(),
        "versions": {"pandas": pd.__version__, "numpy": numpy.__version__, "scikit-learn": sklearn.__version__,
                     "streamlit": streamlit.__version__},
        "data_sha256": hashes, "validation": summary,
        "cabg_recommended_model": cabg_schema["recommended_model"],
    }, art / "pipeline_manifest.json")
    step(f"Done. {len(findings)} key findings written to {art / 'key_findings.json'}")

    if args.report:
        step("Generating project report (DOCX + PDF)")
        from reports.generate_report import main as report_main
        report_main([])
    return 0


if __name__ == "__main__":
    sys.exit(main())
