"""Synthetic-record handling and end-to-end pipeline outputs."""
import json

import numpy as np
import pandas as pd

from src import synthetic_testing


def test_synthetic_records_identified(clean):
    syn = clean["synthetic"]
    assert syn["Synthetic_Record_Flag"].eq(True).all()
    assert syn["Hospital_ID"].str.startswith("SYNTH_").all()
    assert syn["is_synthetic"].all()
    for col in ("prev_readmit_rate", "prev_stroke_rate", "prev_cabg_volume"):
        assert col in syn and np.isfinite(syn[col]).all()


def test_synthetic_scenarios_run_on_saved_models(clean, fast_run):
    from src import cabg_model

    models, schema = cabg_model.load_bundle(fast_run["root"] / "models")
    preds, summary = synthetic_testing.run_scenarios(clean["synthetic"], models, schema)
    assert summary["all_flagged_synthetic"] and summary["all_predictions_finite"]
    assert summary["predictions_in_0_100"] and summary["deterministic"]
    assert len(preds) == 120
    assert "not evidence" in summary["interpretation"].lower()


def test_pipeline_writes_expected_outputs(fast_run):
    root = fast_run["root"]
    assert fast_run["rc"] == 0
    expected = [
        "models/cabg_models.joblib", "models/cabg_feature_schema.json", "models/stroke_models.joblib",
        "artifacts/key_findings.json", "artifacts/validation_checks.csv", "artifacts/preprocessing_log.csv",
        "artifacts/dataset_audit.json", "artifacts/cabg/metrics_test.csv", "artifacts/cabg/bootstrap_mae_diff.csv",
        "artifacts/cabg/hospital_predictions.csv", "artifacts/stroke/metrics_test.csv",
        "artifacts/population/summary.json", "artifacts/synthetic/summary.json", "artifacts/pipeline_manifest.json",
        "data/processed/cabg_clean.csv", "data/processed/allcause_clean.csv", "DATA_DICTIONARY.md",
    ]
    missing = [p for p in expected if not (root / p).exists()]
    assert not missing, missing
    checks = pd.read_csv(root / "artifacts" / "validation_checks.csv")
    assert not (checks["status"] == "fail").any()
    findings = json.loads((root / "artifacts" / "key_findings.json").read_text())
    assert len(findings) >= 8 and all(f["text"] for f in findings)


def test_pipeline_reports_missing_data_file_clearly(tmp_path, monkeypatch, capsys):
    import run_pipeline
    from src import config

    monkeypatch.setattr(config, "RAW_DIR", tmp_path)  # empty directory -> files missing
    rc = run_pipeline.main(["--fast", "--out-dir", str(tmp_path / "out")])
    assert rc == 1
    assert "not found" in capsys.readouterr().err
