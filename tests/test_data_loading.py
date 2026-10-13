"""Dataset loading, required columns and basic integrity."""
from src import config, data_validation


def test_all_four_datasets_load_with_expected_rows(raw):
    assert set(raw) == {"cabg", "stroke", "allcause", "synthetic"}
    assert len(raw["cabg"]) == 1937
    assert len(raw["stroke"]) == 2188
    assert len(raw["allcause"]) == 1057
    assert len(raw["synthetic"]) == 120


def test_required_columns_present(raw):
    for key, spec in config.DATASETS.items():
        cols = {c.strip() for c in raw[key].columns}
        missing = [c for c in spec.required_columns if c.strip() not in cols]
        assert not missing, f"{key} missing {missing}"


def test_windows_1252_files_decoded(raw):
    # The CABG and stroke files contain Windows-1252 en dashes (byte 0x96) and are not valid UTF-8.
    assert raw["cabg"].attrs["encoding"] == "cp1252"
    assert raw["stroke"].attrs["encoding"] == "cp1252"
    assert raw["cabg"]["Hospital"].str.contains("–").any()


def test_raw_validation_has_no_failures(raw):
    checks = data_validation.validate_raw(raw)
    assert not [c for c in checks if c.status == "fail"]


def test_processed_validation_has_no_failures(clean):
    checks = data_validation.validate_processed(clean["cabg"], clean["stroke"], clean["allcause"], clean["synthetic"])
    failed = [c for c in checks if c.status == "fail"]
    assert not failed, failed
    ids = {c.check_id for c in checks if c.status == "warn"}
    assert "cabg.readmission_coverage" in ids  # 2020-2021 sparsity must be surfaced, not hidden
