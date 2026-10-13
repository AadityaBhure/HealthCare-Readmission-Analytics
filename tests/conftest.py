import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import data_loader, preprocessing  # noqa: E402


@pytest.fixture(scope="session")
def raw():
    return data_loader.load_all_raw()


@pytest.fixture(scope="session")
def clean(raw):
    log = preprocessing.PreprocessingLog()
    return {
        "cabg": preprocessing.clean_cabg(raw["cabg"], log),
        "stroke": preprocessing.clean_stroke(raw["stroke"], log),
        "allcause": preprocessing.clean_allcause(raw["allcause"], log),
        "synthetic": preprocessing.prepare_synthetic(raw["synthetic"], log),
        "log": log,
    }


@pytest.fixture(scope="session")
def fast_run(tmp_path_factory):
    """Run the whole pipeline once in fast mode into a temporary directory."""
    import run_pipeline

    out = tmp_path_factory.mktemp("pipeline_run")
    rc = run_pipeline.main(["--fast", "--out-dir", str(out)])
    return {"rc": rc, "root": out}
