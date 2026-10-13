"""Dashboard smoke tests with Streamlit's AppTest (pages run headlessly against repo artifacts)."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
PAGES = ["1_Overview.py", "2_CABG_Prediction.py", "3_Stroke_Analytics.py", "4_Population_Trends.py",
         "5_Model_Evaluation.py"]


@pytest.mark.parametrize("page", PAGES)
def test_page_renders_without_exceptions(page):
    at = AppTest.from_file(str(ROOT / "pages" / page), default_timeout=180).run()
    assert not at.exception, [e.message for e in at.exception]


def test_app_entrypoint_runs():
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180).run()
    assert not at.exception, [e.message for e in at.exception]


def test_cabg_page_hospital_and_period_selection():
    at = AppTest.from_file(str(ROOT / "pages" / "2_CABG_Prediction.py"), default_timeout=180).run()
    hosp = at.selectbox[0]
    hosp.set_value(hosp.options[3]).run()
    assert not at.exception
    period = at.selectbox[1]
    period.set_value(period.options[0]).run()
    assert not at.exception
    text = " ".join(m.value for m in at.markdown)
    assert "hospital-level" in text.lower()


def test_scenario_extrapolation_warning():
    at = AppTest.from_file(str(ROOT / "pages" / "2_CABG_Prediction.py"), default_timeout=180).run()
    at.number_input[0].set_value(55.0).run()  # far above any observed readmission rate
    assert not at.exception
    text = " ".join(m.value for m in at.markdown)
    assert "outside the training range" in text
    assert "Hypothetical scenario" in text
