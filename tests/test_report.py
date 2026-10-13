"""Report generation from pipeline artifacts (DOCX + PDF)."""
import zipfile

import pandas as pd

from src import config


def test_report_generates_docx_and_pdf_from_artifacts(tmp_path):
    from reports.generate_report import main

    assert main(["--out-dir", str(tmp_path)]) == 0
    docx, pdf = tmp_path / "Project_Report.docx", tmp_path / "Project_Report.pdf"
    assert docx.stat().st_size > 100_000 and pdf.stat().st_size > 100_000
    assert pdf.read_bytes()[:5] == b"%PDF-"
    xml = zipfile.ZipFile(docx).read("word/document.xml").decode("utf-8")
    # the comparative table must contain the real persistence MAE written by the pipeline
    m = pd.read_csv(config.ARTIFACTS_DIR / "cabg" / "metrics_test.csv")
    pers = float(m.loc[m["model"] == "persistence", "MAE"].iloc[0])
    assert f"{pers:.3f}" in xml
    for heading in ("Abstract", "Comparative Model Performance", "Limitations", "References"):
        assert heading in xml
    assert len(list((tmp_path / "figures").glob("*.png"))) >= 20
