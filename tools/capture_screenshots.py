"""Capture dashboard screenshots for the report (dev-only helper).

Requires a running dashboard (streamlit run app.py) and `pip install playwright`.
Uses an installed Microsoft Edge or Google Chrome, so no browser download is needed:

    python tools/capture_screenshots.py --url http://localhost:8501
"""
import argparse
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "figures" / "screenshots"
SHOTS = [  # (file name, url path, scroll offset in px)
    ("page1_overview.png", "", 0),
    ("page1_overview_findings.png", "", 1250),
    ("page2_cabg_prediction.png", "cabg-prediction", 0),
    ("page2_cabg_history.png", "cabg-prediction", 560),
    ("page3_stroke.png", "stroke-analytics", 0),
    ("page4_population.png", "population-trends", 0),
    ("page5_model_evaluation.png", "model-evaluation", 0),
    ("page5_model_charts.png", "model-evaluation", 820),
]
SCROLL_JS = """(y) => { const el = document.querySelector('[data-testid="stMain"]') ||
  document.querySelector('section.main') || document.scrollingElement; el.scrollTo(0, y); }"""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8501")
    ap.add_argument("--channel", default="msedge", help="msedge or chrome (installed browser)")
    args = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(channel=args.channel, headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 960}, device_scale_factor=1.5)
        for name, path, y in SHOTS:
            page.goto(f"{args.url.rstrip('/')}/{path}", wait_until="networkidle")
            page.wait_for_selector('[data-testid="stPlotlyChart"]', timeout=60000)
            page.wait_for_timeout(3500)  # let Plotly finish drawing
            if y:
                page.evaluate(SCROLL_JS, y)
                page.wait_for_timeout(1200)
            page.screenshot(path=str(OUT / name))
            print("saved", OUT / name)
        browser.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
