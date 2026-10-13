"""One-command local launcher:  python run_local.py

1. runs the analysis pipeline if its outputs are missing (or always, with --retrain);
2. optionally regenerates the report (--report);
3. starts the Streamlit dashboard (http://localhost:8501), which opens in your browser.
"""
from __future__ import annotations

import argparse
import subprocess
import sys

from src import config


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--retrain", action="store_true", help="re-run the pipeline even if outputs exist")
    ap.add_argument("--report", action="store_true", help="regenerate reports/Project_Report.docx and .pdf")
    ap.add_argument("--port", default="8501")
    args = ap.parse_args(argv)

    if args.retrain or not all(p.exists() for p in config.REQUIRED_ARTIFACTS):
        print(">> Running the analysis pipeline ...", flush=True)
        rc = subprocess.call([sys.executable, "run_pipeline.py"], cwd=config.PROJECT_ROOT)
        if rc != 0:
            print("Pipeline failed - fix the reported problem before starting the dashboard.", file=sys.stderr)
            return rc
    else:
        print(">> Pipeline outputs found (use --retrain to rebuild them).")
    if args.report:
        print(">> Generating the project report ...", flush=True)
        subprocess.call([sys.executable, "reports/generate_report.py"], cwd=config.PROJECT_ROOT)
    print(f">> Starting the dashboard on http://localhost:{args.port}  (Ctrl+C to stop)", flush=True)
    return subprocess.call([sys.executable, "-m", "streamlit", "run", "app.py", "--server.port", args.port,
                            "--server.headless", "false"], cwd=config.PROJECT_ROOT)


if __name__ == "__main__":
    sys.exit(main())
