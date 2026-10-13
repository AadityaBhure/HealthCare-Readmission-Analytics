"""Start the Streamlit dashboard from the project root regardless of the caller's working
directory (used by editor/preview launchers). Equivalent to: streamlit run app.py"""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
port = os.environ.get("PORT", "8501")
sys.exit(subprocess.call([sys.executable, "-m", "streamlit", "run", "app.py", "--server.port", port,
                          "--server.headless", "true"]))
