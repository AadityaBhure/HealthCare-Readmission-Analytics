"""Raw dataset loading (local files or an optional raw-GitHub URL).

Raw files are read as text (``dtype=str``) so that the original formatting
("13.80%", "1,234", ".") reaches the preprocessing layer untouched and can be
parsed explicitly. Raw files on disk are never modified.
"""
from __future__ import annotations

import hashlib
import io
from pathlib import Path

import pandas as pd

from . import config


def raw_path(key: str, raw_dir: Path | None = None) -> Path:
    spec = config.DATASETS[key]
    return (raw_dir or config.RAW_DIR) / spec.filename


def _decode(payload: bytes, encodings: tuple[str, ...]) -> tuple[str, str]:
    """Decode bytes with the first encoding that works (CABG/stroke files are Windows-1252)."""
    last_error: Exception | None = None
    for enc in encodings:
        try:
            return payload.decode(enc), enc
        except UnicodeDecodeError as exc:  # pragma: no cover - exercised with real files
            last_error = exc
    raise UnicodeDecodeError("multi", b"", 0, 1, f"none of {encodings} could decode: {last_error}")


def _fetch_bytes(key: str, raw_dir: Path | None = None, source: str | None = None) -> bytes:
    spec = config.DATASETS[key]
    source = (source or config.DATA_SOURCE).lower()
    if source == "github":
        if not config.DATA_BASE_URL:
            raise RuntimeError(
                "READMISSION_DATA_SOURCE=github requires READMISSION_DATA_BASE_URL "
                "(e.g. https://raw.githubusercontent.com/<user>/<repo>/main/data/raw)."
            )
        import requests  # imported lazily; only needed for the remote source

        url = f"{config.DATA_BASE_URL}/{spec.filename}"
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        return resp.content
    path = raw_path(key, raw_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"Required dataset '{spec.filename}' not found at {path}. "
            "Place the four CSV files in data/raw/ (see README)."
        )
    return path.read_bytes()


def load_raw(key: str, raw_dir: Path | None = None, source: str | None = None) -> pd.DataFrame:
    """Load one raw dataset as strings. Also records the detected encoding in ``df.attrs``."""
    spec = config.DATASETS[key]
    payload = _fetch_bytes(key, raw_dir, source)
    text, encoding = _decode(payload, spec.encodings)
    df = pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False)
    df.attrs.update(
        {
            "dataset_key": key,
            "filename": spec.filename,
            "encoding": encoding,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
        }
    )
    return df


def load_all_raw(raw_dir: Path | None = None, source: str | None = None) -> dict[str, pd.DataFrame]:
    return {key: load_raw(key, raw_dir, source) for key in config.DATASETS}
