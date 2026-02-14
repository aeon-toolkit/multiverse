from __future__ import annotations

import os
from pathlib import Path
import requests
from tqdm import tqdm

from multiverse.utils._hash import sha256_file


def _cache_dir() -> Path:
    base = Path(os.environ.get("MULTIVERSE_CACHE", Path.home() / ".multiverse"))
    d = base / "datasets"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _zenodo_download_url(record_id: str, artifact_path: str) -> str:
    # Simple form which works for public Zenodo records if the filename matches exactly.
    # For robustness, you can query Zenodo's API and locate the file by filename.
    return f"https://zenodo.org/records/{record_id}/files/{artifact_path}?download=1"


def download_artifact(record_id: str, artifact_path: str, expected_sha256: str | None) -> Path:
    cache = _cache_dir()
    out = cache / artifact_path

    if out.exists() and expected_sha256 and expected_sha256 != "REPLACE_ME":
        if sha256_file(out) == expected_sha256.lower():
            return out

    url = _zenodo_download_url(record_id, artifact_path)
    r = requests.get(url, stream=True, timeout=60)
    r.raise_for_status()

    total = int(r.headers.get("Content-Length", 0))
    tmp = out.with_suffix(out.suffix + ".tmp")

    with tmp.open("wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=f"Downloading {artifact_path}") as pbar:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))

    tmp.replace(out)

    if expected_sha256 and expected_sha256 != "REPLACE_ME":
        got = sha256_file(out)
        if got.lower() != expected_sha256.lower():
            out.unlink(missing_ok=True)
            raise ValueError(
                f"Checksum mismatch for {artifact_path}. Expected {expected_sha256}, got {got}."
            )

    return out
