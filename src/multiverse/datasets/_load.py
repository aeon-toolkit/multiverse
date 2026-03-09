from __future__ import annotations

from pathlib import Path
import zipfile

from aeon.datasets import load_from_tsfile

from multiverse.datasets._registry import get_spec
from multiverse.datasets._download import download_artifact


def load_dataset(name: str, split: str = "train"):
    """Load a dataset split into aeon-compatible X, y.

    Assumes the Zenodo artefact is a zip containing `<NAME>_TRAIN.ts` and `<NAME>_TEST.ts`.
    """
    spec = get_spec(name)
    if spec.zenodo_record_id == "REPLACE_ME":
        raise ValueError(
            "Dataset registry contains placeholders. Replace zenodo_record_id and sha256 in mtsc_registry.csv."
        )

    zip_path = download_artifact(spec.zenodo_record_id, spec.artifact_path, spec.sha256)

    split = split.lower()
    if split not in {"train", "test"}:
        raise ValueError("split must be 'train' or 'test'")

    with zipfile.ZipFile(zip_path) as zf:
        target = f"{spec.dataset}_{split.upper()}.ts"
        members = [m for m in zf.namelist() if m.endswith(target)]
        if not members:
            raise FileNotFoundError(f"Could not find {target} inside {spec.artifact_path}")
        ts_member = members[0]

        extract_dir = zip_path.with_suffix("")  # e.g. ~/.multiverse/datasets/BasicMotions/
        extract_dir.mkdir(exist_ok=True)
        out_path = extract_dir / Path(ts_member).name
        if not out_path.exists():
            zf.extract(ts_member, path=extract_dir)
            extracted = extract_dir / ts_member
            if extracted != out_path:
                extracted.replace(out_path)

    X, y = load_from_tsfile(str(out_path))
    return X, y
