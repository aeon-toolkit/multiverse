from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files
import pandas as pd


@dataclass(frozen=True)
class DatasetSpec:
    dataset: str
    zenodo_record_id: str
    artifact_path: str
    sha256: str
    format: str
    notes: str
    licence: str


def load_registry() -> pd.DataFrame:
    reg_path = files("multiverse").joinpath("datasets/mtsc_registry.csv")
    return pd.read_csv(reg_path)


def get_spec(name: str) -> DatasetSpec:
    df = load_registry()
    row = df.loc[df["dataset"].str.lower() == name.lower()]
    if row.empty:
        raise KeyError(f"Unknown dataset: {name}. Use list_datasets().")
    r = row.iloc[0].to_dict()
    return DatasetSpec(**r)


def list_datasets() -> list[str]:
    df = load_registry()
    return sorted(df["dataset"].tolist())
