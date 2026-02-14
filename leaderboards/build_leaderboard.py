from __future__ import annotations

from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "submitted"
OUT_MD = ROOT / "leaderboards" / "leaderboard.md"


def main():
    rows = []
    for metrics in RESULTS.rglob("metrics.csv"):
        algo = metrics.parts[-3]
        version = metrics.parts[-2]
        df = pd.read_csv(metrics)
        df["algorithm"] = algo
        df["archive_version"] = version
        rows.append(df)

    if not rows:
        OUT_MD.write_text("# Leaderboard\n\nNo results submitted yet.\n", encoding="utf-8")
        return

    all_df = pd.concat(rows, ignore_index=True)

    # Example view: pivot for accuracy (max reported score per dataset per algorithm)
    acc = all_df[all_df["metric"] == "accuracy"].copy()
    if acc.empty:
        OUT_MD.write_text("# Leaderboard\n\nNo accuracy results found.\n", encoding="utf-8")
        return

    pivot = acc.pivot_table(index="dataset", columns="algorithm", values="score", aggfunc="max")

    md = []
    md.append("# Leaderboard\n")
    md.append("This file is generated. Do not edit directly.\n\n")
    md.append("## Accuracy (max reported score per dataset)\n\n")
    md.append(pivot.to_markdown(floatfmt=".4f"))
    md.append("\n")

    OUT_MD.write_text("".join(md), encoding="utf-8")


if __name__ == "__main__":
    main()
