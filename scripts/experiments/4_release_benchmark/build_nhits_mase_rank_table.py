"""
Build NHITS MASE tables and average ranks from merged release-benchmark results.

Edit the configuration block to point at the merged result folder you want to
turn into a paper table.
"""

from __future__ import annotations

import os

import pandas as pd


MERGED_RESULTS_PATH = os.path.join(
    "assets",
    "results",
    "release_benchmark",
    "merged_universal_grasynda_da_faithful_lpa_fpa_eps048_24_20260508",
    "merged_results_detailed.csv",
)
OUTPUT_DIR = os.path.dirname(MERGED_RESULTS_PATH)
FORECASTING_MODEL = "NHITS"

DATASET_LABELS = {
    ("M3", "Monthly"): "M3-M",
    ("M3", "Quarterly"): "M3-Q",
    ("Tourism", "Monthly"): "T-M",
    ("Tourism", "Quarterly"): "T-Q",
    ("Gluonts", "m1_monthly"): "M1-M",
    ("Gluonts", "m1_quarterly"): "M1-Q",
    ("NN3", "Monthly"): "NN3-M",
}


def method_label(row: pd.Series) -> str:
    if row["Family"] == "Baseline":
        return "Baseline (No Aug)"
    if pd.notna(row["Epsilon"]):
        return f"{row['Method']} ({float(row['Epsilon']):g})"
    return str(row["Method"])


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = pd.read_csv(MERGED_RESULTS_PATH)
    df = df[df["Forecasting_Model"] == FORECASTING_MODEL].copy()
    df["Dataset_Label"] = [DATASET_LABELS.get((row.Dataset, row.Group), f"{row.Dataset}-{row.Group}") for row in df.itertuples()]
    df["Method_Label"] = df.apply(method_label, axis=1)

    mase_pivot = df.pivot_table(
        index="Method_Label",
        columns="Dataset_Label",
        values="Utility_MASE",
        aggfunc="first",
    )
    ordered_cols = [label for label in DATASET_LABELS.values() if label in mase_pivot.columns]
    mase_pivot = mase_pivot[ordered_cols + [col for col in mase_pivot.columns if col not in ordered_cols]]

    rank_pivot = mase_pivot.rank(axis=0, ascending=True, method="average")
    average_rank = rank_pivot.mean(axis=1).rename("Avg Rank")
    table = mase_pivot.join(average_rank).sort_values("Avg Rank")

    mase_path = os.path.join(OUTPUT_DIR, "nhits_tstr_mase_per_method.csv")
    rank_path = os.path.join(OUTPUT_DIR, "nhits_tstr_rank_per_method.csv")
    table_path = os.path.join(OUTPUT_DIR, "nhits_tstr_mase_with_average_rank.csv")
    mase_pivot.to_csv(mase_path, index_label="Method")
    rank_pivot.to_csv(rank_path, index_label="Method")
    table.to_csv(table_path, index_label="Method")

    print("### DONE ###", flush=True)
    print(f"MASE table: {mase_path}", flush=True)
    print(f"Rank table: {rank_path}", flush=True)
    print(f"Combined:   {table_path}", flush=True)


if __name__ == "__main__":
    main()
