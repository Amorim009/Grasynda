"""
Build compact NHITS utility/privacy summary tables from merged results.

Edit the configuration block to point at the merged result folder.
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
METRIC_COLS = [
    "Utility_MASE",
    "DCR_Mean",
    "NNDR_Mean",
    "PyMDMA_Authenticity",
    "PyMDMA_Fidelity",
    "PyMDMA_Diversity",
    "PyMDMA_Privacy",
]


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
    df["Method_Label"] = df.apply(method_label, axis=1)
    cols = [col for col in METRIC_COLS if col in df.columns]
    summary = df.groupby("Method_Label", dropna=False)[cols].mean().reset_index()
    if "Utility_MASE" in summary.columns:
        summary = summary.sort_values("Utility_MASE")
    path = os.path.join(OUTPUT_DIR, "nhits_privacy_utility_summary.csv")
    summary.to_csv(path, index=False)
    print("### DONE ###", flush=True)
    print(f"Summary: {path}", flush=True)


if __name__ == "__main__":
    main()
