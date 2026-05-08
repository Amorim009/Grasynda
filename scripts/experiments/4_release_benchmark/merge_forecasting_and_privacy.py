"""
Merge released-data forecasting and privacy results.

Configuration lives below; edit paths there rather than using CLI arguments.
"""

from __future__ import annotations

import os

import pandas as pd


FORECASTING_RESULTS_PATH = os.path.join(
    "assets",
    "results",
    "release_benchmark",
    "forecasting_universal_grasynda_da_faithful_lpa_fpa_eps048_24_20260508",
    "forecasting_results_detailed.csv",
)
PRIVACY_RESULTS_PATH = os.path.join(
    "assets",
    "results",
    "release_benchmark",
    "privacy_universal_grasynda_da_faithful_lpa_fpa_eps048_24_20260508",
    "privacy_results_detailed.csv",
)
OUTPUT_DIR = os.path.join(
    "assets",
    "results",
    "release_benchmark",
    "merged_universal_grasynda_da_faithful_lpa_fpa_eps048_24_20260508",
)

JOIN_COLS = ["Dataset", "Group", "Family", "Method", "Variant_Name", "Epsilon", "Seed"]
PRIVACY_METRIC_PRIORITY = [
    "DCR_Mean",
    "DCR_Median",
    "NNDR_Mean",
    "NNDR_Median",
    "RealNN_Mean",
    "PyMDMA_Authenticity",
    "PyMDMA_Fidelity",
    "PyMDMA_Diversity",
    "PyMDMA_Privacy",
]


def get_privacy_metric_cols(privacy_df: pd.DataFrame) -> list[str]:
    return [col for col in PRIVACY_METRIC_PRIORITY if col in privacy_df.columns]


def main() -> None:
    forecasting_df = pd.read_csv(os.path.abspath(FORECASTING_RESULTS_PATH))
    privacy_df = pd.read_csv(os.path.abspath(PRIVACY_RESULTS_PATH))
    privacy_metric_cols = get_privacy_metric_cols(privacy_df)
    merged = forecasting_df.merge(privacy_df, on=JOIN_COLS, how="left")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    detailed_path = os.path.join(OUTPUT_DIR, "merged_results_detailed.csv")
    merged.to_csv(detailed_path, index=False)

    summary_df = (
        merged.groupby(
            ["Family", "Method", "Variant_Name", "Epsilon", "Forecasting_Model", "Evaluation_Protocol"],
            dropna=False,
        )[["Utility_MASE"] + privacy_metric_cols]
        .mean()
        .reset_index()
        .sort_values(["Forecasting_Model", "Evaluation_Protocol", "Utility_MASE"])
    )
    summary_path = os.path.join(OUTPUT_DIR, "merged_results_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    by_method_df = (
        merged.groupby(["Method", "Epsilon", "Forecasting_Model", "Evaluation_Protocol"], dropna=False)[
            ["Utility_MASE"] + privacy_metric_cols
        ]
        .mean()
        .reset_index()
        .sort_values(["Forecasting_Model", "Evaluation_Protocol", "Utility_MASE"])
    )
    by_method_path = os.path.join(OUTPUT_DIR, "merged_results_by_method.csv")
    by_method_df.to_csv(by_method_path, index=False)

    print("### DONE ###", flush=True)
    print(f"Detailed:  {detailed_path}", flush=True)
    print(f"Summary:   {summary_path}", flush=True)
    print(f"By method: {by_method_path}", flush=True)


if __name__ == "__main__":
    main()
