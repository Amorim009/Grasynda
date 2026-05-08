"""
Fill missing privacy rows using the same PyMDMA tabular logic as the older
compute_pymdma_corrected.py workflow.

This script reads an existing privacy result file, identifies any rows present
in the manifest but absent from the privacy output, computes those rows, and
writes patched detailed/summary CSVs.
"""

from __future__ import annotations

import os
import sys

import pandas as pd


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from scripts.experiments.privacy_experiments.compute_release_privacy_metrics import (  # noqa: E402
    compute_privacy_metrics,
    compute_pymdma_metrics,
    prepare_aligned_matrices,
)


MANIFEST_PATH = os.path.join(
    PROJECT_ROOT,
    "assets",
    "results",
    "release_benchmark",
    "release_all_20260505",
    "release_manifest.csv",
)
BASE_RESULTS_PATH = os.path.join(
    PROJECT_ROOT,
    "assets",
    "results",
    "release_benchmark",
    "privacy_all_with_pymdma_20260506",
    "privacy_results_detailed.csv",
)


def main() -> None:
    manifest = pd.read_csv(MANIFEST_PATH)
    manifest = manifest[manifest["Family"].isin(["Grasynda", "OtherAugmentation", "AnonymizedOriginal"])].copy()
    base = pd.read_csv(BASE_RESULTS_PATH)

    key_cols = ["Dataset", "Group", "Family", "Method", "Variant_Name", "Epsilon", "Seed"]
    missing = manifest.merge(base[key_cols], on=key_cols, how="left", indicator=True)
    missing = missing[missing["_merge"] == "left_only"].drop(columns=["_merge"])

    if missing.empty:
        print("No missing rows found.")
        return

    out_dir = os.path.dirname(BASE_RESULTS_PATH)
    patch_rows = []

    print(f"Missing rows to fill: {len(missing)}", flush=True)
    for _, release_row in missing.iterrows():
        print(f"\n### {release_row['Dataset']} - {release_row['Group']} | {release_row['Variant_Name']} ###", flush=True)
        real_train = pd.read_csv(release_row["Real_Train_Path"])
        released_train = pd.read_csv(release_row["Released_Train_Path"])

        metrics = compute_privacy_metrics(real_train, released_train)
        real_matrix, candidate_matrix, real_ids, candidate_ids, _ = prepare_aligned_matrices(real_train, released_train)
        metrics.update(compute_pymdma_metrics(real_matrix, candidate_matrix, real_ids, candidate_ids))

        row = {col: release_row[col] for col in key_cols}
        row.update(metrics)
        patch_rows.append(row)

        print(
            f"  -> DCR={row['DCR_Mean']:.4f} NNDR={row['NNDR_Mean']:.4f} PyMDMA={row['PyMDMA_Privacy']:.4f}",
            flush=True,
        )

    patch_df = pd.DataFrame(patch_rows)
    patch_path = os.path.join(out_dir, "privacy_results_tsmixup_patch.csv")
    patch_df.to_csv(patch_path, index=False)

    complete = pd.concat([base, patch_df], ignore_index=True, sort=False)
    complete = complete.sort_values(key_cols).reset_index(drop=True)
    complete_path = os.path.join(out_dir, "privacy_results_detailed_complete.csv")
    complete.to_csv(complete_path, index=False)

    summary_cols = [
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
    summary = (
        complete.groupby(["Family", "Method", "Variant_Name", "Epsilon"], dropna=False)[summary_cols]
        .mean()
        .reset_index()
        .sort_values(["Family", "DCR_Mean"], ascending=[True, False])
    )
    summary_path = os.path.join(out_dir, "privacy_results_summary_complete.csv")
    summary.to_csv(summary_path, index=False)

    print("\nDONE", flush=True)
    print(f"Patch:    {patch_path}", flush=True)
    print(f"Detailed: {complete_path}", flush=True)
    print(f"Summary:  {summary_path}", flush=True)


if __name__ == "__main__":
    main()
