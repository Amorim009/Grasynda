"""
Compute privacy metrics for pre-materialized released train sets.

This script compares each saved released train set against the same saved real
train split used to create it. It always computes DCR/NNDR and can optionally
also compute PyMDMA tabular privacy/quality metrics on the same aligned series
matrices.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", ".."))
sys.path.insert(0, PROJECT_ROOT)


# =========================
# Configuration
# =========================

MANIFEST_PATH = os.path.join(
    PROJECT_ROOT,
    "assets",
    "results",
    "release_benchmark",
    "release_universal_grasynda_da_faithful_lpa_fpa_20260508",
    "release_manifest.csv",
)
FAMILIES = [
    "Baseline",
    "Grasynda",
    "OtherAugmentation",
    "AnonymizedOriginal",
]
EPSILON_FILTER = [0.48, 24.0]
OUTPUT_TAG = "privacy_universal_grasynda_da_faithful_lpa_fpa_eps048_24_20260508"
WITH_PYMDMA = True
PYMDMA_N_WORKERS = 1


def parse_csv_list(value: Optional[str]) -> List[str]:
    if value is None or value.strip() == "":
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def load_series_as_lists(df: pd.DataFrame) -> Tuple[List[np.ndarray], List[str]]:
    series_list = []
    series_ids = []
    for uid, group_df in df.groupby("unique_id", sort=False):
        values = group_df.sort_values("ds")["y"].to_numpy(dtype=float)
        series_list.append(values)
        series_ids.append(uid)
    return series_list, series_ids


def truncate_to_length(series_list: Sequence[np.ndarray], length: int) -> np.ndarray:
    return np.asarray([series[-length:] for series in series_list], dtype=float)


def load_series_matrix(df: pd.DataFrame) -> Tuple[np.ndarray, int]:
    series_list, _ = load_series_as_lists(df)
    common_len = int(np.min([len(series) for series in series_list]))
    if common_len <= 1:
        raise ValueError(f"Common truncation length too small: {common_len}")
    return truncate_to_length(series_list, common_len), common_len


def zscore_with_real_reference(real_matrix: np.ndarray, candidate_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = real_matrix.mean(axis=0, keepdims=True)
    std = real_matrix.std(axis=0, keepdims=True)
    std = np.where(std <= 1e-12, 1.0, std)
    return (real_matrix - mean) / std, (candidate_matrix - mean) / std


def safe_ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    den = np.where(np.abs(den) <= 1e-12, np.nan, den)
    return num / den


def compute_privacy_metrics(real_df: pd.DataFrame, candidate_df: pd.DataFrame) -> dict:
    real_matrix, real_len = load_series_matrix(real_df)
    candidate_list, _ = load_series_as_lists(candidate_df)
    common_len = int(np.min([real_len] + [len(series) for series in candidate_list]))
    if common_len <= 1:
        raise ValueError(f"Common truncation length too small: {common_len}")

    candidate_matrix = truncate_to_length(candidate_list, common_len)
    real_matrix = real_matrix[:, -common_len:]
    real_z, cand_z = zscore_with_real_reference(real_matrix, candidate_matrix)

    nn = NearestNeighbors(n_neighbors=min(2, len(real_z)), metric="euclidean")
    nn.fit(real_z)

    rr_dist, _ = nn.kneighbors(real_z)
    real_nn = rr_dist[:, 0] if rr_dist.shape[1] == 1 else rr_dist[:, 1]

    cr_dist, _ = nn.kneighbors(cand_z)
    d1 = cr_dist[:, 0]
    d2 = cr_dist[:, 1] if cr_dist.shape[1] > 1 else np.full_like(d1, np.nan)
    nndr = safe_ratio(d1, d2)

    return {
        "DCR_Mean": float(np.nanmean(d1)),
        "DCR_Median": float(np.nanmedian(d1)),
        "NNDR_Mean": float(np.nanmean(nndr)),
        "NNDR_Median": float(np.nanmedian(nndr)),
        "RealNN_Mean": float(np.nanmean(real_nn)),
        "Common_Length": int(common_len),
        "Released_Series": int(candidate_df["unique_id"].nunique()),
    }


def compute_pymdma_metrics(
    real_matrix: np.ndarray,
    candidate_matrix: np.ndarray,
    real_ids: Sequence[str],
    candidate_ids: Sequence[str],
    n_workers: int,
) -> Dict[str, float]:
    try:
        from pymdma.tabular.measures.synthesis_val import (
            Authenticity,
            DCRPrivacy,
            ImprovedPrecision,
            ImprovedRecall,
        )
    except ImportError as exc:
        raise RuntimeError(f"PyMDMA not available: {exc}")

    del real_ids, candidate_ids
    scaler = StandardScaler()
    real_scaled = scaler.fit_transform(np.asarray(real_matrix, dtype=float))
    cand_scaled = scaler.transform(np.asarray(candidate_matrix, dtype=float))

    context = {}
    k = max(1, min(5, len(real_scaled) - 2, len(cand_scaled) - 2))
    auth = Authenticity(n_workers=n_workers)
    imp_prec = ImprovedPrecision(k=k, n_workers=n_workers)
    imp_rec = ImprovedRecall(k=k, n_workers=n_workers)
    dcr_priv = DCRPrivacy()

    return {
        "PyMDMA_Authenticity": float(auth.compute(real_scaled, cand_scaled, context=context).value[0]),
        "PyMDMA_Fidelity": float(imp_prec.compute(real_scaled, cand_scaled, context=context).value[0]),
        "PyMDMA_Diversity": float(imp_rec.compute(real_scaled, cand_scaled, context=context).value[0]),
        "PyMDMA_Privacy": float(dcr_priv.compute(real_scaled, cand_scaled).value[0]["privacy"] / 100),
    }


def prepare_aligned_matrices(
    real_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str], int]:
    real_list, real_ids = load_series_as_lists(real_df)
    candidate_list, candidate_ids = load_series_as_lists(candidate_df)
    common_len = int(np.min([len(series) for series in real_list + candidate_list]))
    if common_len <= 1:
        raise ValueError(f"Common truncation length too small: {common_len}")
    real_matrix = truncate_to_length(real_list, common_len)
    candidate_matrix = truncate_to_length(candidate_list, common_len)
    return real_matrix, candidate_matrix, real_ids, candidate_ids, common_len


def main() -> None:
    manifest_path = os.path.abspath(MANIFEST_PATH)
    manifest = pd.read_csv(manifest_path)
    families = set(FAMILIES)
    manifest = manifest[manifest["Family"].isin(families)].copy()
    if EPSILON_FILTER is not None:
        epsilons = {float(value) for value in EPSILON_FILTER}
        non_anonymized_mask = manifest["Family"] != "AnonymizedOriginal"
        epsilon_mask = manifest["Epsilon"].astype(float).isin(epsilons)
        manifest = manifest[non_anonymized_mask | epsilon_mask].copy()

    manifest_dir = os.path.dirname(manifest_path)
    base_dir = os.path.dirname(manifest_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = OUTPUT_TAG or f"privacy_{timestamp}"
    out_dir = os.path.join(base_dir, tag)
    os.makedirs(out_dir, exist_ok=True)

    config_payload = {
        "created_at": timestamp,
        "manifest": manifest_path,
        "families": sorted(families),
        "epsilon_filter": EPSILON_FILTER,
        "with_pymdma": WITH_PYMDMA,
        "pymdma_n_workers": PYMDMA_N_WORKERS,
    }
    with open(os.path.join(out_dir, "privacy_config.json"), "w", encoding="utf-8") as handle:
        json.dump(config_payload, handle, indent=2)

    rows = []

    for _, release_row in manifest.iterrows():
        print(f"\n### {release_row['Dataset']} - {release_row['Group']} | {release_row['Variant_Name']} ###", flush=True)
        try:
            real_train = pd.read_csv(release_row["Real_Train_Path"])
            released_train = pd.read_csv(release_row["Released_Train_Path"])
            metrics = compute_privacy_metrics(real_train, released_train)
            if WITH_PYMDMA:
                real_matrix, candidate_matrix, real_ids, candidate_ids, _ = prepare_aligned_matrices(real_train, released_train)
                metrics.update(
                    compute_pymdma_metrics(
                        real_matrix,
                        candidate_matrix,
                        real_ids,
                        candidate_ids,
                        n_workers=PYMDMA_N_WORKERS,
                    )
                )
            row = {
                "Dataset": release_row["Dataset"],
                "Group": release_row["Group"],
                "Family": release_row["Family"],
                "Method": release_row["Method"],
                "Variant_Name": release_row["Variant_Name"],
                "Epsilon": release_row["Epsilon"],
                "Seed": release_row["Seed"],
            }
            row.update(metrics)
            rows.append(row)
            print(
                f"  -> DCR={row['DCR_Mean']:.4f} NNDR={row['NNDR_Mean']:.4f}",
                flush=True,
            )
        except Exception as exc:
            print(f"  -> failed: {exc}", flush=True)

        if rows:
            pd.DataFrame(rows).to_csv(os.path.join(out_dir, "privacy_results_checkpoint.csv"), index=False)

    if not rows:
        raise RuntimeError("No privacy results were produced.")

    results_df = pd.DataFrame(rows)
    results_path = os.path.join(out_dir, "privacy_results_detailed.csv")
    results_df.to_csv(results_path, index=False)

    summary_cols = ["DCR_Mean", "DCR_Median", "NNDR_Mean", "NNDR_Median", "RealNN_Mean"]
    if WITH_PYMDMA:
        summary_cols.extend(
            [
                "PyMDMA_Authenticity",
                "PyMDMA_Fidelity",
                "PyMDMA_Diversity",
                "PyMDMA_Privacy",
            ]
        )
    summary_df = (
        results_df.groupby(["Family", "Method", "Variant_Name", "Epsilon"], dropna=False)[summary_cols]
        .mean()
        .reset_index()
        .sort_values(["Family", "DCR_Mean"], ascending=[True, False])
    )
    summary_path = os.path.join(out_dir, "privacy_results_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    print("\n### DONE ###", flush=True)
    print(f"Detailed: {results_path}", flush=True)
    print(f"Summary:  {summary_path}", flush=True)


if __name__ == "__main__":
    main()
