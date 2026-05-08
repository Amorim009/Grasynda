"""
Build privacy comparison tables with sanity-check rows.

This script:
1. Starts from the completed privacy benchmark file.
2. Replaces plain Grasynda with optimized Grasynda privacy rows.
3. Adds two sanity-check methods per dataset:
   - Baseline (No Aug): released data is the real train set itself
   - Random Noise: one synthetic series per real series, sampled uniformly
     within the original train-value range
4. Writes extended detailed/summary CSVs.
5. Writes LaTeX tables for DCR, NNDR, and PyMDMA privacy sorted by average rank.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from scripts.experiments.privacy_experiments.compute_release_privacy_metrics import (  # noqa: E402
    compute_privacy_metrics,
    compute_pymdma_metrics,
    prepare_aligned_matrices,
)


BASE_PRIVACY_PATH = os.path.join(
    PROJECT_ROOT,
    "assets",
    "results",
    "release_benchmark",
    "privacy_all_with_pymdma_20260506",
    "privacy_results_detailed_complete.csv",
)
OPT_GRASYNDA_PRIVACY_PATH = os.path.join(
    PROJECT_ROOT,
    "assets",
    "results",
    "release_benchmark",
    "privacy_all_grasynda_optimized_20260506",
    "privacy_results_detailed.csv",
)
MANIFEST_PATH = os.path.join(
    PROJECT_ROOT,
    "assets",
    "results",
    "release_benchmark",
    "release_all_20260505",
    "release_manifest.csv",
)
OUT_DIR = os.path.join(
    PROJECT_ROOT,
    "assets",
    "results",
    "release_benchmark",
    "privacy_all_with_pymdma_20260506",
)

METHOD_LABELS = {
    "Baseline (No Aug)": "Baseline (No Aug)",
    "Random Noise": "Random Noise",
    "Jittering": "Jitter",
    "SeasonalMBB": "SeasMBB",
    "Scaling": "Scaling",
    "TSMixup": "TSMix",
    "DBA": "DBA",
    "MagnitudeWarping": "MagWarp",
    "TimeVAE": "TimeVAE",
    "TimeWarping": "TimeWarp",
    "Grasynda_Optimized": "Grasynda (Opt)",
    "LPA_24.0": "LPA (24)",
    "LPA_4.8": "LPA (4.8)",
    "LPA_0.48": "LPA (0.48)",
    "tFPA_24.0": "tFPA (24)",
    "tFPA_4.8": "tFPA (4.8)",
    "tFPA_0.48": "tFPA (0.48)",
}

ALLOWED_BASE_METHODS = {
    "Baseline (No Aug)",
    "Random Noise",
    "Jittering",
    "SeasonalMBB",
    "Scaling",
    "TSMixup",
    "DBA",
    "MagnitudeWarping",
    "TimeVAE",
    "TimeWarping",
    "Grasynda_Optimized",
}

ALLOWED_DP_VARIANTS = {
    ("LPA", 24.0),
    ("LPA", 4.8),
    ("LPA", 0.48),
    ("tFPA", 24.0),
    ("tFPA", 4.8),
    ("tFPA", 0.48),
}

DISPLAY_DP_VARIANTS = {
    ("LPA", 24.0),
    ("LPA", 4.8),
    ("tFPA", 24.0),
    ("tFPA", 4.8),
}

NO_RANK_METHODS = {
    "Baseline (No Aug)",
    "Random Noise",
}

DATASET_COLS = {
    ("M3", "Monthly"): "M3-M",
    ("M3", "Quarterly"): "M3-Q",
    ("Tourism", "Monthly"): "T-M",
    ("Tourism", "Quarterly"): "T-Q",
    ("Gluonts", "m1_monthly"): "M1-M",
    ("Gluonts", "m1_quarterly"): "M1-Q",
    ("NN3", "Monthly"): "NN3-M",
}

METRIC_SPECS = {
    "DCR_Mean": {
        "caption": "Privacy evaluation with DCR and average rank. Higher values indicate greater empirical privacy separation. Best values are in bold and second-best values are underlined.",
        "label": "tab:nhits_dcr",
    },
    "NNDR_Mean": {
        "caption": "Privacy evaluation with NNDR and average rank. Higher values indicate greater empirical privacy separation. Best values are in bold and second-best values are underlined.",
        "label": "tab:nhits_nndr",
    },
    "PyMDMA_Privacy": {
        "caption": "Privacy evaluation with PyMDMA privacy and average rank. Higher values indicate stronger privacy according to the PyMDMA DCR-based privacy measure. Best values are in bold and second-best values are underlined.",
        "label": "tab:nhits_pymdma_privacy",
    },
}


def generate_random_noise_release(real_df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    lo = float(real_df["y"].min())
    hi = float(real_df["y"].max())
    frames: List[pd.DataFrame] = []
    for uid, group_df in real_df.groupby("unique_id", sort=False):
        part = group_df.sort_values("ds")[["ds"]].copy()
        part["unique_id"] = f"RandomNoise_{uid}"
        part["y"] = rng.uniform(lo, hi, size=len(part))
        frames.append(part[["unique_id", "ds", "y"]])
    return pd.concat(frames, ignore_index=True)


def compute_row(dataset: str, group: str, method_label: str, real_df: pd.DataFrame, candidate_df: pd.DataFrame) -> Dict[str, object]:
    metrics = compute_privacy_metrics(real_df, candidate_df)
    real_matrix, candidate_matrix, real_ids, candidate_ids, _ = prepare_aligned_matrices(real_df, candidate_df)
    metrics.update(compute_pymdma_metrics(real_matrix, candidate_matrix, real_ids, candidate_ids))
    row = {
        "Dataset": dataset,
        "Group": group,
        "Family": "SanityCheck",
        "Method": method_label,
        "Variant_Name": method_label,
        "Epsilon": np.nan,
        "Seed": 42,
    }
    row.update(metrics)
    return row


def build_sanity_rows(manifest: pd.DataFrame) -> pd.DataFrame:
    baseline_manifest = manifest[manifest["Family"] == "Baseline"].copy()
    rows: List[Dict[str, object]] = []
    for _, item in baseline_manifest.iterrows():
        dataset = item["Dataset"]
        group = item["Group"]
        real_df = pd.read_csv(item["Real_Train_Path"])
        rows.append(compute_row(dataset, group, "Baseline (No Aug)", real_df, real_df.copy()))
        noise_df = generate_random_noise_release(real_df, seed=42)
        rows.append(compute_row(dataset, group, "Random Noise", real_df, noise_df))
    return pd.DataFrame(rows)


def make_table_method(row: pd.Series) -> str:
    method = row["Method"]
    eps = row.get("Epsilon", np.nan)
    if method in ("LPA", "tFPA") and pd.notna(eps):
        key = f"{method}_{float(eps)}"
        if key not in METHOD_LABELS:
            raise KeyError(f"Unsupported DP variant for table rendering: {key}")
        return METHOD_LABELS[key]
    return METHOD_LABELS.get(method, METHOD_LABELS.get(row["Variant_Name"], str(row["Variant_Name"])))


def is_allowed_row(row: pd.Series) -> bool:
    method = row["Method"]
    if method in ALLOWED_BASE_METHODS:
        return True
    if method in ("LPA", "tFPA") and pd.notna(row.get("Epsilon", np.nan)):
        return (method, round(float(row["Epsilon"]), 2)) in ALLOWED_DP_VARIANTS
    return False


def is_display_row(row: pd.Series) -> bool:
    method = row["Method"]
    if method in ALLOWED_BASE_METHODS:
        return True
    if method in ("LPA", "tFPA") and pd.notna(row.get("Epsilon", np.nan)):
        return (method, round(float(row["Epsilon"]), 2)) in DISPLAY_DP_VARIANTS
    return False


def style_cell(value: float, best: float | None, second: float | None) -> str:
    text = f"{value:.3f}"
    if best is not None and np.isclose(value, best):
        return f"\\textbf{{{text}}}"
    if second is not None and np.isclose(value, second):
        return f"\\underline{{{text}}}"
    return text


def style_rank(value: float, best: float | None, second: float | None) -> str:
    text = f"{value:.2f}"
    if best is not None and np.isclose(value, best):
        return f"\\textbf{{{text}}}"
    if second is not None and np.isclose(value, second):
        return f"\\underline{{{text}}}"
    return text


def build_latex_table(df: pd.DataFrame, metric: str) -> str:
    work = df[df.apply(is_display_row, axis=1)].copy()
    work["TableMethod"] = work.apply(make_table_method, axis=1)
    work["DatasetCol"] = work.apply(lambda r: DATASET_COLS[(r["Dataset"], r["Group"])], axis=1)

    piv = work.pivot_table(index="TableMethod", columns="DatasetCol", values=metric, aggfunc="first")
    piv = piv[[c for c in ["M3-M", "M3-Q", "T-M", "T-Q", "M1-M", "M1-Q", "NN3-M"] if c in piv.columns]]
    ranked_methods = [idx for idx in piv.index if idx not in NO_RANK_METHODS]
    sanity_methods = [idx for idx in ["Baseline (No Aug)", "Random Noise"] if idx in piv.index]

    ranks = piv.loc[ranked_methods].rank(axis=0, ascending=False, method="average")
    avg_rank = ranks.mean(axis=1)
    sorted_ranked_methods = avg_rank.sort_values().index.tolist()
    sorted_methods = sanity_methods + sorted_ranked_methods
    piv = piv.loc[sorted_methods]

    best_by_col = {}
    second_by_col = {}
    for col in piv.columns:
        vals = sorted(piv.loc[ranked_methods, col].dropna().unique(), reverse=True)
        best_by_col[col] = vals[0] if vals else None
        second_by_col[col] = vals[1] if len(vals) > 1 else None

    rank_vals = sorted(avg_rank.dropna().unique())
    best_rank = rank_vals[0] if rank_vals else None
    second_rank = rank_vals[1] if len(rank_vals) > 1 else None

    lines = []
    for method in sorted_methods:
        cells = [style_cell(float(piv.loc[method, col]), best_by_col[col], second_by_col[col]) for col in piv.columns]
        if method in NO_RANK_METHODS:
            rank_text = "-"
        else:
            rank_text = style_rank(float(avg_rank.loc[method]), best_rank, second_rank)
        lines.append(f"{method} & " + " & ".join(cells) + f" & {rank_text} \\\\")

    spec = METRIC_SPECS[metric]
    table = [
        "\\begin{table}[h]",
        "\\centering",
        f"\\caption{{{spec['caption']}}}",
        f"\\label{{{spec['label']}}}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{l|ccccccc|c}",
        "\\toprule",
        "\\textbf{Method} & \\textbf{M3-M} & \\textbf{M3-Q} & \\textbf{T-M} & \\textbf{T-Q} & \\textbf{M1-M} & \\textbf{M1-Q} & \\textbf{NN3-M} & \\textbf{Avg Rank} \\\\ \\midrule",
        *lines,
        "\\bottomrule",
        "\\end{tabular}%",
        "}",
        "\\end{table}",
    ]
    return "\n".join(table)


def main() -> None:
    base = pd.read_csv(BASE_PRIVACY_PATH)
    opt = pd.read_csv(OPT_GRASYNDA_PRIVACY_PATH)
    manifest = pd.read_csv(MANIFEST_PATH)

    base = base[base["Family"] != "Grasynda"].copy()
    opt = opt.copy()
    opt["Family"] = "Grasynda"
    opt["Method"] = "Grasynda_Optimized"
    opt["Variant_Name"] = "Grasynda_Optimized"

    sanity = build_sanity_rows(manifest)

    extended = pd.concat([base, opt, sanity], ignore_index=True, sort=False)
    extended = extended[extended.apply(is_allowed_row, axis=1)].copy()
    sort_cols = ["Family", "Method", "Variant_Name", "Dataset", "Group", "Epsilon", "Seed"]
    extended = extended.sort_values(sort_cols, na_position="last").reset_index(drop=True)

    detailed_out = os.path.join(OUT_DIR, "privacy_results_detailed_with_sanity.csv")
    extended.to_csv(detailed_out, index=False)

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
        extended.groupby(["Family", "Method", "Variant_Name", "Epsilon"], dropna=False)[summary_cols]
        .mean()
        .reset_index()
    )
    summary_out = os.path.join(OUT_DIR, "privacy_results_summary_with_sanity.csv")
    summary.to_csv(summary_out, index=False)

    table_df = extended[extended.apply(is_display_row, axis=1)].copy()
    table_detailed_out = os.path.join(OUT_DIR, "privacy_results_detailed_table_only.csv")
    table_df.to_csv(table_detailed_out, index=False)

    tex_parts = []
    for metric in ["DCR_Mean", "NNDR_Mean", "PyMDMA_Privacy"]:
        table_tex = build_latex_table(extended, metric)
        tex_parts.append(table_tex)

    tex_out = os.path.join(OUT_DIR, "nhits_privacy_tables_sorted_with_sanity.tex")
    with open(tex_out, "w", encoding="utf-8") as handle:
        handle.write("\n\n".join(tex_parts) + "\n")

    print(f"Detailed: {detailed_out}")
    print(f"Summary:  {summary_out}")
    print(f"Table:    {table_detailed_out}")
    print(f"LaTeX:    {tex_out}")


if __name__ == "__main__":
    main()
