from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PRIVACY_NODIFF_PATH = PROJECT_ROOT / "assets/results/pymdma_metrics/corrected/final_results_nodiff_sanity.csv"
PRIVACY_DIFF_PATH = PROJECT_ROOT / "assets/results/pymdma_metrics/corrected/final_results_q10_all_datasets.csv"
OUT_DIR = PROJECT_ROOT / "assets/results/pymdma_metrics/corrected"
OUT_CSV = OUT_DIR / "grasynda_nodiff_vs_diff_privacy_comparison.csv"
OUT_PNG = OUT_DIR / "grasynda_nodiff_vs_diff_privacy_scatter.png"
OUT_PDF = OUT_DIR / "grasynda_nodiff_vs_diff_privacy_scatter.pdf"

NODIFF_METHOD = "Hybrid_Q10_NoEnsemble_NoDiff"
DIFF_METHODS = [
    "Hybrid_Q10_NoEnsemble_Continuous",
    "Hybrid_Q10_Ensemble5_Continuous",
]
VARIANT_LABELS = {
    "Hybrid_Q10_NoEnsemble_Continuous": "Diff Trend | No Ensemble",
    "Hybrid_Q10_Ensemble5_Continuous": "Diff Trend | Ensemble5",
}
VARIANT_COLORS = {
    "Hybrid_Q10_NoEnsemble_Continuous": "#d62828",
    "Hybrid_Q10_Ensemble5_Continuous": "#1d3557",
}


def ensure_inputs() -> None:
    for path in [PRIVACY_NODIFF_PATH, PRIVACY_DIFF_PATH]:
        if not path.exists():
            raise FileNotFoundError(f"Missing input CSV: {path}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def dataset_group_label(dataset: str, group: str) -> str:
    return f"{dataset} / {group}"


def load_comparison() -> pd.DataFrame:
    nodiff = pd.read_csv(PRIVACY_NODIFF_PATH)
    diff = pd.read_csv(PRIVACY_DIFF_PATH)

    nodiff = nodiff[nodiff["Method"] == NODIFF_METHOD].copy()
    diff = diff[diff["Method"].isin(DIFF_METHODS)].copy()

    merged = nodiff.merge(
        diff,
        on=["Dataset", "Group"],
        suffixes=("_nodiff", "_diff"),
        how="inner",
    )
    merged["dataset_group"] = merged.apply(lambda row: dataset_group_label(row["Dataset"], row["Group"]), axis=1)
    merged["variant_label"] = merged["Method_diff"].map(VARIANT_LABELS)
    merged["privacy_delta_diff_minus_nodiff"] = merged["Privacy_diff"] - merged["Privacy_nodiff"]
    merged["authenticity_delta_diff_minus_nodiff"] = (
        merged["Authenticity_diff"] - merged["Authenticity_nodiff"]
    )
    merged["fidelity_delta_diff_minus_nodiff"] = merged["Fidelity_diff"] - merged["Fidelity_nodiff"]
    merged["diversity_delta_diff_minus_nodiff"] = merged["Diversity_diff"] - merged["Diversity_nodiff"]
    return merged[
        [
            "Dataset",
            "Group",
            "dataset_group",
            "Method_nodiff",
            "Method_diff",
            "variant_label",
            "Privacy_nodiff",
            "Privacy_diff",
            "privacy_delta_diff_minus_nodiff",
            "Authenticity_nodiff",
            "Authenticity_diff",
            "authenticity_delta_diff_minus_nodiff",
            "Fidelity_nodiff",
            "Fidelity_diff",
            "fidelity_delta_diff_minus_nodiff",
            "Diversity_nodiff",
            "Diversity_diff",
            "diversity_delta_diff_minus_nodiff",
        ]
    ].sort_values(["Method_diff", "Dataset", "Group"]).reset_index(drop=True)


def build_average_points(df: pd.DataFrame) -> pd.DataFrame:
    avg = (
        df.groupby(["Method_diff", "variant_label"], as_index=False)
        .agg(
            Privacy_nodiff=("Privacy_nodiff", "mean"),
            Privacy_diff=("Privacy_diff", "mean"),
            privacy_delta_diff_minus_nodiff=("privacy_delta_diff_minus_nodiff", "mean"),
        )
        .copy()
    )
    avg["dataset_group"] = "Average"
    avg["Dataset"] = "All"
    avg["Group"] = "All"
    return avg


def plot_scatter(df: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(11, 8))

    all_vals = pd.concat([df["Privacy_nodiff"], df["Privacy_diff"]], ignore_index=True)
    min_val = max(0.0, float(all_vals.min()) - 0.05)
    max_val = min(1.0, float(all_vals.max()) + 0.05)

    ax.plot([min_val, max_val], [min_val, max_val], linestyle="--", linewidth=1.5, color="#8d99ae", zorder=1)

    for method_name in DIFF_METHODS:
        slice_df = df[df["Method_diff"] == method_name].copy()
        ax.scatter(
            slice_df["Privacy_nodiff"],
            slice_df["Privacy_diff"],
            s=150,
            color=VARIANT_COLORS[method_name],
            edgecolors="white",
            linewidth=1.0,
            alpha=0.9,
            label=VARIANT_LABELS[method_name],
            zorder=3,
        )
        for _, row in slice_df.iterrows():
            ax.annotate(
                row["dataset_group"],
                (row["Privacy_nodiff"], row["Privacy_diff"]),
                textcoords="offset points",
                xytext=(7, 6),
                fontsize=9,
            )

    avg_df = build_average_points(df)
    for _, row in avg_df.iterrows():
        ax.scatter(
            row["Privacy_nodiff"],
            row["Privacy_diff"],
            s=320,
            marker="D",
            color=VARIANT_COLORS[row["Method_diff"]],
            edgecolors="black",
            linewidth=1.0,
            zorder=5,
        )
        ax.annotate(
            f"Average: {row['variant_label']}",
            (row["Privacy_nodiff"], row["Privacy_diff"]),
            textcoords="offset points",
            xytext=(10, -14),
            fontsize=10,
            fontweight="bold",
        )

    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.set_xlabel("Privacy Score: NoDiff")
    ax.set_ylabel("Privacy Score: Differentiated Trend Variant")
    ax.set_title("Grasynda Privacy: NoDiff vs Differentiated Trend Variants")
    ax.text(
        0.03,
        0.97,
        "Above diagonal = differentiation improved privacy",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        color="#555555",
    )
    ax.legend(loc="lower right", frameon=True, fontsize=10)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=180, bbox_inches="tight")
    plt.savefig(OUT_PDF, bbox_inches="tight")
    plt.close()


def main() -> None:
    ensure_inputs()
    comparison = load_comparison()
    comparison.to_csv(OUT_CSV, index=False)
    plot_scatter(comparison)
    print(f"Saved comparison data: {OUT_CSV}")
    print(f"Saved plot: {OUT_PNG}")
    print(f"Saved plot: {OUT_PDF}")


if __name__ == "__main__":
    main()
