from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = PROJECT_ROOT / "assets/results/pymdma_metrics/corrected"
FORECAST_OUT_CSV = OUT_DIR / "nhits_tstr_ranks_no_sax_recomputed.csv"
PRIVACY_OUT_CSV = OUT_DIR / "privacy_ranks_no_sax_recomputed.csv"
SCATTER_OUT_CSV = OUT_DIR / "nhits_privacy_vs_forecast_no_sax_scatter.csv"
SCATTER_OUT_PNG = OUT_DIR / "nhits_privacy_vs_forecast_no_sax_scatter.png"
SCATTER_OUT_PDF = OUT_DIR / "nhits_privacy_vs_forecast_no_sax_scatter.pdf"

DATASET_COLUMNS = ["M3 M", "M3 Q", "Tour M", "Tour Q", "m1 M", "m1 Q", "NN3 M"]


def build_forecast_table() -> pd.DataFrame:
    df = pd.DataFrame(
        [
            ("Baseline", 0.725, 1.212, 1.192, 1.226, 0.978, 1.361, 1.020),
            ("Hybrid_Q10_Ensemble5_Continuous", 0.726, 1.083, 1.191, 1.238, 1.192, 1.198, 1.271),
            ("Hybrid_Q10_NoEnsemble_Continuous", 0.726, 1.091, 1.184, 1.233, 1.162, 1.237, 1.242),
            ("Jittering", 0.729, 1.198, 1.193, 1.256, 1.048, 1.199, 1.068),
            ("DBA", 0.734, 1.268, 1.203, 1.250, 1.028, 1.350, 1.209),
            ("SeasonalMBB", 0.734, 1.241, 1.211, 1.268, 1.058, 1.439, 1.168),
            ("Scaling", 0.808, 1.215, 1.211, 1.265, 1.076, 1.848, 1.190),
            ("MagnitudeWarping", 0.768, 1.383, 1.192, 1.553, 1.264, 1.285, 1.335),
            ("TSMixup", 0.750, 1.207, 1.215, 1.325, 1.138, 2.230, 1.336),
            ("TimeVAE", 0.992, 1.171, 1.446, 1.283, 1.484, 1.323, 1.324),
            ("TimeWarping", 0.857, 1.827, 2.209, 2.491, 1.228, 1.245, 1.997),
        ],
        columns=["Method"] + DATASET_COLUMNS,
    )
    for column in DATASET_COLUMNS:
        df[f"{column} Rank"] = df[column].rank(method="min", ascending=True)
    df["Avg Rank"] = df[[f"{column} Rank" for column in DATASET_COLUMNS]].mean(axis=1)
    return df.sort_values(["Avg Rank", "Method"]).reset_index(drop=True)


def build_privacy_table() -> pd.DataFrame:
    df = pd.DataFrame(
        [
            ("Hybrid_Q10_NoEnsemble_Continuous", 0.764, 0.937, 0.904, 0.532, 0.764, 0.937, 0.817),
            ("TSMixup", 0.808, 0.799, 0.758, 0.611, 0.789, 0.859, 0.762),
            ("Hybrid_Q10_Ensemble5_Continuous", 0.693, 0.841, 0.890, 0.605, 0.693, 0.841, 0.906),
            ("TimeWarping", 0.788, 0.760, 0.896, 0.414, 0.725, 0.756, 0.761),
            ("TimeVAE", 0.665, 0.614, 0.624, 0.309, 0.704, 0.677, 0.474),
            ("DBA", 0.579, 0.530, 0.726, 0.560, 0.574, 0.352, 0.285),
            ("MagnitudeWarping", 0.025, 0.591, 0.012, 0.042, 0.479, 0.694, 0.741),
            ("SeasonalMBB", 0.711, 0.548, 0.469, 0.034, 0.434, 0.588, 0.648),
            ("Jittering", 0.005, 0.000, 0.012, 0.003, 0.001, 0.013, 0.000),
            # Include Exact_Copy as Baseline for the scatter/rank pool.
            ("Baseline", 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000),
        ],
        columns=["Method"] + DATASET_COLUMNS,
    )
    for column in DATASET_COLUMNS:
        df[f"{column} Rank"] = df[column].rank(method="min", ascending=False)
    df["Avg Rank"] = df[[f"{column} Rank" for column in DATASET_COLUMNS]].mean(axis=1)
    return df.sort_values(["Avg Rank", "Method"]).reset_index(drop=True)


def method_family(method_name: str) -> str:
    if method_name == "Baseline":
        return "Baseline"
    if method_name.startswith("Hybrid_Q10_"):
        return "Grasynda Q10"
    return "Other"


def method_color(method_name: str) -> str:
    family = method_family(method_name)
    if family == "Baseline":
        return "#111111"
    if family == "Grasynda Q10":
        return "#d62828"
    return "#277da1"


def pretty_method_name(method_name: str) -> str:
    if method_name == "Baseline":
        return "Original"
    return (
        method_name.replace("Hybrid_", "")
        .replace("_Continuous", "")
        .replace("_", " ")
    )


def build_scatter_df(forecast_df: pd.DataFrame, privacy_df: pd.DataFrame) -> pd.DataFrame:
    merged = forecast_df[["Method", "Avg Rank"]].merge(
        privacy_df[["Method", "Avg Rank"]],
        on="Method",
        suffixes=("_Forecast", "_Privacy"),
    )
    merged["Family"] = merged["Method"].map(method_family)
    merged["Color"] = merged["Method"].map(method_color)
    merged["Pretty_Method"] = merged["Method"].map(pretty_method_name)
    return merged.sort_values(["Avg Rank_Forecast", "Avg Rank_Privacy", "Method"]).reset_index(drop=True)


def plot_scatter(scatter_df: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(11, 8))

    ax.scatter(
        scatter_df["Avg Rank_Forecast"],
        scatter_df["Avg Rank_Privacy"],
        c=scatter_df["Color"],
        s=180,
        edgecolors="white",
        linewidth=1.1,
        zorder=5,
    )

    for _, row in scatter_df.iterrows():
        ax.annotate(
            row["Pretty_Method"],
            (row["Avg Rank_Forecast"], row["Avg Rank_Privacy"]),
            textcoords="offset points",
            xytext=(7, 6),
            fontsize=9,
        )

    ax.set_xlabel("Average TSTR Forecast Rank (NHITS)\nlower is better")
    ax.set_ylabel("Average Privacy Rank\nlower is better")
    ax.set_title("NHITS: Privacy vs TSTR Forecast Rank\n(No SAX variants, ranks recomputed)")
    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.grid(True, alpha=0.25)
    ax.set_axisbelow(True)

    legend_items = [
        Line2D([0], [0], marker="o", color="w", label="Baseline", markerfacecolor="#111111", markersize=9),
        Line2D([0], [0], marker="o", color="w", label="Grasynda Q10", markerfacecolor="#d62828", markersize=9),
        Line2D([0], [0], marker="o", color="w", label="Other augmentations", markerfacecolor="#277da1", markersize=9),
    ]
    ax.legend(handles=legend_items, loc="upper left", frameon=True, fontsize=9)

    plt.tight_layout()
    plt.savefig(SCATTER_OUT_PNG, dpi=180, bbox_inches="tight")
    plt.savefig(SCATTER_OUT_PDF, bbox_inches="tight")
    plt.close()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    forecast_df = build_forecast_table()
    privacy_df = build_privacy_table()
    scatter_df = build_scatter_df(forecast_df, privacy_df)

    forecast_df.to_csv(FORECAST_OUT_CSV, index=False)
    privacy_df.to_csv(PRIVACY_OUT_CSV, index=False)
    scatter_df.to_csv(SCATTER_OUT_CSV, index=False)
    plot_scatter(scatter_df)

    print(f"Saved forecast table: {FORECAST_OUT_CSV}")
    print(f"Saved privacy table: {PRIVACY_OUT_CSV}")
    print(f"Saved scatter data: {SCATTER_OUT_CSV}")
    print(f"Saved scatter plot: {SCATTER_OUT_PNG}")
    print(f"Saved scatter plot: {SCATTER_OUT_PDF}")


if __name__ == "__main__":
    main()
