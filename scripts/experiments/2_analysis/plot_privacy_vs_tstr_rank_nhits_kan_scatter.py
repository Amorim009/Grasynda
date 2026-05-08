from pathlib import Path
from datetime import datetime

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[3]

FORECAST_PATH = PROJECT_ROOT / "assets/results/systematic_evaluation_all_datasets_all_methods_no_tsdiff_min60_nhits_mlp_kan_20260328.csv"
PRIVACY_SOURCES = [
    (
        "Scaling corrected",
        PROJECT_ROOT / "assets/results/pymdma_metrics/corrected/final_results_scaling_default.csv",
    ),
    (
        "Grasynda Q10 corrected",
        PROJECT_ROOT / "assets/results/pymdma_metrics/corrected/final_results_q10_all_datasets.csv",
    ),
    (
        "Comprehensive PyMDMA",
        PROJECT_ROOT / "assets/results/pymdma_metrics/MOST_RECENT_PYMDMA/final_comprehensive_pymdma_results.csv",
    ),
]

OUT_DIR = PROJECT_ROOT / "assets/results/pymdma_metrics/corrected"
OUT_SUMMARY_CSV = OUT_DIR / "privacy_vs_tstr_rank_nhits_kan_scatter.csv"
OUT_DETAIL_CSV = OUT_DIR / "privacy_vs_tstr_rank_nhits_kan_scatter_detail.csv"
OUT_AUDIT_CSV = OUT_DIR / "privacy_vs_tstr_rank_nhits_kan_scatter_audit.csv"
OUT_PNG = OUT_DIR / "privacy_vs_tstr_rank_nhits_kan_scatter.png"
OUT_PDF = OUT_DIR / "privacy_vs_tstr_rank_nhits_kan_scatter.pdf"

MODELS = ["NHITS", "KAN"]
MODEL_MARKERS = {"NHITS": "o", "KAN": "^"}
MODEL_COLORS = {"NHITS": "#1f77b4", "KAN": "#ff7f0e"}
EXPECTED_DATASET_GROUPS = 7
EXCLUDED_METHODS = {"Hybrid_Q10_Ensemble5_Continuous"}


def ensure_inputs() -> None:
    if not FORECAST_PATH.exists():
        raise FileNotFoundError(f"Missing forecast CSV: {FORECAST_PATH}")
    if not any(path.exists() for _, path in PRIVACY_SOURCES):
        raise FileNotFoundError("No privacy CSVs found in expected locations.")
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_privacy_results() -> pd.DataFrame:
    frames = []
    for priority, (source_name, path) in enumerate(PRIVACY_SOURCES):
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if not {"Dataset", "Group", "Method", "Privacy"}.issubset(df.columns):
            continue
        slim = df[["Dataset", "Group", "Method", "Privacy"]].copy()
        slim["Privacy_Source"] = source_name
        slim["Privacy_Source_Priority"] = priority
        frames.append(slim)

    privacy_df = pd.concat(frames, ignore_index=True)
    privacy_df["Method"] = privacy_df["Method"].replace({"Exact_Copy": "Baseline"})
    privacy_df = privacy_df[privacy_df["Method"] != "Random_Noise"].copy()
    privacy_df = privacy_df[~privacy_df["Method"].str.startswith("Hybrid_SAX_", na=False)].copy()
    privacy_df = privacy_df[~privacy_df["Method"].str.contains("RawY", na=False)].copy()
    privacy_df = privacy_df.sort_values(
        ["Dataset", "Group", "Method", "Privacy_Source_Priority"]
    ).drop_duplicates(["Dataset", "Group", "Method"], keep="first")
    return privacy_df


def method_family(method_name: str) -> str:
    if method_name == "Baseline":
        return "Baseline"
    if method_name.startswith("Hybrid_Q10_"):
        return "Grasynda Q10"
    return "Other"


def family_color(method_name: str) -> str:
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
        .replace("_Optimized", " Opt")
        .replace("_", " ")
    )


def build_outputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    forecast_df = pd.read_csv(FORECAST_PATH)
    forecast_df = forecast_df[
        (forecast_df["Forecasting_Model"].isin(MODELS))
        & (forecast_df["Training_Mode"].isin(["TSTR", "Baseline"]))
        & (forecast_df["Status"] == "Success")
    ].copy()
    forecast_df = forecast_df[~forecast_df["Augmentation_Method"].str.startswith("Hybrid_SAX_", na=False)].copy()
    forecast_df = forecast_df[~forecast_df["Augmentation_Method"].str.contains("RawY", na=False)].copy()
    forecast_df = forecast_df[~forecast_df["Augmentation_Method"].isin(EXCLUDED_METHODS)].copy()

    privacy_df = load_privacy_results()
    privacy_df = privacy_df[~privacy_df["Method"].isin(EXCLUDED_METHODS)].copy()

    overlap_methods = sorted(set(forecast_df["Augmentation_Method"].unique()) & set(privacy_df["Method"].unique()))
    forecast_df = forecast_df[forecast_df["Augmentation_Method"].isin(overlap_methods)].copy()
    privacy_df = privacy_df[privacy_df["Method"].isin(overlap_methods)].copy()

    forecast_rank_parts = []
    for (dataset, group, model), slice_df in forecast_df.groupby(["Dataset", "Group", "Forecasting_Model"]):
        ranked = slice_df[["Augmentation_Method", "MASE", "Training_Mode"]].copy()
        ranked["Forecast_Rank"] = ranked["MASE"].rank(ascending=True, method="min")
        ranked["Dataset"] = dataset
        ranked["Group"] = group
        ranked["Forecasting_Model"] = model
        forecast_rank_parts.append(ranked)

    forecast_rank_df = pd.concat(forecast_rank_parts, ignore_index=True)

    privacy_rank_parts = []
    for (dataset, group), slice_df in privacy_df.groupby(["Dataset", "Group"]):
        ranked = slice_df[["Method", "Privacy"]].copy()
        ranked["Privacy_Rank"] = ranked["Privacy"].rank(ascending=False, method="min")
        ranked["Dataset"] = dataset
        ranked["Group"] = group
        privacy_rank_parts.append(ranked)

    privacy_rank_df = pd.concat(privacy_rank_parts, ignore_index=True)

    detail_df = forecast_rank_df.merge(
        privacy_rank_df,
        left_on=["Dataset", "Group", "Augmentation_Method"],
        right_on=["Dataset", "Group", "Method"],
        how="inner",
    ).drop(columns=["Method"])
    detail_df = detail_df.merge(
        privacy_df[["Dataset", "Group", "Method", "Privacy_Source"]],
        left_on=["Dataset", "Group", "Augmentation_Method"],
        right_on=["Dataset", "Group", "Method"],
        how="left",
    ).drop(columns=["Method"])
    detail_df["Family"] = detail_df["Augmentation_Method"].map(method_family)
    detail_df["Color"] = detail_df["Augmentation_Method"].map(family_color)
    detail_df["Pretty_Method"] = detail_df["Augmentation_Method"].map(pretty_method_name)
    detail_df["Dataset_Group"] = detail_df["Dataset"] + " / " + detail_df["Group"]
    detail_df = detail_df.sort_values(
        ["Forecasting_Model", "Augmentation_Method", "Dataset", "Group"]
    ).reset_index(drop=True)

    summary_df = (
        detail_df.groupby(["Forecasting_Model", "Augmentation_Method", "Pretty_Method", "Family", "Color"], as_index=False)
        .agg(
            Avg_TSTR_Forecast_Rank=("Forecast_Rank", "mean"),
            Avg_Privacy_Rank=("Privacy_Rank", "mean"),
            Dataset_Groups=("Dataset_Group", "nunique"),
        )
        .sort_values(["Augmentation_Method", "Forecasting_Model"])
        .reset_index(drop=True)
    )

    source_df = (
        detail_df.groupby(["Augmentation_Method", "Pretty_Method", "Privacy_Source"], as_index=False)
        .agg(
            Dataset_Groups=("Dataset_Group", "nunique"),
            Models_Covered=("Forecasting_Model", lambda s: ", ".join(sorted(set(s)))),
        )
        .sort_values(["Augmentation_Method", "Privacy_Source"])
        .reset_index(drop=True)
    )

    if detail_df.empty or summary_df.empty:
        raise ValueError("No overlapping methods remained after matching forecast and privacy results.")

    bad_coverage = summary_df[summary_df["Dataset_Groups"] != EXPECTED_DATASET_GROUPS]
    if not bad_coverage.empty:
        raise ValueError(
            "Unexpected dataset-group coverage for some plotted methods:\n"
            f"{bad_coverage[['Forecasting_Model', 'Augmentation_Method', 'Dataset_Groups']].to_string(index=False)}"
        )

    return detail_df, summary_df, source_df


def plot_summary(summary_df: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(11.6, 8.5))

    x_min = float(summary_df["Avg_TSTR_Forecast_Rank"].min()) - 0.35
    x_max = float(summary_df["Avg_TSTR_Forecast_Rank"].max()) + 0.35
    y_min = float(summary_df["Avg_Privacy_Rank"].min()) - 0.35
    y_max = float(summary_df["Avg_Privacy_Rank"].max()) + 0.35

    for model_name in MODELS:
        model_df = summary_df[summary_df["Forecasting_Model"] == model_name].copy()
        ax.scatter(
            model_df["Avg_TSTR_Forecast_Rank"],
            model_df["Avg_Privacy_Rank"],
            c=MODEL_COLORS[model_name],
            marker=MODEL_MARKERS[model_name],
            s=275,
            edgecolors="white",
            linewidth=1.6,
            alpha=0.96,
            zorder=5,
            label=model_name,
        )
        model_df = model_df.sort_values(["Avg_TSTR_Forecast_Rank", "Avg_Privacy_Rank"]).reset_index(drop=True)
        for idx, (_, row) in enumerate(model_df.iterrows()):
            if model_name == "NHITS":
                x_offset = -10
                y_offset = -12 if idx % 2 == 0 else 6
                ha = "right"
            else:
                x_offset = 10
                y_offset = 6 if idx % 2 == 0 else -12
                ha = "left"
            ax.annotate(
                row["Pretty_Method"],
                (row["Avg_TSTR_Forecast_Rank"], row["Avg_Privacy_Rank"]),
                textcoords="offset points",
                xytext=(x_offset, y_offset),
                ha=ha,
                fontsize=10.2,
                color=MODEL_COLORS[model_name],
                bbox={
                    "boxstyle": "round,pad=0.22",
                    "facecolor": "white",
                    "edgecolor": MODEL_COLORS[model_name],
                    "alpha": 0.82,
                    "linewidth": 0.7,
                },
            )

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.set_xlabel("Average TSTR Forecast Rank\n(lower is better)", fontsize=12.5)
    ax.set_ylabel("Average Privacy Rank\n(lower is better)", fontsize=12.5)
    ax.set_title(
        "Privacy vs TSTR Forecast Rank for NHITS and KAN\n"
        "(Baseline included, ensemble/SAX/RawY excluded)",
        fontsize=16.5,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.25)
    ax.set_axisbelow(True)

    legend_items = [
        Line2D([0], [0], marker="o", color="w", label="NHITS", markerfacecolor=MODEL_COLORS["NHITS"], markersize=11),
        Line2D([0], [0], marker="^", color="w", label="KAN", markerfacecolor=MODEL_COLORS["KAN"], markersize=11),
    ]
    ax.legend(handles=legend_items, loc="upper left", frameon=True, fontsize=11, title="Forecast model")

    caption = (
        "Caption: Forecasting ranks come from the shared TSTR benchmark under the same "
        "forecasting conditions: same dataset-group splits, same min_len=60 filtering, and "
        "the same model-specific setup within NHITS or within KAN."
    )
    fig.text(0.02, 0.02, caption, ha="left", va="bottom", fontsize=9.3, wrap=True)

    plt.tight_layout(rect=(0, 0.08, 1, 1))
    save_figure(fig)
    plt.close()


def save_figure(fig) -> None:
    fig.savefig(OUT_PNG, dpi=180, bbox_inches="tight")
    try:
        fig.savefig(OUT_PDF, bbox_inches="tight")
    except PermissionError:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fallback_pdf = OUT_PDF.with_name(f"{OUT_PDF.stem}_{timestamp}{OUT_PDF.suffix}")
        fallback_png = OUT_PNG.with_name(f"{OUT_PNG.stem}_{timestamp}{OUT_PNG.suffix}")
        fig.savefig(fallback_png, dpi=180, bbox_inches="tight")
        fig.savefig(fallback_pdf, bbox_inches="tight")
        print(f"Primary PDF locked; saved fallback PNG: {fallback_png}")
        print(f"Primary PDF locked; saved fallback PDF: {fallback_pdf}")


def main() -> None:
    ensure_inputs()
    detail_df, summary_df, source_df = build_outputs()
    detail_df.to_csv(OUT_DETAIL_CSV, index=False)
    summary_df.to_csv(OUT_SUMMARY_CSV, index=False)
    source_df.to_csv(OUT_AUDIT_CSV, index=False)
    plot_summary(summary_df)
    print(f"Saved detail CSV: {OUT_DETAIL_CSV}")
    print(f"Saved summary CSV: {OUT_SUMMARY_CSV}")
    print(f"Saved audit CSV: {OUT_AUDIT_CSV}")
    print(f"Saved plot: {OUT_PNG}")
    print(f"Saved plot: {OUT_PDF}")
    print("\nMethods plotted per model:")
    print(summary_df.groupby('Forecasting_Model').size().to_string())


if __name__ == "__main__":
    main()
