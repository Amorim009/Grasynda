"""
Plot average privacy rank vs average forecasting rank using NHITS, MLP, and KAN.

- X-axis: average TSTR forecasting rank across datasets and forecasting models
- Y-axis: average privacy rank across datasets

Forecasting ranks include:
- Baseline rows from Training_Mode == Baseline
- Augmented rows from Training_Mode == TSTR

Forecast ranks are computed within each dataset across all included methods
(Baseline + TSTR together).

Privacy ranks include Baseline by mapping Exact_Copy -> Baseline.
RawY Grasynda variants are excluded.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

FORECAST_PATH = PROJECT_ROOT / (
    "assets/results/archive/"
    "all_datasets_all_methods_no_tsdiff_min60_nhits_mlp_kan_20260402/"
    "full_results_all_datasets_all_methods_no_tsdiff_min60_nhits_mlp_kan_20260402.csv"
)

PRIVACY_PATHS = [
    PROJECT_ROOT / "assets/results/pymdma_metrics/corrected/final_results_q10_all_datasets.csv",
    PROJECT_ROOT / "assets/results/pymdma_metrics/corrected/final_results.csv",
    PROJECT_ROOT / "assets/results/pymdma_metrics/MOST_RECENT_PYMDMA/final_comprehensive_pymdma_results.csv",
]

OUT_DIR = PROJECT_ROOT / "assets/results/pymdma_metrics/corrected"
OUT_PLOT = OUT_DIR / "privacy_vs_tstr_forecast_rank_avg_all_models_no_rawy.png"
OUT_DATA = OUT_DIR / "privacy_vs_tstr_forecast_rank_avg_all_models_no_rawy.csv"


def load_privacy_results() -> pd.DataFrame:
    frames = []
    for path in PRIVACY_PATHS:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if not {"Dataset", "Group", "Method", "Privacy"}.issubset(df.columns):
            continue
        df = df[["Dataset", "Group", "Method", "Privacy"]].copy()
        df["Source"] = path.name
        frames.append(df)

    privacy_df = pd.concat(frames, ignore_index=True)
    privacy_df = privacy_df.drop_duplicates(["Dataset", "Group", "Method"], keep="first")
    privacy_df["Method"] = privacy_df["Method"].replace({"Exact_Copy": "Baseline"})
    privacy_df = privacy_df[privacy_df["Method"] != "Random_Noise"]
    privacy_df = privacy_df[~privacy_df["Method"].str.contains("RawY", na=False)]
    return privacy_df


def method_family(method_name: str) -> str:
    if method_name == "Baseline":
        return "Baseline"
    if method_name.startswith("Hybrid_Q10_"):
        return "Grasynda Q10"
    if method_name.startswith("Hybrid_SAX_"):
        return "Grasynda SAX"
    return "Other"


def family_color(method_name: str) -> str:
    family = method_family(method_name)
    if family == "Baseline":
        return "#111111"
    if family == "Grasynda Q10":
        return "#d62828"
    if family == "Grasynda SAX":
        return "#f77f00"
    return "#277da1"


def dataset_key(dataset: str, group: str) -> str:
    return f"{dataset} | {group}"


forecast_df = pd.read_csv(FORECAST_PATH)
forecast_df = forecast_df[forecast_df["Forecasting_Model"].isin(["NHITS", "MLP", "KAN"])].copy()
forecast_df = forecast_df[forecast_df["Training_Mode"].isin(["TSTR", "Baseline"])].copy()
forecast_df = forecast_df[~forecast_df["Augmentation_Method"].str.contains("RawY", na=False)].copy()
privacy_df = load_privacy_results()

merged = forecast_df.merge(
    privacy_df,
    left_on=["Dataset", "Group", "Augmentation_Method"],
    right_on=["Dataset", "Group", "Method"],
    how="inner",
)

forecast_rank_parts = []
for keys, slice_df in merged.groupby(["Dataset", "Group", "Forecasting_Model"]):
    ranked = slice_df[["Augmentation_Method", "MASE"]].copy()
    ranked["Forecast_Rank"] = ranked["MASE"].rank(ascending=True, method="min")
    ranked["Dataset"] = keys[0]
    ranked["Group"] = keys[1]
    ranked["Forecasting_Model"] = keys[2]
    forecast_rank_parts.append(ranked)

forecast_rank_df = pd.concat(forecast_rank_parts, ignore_index=True)
forecast_rank_df = (
    forecast_rank_df.groupby(["Dataset", "Group", "Augmentation_Method"], as_index=False)
    .agg(
        Forecast_Rank=("Forecast_Rank", "mean"),
        Model_Count=("Forecasting_Model", "nunique"),
    )
)

privacy_rank_parts = []
for keys, slice_df in privacy_df.groupby(["Dataset", "Group"]):
    ranked = slice_df[["Method", "Privacy"]].copy()
    ranked["Privacy_Rank"] = ranked["Privacy"].rank(ascending=False, method="min")
    ranked["Dataset"] = keys[0]
    ranked["Group"] = keys[1]
    privacy_rank_parts.append(ranked)

privacy_rank_df = pd.concat(privacy_rank_parts, ignore_index=True)

plot_df = forecast_rank_df.merge(
    privacy_rank_df,
    left_on=["Dataset", "Group", "Augmentation_Method"],
    right_on=["Dataset", "Group", "Method"],
    how="inner",
)
plot_df = plot_df.drop(columns=["Method"])

plot_df["Dataset_Key"] = plot_df.apply(lambda r: dataset_key(r["Dataset"], r["Group"]), axis=1)
plot_df["Family"] = plot_df["Augmentation_Method"].map(method_family)
plot_df["Color"] = plot_df["Augmentation_Method"].map(family_color)
plot_df = plot_df.rename(columns={"Augmentation_Method": "Method"})
plot_df = plot_df[
    ["Dataset", "Group", "Dataset_Key", "Method", "Family", "Color", "Forecast_Rank", "Privacy_Rank", "Model_Count"]
].sort_values(["Method", "Dataset", "Group"])

avg_df = (
    plot_df.groupby(["Method", "Family", "Color"], as_index=False)
    .agg(
        Avg_Forecast_Rank=("Forecast_Rank", "mean"),
        Avg_Privacy_Rank=("Privacy_Rank", "mean"),
        Dataset_Count=("Dataset_Key", "nunique"),
        Avg_Model_Count=("Model_Count", "mean"),
    )
    .sort_values(["Family", "Avg_Forecast_Rank", "Avg_Privacy_Rank"])
)

fig, ax = plt.subplots(figsize=(12, 8))

non_baseline_df = avg_df[avg_df["Method"] != "Baseline"]
baseline_df = avg_df[avg_df["Method"] == "Baseline"]

ax.scatter(
    non_baseline_df["Avg_Forecast_Rank"],
    non_baseline_df["Avg_Privacy_Rank"],
    c=non_baseline_df["Color"],
    s=170,
    edgecolors="white",
    linewidth=1.1,
    alpha=0.95,
    zorder=5,
)

if not baseline_df.empty:
    ax.scatter(
        baseline_df["Avg_Forecast_Rank"],
        baseline_df["Avg_Privacy_Rank"],
        c=baseline_df["Color"],
        marker="D",
        s=220,
        edgecolors="white",
        linewidth=1.2,
        alpha=1.0,
        zorder=6,
    )

for _, row in avg_df.iterrows():
    ax.annotate(
        row["Method"].replace("Hybrid_", "").replace("_Continuous", "").replace("_", " "),
        (row["Avg_Forecast_Rank"], row["Avg_Privacy_Rank"]),
        textcoords="offset points",
        xytext=(8, 5),
        fontsize=8.5,
    )

ax.set_xlabel(
    "Average TSTR Forecast Rank",
    fontsize=11,
)
ax.set_ylabel(
    "Average Privacy Rank",
    fontsize=11,
)
ax.set_title(
    "Privacy vs TSTR Forecast Rank",
    fontsize=14,
    fontweight="bold",
)

ax.invert_xaxis()
ax.invert_yaxis()
ax.grid(True, alpha=0.25)
ax.set_axisbelow(True)

family_legend = [
    Line2D([0], [0], marker="D", color="w", label="Baseline", markerfacecolor="#111111", markersize=9),
    Line2D([0], [0], marker="o", color="w", label="Grasynda Q10", markerfacecolor="#d62828", markersize=9),
    Line2D([0], [0], marker="o", color="w", label="Grasynda SAX", markerfacecolor="#f77f00", markersize=9),
    Line2D([0], [0], marker="o", color="w", label="Other", markerfacecolor="#277da1", markersize=9),
]
ax.legend(handles=family_legend, loc="upper left", title="Group", fontsize=9)

plt.tight_layout()
plt.savefig(OUT_PLOT, dpi=160, bbox_inches="tight")
plt.close()

avg_df.to_csv(OUT_DATA, index=False)

print(f"Methods plotted: {avg_df['Method'].nunique()}")
print(f"Saved plot: {OUT_PLOT}")
print(f"Saved data: {OUT_DATA}")
