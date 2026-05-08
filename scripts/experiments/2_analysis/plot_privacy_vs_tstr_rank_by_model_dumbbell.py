"""
Plot privacy vs TSTR forecast rank separately for NHITS, MLP, and KAN.

- One panel per forecasting model
- Dumbbell per method: left point = TSTR forecast rank, right point = privacy rank
- Baseline included separately
- RawY variants excluded
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
OUT_PLOT = OUT_DIR / "privacy_vs_tstr_forecast_rank_by_model_dumbbell.png"
OUT_PDF = OUT_DIR / "privacy_vs_tstr_forecast_rank_by_model_dumbbell.pdf"
OUT_DATA = OUT_DIR / "privacy_vs_tstr_forecast_rank_by_model_dumbbell.csv"

MODELS = ["NHITS", "MLP", "KAN"]


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


forecast_df = pd.read_csv(FORECAST_PATH)
forecast_df = forecast_df[forecast_df["Forecasting_Model"].isin(MODELS)].copy()
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
    forecast_rank_df.groupby(["Forecasting_Model", "Dataset", "Group", "Augmentation_Method"], as_index=False)
    .agg(Forecast_Rank=("Forecast_Rank", "mean"))
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
).drop(columns=["Method"])

plot_df["Method"] = plot_df["Augmentation_Method"]
plot_df["Family"] = plot_df["Method"].map(method_family)
plot_df["Color"] = plot_df["Method"].map(family_color)

summary_df = (
    plot_df.groupby(["Forecasting_Model", "Method", "Family", "Color"], as_index=False)
    .agg(
        Avg_TSTR_Forecast_Rank=("Forecast_Rank", "mean"),
        Avg_Privacy_Rank=("Privacy_Rank", "mean"),
        Dataset_Count=("Dataset", "nunique"),
    )
)

method_order = (
    summary_df.groupby("Method")[["Avg_TSTR_Forecast_Rank", "Avg_Privacy_Rank"]]
    .mean()
    .assign(Sort=lambda d: (d["Avg_TSTR_Forecast_Rank"] + d["Avg_Privacy_Rank"]) / 2)
    .sort_values(["Sort", "Avg_TSTR_Forecast_Rank", "Avg_Privacy_Rank"])
    .index.tolist()
)

summary_df["Method"] = pd.Categorical(summary_df["Method"], categories=method_order, ordered=True)
summary_df = summary_df.sort_values(["Forecasting_Model", "Method"])

fig, axes = plt.subplots(1, 3, figsize=(16, max(6, 0.55 * len(method_order) + 1.5)), sharey=True)

for ax, model_name in zip(axes, MODELS):
    model_df = summary_df[summary_df["Forecasting_Model"] == model_name].copy()
    model_df = model_df.sort_values("Method")

    for i, (_, row) in enumerate(model_df.iterrows()):
        ax.plot(
            [row["Avg_TSTR_Forecast_Rank"], row["Avg_Privacy_Rank"]],
            [i, i],
            color="#c7c7c7",
            linewidth=2.0,
            zorder=1,
        )
        ax.scatter(
            row["Avg_TSTR_Forecast_Rank"],
            i,
            s=105,
            color=row["Color"],
            marker="o",
            edgecolors="white",
            linewidth=1.0,
            zorder=3,
        )
        privacy_marker = "^" if row["Method"] == "Baseline" else "s"
        ax.scatter(
            row["Avg_Privacy_Rank"],
            i,
            s=125,
            marker=privacy_marker,
            color=row["Color"],
            edgecolors="white",
            linewidth=1.0,
            zorder=4,
        )

    ax.set_title(model_name, fontsize=12, fontweight="bold")
    ax.set_xlabel("Average Rank", fontsize=10)
    ax.grid(True, axis="x", alpha=0.25)
    ax.set_axisbelow(True)
    ax.invert_xaxis()
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

axes[0].set_yticks(range(len(method_order)))
axes[0].set_yticklabels(
    [m.replace("Hybrid_", "").replace("_Continuous", "").replace("_", " ") for m in method_order],
    fontsize=9,
)
axes[0].set_ylabel("Method", fontsize=11)

fig.suptitle("Privacy vs TSTR Forecast Rank", fontsize=15, fontweight="bold", y=0.98)

legend_items = [
    Line2D([0], [0], marker="o", color="w", label="TSTR forecast rank", markerfacecolor="#666666", markersize=8),
    Line2D([0], [0], marker="s", color="w", label="Privacy rank", markerfacecolor="#666666", markersize=8),
    Line2D([0], [0], marker="^", color="w", label="Baseline privacy rank", markerfacecolor="#111111", markersize=8),
]
fig.legend(handles=legend_items, loc="lower center", ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.01))

plt.tight_layout(rect=[0, 0.04, 1, 0.95])
plt.savefig(OUT_PLOT, dpi=180, bbox_inches="tight")
plt.savefig(OUT_PDF, bbox_inches="tight")
plt.close()

summary_df.to_csv(OUT_DATA, index=False)

print(f"Saved plot: {OUT_PLOT}")
print(f"Saved pdf: {OUT_PDF}")
print(f"Saved data: {OUT_DATA}")
