"""
Plot a dumbbell chart of average forecast rank vs average privacy rank.

Uses the precomputed all-model average-rank table:
assets/results/pymdma_metrics/corrected/privacy_vs_tstr_forecast_rank_avg_all_models_no_rawy.csv
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
IN_PATH = PROJECT_ROOT / "assets/results/pymdma_metrics/corrected/privacy_vs_tstr_forecast_rank_avg_all_models_no_rawy.csv"
OUT_PATH = PROJECT_ROOT / "assets/results/pymdma_metrics/corrected/privacy_vs_tstr_forecast_rank_avg_all_models_no_rawy_dumbbell.png"


def family_color(family: str) -> str:
    if family == "Baseline":
        return "#111111"
    if family == "Grasynda Q10":
        return "#d62828"
    if family == "Grasynda SAX":
        return "#f77f00"
    return "#277da1"


df = pd.read_csv(IN_PATH)
df["Sort"] = (df["Avg_Forecast_Rank"] + df["Avg_Privacy_Rank"]) / 2
df = df.sort_values(["Sort", "Avg_Forecast_Rank", "Avg_Privacy_Rank"]).reset_index(drop=True)
df["Color"] = df["Family"].map(family_color)

fig_h = max(6, 0.55 * len(df) + 1.5)
fig, ax = plt.subplots(figsize=(10, fig_h))

for i, row in df.iterrows():
    ax.plot(
        [row["Avg_Forecast_Rank"], row["Avg_Privacy_Rank"]],
        [i, i],
        color="#bdbdbd",
        linewidth=2.0,
        zorder=1,
    )
    ax.scatter(
        row["Avg_Forecast_Rank"],
        i,
        s=110,
        color=row["Color"],
        edgecolors="white",
        linewidth=1.0,
        zorder=3,
    )
    marker = "D" if row["Method"] == "Baseline" else "o"
    ax.scatter(
        row["Avg_Privacy_Rank"],
        i,
        s=130,
        color=row["Color"],
        marker=marker,
        edgecolors="white",
        linewidth=1.0,
        zorder=4,
    )

ax.set_yticks(range(len(df)))
ax.set_yticklabels(
    [m.replace("Hybrid_", "").replace("_Continuous", "").replace("_", " ") for m in df["Method"]],
    fontsize=9,
)
ax.set_xlabel("Average Rank (lower is better)", fontsize=11)
ax.set_title("Privacy vs TSTR Forecast Rank", fontsize=14, fontweight="bold")
ax.grid(True, axis="x", alpha=0.25)
ax.set_axisbelow(True)
ax.invert_xaxis()

from matplotlib.lines import Line2D

legend_items = [
    Line2D([0], [0], marker="o", color="w", label="Forecast rank", markerfacecolor="#666666", markersize=8),
    Line2D([0], [0], marker="o", color="w", label="Privacy rank", markerfacecolor="#666666", markersize=9),
    Line2D([0], [0], marker="D", color="w", label="Baseline", markerfacecolor="#111111", markersize=8),
]
ax.legend(handles=legend_items, loc="lower right", fontsize=9)

for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=180, bbox_inches="tight")
plt.close()

print(f"Saved plot: {OUT_PATH}")
