from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


REPO_ROOT = Path(__file__).resolve().parents[3]
INPUT_DIR = REPO_ROOT / "assets" / "results" / "grasynda_sensitivity"
CSV_DIR = INPUT_DIR / "csv"
EXACT_CONFIG_SUMMARY_PATH = CSV_DIR / "grasynda_random_search_exact_configuration_summary.csv"


def ensure_inputs() -> None:
    if not EXACT_CONFIG_SUMMARY_PATH.exists():
        raise FileNotFoundError(
            f"Missing input CSV: {EXACT_CONFIG_SUMMARY_PATH}. "
            "Run build_grasynda_random_search_rankings.py first."
        )


def base_style() -> None:
    sns.set_theme(style="white", context="talk")


def build_heatmap_tables(summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = summary.copy()
    df["ensemble_axis"] = df["ensemble_size"].apply(lambda x: "no-ens" if pd.isna(x) else str(int(x)))

    quantile_order = sorted(df["n_quantiles"].dropna().astype(int).unique().tolist())
    ensemble_order = ["no-ens"] + [str(int(v)) for v in sorted(df["ensemble_size"].dropna().unique().tolist())]

    rank_heat = df.pivot_table(
        index="n_quantiles",
        columns="ensemble_axis",
        values="mean_percentile_rank",
        aggfunc="first",
    ).reindex(index=quantile_order, columns=ensemble_order)

    coverage_heat = df.pivot_table(
        index="n_quantiles",
        columns="ensemble_axis",
        values="dataset_groups_covered",
        aggfunc="first",
    ).reindex(index=quantile_order, columns=ensemble_order)

    annot = rank_heat.copy().astype(object)
    for ridx in rank_heat.index:
        for cidx in rank_heat.columns:
            rank_val = rank_heat.loc[ridx, cidx]
            cov_val = coverage_heat.loc[ridx, cidx]
            if pd.isna(rank_val):
                annot.loc[ridx, cidx] = ""
            else:
                annot.loc[ridx, cidx] = f"{rank_val:.2f}\n{int(cov_val)}/7"

    return rank_heat, coverage_heat, annot


def plot_heatmap(rank_heat: pd.DataFrame, annot: pd.DataFrame, out_dir: Path) -> None:
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "rank_red_to_blue",
        ["#b71c1c", "#ef5350", "#f6c26b", "#9cc2df", "#2459a6"],
    )
    cmap.set_bad("#eceff1")

    plt.figure(figsize=(12, 7.5))
    ax = sns.heatmap(
        rank_heat,
        annot=annot,
        fmt="",
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
        linewidths=0.8,
        linecolor="white",
        cbar_kws={"label": "Mean Percentile Rank (lower is better)"},
    )
    ax.set_title("Grasynda Random Search: Quantile x Ensemble Region Map")
    ax.set_xlabel("Ensemble Setting")
    ax.set_ylabel("n_quantiles")
    plt.tight_layout()
    plt.savefig(out_dir / "grasynda_random_search_quantile_ensemble_region_heatmap.pdf", dpi=220)
    plt.close()


def main() -> None:
    ensure_inputs()
    base_style()

    summary = pd.read_csv(EXACT_CONFIG_SUMMARY_PATH)
    rank_heat, coverage_heat, annot = build_heatmap_tables(summary)

    rank_heat.to_csv(CSV_DIR / "grasynda_random_search_quantile_ensemble_region_mean_percentile_rank.csv")
    coverage_heat.to_csv(CSV_DIR / "grasynda_random_search_quantile_ensemble_region_coverage.csv")

    plot_heatmap(rank_heat, annot, INPUT_DIR)
    print(f"Saved plot to {INPUT_DIR}")


if __name__ == "__main__":
    main()
