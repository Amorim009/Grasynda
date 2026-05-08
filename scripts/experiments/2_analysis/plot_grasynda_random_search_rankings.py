from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
import seaborn as sns


REPO_ROOT = Path(__file__).resolve().parents[3]
INPUT_DIR = REPO_ROOT / "assets" / "results" / "grasynda_sensitivity"
CSV_DIR = INPUT_DIR / "csv"
CONFIG_SUMMARY_PATH = CSV_DIR / "grasynda_random_search_configuration_family_summary.csv"
BEST_EXACT_PER_FAMILY_PATH = CSV_DIR / "grasynda_random_search_best_exact_per_family_summary.csv"
EXACT_CONFIG_SUMMARY_PATH = CSV_DIR / "grasynda_random_search_exact_configuration_summary.csv"
QUANTILE_SUMMARY_PATH = CSV_DIR / "grasynda_random_search_quantile_summary.csv"
ENSEMBLE_SIZE_SUMMARY_PATH = CSV_DIR / "grasynda_random_search_ensemble_size_summary.csv"


def ensure_inputs() -> None:
    for path in [
        CONFIG_SUMMARY_PATH,
        BEST_EXACT_PER_FAMILY_PATH,
        EXACT_CONFIG_SUMMARY_PATH,
        QUANTILE_SUMMARY_PATH,
        ENSEMBLE_SIZE_SUMMARY_PATH,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing input CSV: {path}. Run build_grasynda_random_search_rankings.py first."
            )


def base_style() -> None:
    sns.set_theme(style="whitegrid", context="talk")


def _annotate_barh(ax, plot_df: pd.DataFrame, value_col: str, coverage_total: int) -> None:
    for idx, row in plot_df.reset_index(drop=True).iterrows():
        label = f"{int(row['dataset_groups_covered'])}/{coverage_total}"
        ax.text(
            row[value_col] + 0.01,
            idx,
            label,
            va="center",
            ha="left",
            fontsize=10,
        )


def _rank_palette(values: pd.Series) -> list[str]:
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "rank_red_to_blue",
        ["#c62828", "#ef5350", "#f6c26b", "#9cc2df", "#2f6fb2"],
    )
    vmin = float(values.min())
    vmax = float(values.max())
    if vmax <= vmin:
        return ["#c62828"] * len(values)
    return [mcolors.to_hex(cmap((float(v) - vmin) / (vmax - vmin))) for v in values]


def plot_config_families(summary: pd.DataFrame, out_dir: Path) -> None:
    plot_df = (
        summary[summary["dataset_groups_covered"] >= 3]
        .sort_values(
            ["mean_percentile_rank", "dataset_groups_covered", "wins_count"],
            ascending=[True, False, False],
        )
        .copy()
    )
    plt.figure(figsize=(12, 8.0))
    colors = _rank_palette(plot_df["mean_percentile_rank"])
    ax = plt.gca()
    ax.barh(plot_df["plot_label"], plot_df["mean_percentile_rank"], color=colors)
    ax.set_title("Grasynda Random Search: Best Exact Configuration per Family")
    ax.set_xlabel("Mean Percentile Rank Across Dataset/Groups\n(lower is better)")
    ax.set_ylabel("")
    ax.invert_yaxis()
    _annotate_barh(ax, plot_df, "mean_percentile_rank", coverage_total=7)
    plt.tight_layout()
    plt.savefig(out_dir / "grasynda_random_search_best_exact_per_family_by_rank.pdf", dpi=200)
    plt.close()


def plot_exact_configurations(summary: pd.DataFrame, out_dir: Path) -> None:
    plot_df = (
        summary[summary["dataset_groups_covered"] >= 4]
        .sort_values(
            ["mean_percentile_rank", "dataset_groups_covered", "wins_count"],
            ascending=[True, False, False],
        )
        .copy()
    )
    plt.figure(figsize=(11, 7.0))
    colors = _rank_palette(plot_df["mean_percentile_rank"])
    ax = plt.gca()
    ax.barh(plot_df["config_label"], plot_df["mean_percentile_rank"], color=colors)
    ax.set_title("Grasynda Random Search: Exact Quantile + Ensemble Configurations")
    ax.set_xlabel("Mean Percentile Rank Across Dataset/Groups\n(lower is better)")
    ax.set_ylabel("")
    ax.invert_yaxis()
    _annotate_barh(ax, plot_df, "mean_percentile_rank", coverage_total=7)
    plt.tight_layout()
    plt.savefig(out_dir / "grasynda_random_search_exact_configurations_by_rank.pdf", dpi=200)
    plt.close()


def plot_quantiles(summary: pd.DataFrame, out_dir: Path) -> None:
    plot_df = (
        summary[summary["dataset_groups_covered"] >= 6]
        .sort_values(
            ["mean_percentile_rank", "dataset_groups_covered", "wins_count"],
            ascending=[True, False, False],
        )
        .copy()
    )
    plot_df["quantile_label"] = plot_df["n_quantiles"].astype(int).astype(str)
    plt.figure(figsize=(10, 5.2))
    colors = _rank_palette(plot_df["mean_percentile_rank"])
    ax = plt.gca()
    ax.barh(plot_df["quantile_label"], plot_df["mean_percentile_rank"], color=colors)
    ax.set_title("Grasynda Random Search: Quantile Comparison by Rank")
    ax.set_xlabel("Mean Percentile Rank Across Dataset/Groups\n(lower is better)")
    ax.set_ylabel("n_quantiles")
    ax.invert_yaxis()
    _annotate_barh(ax, plot_df, "mean_percentile_rank", coverage_total=7)
    plt.tight_layout()
    plt.savefig(out_dir / "grasynda_random_search_quantiles_by_rank.pdf", dpi=200)
    plt.close()


def plot_ensemble_sizes(summary: pd.DataFrame, out_dir: Path) -> None:
    plot_df = (
        summary[summary["dataset_groups_covered"] >= 3]
        .sort_values(
            ["mean_percentile_rank", "dataset_groups_covered", "wins_count"],
            ascending=[True, False, False],
        )
        .copy()
    )
    plot_df["ensemble_size_label"] = plot_df["ensemble_size"].astype(int).astype(str)
    plt.figure(figsize=(10, 4.8))
    colors = _rank_palette(plot_df["mean_percentile_rank"])
    ax = plt.gca()
    ax.barh(plot_df["ensemble_size_label"], plot_df["mean_percentile_rank"], color=colors)
    ax.set_title("Grasynda Random Search: Ensemble Sizes by Rank")
    ax.set_xlabel("Mean Percentile Rank Across Dataset/Groups\n(lower is better)")
    ax.set_ylabel("ensemble_size")
    ax.invert_yaxis()
    _annotate_barh(ax, plot_df, "mean_percentile_rank", coverage_total=5)
    plt.tight_layout()
    plt.savefig(out_dir / "grasynda_random_search_ensemble_sizes_by_rank.pdf", dpi=200)
    plt.close()


def main() -> None:
    ensure_inputs()
    base_style()
    config_summary = pd.read_csv(BEST_EXACT_PER_FAMILY_PATH)
    exact_config_summary = pd.read_csv(EXACT_CONFIG_SUMMARY_PATH)
    quantile_summary = pd.read_csv(QUANTILE_SUMMARY_PATH)
    ensemble_size_summary = pd.read_csv(ENSEMBLE_SIZE_SUMMARY_PATH)
    plot_config_families(config_summary, INPUT_DIR)
    plot_exact_configurations(exact_config_summary, INPUT_DIR)
    plot_quantiles(quantile_summary, INPUT_DIR)
    plot_ensemble_sizes(ensemble_size_summary, INPUT_DIR)
    print(f"Saved plots to {INPUT_DIR}")


if __name__ == "__main__":
    main()
