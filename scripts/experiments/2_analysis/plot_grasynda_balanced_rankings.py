from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


REPO_ROOT = Path(__file__).resolve().parents[3]
INPUT_DIR = REPO_ROOT / "assets" / "results" / "grasynda_sensitivity"
CSV_DIR = INPUT_DIR / "csv"
CONFIG_SUMMARY_PATH = CSV_DIR / "grasynda_balanced_configuration_family_average_rank.csv"
QUANTILE_SUMMARY_PATH = CSV_DIR / "grasynda_balanced_quantile_average_rank.csv"


def ensure_inputs() -> None:
    for path in [CONFIG_SUMMARY_PATH, QUANTILE_SUMMARY_PATH]:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing input CSV: {path}. Run build_grasynda_balanced_rankings.py first."
            )


def base_style() -> None:
    sns.set_theme(style="whitegrid", context="talk")


def plot_balanced_configs(config_summary: pd.DataFrame, out_dir: Path) -> None:
    plot_df = config_summary.sort_values(
        ["mean_rank_across_dataset_groups", "wins_count", "mean_forecast_mase"],
        ascending=[True, False, True],
    )
    plt.figure(figsize=(8, 4.8))
    ax = sns.barplot(
        data=plot_df,
        x="configuration_family_label",
        y="mean_rank_across_dataset_groups",
        hue="configuration_family_label",
        palette="Blues_r",
        legend=False,
    )
    ax.set_title("Balanced Grasynda Configuration Family Ranking")
    ax.set_xlabel("")
    ax.set_ylabel("Average Rank Across Dataset/Groups\n(lower is better)")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / "grasynda_balanced_configuration_family_average_rank.png", dpi=200)
    plt.close()


def plot_balanced_quantiles(quantile_summary: pd.DataFrame, out_dir: Path) -> None:
    plot_df = quantile_summary.sort_values(
        ["mean_rank_across_dataset_groups", "wins_count", "mean_best_forecast_mase"],
        ascending=[True, False, True],
    )
    plt.figure(figsize=(10, 4.8))
    ax = sns.barplot(
        data=plot_df,
        x="n_quantiles",
        y="mean_rank_across_dataset_groups",
        hue="n_quantiles",
        palette="crest",
        legend=False,
    )
    ax.set_title("Balanced Grasynda Quantile Ranking")
    ax.set_xlabel("n_quantiles")
    ax.set_ylabel("Average Rank Across Dataset/Groups\n(lower is better)")
    plt.tight_layout()
    plt.savefig(out_dir / "grasynda_balanced_quantile_average_rank.png", dpi=200)
    plt.close()


def main() -> None:
    ensure_inputs()
    base_style()

    config_summary = pd.read_csv(CONFIG_SUMMARY_PATH)
    quantile_summary = pd.read_csv(QUANTILE_SUMMARY_PATH)

    plot_balanced_configs(config_summary, INPUT_DIR)
    plot_balanced_quantiles(quantile_summary, INPUT_DIR)

    print(f"Saved plots to {INPUT_DIR}")


if __name__ == "__main__":
    main()
