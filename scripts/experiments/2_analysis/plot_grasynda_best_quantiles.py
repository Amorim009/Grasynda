from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


REPO_ROOT = Path(__file__).resolve().parents[3]
INPUT_DIR = REPO_ROOT / "assets" / "results" / "grasynda_sensitivity"
CSV_DIR = INPUT_DIR / "csv"
WINNERS_PATH = CSV_DIR / "grasynda_best_quantile_by_dataset_group.csv"
COLLAPSED_PATH = CSV_DIR / "grasynda_quantile_collapsed.csv"
OVERALL_PATH = CSV_DIR / "grasynda_best_quantile_overall_ranking.csv"


def ensure_inputs() -> None:
    for path in [WINNERS_PATH, COLLAPSED_PATH, OVERALL_PATH]:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing input CSV: {path}. Run build_grasynda_best_quantiles.py first."
            )


def base_style() -> None:
    sns.set_theme(style="whitegrid", context="talk")


def add_dataset_group_label(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["dataset_group"] = out["Dataset"] + " / " + out["Group"]
    return out


def plot_dataset_winners(winners: pd.DataFrame, out_dir: Path) -> None:
    plot_df = add_dataset_group_label(winners).sort_values(
        ["best_n_quantiles", "best_forecast_mase", "dataset_group"],
        ascending=[True, True, True],
    )
    plt.figure(figsize=(12, 6))
    ax = sns.scatterplot(
        data=plot_df,
        x="best_n_quantiles",
        y="dataset_group",
        size="margin_to_runner_up",
        hue="best_forecast_mase",
        palette="viridis_r",
        sizes=(120, 420),
        legend="brief",
    )
    ax.set_title("Best Grasynda Quantile by Dataset/Group")
    ax.set_xlabel("Winning n_quantiles")
    ax.set_ylabel("")
    plt.tight_layout()
    plt.savefig(out_dir / "grasynda_best_quantile_by_dataset_group.png", dpi=200)
    plt.close()


def plot_quantile_heatmap(collapsed: pd.DataFrame, out_dir: Path) -> None:
    plot_df = add_dataset_group_label(collapsed)
    heat = plot_df.pivot_table(
        index="dataset_group",
        columns="n_quantiles",
        values="delta_to_best_mase",
        aggfunc="min",
    )
    plt.figure(figsize=(13, 6))
    ax = sns.heatmap(heat, annot=True, fmt=".3f", cmap="YlGnBu_r")
    ax.set_title("Grasynda Quantile Performance by Dataset/Group\nDelta to Best MASE (lower is better)")
    ax.set_xlabel("n_quantiles")
    ax.set_ylabel("")
    plt.tight_layout()
    plt.savefig(out_dir / "grasynda_quantile_delta_heatmap.png", dpi=200)
    plt.close()


def plot_overall_ranking(overall: pd.DataFrame, out_dir: Path) -> None:
    plot_df = overall.sort_values(
        ["wins_count", "mean_rank_across_dataset_groups", "mean_delta_to_best_mase"],
        ascending=[False, True, True],
    )
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(
        data=plot_df,
        x="n_quantiles",
        y="mean_rank_across_dataset_groups",
        hue="wins_count",
        dodge=False,
        palette="crest",
    )
    ax.set_title("Overall Grasynda Quantile Ranking")
    ax.set_xlabel("n_quantiles")
    ax.set_ylabel("Mean Rank Across Dataset/Groups\n(lower is better)")
    plt.tight_layout()
    plt.savefig(out_dir / "grasynda_best_quantile_overall_ranking.png", dpi=200)
    plt.close()


def main() -> None:
    ensure_inputs()
    base_style()

    winners = pd.read_csv(WINNERS_PATH)
    collapsed = pd.read_csv(COLLAPSED_PATH)
    overall = pd.read_csv(OVERALL_PATH)

    plot_dataset_winners(winners, INPUT_DIR)
    plot_quantile_heatmap(collapsed, INPUT_DIR)
    plot_overall_ranking(overall, INPUT_DIR)

    print(f"Saved plots to {INPUT_DIR}")


if __name__ == "__main__":
    main()
