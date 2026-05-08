from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


REPO_ROOT = Path(__file__).resolve().parents[3]
INPUT_DIR = REPO_ROOT / "assets" / "results" / "grasynda_sensitivity"
CSV_DIR = INPUT_DIR / "csv"
RUNS_PATH = CSV_DIR / "grasynda_parameter_sensitivity_runs.csv"
CONFIG_PATH = CSV_DIR / "grasynda_configuration_ranking.csv"
CHOICE_PATH = CSV_DIR / "grasynda_parameter_choice_ranking.csv"


def ensure_inputs() -> None:
    if not RUNS_PATH.exists():
        raise FileNotFoundError(f"Missing input CSV: {RUNS_PATH}")


def base_style() -> None:
    sns.set_theme(style="whitegrid", context="talk")


def plot_n_quantiles(df: pd.DataFrame, out_dir: Path) -> None:
    plt.figure(figsize=(12, 6))
    order = sorted(df["n_quantiles"].dropna().unique())
    ax = sns.boxplot(
        data=df,
        x="n_quantiles",
        y="rank_within_dataset_group_all_sources",
        color="#c9d7f0",
        order=order,
        showfliers=False,
    )
    sns.stripplot(
        data=df,
        x="n_quantiles",
        y="rank_within_dataset_group_all_sources",
        hue="source_family",
        dodge=False,
        alpha=0.7,
        order=order,
        ax=ax,
    )
    ax.set_title("Grasynda Sensitivity: n_quantiles")
    ax.set_xlabel("n_quantiles")
    ax.set_ylabel("Rank Within Dataset/Group\n(lower is better)")
    ax.legend(title="Source", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(out_dir / "grasynda_sensitivity_n_quantiles.png", dpi=200)
    plt.close()


def plot_ensemble_flag(df: pd.DataFrame, out_dir: Path) -> None:
    plot_df = (
        df.groupby("ensemble_label", as_index=False)
        .agg(
            mean_rank=("rank_within_dataset_group_all_sources", "mean"),
            mean_mase=("forecast_mase", "mean"),
        )
        .sort_values("mean_rank")
    )
    plt.figure(figsize=(7, 5))
    ax = sns.barplot(
        data=plot_df,
        x="ensemble_label",
        y="mean_rank",
        hue="ensemble_label",
        palette="Blues_d",
        legend=False,
    )
    ax.set_title("Grasynda Sensitivity: Ensemble On vs Off")
    ax.set_xlabel("")
    ax.set_ylabel("Mean Rank Within Dataset/Group\n(lower is better)")
    for idx, row in plot_df.reset_index(drop=True).iterrows():
        ax.text(idx, row["mean_rank"] + 0.03, f"{row['mean_rank']:.2f}", ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_dir / "grasynda_sensitivity_ensemble_flag.png", dpi=200)
    plt.close()


def plot_ensemble_size(df: pd.DataFrame, out_dir: Path) -> None:
    ens_df = df[df["ensemble_transitions"]].copy()
    if ens_df.empty:
        return
    plt.figure(figsize=(10, 6))
    order = sorted(ens_df["ensemble_size"].dropna().unique())
    ax = sns.boxplot(
        data=ens_df,
        x="ensemble_size",
        y="rank_within_dataset_group_all_sources",
        color="#f1d4a8",
        order=order,
        showfliers=False,
    )
    sns.stripplot(
        data=ens_df,
        x="ensemble_size",
        y="rank_within_dataset_group_all_sources",
        color="#a65e2e",
        alpha=0.55,
        order=order,
        ax=ax,
    )
    ax.set_title("Grasynda Sensitivity: ensemble_size (when enabled)")
    ax.set_xlabel("ensemble_size")
    ax.set_ylabel("Rank Within Dataset/Group\n(lower is better)")
    plt.tight_layout()
    plt.savefig(out_dir / "grasynda_sensitivity_ensemble_size.png", dpi=200)
    plt.close()


def plot_heatmap(config_df: pd.DataFrame, out_dir: Path) -> None:
    ens_df = config_df[config_df["ensemble_transitions"] == True].copy()
    if ens_df.empty:
        return
    heat = ens_df.pivot_table(
        index="ensemble_size",
        columns="n_quantiles",
        values="mean_rank_within_dataset_group",
        aggfunc="mean",
    )
    plt.figure(figsize=(12, 6))
    ax = sns.heatmap(heat, annot=True, fmt=".2f", cmap="YlGnBu_r")
    ax.set_title("Grasynda Sensitivity Heatmap\nMean Rank for Ensemble Configurations")
    ax.set_xlabel("n_quantiles")
    ax.set_ylabel("ensemble_size")
    plt.tight_layout()
    plt.savefig(out_dir / "grasynda_sensitivity_heatmap_ensemble_configs.png", dpi=200)
    plt.close()


def plot_top_configs(config_df: pd.DataFrame, out_dir: Path) -> None:
    top = config_df.head(12).copy()
    plt.figure(figsize=(12, 7))
    ax = sns.barplot(
        data=top,
        y="config_label",
        x="mean_rank_within_dataset_group",
        hue="config_label",
        palette="viridis",
        legend=False,
    )
    ax.set_title("Top Grasynda Parameter Configurations")
    ax.set_xlabel("Mean Rank Within Dataset/Group\n(lower is better)")
    ax.set_ylabel("")
    plt.tight_layout()
    plt.savefig(out_dir / "grasynda_top_configurations.png", dpi=200)
    plt.close()


def main() -> None:
    ensure_inputs()
    base_style()

    out_dir = INPUT_DIR
    runs = pd.read_csv(RUNS_PATH)
    config = pd.read_csv(CONFIG_PATH)
    _ = pd.read_csv(CHOICE_PATH)

    plot_n_quantiles(runs, out_dir)
    plot_ensemble_flag(runs, out_dir)
    plot_ensemble_size(runs, out_dir)
    plot_heatmap(config, out_dir)
    plot_top_configs(config, out_dir)

    print(f"Saved plots to {out_dir}")


if __name__ == "__main__":
    main()
