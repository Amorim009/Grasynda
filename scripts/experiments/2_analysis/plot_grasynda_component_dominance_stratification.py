from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


REPO_ROOT = Path(__file__).resolve().parents[3]
INPUT_DIR = REPO_ROOT / "assets" / "results" / "grasynda_sensitivity"
CSV_DIR = INPUT_DIR / "csv"
PROFILE_PATH = CSV_DIR / "grasynda_component_dominance_profile_by_dataset_group.csv"
QUANTILE_BUCKET_PATH = CSV_DIR / "grasynda_random_search_quantile_by_dominance_bucket.csv"
BEST_EXACT_BUCKET_PATH = CSV_DIR / "grasynda_random_search_best_exact_by_dominance_bucket.csv"

COMPONENT_COLORS = {
    "trend_share": "#2f6fb2",
    "seasonal_share": "#f2a541",
    "remainder_share": "#c94f4f",
}
BUCKET_ORDER = [
    "seasonal_heavy",
    "seasonal_leaning",
    "mixed",
    "trend_leaning",
    "trend_heavy",
    "remainder_leaning",
    "remainder_heavy",
]


def ensure_inputs() -> None:
    for path in [PROFILE_PATH, QUANTILE_BUCKET_PATH, BEST_EXACT_BUCKET_PATH]:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing input CSV: {path}. "
                "Run build_grasynda_component_dominance_stratification.py first."
            )


def base_style() -> None:
    sns.set_theme(style="whitegrid", context="talk")


def prettify_bucket(bucket: str) -> str:
    return bucket.replace("_", " ").title()


def bucket_sort_key(bucket: str) -> int:
    try:
        return BUCKET_ORDER.index(bucket)
    except ValueError:
        return len(BUCKET_ORDER)


def ordered_buckets(values: pd.Series) -> list[str]:
    unique_values = pd.Series(values.dropna().unique())
    return sorted(unique_values.tolist(), key=bucket_sort_key)


def plot_component_shares(profile: pd.DataFrame, out_dir: Path) -> None:
    plot_df = profile.copy().sort_values(
        ["dominance_bucket", "dominant_share", "dataset_group"],
        ascending=[True, False, True],
        key=lambda col: col.map(bucket_sort_key) if col.name == "dominance_bucket" else col,
    )

    fig, ax = plt.subplots(figsize=(12, 6.5))
    left = pd.Series(0.0, index=plot_df.index)
    for column in ["trend_share", "seasonal_share", "remainder_share"]:
        ax.barh(
            plot_df["dataset_group"],
            plot_df[column],
            left=left,
            color=COMPONENT_COLORS[column],
            edgecolor="white",
            label=column.replace("_share", "").title(),
        )
        left = left + plot_df[column]

    for idx, row in plot_df.reset_index(drop=True).iterrows():
        ax.text(
            1.01,
            idx,
            prettify_bucket(row["dominance_bucket"]),
            va="center",
            ha="left",
            fontsize=10,
        )

    ax.set_xlim(0, 1.15)
    ax.set_title("Grasynda Dataset/Group Component-Dominance Profiles")
    ax.set_xlabel("Share of Series with Dominant Component")
    ax.set_ylabel("")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_dir / "grasynda_component_dominance_by_dataset_group.pdf", dpi=220)
    plt.close()


def build_quantile_heatmap_tables(summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    plot_df = summary.copy()
    bucket_order = ordered_buckets(plot_df["dominance_bucket"])
    quantile_order = sorted(plot_df["n_quantiles"].dropna().astype(int).unique().tolist())

    heat = plot_df.pivot_table(
        index="dominance_bucket",
        columns="n_quantiles",
        values="mean_percentile_rank",
        aggfunc="first",
    ).reindex(index=bucket_order, columns=quantile_order)

    coverage = plot_df.pivot_table(
        index="dominance_bucket",
        columns="n_quantiles",
        values="dataset_groups_covered",
        aggfunc="first",
    ).reindex(index=bucket_order, columns=quantile_order)

    annot = heat.copy().astype(object)
    for bucket in heat.index:
        for n_quantiles in heat.columns:
            rank_val = heat.loc[bucket, n_quantiles]
            cov_val = coverage.loc[bucket, n_quantiles]
            if pd.isna(rank_val):
                annot.loc[bucket, n_quantiles] = ""
            else:
                annot.loc[bucket, n_quantiles] = f"{rank_val:.2f}\n{int(cov_val)} grp"

    heat.index = [prettify_bucket(bucket) for bucket in heat.index]
    annot.index = heat.index
    return heat, annot


def plot_quantile_heatmap(summary: pd.DataFrame, out_dir: Path) -> None:
    heat, annot = build_quantile_heatmap_tables(summary)
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "rank_red_to_blue",
        ["#c62828", "#ef5350", "#f6c26b", "#9cc2df", "#2f6fb2"],
    )
    cmap.set_bad("#eceff1")

    plt.figure(figsize=(12.5, 5.4))
    ax = sns.heatmap(
        heat,
        annot=annot,
        fmt="",
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
        linewidths=0.8,
        linecolor="white",
        cbar_kws={"label": "Mean Percentile Rank (lower is better)"},
    )
    ax.set_title("Grasynda Random Search: Quantiles by Component-Dominance Bucket")
    ax.set_xlabel("n_quantiles")
    ax.set_ylabel("")
    plt.tight_layout()
    plt.savefig(out_dir / "grasynda_random_search_quantiles_by_dominance_bucket.pdf", dpi=220)
    plt.close()


def plot_best_exact(best_exact: pd.DataFrame, out_dir: Path) -> None:
    plot_df = best_exact.copy().sort_values(
        ["dominance_bucket", "mean_percentile_rank"],
        ascending=[True, True],
        key=lambda col: col.map(bucket_sort_key) if col.name == "dominance_bucket" else col,
    )
    plot_df["bucket_label"] = plot_df["dominance_bucket"].map(prettify_bucket)

    plt.figure(figsize=(11.5, 4.8))
    ax = plt.gca()
    ax.barh(plot_df["bucket_label"], plot_df["mean_percentile_rank"], color="#4c78a8")
    for idx, row in plot_df.reset_index(drop=True).iterrows():
        ax.text(
            row["mean_percentile_rank"] + 0.01,
            idx,
            row["config_label"],
            va="center",
            ha="left",
            fontsize=10,
        )
    ax.set_title("Best Grasynda Exact Configuration per Component-Dominance Bucket")
    ax.set_xlabel("Mean Percentile Rank Within Bucket\n(lower is better)")
    ax.set_ylabel("")
    plt.tight_layout()
    plt.savefig(out_dir / "grasynda_random_search_best_exact_by_dominance_bucket.pdf", dpi=220)
    plt.close()


def main() -> None:
    ensure_inputs()
    base_style()

    profile = pd.read_csv(PROFILE_PATH)
    quantile_summary = pd.read_csv(QUANTILE_BUCKET_PATH)
    best_exact = pd.read_csv(BEST_EXACT_BUCKET_PATH)

    plot_component_shares(profile, INPUT_DIR)
    plot_quantile_heatmap(quantile_summary, INPUT_DIR)
    plot_best_exact(best_exact, INPUT_DIR)
    print(f"Saved plots to {INPUT_DIR}")


if __name__ == "__main__":
    main()
