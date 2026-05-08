from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
INPUT_DIR = REPO_ROOT / "assets" / "results" / "grasynda_sensitivity"
CSV_DIR = INPUT_DIR / "csv"
RUNS_PATH = CSV_DIR / "grasynda_parameter_sensitivity_runs.csv"


def ensure_inputs() -> None:
    if not RUNS_PATH.exists():
        raise FileNotFoundError(
            f"Missing merged sensitivity runs at {RUNS_PATH}. "
            "Run build_grasynda_parameter_sensitivity.py first."
        )


def add_dataset_group(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["dataset_group"] = out["Dataset"] + " / " + out["Group"]
    return out


def best_row_per_configuration_family(runs: pd.DataFrame) -> pd.DataFrame:
    sort_cols = [
        "Dataset",
        "Group",
        "n_quantiles",
        "ensemble_transitions",
        "forecast_mase",
        "file_mtime",
    ]
    return (
        runs.sort_values(sort_cols, ascending=[True, True, True, True, True, False])
        .drop_duplicates(
            subset=["Dataset", "Group", "n_quantiles", "ensemble_transitions"],
            keep="first",
        )
        .reset_index(drop=True)
    )


def build_balanced_configuration_family_tables(runs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    families = best_row_per_configuration_family(runs)
    n_dataset_groups = families[["Dataset", "Group"]].drop_duplicates().shape[0]

    coverage = (
        families.groupby(
            ["n_quantiles", "ensemble_transitions"], dropna=False
        )[["Dataset", "Group"]]
        .apply(lambda x: x.drop_duplicates().shape[0])
        .reset_index(name="dataset_groups_covered")
    )
    common_configs = coverage[coverage["dataset_groups_covered"] == n_dataset_groups].copy()

    detailed = families.merge(
        common_configs,
        on=["n_quantiles", "ensemble_transitions"],
        how="inner",
    )
    detailed["configuration_family_label"] = detailed.apply(
        lambda row: (
            f"q={int(row['n_quantiles'])}, no-ens"
            if not row["ensemble_transitions"]
            else f"q={int(row['n_quantiles'])}, ensemble"
        ),
        axis=1,
    )
    detailed["rank_within_dataset_group"] = (
        detailed.groupby(["Dataset", "Group"])["forecast_mase"]
        .rank(method="dense", ascending=True)
        .astype(int)
    )
    detailed = detailed.sort_values(
        ["Dataset", "Group", "rank_within_dataset_group", "forecast_mase", "n_quantiles"],
        ascending=[True, True, True, True, True],
    ).reset_index(drop=True)

    summary = (
        detailed.groupby(
            ["n_quantiles", "ensemble_transitions", "configuration_family_label", "dataset_groups_covered"],
            dropna=False,
        )
        .agg(
            mean_rank_across_dataset_groups=("rank_within_dataset_group", "mean"),
            median_rank_across_dataset_groups=("rank_within_dataset_group", "median"),
            mean_forecast_mase=("forecast_mase", "mean"),
            median_forecast_mase=("forecast_mase", "median"),
            wins_count=("rank_within_dataset_group", lambda x: int((x == 1).sum())),
        )
        .reset_index()
        .sort_values(
            ["mean_rank_across_dataset_groups", "wins_count", "mean_forecast_mase"],
            ascending=[True, False, True],
        )
        .reset_index(drop=True)
    )
    summary.insert(0, "balanced_configuration_family_rank", range(1, len(summary) + 1))
    return detailed, summary


def build_balanced_quantile_tables(runs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    quantile_best = (
        runs.sort_values(
            ["Dataset", "Group", "n_quantiles", "forecast_mase", "file_mtime"],
            ascending=[True, True, True, True, False],
        )
        .drop_duplicates(subset=["Dataset", "Group", "n_quantiles"], keep="first")
        .reset_index(drop=True)
    )

    n_dataset_groups = quantile_best[["Dataset", "Group"]].drop_duplicates().shape[0]
    coverage = (
        quantile_best.groupby("n_quantiles")[["Dataset", "Group"]]
        .apply(lambda x: x.drop_duplicates().shape[0])
        .reset_index(name="dataset_groups_covered")
    )
    common_quantiles = coverage[coverage["dataset_groups_covered"] == n_dataset_groups].copy()

    detailed = quantile_best.merge(common_quantiles, on="n_quantiles", how="inner")
    detailed = detailed.rename(
        columns={
            "forecast_mase": "best_forecast_mase",
            "source_family": "best_source_family",
            "config_label": "best_config_label",
            "ensemble_transitions": "best_ensemble_transitions",
            "ensemble_size": "best_ensemble_size",
        }
    )
    detailed["rank_within_dataset_group"] = (
        detailed.groupby(["Dataset", "Group"])["best_forecast_mase"]
        .rank(method="dense", ascending=True)
        .astype(int)
    )
    detailed = detailed.sort_values(
        ["Dataset", "Group", "rank_within_dataset_group", "best_forecast_mase", "n_quantiles"],
        ascending=[True, True, True, True, True],
    ).reset_index(drop=True)

    summary = (
        detailed.groupby(["n_quantiles", "dataset_groups_covered"])
        .agg(
            mean_rank_across_dataset_groups=("rank_within_dataset_group", "mean"),
            median_rank_across_dataset_groups=("rank_within_dataset_group", "median"),
            mean_best_forecast_mase=("best_forecast_mase", "mean"),
            median_best_forecast_mase=("best_forecast_mase", "median"),
            wins_count=("rank_within_dataset_group", lambda x: int((x == 1).sum())),
        )
        .reset_index()
        .sort_values(
            ["mean_rank_across_dataset_groups", "wins_count", "mean_best_forecast_mase"],
            ascending=[True, False, True],
        )
        .reset_index(drop=True)
    )
    summary.insert(0, "balanced_quantile_rank", range(1, len(summary) + 1))
    return detailed, summary


def main() -> None:
    ensure_inputs()
    runs = add_dataset_group(pd.read_csv(RUNS_PATH))

    config_detailed, config_summary = build_balanced_configuration_family_tables(runs)
    quantile_detailed, quantile_summary = build_balanced_quantile_tables(runs)

    CSV_DIR.mkdir(parents=True, exist_ok=True)
    config_detailed_path = CSV_DIR / "grasynda_balanced_configuration_family_ranks.csv"
    config_summary_path = CSV_DIR / "grasynda_balanced_configuration_family_average_rank.csv"
    quantile_detailed_path = CSV_DIR / "grasynda_balanced_quantile_ranks.csv"
    quantile_summary_path = CSV_DIR / "grasynda_balanced_quantile_average_rank.csv"

    config_detailed.to_csv(config_detailed_path, index=False)
    config_summary.to_csv(config_summary_path, index=False)
    quantile_detailed.to_csv(quantile_detailed_path, index=False)
    quantile_summary.to_csv(quantile_summary_path, index=False)

    print(f"Saved balanced configuration family ranks: {config_detailed_path}")
    print(f"Saved balanced configuration family summary: {config_summary_path}")
    print(f"Saved balanced quantile ranks: {quantile_detailed_path}")
    print(f"Saved balanced quantile summary: {quantile_summary_path}")
    print("\nBalanced configuration family ranking:")
    print(config_summary.to_string(index=False))
    print("\nBalanced quantile ranking:")
    print(quantile_summary.to_string(index=False))


if __name__ == "__main__":
    main()
