from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
INPUT_DIR = REPO_ROOT / "assets" / "results" / "grasynda_sensitivity"
CSV_DIR = INPUT_DIR / "csv"
OUTPUT_DIR = INPUT_DIR
RUNS_PATH = CSV_DIR / "grasynda_parameter_sensitivity_runs.csv"


def ensure_inputs() -> None:
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    if not RUNS_PATH.exists():
        raise FileNotFoundError(
            f"Missing merged sensitivity runs at {RUNS_PATH}. "
            "Run build_grasynda_parameter_sensitivity.py first."
        )


def collapse_to_quantiles(runs: pd.DataFrame) -> pd.DataFrame:
    sort_cols = [
        "Dataset",
        "Group",
        "n_quantiles",
        "forecast_mase",
        "rank_within_dataset_group_all_sources",
        "file_mtime",
    ]
    best_per_quantile = (
        runs.sort_values(sort_cols, ascending=[True, True, True, True, True, False])
        .drop_duplicates(subset=["Dataset", "Group", "n_quantiles"], keep="first")
        .reset_index(drop=True)
    )
    best_per_quantile = best_per_quantile.rename(
        columns={
            "forecast_mase": "best_forecast_mase",
            "source_family": "best_source_family",
            "config_label": "best_config_label",
            "ensemble_transitions": "best_ensemble_transitions",
            "ensemble_size": "best_ensemble_size",
        }
    )
    best_per_quantile["rank_within_dataset_group"] = (
        best_per_quantile.groupby(["Dataset", "Group"])["best_forecast_mase"]
        .rank(method="dense", ascending=True)
        .astype(int)
    )
    best_per_quantile["best_mase_for_dataset_group"] = (
        best_per_quantile.groupby(["Dataset", "Group"])["best_forecast_mase"].transform("min")
    )
    best_per_quantile["delta_to_best_mase"] = (
        best_per_quantile["best_forecast_mase"] - best_per_quantile["best_mase_for_dataset_group"]
    )
    return best_per_quantile.sort_values(
        ["Dataset", "Group", "rank_within_dataset_group", "best_forecast_mase"],
        ascending=[True, True, True, True],
    ).reset_index(drop=True)


def build_dataset_winners(collapsed: pd.DataFrame) -> pd.DataFrame:
    winners = (
        collapsed.sort_values(
            ["Dataset", "Group", "best_forecast_mase", "n_quantiles"],
            ascending=[True, True, True, True],
        )
        .drop_duplicates(subset=["Dataset", "Group"], keep="first")
        .reset_index(drop=True)
    )

    runner_up = (
        collapsed.sort_values(
            ["Dataset", "Group", "best_forecast_mase", "n_quantiles"],
            ascending=[True, True, True, True],
        )
        .groupby(["Dataset", "Group"], group_keys=False)
        .nth(1)
        .reset_index()
    )
    runner_up = runner_up.rename(
        columns={
            "n_quantiles": "runner_up_n_quantiles",
            "best_forecast_mase": "runner_up_forecast_mase",
        }
    )
    winners = winners.merge(
        runner_up[["Dataset", "Group", "runner_up_n_quantiles", "runner_up_forecast_mase"]],
        on=["Dataset", "Group"],
        how="left",
    )
    winners["margin_to_runner_up"] = (
        winners["runner_up_forecast_mase"] - winners["best_forecast_mase"]
    )
    winners = winners.rename(columns={"n_quantiles": "best_n_quantiles"})
    return winners[
        [
            "Dataset",
            "Group",
            "best_n_quantiles",
            "best_forecast_mase",
            "runner_up_n_quantiles",
            "runner_up_forecast_mase",
            "margin_to_runner_up",
            "best_source_family",
            "best_config_label",
            "best_ensemble_transitions",
            "best_ensemble_size",
        ]
    ].sort_values(["Dataset", "Group"]).reset_index(drop=True)


def build_overall_quantile_ranking(collapsed: pd.DataFrame) -> pd.DataFrame:
    overall = (
        collapsed.groupby("n_quantiles")
        .agg(
            dataset_groups_covered=("best_forecast_mase", "size"),
            wins_count=("rank_within_dataset_group", lambda x: int((x == 1).sum())),
            mean_best_forecast_mase=("best_forecast_mase", "mean"),
            median_best_forecast_mase=("best_forecast_mase", "median"),
            mean_rank_across_dataset_groups=("rank_within_dataset_group", "mean"),
            mean_delta_to_best_mase=("delta_to_best_mase", "mean"),
            median_delta_to_best_mase=("delta_to_best_mase", "median"),
        )
        .reset_index()
        .sort_values(
            ["wins_count", "mean_rank_across_dataset_groups", "mean_delta_to_best_mase"],
            ascending=[False, True, True],
        )
        .reset_index(drop=True)
    )
    overall.insert(0, "overall_quantile_rank", range(1, len(overall) + 1))
    return overall


def main() -> None:
    ensure_inputs()
    runs = pd.read_csv(RUNS_PATH)

    collapsed = collapse_to_quantiles(runs)
    winners = build_dataset_winners(collapsed)
    overall = build_overall_quantile_ranking(collapsed)

    winners_path = CSV_DIR / "grasynda_best_quantile_by_dataset_group.csv"
    overall_path = CSV_DIR / "grasynda_best_quantile_overall_ranking.csv"
    collapsed_path = CSV_DIR / "grasynda_quantile_collapsed.csv"

    collapsed.to_csv(collapsed_path, index=False)
    winners.to_csv(winners_path, index=False)
    overall.to_csv(overall_path, index=False)

    print(f"Saved collapsed quantile results: {collapsed_path}")
    print(f"Saved dataset/group winners: {winners_path}")
    print(f"Saved overall quantile ranking: {overall_path}")
    print("\nBest quantile by dataset/group:")
    print(winners.to_string(index=False))
    print("\nOverall quantile ranking:")
    print(overall.to_string(index=False))


if __name__ == "__main__":
    main()
