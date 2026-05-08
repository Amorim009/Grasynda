from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
INPUT_DIR = REPO_ROOT / "assets" / "results" / "grasynda_sensitivity"
CSV_DIR = INPUT_DIR / "csv"
RUNS_PATH = CSV_DIR / "grasynda_parameter_sensitivity_runs.csv"
OUTPUT_DIR = CSV_DIR


def ensure_inputs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not RUNS_PATH.exists():
        raise FileNotFoundError(
            f"Missing merged sensitivity runs at {RUNS_PATH}. "
            "Run build_grasynda_parameter_sensitivity.py first."
        )


def load_random_search_runs() -> pd.DataFrame:
    df = pd.read_csv(RUNS_PATH)
    out = df[df["source_family"] == "random_search"].copy()
    if out.empty:
        raise ValueError("No Grasynda random-search rows found.")
    out["dataset_group"] = out["Dataset"] + " / " + out["Group"]
    return out


def add_normalized_rank(df: pd.DataFrame, metric_col: str) -> pd.DataFrame:
    out = df.copy()
    out["rank_within_dataset_group"] = (
        out.groupby(["Dataset", "Group"])[metric_col]
        .rank(method="dense", ascending=True)
        .astype(int)
    )
    out["n_items_within_dataset_group"] = out.groupby(["Dataset", "Group"])[metric_col].transform("size")
    denom = (out["n_items_within_dataset_group"] - 1).replace(0, 1)
    out["percentile_rank_within_dataset_group"] = (out["rank_within_dataset_group"] - 1) / denom
    return out


def best_configuration_families(rs: pd.DataFrame) -> pd.DataFrame:
    out = (
        rs.sort_values(
            ["Dataset", "Group", "n_quantiles", "ensemble_transitions", "forecast_mase", "file_mtime"],
            ascending=[True, True, True, True, True, False],
        )
        .drop_duplicates(subset=["Dataset", "Group", "n_quantiles", "ensemble_transitions"], keep="first")
        .reset_index(drop=True)
    )
    out["configuration_family_label"] = out.apply(
        lambda row: (
            f"q={int(row['n_quantiles'])}, no-ens"
            if not row["ensemble_transitions"]
            else f"q={int(row['n_quantiles'])}, ensemble"
        ),
        axis=1,
    )
    return add_normalized_rank(out, "forecast_mase")


def best_exact_configurations(rs: pd.DataFrame) -> pd.DataFrame:
    out = (
        rs.sort_values(
            [
                "Dataset",
                "Group",
                "n_quantiles",
                "ensemble_transitions",
                "ensemble_size",
                "forecast_mase",
                "file_mtime",
            ],
            ascending=[True, True, True, True, True, True, False],
        )
        .drop_duplicates(
            subset=[
                "Dataset",
                "Group",
                "n_quantiles",
                "ensemble_transitions",
                "ensemble_size",
                "config_label",
            ],
            keep="first",
        )
        .reset_index(drop=True)
    )
    return add_normalized_rank(out, "forecast_mase")


def best_quantiles(rs: pd.DataFrame) -> pd.DataFrame:
    out = (
        rs.sort_values(
            ["Dataset", "Group", "n_quantiles", "forecast_mase", "file_mtime"],
            ascending=[True, True, True, True, False],
        )
        .drop_duplicates(subset=["Dataset", "Group", "n_quantiles"], keep="first")
        .reset_index(drop=True)
    )
    out = out.rename(
        columns={
            "forecast_mase": "best_forecast_mase",
            "source_file": "best_source_file",
            "config_label": "best_config_label",
            "ensemble_transitions": "best_ensemble_transitions",
            "ensemble_size": "best_ensemble_size",
        }
    )
    return add_normalized_rank(out, "best_forecast_mase")


def best_ensemble_sizes(rs: pd.DataFrame) -> pd.DataFrame:
    ens = rs[rs["ensemble_transitions"] == True].copy()
    out = (
        ens.sort_values(
            ["Dataset", "Group", "ensemble_size", "forecast_mase", "file_mtime"],
            ascending=[True, True, True, True, False],
        )
        .drop_duplicates(subset=["Dataset", "Group", "ensemble_size"], keep="first")
        .reset_index(drop=True)
    )
    out = out.rename(columns={"forecast_mase": "best_forecast_mase_for_size"})
    return add_normalized_rank(out, "best_forecast_mase_for_size")


def summarize_config_families(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(["n_quantiles", "ensemble_transitions", "configuration_family_label"])
        .agg(
            dataset_groups_covered=("dataset_group", "nunique"),
            mean_rank=("rank_within_dataset_group", "mean"),
            median_rank=("rank_within_dataset_group", "median"),
            mean_percentile_rank=("percentile_rank_within_dataset_group", "mean"),
            median_percentile_rank=("percentile_rank_within_dataset_group", "median"),
            wins_count=("rank_within_dataset_group", lambda x: int((x == 1).sum())),
            mean_forecast_mase=("forecast_mase", "mean"),
            median_forecast_mase=("forecast_mase", "median"),
        )
        .reset_index()
        .sort_values(
            ["dataset_groups_covered", "mean_percentile_rank", "wins_count", "mean_forecast_mase"],
            ascending=[False, True, False, True],
        )
        .reset_index(drop=True)
    )
    summary.insert(0, "random_search_configuration_family_rank", range(1, len(summary) + 1))
    return summary


def summarize_exact_configurations(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(["config_label", "n_quantiles", "ensemble_transitions", "ensemble_size"], dropna=False)
        .agg(
            dataset_groups_covered=("dataset_group", "nunique"),
            mean_rank=("rank_within_dataset_group", "mean"),
            median_rank=("rank_within_dataset_group", "median"),
            mean_percentile_rank=("percentile_rank_within_dataset_group", "mean"),
            median_percentile_rank=("percentile_rank_within_dataset_group", "median"),
            wins_count=("rank_within_dataset_group", lambda x: int((x == 1).sum())),
            mean_forecast_mase=("forecast_mase", "mean"),
            median_forecast_mase=("forecast_mase", "median"),
        )
        .reset_index()
        .sort_values(
            ["dataset_groups_covered", "mean_percentile_rank", "wins_count", "mean_forecast_mase"],
            ascending=[False, True, False, True],
        )
        .reset_index(drop=True)
    )
    summary.insert(0, "random_search_exact_configuration_rank", range(1, len(summary) + 1))
    return summary


def summarize_best_exact_per_family(exact_summary: pd.DataFrame) -> pd.DataFrame:
    ordered = exact_summary.sort_values(
        [
            "n_quantiles",
            "ensemble_transitions",
            "dataset_groups_covered",
            "mean_percentile_rank",
            "wins_count",
            "mean_forecast_mase",
        ],
        ascending=[True, True, False, True, False, True],
    )
    best = (
        ordered.drop_duplicates(subset=["n_quantiles", "ensemble_transitions"], keep="first")
        .reset_index(drop=True)
        .copy()
    )
    best["configuration_family_label"] = best.apply(
        lambda row: (
            f"q={int(row['n_quantiles'])}, no-ens"
            if not row["ensemble_transitions"]
            else f"q={int(row['n_quantiles'])}, ensemble"
        ),
        axis=1,
    )
    best["plot_label"] = best.apply(
        lambda row: (
            row["configuration_family_label"]
            if not row["ensemble_transitions"]
            else f"{row['configuration_family_label']} | best: ens={int(row['ensemble_size'])}"
        ),
        axis=1,
    )
    best = best.sort_values(
        ["mean_percentile_rank", "dataset_groups_covered", "wins_count", "mean_forecast_mase"],
        ascending=[True, False, False, True],
    ).reset_index(drop=True)
    best.insert(0, "random_search_best_exact_per_family_rank", range(1, len(best) + 1))
    return best


def summarize_quantiles(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby("n_quantiles")
        .agg(
            dataset_groups_covered=("dataset_group", "nunique"),
            mean_rank=("rank_within_dataset_group", "mean"),
            median_rank=("rank_within_dataset_group", "median"),
            mean_percentile_rank=("percentile_rank_within_dataset_group", "mean"),
            median_percentile_rank=("percentile_rank_within_dataset_group", "median"),
            wins_count=("rank_within_dataset_group", lambda x: int((x == 1).sum())),
            mean_best_forecast_mase=("best_forecast_mase", "mean"),
            median_best_forecast_mase=("best_forecast_mase", "median"),
        )
        .reset_index()
        .sort_values(
            ["dataset_groups_covered", "mean_percentile_rank", "wins_count", "mean_best_forecast_mase"],
            ascending=[False, True, False, True],
        )
        .reset_index(drop=True)
    )
    summary.insert(0, "random_search_quantile_rank", range(1, len(summary) + 1))
    return summary


def summarize_ensemble_sizes(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby("ensemble_size")
        .agg(
            dataset_groups_covered=("dataset_group", "nunique"),
            mean_rank=("rank_within_dataset_group", "mean"),
            median_rank=("rank_within_dataset_group", "median"),
            mean_percentile_rank=("percentile_rank_within_dataset_group", "mean"),
            median_percentile_rank=("percentile_rank_within_dataset_group", "median"),
            wins_count=("rank_within_dataset_group", lambda x: int((x == 1).sum())),
            mean_best_forecast_mase=("best_forecast_mase_for_size", "mean"),
            median_best_forecast_mase=("best_forecast_mase_for_size", "median"),
        )
        .reset_index()
        .sort_values(
            ["dataset_groups_covered", "mean_percentile_rank", "wins_count", "mean_best_forecast_mase"],
            ascending=[False, True, False, True],
        )
        .reset_index(drop=True)
    )
    summary.insert(0, "random_search_ensemble_size_rank", range(1, len(summary) + 1))
    return summary


def add_threshold_exports(summary: pd.DataFrame, prefix: str, coverage_col: str = "dataset_groups_covered") -> None:
    for threshold in [7, 6, 5, 4, 3]:
        filtered = summary[summary[coverage_col] >= threshold].copy()
        if filtered.empty:
            continue
        filtered.to_csv(OUTPUT_DIR / f"{prefix}_coverage_gte_{threshold}.csv", index=False)


def main() -> None:
    ensure_inputs()
    rs = load_random_search_runs()

    config_families = best_configuration_families(rs)
    exact_configurations = best_exact_configurations(rs)
    quantiles = best_quantiles(rs)
    ensemble_sizes = best_ensemble_sizes(rs)
    config_summary = summarize_config_families(config_families)
    exact_config_summary = summarize_exact_configurations(exact_configurations)
    best_exact_per_family_summary = summarize_best_exact_per_family(exact_config_summary)
    quantile_summary = summarize_quantiles(quantiles)
    ensemble_size_summary = summarize_ensemble_sizes(ensemble_sizes)

    config_detail_path = OUTPUT_DIR / "grasynda_random_search_configuration_family_ranks.csv"
    config_summary_path = OUTPUT_DIR / "grasynda_random_search_configuration_family_summary.csv"
    exact_config_detail_path = OUTPUT_DIR / "grasynda_random_search_exact_configuration_ranks.csv"
    exact_config_summary_path = OUTPUT_DIR / "grasynda_random_search_exact_configuration_summary.csv"
    best_exact_per_family_path = OUTPUT_DIR / "grasynda_random_search_best_exact_per_family_summary.csv"
    quantile_detail_path = OUTPUT_DIR / "grasynda_random_search_quantile_ranks.csv"
    quantile_summary_path = OUTPUT_DIR / "grasynda_random_search_quantile_summary.csv"
    ensemble_size_detail_path = OUTPUT_DIR / "grasynda_random_search_ensemble_size_ranks.csv"
    ensemble_size_summary_path = OUTPUT_DIR / "grasynda_random_search_ensemble_size_summary.csv"

    config_families.to_csv(config_detail_path, index=False)
    config_summary.to_csv(config_summary_path, index=False)
    exact_configurations.to_csv(exact_config_detail_path, index=False)
    exact_config_summary.to_csv(exact_config_summary_path, index=False)
    best_exact_per_family_summary.to_csv(best_exact_per_family_path, index=False)
    quantiles.to_csv(quantile_detail_path, index=False)
    quantile_summary.to_csv(quantile_summary_path, index=False)
    ensemble_sizes.to_csv(ensemble_size_detail_path, index=False)
    ensemble_size_summary.to_csv(ensemble_size_summary_path, index=False)

    add_threshold_exports(config_summary, "grasynda_random_search_configuration_family_summary")
    add_threshold_exports(exact_config_summary, "grasynda_random_search_exact_configuration_summary")
    add_threshold_exports(best_exact_per_family_summary, "grasynda_random_search_best_exact_per_family_summary")
    add_threshold_exports(quantile_summary, "grasynda_random_search_quantile_summary")
    add_threshold_exports(ensemble_size_summary, "grasynda_random_search_ensemble_size_summary")

    print(f"Saved random-search configuration family ranks: {config_detail_path}")
    print(f"Saved random-search configuration family summary: {config_summary_path}")
    print(f"Saved random-search exact configuration ranks: {exact_config_detail_path}")
    print(f"Saved random-search exact configuration summary: {exact_config_summary_path}")
    print(f"Saved random-search best exact-per-family summary: {best_exact_per_family_path}")
    print(f"Saved random-search quantile ranks: {quantile_detail_path}")
    print(f"Saved random-search quantile summary: {quantile_summary_path}")
    print(f"Saved random-search ensemble size ranks: {ensemble_size_detail_path}")
    print(f"Saved random-search ensemble size summary: {ensemble_size_summary_path}")
    print("\nRandom-search configuration families:")
    print(config_summary.to_string(index=False))
    print("\nRandom-search exact configurations:")
    print(exact_config_summary.to_string(index=False))
    print("\nRandom-search best exact configuration per family:")
    print(best_exact_per_family_summary.to_string(index=False))
    print("\nRandom-search quantiles:")
    print(quantile_summary.to_string(index=False))
    print("\nRandom-search ensemble sizes:")
    print(ensemble_size_summary.to_string(index=False))


if __name__ == "__main__":
    main()
