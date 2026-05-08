import sys
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.load_data.config import DATASETS, DATA_GROUPS


INPUT_DIR = REPO_ROOT / "assets" / "results" / "grasynda_sensitivity"
CSV_DIR = INPUT_DIR / "csv"
RUNS_PATH = CSV_DIR / "grasynda_parameter_sensitivity_runs.csv"

COMPONENTS = ["trend", "seasonal", "remainder"]
DOMINANCE_BUCKET_ORDER = [
    "seasonal_heavy",
    "seasonal_leaning",
    "mixed",
    "trend_leaning",
    "trend_heavy",
    "remainder_leaning",
    "remainder_heavy",
]


def ensure_inputs() -> None:
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    if not RUNS_PATH.exists():
        raise FileNotFoundError(
            f"Missing merged sensitivity runs at {RUNS_PATH}. "
            "Run build_grasynda_parameter_sensitivity.py first."
        )


def dominance_bucket(dominant_component: str, dominant_share: float) -> str:
    if dominant_share >= 0.80:
        return f"{dominant_component}_heavy"
    if dominant_share >= 0.60:
        return f"{dominant_component}_leaning"
    return "mixed"


def compute_component_dominance(dataset: str, group: str) -> pd.DataFrame:
    data_loader = DATASETS[dataset]
    df, _, _, _, period = data_loader.load_everything(group)
    rows = []

    for unique_id, uid_df in df.groupby("unique_id"):
        y = uid_df["y"].to_numpy()
        if len(y) < max((2 * period), period + 2):
            continue

        try:
            decomposition = STL(y, period=period, robust=False).fit()
        except Exception:
            continue

        y_std = float(np.std(y))
        if not np.isfinite(y_std) or y_std <= 0:
            continue

        strengths = {
            "trend": float(np.std(decomposition.trend) / y_std),
            "seasonal": float(np.std(decomposition.seasonal) / y_std),
            "remainder": float(np.std(decomposition.resid) / y_std),
        }
        dominant_component = max(strengths, key=strengths.get)
        ordered_strengths = sorted(strengths.values(), reverse=True)
        dominance_margin = ordered_strengths[0] - ordered_strengths[1]

        rows.append(
            {
                "Dataset": dataset,
                "Group": group,
                "dataset_group": f"{dataset} / {group}",
                "unique_id": unique_id,
                "series_length": int(len(y)),
                "period": int(period),
                "y_std": y_std,
                "trend_rel_strength": strengths["trend"],
                "seasonal_rel_strength": strengths["seasonal"],
                "remainder_rel_strength": strengths["remainder"],
                "dominant_component": dominant_component,
                "dominance_margin": dominance_margin,
            }
        )

    return pd.DataFrame(rows)


def build_dominance_outputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    per_series_frames = []
    profile_rows = []

    for dataset, group in DATA_GROUPS:
        per_series = compute_component_dominance(dataset, group)
        if per_series.empty:
            continue
        per_series_frames.append(per_series)

        share_map = per_series["dominant_component"].value_counts(normalize=True).to_dict()
        dominant_component = max(COMPONENTS, key=lambda name: share_map.get(name, 0.0))
        dominant_share = float(share_map.get(dominant_component, 0.0))
        share_values = sorted([float(share_map.get(name, 0.0)) for name in COMPONENTS], reverse=True)
        dominance_margin = share_values[0] - share_values[1]

        row = {
            "Dataset": dataset,
            "Group": group,
            "dataset_group": f"{dataset} / {group}",
            "n_series_evaluated": int(len(per_series)),
            "dominant_component": dominant_component,
            "dominant_share": dominant_share,
            "dominance_margin": dominance_margin,
            "dominance_bucket": dominance_bucket(dominant_component, dominant_share),
        }
        for component in COMPONENTS:
            row[f"{component}_share"] = float(share_map.get(component, 0.0))
            row[f"{component}_rel_strength_mean"] = float(per_series[f"{component}_rel_strength"].mean())
            row[f"{component}_rel_strength_median"] = float(per_series[f"{component}_rel_strength"].median())
        profile_rows.append(row)

    per_series_df = pd.concat(per_series_frames, axis=0, ignore_index=True)
    profile_df = pd.DataFrame(profile_rows).sort_values(
        ["dominant_component", "dominant_share", "dataset_group"],
        ascending=[True, False, True],
    ).reset_index(drop=True)
    return per_series_df, profile_df


def load_random_search_runs() -> pd.DataFrame:
    runs = pd.read_csv(RUNS_PATH)
    runs = runs[runs["source_family"] == "random_search"].copy()
    if runs.empty:
        raise ValueError("No Grasynda random-search rows found.")

    runs["dataset_group"] = runs["Dataset"] + " / " + runs["Group"]
    runs["rank_within_dataset_group"] = (
        runs.groupby(["Dataset", "Group"])["forecast_mase"]
        .rank(method="dense", ascending=True)
        .astype(int)
    )
    runs["n_configs_within_dataset_group"] = runs.groupby(["Dataset", "Group"])["forecast_mase"].transform("size")
    denom = (runs["n_configs_within_dataset_group"] - 1).replace(0, 1)
    runs["percentile_rank_within_dataset_group"] = (runs["rank_within_dataset_group"] - 1) / denom
    runs["best_forecast_mase_within_dataset_group"] = runs.groupby(["Dataset", "Group"])["forecast_mase"].transform(
        "min"
    )
    runs["delta_to_best_mase"] = runs["forecast_mase"] - runs["best_forecast_mase_within_dataset_group"]
    return runs


def add_bucket_rank(summary: pd.DataFrame, bucket_col: str, rank_name: str) -> pd.DataFrame:
    out = summary.copy()
    out[rank_name] = out.groupby(bucket_col).cumcount() + 1
    return out


def summarize_quantiles_by_bucket(runs: pd.DataFrame, bucket_col: str) -> pd.DataFrame:
    summary = (
        runs.groupby([bucket_col, "n_quantiles"])
        .agg(
            dataset_groups_covered=("dataset_group", "nunique"),
            mean_rank=("rank_within_dataset_group", "mean"),
            median_rank=("rank_within_dataset_group", "median"),
            mean_percentile_rank=("percentile_rank_within_dataset_group", "mean"),
            median_percentile_rank=("percentile_rank_within_dataset_group", "median"),
            mean_delta_to_best_mase=("delta_to_best_mase", "mean"),
            median_delta_to_best_mase=("delta_to_best_mase", "median"),
            wins_count=("rank_within_dataset_group", lambda x: int((x == 1).sum())),
            mean_forecast_mase=("forecast_mase", "mean"),
        )
        .reset_index()
        .sort_values(
            [bucket_col, "dataset_groups_covered", "mean_percentile_rank", "wins_count", "mean_delta_to_best_mase", "n_quantiles"],
            ascending=[True, False, True, False, True, True],
        )
        .reset_index(drop=True)
    )
    return add_bucket_rank(summary, bucket_col, f"random_search_quantile_rank_within_{bucket_col}")


def summarize_exact_configs_by_bucket(runs: pd.DataFrame, bucket_col: str) -> pd.DataFrame:
    summary = (
        runs.groupby([bucket_col, "config_label", "n_quantiles", "ensemble_transitions", "ensemble_size"], dropna=False)
        .agg(
            dataset_groups_covered=("dataset_group", "nunique"),
            mean_rank=("rank_within_dataset_group", "mean"),
            median_rank=("rank_within_dataset_group", "median"),
            mean_percentile_rank=("percentile_rank_within_dataset_group", "mean"),
            median_percentile_rank=("percentile_rank_within_dataset_group", "median"),
            mean_delta_to_best_mase=("delta_to_best_mase", "mean"),
            median_delta_to_best_mase=("delta_to_best_mase", "median"),
            wins_count=("rank_within_dataset_group", lambda x: int((x == 1).sum())),
            mean_forecast_mase=("forecast_mase", "mean"),
        )
        .reset_index()
        .sort_values(
            [bucket_col, "dataset_groups_covered", "mean_percentile_rank", "wins_count", "mean_delta_to_best_mase", "config_label"],
            ascending=[True, False, True, False, True, True],
        )
        .reset_index(drop=True)
    )
    return add_bucket_rank(summary, bucket_col, f"random_search_exact_config_rank_within_{bucket_col}")


def best_rows_within_bucket(summary: pd.DataFrame, bucket_col: str, rank_col: str) -> pd.DataFrame:
    return (
        summary.sort_values([bucket_col, rank_col], ascending=[True, True])
        .drop_duplicates(subset=[bucket_col], keep="first")
        .reset_index(drop=True)
    )


def sort_by_bucket(df: pd.DataFrame, bucket_col: str) -> pd.DataFrame:
    order_map = {name: idx for idx, name in enumerate(DOMINANCE_BUCKET_ORDER)}
    out = df.copy()
    out["_bucket_order"] = out[bucket_col].map(lambda x: order_map.get(x, len(order_map)))
    out = out.sort_values(["_bucket_order"] + [c for c in out.columns if c not in {"_bucket_order", bucket_col}])
    return out.drop(columns="_bucket_order").reset_index(drop=True)


def main() -> None:
    ensure_inputs()

    per_series, profile = build_dominance_outputs()
    runs = load_random_search_runs().merge(
        profile[
            [
                "Dataset",
                "Group",
                "dataset_group",
                "dominant_component",
                "dominant_share",
                "dominance_margin",
                "dominance_bucket",
                "trend_share",
                "seasonal_share",
                "remainder_share",
            ]
        ],
        on=["Dataset", "Group", "dataset_group"],
        how="left",
    )

    quantiles_by_bucket = sort_by_bucket(summarize_quantiles_by_bucket(runs, "dominance_bucket"), "dominance_bucket")
    exact_by_bucket = sort_by_bucket(summarize_exact_configs_by_bucket(runs, "dominance_bucket"), "dominance_bucket")
    best_quantile_by_bucket = sort_by_bucket(
        best_rows_within_bucket(
            quantiles_by_bucket, "dominance_bucket", "random_search_quantile_rank_within_dominance_bucket"
        ),
        "dominance_bucket",
    )
    best_exact_by_bucket = sort_by_bucket(
        best_rows_within_bucket(
            exact_by_bucket, "dominance_bucket", "random_search_exact_config_rank_within_dominance_bucket"
        ),
        "dominance_bucket",
    )

    per_series_path = CSV_DIR / "grasynda_component_dominance_per_series.csv"
    profile_path = CSV_DIR / "grasynda_component_dominance_profile_by_dataset_group.csv"
    merged_runs_path = CSV_DIR / "grasynda_random_search_runs_with_component_dominance.csv"
    quantiles_by_bucket_path = CSV_DIR / "grasynda_random_search_quantile_by_dominance_bucket.csv"
    exact_by_bucket_path = CSV_DIR / "grasynda_random_search_exact_configuration_by_dominance_bucket.csv"
    best_quantile_by_bucket_path = CSV_DIR / "grasynda_random_search_best_quantile_by_dominance_bucket.csv"
    best_exact_by_bucket_path = CSV_DIR / "grasynda_random_search_best_exact_by_dominance_bucket.csv"

    per_series.to_csv(per_series_path, index=False)
    sort_by_bucket(profile, "dominance_bucket").to_csv(profile_path, index=False)
    sort_by_bucket(runs, "dominance_bucket").to_csv(merged_runs_path, index=False)
    quantiles_by_bucket.to_csv(quantiles_by_bucket_path, index=False)
    exact_by_bucket.to_csv(exact_by_bucket_path, index=False)
    best_quantile_by_bucket.to_csv(best_quantile_by_bucket_path, index=False)
    best_exact_by_bucket.to_csv(best_exact_by_bucket_path, index=False)

    print(f"Saved per-series dominance detail: {per_series_path}")
    print(f"Saved dataset-group dominance profile: {profile_path}")
    print(f"Saved merged random-search runs with dominance metadata: {merged_runs_path}")
    print(f"Saved quantile summary by dominance bucket: {quantiles_by_bucket_path}")
    print(f"Saved exact-configuration summary by dominance bucket: {exact_by_bucket_path}")
    print(f"Saved best quantile by dominance bucket: {best_quantile_by_bucket_path}")
    print(f"Saved best exact configuration by dominance bucket: {best_exact_by_bucket_path}")
    print("\nDominance profile:")
    print(sort_by_bucket(profile, "dominance_bucket").to_string(index=False))
    print("\nBest quantile by dominance bucket:")
    print(best_quantile_by_bucket.to_string(index=False))
    print("\nBest exact configuration by dominance bucket:")
    print(best_exact_by_bucket.to_string(index=False))


if __name__ == "__main__":
    main()
