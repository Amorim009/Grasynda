import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = REPO_ROOT / "assets" / "results" / "grasynda_sensitivity"
CSV_DIR = OUTPUT_DIR / "csv"
GRID_DIR = REPO_ROOT / "assets" / "results" / "grid_search_grasy"
RANDOM_DIR = REPO_ROOT / "assets" / "results" / "random_search" / "grasynda"


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CSV_DIR.mkdir(parents=True, exist_ok=True)


def parse_target_from_name(name: str, prefix: str) -> Tuple[str, str]:
    stem = name
    if stem.endswith("_summary.csv"):
        stem = stem[: -len("_summary.csv")]
    if not stem.startswith(prefix):
        raise ValueError(f"Unexpected filename for prefix '{prefix}': {name}")

    remainder = stem[len(prefix) :]
    if remainder.endswith("_checkpoint"):
        remainder = remainder[: -len("_checkpoint")]

    tokens = remainder.split("_")
    split_idx = None
    for idx, token in enumerate(tokens):
        if re.fullmatch(r"\d{8}", token):
            split_idx = idx
            break
    if split_idx is not None:
        tokens = tokens[:split_idx]

    if not tokens:
        raise ValueError(f"Could not parse dataset/group from filename: {name}")

    dataset = tokens[0]
    group_tokens = tokens[1:]
    if not group_tokens:
        raise ValueError(f"Could not parse group from filename: {name}")

    if dataset in {"M3", "Tourism", "NN3"}:
        group = " ".join(group_tokens)
    else:
        group = "_".join(group_tokens)
    return dataset, group


def normalize_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def normalize_ensemble_size(value, ensemble_transitions: bool) -> Optional[int]:
    if not ensemble_transitions:
        return None
    if pd.isna(value):
        return None
    return int(float(value))


def load_summary(path: Path, source_family: str, prefix: Optional[str] = None) -> pd.DataFrame:
    df = pd.read_csv(path)

    if "Dataset" in df.columns and "Group" in df.columns:
        dataset = str(df["Dataset"].iloc[0])
        group = str(df["Group"].iloc[0])
    else:
        if prefix is None:
            raise ValueError(f"Missing filename prefix for parsing dataset/group: {path}")
        dataset, group = parse_target_from_name(path.name, prefix)

    out = df.copy()
    out["Dataset"] = dataset
    out["Group"] = group
    out["source_family"] = source_family
    out["source_file"] = str(path.relative_to(REPO_ROOT))
    out["file_mtime"] = path.stat().st_mtime

    if "mean_cv_mase" in out.columns:
        out["forecast_mase"] = out["mean_cv_mase"]
        out["metric_name"] = "mean_cv_mase"
    elif "mean_augmented_mase" in out.columns:
        out["forecast_mase"] = out["mean_augmented_mase"]
        out["metric_name"] = "mean_augmented_mase"
    else:
        raise ValueError(f"Unsupported summary columns in {path}")

    if "std_cv_mase" not in out.columns:
        out["std_cv_mase"] = pd.NA
    if "mean_baseline_mase" not in out.columns:
        out["mean_baseline_mase"] = pd.NA
    if "mean_mase_gain" not in out.columns:
        out["mean_mase_gain"] = pd.NA
    if "trial_id" not in out.columns:
        out["trial_id"] = pd.NA

    out["ensemble_transitions"] = out["ensemble_transitions"].map(normalize_bool)
    out["ensemble_size"] = [
        normalize_ensemble_size(v, flag)
        for v, flag in zip(out["ensemble_size"], out["ensemble_transitions"])
    ]
    out["n_quantiles"] = out["n_quantiles"].astype(int)
    out["ensemble_label"] = out["ensemble_transitions"].map({False: "No Ensemble", True: "Ensemble"})
    out["config_label"] = out.apply(
        lambda row: (
            f"q={int(row['n_quantiles'])}, no-ens"
            if not row["ensemble_transitions"]
            else f"q={int(row['n_quantiles'])}, ens={int(row['ensemble_size'])}"
        ),
        axis=1,
    )
    return out[
        [
            "Dataset",
            "Group",
            "source_family",
            "source_file",
            "file_mtime",
            "metric_name",
            "forecast_mase",
            "std_cv_mase",
            "mean_baseline_mase",
            "mean_mase_gain",
            "n_quantiles",
            "ensemble_transitions",
            "ensemble_size",
            "ensemble_label",
            "config_label",
            "trial_id",
        ]
    ]


def collect_latest_summaries() -> pd.DataFrame:
    frames: List[pd.DataFrame] = []

    for path in GRID_DIR.glob("grasynda_gridsearch_*_summary.csv"):
        frames.append(load_summary(path, source_family="legacy_grid"))

    for path in GRID_DIR.glob("grasynda_nhits_grid_*_summary.csv"):
        if "checkpoint" in path.name.lower():
            continue
        frames.append(load_summary(path, source_family="nhits_grid", prefix="grasynda_nhits_grid_"))

    latest_random: Dict[Tuple[str, str], Path] = {}
    for path in RANDOM_DIR.glob("grasynda_nhits_random_search_*_summary.csv"):
        if "checkpoint" in path.name.lower():
            continue
        dataset, group = parse_target_from_name(path.name, "grasynda_nhits_random_search_")
        key = (dataset, group)
        prev = latest_random.get(key)
        if prev is None or path.stat().st_mtime > prev.stat().st_mtime:
            latest_random[key] = path
    for path in latest_random.values():
        frames.append(load_summary(path, source_family="random_search", prefix="grasynda_nhits_random_search_"))

    if not frames:
        raise ValueError("No Grasynda summary files found.")

    merged = pd.concat(frames, axis=0, ignore_index=True)
    merged["rank_within_source_dataset_group"] = (
        merged.groupby(["source_family", "Dataset", "Group"])["forecast_mase"]
        .rank(method="dense", ascending=True)
        .astype(int)
    )
    merged["rank_within_dataset_group_all_sources"] = (
        merged.groupby(["Dataset", "Group"])["forecast_mase"]
        .rank(method="dense", ascending=True)
        .astype(int)
    )
    return merged.sort_values(
        ["source_family", "Dataset", "Group", "forecast_mase"],
        ascending=[True, True, True, True],
    ).reset_index(drop=True)


def build_configuration_ranking(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["n_quantiles", "ensemble_transitions", "ensemble_size", "config_label"]
    ranking = (
        df.groupby(group_cols, dropna=False)
        .agg(
            n_runs=("forecast_mase", "size"),
            mean_forecast_mase=("forecast_mase", "mean"),
            median_forecast_mase=("forecast_mase", "median"),
            mean_rank_within_dataset_group=("rank_within_dataset_group_all_sources", "mean"),
            mean_rank_within_source_dataset_group=("rank_within_source_dataset_group", "mean"),
        )
        .reset_index()
    )
    coverage = (
        df[group_cols + ["Dataset", "Group"]]
        .drop_duplicates()
        .groupby(group_cols, dropna=False)
        .size()
        .reset_index(name="n_dataset_groups")
    )
    ranking = (
        ranking.merge(coverage, on=group_cols, how="left")
        .sort_values(
            ["mean_rank_within_dataset_group", "mean_forecast_mase", "n_dataset_groups", "n_runs"],
            ascending=[True, True, False, False],
        )
        .reset_index(drop=True)
    )
    ranking.insert(0, "configuration_rank", range(1, len(ranking) + 1))
    return ranking


def build_parameter_choice_ranking(df: pd.DataFrame) -> pd.DataFrame:
    parts = []

    n_quant = (
        df.groupby("n_quantiles")
        .agg(
            n_runs=("forecast_mase", "size"),
            mean_forecast_mase=("forecast_mase", "mean"),
            median_forecast_mase=("forecast_mase", "median"),
            mean_rank_within_dataset_group=("rank_within_dataset_group_all_sources", "mean"),
            mean_rank_within_source_dataset_group=("rank_within_source_dataset_group", "mean"),
        )
        .reset_index()
        .rename(columns={"n_quantiles": "parameter_value"})
    )
    n_quant["parameter_name"] = "n_quantiles"
    parts.append(n_quant)

    ensemble_flag = (
        df.groupby("ensemble_transitions")
        .agg(
            n_runs=("forecast_mase", "size"),
            mean_forecast_mase=("forecast_mase", "mean"),
            median_forecast_mase=("forecast_mase", "median"),
            mean_rank_within_dataset_group=("rank_within_dataset_group_all_sources", "mean"),
            mean_rank_within_source_dataset_group=("rank_within_source_dataset_group", "mean"),
        )
        .reset_index()
        .rename(columns={"ensemble_transitions": "parameter_value"})
    )
    ensemble_flag["parameter_name"] = "ensemble_transitions"
    parts.append(ensemble_flag)

    ensemble_size = (
        df[df["ensemble_transitions"]]
        .groupby("ensemble_size")
        .agg(
            n_runs=("forecast_mase", "size"),
            mean_forecast_mase=("forecast_mase", "mean"),
            median_forecast_mase=("forecast_mase", "median"),
            mean_rank_within_dataset_group=("rank_within_dataset_group_all_sources", "mean"),
            mean_rank_within_source_dataset_group=("rank_within_source_dataset_group", "mean"),
        )
        .reset_index()
        .rename(columns={"ensemble_size": "parameter_value"})
    )
    ensemble_size["parameter_name"] = "ensemble_size_when_enabled"
    parts.append(ensemble_size)

    out = pd.concat(parts, axis=0, ignore_index=True)
    out = out.sort_values(
        ["parameter_name", "mean_rank_within_dataset_group", "mean_forecast_mase"],
        ascending=[True, True, True],
    ).reset_index(drop=True)
    out["parameter_rank"] = (
        out.groupby("parameter_name")["mean_rank_within_dataset_group"]
        .rank(method="dense", ascending=True)
        .astype(int)
    )
    return out[
        [
            "parameter_name",
            "parameter_value",
            "parameter_rank",
            "n_runs",
            "mean_forecast_mase",
            "median_forecast_mase",
            "mean_rank_within_dataset_group",
            "mean_rank_within_source_dataset_group",
        ]
    ]


def main() -> None:
    ensure_output_dir()
    merged = collect_latest_summaries()
    config_ranking = build_configuration_ranking(merged)
    choice_ranking = build_parameter_choice_ranking(merged)

    merged_path = CSV_DIR / "grasynda_parameter_sensitivity_runs.csv"
    config_path = CSV_DIR / "grasynda_configuration_ranking.csv"
    choice_path = CSV_DIR / "grasynda_parameter_choice_ranking.csv"

    merged.to_csv(merged_path, index=False)
    config_ranking.to_csv(config_path, index=False)
    choice_ranking.to_csv(choice_path, index=False)

    print(f"Saved merged runs: {merged_path}")
    print(f"Saved configuration ranking: {config_path}")
    print(f"Saved parameter choice ranking: {choice_path}")
    print("\nTop configurations:")
    print(config_ranking.head(10).to_string(index=False))
    print("\nTop parameter choices:")
    print(choice_ranking.groupby("parameter_name").head(5).to_string(index=False))


if __name__ == "__main__":
    main()
