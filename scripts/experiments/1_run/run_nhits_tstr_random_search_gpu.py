"""
Run paper-level NHITS TSTR random search for Grasynda and augmentation baselines.

This is intentionally configured from the block below, matching the rest of the
paper pipeline style. It samples 32 configurations per method/target by default,
trains NHITS for 1000 steps, evaluates TSTR MASE on the real held-out horizon,
and writes per-target *_best_config.json files consumed by
build_final_nhits_optimal_parameters.py.
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from neuralforecast import NeuralForecast
from utilsforecast.evaluation import evaluate
from utilsforecast.losses import mase


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import run_universal_experiments_gpu as universal
from src.grasynda_unified import GrasyndaUnified
from src.workflow_gpu import ExpWorkflow
from utils.augmentation_random_search_spaces import (
    DEFAULT_RANDOM_SEARCH_TRIALS,
    dba_search_space,
    grasynda_search_space,
    jittering_search_space,
    magnitudewarping_search_space,
    scaling_search_space,
    seasonalmbb_search_space,
    timevae_search_space,
    timewarping_search_space,
    tsdiff_search_space,
    tsmixup_search_space,
)
from utils.config_gpu import get_runtime_hardware_summary
from utils.load_data.base import LoadDataset
from utils.load_data.config import DATASETS
from utils.random_search_utils import (
    ALL_NHITS_TARGETS,
    build_unique_param_sets,
    params_key,
    stable_target_seed,
)


# =========================
# Configuration
# =========================

TARGETS = list(ALL_NHITS_TARGETS)

METHODS = [
    "Grasynda",
    "SeasonalMBB",
    "Jittering",
    "Scaling",
    "MagnitudeWarping",
    "TimeWarping",
    "DBA",
    "TSMixup",
    "TimeVAE",
    # "TSDiff",
]

FORECASTING_MODEL = "NHITS"
MAX_STEPS = 1000
N_TRIALS = DEFAULT_RANDOM_SEARCH_TRIALS
SEED = 42
AUGMENTATION_SAMPLES = 1

USE_MIN_SAMPLES_FILTER = True
MIN_SAMPLES_BY_FREQUENCY = {"monthly": 48, "quarterly": 16}

# False matches the release benchmark setup: one real train/test split, tune by
# TSTR MASE under the same train-on-synthetic/test-on-real protocol.
NESTED_VALIDATION = False

SKIP_FINISHED_TARGETS = True
STOP_ON_FIRST_ERROR = False
WRITE_FINAL_OPTIMAL_JSON = True
OUTPUT_ROOT = PROJECT_ROOT / "assets" / "results" / "random_search"


SEARCH_SPACES = {
    "Grasynda": grasynda_search_space,
    "SeasonalMBB": seasonalmbb_search_space,
    "Jittering": jittering_search_space,
    "Scaling": scaling_search_space,
    "MagnitudeWarping": magnitudewarping_search_space,
    "TimeWarping": timewarping_search_space,
    "DBA": dba_search_space,
    "TSMixup": tsmixup_search_space,
    "TimeVAE": timevae_search_space,
    "TSDiff": tsdiff_search_space,
}


# =========================
# Helpers
# =========================

def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def slugify_target(dataset: str, group: str) -> str:
    return f"{dataset}_{group}".replace(" ", "_").replace("/", "_").lower()


def method_folder(method: str) -> str:
    return method.lower()


def resolve_frequency_key(group: str) -> Optional[str]:
    lowered = str(group).strip().lower()
    if "monthly" in lowered:
        return "monthly"
    if "quarterly" in lowered:
        return "quarterly"
    return None


def resolve_horizon(group: str, fallback: int) -> int:
    freq_key = resolve_frequency_key(group)
    if freq_key == "monthly":
        return 12
    if freq_key == "quarterly":
        return 4
    return int(fallback)


def resolve_min_samples(group: str) -> Optional[int]:
    if not USE_MIN_SAMPLES_FILTER:
        return None
    freq_key = resolve_frequency_key(group)
    if freq_key is None:
        return None
    return int(MIN_SAMPLES_BY_FREQUENCY[freq_key])


def normalize_params(params: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(params)
    if "dba_iter" in out:
        out["max_iter"] = out.pop("dba_iter")

    int_like = {
        "knot",
        "max_samples_in_stl",
        "max_n_uids",
        "latent_dim",
        "max_epochs",
        "max_iter",
        "n_quantiles",
        "ensemble_size",
        "num_batches_per_epoch",
        "batch_size",
        "context_length",
        "prediction_length",
    }
    for key in int_like:
        if key in out and out[key] is not None:
            out[key] = int(round(float(out[key])))
    return out


def set_trial_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except Exception:
        pass


def load_target(dataset: str, group: str) -> Tuple[pd.DataFrame, int, int, str, int]:
    loader = DATASETS[dataset]
    load_kwargs = {}
    min_samples = resolve_min_samples(group)
    if min_samples is not None:
        load_kwargs["min_n_instances"] = min_samples
    df, loader_horizon, n_lags, freq_str, freq_int = loader.load_everything(group, **load_kwargs)
    horizon = resolve_horizon(group, loader_horizon)
    if df.empty:
        raise ValueError(f"{dataset} {group} is empty after min-length filtering.")
    return df, int(horizon), int(n_lags), str(freq_str), int(freq_int)


def split_target(df: pd.DataFrame, horizon: int):
    return LoadDataset.train_test_split(df, horizon)


def extract_synthetic_only(candidate_df: pd.DataFrame, real_df: pd.DataFrame) -> pd.DataFrame:
    real_ids = set(real_df["unique_id"].unique())
    synth_only = candidate_df[~candidate_df["unique_id"].isin(real_ids)].copy()
    if synth_only.empty and len(candidate_df) >= len(real_df):
        synth_only = candidate_df.iloc[len(real_df):].copy()
    return synth_only.sort_values(["unique_id", "ds"]).reset_index(drop=True)


def base_augmentation_params(
    train_df: pd.DataFrame,
    freq_int: int,
    freq_str: str,
    horizon: int,
) -> Dict[str, Any]:
    lengths = train_df.groupby("unique_id").size()
    n_uids = int(train_df["unique_id"].nunique())
    max_n_uids = max(2, int(round(np.log(max(n_uids, 2)), 0)))
    return {
        "seas_period": freq_int,
        "freq": freq_str,
        "max_n_uids": max_n_uids,
        "max_len": int(lengths.max()),
        "min_len": int(lengths.min()),
        "max_steps": MAX_STEPS,
        "window_size": int(max(horizon * 2, freq_int)),
        "context_length": int(max(horizon * 2, freq_int)),
        "prediction_length": int(horizon),
        "device": universal.TORCH_DEVICE,
        "require_gpu": universal.REQUIRE_GPU,
    }


def generate_grasynda_synthetic(
    train_df: pd.DataFrame,
    freq_int: int,
    params: Dict[str, Any],
) -> pd.DataFrame:
    generator = GrasyndaUnified(
        period=freq_int,
        n_quantiles=params.get("n_quantiles", 10),
        sampling_type="continuous_uniform",
        graph_type="quantile",
        components_to_model=["trend", "remainder"],
        component_params={"trend": {"apply_differentiation": True}},
        ensemble_transitions=params.get("ensemble_transitions", False),
        ensemble_size=params.get("ensemble_size"),
    )
    return extract_synthetic_only(generator.transform(train_df), train_df)


def generate_augmentation_synthetic(
    method: str,
    train_df: pd.DataFrame,
    params: Dict[str, Any],
) -> Tuple[pd.DataFrame, int, Dict[str, Any]]:
    method_params = normalize_params(params)
    n_series = AUGMENTATION_SAMPLES
    n_series_factor = method_params.pop("n_series_factor", None)
    if n_series_factor is not None:
        n_series = max(1, int(round(float(n_series) * float(n_series_factor))))

    if method == "DBA":
        method_params["max_n_uids"] = min(
            int(method_params.get("max_n_uids", 2)),
            int(train_df["unique_id"].nunique()),
        )

    augmented = ExpWorkflow.get_offline_augmented_data(
        train_=train_df,
        generator_name=method,
        augmentation_params=method_params,
        n_series_by_uid=n_series,
    )
    return extract_synthetic_only(augmented, train_df), n_series, method_params


def generate_synthetic(
    method: str,
    train_df: pd.DataFrame,
    freq_int: int,
    params: Dict[str, Any],
) -> Tuple[pd.DataFrame, int, Dict[str, Any]]:
    if method == "Grasynda":
        normalized = normalize_params(params)
        return generate_grasynda_synthetic(train_df, freq_int, normalized), 1, normalized
    return generate_augmentation_synthetic(method, train_df, params)


def compute_mase(
    fit_df: pd.DataFrame,
    predict_context_df: pd.DataFrame,
    test_df: pd.DataFrame,
    horizon: int,
    n_lags: int,
    freq_str: str,
    freq_int: int,
    alias: str,
) -> float:
    model_inst = universal.build_model_instance(
        model_name=FORECASTING_MODEL,
        horizon=horizon,
        n_lags=n_lags,
        max_steps=MAX_STEPS,
        alias=alias,
    )
    nf = NeuralForecast(models=[model_inst], freq=freq_str)
    nf.fit(df=fit_df, val_size=horizon)
    fcst = nf.predict(df=predict_context_df)
    fcst_out = fcst.reset_index()
    pred_cols = [col for col in fcst_out.columns if col not in {"unique_id", "ds"}]
    if len(pred_cols) != 1:
        raise ValueError(f"Expected exactly one prediction column, got {pred_cols}")
    pred_col = pred_cols[0]
    test_with_fcst = test_df.merge(fcst_out, on=["unique_id", "ds"], how="left")
    if test_with_fcst[pred_col].isna().any():
        missing = int(test_with_fcst[pred_col].isna().sum())
        raise ValueError(f"Forecast merge produced {missing} missing predictions.")
    eval_df = evaluate(
        test_with_fcst,
        [partial(mase, seasonality=freq_int)],
        train_df=predict_context_df,
    )
    return float(eval_df.query('metric=="mase"')[pred_col].mean())


def evaluate_baseline(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    horizon: int,
    n_lags: int,
    freq_str: str,
    freq_int: int,
    alias: str,
) -> float:
    model_inst = universal.build_model_instance(
        model_name=FORECASTING_MODEL,
        horizon=horizon,
        n_lags=n_lags,
        max_steps=MAX_STEPS,
        alias=alias,
    )
    nf = NeuralForecast(models=[model_inst], freq=freq_str)
    nf.fit(df=train_df, val_size=horizon)
    fcst = nf.predict()
    fcst_out = fcst.reset_index()
    pred_cols = [col for col in fcst_out.columns if col not in {"unique_id", "ds"}]
    pred_col = pred_cols[0]
    test_with_fcst = test_df.merge(fcst_out, on=["unique_id", "ds"], how="left")
    eval_df = evaluate(
        test_with_fcst,
        [partial(mase, seasonality=freq_int)],
        train_df=train_df,
    )
    return float(eval_df.query('metric=="mase"')[pred_col].mean())


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2), encoding="utf-8")


def run_target(method: str, dataset: str, group: str) -> Optional[Dict[str, Any]]:
    method_dir = OUTPUT_ROOT / method_folder(method)
    method_dir.mkdir(parents=True, exist_ok=True)
    target_slug = slugify_target(dataset, group)
    best_path = method_dir / f"{method_folder(method)}_nhits_random_search_{target_slug}_best_config.json"
    summary_path = method_dir / f"{method_folder(method)}_nhits_random_search_{target_slug}_summary.csv"

    if SKIP_FINISHED_TARGETS and best_path.exists():
        print(f"  Existing best config found, skipping: {best_path}")
        return json.loads(best_path.read_text(encoding="utf-8"))

    df, horizon, n_lags, freq_str, freq_int = load_target(dataset, group)
    outer_train, outer_test = split_target(df, horizon)

    if NESTED_VALIDATION:
        search_train, search_test = split_target(outer_train, horizon)
        holdout_train, holdout_test = outer_train, outer_test
    else:
        search_train, search_test = outer_train, outer_test
        holdout_train, holdout_test = outer_train, outer_test

    baseline_mase = evaluate_baseline(
        train_df=search_train,
        test_df=search_test,
        horizon=horizon,
        n_lags=n_lags,
        freq_str=freq_str,
        freq_int=freq_int,
        alias=f"baseline_{target_slug}",
    )

    search_space = SEARCH_SPACES[method](dataset, group)
    ordered_keys = list(search_space.keys())
    target_seed = stable_target_seed(dataset, group, SEED) + sum(ord(ch) for ch in method)
    param_sets = build_unique_param_sets(
        param_distributions=search_space,
        ordered_keys=ordered_keys,
        n_trials=N_TRIALS,
        seed=target_seed,
    )

    rows = []
    best_row = None
    start_target = time.time()
    print(
        f"  {method} | {dataset}-{group}: "
        f"{len(param_sets)} trials, horizon={horizon}, n_lags={n_lags}, "
        f"min_samples={resolve_min_samples(group)}"
    )
    print(f"  Baseline TRTR MASE for search split: {baseline_mase:.4f}")

    for trial_idx, sampled_params in enumerate(param_sets, start=1):
        trial_seed = target_seed + trial_idx
        set_trial_seed(trial_seed)
        start_trial = time.time()
        complete_params = base_augmentation_params(search_train, freq_int, freq_str, horizon)
        complete_params.update(sampled_params)
        complete_params = normalize_params(complete_params)
        trial_row = {
            "Method": method,
            "Dataset": dataset,
            "Group": group,
            "Forecast_Model": FORECASTING_MODEL,
            "Trial_Index": trial_idx,
            "Trial_Seed": trial_seed,
            "Status": "Success",
            "MASE": np.nan,
            "Baseline_CV_MASE": baseline_mase,
            "Params_JSON": json.dumps(json_safe(sampled_params), sort_keys=True),
            "Effective_Params_JSON": json.dumps(json_safe(complete_params), sort_keys=True),
            "N_Series_By_UID": AUGMENTATION_SAMPLES,
            "Elapsed_Sec": np.nan,
            "Error": "",
        }
        for key, value in sampled_params.items():
            trial_row[f"param_{key}"] = value

        try:
            synth_df, n_series, effective_params = generate_synthetic(
                method=method,
                train_df=search_train,
                freq_int=freq_int,
                params=complete_params,
            )
            if synth_df.empty:
                raise ValueError("Synthetic training set is empty.")
            trial_row["N_Series_By_UID"] = n_series
            trial_row["Effective_Params_JSON"] = json.dumps(json_safe(effective_params), sort_keys=True)
            mase_score = compute_mase(
                fit_df=synth_df,
                predict_context_df=search_train,
                test_df=search_test,
                horizon=horizon,
                n_lags=n_lags,
                freq_str=freq_str,
                freq_int=freq_int,
                alias=f"{method}_{target_slug}_trial{trial_idx}",
            )
            trial_row["MASE"] = mase_score
            print(f"    Trial {trial_idx:02d}/{len(param_sets)} MASE={mase_score:.4f} params={sampled_params}")
            if best_row is None or mase_score < best_row["MASE"]:
                best_row = dict(trial_row)
        except Exception as exc:
            trial_row["Status"] = "Failed"
            trial_row["Error"] = repr(exc)
            print(f"    Trial {trial_idx:02d}/{len(param_sets)} FAILED: {exc}")
            if STOP_ON_FIRST_ERROR:
                raise
        finally:
            trial_row["Elapsed_Sec"] = time.time() - start_trial
            rows.append(trial_row)
            pd.DataFrame(rows).to_csv(summary_path, index=False)

    if best_row is None:
        print(f"  No successful trials for {method} {dataset}-{group}.")
        return None

    best_params_for_experiment = json.loads(best_row["Params_JSON"])
    best_effective_params = json.loads(best_row["Effective_Params_JSON"])
    holdout_mase = float(best_row["MASE"])
    holdout_baseline_mase = baseline_mase

    if NESTED_VALIDATION:
        holdout_complete_params = base_augmentation_params(holdout_train, freq_int, freq_str, horizon)
        holdout_complete_params.update(best_params_for_experiment)
        holdout_complete_params = normalize_params(holdout_complete_params)
        synth_df, _, _ = generate_synthetic(
            method=method,
            train_df=holdout_train,
            freq_int=freq_int,
            params=holdout_complete_params,
        )
        holdout_baseline_mase = evaluate_baseline(
            train_df=holdout_train,
            test_df=holdout_test,
            horizon=horizon,
            n_lags=n_lags,
            freq_str=freq_str,
            freq_int=freq_int,
            alias=f"baseline_holdout_{target_slug}",
        )
        holdout_mase = compute_mase(
            fit_df=synth_df,
            predict_context_df=holdout_train,
            test_df=holdout_test,
            horizon=horizon,
            n_lags=n_lags,
            freq_str=freq_str,
            freq_int=freq_int,
            alias=f"{method}_{target_slug}_holdout_best",
        )

    payload = {
        "Method": method,
        "Dataset": dataset,
        "Group": group,
        "Forecast_Model": FORECASTING_MODEL,
        "Protocol": "TSTR",
        "Search_Trials_Configured": N_TRIALS,
        "Trials_Evaluated": int(sum(row["Status"] == "Success" for row in rows)),
        "Best_Trial_Index": int(best_row["Trial_Index"]),
        "Best_Params": best_params_for_experiment,
        "Best_Params_For_Experiment": best_params_for_experiment,
        "Best_Effective_Params": best_effective_params,
        "Best_CV_MASE": float(best_row["MASE"]),
        "Baseline_CV_MASE": float(baseline_mase),
        "Holdout_Best_MASE": float(holdout_mase),
        "Holdout_Baseline_MASE": float(holdout_baseline_mase),
        "Nested_Validation": bool(NESTED_VALIDATION),
        "Max_Steps": MAX_STEPS,
        "Min_Samples": resolve_min_samples(group),
        "Monthly_Min_Samples": MIN_SAMPLES_BY_FREQUENCY["monthly"],
        "Quarterly_Min_Samples": MIN_SAMPLES_BY_FREQUENCY["quarterly"],
        "Horizon": horizon,
        "NLags": n_lags,
        "Freq_Str": freq_str,
        "Freq_Int": freq_int,
        "Summary_Path": str(summary_path.relative_to(PROJECT_ROOT)),
        "Elapsed_Sec": time.time() - start_target,
        "Generated_At": datetime.now().isoformat(timespec="seconds"),
    }
    write_json(best_path, payload)
    print(f"  Best {method} {dataset}-{group}: MASE={payload['Best_CV_MASE']:.4f}")
    print(f"  Saved best config: {best_path}")
    return payload


def write_consolidated(method: str, rows: list[Dict[str, Any]]) -> Optional[Path]:
    if not rows:
        return None
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = OUTPUT_ROOT / method_folder(method) / f"{method_folder(method)}_best_configs_NHITS_{timestamp}.json"
    write_json(path, rows)
    print(f"Saved consolidated {method} best configs: {path}")
    return path


def rebuild_final_optimal_json_direct() -> None:
    if not WRITE_FINAL_OPTIMAL_JSON:
        return
    script_path = PROJECT_ROOT / "scripts" / "experiments" / "1_run" / "build_final_nhits_optimal_parameters.py"
    namespace: Dict[str, Any] = {
        "__file__": str(script_path),
        "__name__": "_build_final_nhits_optimal_parameters",
    }
    exec(script_path.read_text(encoding="utf-8"), namespace)
    namespace["main"]()


def main() -> None:
    runtime = get_runtime_hardware_summary()
    print("NHITS TSTR random search")
    print(f"  Model: {FORECASTING_MODEL}")
    print(f"  Max steps: {MAX_STEPS}")
    print(f"  Trials per target: {N_TRIALS}")
    print(f"  Targets: {len(TARGETS)}")
    print(f"  Methods: {', '.join(METHODS)}")
    print(f"  Min samples: monthly={MIN_SAMPLES_BY_FREQUENCY['monthly']}, quarterly={MIN_SAMPLES_BY_FREQUENCY['quarterly']}")
    print(f"  Nested validation: {NESTED_VALIDATION}")
    print(f"  Torch CUDA available: {runtime['torch_cuda_available']}")
    print(f"  Torch CUDA device count: {runtime['torch_cuda_device_count']}")
    universal.assert_runtime_ready_for_requested_accelerator()

    all_best_by_method: Dict[str, list[Dict[str, Any]]] = {method: [] for method in METHODS}
    for method in METHODS:
        print(f"\n{'=' * 100}")
        print(f"METHOD: {method}")
        print("=" * 100)
        if method not in SEARCH_SPACES:
            raise ValueError(f"No search space registered for method: {method}")
        for dataset, group in TARGETS:
            print(f"\n### {method} | {dataset} - {group} ###")
            result = run_target(method, dataset, group)
            if result is not None:
                all_best_by_method[method].append(result)
        write_consolidated(method, all_best_by_method[method])

    rebuild_final_optimal_json_direct()
    print("\n### DONE ###")
    print(f"Random-search root: {OUTPUT_ROOT}")
    print(f"Final optimal JSON: {OUTPUT_ROOT / 'nhits_optimal_parameters_final.json'}")


if __name__ == "__main__":
    main()
