"""
Universal experiment runner.

Results:
- assets/results/systematic_evaluation_mase_results.csv
- assets/results/systematic_eval_{MODEL}_{MODE}_Results.csv
"""

import os
import sys
import glob
import json
from copy import deepcopy
from datetime import datetime
from functools import partial
from typing import Any, Dict

import numpy as np
import pandas as pd
from neuralforecast import NeuralForecast
from utilsforecast.evaluation import evaluate
from utilsforecast.losses import mase

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.workflow import ExpWorkflow
from utils.config import AUTO_MODELS, MODEL_CONFIG, MODELS
from utils.load_data.base import LoadDataset
from utils.load_data.config import DATASETS

# =========================
# Configuration
# =========================

DATASETS_TO_TEST = [
    ("Gluonts", "m1_monthly"),
    ("Gluonts", "m1_quarterly"),
    ("M3", "Monthly"),
    ("M3", "Quarterly"),
    ("NN3", "Monthly"),
    ("Tourism", "Monthly"),
    ("Tourism", "Quarterly"),
]

BASELINE_METHOD = [
    "Baseline",
]

OTHER_METHODS = [
    # "TSMixup_Optimized",
    # "SeasonalMBB",
    # "Jittering",
    # "TimeWarping",
    # "MagnitudeWarping",
    # "MagnitudeWarping_Optimized",
    # "TSMixup",
    # "TSMixup_Optimized",
    # "TimeVAE",
    # "TimeVAE_Optimized",
    # "TSDiff",
    # "TSDiff_Optimized",
]

GRASYNDA_METHODS = [
    "Hybrid_Q10_ShortSeriesEnsemble5_Continuous",  # ensemble only for short series
    # "Hybrid_Q10_Ensemble5_Continuous",
    # "Hybrid_Q10_NoEnsemble_Continuous",
    # "Grasynda_Optimized",
    # "Hybrid_Q10_NoEnsemble_RawY",
    # "Hybrid_Q10_NoEnsemble_NoDiff",  # trend+remainder, no trend differentiation
    # "Hybrid_SAX_Q10_Ensemble5_Continuous",
    # "Hybrid_SAX_Q10_NoEnsemble_Continuous",
    # "Hybrid_SAX_Q10_Ensemble5_RawY",
    # "Hybrid_SAX_Q10_NoEnsemble_RawY",
    # "Hybrid_VisH_Ensemble10",
]

FORECASTING_MODELS = [
    "NHITS",
    # "MLP",
    # "KAN",
]

TRAINING_MODES = [
    "Train+Real",
    "TSTR",
]

MAX_STEPS = 1000
AUGMENTATION_SAMPLES = 1
# Easy toggle: set to False to run on all series without min-length filtering.
USE_MIN_SAMPLES_FILTER = True
MIN_SAMPLES_BY_FREQUENCY = {
    "monthly": 48,
}
RESULTS_TAG = "all_datasets_baseline_grasynda_shortseries_ensemble5_nhits_cpu_20260428"
FINAL_NHITS_OPTIMAL_PARAMS_PATH = os.path.join(
    "assets",
    "results",
    "random_search",
    "nhits_optimal_parameters_final.json",
)


# --- Grasynda ---
DEFAULT_GRASYNDA_PARAMS = {
    "n_quantiles": 10,
    "ensemble_transitions": False,
    "ensemble_size": None,
}
OPTIMAL_GRASYNDA_PARAMS = {}

# --- SeasonalMBB ---
DEFAULT_SEASONAL_MBB_PARAMS = {
    "log": True,
    "max_samples_in_stl": None,
}
OPTIMAL_SEASONAL_MBB_PARAMS = {}

# --- Jittering ---
DEFAULT_JITTERING_PARAMS = {
    "sigma": 0.03,
}
OPTIMAL_JITTERING_PARAMS = {}

# --- Scaling ---
DEFAULT_SCALING_PARAMS = {
    "sigma": 0.1,
}
OPTIMAL_SCALING_PARAMS = {}

# --- TimeWarping ---
DEFAULT_TIMEWARPING_PARAMS = {
    "sigma": 0.2,
    "knot": 4,
}
OPTIMAL_TIMEWARPING_PARAMS = {}

# --- TSMixup ---
DEFAULT_TSMIXUP_PARAMS = {
    "max_n_uids": 3,
    "dirichlet_alpha": 1.0,
}
OPTIMAL_TSMIXUP_PARAMS = {
}

# --- TimeVAE ---
DEFAULT_TIMEVAE_PARAMS = {
    "latent_dim": 8,
    "reconstruction_wt": 3.0,
    "max_epochs": 100,
}
OPTIMAL_TIMEVAE_PARAMS = {
}

# --- DBA ---
DEFAULT_DBA_PARAMS = {
    "max_n_uids": 2,
    "dirichlet_alpha": 1.0,
    "max_iter": 10,
}
OPTIMAL_DBA_PARAMS = {
}

# --- MagnitudeWarping ---
DEFAULT_MAGNITUDE_WARPING_PARAMS = {
    "sigma": 0.2,
    "knot": 4,
}
OPTIMAL_MAGNITUDE_WARPING_PARAMS = {
}

# --- TSDiff ---
DEFAULT_TSDIFF_PARAMS = {
    "twin_noise_level": 0.08,
    "max_epochs": 20,
    "learning_rate": 1e-3,
}
OPTIMAL_TSDIFF_PARAMS = {
}


# =========================
# Helper functions
# =========================

def load_latest_random_search_best_params(
    method_name: str,
    default_params: Dict[tuple, Dict[str, Any]],
) -> Dict[tuple, Dict[str, Any]]:
    method_dir = os.path.join("assets", "results", "random_search", method_name.lower())
    pattern = os.path.join(method_dir, "*_best_configs_NHITS_*.json")
    candidates = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    if not candidates:
        return default_params

    latest_path = candidates[0]
    try:
        with open(latest_path, "r", encoding="utf-8") as handle:
            rows = json.load(handle)
    except Exception as exc:
        print(f"Warning: could not read random-search config file {latest_path}: {exc}")
        return default_params

    merged = dict(default_params)
    for row in rows:
        dataset = row.get("Dataset")
        group = row.get("Group")
        best_params = row.get("Best_Params_For_Experiment") or row.get("Best_Params")
        if dataset is None or group is None or not isinstance(best_params, dict):
            continue
        merged[(dataset, group)] = best_params

    print(f"Loaded optimized {method_name} params from {latest_path}")
    return merged


def load_final_nhits_best_params(
    method_name: str,
    default_params: Dict[tuple, Dict[str, Any]],
) -> Dict[tuple, Dict[str, Any]]:
    if not os.path.exists(FINAL_NHITS_OPTIMAL_PARAMS_PATH):
        return load_latest_random_search_best_params(method_name.lower(), default_params)

    try:
        with open(FINAL_NHITS_OPTIMAL_PARAMS_PATH, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception as exc:
        print(
            f"Warning: could not read final NHITS optimal-parameter file "
            f"{FINAL_NHITS_OPTIMAL_PARAMS_PATH}: {exc}"
        )
        return load_latest_random_search_best_params(method_name.lower(), default_params)

    method_payload = payload.get("Methods", {}).get(method_name, {})
    targets = method_payload.get("Targets", [])
    if not targets:
        return load_latest_random_search_best_params(method_name.lower(), default_params)

    merged = dict(default_params)
    for row in targets:
        dataset = row.get("Dataset")
        group = row.get("Group")
        params = row.get("Params")
        if dataset is None or group is None or not isinstance(params, dict):
            continue
        merged[(dataset, group)] = params

    print(f"Loaded final optimized {method_name} params from {FINAL_NHITS_OPTIMAL_PARAMS_PATH}")
    return merged


OPTIMAL_GRASYNDA_PARAMS = load_final_nhits_best_params("Grasynda", OPTIMAL_GRASYNDA_PARAMS)
OPTIMAL_SEASONAL_MBB_PARAMS = load_final_nhits_best_params("SeasonalMBB", OPTIMAL_SEASONAL_MBB_PARAMS)
OPTIMAL_JITTERING_PARAMS = load_final_nhits_best_params("Jittering", OPTIMAL_JITTERING_PARAMS)
OPTIMAL_SCALING_PARAMS = load_final_nhits_best_params("Scaling", OPTIMAL_SCALING_PARAMS)
OPTIMAL_TIMEWARPING_PARAMS = load_final_nhits_best_params("TimeWarping", OPTIMAL_TIMEWARPING_PARAMS)
OPTIMAL_TSMIXUP_PARAMS = load_final_nhits_best_params("TSMixup", OPTIMAL_TSMIXUP_PARAMS)
OPTIMAL_TIMEVAE_PARAMS = load_final_nhits_best_params("TimeVAE", OPTIMAL_TIMEVAE_PARAMS)
OPTIMAL_DBA_PARAMS = load_final_nhits_best_params("DBA", OPTIMAL_DBA_PARAMS)
OPTIMAL_MAGNITUDE_WARPING_PARAMS = load_final_nhits_best_params(
    "MagnitudeWarping",
    OPTIMAL_MAGNITUDE_WARPING_PARAMS,
)
OPTIMAL_TSDIFF_PARAMS = load_final_nhits_best_params("TSDiff", OPTIMAL_TSDIFF_PARAMS)

def build_hybrid_q10_generator(freq_int: int, apply_differentiation: bool, ensemble_transitions: bool, ensemble_size):
    from src.grasynda_unified import GrasyndaUnified

    return GrasyndaUnified(
        period=freq_int,
        n_quantiles=10,
        sampling_type="continuous_uniform",
        graph_type="quantile",
        components_to_model=["trend", "remainder"],
        component_params={"trend": {"apply_differentiation": apply_differentiation}},
        ensemble_transitions=ensemble_transitions,
        ensemble_size=ensemble_size,
    )


def resolve_frequency_key(group: str):
    g = str(group).strip().lower()
    if "monthly" in g:
        return "monthly"
    if "quarterly" in g:
        return "quarterly"
    return None


def resolve_horizon(group: str, fallback: int) -> int:
    freq_key = resolve_frequency_key(group)
    if freq_key == "monthly":
        return 12
    if freq_key == "quarterly":
        return 4
    return fallback


def resolve_min_samples(group: str):
    if not USE_MIN_SAMPLES_FILTER:
        return None
    freq_key = resolve_frequency_key(group)
    if freq_key == "monthly":
        return MIN_SAMPLES_BY_FREQUENCY["monthly"]
    if freq_key == "quarterly":
        return 4 * resolve_horizon(group, 4)
    return None


def get_short_series_uids(train_df: pd.DataFrame, horizon: int, group: str):
    freq_key = resolve_frequency_key(group)
    if freq_key == "monthly":
        threshold = MIN_SAMPLES_BY_FREQUENCY["monthly"]
    elif freq_key == "quarterly":
        threshold = 4 * horizon
    else:
        threshold = None
    if threshold is None:
        return [], [], None

    total_lengths = train_df.groupby("unique_id").size() + horizon
    short_uids = total_lengths[total_lengths < threshold].index.tolist()
    regular_uids = total_lengths[total_lengths >= threshold].index.tolist()
    return short_uids, regular_uids, threshold


def pick_synthetic_uids(synth_df: pd.DataFrame, original_uids, alias: str) -> pd.DataFrame:
    if not original_uids:
        return synth_df.iloc[0:0].copy()
    synthetic_uids = {f"{alias}_{uid}" for uid in original_uids}
    return synth_df[synth_df["unique_id"].isin(synthetic_uids)].copy()


def normalize_augmentation_params(params: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(params)
    int_like_keys = {
        "knot",
        "max_samples_in_stl",
        "max_n_uids",
        "latent_dim",
        "max_epochs",
        "max_iter",
        "dba_iter",
        "n_quantiles",
        "ensemble_size",
    }
    for key in int_like_keys:
        if key in normalized and normalized[key] is not None:
            normalized[key] = int(round(normalized[key]))
    return normalized


def generate_grasynda_data(method_name, train_df, freq_int, group, horizon, data_name):
    from src.grasynda_unified import GrasyndaUnified
    from src.grasynda_sax import GrasyndaSAX

    if method_name == "Grasynda_Optimized":
        params = DEFAULT_GRASYNDA_PARAMS.copy()
        params.update(OPTIMAL_GRASYNDA_PARAMS.get((data_name, group), {}))
        params = normalize_augmentation_params(params)
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
        return generator.transform(train_df)

    if method_name == "Hybrid_Q10_ShortSeriesEnsemble5_Continuous":
        short_uids, regular_uids, threshold = get_short_series_uids(train_df, horizon, group)
        print(
            f"    Short-series split: {len(short_uids)} short series (<{threshold}) and "
            f"{len(regular_uids)} non-short series (>= {threshold})."
        )
        regular_synth = build_hybrid_q10_generator(
            freq_int=freq_int,
            apply_differentiation=False,
            ensemble_transitions=False,
            ensemble_size=None,
        ).transform(train_df)

        if not short_uids:
            print("    No short series found for conditional ensemble; using regular Grasynda for all series.")
            return regular_synth

        ensemble_synth = build_hybrid_q10_generator(
            freq_int=freq_int,
            apply_differentiation=False,
            ensemble_transitions=True,
            ensemble_size=5,
        ).transform(train_df)
        synth_df = pd.concat(
            [
                pick_synthetic_uids(ensemble_synth, short_uids, alias="GrasyndaUnified"),
                pick_synthetic_uids(regular_synth, regular_uids, alias="GrasyndaUnified"),
            ],
            ignore_index=True,
        ).sort_values(["unique_id", "ds"]).reset_index(drop=True)
        print(
            f"    Conditional ensemble applied: {len(short_uids)} short series use ensemble=5; "
            f"{len(regular_uids)} non-short series use regular Grasynda. "
            f"Both branches model trend+remainder with no trend differentiation."
        )
        return synth_df

    if method_name == "Hybrid_Q10_Ensemble5_Continuous":
        generator = build_hybrid_q10_generator(
            freq_int=freq_int,
            apply_differentiation=True,
            ensemble_transitions=True,
            ensemble_size=5,
        )
        return generator.transform(train_df)

    if method_name == "Hybrid_Q10_NoEnsemble_Continuous":
        generator = build_hybrid_q10_generator(
            freq_int=freq_int,
            apply_differentiation=True,
            ensemble_transitions=False,
            ensemble_size=None,
        )
        return generator.transform(train_df)

    if method_name == "Hybrid_Q10_NoEnsemble_RawY":
        generator = GrasyndaUnified(
            period=freq_int,
            n_quantiles=10,
            sampling_type="continuous_uniform",
            graph_type="quantile",
            components_to_model=["y"],  # raw-y: no STL decomposition
            ensemble_transitions=False,
            ensemble_size=None,
        )
        return generator.transform(train_df)

    if method_name == "Hybrid_Q10_NoEnsemble_NoDiff":
        generator = GrasyndaUnified(
            period=freq_int,
            n_quantiles=10,
            sampling_type="continuous_uniform",
            graph_type="quantile",
            components_to_model=["trend", "remainder"],
            component_params={"trend": {"apply_differentiation": False}},
            ensemble_transitions=False,
            ensemble_size=None,
        )
        return generator.transform(train_df)

    if method_name == "Hybrid_SAX_Q10_Ensemble5_Continuous":
        generator = GrasyndaSAX(
            period=freq_int,
            n_symbols=10,
            n_sax_windows=1,
            sax_normalize=True,
            sampling_type="continuous_uniform",
            graph_type="quantile",
            components_to_model=["trend", "remainder"],
            component_params={"trend": {"apply_differentiation": True}},
            ensemble_transitions=True,
            ensemble_size=5,
        )
        return generator.transform(train_df)

    if method_name == "Hybrid_SAX_Q10_NoEnsemble_Continuous":
        generator = GrasyndaSAX(
            period=freq_int,
            n_symbols=10,
            n_sax_windows=1,
            sax_normalize=True,
            sampling_type="continuous_uniform",
            graph_type="quantile",
            components_to_model=["trend", "remainder"],
            component_params={"trend": {"apply_differentiation": True}},
            ensemble_transitions=False,
            ensemble_size=None,
        )
        return generator.transform(train_df)

    if method_name == "Hybrid_SAX_Q10_Ensemble5_RawY":
        generator = GrasyndaSAX(
            period=freq_int,
            n_symbols=10,
            n_sax_windows=1,
            sax_normalize=True,
            sampling_type="continuous_uniform",
            graph_type="quantile",
            components_to_model=["y"],  # raw-y: no STL decomposition
            ensemble_transitions=True,
            ensemble_size=5,
        )
        return generator.transform(train_df)

    if method_name == "Hybrid_SAX_Q10_NoEnsemble_RawY":
        generator = GrasyndaSAX(
            period=freq_int,
            n_symbols=10,
            n_sax_windows=1,
            sax_normalize=True,
            sampling_type="continuous_uniform",
            graph_type="quantile",
            components_to_model=["y"],  # raw-y: no STL decomposition
            ensemble_transitions=False,
            ensemble_size=None,
        )
        return generator.transform(train_df)

    if method_name == "Hybrid_VisH_Ensemble10":
        generator = GrasyndaUnified(
            period=freq_int,
            sampling_type="discrete",
            graph_type="visibility",
            visibility_type="horizontal",
            components_to_model=["trend", "remainder"],
            component_params={
                "trend": {"apply_differentiation": True, "sampling_type": "discrete"},
                "remainder": {
                    "sampling_type": "discrete",
                    "graph_type": "visibility",
                    "visibility_type": "horizontal",
                },
            },
            ensemble_transitions=True,
            ensemble_size=10,
        )
        return generator.transform(train_df)

    raise ValueError(f"Unknown Grasynda method: {method_name}")


def generate_other_augmentation_data(method_name, train_df, augmentation_params, n_series):
    return ExpWorkflow.get_offline_augmented_data(
        train_=train_df,
        generator_name=method_name,
        augmentation_params=augmentation_params,
        n_series_by_uid=n_series,
    )


def extract_synthetic_only(train_aug, real_train):
    """
    Given an augmented dataframe (Real + Synth), return only the synthetic part.
    Assumes synthetic data was appended to real data.
    """
    return train_aug.iloc[len(real_train):].reset_index(drop=True)


def build_model_instance(model_name, horizon, n_lags, max_steps, alias):
    model_cfg = deepcopy(MODEL_CONFIG.get(model_name, {}))
    if model_name == "AutoNHITS":
        return MODELS[model_name](
            h=horizon,
            config=None,
            num_samples=5,
            backend="optuna",
        )

    if model_name in AUTO_MODELS:
        return MODELS[model_name](
            h=horizon,
            config=None,
            num_samples=model_cfg.get("num_samples", 10),
            backend=model_cfg.get("backend", "optuna"),
            alias=alias,
        )

    model_params = {
        k: v
        for k, v in model_cfg.items()
        if k not in {"auto", "backend", "num_samples", "config"}
    }
    model_params["max_steps"] = model_params.get("max_steps", max_steps)

    return MODELS[model_name](
        input_size=n_lags,
        h=horizon,
        alias=alias,
        **model_params,
    )


# =========================
# Main loop
# =========================

def run_universal_experiments():
    print("=" * 100)
    print("UNIVERSAL TIME SERIES FORECASTING EXPERIMENT SUITE")
    print("=" * 100)
    print("\nConfiguration:")
    print(f"  Datasets: {len(DATASETS_TO_TEST)}")
    print(f"  Baseline Method: {len(BASELINE_METHOD)}")
    print(f"  Grasynda Methods: {len(GRASYNDA_METHODS)}")
    print(f"  Other Methods: {len(OTHER_METHODS)}")
    print(f"  Forecasting Models: {len(FORECASTING_MODELS)}")
    print(f"  Training Modes: {len(TRAINING_MODES)}")
    print(f"  Min-length filter enabled: {USE_MIN_SAMPLES_FILTER}")
    print(
        "  Min-length thresholds (total observations): "
        f"monthly>={MIN_SAMPLES_BY_FREQUENCY['monthly']}, "
        "quarterly>=4*horizon (=16 for current quarterly horizon=4)"
    )
    print(
        f"\n  Total Experiments: "
        f"{len(DATASETS_TO_TEST) * (len(BASELINE_METHOD) + len(GRASYNDA_METHODS) + len(OTHER_METHODS)) * len(FORECASTING_MODELS)}"
    )
    print("=" * 100)

    results_path = f"assets/results/systematic_evaluation_{RESULTS_TAG}.csv"
    all_results = []
    experiment_count = 0
    start_time = datetime.now()

    for dataset_idx, (data_name, group) in enumerate(DATASETS_TO_TEST):
        print(f"\n{'=' * 100}")
        print(f"DATASET {dataset_idx + 1}/{len(DATASETS_TO_TEST)}: {data_name} - {group}")
        print("=" * 100)

        data_loader = DATASETS[data_name]
        min_samples = resolve_min_samples(group)
        load_kwargs = {}
        if min_samples is not None:
            load_kwargs["min_n_instances"] = min_samples
        df, loader_horizon, n_lags, freq_str, freq_int = data_loader.load_everything(group, **load_kwargs)
        horizon = resolve_horizon(group, loader_horizon)

        if df.empty or df["unique_id"].nunique() == 0:
            print("  Warning: dataset is empty after min-length filtering; skipping.")
            continue

        print(f"  Data: {df.shape}, Unique IDs: {df['unique_id'].nunique()}")
        print(f"  NUMBER OF TIME SERIES: {df['unique_id'].nunique()}")
        print(f"  Horizon: {horizon}, Lags: {n_lags}, Frequency: {freq_str}")

        train, test = LoadDataset.train_test_split(df, horizon)

        max_len = df["unique_id"].value_counts().max() - (2 * horizon)
        min_len = df["unique_id"].value_counts().min() - (2 * horizon)
        n_uids = df["unique_id"].nunique()
        max_n_uids = int(np.round(np.log(n_uids), 0))
        max_n_uids = 2 if max_n_uids < 2 else max_n_uids

        augmentation_params = {
            "seas_period": freq_int,
            "freq": freq_str,
            "max_n_uids": max_n_uids,
            "max_len": max_len,
            "min_len": min_len,
            "max_steps": MAX_STEPS,  # used by TSDiff wrapper
        }

        all_methods = BASELINE_METHOD + GRASYNDA_METHODS + OTHER_METHODS

        for method_idx, method_name in enumerate(all_methods):
            variant_name = method_name

            print(f"\n  [{method_idx + 1}/{len(all_methods)}] Method: {variant_name}")

            if method_name == "Baseline":
                print("    No augmentation (baseline)...")
                training_sets = {"Baseline": train}
                modes_to_test = ["Baseline"]
            elif method_name in GRASYNDA_METHODS:
                print("    Grasynda...")
                synth = generate_grasynda_data(method_name, train, freq_int, group, horizon, data_name)
                training_sets = {
                    "Train+Real": pd.concat([train, synth]).reset_index(drop=True),
                    "TSTR": synth,
                }
                modes_to_test = TRAINING_MODES
            else:
                print(f"    Augmenting ({variant_name})...")
                current_params = augmentation_params.copy()
                
                is_optimized = variant_name.endswith("_Optimized")
                base_method = variant_name.replace("_Optimized", "")

                if base_method == "TSMixup":
                    current_params.update(DEFAULT_TSMIXUP_PARAMS)
                    if is_optimized:
                        current_params.update(
                            OPTIMAL_TSMIXUP_PARAMS.get((data_name, group), {})
                        )
                elif base_method == "DBA":
                    current_params.update(DEFAULT_DBA_PARAMS)
                    if is_optimized:
                        current_params.update(
                            OPTIMAL_DBA_PARAMS.get((data_name, group), {})
                        )
                elif base_method == "TimeVAE":
                    current_params.update(DEFAULT_TIMEVAE_PARAMS)
                    if is_optimized:
                        current_params.update(
                            OPTIMAL_TIMEVAE_PARAMS.get(
                                (data_name, group), {}
                            )
                        )
                elif base_method == "MagnitudeWarping":
                    current_params.update(DEFAULT_MAGNITUDE_WARPING_PARAMS)
                    if is_optimized:
                        current_params.update(
                            OPTIMAL_MAGNITUDE_WARPING_PARAMS.get(
                                (data_name, group), {}
                            )
                        )
                elif base_method == "TSDiff":
                    current_params.update(DEFAULT_TSDIFF_PARAMS)
                    if is_optimized:
                        current_params.update(
                            OPTIMAL_TSDIFF_PARAMS.get((data_name, group), {})
                        )
                    current_params["transform_mode"] = "sample"
                    current_params["normalization"] = "mean"
                    current_params["restore_scale"] = True
                    current_params["clip_to_observed_range"] = True
                    current_params["use_lags"] = False
                    current_params["use_features"] = False
                    current_params["clip_scaled"] = False
                    current_params["show_progress"] = False
                elif base_method == "SeasonalMBB":
                    current_params.update(DEFAULT_SEASONAL_MBB_PARAMS)
                    if is_optimized:
                        current_params.update(
                            OPTIMAL_SEASONAL_MBB_PARAMS.get((data_name, group), {})
                        )
                elif base_method == "Jittering":
                    current_params.update(DEFAULT_JITTERING_PARAMS)
                    if is_optimized:
                        current_params.update(
                            OPTIMAL_JITTERING_PARAMS.get((data_name, group), {})
                        )
                elif base_method == "Scaling":
                    current_params.update(DEFAULT_SCALING_PARAMS)
                    if is_optimized:
                        current_params.update(
                            OPTIMAL_SCALING_PARAMS.get((data_name, group), {})
                        )
                elif base_method == "TimeWarping":
                    current_params.update(DEFAULT_TIMEWARPING_PARAMS)
                    if is_optimized:
                        current_params.update(
                            OPTIMAL_TIMEWARPING_PARAMS.get((data_name, group), {})
                        )

                current_params = normalize_augmentation_params(current_params)

                n_series_val = AUGMENTATION_SAMPLES
                if "n_series_factor" in current_params:
                    factor = current_params.pop("n_series_factor")
                    n_series_val = int(round(n_series_val * factor))
                    n_series_val = max(1, n_series_val)

                synth = generate_other_augmentation_data(
                    base_method, train, current_params, n_series=n_series_val
                )
                training_sets = {
                    "Train+Real": synth,
                    "TSTR": extract_synthetic_only(synth, train),
                }
                modes_to_test = TRAINING_MODES

            for model_name in FORECASTING_MODELS:
                for mode in modes_to_test:
                    experiment_count += 1
                    train_data = training_sets[mode]
                    print(f"      [{experiment_count}] {model_name} - {mode} (n={len(train_data)})")

                    model_inst = build_model_instance(
                        model_name=model_name,
                        horizon=horizon,
                        n_lags=n_lags,
                        max_steps=MAX_STEPS,
                        alias=f"{variant_name}_{mode}",
                    )
                    nf = NeuralForecast(models=[model_inst], freq=freq_str)
                    Y_df = train_data
                    if model_name == "AutoNHITS":
                        nf.fit(df=Y_df)
                    else:
                        nf.fit(df=Y_df, val_size=horizon)

                    if mode == "TSTR":
                        fcst = nf.predict(df=train)
                    else:
                        fcst = nf.predict()

                    fcst_out = fcst.reset_index()
                    pred_cols = [c for c in fcst_out.columns if c not in {"unique_id", "ds"}]
                    if len(pred_cols) != 1:
                        raise ValueError(f"Expected exactly 1 prediction column, got: {pred_cols}")
                    pred_col = pred_cols[0]

                    test_with_fcst = test.merge(
                        fcst_out, on=["unique_id", "ds"], how="left"
                    )
                    eval_df = evaluate(
                        test_with_fcst,
                        [partial(mase, seasonality=freq_int)],
                        train_df=train,
                    )
                    mase_score = eval_df.query('metric=="mase"')[pred_col].mean()

                    all_results.append(
                        {
                            "Dataset": data_name,
                            "Group": group,
                            "Augmentation_Method": variant_name,
                            "Forecasting_Model": model_name,
                            "Training_Mode": mode,
                            "MASE": mase_score,
                            "Train_Size": len(train_data),
                            "Test_Size": len(test),
                            "Status": "Success",
                        }
                    )
                    print(f"        -> MASE: {mase_score:.4f}")

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(results_path, index=False)

    print("\n" + "=" * 100)
    print("CREATING PIVOT TABLES (Methods as Columns, Datasets as Rows)")
    print("=" * 100)

    pivot_modes = ["Baseline"] + TRAINING_MODES

    for model_name in FORECASTING_MODELS:
        for mode in pivot_modes:
            filtered = results_df[
                (results_df["Forecasting_Model"] == model_name)
                & (results_df["Training_Mode"] == mode)
                & (results_df["Status"] == "Success")
            ].copy()
            if len(filtered) == 0:
                continue

            filtered["Dataset_Full"] = filtered["Dataset"] + " - " + filtered["Group"]
            pivot = filtered.pivot_table(
                index="Dataset_Full",
                columns="Augmentation_Method",
                values="MASE",
                aggfunc="first",
            )
            grasynda_cols = [c for c in pivot.columns if c in GRASYNDA_METHODS]
            other_cols = [c for c in pivot.columns if c in OTHER_METHODS]
            ordered_cols = grasynda_cols + other_cols
            if ordered_cols:
                pivot = pivot[ordered_cols]

            filename = f"assets/results/systematic_eval_{RESULTS_TAG}_{model_name}_{mode}_Results.csv"
            pivot.to_csv(filename)
            print(f"  Saved: {filename}")

    elapsed = datetime.now() - start_time
    print("\n" + "=" * 100)
    print("COMPLETE")
    print("=" * 100)
    print(f"Experiments: {len(all_results)}")
    print(f"Success: {sum(1 for r in all_results if r['Status'] == 'Success')}")
    print(f"Failed: {sum(1 for r in all_results if r['Status'] != 'Success')}")
    print(f"Time: {elapsed}")
    print("\nFiles created:")
    print(f"  - {results_path} (all data)")
    for model_name in FORECASTING_MODELS:
        for mode in pivot_modes:
            print(f"  - assets/results/systematic_eval_{RESULTS_TAG}_{model_name}_{mode}_Results.csv (pivot table)")
    print("=" * 100)

    return results_df


if __name__ == "__main__":
    os.makedirs("assets/results", exist_ok=True)
    run_universal_experiments()

