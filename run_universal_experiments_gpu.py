from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict

import run_universal_experiments as base
from src.grasynda_unified import GrasyndaUnified
from src.workflow_gpu import ExpWorkflow
from utils.config_gpu import (
    ACCELERATOR,
    AUTO_MODELS,
    MODEL_CONFIG,
    MODELS,
    REQUIRE_GPU,
    TORCH_DEVICE,
    assert_runtime_ready_for_requested_accelerator,
    get_runtime_hardware_summary,
)

# Captured GPU-runner defaults from the deleted local file.
DATASETS_TO_TEST = [
    ("M3", "Monthly"),
    ("M3", "Quarterly"),
]

BASELINE_METHOD = [
    "Baseline",
]

OTHER_METHODS = [
    "TSDiff",
]

GRASYNDA_METHODS = [
    "Hybrid_Q10_NoEnsemble_Continuous",
    "Hybrid_Q10_NoEnsemble_KDE",
]

FORECASTING_MODELS = [
    "NHITS",
]

TRAINING_MODES = [
    "Train+Real",
]

MAX_STEPS = 1000
AUGMENTATION_SAMPLES = 1


USE_MIN_SAMPLES_FILTER = True
MIN_SAMPLES_BY_FREQUENCY = {"monthly": 48, "quarterly": 16}
RESULTS_TAG = "m3_tsdiff_grasynda_q10_cont_kde_trainreal_nhits_gpu_min48m_q16_20260507"

# Re-export common parameter dictionaries used by release-benchmark helpers.
DEFAULT_GRASYNDA_PARAMS = base.DEFAULT_GRASYNDA_PARAMS
OPTIMAL_GRASYNDA_PARAMS = base.OPTIMAL_GRASYNDA_PARAMS
DEFAULT_SEASONAL_MBB_PARAMS = base.DEFAULT_SEASONAL_MBB_PARAMS
OPTIMAL_SEASONAL_MBB_PARAMS = base.OPTIMAL_SEASONAL_MBB_PARAMS
DEFAULT_JITTERING_PARAMS = base.DEFAULT_JITTERING_PARAMS
OPTIMAL_JITTERING_PARAMS = base.OPTIMAL_JITTERING_PARAMS
DEFAULT_SCALING_PARAMS = base.DEFAULT_SCALING_PARAMS
OPTIMAL_SCALING_PARAMS = base.OPTIMAL_SCALING_PARAMS
DEFAULT_TIMEWARPING_PARAMS = base.DEFAULT_TIMEWARPING_PARAMS
OPTIMAL_TIMEWARPING_PARAMS = base.OPTIMAL_TIMEWARPING_PARAMS
DEFAULT_TSMIXUP_PARAMS = base.DEFAULT_TSMIXUP_PARAMS
OPTIMAL_TSMIXUP_PARAMS = base.OPTIMAL_TSMIXUP_PARAMS
DEFAULT_TIMEVAE_PARAMS = base.DEFAULT_TIMEVAE_PARAMS
OPTIMAL_TIMEVAE_PARAMS = base.OPTIMAL_TIMEVAE_PARAMS
DEFAULT_DBA_PARAMS = base.DEFAULT_DBA_PARAMS
OPTIMAL_DBA_PARAMS = base.OPTIMAL_DBA_PARAMS
DEFAULT_MAGNITUDE_WARPING_PARAMS = base.DEFAULT_MAGNITUDE_WARPING_PARAMS
OPTIMAL_MAGNITUDE_WARPING_PARAMS = base.OPTIMAL_MAGNITUDE_WARPING_PARAMS
DEFAULT_TSDIFF_PARAMS = base.DEFAULT_TSDIFF_PARAMS
OPTIMAL_TSDIFF_PARAMS = base.OPTIMAL_TSDIFF_PARAMS

_base_generate_grasynda_data = base.generate_grasynda_data


def build_hybrid_q10_generator(
    freq_int: int,
    apply_differentiation: bool,
    ensemble_transitions: bool,
    ensemble_size,
    sampling_type: str = "continuous_uniform",
):
    return GrasyndaUnified(
        period=freq_int,
        n_quantiles=10,
        sampling_type=sampling_type,
        graph_type="quantile",
        components_to_model=["trend", "remainder"],
        component_params={"trend": {"apply_differentiation": apply_differentiation}},
        ensemble_transitions=ensemble_transitions,
        ensemble_size=ensemble_size,
    )


def generate_grasynda_data(method_name, train_df, freq_int, group, horizon, data_name):
    if method_name == "Hybrid_Q10_NoEnsemble_KDE":
        generator = build_hybrid_q10_generator(
            freq_int=freq_int,
            apply_differentiation=True,
            ensemble_transitions=False,
            ensemble_size=None,
            sampling_type="kde",
        )
        return generator.transform(train_df)

    return _base_generate_grasynda_data(method_name, train_df, freq_int, group, horizon, data_name)


def build_model_instance(model_name: str, horizon: int, n_lags: int, max_steps: int, alias: str):
    if model_name not in MODELS:
        raise ValueError(f"Unsupported forecasting model: {model_name}")

    model_cls = MODELS[model_name]
    model_cfg: Dict[str, Any] = dict(MODEL_CONFIG.get(model_name, {}))

    if model_name in AUTO_MODELS:
        kwargs = {key: value for key, value in model_cfg.items() if key in {"backend", "num_samples", "config"}}
        return model_cls(h=horizon, alias=alias, **kwargs)

    kwargs = {
        key: value
        for key, value in model_cfg.items()
        if key not in {"auto", "backend", "num_samples", "config"}
    }
    return model_cls(
        h=horizon,
        input_size=n_lags,
        max_steps=max_steps,
        alias=alias,
        **kwargs,
    )


def _configure_base_runner() -> None:
    base.ExpWorkflow = ExpWorkflow
    base.MODELS = MODELS
    base.MODEL_CONFIG = MODEL_CONFIG
    base.AUTO_MODELS = AUTO_MODELS
    base.DATASETS_TO_TEST = DATASETS_TO_TEST
    base.BASELINE_METHOD = BASELINE_METHOD
    base.OTHER_METHODS = OTHER_METHODS
    base.GRASYNDA_METHODS = GRASYNDA_METHODS
    base.FORECASTING_MODELS = FORECASTING_MODELS
    base.TRAINING_MODES = TRAINING_MODES
    base.MAX_STEPS = MAX_STEPS
    base.AUGMENTATION_SAMPLES = AUGMENTATION_SAMPLES
    base.USE_MIN_SAMPLES_FILTER = USE_MIN_SAMPLES_FILTER
    base.MIN_SAMPLES_BY_FREQUENCY = MIN_SAMPLES_BY_FREQUENCY
    base.RESULTS_TAG = RESULTS_TAG
    base.generate_grasynda_data = generate_grasynda_data
    base.build_model_instance = build_model_instance


def run_universal_experiments():
    runtime = get_runtime_hardware_summary()
    print("GPU runtime:")
    print(f"  Requested accelerator: {runtime['requested_accelerator']}")
    print(f"  NeuralForecast accelerator: {ACCELERATOR}")
    print(f"  Torch device: {TORCH_DEVICE}")
    print(f"  Torch CUDA available: {runtime['torch_cuda_available']}")
    print(f"  Torch CUDA device count: {runtime['torch_cuda_device_count']}")
    print(f"  GPU required: {REQUIRE_GPU}")
    assert_runtime_ready_for_requested_accelerator()

    _configure_base_runner()
    return base.run_universal_experiments()


if __name__ == "__main__":
    run_universal_experiments()
