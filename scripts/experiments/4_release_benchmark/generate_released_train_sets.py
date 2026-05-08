"""
Materialize released training datasets for the paper privacy-utility pipeline.

Current config generates only the paper-level faithful LPA/FPA anonymization
baselines plus an OriginalBaseline row. Keep behavior controlled from the
configuration block below rather than command-line arguments.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

import run_universal_experiments_gpu as universal
from src.grasynda_unified import GrasyndaUnified
from src.workflow import ExpWorkflow
from utils.load_data.config import DATASETS, DATA_GROUPS


# =========================
# Configuration
# =========================

DATASETS_TO_TEST = list(DATA_GROUPS)
GRASYNDA_METHODS = [
    "Grasynda_Optimized",
]
AUGMENTATION_METHODS = [
    "SeasonalMBB",
    "Jittering",
    "Scaling",
    "MagnitudeWarping",
    "TimeWarping",
    "DBA",
    "TSMixup",
    "TimeVAE",
]
ANONYMIZATION_METHODS = [
    "LPA",
    "FPA",
]
ANONYMIZATION_EPSILONS = [0.48, 2.4, 4.8, 24.0]
SAMPLE_N_UIDS = None
SEED = 42
USE_MIN_SAMPLES_FILTER = True
FIXED_N_SYNTH_PER_UID = 1
CLIP_LOWER = None
CLIP_UPPER = None
LOWER_QUANTILE = 0.01
UPPER_QUANTILE = 0.99
SENSITIVITY_OVERRIDE = None
FOURIER_K = 30
FOURIER_KEEP_RATIO = 0.15
OUTPUT_TAG = "release_universal_grasynda_da_faithful_lpa_fpa_20260508"
FINAL_NHITS_OPTIMAL_PARAMS_PATH = os.path.join(
    PROJECT_ROOT,
    "assets",
    "results",
    "random_search",
    "nhits_optimal_parameters_final.json",
)

SUPPORTED_GRASYNDA_METHODS = ["Grasynda_Optimized"]
SUPPORTED_AUGMENTATION_METHODS = [
    "SeasonalMBB",
    "Jittering",
    "Scaling",
    "MagnitudeWarping",
    "TimeWarping",
    "DBA",
    "TSMixup",
    "TimeVAE",
]
SUPPORTED_ANONYMIZATION_METHODS = ["LPA", "FPA", "sFPA", "tFPA"]
METHOD_ALIASES = {
    "DP_Laplace": "LPA",
    "DP_Fourier": "FPA",
    "STL_DP_Seasonal": "sFPA",
    "STL_DP_Trend": "tFPA",
}

DEFAULT_GRASYNDA_PARAMS = {"n_quantiles": 10, "ensemble_transitions": False, "ensemble_size": None}
OPTIMAL_GRASYNDA_PARAMS = {
    ("Gluonts", "m1_monthly"): {"n_quantiles": 75, "ensemble_transitions": True, "ensemble_size": 15},
    ("Gluonts", "m1_quarterly"): {"n_quantiles": 50, "ensemble_transitions": True, "ensemble_size": 3},
    ("M3", "Monthly"): {"n_quantiles": 5, "ensemble_transitions": False, "ensemble_size": None},
    ("M3", "Quarterly"): {"n_quantiles": 3, "ensemble_transitions": True, "ensemble_size": 2},
    ("NN3", "Monthly"): {"n_quantiles": 30, "ensemble_transitions": True, "ensemble_size": 8},
    ("Tourism", "Monthly"): {"n_quantiles": 5, "ensemble_transitions": False, "ensemble_size": None},
    ("Tourism", "Quarterly"): {"n_quantiles": 5, "ensemble_transitions": True, "ensemble_size": 2},
}

DEFAULT_AUGMENTATION_PARAMS = {
    "SeasonalMBB": {"log": True, "max_samples_in_stl": None},
    "Jittering": {"sigma": 0.03},
    "Scaling": {"sigma": 0.1},
    "MagnitudeWarping": {"sigma": 0.2, "knot": 4},
    "TimeWarping": {"sigma": 0.2, "knot": 4},
    "DBA": {"max_n_uids": 2, "dirichlet_alpha": 1.0, "max_iter": 10},
    "TSMixup": {"max_n_uids": 3, "dirichlet_alpha": 1.0},
    "TimeVAE": {"latent_dim": 8, "reconstruction_wt": 3.0, "max_epochs": 100},
}
OPTIMAL_AUGMENTATION_PARAMS = {
    "SeasonalMBB": {
        ("Gluonts", "m1_monthly"): {"log": False, "max_samples_in_stl": None},
        ("Gluonts", "m1_quarterly"): {"log": True, "max_samples_in_stl": 128},
        ("M3", "Monthly"): {"log": True, "max_samples_in_stl": 512},
        ("M3", "Quarterly"): {"log": True, "max_samples_in_stl": 512},
        ("NN3", "Monthly"): {"log": True, "max_samples_in_stl": 256},
        ("Tourism", "Monthly"): {"log": False, "max_samples_in_stl": 256},
        ("Tourism", "Quarterly"): {"log": False, "max_samples_in_stl": 512},
    },
    "Jittering": {
        ("Gluonts", "m1_monthly"): {"sigma": 0.1174},
        ("Gluonts", "m1_quarterly"): {"sigma": 0.0275},
        ("M3", "Monthly"): {"sigma": 0.0081},
        ("M3", "Quarterly"): {"sigma": 0.1002},
        ("NN3", "Monthly"): {"sigma": 0.0551},
        ("Tourism", "Monthly"): {"sigma": 0.0517},
        ("Tourism", "Quarterly"): {"sigma": 0.0847},
    },
    "Scaling": {
        ("Gluonts", "m1_monthly"): {"sigma": 0.0787},
        ("Gluonts", "m1_quarterly"): {"sigma": 0.0703},
        ("M3", "Monthly"): {"sigma": 0.0674},
        ("M3", "Quarterly"): {"sigma": 0.1077},
        ("NN3", "Monthly"): {"sigma": 0.0139},
        ("Tourism", "Monthly"): {"sigma": 0.0319},
        ("Tourism", "Quarterly"): {"sigma": 0.0354},
    },
    "MagnitudeWarping": {
        ("Gluonts", "m1_monthly"): {"sigma": 0.0905, "knot": 6},
        ("Gluonts", "m1_quarterly"): {"sigma": 0.0944, "knot": 2},
        ("M3", "Monthly"): {"sigma": 0.0358, "knot": 3},
        ("M3", "Quarterly"): {"sigma": 0.1766, "knot": 3},
        ("NN3", "Monthly"): {"sigma": 0.0497, "knot": 5},
        ("Tourism", "Monthly"): {"sigma": 0.0694, "knot": 6},
        ("Tourism", "Quarterly"): {"sigma": 0.0510, "knot": 2},
    },
    "TimeWarping": {
        ("Gluonts", "m1_monthly"): {"sigma": 0.0302, "knot": 4},
        ("Gluonts", "m1_quarterly"): {"sigma": 0.0104, "knot": 2},
        ("M3", "Monthly"): {"sigma": 0.1186, "knot": 6},
        ("M3", "Quarterly"): {"sigma": 0.0362, "knot": 5},
        ("NN3", "Monthly"): {"sigma": 0.0118, "knot": 3},
        ("Tourism", "Monthly"): {"sigma": 0.1232, "knot": 6},
        ("Tourism", "Quarterly"): {"sigma": 0.0154, "knot": 2},
    },
    "DBA": {
        ("Gluonts", "m1_monthly"): {"max_n_uids": 4, "dirichlet_alpha": 0.6093, "max_iter": 40},
        ("Gluonts", "m1_quarterly"): {"max_n_uids": 2, "dirichlet_alpha": 0.4453, "max_iter": 25},
        ("M3", "Monthly"): {"max_n_uids": 5, "dirichlet_alpha": 1.1399, "max_iter": 12},
        ("M3", "Quarterly"): {"max_n_uids": 4, "dirichlet_alpha": 1.2556, "max_iter": 15},
        ("NN3", "Monthly"): {"max_n_uids": 5, "dirichlet_alpha": 0.6068, "max_iter": 12},
    },
    "TSMixup": {
        ("Gluonts", "m1_monthly"): {"max_n_uids": 2, "dirichlet_alpha": 0.9109, "n_series_factor": 1.83},
        ("Gluonts", "m1_quarterly"): {"max_n_uids": 1, "dirichlet_alpha": 0.8648, "n_series_factor": 1.61},
        ("M3", "Monthly"): {"max_n_uids": 6, "dirichlet_alpha": 2.4165, "n_series_factor": 1.48},
        ("M3", "Quarterly"): {"max_n_uids": 1, "dirichlet_alpha": 2.9405, "n_series_factor": 1.45},
        ("NN3", "Monthly"): {"max_n_uids": 1, "dirichlet_alpha": 2.4129, "n_series_factor": 1.43},
        ("Tourism", "Monthly"): {"max_n_uids": 7, "dirichlet_alpha": 1.2358, "n_series_factor": 2.01},
        ("Tourism", "Quarterly"): {"max_n_uids": 5, "dirichlet_alpha": 0.2544, "n_series_factor": 1.85},
    },
    "TimeVAE": {
        ("Gluonts", "m1_monthly"): {"latent_dim": 4, "reconstruction_wt": 3.0, "max_epochs": 40},
        ("Gluonts", "m1_quarterly"): {"latent_dim": 4, "reconstruction_wt": 3.0, "max_epochs": 80},
        ("M3", "Monthly"): {"latent_dim": 8, "reconstruction_wt": 7.5, "max_epochs": 50},
        ("M3", "Quarterly"): {"latent_dim": 8, "reconstruction_wt": 1.0, "max_epochs": 60},
        ("NN3", "Monthly"): {"latent_dim": 4, "reconstruction_wt": 5.0, "max_epochs": 150},
        ("Tourism", "Monthly"): {"latent_dim": 4, "reconstruction_wt": 2.0, "max_epochs": 150},
        ("Tourism", "Quarterly"): {"latent_dim": 4, "reconstruction_wt": 2.0, "max_epochs": 40},
    },
}


def load_final_nhits_optimal_params() -> None:
    """Prefer audited random-search params when the final JSON exists."""
    global OPTIMAL_GRASYNDA_PARAMS, OPTIMAL_AUGMENTATION_PARAMS

    if not os.path.exists(FINAL_NHITS_OPTIMAL_PARAMS_PATH):
        return

    try:
        with open(FINAL_NHITS_OPTIMAL_PARAMS_PATH, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception as exc:
        print(f"Warning: could not read {FINAL_NHITS_OPTIMAL_PARAMS_PATH}: {exc}")
        return

    methods = payload.get("Methods", {})
    if not isinstance(methods, dict):
        return

    grasynda_targets = methods.get("Grasynda", {}).get("Targets", [])
    if grasynda_targets:
        loaded = {}
        for row in grasynda_targets:
            params = row.get("Params")
            if isinstance(params, dict):
                loaded[(row.get("Dataset"), row.get("Group"))] = params
        if loaded:
            OPTIMAL_GRASYNDA_PARAMS = loaded
            print(f"Loaded final NHITS Grasynda params from {FINAL_NHITS_OPTIMAL_PARAMS_PATH}")

    for method_name in SUPPORTED_AUGMENTATION_METHODS:
        targets = methods.get(method_name, {}).get("Targets", [])
        if not targets:
            continue
        loaded = {}
        for row in targets:
            params = row.get("Params")
            if isinstance(params, dict):
                loaded[(row.get("Dataset"), row.get("Group"))] = params
        if loaded:
            OPTIMAL_AUGMENTATION_PARAMS[method_name] = loaded
            print(f"Loaded final NHITS {method_name} params from {FINAL_NHITS_OPTIMAL_PARAMS_PATH}")


load_final_nhits_optimal_params()


@dataclass
class ClipConfig:
    lower: float
    upper: float
    source: str
    formal_dp_ready: bool


def stable_int(*parts: object) -> int:
    payload = "|".join(map(str, parts)).encode("utf-8")
    return int(hashlib.md5(payload).hexdigest()[:8], 16)


def canonicalize_method(method_name: str) -> str:
    return METHOD_ALIASES.get(method_name, method_name)


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


def resolve_frequency_int(group: str, fallback: Optional[int]) -> int:
    freq_key = resolve_frequency_key(group)
    if freq_key == "monthly":
        return 12
    if freq_key == "quarterly":
        return 4
    if fallback is None:
        raise ValueError(f"Could not infer seasonal period for group '{group}'")
    return int(fallback)


def resolve_min_samples(group: str, use_filter: bool) -> Optional[int]:
    if not use_filter:
        return None
    freq_key = resolve_frequency_key(group)
    if freq_key == "monthly":
        return universal.MIN_SAMPLES_BY_FREQUENCY["monthly"]
    if freq_key == "quarterly":
        return universal.MIN_SAMPLES_BY_FREQUENCY["quarterly"]
    return None


def save_release_csv(df: pd.DataFrame, path: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    return os.path.abspath(path)


def normalize_params(params: dict) -> dict:
    out = dict(params)
    int_like = {"knot", "max_samples_in_stl", "max_n_uids", "latent_dim", "max_epochs", "max_iter", "n_quantiles", "ensemble_size"}
    for key in int_like:
        if key in out and out[key] is not None:
            out[key] = int(round(out[key]))
    return out


def extract_synthetic_only(candidate_df: pd.DataFrame, real_df: pd.DataFrame) -> pd.DataFrame:
    real_ids = set(real_df["unique_id"].unique())
    synth_only = candidate_df[~candidate_df["unique_id"].isin(real_ids)].copy()
    if synth_only.empty and len(candidate_df) >= len(real_df):
        synth_only = candidate_df.iloc[len(real_df):].copy()
    return synth_only.reset_index(drop=True)


def generate_grasynda_data(method_name: str, train_df: pd.DataFrame, freq_int: int, dataset_name: str, group: str) -> pd.DataFrame:
    if method_name not in SUPPORTED_GRASYNDA_METHODS:
        raise ValueError(f"Unsupported Grasynda method: {method_name}")
    params = dict(DEFAULT_GRASYNDA_PARAMS)
    params.update(OPTIMAL_GRASYNDA_PARAMS.get((dataset_name, group), {}))
    params = normalize_params(params)
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


def build_augmentation_params(
    method_name: str,
    train_df: pd.DataFrame,
    freq_int: int,
    dataset_name: str,
    group: str,
) -> tuple[dict, int]:
    if method_name not in SUPPORTED_AUGMENTATION_METHODS:
        raise ValueError(f"Unsupported augmentation method: {method_name}")
    params = {"seas_period": freq_int}
    params.update(DEFAULT_AUGMENTATION_PARAMS.get(method_name, {}))
    params.update(OPTIMAL_AUGMENTATION_PARAMS.get(method_name, {}).get((dataset_name, group), {}))

    n_series_by_uid = FIXED_N_SYNTH_PER_UID
    n_series_factor = params.pop("n_series_factor", None)
    if n_series_factor is not None:
        n_series_by_uid = max(1, int(round(float(n_series_factor))))

    if method_name == "TSMixup":
        lengths = train_df.groupby("unique_id").size()
        params["min_len"] = int(lengths.min())
        params["max_len"] = int(lengths.max())
    if method_name == "DBA":
        params["max_n_uids"] = min(int(params["max_n_uids"]), int(train_df["unique_id"].nunique()))

    return normalize_params(params), n_series_by_uid


def generate_augmentation_data(method_name: str, train_df: pd.DataFrame, params: dict, n_series_by_uid: int) -> pd.DataFrame:
    augmented = ExpWorkflow.get_offline_augmented_data(
        train_=train_df,
        generator_name=method_name,
        augmentation_params=params,
        n_series_by_uid=n_series_by_uid,
    )
    return extract_synthetic_only(augmented, train_df)


def estimate_clip_config(df_real: pd.DataFrame) -> ClipConfig:
    if CLIP_LOWER is not None and CLIP_UPPER is not None:
        return ClipConfig(float(CLIP_LOWER), float(CLIP_UPPER), "fixed_bounds", True)
    lower = float(df_real["y"].quantile(LOWER_QUANTILE))
    upper = float(df_real["y"].quantile(UPPER_QUANTILE))
    if upper <= lower:
        lower = float(df_real["y"].min())
        upper = float(df_real["y"].max())
    if upper <= lower:
        upper = lower + 1.0
    return ClipConfig(lower, upper, f"quantile_{LOWER_QUANTILE:g}_{UPPER_QUANTILE:g}", False)


def clip_values(values: np.ndarray, clip_cfg: ClipConfig) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=float), clip_cfg.lower, clip_cfg.upper)


def laplace_noise(rng: np.random.Generator, size, scale: float) -> np.ndarray:
    if scale <= 0:
        return np.zeros(size, dtype=float)
    return rng.laplace(loc=0.0, scale=scale, size=size)


def determine_fourier_k(series_len: int, requested_k: Optional[int], keep_ratio: float) -> int:
    max_coeffs = series_len // 2 + 1
    if requested_k is not None:
        return max(1, min(int(requested_k), max_coeffs))
    return max(1, min(max_coeffs, int(np.ceil(max_coeffs * keep_ratio))))


def sequence_l1_sensitivity(series_len: int, per_timestamp_sensitivity: float) -> float:
    return float(series_len) * float(per_timestamp_sensitivity)


def sequence_l2_sensitivity(series_len: int, per_timestamp_sensitivity: float) -> float:
    return float(np.sqrt(float(series_len)) * float(per_timestamp_sensitivity))


def lpa_noise_scale(series_len: int, per_timestamp_sensitivity: float, epsilon: float) -> float:
    return sequence_l1_sensitivity(series_len, per_timestamp_sensitivity) / max(epsilon, 1e-12)


def fpa_noise_scale(series_len: int, fourier_k: int, per_timestamp_sensitivity: float, epsilon: float) -> float:
    """Real-coordinate FPA scale for an orthonormal Fourier transform.

    Rastogi and Nath Theorem 4.1 gives sqrt(k) * Delta_2(Q) / epsilon for
    k complex coefficients. This implementation perturbs real and imaginary
    coordinates with ordinary real Laplace noise, so it uses the conservative
    sqrt(2k) coordinate bound.
    """
    delta2 = sequence_l2_sensitivity(series_len, per_timestamp_sensitivity)
    return float(np.sqrt(2.0 * float(fourier_k)) * delta2 / max(epsilon, 1e-12))


def dp_laplace_series(values: np.ndarray, epsilon: float, sensitivity: float, clip_cfg: ClipConfig, rng: np.random.Generator) -> np.ndarray:
    clipped = clip_values(values, clip_cfg)
    scale = lpa_noise_scale(len(clipped), sensitivity, epsilon)
    noisy = clipped + laplace_noise(rng, clipped.shape, scale=scale)
    return clip_values(noisy, clip_cfg)


def dp_fourier_series(
    values: np.ndarray,
    epsilon: float,
    sensitivity: float,
    clip_cfg: ClipConfig,
    rng: np.random.Generator,
    fourier_k: Optional[int],
    keep_ratio: float,
) -> np.ndarray:
    clipped = clip_values(values, clip_cfg)
    coeffs = np.fft.rfft(clipped, norm="ortho")
    k = determine_fourier_k(len(clipped), fourier_k, keep_ratio)
    noisy = np.zeros_like(coeffs)
    retained = coeffs[:k].copy()
    scale = fpa_noise_scale(len(clipped), k, sensitivity, epsilon)
    retained_real = retained.real + laplace_noise(rng, retained.shape, scale=scale)
    retained_imag = retained.imag + laplace_noise(rng, retained.shape, scale=scale)
    noisy[:k] = retained_real + 1j * retained_imag
    recon = np.fft.irfft(noisy, n=len(clipped), norm="ortho")
    return clip_values(recon, clip_cfg)


def resolve_stl_period(freq_int: int, series_len: int) -> int:
    period = max(2, int(freq_int))
    if period >= series_len:
        period = max(2, series_len // 2)
    return max(2, period)


def decompose_stl(values: np.ndarray, freq_int: int):
    period = resolve_stl_period(freq_int, len(values))
    fitted = STL(values, period=period, robust=True).fit()
    return fitted.trend, fitted.seasonal, fitted.resid


def dp_stl_series(
    values: np.ndarray,
    epsilon: float,
    sensitivity: float,
    clip_cfg: ClipConfig,
    rng: np.random.Generator,
    freq_int: int,
    component: str,
    fourier_k: Optional[int],
    keep_ratio: float,
) -> np.ndarray:
    clipped = clip_values(values, clip_cfg)
    trend, seasonal, resid = decompose_stl(clipped, freq_int)
    if component == "seasonal":
        seasonal_noisy = dp_fourier_series(seasonal, epsilon, sensitivity, clip_cfg, rng, fourier_k, keep_ratio)
        return clip_values(trend + seasonal_noisy + resid, clip_cfg)
    if component == "trend":
        trend_noisy = dp_fourier_series(trend, epsilon, sensitivity, clip_cfg, rng, fourier_k, keep_ratio)
        return clip_values(trend_noisy + seasonal + resid, clip_cfg)
    raise ValueError(f"Unsupported STL component: {component}")


def anonymize_dataframe(
    df_real: pd.DataFrame,
    method_name: str,
    epsilon: float,
    freq_int: int,
    clip_cfg: ClipConfig,
    seed: int,
) -> pd.DataFrame:
    method_name = canonicalize_method(method_name)
    sensitivity = float(SENSITIVITY_OVERRIDE) if SENSITIVITY_OVERRIDE is not None else clip_cfg.upper - clip_cfg.lower
    rows = []
    for uid, uid_df in df_real.groupby("unique_id", sort=False):
        uid_sorted = uid_df.sort_values("ds").copy()
        values = uid_sorted["y"].to_numpy(dtype=float)
        rng = np.random.default_rng(stable_int(seed, method_name, epsilon, uid))
        if method_name == "LPA":
            anonymized = dp_laplace_series(values, epsilon, sensitivity, clip_cfg, rng)
        elif method_name == "FPA":
            anonymized = dp_fourier_series(values, epsilon, sensitivity, clip_cfg, rng, FOURIER_K, FOURIER_KEEP_RATIO)
        elif method_name == "sFPA":
            anonymized = dp_stl_series(values, epsilon, sensitivity, clip_cfg, rng, freq_int, "seasonal", FOURIER_K, FOURIER_KEEP_RATIO)
        elif method_name == "tFPA":
            anonymized = dp_stl_series(values, epsilon, sensitivity, clip_cfg, rng, freq_int, "trend", FOURIER_K, FOURIER_KEEP_RATIO)
        else:
            raise ValueError(f"Unsupported anonymization method: {method_name}")
        uid_sorted["y"] = anonymized
        rows.append(uid_sorted)
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = OUTPUT_TAG or f"release_materialization_{timestamp}"
    out_dir = os.path.join(PROJECT_ROOT, "assets", "results", "release_benchmark", tag)
    releases_dir = os.path.join(out_dir, "releases")
    os.makedirs(releases_dir, exist_ok=True)

    methods = [canonicalize_method(method) for method in ANONYMIZATION_METHODS]
    for method in GRASYNDA_METHODS:
        if method not in SUPPORTED_GRASYNDA_METHODS:
            raise ValueError(f"Unsupported Grasynda method: {method}")
    for method in AUGMENTATION_METHODS:
        if method not in SUPPORTED_AUGMENTATION_METHODS:
            raise ValueError(f"Unsupported augmentation method: {method}")
    for method in methods:
        if method not in SUPPORTED_ANONYMIZATION_METHODS:
            raise ValueError(f"Unsupported anonymization method: {method}")

    config_payload = {
        "created_at": timestamp,
        "targets": DATASETS_TO_TEST,
        "grasynda_methods": GRASYNDA_METHODS,
        "augmentation_methods": AUGMENTATION_METHODS,
        "anonymization_methods": methods,
        "epsilons": ANONYMIZATION_EPSILONS,
        "sample_n_uids": SAMPLE_N_UIDS,
        "seed": SEED,
        "use_min_samples_filter": USE_MIN_SAMPLES_FILTER,
        "fixed_n_synth_per_uid": FIXED_N_SYNTH_PER_UID,
        "clip_lower": CLIP_LOWER,
        "clip_upper": CLIP_UPPER,
        "lower_quantile": LOWER_QUANTILE,
        "upper_quantile": UPPER_QUANTILE,
        "sensitivity_override": SENSITIVITY_OVERRIDE,
        "fourier_k": FOURIER_K,
        "fourier_keep_ratio": FOURIER_KEEP_RATIO,
        "dp_calibration": "Rastogi_Nath_2010_sequence_l1_l2_orthonormal_real_coordinate_fpa",
        "sensitivity_interpretation": (
            "sensitivity_override or clip range is treated as per-timestamp sensitivity; "
            "LPA lambda=n*s/epsilon; FPA uses orthonormal RFFT and real/imag coordinate "
            "Laplace with lambda=sqrt(2k)*sqrt(n)*s/epsilon"
        ),
    }
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as handle:
        json.dump(config_payload, handle, indent=2)

    rows = []
    for dataset_name, group in DATASETS_TO_TEST:
        print(f"\n### {dataset_name} - {group} ###", flush=True)
        loader_cls = DATASETS[dataset_name]
        load_kwargs = {}
        min_samples = resolve_min_samples(group, USE_MIN_SAMPLES_FILTER)
        if min_samples is not None:
            load_kwargs["min_n_instances"] = min_samples
        if SAMPLE_N_UIDS is not None:
            load_kwargs["sample_n_uid"] = SAMPLE_N_UIDS
        df, loader_horizon, n_lags, freq_str, freq_int = loader_cls.load_everything(group, **load_kwargs)
        if df.empty or df["unique_id"].nunique() == 0:
            print("  Warning: dataset is empty after min-length filtering; skipping.", flush=True)
            continue
        horizon = resolve_horizon(group, loader_horizon)
        freq_int = resolve_frequency_int(group, freq_int)
        train, test = loader_cls.train_test_split(df, horizon)
        clip_cfg = estimate_clip_config(train)
        dataset_slug = f"{dataset_name}_{group}".replace(":", "_").replace(" ", "_")
        train_path = save_release_csv(train, os.path.join(releases_dir, f"{dataset_slug}__original_train.csv"))
        test_path = save_release_csv(test, os.path.join(releases_dir, f"{dataset_slug}__real_test.csv"))
        print(
            f"  Train shape={train.shape}, Test shape={test.shape}, UIDs={train['unique_id'].nunique()}\n"
            f"  Clip bounds: [{clip_cfg.lower:.4f}, {clip_cfg.upper:.4f}] from {clip_cfg.source}",
            flush=True,
        )
        rows.append(
            {
                "Dataset": dataset_name,
                "Group": group,
                "Family": "Baseline",
                "Method": "OriginalBaseline",
                "Variant_Name": "OriginalBaseline",
                "Epsilon": np.nan,
                "Seed": SEED,
                "Freq_Str": freq_str,
                "Freq_Int": freq_int,
                "Horizon": horizon,
                "NLags": n_lags,
                "Real_Train_Path": train_path,
                "Real_Test_Path": test_path,
                "Released_Train_Path": train_path,
                "Released_Size": int(len(train)),
                "Clip_Lower": np.nan,
                "Clip_Upper": np.nan,
                "Clip_Source": "not_applicable",
                "Formal_DP_Ready": np.nan,
            }
        )

        for method in GRASYNDA_METHODS:
            print(f"  -> Grasynda: {method}", flush=True)
            released_train = generate_grasynda_data(method, train, freq_int, dataset_name, group)
            release_name = f"{dataset_slug}__Grasynda__{method}__seed{SEED}.csv"
            release_path = save_release_csv(released_train, os.path.join(releases_dir, release_name))
            rows.append(
                {
                    "Dataset": dataset_name,
                    "Group": group,
                    "Family": "Grasynda",
                    "Method": method,
                    "Variant_Name": method,
                    "Epsilon": np.nan,
                    "Seed": SEED,
                    "Freq_Str": freq_str,
                    "Freq_Int": freq_int,
                    "Horizon": horizon,
                    "NLags": n_lags,
                    "Real_Train_Path": train_path,
                    "Real_Test_Path": test_path,
                    "Released_Train_Path": release_path,
                    "Released_Size": int(len(released_train)),
                    "Clip_Lower": np.nan,
                    "Clip_Upper": np.nan,
                    "Clip_Source": "not_applicable",
                    "Formal_DP_Ready": False,
                }
            )

        for method in AUGMENTATION_METHODS:
            print(f"  -> Augmentation: {method}", flush=True)
            params, n_series_by_uid = build_augmentation_params(method, train, freq_int, dataset_name, group)
            released_train = generate_augmentation_data(method, train, params, n_series_by_uid)
            release_name = f"{dataset_slug}__OtherAugmentation__{method}__seed{SEED}.csv"
            release_path = save_release_csv(released_train, os.path.join(releases_dir, release_name))
            rows.append(
                {
                    "Dataset": dataset_name,
                    "Group": group,
                    "Family": "OtherAugmentation",
                    "Method": method,
                    "Variant_Name": method,
                    "Epsilon": np.nan,
                    "Seed": SEED,
                    "Freq_Str": freq_str,
                    "Freq_Int": freq_int,
                    "Horizon": horizon,
                    "NLags": n_lags,
                    "Real_Train_Path": train_path,
                    "Real_Test_Path": test_path,
                    "Released_Train_Path": release_path,
                    "Released_Size": int(len(released_train)),
                    "Clip_Lower": np.nan,
                    "Clip_Upper": np.nan,
                    "Clip_Source": "not_applicable",
                    "Formal_DP_Ready": False,
                }
            )

        for method in methods:
            for epsilon in ANONYMIZATION_EPSILONS:
                print(f"  -> Anonymization: {method} | epsilon={epsilon}", flush=True)
                released_train = anonymize_dataframe(train, method, float(epsilon), freq_int, clip_cfg, SEED)
                release_name = f"{dataset_slug}__AnonymizedOriginal__{method}__eps{epsilon:g}__seed{SEED}.csv"
                release_path = save_release_csv(released_train, os.path.join(releases_dir, release_name))
                rows.append(
                    {
                        "Dataset": dataset_name,
                        "Group": group,
                        "Family": "AnonymizedOriginal",
                        "Method": method,
                        "Variant_Name": f"{method}_eps{epsilon:g}",
                        "Epsilon": epsilon,
                        "Seed": SEED,
                        "Freq_Str": freq_str,
                        "Freq_Int": freq_int,
                        "Horizon": horizon,
                        "NLags": n_lags,
                        "Real_Train_Path": train_path,
                        "Real_Test_Path": test_path,
                        "Released_Train_Path": release_path,
                        "Released_Size": int(len(released_train)),
                        "Clip_Lower": clip_cfg.lower,
                        "Clip_Upper": clip_cfg.upper,
                        "Clip_Source": clip_cfg.source,
                        "Formal_DP_Ready": clip_cfg.formal_dp_ready,
                    }
                )

        manifest_path = os.path.join(out_dir, "release_manifest.csv")
        pd.DataFrame(rows).to_csv(manifest_path, index=False)

    print("\n### DONE ###", flush=True)
    print(f"Manifest: {os.path.join(out_dir, 'release_manifest.csv')}", flush=True)


if __name__ == "__main__":
    main()
