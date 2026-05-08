from typing import Any, Dict

from utils.random_search_utils import (
    LogUniformFloatDistribution,
    UniformFloatDistribution,
    is_monthly_group,
    is_quarterly_group,
)


def _is_m1(dataset: str, group: str) -> bool:
    return dataset == "Gluonts" and str(group).lower().startswith("m1_")


def grasynda_search_space(dataset: str, group: str) -> Dict[str, Any]:
    if is_quarterly_group(group):
        n_quantiles = [3, 5, 8, 10, 12, 15, 20, 30, 50]
        ensemble_size = [2, 3, 5, 8, 10]
    else:
        n_quantiles = [3, 5, 8, 10, 12, 15, 20, 30, 50, 75, 100]
        ensemble_size = [2, 3, 5, 8, 10, 15, 20]

    if dataset == "NN3":
        n_quantiles = [5, 8, 10, 12, 15, 20, 30, 50]

    return {
        "n_quantiles": n_quantiles,
        "ensemble_transitions": [False, True],
        "ensemble_size": ensemble_size,
    }


def seasonalmbb_search_space(dataset: str, group: str) -> Dict[str, Any]:
    chunk_choices = [None, 128, 256, 512]
    if dataset == "NN3":
        chunk_choices = [None, 64, 128, 256]
    if _is_m1(dataset, group):
        chunk_choices = [None, 64, 128, 256]

    return {
        "log": [True, False],
        "max_samples_in_stl": chunk_choices,
    }


def jittering_search_space(dataset: str, group: str) -> Dict[str, Any]:
    high = 0.12 if is_quarterly_group(group) else 0.18
    if _is_m1(dataset, group):
        high = 0.12
    return {
        "sigma": LogUniformFloatDistribution(0.003, high, decimals=4),
    }


def scaling_search_space(dataset: str, group: str) -> Dict[str, Any]:
    high = 0.3 if is_monthly_group(group) else 0.2
    if _is_m1(dataset, group):
        high = 0.2
    return {
        "sigma": LogUniformFloatDistribution(0.01, high, decimals=4),
    }


def timewarping_search_space(dataset: str, group: str) -> Dict[str, Any]:
    sigma_high = 0.25 if is_monthly_group(group) else 0.18
    if _is_m1(dataset, group):
        sigma_high = 0.16
    return {
        "sigma": LogUniformFloatDistribution(0.01, sigma_high, decimals=4),
        "knot": [2, 3, 4, 5, 6],
    }


def magnitudewarping_search_space(dataset: str, group: str) -> Dict[str, Any]:
    sigma_low, sigma_high = (0.01, 0.14) if is_monthly_group(group) else (0.01, 0.18)
    if _is_m1(dataset, group):
        sigma_low, sigma_high = (0.01, 0.12) if is_monthly_group(group) else (0.01, 0.10)

    return {
        "sigma": LogUniformFloatDistribution(sigma_low, sigma_high, decimals=4),
        "knot": [2, 3, 4, 5, 6],
    }


def tsmixup_search_space(dataset: str, group: str) -> Dict[str, Any]:
    max_n_upper = 8 if is_monthly_group(group) else 5
    if dataset == "NN3":
        max_n_upper = 6
    if _is_m1(dataset, group):
        max_n_upper = 6 if is_monthly_group(group) else 4

    alpha_high = 8.0 if is_monthly_group(group) else 5.0
    factor_high = 2.5 if is_monthly_group(group) else 2.0

    return {
        "max_n_uids": list(range(1, max_n_upper + 1)),
        "dirichlet_alpha": LogUniformFloatDistribution(0.2, alpha_high, decimals=4),
        "n_series_factor": UniformFloatDistribution(0.5, factor_high, decimals=2),
    }


def dba_search_space(dataset: str, group: str) -> Dict[str, Any]:
    max_n_upper = 5 if is_quarterly_group(group) else 6
    if _is_m1(dataset, group):
        max_n_upper = 4

    iter_choices = [5, 8, 10, 12, 15, 20, 25, 30, 40]
    if is_quarterly_group(group):
        iter_choices = [5, 8, 10, 12, 15, 20, 25, 30]

    return {
        "max_n_uids": list(range(1, max_n_upper + 1)),
        "dirichlet_alpha": LogUniformFloatDistribution(0.35, 5.0, decimals=4),
        "dba_iter": iter_choices,
    }


def timevae_search_space(dataset: str, group: str) -> Dict[str, Any]:
    latent_choices = [4, 8, 16, 24, 32]
    epoch_choices = [50, 80, 100, 150, 200]

    if is_quarterly_group(group):
        latent_choices = [4, 8, 12, 16, 24]
        epoch_choices = [40, 60, 80, 100, 150]

    if _is_m1(dataset, group):
        epoch_choices = [40, 60, 80, 100, 150]

    return {
        "latent_dim": latent_choices,
        "reconstruction_wt": [1.0, 2.0, 3.0, 5.0, 7.5],
        "max_epochs": epoch_choices,
    }


def tsdiff_search_space(dataset: str, group: str) -> Dict[str, Any]:
    epoch_choices = [10, 15, 20, 30, 40]
    if _is_m1(dataset, group):
        epoch_choices = [8, 10, 15, 20, 30]

    return {
        "max_epochs": epoch_choices,
        "learning_rate": [1e-4, 3e-4, 5e-4, 1e-3, 2e-3],
        "num_batches_per_epoch": [32, 48, 64, 96, 128],
    }


DEFAULT_RANDOM_SEARCH_TRIALS = 32


def default_trials_for_method(method_name: str, dataset: str, group: str) -> int:
    return DEFAULT_RANDOM_SEARCH_TRIALS
