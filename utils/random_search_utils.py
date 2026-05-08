import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.model_selection import ParameterSampler

ALL_NHITS_TARGETS: List[Tuple[str, str]] = [
    ("Gluonts", "m1_monthly"),
    ("Gluonts", "m1_quarterly"),
    ("M3", "Monthly"),
    ("M3", "Quarterly"),
    ("NN3", "Monthly"),
    ("Tourism", "Monthly"),
    ("Tourism", "Quarterly"),
]


def is_monthly_group(group: str) -> bool:
    group_lower = str(group).strip().lower()
    return group_lower == "monthly" or group_lower.endswith("_monthly")


def is_quarterly_group(group: str) -> bool:
    group_lower = str(group).strip().lower()
    return group_lower == "quarterly" or group_lower.endswith("_quarterly")


def resolve_horizon_for_search(group: str, fallback: int) -> int:
    # Keep legacy overrides for the plain M3/Tourism labels, but preserve
    # dataset-native horizons for groups such as m1_quarterly.
    if str(group).strip() == "Monthly":
        return 12
    if str(group).strip() == "Quarterly":
        return 8
    return fallback


def stable_target_seed(dataset: str, group: str, base_seed: int) -> int:
    text = f"{dataset}::{group}"
    offset = sum((idx + 1) * ord(ch) for idx, ch in enumerate(text))
    return int(base_seed) + offset


class ChoiceDistribution:
    def __init__(self, values: Sequence[Any]):
        if not values:
            raise ValueError("ChoiceDistribution requires at least one value.")
        self.values = list(values)

    def rvs(self, random_state=None) -> Any:
        rng = _coerce_random_state(random_state)
        idx = int(rng.randint(0, len(self.values)))
        return self.values[idx]


class UniformFloatDistribution:
    def __init__(self, low: float, high: float, decimals: int = 6):
        self.low = float(low)
        self.high = float(high)
        self.decimals = int(decimals)

    def rvs(self, random_state=None) -> float:
        rng = _coerce_random_state(random_state)
        value = rng.uniform(self.low, self.high)
        return round(float(value), self.decimals)


class LogUniformFloatDistribution:
    def __init__(self, low: float, high: float, decimals: int = 6):
        if low <= 0 or high <= 0:
            raise ValueError("Log-uniform bounds must be > 0.")
        self.low = float(low)
        self.high = float(high)
        self.decimals = int(decimals)

    def rvs(self, random_state=None) -> float:
        rng = _coerce_random_state(random_state)
        value = math.exp(rng.uniform(math.log(self.low), math.log(self.high)))
        return round(float(value), self.decimals)


def _coerce_random_state(random_state=None):
    if random_state is None:
        return np.random.RandomState()
    if isinstance(random_state, np.random.RandomState):
        return random_state
    if isinstance(random_state, np.random.Generator):
        return random_state
    return np.random.RandomState(random_state)


def canonicalize_value(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 10)
    return value


def params_key(params: Dict[str, Any], ordered_keys: Iterable[str]) -> Tuple[Any, ...]:
    return tuple(canonicalize_value(params[key]) for key in ordered_keys)


def finite_search_space_size(param_distributions: Dict[str, Any]) -> Optional[int]:
    total = 1
    for values in param_distributions.values():
        if hasattr(values, "rvs"):
            return None
        total *= len(values)
    return total


def build_unique_param_sets(
    param_distributions: Dict[str, Any],
    ordered_keys: Sequence[str],
    n_trials: int,
    seed: int,
    existing_keys: Optional[Iterable[Tuple[Any, ...]]] = None,
    oversample_factor: int = 6,
    normalize_fn=None,
) -> List[Dict[str, Any]]:
    if n_trials <= 0:
        return []

    seen = set(existing_keys or [])
    selected: List[Dict[str, Any]] = []
    finite_total = finite_search_space_size(param_distributions)
    effective_trials = n_trials if finite_total is None else min(n_trials, finite_total)
    requested_unique = max(0, effective_trials - len(seen))
    if requested_unique == 0:
        return []

    batch_size = max(requested_unique * oversample_factor, requested_unique, 50)
    if finite_total is not None:
        batch_size = min(batch_size, finite_total)
    total_requested = 0
    max_total_requests = batch_size * 10 if finite_total is None else finite_total

    while len(selected) < requested_unique and total_requested < max_total_requests:
        current_n_iter = batch_size
        if finite_total is not None:
            current_n_iter = min(batch_size, max_total_requests - total_requested)

        sampler = ParameterSampler(
            param_distributions=param_distributions,
            n_iter=current_n_iter,
            random_state=seed + total_requested,
        )
        total_requested += current_n_iter
        for params in sampler:
            if normalize_fn is not None:
                params = normalize_fn(params)
            key = params_key(params, ordered_keys)
            if key in seen:
                continue
            seen.add(key)
            selected.append(params)
            if len(selected) >= requested_unique:
                break

    if len(selected) < requested_unique:
        raise ValueError(
            f"Unable to draw {effective_trials} unique parameter sets with sklearn ParameterSampler. "
            "Consider widening the search space or reducing n_trials."
        )

    return selected
