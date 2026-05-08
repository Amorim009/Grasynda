"""
Audit direct time-series anonymization baselines from the LPA/FPA/STL-DP family.

This script is standalone and does not modify the existing experiment pipeline.
It adapts the published method family to the repo's per-series forecasting format
(`unique_id`, `ds`, `y`) and reports empirical privacy-risk diagnostics that are
useful in a paper:

- DCR: distance to closest real record
- NNDR: nearest-neighbor distance ratio
- Source-nearest rate: how often the original source record is still the nearest
- Exact-copy rate under a tolerance
- Real-vs-real nearest-neighbor baseline for context
- Optional PyMDMA metrics

Canonical anonymization methods exposed by this script:
- LPA: independent Laplace Perturbation Algorithm baseline
- FPA: Fourier Perturbation Algorithm baseline
- sFPA: STL-DP seasonal-only variant from the STL-DP paper
- tFPA: STL-DP trend-only variant from the STL-DP paper

Primary references:
- Vibhor Rastogi and Suman Nath. 2010.
  "Differentially Private Aggregation of Distributed Time-Series with
  Transformation and Encryption." SIGMOD 2010.
  https://www.microsoft.com/en-us/research/publication/differentially-private-aggregation-of-distributed-time-series-with-transformation-and-encryption-2/
- Kyunghee Kim, Minha Kim, and Simon Woo. 2022.
  "STL-DP: Differentially Private Time Series Exploring Decomposition and
  Compression Methods." CIKM Workshops 2022.
  https://ceur-ws.org/Vol-3318/short5.pdf

Faithfulness note:
- This is a careful benchmark adaptation of the published ideas, not a claim of
  exact reproduction of the original authors' code or dataset-specific setup.
- In particular, clipping, sensitivity choice, and low-frequency coefficient
  count are configurable so the methods can be applied across your datasets.

Formal-DP note:
- If clipping bounds are derived from the private dataset, the release is best
  interpreted as an empirical DP-inspired baseline, not a fully clean formal DP
  release. For a cleaner DP story, pass fixed public bounds using
  --clip-lower and --clip-upper, and optionally --sensitivity-override.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import warnings
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import STL

warnings.filterwarnings("ignore")


# =========================
# Project setup
# =========================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from utils.load_data.config import DATASETS, DATA_GROUPS


# =========================
# Configuration
# =========================

SUPPORTED_METHODS = [
    "LPA",
    "FPA",
    "sFPA",
    "tFPA",
]

METHOD_ALIASES = {
    "DP_Laplace": "LPA",
    "DP_Fourier": "FPA",
    "STL_DP_Seasonal": "sFPA",
    "STL_DP_Trend": "tFPA",
}


@dataclass
class ClipConfig:
    lower: float
    upper: float
    source: str
    formal_dp_ready: bool


def stable_int(*parts: object) -> int:
    payload = "|".join(map(str, parts)).encode("utf-8")
    return int(hashlib.md5(payload).hexdigest()[:8], 16)


def parse_csv_list(value: Optional[str]) -> List[str]:
    if value is None or value.strip() == "":
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def parse_targets(targets_arg: Optional[str]) -> List[Tuple[str, str]]:
    if not targets_arg:
        return list(DATA_GROUPS)

    parsed = []
    for item in parse_csv_list(targets_arg):
        if ":" not in item:
            raise ValueError(
                f"Invalid target '{item}'. Expected DATASET:GROUP, for example M3:Monthly"
            )
        dataset, group = item.split(":", 1)
        parsed.append((dataset.strip(), group.strip()))
    return parsed


def canonicalize_method(method_name: str) -> str:
    canonical = METHOD_ALIASES.get(method_name, method_name)
    if canonical == "STL_DP_TrendSeasonal":
        raise ValueError(
            "STL_DP_TrendSeasonal is not part of the paper-faithful STL-DP baseline family. "
            "Use sFPA and/or tFPA instead."
        )
    return canonical


def resolve_frequency_int(group: str, fallback: Optional[int]) -> int:
    g = str(group).lower()
    if "monthly" in g:
        return 12
    if "quarterly" in g:
        return 4
    if fallback is None:
        raise ValueError(f"Could not infer seasonal period for group '{group}'")
    return int(fallback)


def resolve_stl_period(freq_int: int, series_len: int) -> int:
    period = max(2, int(freq_int))
    if period >= series_len:
        period = max(2, series_len // 2)
    return max(2, period)


def compute_clip_config(
    df_real: pd.DataFrame,
    clip_lower: Optional[float],
    clip_upper: Optional[float],
    lower_quantile: float,
    upper_quantile: float,
) -> ClipConfig:
    if clip_lower is not None and clip_upper is not None:
        return ClipConfig(
            lower=float(clip_lower),
            upper=float(clip_upper),
            source="fixed_public_bounds",
            formal_dp_ready=True,
        )

    lower = float(df_real["y"].quantile(lower_quantile))
    upper = float(df_real["y"].quantile(upper_quantile))
    if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
        lower = float(df_real["y"].min())
        upper = float(df_real["y"].max())

    return ClipConfig(
        lower=lower,
        upper=upper,
        source=f"data_quantiles_{lower_quantile:g}_{upper_quantile:g}",
        formal_dp_ready=False,
    )


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
    """Paper-faithful L1 sensitivity for a length-n recurring query sequence."""
    return float(series_len) * float(per_timestamp_sensitivity)


def sequence_l2_sensitivity(series_len: int, per_timestamp_sensitivity: float) -> float:
    """Paper-faithful L2 sensitivity for a length-n recurring query sequence."""
    return float(np.sqrt(float(series_len)) * float(per_timestamp_sensitivity))


def lpa_noise_scale(series_len: int, per_timestamp_sensitivity: float, epsilon: float) -> float:
    return sequence_l1_sensitivity(series_len, per_timestamp_sensitivity) / max(epsilon, 1e-12)


def fpa_noise_scale(series_len: int, fourier_k: int, per_timestamp_sensitivity: float, epsilon: float) -> float:
    """Real-coordinate FPA scale for an orthonormal Fourier transform.

    Theorem 4.1 gives sqrt(k) * Delta_2(Q) / epsilon for k complex
    coefficients. This implementation perturbs real and imaginary coordinates
    with ordinary real Laplace noise, so it uses the conservative sqrt(2k)
    coordinate bound.
    """
    delta2 = sequence_l2_sensitivity(series_len, per_timestamp_sensitivity)
    return float(np.sqrt(2.0 * float(fourier_k)) * delta2 / max(epsilon, 1e-12))


def dp_laplace_series(
    values: np.ndarray,
    epsilon: float,
    sensitivity: float,
    clip_cfg: ClipConfig,
    rng: np.random.Generator,
) -> np.ndarray:
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

    recon = np.fft.irfft(noisy, n=len(clipped), norm="ortho").real
    return clip_values(recon, clip_cfg)


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
    trend, seasonal, resid = decompose_stl(clipped, freq_int=freq_int)

    if component == "trend":
        trend_noisy = dp_fourier_series(
            trend, epsilon, sensitivity, clip_cfg, rng, fourier_k, keep_ratio
        )
        recon = trend_noisy + seasonal + resid
    elif component == "seasonal":
        seasonal_noisy = dp_fourier_series(
            seasonal, epsilon, sensitivity, clip_cfg, rng, fourier_k, keep_ratio
        )
        recon = trend + seasonal_noisy + resid
    elif component == "trend_seasonal":
        split_eps = max(epsilon, 1e-12) / 2.0
        trend_noisy = dp_fourier_series(
            trend, split_eps, sensitivity, clip_cfg, rng, fourier_k, keep_ratio
        )
        seasonal_noisy = dp_fourier_series(
            seasonal, split_eps, sensitivity, clip_cfg, rng, fourier_k, keep_ratio
        )
        recon = trend_noisy + seasonal_noisy + resid
    else:
        raise ValueError(f"Unsupported STL component mode: {component}")

    return clip_values(recon, clip_cfg)


def anonymize_dataframe(
    df_real: pd.DataFrame,
    method_name: str,
    epsilon: float,
    freq_int: int,
    clip_cfg: ClipConfig,
    fourier_k: Optional[int],
    keep_ratio: float,
    sensitivity_override: Optional[float],
    seed: int,
) -> pd.DataFrame:
    outputs = []
    sensitivity = float(sensitivity_override) if sensitivity_override is not None else (clip_cfg.upper - clip_cfg.lower)

    for uid, uid_df in df_real.groupby("unique_id", sort=False):
        uid_df = uid_df.sort_values("ds").copy()
        values = uid_df["y"].to_numpy(dtype=float)
        canonical_method = canonicalize_method(method_name)
        rng = np.random.default_rng(stable_int(seed, canonical_method, epsilon, uid))

        if canonical_method == "LPA":
            anon = dp_laplace_series(values, epsilon, sensitivity, clip_cfg, rng)
        elif canonical_method == "FPA":
            anon = dp_fourier_series(values, epsilon, sensitivity, clip_cfg, rng, fourier_k, keep_ratio)
        elif canonical_method == "tFPA":
            anon = dp_stl_series(
                values, epsilon, sensitivity, clip_cfg, rng, freq_int, "trend", fourier_k, keep_ratio
            )
        elif canonical_method == "sFPA":
            anon = dp_stl_series(
                values, epsilon, sensitivity, clip_cfg, rng, freq_int, "seasonal", fourier_k, keep_ratio
            )
        else:
            raise ValueError(f"Unsupported method: {canonical_method}")

        uid_df["y"] = anon
        outputs.append(uid_df)

    return pd.concat(outputs, ignore_index=True)


def load_series_matrix(df: pd.DataFrame) -> Tuple[np.ndarray, List[str], int]:
    series = []
    ids = []
    for uid, group_df in df.groupby("unique_id", sort=False):
        ids.append(uid)
        series.append(group_df.sort_values("ds")["y"].to_numpy(dtype=float))

    common_len = int(min(len(row) for row in series))
    matrix = np.asarray([row[-common_len:] for row in series], dtype=float)
    return matrix, ids, common_len


def zscore_with_real_reference(real_matrix: np.ndarray, candidate_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = real_matrix.mean(axis=0, keepdims=True)
    std = real_matrix.std(axis=0, keepdims=True)
    std = np.where(std <= 1e-12, 1.0, std)
    return (real_matrix - mean) / std, (candidate_matrix - mean) / std


def build_nn(real_matrix_z: np.ndarray, n_neighbors: int) -> NearestNeighbors:
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
    nn.fit(real_matrix_z)
    return nn


def safe_ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    den = np.where(np.abs(den) <= 1e-12, np.nan, den)
    return num / den


def summarize_distance_array(prefix: str, values: np.ndarray) -> Dict[str, float]:
    return {
        f"{prefix}_Mean": float(np.nanmean(values)),
        f"{prefix}_Median": float(np.nanmedian(values)),
        f"{prefix}_Min": float(np.nanmin(values)),
        f"{prefix}_P05": float(np.nanpercentile(values, 5)),
        f"{prefix}_P95": float(np.nanpercentile(values, 95)),
    }


def compute_exact_copy_rate(
    real_matrix: np.ndarray,
    candidate_matrix: np.ndarray,
    ids: Sequence[str],
    tolerance: float,
) -> Tuple[float, float]:
    id_to_idx = {uid: idx for idx, uid in enumerate(ids)}
    exact_matches = 0
    near_matches = 0

    for idx, uid in enumerate(ids):
        real_row = real_matrix[id_to_idx[uid]]
        cand_row = candidate_matrix[idx]
        max_abs_diff = float(np.max(np.abs(cand_row - real_row)))
        if max_abs_diff <= tolerance:
            exact_matches += 1
        if max_abs_diff <= max(tolerance, 1e-3):
            near_matches += 1

    n = len(ids)
    return exact_matches / n, near_matches / n


def compute_nn_privacy_metrics(
    real_matrix: np.ndarray,
    candidate_matrix: np.ndarray,
    ids: Sequence[str],
    tolerance: float,
) -> Dict[str, float]:
    real_z, cand_z = zscore_with_real_reference(real_matrix, candidate_matrix)

    nn_real = build_nn(real_z, n_neighbors=min(2, len(real_z)))
    rr_dist, rr_idx = nn_real.kneighbors(real_z)

    if rr_dist.shape[1] == 1:
        real_nn = rr_dist[:, 0]
    else:
        real_nn = rr_dist[:, 1]

    nn_cand = build_nn(real_z, n_neighbors=min(2, len(real_z)))
    cr_dist, cr_idx = nn_cand.kneighbors(cand_z)
    d1 = cr_dist[:, 0]
    d2 = cr_dist[:, 1] if cr_dist.shape[1] > 1 else np.full_like(d1, np.nan)
    nndr = safe_ratio(d1, d2)

    id_to_idx = {uid: idx for idx, uid in enumerate(ids)}
    nearest_is_source = []
    source_rank_distance = []
    source_is_top2 = []

    for idx, uid in enumerate(ids):
        source_idx = id_to_idx[uid]
        neigh_idx = cr_idx[idx]
        neigh_dist = cr_dist[idx]
        nearest_is_source.append(int(neigh_idx[0] == source_idx))
        source_is_top2.append(int(source_idx in neigh_idx[: min(2, len(neigh_idx))]))

        match_pos = np.where(neigh_idx == source_idx)[0]
        if len(match_pos) > 0:
            source_rank_distance.append(float(neigh_dist[int(match_pos[0])]))
        else:
            source_rank_distance.append(np.nan)

    exact_copy_rate, near_copy_rate = compute_exact_copy_rate(real_matrix, candidate_matrix, ids, tolerance)

    metrics = {}
    metrics.update(summarize_distance_array("RealNN", real_nn))
    metrics.update(summarize_distance_array("DCR", d1))
    metrics.update(summarize_distance_array("NNDR", nndr))
    metrics.update(summarize_distance_array("SourceDist", np.asarray(source_rank_distance, dtype=float)))
    metrics["DCR_to_RealNN_MeanRatio"] = float(np.nanmean(d1) / max(np.nanmean(real_nn), 1e-12))
    metrics["NearestIsSourceRate"] = float(np.mean(nearest_is_source))
    metrics["SourceInTop2Rate"] = float(np.mean(source_is_top2))
    metrics["ExactCopyRate"] = float(exact_copy_rate)
    metrics["NearCopyRate"] = float(near_copy_rate)
    metrics["SeriesCount"] = int(len(ids))
    metrics["CommonLength"] = int(real_matrix.shape[1])
    return metrics


def compute_pymdma_metrics(real_matrix: np.ndarray, candidate_matrix: np.ndarray, ids: Sequence[str]) -> Dict[str, float]:
    try:
        from pymdma.tabular.measures.synthesis_val import (
            Authenticity,
            DCRPrivacy,
            ImprovedPrecision,
            ImprovedRecall,
        )
    except ImportError as exc:
        raise RuntimeError(f"PyMDMA not available: {exc}")

    del ids
    scaler = StandardScaler()
    real_scaled = scaler.fit_transform(np.asarray(real_matrix, dtype=float))
    cand_scaled = scaler.transform(np.asarray(candidate_matrix, dtype=float))

    context = {}
    k = max(1, min(5, len(real_scaled) - 2, len(cand_scaled) - 2))
    auth = Authenticity(n_workers=1)
    imp_prec = ImprovedPrecision(k=k, n_workers=1)
    imp_rec = ImprovedRecall(k=k, n_workers=1)
    dcr_priv = DCRPrivacy()

    return {
        "PyMDMA_Authenticity": float(auth.compute(real_scaled, cand_scaled, context=context).value[0]),
        "PyMDMA_Fidelity": float(imp_prec.compute(real_scaled, cand_scaled, context=context).value[0]),
        "PyMDMA_Diversity": float(imp_rec.compute(real_scaled, cand_scaled, context=context).value[0]),
        "PyMDMA_Privacy": float(dcr_priv.compute(real_scaled, cand_scaled).value[0]["privacy"] / 100),
    }


def run_audit(args) -> str:
    targets = parse_targets(args.targets)
    methods = parse_csv_list(args.methods) or SUPPORTED_METHODS
    methods = [canonicalize_method(method) for method in methods]
    methods = list(dict.fromkeys(methods))
    epsilons = [float(x) for x in parse_csv_list(args.epsilons)] or [0.48, 2.4, 4.8, 24.0]

    for method in methods:
        if method not in SUPPORTED_METHODS:
            raise ValueError(f"Unsupported method '{method}'")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = args.output_tag or f"dp_time_series_privacy_audit_{timestamp}"
    out_dir = os.path.join(PROJECT_ROOT, "assets", "results", "dp_privacy_audit", tag)
    os.makedirs(out_dir, exist_ok=True)

    config = {
        "created_at": timestamp,
        "targets": targets,
        "methods": methods,
        "epsilons": epsilons,
        "sample_n_uids": args.sample_n_uids,
        "clip_lower": args.clip_lower,
        "clip_upper": args.clip_upper,
        "sensitivity_override": args.sensitivity_override,
        "lower_quantile": args.lower_quantile,
        "upper_quantile": args.upper_quantile,
        "fourier_k": args.fourier_k,
        "fourier_keep_ratio": args.fourier_keep_ratio,
        "dp_calibration": "Rastogi_Nath_2010_sequence_l1_l2_orthonormal_real_coordinate_fpa",
        "sensitivity_interpretation": (
            "sensitivity_override or clip range is treated as per-timestamp sensitivity; "
            "LPA lambda=n*s/epsilon; FPA uses orthonormal RFFT and real/imag coordinate "
            "Laplace with lambda=sqrt(2k)*sqrt(n)*s/epsilon"
        ),
        "copy_tolerance": args.copy_tolerance,
        "seed": args.seed,
        "with_pymdma": args.with_pymdma,
        "note": "Formal DP claims are cleaner when fixed public clipping bounds are supplied.",
    }
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)

    rows = []

    for dataset_name, group in targets:
        print(f"\n### {dataset_name} - {group} ###", flush=True)
        loader = DATASETS[dataset_name]
        df_real, _, _, _, freq_int = loader.load_everything(group, sample_n_uid=args.sample_n_uids)
        freq_int = resolve_frequency_int(group, freq_int)
        clip_cfg = compute_clip_config(
            df_real,
            clip_lower=args.clip_lower,
            clip_upper=args.clip_upper,
            lower_quantile=args.lower_quantile,
            upper_quantile=args.upper_quantile,
        )
        print(
            f"Clip bounds: [{clip_cfg.lower:.4f}, {clip_cfg.upper:.4f}] "
            f"from {clip_cfg.source}; formal_dp_ready={clip_cfg.formal_dp_ready}",
            flush=True,
        )

        for method_name in methods:
            for epsilon in epsilons:
                print(f"  -> {method_name} | epsilon={epsilon}", flush=True)
                start = time.time()
                try:
                    candidate_df = anonymize_dataframe(
                        df_real=df_real,
                        method_name=method_name,
                        epsilon=epsilon,
                        freq_int=freq_int,
                        clip_cfg=clip_cfg,
                        fourier_k=args.fourier_k,
                        keep_ratio=args.fourier_keep_ratio,
                        sensitivity_override=args.sensitivity_override,
                        seed=args.seed,
                    )
                    real_matrix, ids, common_len = load_series_matrix(df_real)
                    cand_matrix, cand_ids, cand_len = load_series_matrix(candidate_df)
                    if ids != cand_ids:
                        raise ValueError("Candidate series ids no longer align with the original ids.")
                    if common_len != cand_len:
                        use_len = min(common_len, cand_len)
                        real_matrix = real_matrix[:, -use_len:]
                        cand_matrix = cand_matrix[:, -use_len:]

                    metrics = compute_nn_privacy_metrics(
                        real_matrix=real_matrix,
                        candidate_matrix=cand_matrix,
                        ids=ids,
                        tolerance=args.copy_tolerance,
                    )
                    if args.with_pymdma:
                        metrics.update(compute_pymdma_metrics(real_matrix, cand_matrix, ids))

                    row = {
                        "Dataset": dataset_name,
                        "Group": group,
                        "Method": method_name,
                        "Epsilon": epsilon,
                        "Clip_Lower": clip_cfg.lower,
                        "Clip_Upper": clip_cfg.upper,
                        "Clip_Source": clip_cfg.source,
                        "Formal_DP_Ready": clip_cfg.formal_dp_ready,
                        "Time_Sec": time.time() - start,
                    }
                    row.update(metrics)
                    rows.append(row)
                    print(
                        f"     ok DCR_mean={row['DCR_Mean']:.4f} "
                        f"NNDR_mean={row['NNDR_Mean']:.4f} "
                        f"nearest_source={row['NearestIsSourceRate']:.4f}",
                        flush=True,
                    )
                except Exception as exc:
                    print(f"     failed: {exc}", flush=True)

        if rows:
            pd.DataFrame(rows).to_csv(os.path.join(out_dir, "audit_checkpoint.csv"), index=False)

    if not rows:
        raise RuntimeError("No results were produced.")

    detailed = pd.DataFrame(rows)
    detailed_path = os.path.join(out_dir, "audit_detailed.csv")
    detailed.to_csv(detailed_path, index=False)

    agg_cols = [
        "DCR_Mean",
        "DCR_Median",
        "NNDR_Mean",
        "NearestIsSourceRate",
        "SourceInTop2Rate",
        "ExactCopyRate",
        "NearCopyRate",
        "DCR_to_RealNN_MeanRatio",
        "Time_Sec",
    ]
    if args.with_pymdma:
        agg_cols.extend([
            "PyMDMA_Authenticity",
            "PyMDMA_Fidelity",
            "PyMDMA_Diversity",
            "PyMDMA_Privacy",
        ])

    summary = (
        detailed.groupby(["Method", "Epsilon"], dropna=False)[agg_cols]
        .mean()
        .reset_index()
        .sort_values(["Method", "Epsilon"])
    )
    summary_path = os.path.join(out_dir, "audit_summary.csv")
    summary.to_csv(summary_path, index=False)

    print("\n### DONE ###", flush=True)
    print(f"Detailed results: {detailed_path}", flush=True)
    print(f"Summary results:  {summary_path}", flush=True)
    return out_dir


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit published LPA/FPA/STL-DP-style time-series anonymization baselines with nearest-neighbor disclosure metrics."
    )
    parser.add_argument(
        "--targets",
        type=str,
        default=None,
        help="Comma-separated DATASET:GROUP values, e.g. M3:Monthly,Tourism:Quarterly. Default: all targets.",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default="tFPA,sFPA,FPA,LPA",
        help=f"Comma-separated methods. Supported: {', '.join(SUPPORTED_METHODS)}",
    )
    parser.add_argument(
        "--epsilons",
        type=str,
        default="0.48,2.4,4.8,24.0",
        help="Comma-separated privacy budgets. The default values follow the STL-DP paper's experimental sweep.",
    )
    parser.add_argument("--sample-n-uids", type=int, default=None, help="Optional small subset for smoke tests.")
    parser.add_argument("--output-tag", type=str, default=None, help="Optional output tag.")
    parser.add_argument("--seed", type=int, default=42, help="Base RNG seed.")
    parser.add_argument(
        "--clip-lower",
        type=float,
        default=None,
        help="Fixed public lower clipping bound.",
    )
    parser.add_argument(
        "--clip-upper",
        type=float,
        default=None,
        help="Fixed public upper clipping bound.",
    )
    parser.add_argument(
        "--sensitivity-override",
        type=float,
        default=None,
        help="Optional fixed sensitivity override. Useful when you want to separate DP sensitivity from the clip range.",
    )
    parser.add_argument(
        "--lower-quantile",
        type=float,
        default=0.01,
        help="Lower quantile used when clip bounds are estimated from private data.",
    )
    parser.add_argument(
        "--upper-quantile",
        type=float,
        default=0.99,
        help="Upper quantile used when clip bounds are estimated from private data.",
    )
    parser.add_argument(
        "--fourier-k",
        type=int,
        default=None,
        help="Number of low-frequency Fourier coefficients to keep. Default: auto from --fourier-keep-ratio.",
    )
    parser.add_argument(
        "--fourier-keep-ratio",
        type=float,
        default=0.15,
        help="Ratio used to choose low-frequency Fourier coefficients when --fourier-k is omitted.",
    )
    parser.add_argument(
        "--copy-tolerance",
        type=float,
        default=1e-8,
        help="Tolerance used for exact-copy detection.",
    )
    parser.add_argument(
        "--with-pymdma",
        action="store_true",
        help="Also compute PyMDMA metrics. Slower but useful for comparison with your existing results.",
    )
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    run_audit(args)


if __name__ == "__main__":
    main()
