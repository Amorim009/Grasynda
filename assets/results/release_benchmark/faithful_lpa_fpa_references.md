# Faithful LPA/FPA Reference Notes

## Primary Paper

Rastogi, V., and Nath, S. (2010). *Differentially Private Aggregation of Distributed Time-Series with Transformation and Encryption*. Proceedings of ACM SIGMOD 2010.

Local source inspected: `C:\Users\lhenr\Downloads\lpa.pdf`

## Implemented Calibration

The central LPA/FPA implementation follows Rastogi and Nath:

- LPA: add independent Laplace noise to each timestamp with scale `Delta_1(Q) / epsilon`.
- FPA_k: DFT the query sequence, retain the first `k` low-frequency coefficients, perturb retained coefficients, pad the rest with zeros, and inverse-transform.

For per-series releases with per-timestamp sensitivity `s` and length `n`, the implemented sequence sensitivities are:

- `Delta_1(Q) = n * s`
- `Delta_2(Q) = sqrt(n) * s`
- LPA scale: `n * s / epsilon`
- Paper FPA complex-coefficient scale: `sqrt(k) * sqrt(n) * s / epsilon`

Implementation note: NumPy's default FFT is not L2-normalized, while the paper's Theorem 4.1 relies on the Fourier transform preserving L2 norm. The code therefore uses orthonormal `rfft`/`irfft` for FPA. Since the implementation perturbs real and imaginary coordinates with ordinary real Laplace noise, it uses the conservative real-coordinate scale `sqrt(2k) * sqrt(n) * s / epsilon`.

The distributed encrypted DLPA protocol from the paper is not implemented here; these experiments implement central LPA/FPA release baselines.

## Implementation Files

- `scripts/experiments/4_release_benchmark/generate_released_train_sets.py`
- `scripts/experiments/4_release_benchmark/run_forecasting_on_releases_gpu.py`
- `scripts/experiments/privacy_experiments/compute_release_privacy_metrics.py`
- `scripts/experiments/privacy_experiments/audit_dp_time_series_privacy.py`
- `scripts/experiments/privacy_experiments/compare_grasynda_vs_dp_anonymization.py`

## DP Guarantee Note

The noise calibration is faithful to the paper formulas, with an implementation-level conservative `sqrt(2)` adjustment for real/imaginary Fourier-coordinate perturbation. A formal DP claim additionally needs public/fixed clipping bounds or a separately private clipping-bound selection. Runs that use data-derived 0.01/0.99 quantile clipping should be described as faithfully calibrated LPA/FPA after data-derived clipping, not as end-to-end formal DP releases unless clipping is fixed/public or privatized.
