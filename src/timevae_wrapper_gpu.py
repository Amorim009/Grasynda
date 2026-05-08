import random

import numpy as np
import pandas as pd
import tensorflow as tf

from src.timevae.vae_utils import instantiate_vae_model, train_vae


class MaskedGlobalMinMaxScaler:
    def __init__(self):
        self.mini = None
        self.range = None

    def fit(self, data: np.ndarray, mask: np.ndarray):
        n, t, f = data.shape
        d2 = data.reshape(-1, f)
        m = mask.reshape(-1) > 0
        if not np.any(m):
            raise ValueError("Cannot fit TimeVAE scaler: mask has no observed values.")
        self.mini = np.min(d2[m], axis=0, keepdims=True).astype(np.float32)
        self.range = (
            np.max(d2[m], axis=0, keepdims=True).astype(np.float32) - self.mini
        )
        self.range[self.range < 1e-7] = 1.0
        return self

    def fit_transform(self, data: np.ndarray, mask: np.ndarray) -> np.ndarray:
        self.fit(data, mask)
        return self.transform(data, mask)

    def transform(self, data: np.ndarray, mask: np.ndarray) -> np.ndarray:
        n, t, f = data.shape
        d2 = data.reshape(-1, f).astype(np.float32)
        out = np.zeros_like(d2, dtype=np.float32)
        m = mask.reshape(-1) > 0
        out[m] = (d2[m] - self.mini) / self.range
        return out.reshape(n, t, f)

    def inverse_transform(self, data: np.ndarray, mask: np.ndarray) -> np.ndarray:
        n, t, f = data.shape
        d2 = data.reshape(-1, f).astype(np.float32)
        out = np.zeros_like(d2, dtype=np.float32)
        m = mask.reshape(-1) > 0
        out[m] = d2[m] * self.range + self.mini
        return out.reshape(n, t, f)


def masked_mse_per_series(x: np.ndarray, xhat: np.ndarray, mask: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    xhat = np.asarray(xhat, dtype=np.float32)
    mask = np.asarray(mask, dtype=np.float32)
    if mask.ndim == 3:
        mask = mask[..., 0]
    mask3 = mask[..., None]
    num = np.sum(((x - xhat) ** 2) * mask3, axis=(1, 2))
    den = (np.sum(mask3, axis=(1, 2)) * x.shape[2]) + 1e-8
    return (num / den).astype(np.float32)


def masked_mse_np(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    mask = np.asarray(mask, dtype=np.float32)
    if mask.ndim == 3:
        mask = mask[..., 0]
    mask3 = mask[..., None]
    num = np.sum(((a - b) ** 2) * mask3)
    den = np.sum(mask3) * a.shape[2] + 1e-8
    return float(num / den)


def make_alpha_per_series(
    alpha_base: float,
    rec_err_per_series: np.ndarray,
    clip: tuple[float, float] = (0.8, 1.2),
) -> np.ndarray:
    med = np.median(rec_err_per_series) + 1e-12
    scale = np.sqrt(rec_err_per_series / med).astype(np.float32)
    lo, hi = clip
    scale = np.clip(scale, lo, hi)
    return (alpha_base * scale).astype(np.float32)


class TimeVAEWrapper:
    def __init__(
        self,
        latent_dim=8,
        hidden_layer_sizes=None,
        reconstruction_wt=3.0,
        batch_size=16,
        max_epochs=100,
        window_size=None,
        alpha_base=0.1,
        alpha_candidates=None,
        alpha_target_ratio=0.20,
        auto_tune_alpha=True,
        seas_period=None,
        trend_poly=0,
        custom_seas=None,
        use_residual_conn=True,
        enable_length_bucketing=True,
        max_length_ratio_per_bucket=1.35,
        min_bucket_size=24,
        max_buckets=6,
        random_seed=42,
        device="auto",
        require_gpu=False,
    ):
        if hidden_layer_sizes is None:
            hidden_layer_sizes = [50, 100, 200]
        if alpha_candidates is None:
            alpha_candidates = [0.01, 0.02, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.5, 0.8, 1.0]

        self.latent_dim = latent_dim
        self.hidden_layer_sizes = hidden_layer_sizes
        self.reconstruction_wt = reconstruction_wt
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.window_size = None if window_size in (None, 0) else int(window_size)
        self.alpha_base = float(alpha_base)
        self.alpha_candidates = [float(a) for a in alpha_candidates]
        self.alpha_target_ratio = float(alpha_target_ratio)
        self.auto_tune_alpha = bool(auto_tune_alpha)
        self.seas_period = seas_period
        self.trend_poly = int(trend_poly)
        self.custom_seas = custom_seas
        self.use_residual_conn = bool(use_residual_conn)
        self.enable_length_bucketing = bool(enable_length_bucketing)
        self.max_length_ratio_per_bucket = float(max_length_ratio_per_bucket)
        self.min_bucket_size = int(min_bucket_size)
        self.max_buckets = int(max_buckets)
        self.random_seed = int(random_seed)
        self.device = device
        self.require_gpu = bool(require_gpu)

        self.vae_model = None
        self.scaler = MaskedGlobalMinMaxScaler()
        self._transform_calls = 0

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        call_id = self._transform_calls
        self._transform_calls += 1

        print(f"[TimeVAE] Starting with {df['unique_id'].nunique()} series")
        lengths_by_uid = df.groupby("unique_id").size().sort_index()
        buckets = self._build_length_buckets(lengths_by_uid)
        bucket_sizes = [len(bucket) for bucket in buckets]
        print(
            f"[TimeVAE] Using {len(buckets)} length bucket(s): "
            + ", ".join(str(size) for size in bucket_sizes)
        )

        synth_parts = []
        for bucket_id, bucket_uids in enumerate(buckets):
            bucket_df = df[df["unique_id"].isin(bucket_uids)].copy()
            synth_parts.append(
                self._transform_bucket(
                    df=bucket_df,
                    call_id=call_id,
                    bucket_id=bucket_id,
                )
            )

        synthetic_df = pd.concat(synth_parts, ignore_index=True)
        print(
            f"[TimeVAE] Complete! Generated {synthetic_df['unique_id'].nunique()} series"
        )
        return synthetic_df

    def _transform_bucket(self, df: pd.DataFrame, call_id: int, bucket_id: int) -> pd.DataFrame:
        uids = sorted(df["unique_id"].unique())
        (
            data_3d,
            mask_3d,
            seq_len,
            original_dates_per_series,
            original_lengths,
        ) = self._df_to_3d_array(df, uids)
        print(
            f"[TimeVAE] bucket={bucket_id} series={len(uids)} "
            f"shape={data_3d.shape} seq_len={seq_len}"
        )

        self.scaler = MaskedGlobalMinMaxScaler()
        scaled_3d = self.scaler.fit_transform(data_3d, mask_3d)
        scaled_3d = scaled_3d * mask_3d

        device = self._resolve_device()
        print(f"[TimeVAE] device={device}")

        with tf.device(device):
            self.vae_model = instantiate_vae_model(
                vae_type="timeVAE",
                sequence_length=seq_len,
                feature_dim=1,
                batch_size=self.batch_size,
                latent_dim=self.latent_dim,
                hidden_layer_sizes=self.hidden_layer_sizes,
                reconstruction_wt=self.reconstruction_wt,
                trend_poly=self.trend_poly,
                custom_seas=self._resolve_custom_seas(),
                use_residual_conn=self.use_residual_conn,
            )

            train_vae(
                vae=self.vae_model,
                train_data=scaled_3d,
                max_epochs=self.max_epochs,
                verbose=0,
                train_mask=mask_3d,
            )

            alpha_base = self.alpha_base
            z_chol_small = self._estimate_latent_cholesky(
                scaled_3d, mask_3d, max_n=2048
            )
            if self.auto_tune_alpha:
                alpha_base, rec_mse, var_mse, ratio = self._pick_alpha(
                    scaled_3d,
                    mask_3d,
                    self.alpha_candidates,
                    z_chol_small,
                    target_ratio=self.alpha_target_ratio,
                    seed=self.random_seed + call_id + bucket_id,
                )
                print(
                    f"[TimeVAE] bucket={bucket_id} alpha={alpha_base:.4f} "
                    f"(recon_mse={rec_mse:.6f}, variant_mse={var_mse:.6f}, ratio={ratio:.4f})"
                )

            z_chol = self._estimate_latent_cholesky(scaled_3d, mask_3d, max_n=4096)
            variants_scaled, _, _ = self._generate_variants_scaled(
                scaled_3d,
                mask_3d,
                n_variants=1,
                alpha_base=alpha_base,
                z_chol=z_chol,
                seed=self.random_seed + call_id + bucket_id,
                adaptive=True,
                adaptive_clip=(0.8, 1.2),
            )

        synthetic_3d = self.scaler.inverse_transform(
            variants_scaled[:, 0], mask_3d
        ) * mask_3d

        return self._3d_array_to_df(
            synthetic_3d,
            uids,
            original_dates_per_series,
            original_lengths,
            suffix=f"timevae_{call_id}",
        )

    def _resolve_device(self) -> str:
        requested = "auto" if self.device is None else str(self.device).strip()
        requested_lower = requested.lower()
        gpu_devices = tf.config.list_physical_devices("GPU")

        if requested_lower == "auto":
            resolved = "/GPU:0" if gpu_devices else "/CPU:0"
        elif requested_lower in {"gpu", "cuda", "cuda:0", "/gpu:0"}:
            resolved = "/GPU:0"
        elif requested_lower in {"cpu", "/cpu:0"}:
            resolved = "/CPU:0"
        elif requested_lower.startswith("cuda:"):
            resolved = f"/GPU:{requested_lower.split(':', 1)[1]}"
        elif requested_lower.startswith("/gpu:"):
            resolved = f"/GPU:{requested_lower.split(':', 1)[1]}"
        elif requested_lower.startswith("/cpu:"):
            resolved = f"/CPU:{requested_lower.split(':', 1)[1]}"
        else:
            resolved = requested

        if resolved.startswith("/GPU:") and not gpu_devices:
            if self.require_gpu or requested_lower != "auto":
                raise RuntimeError(
                    "TimeVAE requested GPU execution, but TensorFlow does not detect a GPU."
                )
            return "/CPU:0"

        return resolved

    def _df_to_3d_array(self, df: pd.DataFrame, uids: list[str]):
        grouped = df.groupby("unique_id")
        series_lengths = grouped.size()
        seq_len = (
            int(self.window_size)
            if self.window_size is not None
            else int(series_lengths.max())
        )

        n_series = len(uids)
        array_3d = np.zeros((n_series, seq_len, 1), dtype=np.float32)
        mask_3d = np.zeros((n_series, seq_len, 1), dtype=np.float32)
        original_dates_per_series = []
        original_lengths = np.zeros(n_series, dtype=np.int64)

        for i, uid in enumerate(uids):
            group = grouped.get_group(uid).sort_values("ds")
            values = group["y"].to_numpy(dtype=np.float32)
            dates = pd.to_datetime(group["ds"]).reset_index(drop=True)

            if len(values) > seq_len:
                values = values[-seq_len:]
                dates = dates.iloc[-seq_len:].reset_index(drop=True)

            n = len(values)
            array_3d[i, :n, 0] = values
            mask_3d[i, :n, 0] = 1.0
            original_dates_per_series.append(dates)
            original_lengths[i] = n

        return array_3d, mask_3d, seq_len, original_dates_per_series, original_lengths

    def _estimate_latent_cholesky(
        self, X_scaled: np.ndarray, mask: np.ndarray, max_n: int = 4096
    ) -> np.ndarray:
        Xs = X_scaled[:max_n] * mask[:max_n]
        z_mean, _, _ = self.vae_model.encoder(Xs, training=False)
        z = z_mean.numpy().astype(np.float32)
        mu = z.mean(axis=0)
        zc = z - mu[None, :]
        cov = (zc.T @ zc) / max(1, zc.shape[0] - 1)
        cov += 1e-6 * np.eye(cov.shape[0], dtype=np.float32)
        return np.linalg.cholesky(cov).astype(np.float32)

    def _generate_variants_scaled(
        self,
        X_scaled: np.ndarray,
        mask: np.ndarray,
        n_variants: int,
        alpha_base: float,
        z_chol: np.ndarray,
        seed: int,
        adaptive: bool,
        adaptive_clip: tuple[float, float] = (0.8, 1.2),
    ):
        X_scaled = np.asarray(X_scaled, dtype=np.float32)
        mask = np.asarray(mask, dtype=np.float32)
        X_in = X_scaled * mask

        X_rec = self.vae_model.predict(X_in, verbose=0).astype(np.float32)
        X_rec = X_rec * mask
        rec_err = masked_mse_per_series(X_in, X_rec, mask)

        if adaptive:
            alpha_i = make_alpha_per_series(alpha_base, rec_err, clip=adaptive_clip)
        else:
            alpha_i = np.full((X_in.shape[0],), float(alpha_base), dtype=np.float32)

        z_mean, _, _ = self.vae_model.encoder(X_in, training=False)
        z_mean = z_mean.numpy().astype(np.float32)
        n, latent_dim = z_mean.shape

        rng = np.random.default_rng(seed)
        eps = rng.normal(size=(n, n_variants, latent_dim)).astype(np.float32)
        noise = eps @ z_chol.T
        z = z_mean[:, None, :] + (alpha_i[:, None, None] * noise)
        z_flat = z.reshape(n * n_variants, latent_dim)

        X_var = self.vae_model.decoder.predict(z_flat, verbose=0).astype(np.float32)
        T, F = X_var.shape[1], X_var.shape[2]
        X_var = X_var.reshape(n, n_variants, T, F)
        X_var = X_var * mask[:, None, :, :]
        return X_var, alpha_i, rec_err

    def _pick_alpha(
        self,
        X_scaled: np.ndarray,
        mask: np.ndarray,
        alpha_candidates: list[float],
        z_chol_small: np.ndarray,
        target_ratio: float,
        seed: int,
        ncheck: int = 256,
    ):
        Xs = X_scaled[:ncheck]
        Ms = mask[:ncheck]
        X_rec = self.vae_model.predict(Xs, verbose=0).astype(np.float32)
        base = masked_mse_np(Xs, X_rec, Ms)

        best = None
        target = 1.0 + float(target_ratio)

        for alpha in alpha_candidates:
            V, _, _ = self._generate_variants_scaled(
                Xs,
                Ms,
                n_variants=1,
                alpha_base=alpha,
                z_chol=z_chol_small,
                seed=seed,
                adaptive=False,
                adaptive_clip=(1.0, 1.0),
            )
            V = V[:, 0]
            mse = masked_mse_np(Xs, V, Ms)
            ratio = mse / (base + 1e-12)
            score = abs(ratio - target)
            if best is None or score < best[0]:
                best = (score, alpha, mse, ratio, base)

        _, alpha_best, mse_best, ratio_best, base = best
        return alpha_best, base, mse_best, ratio_best

    def _resolve_custom_seas(self):
        if self.custom_seas is not None:
            return self.custom_seas
        if self.seas_period is None:
            return None
        try:
            period = int(self.seas_period)
        except Exception:
            return None
        if period > 1:
            return [(period, 1)]
        return None

    def _build_length_buckets(self, lengths_by_uid: pd.Series):
        items = sorted(
            ((str(uid), int(length)) for uid, length in lengths_by_uid.items()),
            key=lambda item: item[1],
        )
        if not items:
            return []

        if (not self.enable_length_bucketing) or len(items) <= self.min_bucket_size:
            return [[uid for uid, _ in items]]

        global_ratio = items[-1][1] / max(1, items[0][1])
        if global_ratio <= self.max_length_ratio_per_bucket:
            return [[uid for uid, _ in items]]

        buckets = []
        current = []
        cur_min = None
        cur_max = None

        for uid, length in items:
            next_min = length if cur_min is None else min(cur_min, length)
            next_max = length if cur_max is None else max(cur_max, length)
            next_ratio = next_max / max(1, next_min)

            if current and next_ratio > self.max_length_ratio_per_bucket and len(current) >= self.min_bucket_size:
                buckets.append(current)
                current = [(uid, length)]
                cur_min = length
                cur_max = length
            else:
                current.append((uid, length))
                cur_min = next_min
                cur_max = next_max

        if current:
            buckets.append(current)

        buckets = self._merge_small_buckets(buckets)

        while len(buckets) > self.max_buckets:
            best_idx = None
            best_ratio = None
            for i in range(len(buckets) - 1):
                merged = buckets[i] + buckets[i + 1]
                merged_lengths = [length for _, length in merged]
                ratio = max(merged_lengths) / max(1, min(merged_lengths))
                if best_ratio is None or ratio < best_ratio:
                    best_ratio = ratio
                    best_idx = i
            buckets[best_idx] = buckets[best_idx] + buckets[best_idx + 1]
            del buckets[best_idx + 1]

        return [[uid for uid, _ in bucket] for bucket in buckets]

    def _merge_small_buckets(self, buckets):
        if len(buckets) <= 1:
            return buckets

        buckets = [list(bucket) for bucket in buckets]
        changed = True
        while changed and len(buckets) > 1:
            changed = False
            for idx, bucket in enumerate(list(buckets)):
                if len(bucket) >= self.min_bucket_size:
                    continue

                candidates = []
                if idx > 0:
                    merged = buckets[idx - 1] + bucket
                    lengths = [length for _, length in merged]
                    candidates.append((max(lengths) / max(1, min(lengths)), idx - 1))
                if idx < len(buckets) - 1:
                    merged = bucket + buckets[idx + 1]
                    lengths = [length for _, length in merged]
                    candidates.append((max(lengths) / max(1, min(lengths)), idx))

                if not candidates:
                    continue

                _, merge_idx = min(candidates, key=lambda item: item[0])
                buckets[merge_idx] = buckets[merge_idx] + buckets[merge_idx + 1]
                del buckets[merge_idx + 1]
                changed = True
                break

        return buckets

    def _3d_array_to_df(
        self,
        array_3d: np.ndarray,
        uids: list[str],
        original_dates_per_series: list[pd.Series],
        original_lengths: np.ndarray,
        suffix: str,
    ) -> pd.DataFrame:
        dfs = []
        for i, uid in enumerate(uids):
            n = int(original_lengths[i])
            values = array_3d[i, :n, 0]
            dates = pd.to_datetime(original_dates_per_series[i]).iloc[:n]
            dfs.append(
                pd.DataFrame(
                    {
                        "unique_id": f"{uid}_{suffix}",
                        "ds": dates.to_numpy(),
                        "y": values,
                    }
                )
            )
        return pd.concat(dfs, ignore_index=True)
