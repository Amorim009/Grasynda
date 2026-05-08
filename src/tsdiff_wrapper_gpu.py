import numpy as np
import pandas as pd
import torch
from typing import Any
from pandas.tseries.frequencies import to_offset
from gluonts.dataset.common import ListDataset
from gluonts.dataset.loader import TrainDataLoader
from gluonts.itertools import Cached
from gluonts.time_feature import time_features_from_frequency_str
from gluonts.torch.batchify import batchify
from tqdm.auto import tqdm

from src.uncond_ts_diff.model import TSDiff
from src.uncond_ts_diff.utils import create_splitter, create_transforms, get_lags_for_freq
import src.uncond_ts_diff.configs as diffusion_configs


class TSDiffWrapper:
    def __init__(
        self,
        window_size=None,
        diffusion_config="diffusion_small_config",
        max_epochs=20,
        max_steps=None,
        batch_size=32,
        num_batches_per_epoch=64,
        learning_rate=1e-3,
        context_length=None,
        prediction_length=None,
        transform_mode="sample",
        twin_noise_level=0.08,
        freq=None,
        use_lags=False,
        use_features=False,
        normalization="mean",
        restore_scale=True,
        clip_to_observed_range=True,
        init_skip=True,
        grad_clip=1.0,
        clip_scaled=False,
        use_rolling_windows=True,
        rolling_stride=1,
        max_windows_per_series=64,
        use_length_bucketing=True,
        max_length_ratio_per_bucket=1.35,
        min_bucket_size=12,
        max_buckets=6,
        preserve_train_size=True,
        max_samples_per_uid=8,
        show_progress=True,
        output_suffix="tsdiff",
        random_seed=42,
        device=None,
        require_gpu=False,
    ):
        if window_size is None:
            self.window_size = None
        else:
            w = int(window_size)
            self.window_size = w if w > 0 else None
        self.diffusion_config = diffusion_config
        self.max_epochs = int(max_epochs)
        self.max_steps = None if max_steps is None else max(1, int(max_steps))
        self.batch_size = int(batch_size)
        self.num_batches_per_epoch = max(1, int(num_batches_per_epoch))
        self.learning_rate = float(learning_rate)
        self.context_length = context_length
        self.prediction_length = (
            None if prediction_length is None else max(1, int(prediction_length))
        )
        self.transform_mode = str(transform_mode).strip().lower()
        self.twin_noise_level = float(twin_noise_level)
        self.freq = freq
        self.model_freq = None if freq is None else self._canonicalize_freq(freq)
        self.use_lags = bool(use_lags)
        self.use_features = bool(use_features)
        self.normalization = normalization
        self.restore_scale = bool(restore_scale)
        self.clip_to_observed_range = bool(clip_to_observed_range)
        self.init_skip = bool(init_skip)
        self.grad_clip = float(grad_clip)
        self.clip_scaled = bool(clip_scaled)
        self.use_rolling_windows = bool(use_rolling_windows)
        self.rolling_stride = max(1, int(rolling_stride))
        self.max_windows_per_series = max(1, int(max_windows_per_series))
        self.use_length_bucketing = bool(use_length_bucketing)
        self.max_length_ratio_per_bucket = float(max_length_ratio_per_bucket)
        self.min_bucket_size = max(1, int(min_bucket_size))
        self.max_buckets = max(1, int(max_buckets))
        self.preserve_train_size = bool(preserve_train_size)
        self.max_samples_per_uid = max(1, int(max_samples_per_uid))
        self.show_progress = bool(show_progress)
        self.output_suffix = output_suffix
        self.random_seed = int(random_seed)
        self.device = device
        self.require_gpu = bool(require_gpu)

        self.model = None

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self._set_seed()

        df_local = df.copy()
        df_local["ds"] = pd.to_datetime(df_local["ds"])
        df_local["y"] = pd.to_numeric(df_local["y"], errors="coerce")
        df_local = df_local.dropna(subset=["y"]).reset_index(drop=True)

        if df_local.empty:
            raise ValueError("TSDiff received an empty dataframe after cleaning.")

        if self.model_freq is None:
            self.model_freq = self._infer_model_freq(df_local)

        lengths_by_uid = {
            uid: int(len(g)) for uid, g in df_local.groupby("unique_id")
        }
        if self.use_length_bucketing:
            buckets = self._build_length_buckets(lengths_by_uid)
        else:
            buckets = [sorted(lengths_by_uid)]

        if self.show_progress:
            bucket_sizes = ", ".join(str(len(bucket)) for bucket in buckets)
            print(f"[TSDiff] Using {len(buckets)} length bucket(s): {bucket_sizes}")

        outputs = []
        for bucket_idx, bucket_uids in enumerate(buckets, start=1):
            bucket_df = df_local[df_local["unique_id"].isin(bucket_uids)].copy()
            outputs.append(
                self._transform_bucket(bucket_df, bucket_uids, bucket_idx)
            )

        return pd.concat(outputs, ignore_index=True)

    def _transform_bucket(
        self,
        df_bucket: pd.DataFrame,
        uids: list[str],
        bucket_idx: int,
    ) -> pd.DataFrame:
        grouped = {
            uid: g.sort_values("ds").reset_index(drop=True)
            for uid, g in df_bucket.groupby("unique_id")
        }
        lengths = {uid: int(len(g)) for uid, g in grouped.items()}
        effective_use_lags, lag_penalty = self._resolve_lag_behavior(lengths)
        total_length = self._resolve_total_length(lengths, lag_penalty)
        context_length, prediction_length = self._resolve_lengths(total_length)

        if total_length <= prediction_length:
            raise ValueError(
                "TSDiff bucket total length must exceed prediction length. "
                f"Got total_length={total_length}, prediction_length={prediction_length}."
            )

        if self.show_progress:
            print(
                f"[TSDiff] bucket={bucket_idx} series={len(uids)} "
                f"total_length={total_length} context={context_length} "
                f"prediction={prediction_length} use_lags={effective_use_lags} "
                f"freq={self.model_freq} restore_scale={self.restore_scale}"
            )

        dataset = self._build_gluonts_dataset(grouped, uids)
        device = self._resolve_device()
        self.model = self._build_model(
            context_length,
            prediction_length,
            device,
            use_lags=effective_use_lags,
        )
        train_loader = self._build_train_dataloader(
            dataset=dataset,
            context_length=context_length,
            prediction_length=prediction_length,
        )
        self._fit_model(train_loader, device)

        if self.transform_mode != "sample":
            raise ValueError(
                "The faithful TSDiff wrapper currently supports only "
                "transform_mode='sample'."
            )

        samples_per_uid = self._resolve_samples_per_uid(lengths, total_length)
        num_samples = int(sum(samples_per_uid))
        synth_2d = self.model.sample_n(num_samples=num_samples).astype(np.float32)
        if self.clip_scaled:
            synth_2d = np.clip(synth_2d, -5.0, 5.0)

        return self._samples_to_df(
            synth_2d=synth_2d,
            grouped=grouped,
            uids=uids,
            total_length=total_length,
            context_length=context_length,
            samples_per_uid=samples_per_uid,
            observed_bounds=(
                float(df_bucket["y"].min()),
                float(df_bucket["y"].max()),
            ),
        )

    def _build_model(
        self,
        context_length: int,
        prediction_length: int,
        device: str,
        use_lags: bool,
    ):
        cfg = getattr(diffusion_configs, self.diffusion_config)
        model = TSDiff(
            **cfg,
            freq=self.model_freq,
            use_features=self.use_features,
            use_lags=use_lags,
            normalization=self.normalization,
            context_length=context_length,
            prediction_length=prediction_length,
            lr=self.learning_rate,
            init_skip=self.init_skip,
        )
        model.to(device)
        return model

    def _build_gluonts_dataset(
        self,
        grouped: dict[str, pd.DataFrame],
        uids: list[str],
    ) -> ListDataset:
        entries = []
        for uid in uids:
            g = grouped[uid]
            entries.append(
                {
                    "item_id": str(uid),
                    "start": pd.Timestamp(g["ds"].iloc[0]),
                    "target": g["y"].to_numpy(dtype=np.float32),
                }
            )
        return ListDataset(entries, freq=self.model_freq)

    def _build_train_dataloader(
        self,
        dataset: ListDataset,
        context_length: int,
        prediction_length: int,
    ):
        transformation = create_transforms(
            num_feat_dynamic_real=0,
            num_feat_static_cat=0,
            num_feat_static_real=0,
            time_features=time_features_from_frequency_str(self.model_freq),
            prediction_length=prediction_length,
        )
        transformed_data = transformation.apply(dataset, is_train=True)
        splitter = create_splitter(
            past_length=context_length + max(self.model.lags_seq),
            future_length=prediction_length,
            mode="train",
        )
        return TrainDataLoader(
            Cached(transformed_data),
            batch_size=self.batch_size,
            stack_fn=batchify,
            transform=splitter,
            num_batches_per_epoch=self.num_batches_per_epoch,
        )

    def _fit_model(self, data_loader, device: str) -> None:
        optim = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.model.train()
        epoch_iter = range(self.max_epochs)
        if self.show_progress:
            epoch_iter = tqdm(epoch_iter, desc="TSDiff train", leave=False)

        step_count = 0
        reached_step_limit = False
        for _ in epoch_iter:
            last_loss = None
            for batch in data_loader:
                batch = self._move_batch_to_device(batch, device)
                optim.zero_grad(set_to_none=True)
                loss = self.model.training_step(batch, 0)["loss"]
                loss.backward()
                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_clip
                    )
                optim.step()
                last_loss = float(loss.detach().cpu())
                step_count += 1
                if self.max_steps is not None and step_count >= self.max_steps:
                    reached_step_limit = True
                    break
            if self.show_progress and last_loss is not None:
                epoch_iter.set_postfix(loss=f"{last_loss:.4f}", steps=step_count)
            if reached_step_limit:
                break
        self.model.eval()

    def _move_batch_to_device(self, batch: dict, device: str) -> dict:
        out = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                out[key] = value.to(device)
            else:
                out[key] = value
        return out

    def _resolve_lengths(self, total_length: int):
        pred = self.prediction_length
        context = self.context_length

        if pred is None and context is None:
            pred = max(1, total_length // 4)
            context = total_length - pred
        elif pred is None:
            context = int(context)
            pred = total_length - context
        elif context is None:
            pred = int(pred)
            context = total_length - pred
        else:
            context = int(context)
            pred = int(pred)

        if context < 1 or pred < 1:
            raise ValueError(
                f"Invalid TSDiff lengths: context={context}, prediction={pred}, total={total_length}."
            )

        return context, pred

    def _resolve_device(self) -> str:
        requested = "auto" if self.device is None else str(self.device).strip()
        requested_lower = requested.lower()

        if requested_lower == "auto":
            resolved = "cuda:0" if torch.cuda.is_available() else "cpu"
        elif requested_lower in {"gpu", "cuda"}:
            resolved = "cuda:0"
        else:
            resolved = requested

        if resolved.lower().startswith("cuda") and not torch.cuda.is_available():
            if self.require_gpu or requested_lower.startswith("cuda") or requested_lower in {"gpu", "cuda"}:
                raise RuntimeError(
                    "TSDiff requested GPU execution, but PyTorch CUDA is unavailable."
                )
            return "cpu"

        return resolved

    def _resolve_total_length(self, lengths_by_uid: dict[str, int], lag_penalty: int) -> int:
        min_length = min(lengths_by_uid.values())
        max_total_length = max(2, min_length - lag_penalty)
        if self.window_size is not None:
            total_length = min(int(self.window_size), max_total_length)
        else:
            total_length = max_total_length

        if self.prediction_length is not None:
            total_length = max(total_length, int(self.prediction_length) + 1)
            if total_length > max_total_length:
                total_length = max_total_length

        return int(total_length)

    def _resolve_lag_behavior(self, lengths_by_uid: dict[str, int]) -> tuple[bool, int]:
        if not self.use_lags:
            return False, 0

        try:
            lag_penalty = max(get_lags_for_freq(self.model_freq))
        except Exception:
            return False, 0

        min_length = min(lengths_by_uid.values())
        if self.prediction_length is None:
            min_required = lag_penalty + 2
        else:
            min_required = lag_penalty + int(self.prediction_length) + 2

        if min_length <= min_required:
            return False, 0
        return True, int(lag_penalty)

    def _build_length_buckets(
        self,
        lengths_by_uid: dict[str, int],
    ) -> list[list[str]]:
        items = sorted(lengths_by_uid.items(), key=lambda kv: kv[1])
        buckets: list[list[tuple[str, int]]] = []
        current: list[tuple[str, int]] = []

        for uid, length in items:
            if not current:
                current = [(uid, length)]
                continue

            current_min = current[0][1]
            proposed_ratio = float(length) / float(max(1, current_min))
            if (
                proposed_ratio <= self.max_length_ratio_per_bucket
                or len(current) < self.min_bucket_size
            ):
                current.append((uid, length))
            else:
                buckets.append(current)
                current = [(uid, length)]

        if current:
            buckets.append(current)

        merged: list[list[tuple[str, int]]] = []
        for bucket in buckets:
            if merged and len(bucket) < self.min_bucket_size:
                merged[-1].extend(bucket)
            else:
                merged.append(bucket)

        while len(merged) > self.max_buckets:
            smallest_idx = min(range(len(merged)), key=lambda i: len(merged[i]))
            if smallest_idx == 0:
                merged[1] = merged[0] + merged[1]
                del merged[0]
            else:
                merged[smallest_idx - 1].extend(merged[smallest_idx])
                del merged[smallest_idx]

        return [[uid for uid, _ in bucket] for bucket in merged]

    def _samples_to_df(
        self,
        synth_2d: np.ndarray,
        grouped: dict[str, pd.DataFrame],
        uids: list[str],
        total_length: int,
        context_length: int,
        samples_per_uid: list[int],
        observed_bounds: tuple[float, float],
    ) -> pd.DataFrame:
        dfs = []
        sample_idx = 0
        for i, uid in enumerate(uids):
            g = grouped[uid]
            window_refs = self._build_window_references(
                g=g,
                total_length=total_length,
                context_length=context_length,
                n_refs=samples_per_uid[i],
            )
            for rep in range(samples_per_uid[i]):
                ref = window_refs[rep % len(window_refs)]
                dates = ref["dates"]
                # Amazon's TSTR pipeline trains downstream models on synthetic
                # samples in the diffusion model's scaled space. We keep those
                # values intact here and only adapt them to our long-format
                # dataframe interface by assigning a valid date scaffold.
                values = np.asarray(
                    synth_2d[sample_idx, -len(dates):],
                    dtype=np.float32,
                )
                if self.restore_scale:
                    values = values * np.float32(ref["scale"])
                    if self.clip_to_observed_range:
                        values = np.clip(
                            values,
                            np.float32(observed_bounds[0]),
                            np.float32(observed_bounds[1]),
                        )
                dfs.append(
                    pd.DataFrame(
                        {
                            "unique_id": f"{uid}_{self.output_suffix}_{rep+1}",
                            "ds": dates,
                            "y": values,
                        }
                    )
                )
                sample_idx += 1
        return pd.concat(dfs, ignore_index=True)

    def _build_window_references(
        self,
        g: pd.DataFrame,
        total_length: int,
        context_length: int,
        n_refs: int,
    ) -> list[dict[str, Any]]:
        dates = pd.to_datetime(g["ds"]).reset_index(drop=True)
        series_values = g["y"].to_numpy(dtype=np.float32)
        series_len = len(dates)

        if series_len <= total_length:
            starts = [0]
        else:
            max_start = series_len - total_length
            n_starts = min(max_start + 1, max(1, n_refs))
            starts = np.linspace(0, max_start, num=n_starts, dtype=int).tolist()

        refs: list[dict[str, Any]] = []
        for start in starts:
            end = start + total_length
            window_dates = dates.iloc[start:end].reset_index(drop=True)
            context_values = series_values[start : start + context_length]
            finite_context = context_values[np.isfinite(context_values)]
            if finite_context.size:
                scale = float(np.mean(np.abs(finite_context)))
            else:
                scale = 1.0
            if not np.isfinite(scale) or scale < 1e-7:
                scale = 1.0
            refs.append(
                {
                    "dates": window_dates,
                    "scale": scale,
                }
            )

        return refs

    def _resolve_samples_per_uid(
        self,
        lengths_by_uid: dict[str, int],
        total_length: int,
    ) -> list[int]:
        counts = []
        for uid in lengths_by_uid:
            orig_len = int(lengths_by_uid[uid])
            if self.preserve_train_size:
                n = int(np.ceil(orig_len / float(max(1, total_length))))
            else:
                n = 1
            counts.append(min(self.max_samples_per_uid, max(1, n)))
        return counts

    def _canonicalize_freq(self, freq: str) -> str:
        text = str(freq).strip()
        try:
            offset = to_offset(text)
            name = str(offset.name).upper()
        except Exception:
            name = text.upper()

        if name.startswith("MS") or name.startswith("ME") or name.startswith("M"):
            return "M"
        if name.startswith("QS") or name.startswith("QE") or name.startswith("Q"):
            return "Q"
        if name.startswith("AS") or name.startswith("A") or name.startswith("YS"):
            return "A"
        if name.startswith("W"):
            return "W"
        if name.startswith("B"):
            return "B"
        if name.startswith("D"):
            return "D"
        if name.startswith("H"):
            return "H"
        return text

    def _infer_model_freq(self, df: pd.DataFrame) -> str:
        for _, g in df.groupby("unique_id"):
            dates = pd.to_datetime(g["ds"]).sort_values().drop_duplicates()
            if len(dates) >= 3:
                inferred = pd.infer_freq(dates)
                if inferred:
                    return self._canonicalize_freq(inferred)

        diffs = []
        for _, g in df.groupby("unique_id"):
            dates = pd.to_datetime(g["ds"]).sort_values().drop_duplicates()
            delta_days = dates.diff().dropna().dt.total_seconds().to_numpy() / 86400.0
            diffs.extend(delta_days[np.isfinite(delta_days)].tolist())

        if not diffs:
            return "D"

        median_days = float(np.median(diffs))
        if 80 <= median_days <= 100:
            return "Q"
        if 27 <= median_days <= 32:
            return "M"
        if 360 <= median_days <= 370:
            return "A"
        if 6 <= median_days <= 8:
            return "W"
        if 0.9 <= median_days <= 1.1:
            return "D"
        return "D"

    def _set_seed(self):
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_seed)
