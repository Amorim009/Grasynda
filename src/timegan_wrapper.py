import numpy as np
import pandas as pd
from types import SimpleNamespace
import torch

from src.timegan.timegan import TimeGAN


class PerSeriesMinMaxScaler:
    """Scale each series independently to [0, 1]."""

    def __init__(self):
        self.mins = None
        self.ranges = None

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        self.mins = np.min(x, axis=1, keepdims=True)
        maxs = np.max(x, axis=1, keepdims=True)
        self.ranges = maxs - self.mins
        self.ranges[self.ranges == 0] = 1.0
        return (x - self.mins) / self.ranges

    def inverse_transform(self, x_scaled: np.ndarray, indices: np.ndarray | None = None) -> np.ndarray:
        if self.mins is None or self.ranges is None:
            raise RuntimeError("Scaler is not fitted.")

        if indices is None:
            if x_scaled.shape[0] != self.mins.shape[0]:
                raise ValueError("inverse_transform requires indices when n_samples differs from training.")
            mins = self.mins
            ranges = self.ranges
        else:
            mins = self.mins[indices]
            ranges = self.ranges[indices]
        return x_scaled * ranges + mins


class TimeGANWrapper:
    """
    Wrapper for TimeGAN adapted for panel forecasting datasets:
    - input: DataFrame ['unique_id', 'ds', 'y']
    - default transform mode: digital twins (1 synthetic per real series)
    """

    def __init__(
        self,
        hidden_dim=24,
        num_layer=3,
        z_dim=None,
        batch_size=128,
        iteration=2000,
        lr=0.001,
        beta1=0.9,
        w_gamma=1.0,
        w_g=100.0,
        window_size=48,
        twin_noise_scale=0.0,
        twin_use_supervisor=False,
        transform_mode="gan",
        recovery_sigmoid=False,
        w_deriv=1.0,
        output_suffix="timegan",
        device=None,
    ):
        self.hidden_dim = hidden_dim
        self.num_layer = num_layer
        self.z_dim = z_dim
        self.batch_size = batch_size
        self.iteration = iteration
        self.lr = lr
        self.beta1 = beta1
        self.w_gamma = w_gamma
        self.w_g = w_g
        self.window_size = window_size
        self.twin_noise_scale = twin_noise_scale
        self.twin_use_supervisor = twin_use_supervisor
        self.transform_mode = transform_mode
        self.recovery_sigmoid = recovery_sigmoid
        self.w_deriv = w_deriv
        self.output_suffix = output_suffix
        self.device = device
        self.scaler = PerSeriesMinMaxScaler()
        self._transform_calls = 0
        self._train_scaled_3d = None

    def fit(self, df: pd.DataFrame, plot_path=None):
        required_cols = {"unique_id", "ds", "y"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")
        if df.empty:
            raise ValueError("Input dataframe is empty.")

        df_local = df.copy()
        df_local["ds"] = pd.to_datetime(df_local["ds"])
        df_local["y"] = pd.to_numeric(df_local["y"], errors="coerce")
        if df_local["y"].isna().any():
            raise ValueError("Column 'y' contains NaN/non-numeric values.")

        self.uids = sorted(df_local["unique_id"].unique().tolist())
        n_series = len(self.uids)
        print(f"[TimeGAN] Starting fit with {n_series} series, window_size={self.window_size}")

        data_3d, _, timestamps_per_series = self._df_to_3d_array(df_local, self.uids)
        self.timestamps_per_series = timestamps_per_series

        scaled_3d = self.scaler.fit_transform(data_3d)
        self._train_scaled_3d = scaled_3d
        ori_data_list = [scaled_3d[i] for i in range(n_series)]

        z_dim = self.z_dim if self.z_dim is not None else data_3d.shape[2]

        if self.device is None:
            resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
        elif self.device == "cuda" and not torch.cuda.is_available():
            print("[TimeGAN] CUDA requested but unavailable. Falling back to CPU.")
            resolved_device = "cpu"
        else:
            resolved_device = self.device

        self.opt = SimpleNamespace(
            z_dim=z_dim,
            hidden_dim=self.hidden_dim,
            num_layer=self.num_layer,
            batch_size=min(self.batch_size, n_series),
            iteration=self.iteration,
            lr=self.lr,
            beta1=self.beta1,
            w_gamma=self.w_gamma,
            w_g=self.w_g,
            device=resolved_device,
            isTrain=True,
            resume="",
            manualseed=-1,
            verbose=1,
            plot_path=plot_path,
            recovery_sigmoid=self.recovery_sigmoid,
            w_deriv=self.w_deriv,
        )

        self.model = TimeGAN(self.opt, ori_data_list)
        print("[TimeGAN] Training...")
        self.model.train()
        return self

    def generate(
        self,
        n_samples: int | None = None,
        mode: str = "gan",
        noise_scale: float | None = None,
        id_suffix: str | None = None,
    ) -> pd.DataFrame:
        if not hasattr(self, "model"):
            raise RuntimeError("Model not fitted. Call fit() first.")
        if mode == "reconstruction":
            mode = "digital_twin"
        if mode not in {"gan", "digital_twin"}:
            raise ValueError("mode must be 'gan', 'digital_twin', or 'reconstruction'")

        if mode == "gan":
            if n_samples is None:
                raise ValueError("n_samples is required when mode='gan'.")
            synth_scaled = self._generate_from_gan_scaled(n_samples)
            mapped_idx = np.arange(n_samples) % len(self.uids)
            synth_denorm = self.scaler.inverse_transform(synth_scaled, indices=mapped_idx)
            synth_uids = [f"gen_{i}" for i in range(n_samples)]
            synth_dates = [self.timestamps_per_series[i] for i in mapped_idx]
            print("[TimeGAN] Generation complete (GAN mode).")
            return self._3d_array_to_df(synth_denorm, synth_uids, synth_dates)

        if n_samples is None:
            n_samples = len(self.uids)
        n_samples = min(n_samples, len(self.uids))
        idx = np.arange(n_samples)
        noise = self.twin_noise_scale if noise_scale is None else noise_scale
        synth_scaled = self._generate_digital_twins_scaled(idx, noise)
        synth_denorm = self.scaler.inverse_transform(synth_scaled, indices=idx)
        suffix = self.output_suffix if id_suffix is None else id_suffix
        synth_uids = [f"{self.uids[i]}_{suffix}" for i in idx]
        synth_dates = [self.timestamps_per_series[i] for i in idx]
        print("[TimeGAN] Generation complete (digital_twin mode).")
        return self._3d_array_to_df(synth_denorm, synth_uids, synth_dates)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.fit(df)
        if self._transform_calls == 0:
            suffix = self.output_suffix
        else:
            suffix = f"{self.output_suffix}_{self._transform_calls}"
        self._transform_calls += 1
        if self.transform_mode == "gan":
            n = len(self.uids)
            synth_scaled = self._generate_from_gan_scaled(n)
            idx = np.arange(n)
            synth_denorm = self.scaler.inverse_transform(synth_scaled, indices=idx)
            synth_uids = [f"{self.uids[i]}_{suffix}" for i in idx]
            synth_dates = [self.timestamps_per_series[i] for i in idx]
            print("[TimeGAN] Transform complete (GAN mode mapped to training UIDs).")
            return self._3d_array_to_df(synth_denorm, synth_uids, synth_dates)
        if self.transform_mode in {"digital_twin", "reconstruction"}:
            return self.generate(mode="digital_twin", id_suffix=suffix)
        raise ValueError("transform_mode must be 'gan', 'digital_twin', or 'reconstruction'")

    def _generate_from_gan_scaled(self, n_samples: int) -> np.ndarray:
        generated_list = []
        remaining = n_samples
        batch_size = self.opt.batch_size

        print(f"[TimeGAN] Generating {n_samples} series (GAN mode)...")
        while remaining > 0:
            current_batch = min(remaining, batch_size)
            chunk = self.model.generation(current_batch)
            generated_list.extend(chunk)
            remaining -= current_batch

        synth_3d = np.zeros((n_samples, self.window_size, 1), dtype=np.float32)
        for i, arr in enumerate(generated_list):
            arr_np = np.asarray(arr)
            if arr_np.ndim == 1:
                arr_np = arr_np[:, None]
            length = min(arr_np.shape[0], self.window_size)
            synth_3d[i, :length, 0] = arr_np[:length, 0]
        return synth_3d

    def _generate_digital_twins_scaled(self, indices: np.ndarray, noise_scale: float) -> np.ndarray:
        if self._train_scaled_3d is None:
            raise RuntimeError("Digital twin generation requires fitted in-memory training data.")

        source = np.asarray(self.model.ori_data)[indices]
        out_batches = []
        batch_size = self.opt.batch_size

        self.model.nete.eval()
        self.model.nets.eval()
        self.model.netr.eval()

        print(f"[TimeGAN] Generating {len(indices)} series (digital_twin mode)...")
        with torch.no_grad():
            for start in range(0, len(source), batch_size):
                x_np = source[start:start + batch_size]
                x = torch.tensor(x_np, dtype=torch.float32).to(self.model.device)

                h = self.model.nete(x)
                if noise_scale > 0:
                    h = h + torch.randn_like(h) * noise_scale
                if self.twin_use_supervisor:
                    h = self.model.nets(h)
                x_tilde = self.model.netr(h)
                out_batches.append(x_tilde.cpu().numpy())

        return np.concatenate(out_batches, axis=0)

    def _df_to_3d_array(self, df: pd.DataFrame, uids: list) -> tuple:
        n_series = len(uids)
        seq_len = self.window_size
        array_3d = np.zeros((n_series, seq_len, 1), dtype=np.float32)
        timestamps_per_series = []
        grouped = df.groupby("unique_id")

        for i, uid in enumerate(uids):
            if uid not in grouped.groups:
                timestamps_per_series.append(pd.date_range("2000-01-01", periods=seq_len, freq="D"))
                continue

            group = grouped.get_group(uid).sort_values("ds")
            values = group["y"].to_numpy(dtype=float)
            dates = pd.to_datetime(group["ds"]).to_numpy()

            if len(values) < seq_len:
                pad_len = seq_len - len(values)
                values = np.concatenate([np.repeat(values[0], pad_len), values])
                if len(dates) > 1:
                    step = dates[1] - dates[0]
                    pad_dates = [dates[0] - step * (pad_len - j) for j in range(pad_len)]
                    dates = np.concatenate([np.asarray(pad_dates), dates])
                else:
                    dates = pd.date_range(end=pd.Timestamp(dates[0]), periods=seq_len, freq="D").to_numpy()
            else:
                values = values[-seq_len:]
                dates = dates[-seq_len:]

            array_3d[i, :, 0] = values
            timestamps_per_series.append(pd.to_datetime(dates))

        return array_3d, seq_len, timestamps_per_series

    def _3d_array_to_df(self, array_3d: np.ndarray, uids: list, timestamps_per_series: list) -> pd.DataFrame:
        n_series, _, _ = array_3d.shape
        dfs = []
        for i in range(n_series):
            dfs.append(
                pd.DataFrame(
                    {
                        "unique_id": uids[i],
                        "ds": pd.to_datetime(timestamps_per_series[i]),
                        "y": array_3d[i, :, 0],
                    }
                )
            )
        return pd.concat(dfs, ignore_index=True)
