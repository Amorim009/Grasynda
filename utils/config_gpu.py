from neuralforecast.auto import AutoKAN, AutoMLP, AutoNHITS
from neuralforecast.models import KAN, MLP, NHITS
from metaforecast.synth import (
    DBA,
    Jittering,
    MagnitudeWarping,
    Scaling,
    SeasonalMBB,
    TimeWarping,
    TSMixup,
)
from src.timevae_wrapper_gpu import TimeVAEWrapper
from src.tsdiff_wrapper_gpu import TSDiffWrapper

try:
    import torch
except Exception:  # pragma: no cover - keep config importable without torch
    torch = None


def torch_cuda_available() -> bool:
    return bool(torch is not None and torch.cuda.is_available())


def get_runtime_hardware_summary() -> dict:
    return {
        "requested_accelerator": "gpu",
        "accelerator": "gpu",
        "torch_device": "cuda:0",
        "require_gpu": True,
        "torch_version": getattr(torch, "__version__", "unavailable"),
        "torch_cuda_available": torch_cuda_available(),
        "torch_cuda_version": getattr(getattr(torch, "version", None), "cuda", None),
        "torch_cuda_device_count": torch.cuda.device_count() if torch_cuda_available() else 0,
    }


def assert_runtime_ready_for_requested_accelerator() -> None:
    if not torch_cuda_available():
        raise RuntimeError(
            "GPU execution was requested, but PyTorch CUDA is unavailable in this "
            "environment. Install a CUDA-enabled PyTorch build before using the "
            "GPU universal runner."
        )


ACCELERATOR = "gpu"
TORCH_DEVICE = "cuda:0"
REQUIRE_GPU = True

MODELS = {
    "NHITS": NHITS,
    "MLP": MLP,
    "KAN": KAN,
    "AutoMLP": AutoMLP,
    "AutoNHITS": AutoNHITS,
    "AutoKAN": AutoKAN,
}
AUTO_MODELS = {"AutoMLP", "AutoNHITS", "AutoKAN"}

MODEL_CONFIG = {
    "AutoMLP": {"auto": True, "backend": "optuna", "num_samples": 5},
    "AutoKAN": {"auto": True, "backend": "optuna", "num_samples": 5},
    "AutoNHITS": {"auto": True, "backend": "optuna", "num_samples": 5, "config": None},
    "NHITS": {"accelerator": ACCELERATOR, "scaler_type": "standard"},
    "MLP": {"accelerator": ACCELERATOR, "scaler_type": "standard"},
    "KAN": {"accelerator": ACCELERATOR, "scaler_type": "standard"},
}

SYNTH_METHODS = {
    "SeasonalMBB": SeasonalMBB,
    "Jittering": Jittering,
    "Scaling": Scaling,
    "TimeWarping": TimeWarping,
    "MagnitudeWarping": MagnitudeWarping,
    "TSMixup": TSMixup,
    "DBA": DBA,
    "TimeVAE": TimeVAEWrapper,
    "TSDiff": TSDiffWrapper,
}

SYNTH_METHODS_ARGS = {
    "SeasonalMBB": ["seas_period", "log", "max_samples_in_stl"],
    "Jittering": ["sigma"],
    "Scaling": ["sigma"],
    "MagnitudeWarping": ["sigma", "knot"],
    "TimeWarping": ["sigma", "knot"],
    "DBA": ["max_n_uids", "dirichlet_alpha", "max_iter"],
    "TSMixup": ["max_n_uids", "max_len", "min_len", "dirichlet_alpha"],
    "TimeVAE": [
        "seas_period",
        "window_size",
        "latent_dim",
        "reconstruction_wt",
        "max_epochs",
        "alpha_target_ratio",
        "device",
        "require_gpu",
    ],
    "TSDiff": [
        "window_size",
        "max_epochs",
        "max_steps",
        "batch_size",
        "num_batches_per_epoch",
        "learning_rate",
        "context_length",
        "prediction_length",
        "transform_mode",
        "twin_noise_level",
        "clip_scaled",
        "freq",
        "diffusion_config",
        "normalization",
        "restore_scale",
        "clip_to_observed_range",
        "use_lags",
        "use_features",
        "use_rolling_windows",
        "rolling_stride",
        "max_windows_per_series",
        "use_length_bucketing",
        "max_length_ratio_per_bucket",
        "min_bucket_size",
        "max_buckets",
        "preserve_train_size",
        "max_samples_per_uid",
        "show_progress",
        "device",
        "require_gpu",
    ],
}
