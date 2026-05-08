from neuralforecast.models import NHITS, MLP, KAN
from neuralforecast.auto import AutoMLP, AutoNHITS, AutoKAN
from metaforecast.synth import (SeasonalMBB,
                                Jittering,
                                Scaling,
                                MagnitudeWarping,
                                TimeWarping,
                                DBA,
                                TSMixup)
from src.timevae_wrapper import TimeVAEWrapper
from src.tsdiff_wrapper import TSDiffWrapper

ACCELERATOR = 'cpu'
DEVICES = 1

MODELS = {
    'NHITS': NHITS,
    'MLP': MLP,
    'KAN': KAN,
    'AutoMLP': AutoMLP,
    'AutoNHITS': AutoNHITS,
    'AutoKAN': AutoKAN,
}

AUTO_MODELS = {'AutoMLP', 'AutoNHITS', 'AutoKAN'}

MODEL_CONFIG = {
    'AutoMLP': {
        'auto': True,
        'backend': 'optuna',
        'num_samples': 5,
    },
    'AutoKAN': {
        'auto': True,
        'backend': 'optuna',
        'num_samples': 5,
    },
    'AutoNHITS': {
        'auto': True,
        'backend': 'optuna',
        'num_samples': 5,
        'config': None,
    },
    'NHITS': {
        # 'start_padding_enabled': False,
        'accelerator': ACCELERATOR,
        'devices': DEVICES,
        'scaler_type': 'standard',
    },
    'MLP': {
        # 'start_padding_enabled': False,
        'accelerator': ACCELERATOR,
        'devices': DEVICES,
        'scaler_type': 'standard',
    },
    'KAN': {
        'accelerator': ACCELERATOR,
        'devices': DEVICES,
        'scaler_type': 'standard',
    },
}

SYNTH_METHODS = {
    'SeasonalMBB': SeasonalMBB,
    'Jittering': Jittering,
    'Scaling': Scaling,
    'TimeWarping': TimeWarping,
    'MagnitudeWarping': MagnitudeWarping,
    'TSMixup': TSMixup,
    'DBA': DBA,
    'TimeVAE': TimeVAEWrapper,
    'TSDiff': TSDiffWrapper,
}

SYNTH_METHODS_ARGS = {
    'SeasonalMBB': ['seas_period', 'log', 'max_samples_in_stl'],
    'Jittering': ['sigma'],
    'Scaling': ['sigma'],
    'MagnitudeWarping': ['sigma', 'knot'],
    'TimeWarping': ['sigma', 'knot'],
    'DBA': ['max_n_uids', 'dirichlet_alpha', 'max_iter'],
    'TSMixup': ['max_n_uids', 'max_len', 'min_len', 'dirichlet_alpha'],
    'TimeVAE': ['seas_period', 'window_size', 'latent_dim', 'reconstruction_wt', 'max_epochs'],
    'TSDiff': [
        'window_size',
        'max_epochs',
        'max_steps',
        'batch_size',
        'num_batches_per_epoch',
        'learning_rate',
        'context_length',
        'prediction_length',
        'transform_mode',
        'twin_noise_level',
        'clip_scaled',
        'freq',
        'diffusion_config',
        'normalization',
        'restore_scale',
        'clip_to_observed_range',
        'use_lags',
        'use_features',
        'use_rolling_windows',
        'rolling_stride',
        'max_windows_per_series',
        'use_length_bucketing',
        'max_length_ratio_per_bucket',
        'min_bucket_size',
        'max_buckets',
        'preserve_train_size',
        'max_samples_per_uid',
        'show_progress',
    ],
}
