from typing import Union, List, Optional
from tensorflow.keras.optimizers import Adam
import numpy as np
import random
import os
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # or any {'0', '1', '2'}
warnings.filterwarnings("ignore")

import tensorflow as tf

from src.timevae.vae_conv_model import VariationalAutoencoderConv as VAE_Conv
from src.timevae.timevae import TimeVAE


def set_seeds(seed: int = 111) -> None:
    tf.random.set_seed(seed)

    np.random.seed(seed)

    random.seed(seed)


def instantiate_vae_model(
    vae_type: str, sequence_length: int, feature_dim: int, batch_size: int, **kwargs
):
    set_seeds(seed=123)
    kwargs.setdefault("hidden_layer_sizes", None)

    if vae_type == "vae_conv":
        vae = VAE_Conv(
            seq_len=sequence_length,
            feat_dim=feature_dim,
            batch_size=batch_size,
            **kwargs,
        )
    elif vae_type == "timeVAE":
        vae = TimeVAE(
            seq_len=sequence_length,
            feat_dim=feature_dim,
            batch_size=batch_size,
            **kwargs,
        )
    else:
        raise ValueError(
            f"Unrecognized model type [{vae_type}]. "
            "For Grasynda integration, use 'vae_conv' or 'timeVAE'."
        )

    return vae


def train_vae(vae, train_data, max_epochs, verbose=0, train_mask=None):
    vae.fit_on_data(train_data, max_epochs, verbose, train_mask=train_mask)


def save_vae_model(vae, dir_path: str) -> None:
    vae.save(dir_path)


def load_vae_model(vae_type: str, dir_path: str):
    if vae_type == "vae_conv":
        vae = VAE_Conv.load(dir_path)
    elif vae_type == "timeVAE":
        vae = TimeVAE.load(dir_path)
    else:
        raise ValueError(
            f"Unrecognized model type [{vae_type}]. "
            "For Grasynda integration, use 'vae_conv' or 'timeVAE'."
        )

    return vae


def get_posterior_samples(vae, data):
    return vae.predict(data, verbose=0)


def get_prior_samples(vae, num_samples: int):
    return vae.get_prior_samples(num_samples=num_samples)
