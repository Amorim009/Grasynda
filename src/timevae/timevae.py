import os
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import (
    Conv1D,
    Conv1DTranspose,
    Dense,
    Input,
    Layer,
    Reshape,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from src.timevae.vae_base import BaseVariationalAutoencoder, Sampling


class TrendLayer(Layer):
    def __init__(self, feat_dim, trend_poly, seq_len, **kwargs):
        super(TrendLayer, self).__init__(**kwargs)
        self.feat_dim = feat_dim
        self.trend_poly = trend_poly
        self.seq_len = seq_len
        self.trend_dense1 = Dense(
            self.feat_dim * self.trend_poly, activation="relu", name="trend_params"
        )
        self.trend_dense2 = Dense(self.feat_dim * self.trend_poly, name="trend_params2")
        self.reshape_layer = Reshape(target_shape=(self.feat_dim, self.trend_poly))

    def call(self, z):
        trend_params = self.trend_dense1(z)
        trend_params = self.trend_dense2(trend_params)
        trend_params = self.reshape_layer(trend_params)

        lin_space = tf.range(0, float(self.seq_len), 1) / self.seq_len
        poly_space = tf.stack(
            [lin_space ** float(p + 1) for p in range(self.trend_poly)], axis=0
        )

        trend_vals = tf.matmul(trend_params, poly_space)
        trend_vals = tf.transpose(trend_vals, perm=[0, 2, 1])
        trend_vals = tf.cast(trend_vals, tf.float32)
        return trend_vals


class SeasonalLayer(Layer):
    def __init__(self, feat_dim, seq_len, custom_seas, **kwargs):
        super(SeasonalLayer, self).__init__(**kwargs)
        self.feat_dim = feat_dim
        self.seq_len = seq_len
        self.custom_seas = custom_seas
        self.dense_layers = [
            Dense(feat_dim * num_seasons, name=f"season_params_{i}")
            for i, (num_seasons, len_per_season) in enumerate(custom_seas)
        ]
        self.reshape_layers = [
            Reshape(target_shape=(feat_dim, num_seasons))
            for num_seasons, len_per_season in custom_seas
        ]

    def _get_season_indexes_over_seq(self, num_seasons, len_per_season):
        season_indexes = tf.range(num_seasons)[:, None] + tf.zeros(
            (num_seasons, len_per_season), dtype=tf.int32
        )
        season_indexes = tf.reshape(season_indexes, [-1])
        season_indexes = tf.tile(season_indexes, [self.seq_len // len_per_season + 1])[
            : self.seq_len
        ]
        return season_indexes

    def call(self, z):
        n = tf.shape(z)[0]
        ones_tensor = tf.ones(shape=[n, self.feat_dim, self.seq_len], dtype=tf.int32)

        all_seas_vals = []
        for i, (num_seasons, len_per_season) in enumerate(self.custom_seas):
            season_params = self.dense_layers[i](z)
            season_params = self.reshape_layers[i](season_params)

            season_indexes_over_time = self._get_season_indexes_over_seq(
                num_seasons, len_per_season
            )
            dim2_idxes = ones_tensor * tf.reshape(
                season_indexes_over_time, shape=(1, 1, -1)
            )
            season_vals = tf.gather(season_params, dim2_idxes, batch_dims=-1)
            all_seas_vals.append(season_vals)

        all_seas_vals = K.stack(all_seas_vals, axis=-1)
        all_seas_vals = tf.reduce_sum(all_seas_vals, axis=-1)
        all_seas_vals = tf.transpose(all_seas_vals, perm=[0, 2, 1])
        return all_seas_vals

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.seq_len, self.feat_dim)


class TimeVAE(BaseVariationalAutoencoder):
    model_name = "TimeVAE"

    def __init__(
        self,
        hidden_layer_sizes,
        trend_poly=0,
        custom_seas=None,
        use_residual_conn=True,
        **kwargs,
    ):
        super(TimeVAE, self).__init__(**kwargs)

        if hidden_layer_sizes is None:
            hidden_layer_sizes = [50, 100, 200]

        self.hidden_layer_sizes = hidden_layer_sizes
        self.trend_poly = trend_poly
        self.custom_seas = custom_seas
        self.use_residual_conn = use_residual_conn
        self.encoder = self._get_encoder()
        self.decoder = self._get_decoder()
        self.compile(optimizer=Adam())

    def _get_encoder(self):
        encoder_inputs = Input(shape=(self.seq_len, self.feat_dim), name="encoder_input")
        x = encoder_inputs
        for i, num_filters in enumerate(self.hidden_layer_sizes):
            x = Conv1D(
                filters=num_filters,
                kernel_size=3,
                strides=2,
                activation="relu",
                padding="same",
                name=f"enc_conv_{i}",
            )(x)

        x = tf.keras.layers.Flatten(name="enc_flatten")(x)
        self.encoder_last_dense_dim = x.shape[-1]

        z_mean = Dense(self.latent_dim, name="z_mean")(x)
        z_log_var = Dense(self.latent_dim, name="z_log_var")(x)

        encoder_output = Sampling()([z_mean, z_log_var])
        self.encoder_output = encoder_output

        return Model(encoder_inputs, [z_mean, z_log_var, encoder_output], name="encoder")

    def _get_decoder(self):
        decoder_inputs = Input(shape=(self.latent_dim,), name="decoder_input")

        outputs = self.level_model(decoder_inputs)

        if self.trend_poly is not None and self.trend_poly > 0:
            trend_vals = TrendLayer(self.feat_dim, self.trend_poly, self.seq_len)(
                decoder_inputs
            )
            outputs = trend_vals if outputs is None else outputs + trend_vals

        if self.custom_seas is not None and len(self.custom_seas) > 0:
            cust_seas_vals = SeasonalLayer(
                feat_dim=self.feat_dim,
                seq_len=self.seq_len,
                custom_seas=self.custom_seas,
            )(decoder_inputs)
            outputs = cust_seas_vals if outputs is None else outputs + cust_seas_vals

        if self.use_residual_conn:
            residuals = self._get_decoder_residual(decoder_inputs)
            outputs = residuals if outputs is None else outputs + residuals

        if outputs is None:
            raise ValueError(
                "Error: No decoder model to use. You must use one or more of trend, "
                "custom seasonalities, and/or residual connection."
            )

        return Model(decoder_inputs, [outputs], name="decoder")

    def level_model(self, z):
        level_params = Dense(self.feat_dim, name="level_params", activation="relu")(z)
        level_params = Dense(self.feat_dim, name="level_params2")(level_params)
        level_params = Reshape(target_shape=(1, self.feat_dim))(level_params)
        ones_tensor = tf.ones(shape=[1, self.seq_len, 1], dtype=tf.float32)
        return level_params * ones_tensor

    def _get_decoder_residual(self, x):
        x = Dense(self.encoder_last_dense_dim, name="dec_dense", activation="relu")(x)
        x = Reshape(target_shape=(-1, self.hidden_layer_sizes[-1]), name="dec_reshape")(x)

        for i, num_filters in enumerate(reversed(self.hidden_layer_sizes[:-1])):
            x = Conv1DTranspose(
                filters=num_filters,
                kernel_size=3,
                strides=2,
                padding="same",
                activation="relu",
                name=f"dec_deconv_{i}",
            )(x)

        x = Conv1DTranspose(
            filters=self.feat_dim,
            kernel_size=3,
            strides=2,
            padding="same",
            activation="relu",
            name=f"dec_deconv__{i+1}",
        )(x)

        x = tf.keras.layers.Flatten(name="dec_flatten")(x)
        x = Dense(self.seq_len * self.feat_dim, name="decoder_dense_final")(x)
        return Reshape(target_shape=(self.seq_len, self.feat_dim))(x)

    def save(self, model_dir: str):
        os.makedirs(model_dir, exist_ok=True)
        super().save(model_dir)

        if self.custom_seas is not None:
            custom_seas_serializable = [
                (int(num_seasons), int(len_per_season))
                for num_seasons, len_per_season in self.custom_seas
            ]
        else:
            custom_seas_serializable = None

        dict_params = {
            "seq_len": self.seq_len,
            "feat_dim": self.feat_dim,
            "latent_dim": self.latent_dim,
            "reconstruction_wt": self.reconstruction_wt,
            "hidden_layer_sizes": list(self.hidden_layer_sizes),
            "trend_poly": self.trend_poly,
            "custom_seas": custom_seas_serializable,
            "use_residual_conn": self.use_residual_conn,
        }
        params_file = os.path.join(model_dir, f"{self.model_name}_parameters.pkl")
        joblib.dump(dict_params, params_file)

    @classmethod
    def load(cls, model_dir: str) -> "TimeVAE":
        params_file = os.path.join(model_dir, f"{cls.model_name}_parameters.pkl")
        dict_params = joblib.load(params_file)
        vae_model = TimeVAE(**dict_params)
        vae_model.load_weights(model_dir)
        vae_model.compile(optimizer=Adam())
        return vae_model
