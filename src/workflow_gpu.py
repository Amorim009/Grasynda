import pandas as pd

from utils.config_gpu import SYNTH_METHODS, SYNTH_METHODS_ARGS


class ExpWorkflow:

    @staticmethod
    def get_offline_augmented_data(train_, generator_name, augmentation_params, n_series_by_uid):
        train = train_.copy()

        tsgen_params = {
            k: v
            for k, v in augmentation_params.items()
            if k in SYNTH_METHODS_ARGS[generator_name]
        }

        offline_def_tsgen = SYNTH_METHODS[generator_name](**tsgen_params)

        train_synth = pd.concat(
            [offline_def_tsgen.transform(train) for _ in range(n_series_by_uid)]
        ).reset_index(drop=True)

        train_augmented = pd.concat([train, train_synth]).reset_index(drop=True)

        return train_augmented
