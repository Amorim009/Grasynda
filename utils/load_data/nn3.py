import pandas as pd

from utils.load_data.base import LoadDataset


class NN3Dataset(LoadDataset):
    DATASET_PATH = 'assets/datasets/NN3_COMPLETE(NN3 COMPLETE Data).csv'
    DATASET_NAME = 'NN3'

    horizons_map = {
        'Monthly': 12,
    }

    frequency_map = {
        'Monthly': 12,
    }

    context_length = {
        'Monthly': 24,
    }

    min_samples = {
        'Monthly': 50,
    }

    frequency_pd = {
        'Monthly': 'MS',
    }

    data_group = [*horizons_map]
    horizons = [*horizons_map.values()]
    frequency = [*frequency_map.values()]

    _HEADER_ROW = 12
    _START_YEAR_ROW = 13
    _START_MONTH_ROW = 14
    _DATA_START_ROW = 17

    @classmethod
    def load_data(cls, group, min_n_instances=None):
        assert group in cls.data_group

        raw = pd.read_csv(cls.DATASET_PATH, header=None)

        ids = raw.iloc[cls._HEADER_ROW, 1:].astype(str).str.strip()
        start_years = pd.to_numeric(
            raw.iloc[cls._START_YEAR_ROW, 1:], errors='coerce'
        ).astype('Int64')
        start_months = pd.to_numeric(
            raw.iloc[cls._START_MONTH_ROW, 1:], errors='coerce'
        ).astype('Int64')
        # The competition template keeps future forecast slots as blanks, so
        # only numeric observed values are carried into the pipeline.
        values = raw.iloc[cls._DATA_START_ROW:, 1:].apply(
            pd.to_numeric, errors='coerce'
        )

        series_frames = []
        for col_idx, unique_id in enumerate(ids):
            series = values.iloc[:, col_idx].dropna().reset_index(drop=True)
            if series.empty:
                continue

            start = pd.Timestamp(
                year=int(start_years.iloc[col_idx]),
                month=int(start_months.iloc[col_idx]),
                day=1,
            )
            ds = pd.date_range(
                start=start,
                periods=len(series),
                freq=cls.frequency_pd[group],
            )

            series_frames.append(
                pd.DataFrame(
                    {
                        'unique_id': unique_id,
                        'ds': ds,
                        'y': series.to_numpy(),
                    }
                )
            )

        df = pd.concat(series_frames, ignore_index=True)

        if min_n_instances is not None:
            df = cls.prune_df_by_size(df, min_n_instances)

        return df
