
import numpy as np
import pandas as pd
from typing import Dict
from src.qgraph_ts import Grasynda

class GrasyndaContinuousComponent(Grasynda):
    """
    Grasynda variant that uses Continuous Uniform Sampling within quantile bins.
    Retains the standard transform() method (STL decomposition).
    Useful for generating synthetic components (e.g., Remainder) continuously.
    """
    def _create_synthetic_ts(self, df: pd.DataFrame) -> Dict:
        """
        Continuous Uniform Sampling implementation.
        """
        quantile_series = self._generate_quantile_series(df)
        generated_time_series = {}

        uids = df['unique_id'].unique().tolist()
        for uid in uids:
            uid_df = df.query(f'unique_id=="{uid}"')
            uid_s = uid_df[self.quantile_on]
            uid_quantiles = uid_df['Quantile']

            # Store min/max for each quantile
            uid_q_bounds = {}
            for q in range(self.n_quantiles):
                vals = uid_s[uid_quantiles == q].values
                if len(vals) > 0:
                    uid_q_bounds[q] = (vals.min(), vals.max())
                else:
                    uid_q_bounds[q] = (None, None)

            synth_ts = np.zeros(len(uid_s))
            synth_ts[0] = uid_s.values[0]

            for i in range(1, len(uid_quantiles)):
                current_quantile = quantile_series[uid][i]
                min_val, max_val = uid_q_bounds.get(current_quantile, (None, None))

                if min_val is None:
                    # No samples - repeat last value
                    synth_ts[i] = synth_ts[i - 1]
                else:
                    # Continuous Uniform sampling
                    if min_val == max_val:
                        sampled_val = min_val
                    else:
                        sampled_val = np.random.uniform(min_val, max_val)
                    synth_ts[i] = sampled_val

            generated_time_series[uid] = pd.Series(synth_ts, index=uid_df.index)

        return generated_time_series


class GrasyndaContinuous(GrasyndaContinuousComponent):
    """
    Grasynda variant that applies directly to the original time series (no decomposition)
    and uses Continuous Uniform Sampling within quantile bins.
    """
    def __init__(self,
                 n_quantiles: int,
                 period: int,
                 ensemble_transitions: bool = False,
                 ensemble_size: int = 5,
                 robust: bool = False):
        
        # We force quantile_on='y' as we are working on original series
        super().__init__(
            n_quantiles=n_quantiles,
            quantile_on='y',
            period=period,
            ensemble_transitions=ensemble_transitions,
            ensemble_size=ensemble_size,
            robust=robust
        )
        self.alias = 'GrasyndaContinuous'

    def transform(self, df: pd.DataFrame, **kwargs):
        df_ = df.copy()
        
        # Skip decomposition - work directly on 'y'
        
        # Calculate Quantiles on 'y'
        df_['Quantile'] = self._get_quantiles(df_)
        
        # Calc Transitions
        self._calc_transition_matrix(df_)
        if self.ensemble_transitions:
            self.ensemble_transition_mats = self._get_ensemble_transition_mats()
            
        # Generate Synthetic Series (Continuous)
        synth_ts_dict = self._create_synthetic_ts(df_)
        
        # Post-process (Simple replacement)
        synth_list = []
        for uid, uid_df in df_.groupby('unique_id'):
            # Create new dataframe with synthetic y
            new_df = uid_df.copy()
            new_df['y'] = synth_ts_dict[uid].values
            synth_list.append(new_df)
            
        synth_df = pd.concat(synth_list)
        
        # Cleanup
        if 'Quantile' in synth_df.columns:
            synth_df = synth_df.drop(columns=['Quantile'])
            
        synth_df['unique_id'] = synth_df['unique_id'].apply(lambda x: f'{self.alias}_{x}')
        synth_df = synth_df[['ds', 'unique_id', 'y']]
        
        return synth_df
