
import numpy as np
import pandas as pd
from typing import Dict

from src.qgraph_ts import Grasynda


class GrasyndaTrend(Grasynda):
    """
    Grasynda variant that applies quantile-based generation to the differentiated trend.
    
    Instead of generating on the remainder component, this variant:
    - Differentiates the trend
    - Generates synthetic differentiated trend using quantiles
    - Integrates back to get synthetic trend
    - Combines with original remainder and seasonal components
    """
    
    def __init__(self,
                 n_quantiles: int,
                 period: int,
                 ensemble_transitions: bool = False,
                 ensemble_size: int = 5,
                 robust: bool = False):
        
        # Initialize parent with quantile_on='trend' (we'll override this)
        super().__init__(
            n_quantiles=n_quantiles,
            quantile_on='remainder',  # We'll manually handle this
            period=period,
            ensemble_transitions=ensemble_transitions,
            ensemble_size=ensemble_size,
            robust=robust
        )
        
        self.alias = 'GrasyndaTrend'
    
    def transform(self, df: pd.DataFrame, **kwargs):

        df_ = df.copy()
        
        df_ = self.decompose_tsd(df_, period=self.period, robust=self.robust)
        
        df_ = self._differentiate_trend(df_)
        
        df_['Quantile'] = self._get_quantiles_on_diff_trend(df_)
        

        self._calc_transition_matrix(df_)
        if self.ensemble_transitions:
            self.ensemble_transition_mats = self._get_ensemble_transition_mats()
        
        synth_diff_trend_dict = self._create_synthetic_diff_trend(df_)
        
        #Integrate to get synthetic trend
        synth_trend_dict = self._integrate_diff_trend(df_, synth_diff_trend_dict)
        
        #Combine synthetic trend with original remainder and seasonal
        synth_df = self._postprocess_trend_df(df_, synth_trend_dict)
        
        return synth_df
    
    def _differentiate_trend(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add a 'diff_trend' column containing the first-order difference of the trend.
        """
        diff_trend_list = []
        
        for uid, group in df.groupby('unique_id'):
            trend = group['trend'].values
            
            # First-order difference: trend[t] - trend[t-1]
            # Instead of prepending trend[0] (which gives 0), we prepend the first difference
            # to maintain length N and avoid a spike at 0.
            first_diff = trend[1] - trend[0] if len(trend) > 1 else 0
            diff_trend = np.diff(trend, prepend=trend[0] - first_diff)
            
            # This makes diff_trend[0] = trend[0] - (trend[0] - first_diff) = first_diff
            
            diff_trend_list.append(pd.DataFrame({
                'unique_id': uid,
                'ds': group['ds'],
                'diff_trend': diff_trend
            }))
        
        diff_trend_df = pd.concat(diff_trend_list)
        df_ = pd.merge(df, diff_trend_df, on=['unique_id', 'ds'])
        
        return df_
    
    def _get_quantiles_on_diff_trend(self, df: pd.DataFrame):
        """
        Calculate quantiles based on the differentiated trend.
        """
        quantiles = df.groupby('unique_id')['diff_trend'].transform(
            lambda x: pd.qcut(x, self.n_quantiles, labels=False, duplicates='drop')
        )
        return quantiles
    
    def _create_synthetic_diff_trend(self, df: pd.DataFrame) -> Dict:

        quantile_series = self._generate_quantile_series(df)
        generated_diff_trend = {}
        
        uids = df['unique_id'].unique().tolist()
        for uid in uids:
            uid_df = df.query(f'unique_id=="{uid}"')
            uid_diff_trend = uid_df['diff_trend']
            uid_quantiles = uid_df['Quantile']
            
            # Store min/max values for each quantile to define the bin range
            uid_q_ranges = {}
            for q in range(self.n_quantiles):
                vals = uid_diff_trend[uid_quantiles == q].values
                if len(vals) > 0:
                    uid_q_ranges[q] = (np.min(vals), np.max(vals))
                else:
                    uid_q_ranges[q] = None
            
            synth_diff = np.zeros(len(uid_diff_trend))
            synth_diff[0] = uid_diff_trend.values[0]
            
            for i in range(1, len(uid_quantiles)):
                current_quantile = quantile_series[uid][i]
                bin_range = uid_q_ranges.get(current_quantile, None)
                
                if bin_range is None:
                    # No samples - repeat last value
                    synth_diff[i] = synth_diff[i - 1]
                else:
                    # Continuous Uniform sampling within the bin range [min, max]
                    min_val, max_val = bin_range
                    if min_val == max_val:
                        sampled_val = min_val
                    else:
                        sampled_val = np.random.uniform(min_val, max_val)
                    synth_diff[i] = sampled_val
            
            generated_diff_trend[uid] = pd.Series(synth_diff, index=uid_df.index)
        
        return generated_diff_trend
    
    def _integrate_diff_trend(self, df: pd.DataFrame, synth_diff_trend_dict: Dict) -> Dict:
        """
        Reverse the differentiation using cumulative sum to get synthetic trend.

        """
        synth_trend_dict = {}
        
        for uid, uid_df in df.groupby('unique_id'):
            synth_diff = synth_diff_trend_dict[uid].values
            original_trend = uid_df['trend'].values
            
            # Start with the original first trend value
            synth_trend = np.zeros(len(synth_diff))
            synth_trend[0] = original_trend[0]
            
            # Cumulative sum to reverse differentiation
            # trend[t] = trend[0] + sum(diff[1:t+1])
            synth_trend[1:] = original_trend[0] + np.cumsum(synth_diff[1:])
            
            synth_trend_dict[uid] = pd.Series(synth_trend, index=uid_df.index)
        
        return synth_trend_dict
    
    def _postprocess_trend_df(self, df: pd.DataFrame, synth_trend_dict: Dict):

        synth_list = []
        
        for uid, uid_df in df.groupby('unique_id'):
            uid_df_copy = uid_df.copy()
            
            # Replace trend with synthetic trend
            uid_df_copy['trend'] = synth_trend_dict[uid].values
            
            synth_list.append(uid_df_copy)
        
        synth_df = pd.concat(synth_list)
        
        # Reconstruct
        synth_df['y'] = synth_df[['trend', 'seasonal', 'remainder']].sum(axis=1)
          
        synth_df = synth_df.drop(columns=['trend', 'seasonal', 'remainder', 'Quantile', 'diff_trend'])
        synth_df['unique_id'] = synth_df['unique_id'].apply(lambda x: f'{self.alias}_{x}')
        synth_df = synth_df[['ds', 'unique_id', 'y']]
        
        return synth_df
