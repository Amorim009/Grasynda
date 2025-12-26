
import numpy as np
import pandas as pd
from typing import Dict

from src.qgraph_ts import Grasynda
from src.grasynda_trend import GrasyndaTrend
from src.grasynda_continuous import GrasyndaContinuousComponent

class GrasyndaHybrid(GrasyndaTrend):
    """
    Grasynda variant that generates BOTH synthetic trend and synthetic remainder.
    
    Final Series = Synthetic Trend + Synthetic Remainder + Original Seasonality
    
    - Trend: Generated using GrasyndaTrend logic (Derivative + Continuous Sampling)
    - Remainder: Generated using GrasyndaContinuousComponent logic (Continuous Sampling)
    """
    
    def __init__(self,
                 n_quantiles: int,
                 period: int,
                 ensemble_transitions: bool = False,
                 ensemble_size: int = 5,
                 robust: bool = False):
        
        super().__init__(
            n_quantiles=n_quantiles,
            period=period,
            ensemble_transitions=ensemble_transitions,
            ensemble_size=ensemble_size,
            robust=robust
        )
        self.alias = 'GrasyndaHybrid'
        
        # We need a separate instance for the remainder generation to avoid state conflicts
        # We use GrasyndaContinuousComponent to ensure continuous sampling for the remainder
        self.remainder_gen = GrasyndaContinuousComponent(
            n_quantiles=n_quantiles,
            quantile_on='remainder',
            period=period,
            ensemble_transitions=ensemble_transitions,
            ensemble_size=ensemble_size,
            robust=robust
        )

    def transform(self, df: pd.DataFrame, **kwargs):
        
        # 1. Decompose (Shared)
        df_ = df.copy()
        df_ = self.decompose_tsd(df_, period=self.period, robust=self.robust)
        
        # --- TREND GENERATION (GrasyndaTrend Logic) ---
        # We use 'self' which is a GrasyndaTrend instance
        
        # Differentiate Trend
        df_trend = self._differentiate_trend(df_)
        
        # Calculate Quantiles on Diff Trend
        df_trend['Quantile'] = self._get_quantiles_on_diff_trend(df_trend)
        
        # Calc Transitions for Trend
        self._calc_transition_matrix(df_trend)
        if self.ensemble_transitions:
            self.ensemble_transition_mats = self._get_ensemble_transition_mats()
            
        # Generate Synthetic Diff Trend
        synth_diff_trend_dict = self._create_synthetic_diff_trend(df_trend)
        
        # Integrate to get Synthetic Trend
        synth_trend_dict = self._integrate_diff_trend(df_trend, synth_diff_trend_dict)
        
        
        # --- REMAINDER GENERATION (Continuous Logic) ---
        # We use 'self.remainder_gen'
        
        # Calculate Quantiles on Remainder
        # Note: Grasynda._get_quantiles expects 'remainder' column which exists in df_
        df_rem = df_.copy()
        df_rem['Quantile'] = self.remainder_gen._get_quantiles(df_rem)
        
        # Calc Transitions for Remainder
        self.remainder_gen._calc_transition_matrix(df_rem)
        if self.remainder_gen.ensemble_transitions:
            self.remainder_gen.ensemble_transition_mats = self.remainder_gen._get_ensemble_transition_mats()
            
        # Generate Synthetic Remainder (Continuous)
        synth_remainder_dict = self.remainder_gen._create_synthetic_ts(df_rem)
        
        
        # --- COMBINE ---
        synth_df = self._postprocess_hybrid(df_, synth_trend_dict, synth_remainder_dict)
        
        return synth_df

    def _postprocess_hybrid(self, df: pd.DataFrame, synth_trend_dict: Dict, synth_remainder_dict: Dict):
        
        synth_list = []
        
        for uid, uid_df in df.groupby('unique_id'):
            uid_df_copy = uid_df.copy()
            
            # Replace trend with synthetic trend
            uid_df_copy['trend'] = synth_trend_dict[uid].values
            
            # Replace remainder with synthetic remainder
            uid_df_copy['remainder'] = synth_remainder_dict[uid].values
            
            synth_list.append(uid_df_copy)
        
        synth_df = pd.concat(synth_list)
        
        # Reconstruct: SynthTrend + SynthRemainder + OriginalSeasonal
        synth_df['y'] = synth_df[['trend', 'seasonal', 'remainder']].sum(axis=1)
          
        cols_to_drop = ['trend', 'seasonal', 'remainder', 'Quantile', 'diff_trend']
        synth_df = synth_df.drop(columns=cols_to_drop, errors='ignore')
            
        synth_df['unique_id'] = synth_df['unique_id'].apply(lambda x: f'{self.alias}_{x}')
        synth_df = synth_df[['ds', 'unique_id', 'y']]
        
        return synth_df
