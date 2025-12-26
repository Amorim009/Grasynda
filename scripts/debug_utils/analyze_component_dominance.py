
import os
import numpy as np
import pandas as pd
from utils.load_data.config import DATASETS
from src.grasynda_unified import GrasyndaUnified

def analyze_components(dataset_name='M3', group='Monthly', n_samples=5):
    data_loader = DATASETS[dataset_name]
    df, horizon, n_lags, freq_str, freq_int = data_loader.load_everything(group)
    
    sample_uids = df['unique_id'].unique()[:n_samples]
    df_sample = df[df['unique_id'].isin(sample_uids)].copy()
    
    grasynda = GrasyndaUnified(period=freq_int)
    # decompose_tsd requires the dataframe to have 'y' and 'unique_id'
    df_decomposed = grasynda.decompose_tsd(df_sample, freq_int, False)
    
    results = []
    for uid in sample_uids:
        series_df = df_decomposed[df_decomposed['unique_id'] == uid]
        y_std = series_df['y'].std()
        trend_std = series_df['trend'].std()
        seasonal_std = series_df['seasonal'].std()
        rem_std = series_df['remainder'].std()
        
        results.append({
            'uid': uid,
            'y_std': y_std,
            'trend_rel': trend_std / y_std,
            'seasonal_rel': seasonal_std / y_std,
            'rem_rel': rem_std / y_std
        })
    
    return pd.DataFrame(results)

if __name__ == "__main__":
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print("M3 Monthly Analysis:")
    print(analyze_components('M3', 'Monthly'))
    print("\nTourism Monthly Analysis:")
    print(analyze_components('Tourism', 'Monthly'))
