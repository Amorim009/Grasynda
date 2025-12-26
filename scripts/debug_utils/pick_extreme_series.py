
import os
import numpy as np
import pandas as pd
from utils.load_data.config import DATASETS
from src.grasynda_unified import GrasyndaUnified

def find_extremes(dataset_name='M3', group='Monthly', n=3):
    data_loader = DATASETS[dataset_name]
    df, horizon, n_lags, freq_str, freq_int = data_loader.load_everything(group)
    
    uids = df['unique_id'].unique()[:200] # Process 200 for speed
    df_sample = df[df['unique_id'].isin(uids)].copy()
    
    grasynda = GrasyndaUnified(period=freq_int)
    decomposed = grasynda.decompose_tsd(df_sample, freq_int, False)
    
    stats = []
    for uid in uids:
        group_df = decomposed[decomposed['unique_id'] == uid]
        y_std = group_df['y'].std()
        s_std = group_df['seasonal'].std()
        stats.append({'uid': uid, 's_ratio': s_std / y_std if y_std > 0 else 0})
    
    res = pd.DataFrame(stats).sort_values('s_ratio')
    low = res.head(n)['uid'].tolist()
    high = res.tail(n)['uid'].tolist()
    return low, high

if __name__ == "__main__":
    for d in ['M3', 'Tourism']:
        low, high = find_extremes(d, 'Monthly')
        print(f"\n{d} Monthly:")
        print(f"  Low Seasonality:  {low}")
        print(f"  High Seasonality: {high}")
