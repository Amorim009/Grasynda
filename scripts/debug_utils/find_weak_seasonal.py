
import os
import numpy as np
import pandas as pd
from utils.load_data.config import DATASETS
from src.grasynda_unified import GrasyndaUnified

def find_weakest_seasonal(dataset_name='M3', group='Monthly', top_n=5):
    data_loader = DATASETS[dataset_name]
    df, horizon, n_lags, freq_str, freq_int = data_loader.load_everything(group)
    
    uids = df['unique_id'].unique()
    grasynda = GrasyndaUnified(period=freq_int)
    
    # Sample a subset to keep it fast, or process all if small
    subset_uids = uids[:200]
    df_subset = df[df['unique_id'].isin(subset_uids)].copy()
    
    decomposed = grasynda.decompose_tsd(df_subset, freq_int, False)
    
    results = []
    for uid in subset_uids:
        group_df = decomposed[decomposed['unique_id'] == uid]
        y_std = group_df['y'].std()
        seasonal_std = group_df['seasonal'].std()
        
        # We want the lowest ratio of Seasonal / Y
        results.append({
            'unique_id': uid,
            'seasonal_ratio': seasonal_std / y_std if y_std > 0 else 1.0
        })
    
    res_df = pd.DataFrame(results).sort_values('seasonal_ratio')
    return res_df['unique_id'].head(top_n).tolist()

if __name__ == "__main__":
    print("Finding weakest seasonal series...")
    m3_weak = find_weakest_seasonal('M3', 'Monthly')
    print(f"M3 Weakest Seasonal UIDs: {m3_weak}")
    
    tourism_weak = find_weakest_seasonal('Tourism', 'Monthly')
    print(f"Tourism Weakest Seasonal UIDs: {tourism_weak}")
