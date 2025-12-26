
import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from utils.load_data.config import DATASETS
from src.grasynda_unified import GrasyndaUnified

def analyze_synthetic_impact(dataset_name='M3', group='Monthly', n_samples=50):
    print(f"Analyzing Synthetic Impact for {dataset_name} - {group} (n={n_samples})...")
    
    data_loader = DATASETS[dataset_name]
    df, horizon, n_lags, freq_str, freq_int = data_loader.load_everything(group)
    
    # Randomly select series
    all_uids = df['unique_id'].unique()
    if len(all_uids) > n_samples:
        np.random.seed(42)
        sample_uids = np.random.choice(all_uids, n_samples, replace=False)
    else:
        sample_uids = all_uids
        
    df_sample = df[df['unique_id'].isin(sample_uids)].copy()
    
    # Initialize Grasynda (Standard: Remainder only)
    model = GrasyndaUnified(
        period=freq_int,
        n_quantiles=25,
        components_to_model=['remainder'],
        sampling_type='discrete' # Standard default
    )
    
    # Generate Synthetic Data
    synth_df = model.transform(df_sample)
    
    # Analysis
    results = []
    
    # Pre-calculate components for dominance check
    decomposed = model.decompose_tsd(df_sample, freq_int, False)
    
    for uid in sample_uids:
        orig_y = df_sample[df_sample['unique_id'] == uid]['y'].values
        synth_row = synth_df[synth_df['unique_id'] == f'GrasyndaUnified_{uid}']
        
        if synth_row.empty:
            continue
            
        synth_y = synth_row['y'].values
        
        # 1. Similarity Metrics
        # Correlation
        corr, _ = pearsonr(orig_y, synth_y)
        
        # MAPE / MAE relative to mean
        mae = np.mean(np.abs(orig_y - synth_y))
        mape_mean = mae / np.mean(np.abs(orig_y)) 
        
        # 2. Component Dominance
        uid_decomp = decomposed[decomposed['unique_id'] == uid]
        std_y = uid_decomp['y'].std()
        std_trend = uid_decomp['trend'].std()
        std_seas = uid_decomp['seasonal'].std()
        std_rem = uid_decomp['remainder'].std()
        
        # Relative Contribution (simple ratio of stds)
        rel_trend = std_trend / std_y if std_y > 0 else 0
        rel_seas = std_seas / std_y if std_y > 0 else 0
        rel_rem = std_rem / std_y if std_y > 0 else 0
        
        results.append({
            'unique_id': uid,
            'Correlation': corr,
            'Rel_MAE': mape_mean,
            'Rel_Trend': rel_trend,
            'Rel_Seas': rel_seas,
            'Rel_Rem': rel_rem,
            'Dominant_Comp': 'Seasonal' if rel_seas > rel_trend and rel_seas > rel_rem else 
                             ('Trend' if rel_trend > rel_rem else 'Remainder')
        })
        
    res_df = pd.DataFrame(results)
    
    # Sort by Correlation (High correlation = Low Impact/Too Similar)
    res_df = res_df.sort_values('Correlation', ascending=False)
    
    print("\n--- Top 10 Most Similar (Least Impact) ---")
    print(res_df.head(10)[['unique_id', 'Correlation', 'Rel_MAE', 'Dominant_Comp', 'Rel_Seas', 'Rel_Rem']].to_string())
    
    print("\n--- Top 10 Least Similar (Most Impact) ---")
    print(res_df.tail(10)[['unique_id', 'Correlation', 'Rel_MAE', 'Dominant_Comp', 'Rel_Seas', 'Rel_Rem']].to_string())
    
    print("\n--- Summary Statistics ---")
    print(res_df.describe())
    
    return res_df

if __name__ == "__main__":
    analyze_synthetic_impact('M3', 'Monthly')
