
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.load_data.config import DATASETS
from src.grasynda_unified import GrasyndaUnified

def audit_similarity(dataset_name='M3', group='Monthly', uid=None):
    data_loader = DATASETS[dataset_name]
    df, horizon, n_lags, freq_str, freq_int = data_loader.load_everything(group)
    
    if uid is None:
        uid = df['unique_id'].unique()[0]
    
    uid_df = df[df['unique_id'] == uid].copy()
    
    grasynda = GrasyndaUnified(period=freq_int)
    decomposed = grasynda.decompose_tsd(uid_df, freq_int, False)
    
    # Components
    trend = decomposed['trend'].values
    seasonal = decomposed['seasonal'].values
    remainder = decomposed['remainder'].values
    
    # 1. Visualize Components
    fig, axes = plt.subplots(4, 1, figsize=(15, 12), sharex=True)
    axes[0].plot(uid_df['ds'], uid_df['y'], color='black', label='Original Y')
    axes[1].plot(uid_df['ds'], trend, color='blue', label='Trend')
    axes[2].plot(uid_df['ds'], seasonal, color='red', label='Seasonal (Fixed Timing)')
    axes[3].plot(uid_df['ds'], remainder, color='green', label='Remainder (Jitter Source)')
    
    for ax in axes: ax.legend()
    plt.suptitle(f"Component Audit: {uid}", fontsize=16)
    
    results_dir = 'scripts/debug_utils/audit_plots'
    os.makedirs(results_dir, exist_ok=True)
    plt.savefig(f"{results_dir}/audit_{uid}.png")
    
    # 2. Sensitivity Test: If we jitter the remainder 100%, how much does Y change?
    random_rem = np.random.choice(remainder, size=len(remainder))
    synth_y_fixed_season = trend + seasonal + random_rem
    synth_y_no_season = trend + np.mean(seasonal) + random_rem
    
    # Correlation with original Y
    corr_fixed = np.corrcoef(uid_df['y'], synth_y_fixed_season)[0, 1]
    corr_no_season = np.corrcoef(uid_df['y'], synth_y_no_season)[0, 1]
    
    print(f"\nSimilarity Audit for {uid}:")
    print(f"  - Correlation (with Original Seasonal): {corr_fixed:.4f}  <-- Still very high?")
    print(f"  - Correlation (NO Seasonal): {corr_no_season:.4f}            <-- This should be much lower")
    print(f"  - Std Ratio (Seasonal / Remainder): {np.std(seasonal)/np.std(remainder):.2f}")

if __name__ == "__main__":
    audit_similarity('M3', 'Monthly', 'M1065')
    audit_similarity('Tourism', 'Monthly', 'm1')
