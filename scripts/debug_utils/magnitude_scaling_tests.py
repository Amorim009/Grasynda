
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
from utils.load_data.config import DATASETS
from src.grasynda_unified import GrasyndaUnified

def create_balanced_input_corrected(df_series, period):
    """
    Balance Trend and Seasonality magnitudes while preserving means.
    
    Key fix: Center components before scaling, then add back means.
    """
    series = df_series['y'].values
    
    if len(series) < 2 * period:
        return df_series.copy()
    
    # Decompose
    res = STL(series, period=period, robust=False).fit()
    trend = res.trend
    seasonal = res.seasonal
    remainder = res.resid
    
    # Calculate magnitudes (std of centered components)
    std_t = np.std(trend)
    std_s = np.std(seasonal)
    
    # Target: average of the two
    target_std = (std_t + std_s) / 2
    
    # Scale factors
    scale_t = target_std / std_t if std_t > 0 else 1.0
    scale_s = target_std / std_s if std_s > 0 else 1.0
    
    # CORRECTED LOGIC: Center, scale, then restore mean
    mean_t = trend.mean()
    mean_s = seasonal.mean()  # Should be ~0 for seasonal
    
    trend_centered = trend - mean_t
    seasonal_centered = seasonal - mean_s
    
    trend_scaled = trend_centered * scale_t + mean_t
    seasonal_scaled = seasonal_centered * scale_s + mean_s
    
    # Reconstruct
    y_balanced = trend_scaled + seasonal_scaled + remainder
    
    # Create balanced dataframe
    df_balanced = df_series.copy()
    df_balanced['y'] = y_balanced
    
    return df_balanced

def demonstrate_corrected_multi(dataset_name='M3', group='Monthly', n_series=5):
    print(f"Demonstrating CORRECTED Balanced Input on {n_series} series...")
    
    # Load Data
    data_loader = DATASETS[dataset_name]
    df, _, _, _, freq_int = data_loader.load_everything(group)
    
    # Select series
    all_uids = df['unique_id'].unique()
    selected_uids = all_uids[:n_series]
    
    # Configure Model
    config = {
        'period': freq_int,
        'n_quantiles': 25,
        'components_to_model': ['trend', 'remainder'],
        'component_params': {
            'trend': {'sampling_type': 'discrete', 'apply_differentiation': True},
            'remainder': {'sampling_type': 'discrete'}
        }
    }
    model = GrasyndaUnified(**config)
    
    # Process each series
    output_dir = 'assets/balanced_corrected_demo'
    os.makedirs(output_dir, exist_ok=True)
    
    for idx, uid in enumerate(selected_uids):
        print(f"\n[{idx+1}/{n_series}] Processing {uid}...")
        
        df_series = df[df['unique_id'] == uid].copy()
        
        # Create balanced input (CORRECTED)
        df_balanced = create_balanced_input_corrected(df_series, freq_int)
        
        # Generate baseline and balanced
        synth_base = model.transform(df_series)
        synth_balanced = model.transform(df_balanced)
        
        base_ts = synth_base[synth_base['unique_id'] == f'GrasyndaUnified_{uid}']
        bal_ts = synth_balanced[synth_balanced['unique_id'] == f'GrasyndaUnified_{uid}']
        
        # Create side-by-side plot
        fig, axes = plt.subplots(1, 2, figsize=(16, 5))
        
        # Left: Original + Baseline
        axes[0].plot(df_series['ds'], df_series['y'], color='black', label='Original', lw=2)
        axes[0].plot(base_ts['ds'], base_ts['y'], color='blue', linestyle='--', label='Baseline', lw=1.5, alpha=0.7)
        axes[0].set_title(f"{uid} - Original vs Baseline")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Right: Original + Balanced
        axes[1].plot(df_series['ds'], df_series['y'], color='black', label='Original', lw=2)
        axes[1].plot(bal_ts['ds'], bal_ts['y'], color='red', label='Balanced (Corrected)', lw=2)
        axes[1].set_title(f"{uid} - Original vs Balanced")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = f"{output_dir}/{uid}_comparison.png"
        plt.savefig(save_path, dpi=120)
        plt.close()
        print(f"Saved to {save_path}")
        
        # Print stats
        print(f"  Original range: [{df_series['y'].min():.1f}, {df_series['y'].max():.1f}]")
        print(f"  Baseline range: [{base_ts['y'].min():.1f}, {base_ts['y'].max():.1f}]")
        print(f"  Balanced range: [{bal_ts['y'].min():.1f}, {bal_ts['y'].max():.1f}]")
        print(f"  Range preserved: {abs((bal_ts['y'].max() - bal_ts['y'].min()) / (df_series['y'].max() - df_series['y'].min()) - 1.0) < 0.2}")

if __name__ == "__main__":
    demonstrate_corrected_multi(n_series=15)
