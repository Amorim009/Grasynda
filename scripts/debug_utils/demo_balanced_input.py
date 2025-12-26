
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
from utils.load_data.config import DATASETS
from src.grasynda_unified import GrasyndaUnified

def create_balanced_input(df_series, period):
    """
    Balance Trend and Seasonality magnitudes.
    
    Strategy:
    1. Decompose into T, S, R
    2. Calculate std(T) and std(S)
    3. Scale both to average magnitude
    4. Reconstruct balanced series
    """
    series = df_series['y'].values
    
    if len(series) < 2 * period:
        return df_series.copy()
    
    # Decompose
    res = STL(series, period=period, robust=False).fit()
    trend = res.trend
    seasonal = res.seasonal
    remainder = res.resid
    
    # Calculate magnitudes
    std_t = np.std(trend)
    std_s = np.std(seasonal)
    
    print(f"Original Magnitudes: Trend std={std_t:.2f}, Seasonal std={std_s:.2f}")
    
    # Target: average of the two
    target_std = (std_t + std_s) / 2
    
    # Scale factors
    scale_t = target_std / std_t if std_t > 0 else 1.0
    scale_s = target_std / std_s if std_s > 0 else 1.0
    
    print(f"Scaling: Trend by {scale_t:.2f}, Seasonal by {scale_s:.2f} -> Target std={target_std:.2f}")
    
    # Apply scaling
    trend_balanced = trend * scale_t
    seasonal_balanced = seasonal * scale_s
    
    # Reconstruct
    y_balanced = trend_balanced + seasonal_balanced + remainder
    
    # Create balanced dataframe
    df_balanced = df_series.copy()
    df_balanced['y'] = y_balanced
    
    return df_balanced

def demonstrate_balanced_generation(dataset_name='M3', group='Monthly', uid='M806'):
    print(f"Demonstrating Balanced Input Generation on {uid}...")
    
    # Load Data
    data_loader = DATASETS[dataset_name]
    df, _, _, _, freq_int = data_loader.load_everything(group)
    df_series = df[df['unique_id'] == uid].copy()
    
    # Configure Model (Standard Hybrid)
    config = {
        'period': freq_int,
        'n_quantiles': 25,
        'components_to_model': ['trend', 'remainder'],
        'component_params': {
            'trend': {'sampling_type': 'continuous_uniform', 'apply_differentiation': True},
            'remainder': {'sampling_type': 'discrete'}
        }
    }
    model = GrasyndaUnified(**config)
    
    # 1. Baseline (Original Input)
    print("\n=== Generating Baseline (Original) ===")
    synth_base = model.transform(df_series)
    
    # 2. Balanced Input Strategy
    print("\n=== Creating Balanced Input ===")
    df_balanced = create_balanced_input(df_series, freq_int)
    
    print("\n=== Generating from Balanced Input ===")
    synth_balanced = model.transform(df_balanced)
    
    # VALIDATION: Check generated data
    print("\n=== Validation ===")
    print(f"Original Series Range: {df_series['y'].min():.2f} to {df_series['y'].max():.2f}")
    print(f"Balanced Input Range: {df_balanced['y'].min():.2f} to {df_balanced['y'].max():.2f}")
    
    bal_ts = synth_balanced[synth_balanced['unique_id'] == f'GrasyndaUnified_{uid}']
    print(f"Balanced Generation Rows: {len(bal_ts)}")
    
    if len(bal_ts) == 0:
        print("ERROR: No balanced generation data found!")
        return
    
    print(f"Balanced Generation Range: {bal_ts['y'].min():.2f} to {bal_ts['y'].max():.2f}")
    print(f"Balanced Generation NaNs: {bal_ts['y'].isna().sum()}")
    print(f"First 5 values: {bal_ts['y'].head().values}")
    
    # 3. Plotting
    output_dir = 'assets/balanced_input_demo'
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Top: Original vs Baseline
    axes[0].plot(df_series['ds'], df_series['y'], color='black', label='Original', lw=2, alpha=0.8)
    
    base_ts = synth_base[synth_base['unique_id'] == f'GrasyndaUnified_{uid}']
    axes[0].plot(base_ts['ds'], base_ts['y'], color='gray', linestyle='--', label='Baseline (Rigid Seasonality)', alpha=0.6, lw=1.5)
    
    axes[0].set_title(f"Original Input → Baseline Generation ({uid})")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Bottom: ONLY Balanced Output vs Original (clear comparison)
    axes[1].plot(df_series['ds'], df_series['y'], color='black', label='Original', lw=2, alpha=0.8)
    axes[1].plot(bal_ts['ds'], bal_ts['y'], color='red', label='Balanced Generation (Variable)', lw=2.5)
    
    axes[1].set_title(f"Balanced Generation vs Original ({uid})")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = f"{output_dir}/{uid}_balanced_input.png"
    plt.savefig(save_path, dpi=120)
    print(f"\nSaved to {save_path}")

if __name__ == "__main__":
    demonstrate_balanced_generation()
