
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
from utils.load_data.config import DATASETS
from src.grasynda_unified import GrasyndaUnified

def scale_seasonality(series, period, scale_range=(0.8, 1.2)):
    """
    Decomposes the series, scales the seasonal component by a random factor 
    within scale_range, and reconstructs the series.
    """
    if len(series) < 2 * period:
        return series
        
    res = STL(series, period=period, robust=False).fit()
    trend = res.trend
    seasonal = res.seasonal
    resid = res.resid
    
    # 1. Global Amplitude Scaling (Simplest)
    factor = np.random.uniform(*scale_range)
    seasonal_scaled = seasonal * factor
    
    # Reconstruct
    return trend + seasonal_scaled + resid

def demonstrate_perturbation(dataset_name='M3', group='Monthly', uid='M806'):
    print(f"Demonstrating Seasonal Perturbation on {uid}...")
    
    # 1. Load Data
    data_loader = DATASETS[dataset_name]
    df, _, _, _, freq_int = data_loader.load_everything(group)
    
    df_series = df[df['unique_id'] == uid].copy()
    if df_series.empty:
        print(f"Series {uid} not found.")
        return

    # 2. Configure Grasynda (Hybrid Fixed)
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

    # 3. Method A: Standard Execution (Baseline)
    print("Running Baseline Generation...")
    synth_base = model.transform(df_series)
    
    # 4. Method B: Input Seasonal Perturbation
    print("Applying Input Seasonal Perturbation...")
    # Create perturbed input
    y_perturbed = scale_seasonality(df_series['y'].values, freq_int, scale_range=(0.90, 1.10))
    df_perturbed = df_series.copy()
    df_perturbed['y'] = y_perturbed
    
    # Run model on perturbed input
    synth_perturbed = model.transform(df_perturbed)
    
    # 5. Method C: Seasonal warping (Non-linear scaling)
    # Scale peaks more than troughs? Or time shift?
    # Let's try simple Random Noise injection into seasonality specifically
    print("Applying Noisy Seasonal Perturbation...")
    def noisy_seasonality(series, period):
        res = STL(series, period=period).fit()
        # Add 5% proportional noise to seasonality
        noise = np.random.normal(0, 0.05, size=len(series)) * res.seasonal 
        return res.trend + (res.seasonal + noise) + res.resid
        
    y_noisy = noisy_seasonality(df_series['y'].values, freq_int)
    df_noisy = df_series.copy()
    df_noisy['y'] = y_noisy
    synth_noisy = model.transform(df_noisy)

    # 6. Plotting
    output_dir = 'assets/seasonal_perturbation_demo'
    os.makedirs(output_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Original
    ax.plot(df_series['ds'], df_series['y'], color='black', label='Original', lw=2, alpha=0.8)
    
    # Baseline (Overlap issues expected)
    base_ts = synth_base[synth_base['unique_id'] == f'GrasyndaUnified_{uid}']
    ax.plot(base_ts['ds'], base_ts['y'], color='gray', linestyle='--', label='Baseline Grasynda (Identical Seasonality)', alpha=0.6)
    
    # Perturbed Scaled
    pert_ts = synth_perturbed[synth_perturbed['unique_id'] == f'GrasyndaUnified_{uid}']
    ax.plot(pert_ts['ds'], pert_ts['y'], color='#2980b9', label='Perturbed Input (Scaled Seasonality)', lw=1.5)
    
    # Perturbed Noisy
    noisy_ts = synth_noisy[synth_noisy['unique_id'] == f'GrasyndaUnified_{uid}']
    ax.plot(noisy_ts['ds'], noisy_ts['y'], color='#e74c3c', label='Perturbed Input (Noisy Seasonality)', lw=1.5, alpha=0.8)
    
    ax.set_title(f"Addressing Seasonal Dominance: Input Perturbation Strategies ({uid})")
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = f"{output_dir}/perturbation_{uid}.png"
    plt.savefig(save_path, dpi=120)
    print(f"Saved to {save_path}")

if __name__ == "__main__":
    demonstrate_perturbation(uid='M806') # Known seasonal dominant
