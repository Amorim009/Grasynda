
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.load_data.config import DATASETS
from src.grasynda_unified import GrasyndaUnified

def compare_sampling_methods(dataset_name='M3', group='Monthly', uid='M1165'):
    print(f"Comparing Sampling Methods for {uid} (N=1)...")
    
    data_loader = DATASETS[dataset_name]
    df, horizon, n_lags, freq_str, freq_int = data_loader.load_everything(group)
    uid_df = df[df['unique_id'] == uid].copy()
    
    # 1. Discrete Sampling (Original Grasynda behavior)
    discrete_model = GrasyndaUnified(
        period=freq_int,
        n_quantiles=1,
        components_to_model=['trend', 'remainder'],
        apply_differentiation=True,
        sampling_type='discrete'
    )
    
    # 2. Continuous Uniform Sampling (Min-Max Box)
    uniform_model = GrasyndaUnified(
        period=freq_int,
        n_quantiles=1,
        components_to_model=['trend', 'remainder'],
        apply_differentiation=True,
        sampling_type='continuous_uniform'
    )
    
    # Run transformations
    synth_discrete = discrete_model.transform(uid_df)
    synth_uniform = uniform_model.transform(uid_df)
    
    # Extraction for custom plotting (Intermediate components)
    decomposed = discrete_model.decompose_tsd(uid_df, freq_int, False)
    orig_rem = decomposed['remainder'].values
    
    trend_prep = discrete_model._differentiate_component(decomposed, 'trend')
    orig_slopes = trend_prep['diff_trend'].values
    
    # Manually extract the "uniform" components for visualization
    # Remainder
    rem_min, rem_max = orig_rem.min(), orig_rem.max()
    uni_rem = np.random.uniform(rem_min, rem_max, size=len(orig_rem))
    
    # Slopes
    slope_min, slope_max = orig_slopes.min(), orig_slopes.max()
    uni_slopes = np.random.uniform(slope_min, slope_max, size=len(orig_slopes))
    
    # --- Plotting ---
    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(3, 2)
    
    # Row 0: Full Series Comparison
    ax00 = fig.add_subplot(gs[0, 0])
    ax00.plot(uid_df['ds'], uid_df['y'], color='black', label='Original', alpha=0.5, linewidth=2)
    ax00.plot(uid_df['ds'], synth_discrete['y'], color='#3498db', label='Discrete (Shuffled)')
    ax00.set_title(f"Full Series: Discrete Sampling (N=1)")
    ax00.legend()
    
    ax01 = fig.add_subplot(gs[0, 1])
    ax01.plot(uid_df['ds'], uid_df['y'], color='black', label='Original', alpha=0.5, linewidth=2)
    ax01.plot(uid_df['ds'], synth_uniform['y'], color='#e74c3c', label='Continuous Uniform (Box)')
    ax01.set_title(f"Full Series: Continuous Uniform Sampling (N=1)")
    ax01.legend()
    
    # Row 1: Trend Slopes Comparison
    ax10 = fig.add_subplot(gs[1, 0])
    ax10.plot(uid_df['ds'], orig_slopes, color='black', alpha=0.3, label='Orig Slopes')
    # Extract from discrete run? Transform doesn't expose them easily, so we reproduce logic
    ax10.plot(uid_df['ds'], np.random.choice(orig_slopes, len(orig_slopes)), color='#3498db', label='Shuffled Slopes')
    ax10.set_title("Trend Slopes: Discrete (Related to Data)")
    ax10.legend()
    
    ax11 = fig.add_subplot(gs[1, 1])
    ax11.plot(uid_df['ds'], orig_slopes, color='black', alpha=0.3, label='Orig Slopes')
    ax11.plot(uid_df['ds'], uni_slopes, color='#e74c3c', label='Uniform Slopes (Min/Max)')
    ax11.set_title("Trend Slopes: Continuous Uniform (Flat Noise)")
    ax11.legend()
    
    # Row 2: Remainder Comparison
    ax20 = fig.add_subplot(gs[2, 0])
    ax20.plot(uid_df['ds'], orig_rem, color='black', alpha=0.3, label='Orig Rem')
    ax20.plot(uid_df['ds'], np.random.choice(orig_rem, len(orig_rem)), color='#2ecc71', label='Shuffled Rem')
    ax20.set_title("Remainder: Discrete (Preserves Spikes)")
    ax20.legend()
    
    ax21 = fig.add_subplot(gs[2, 1])
    ax21.plot(uid_df['ds'], orig_rem, color='black', alpha=0.3, label='Orig Rem')
    ax21.plot(uid_df['ds'], uni_rem, color='#2ecc71', label='Uniform Rem (Min/Max)')
    ax21.set_title("Remainder: Continuous Uniform (Loss of Character)")
    ax21.legend()
    
    plt.tight_layout()
    output_dir = "assets/sampling_audit"
    os.makedirs(output_dir, exist_ok=True)
    save_path = f"{output_dir}/compare_sampling_{uid}.png"
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"Saved comparison to {save_path}")

if __name__ == "__main__":
    compare_sampling_methods('M3', 'Monthly', 'M1165')
    compare_sampling_methods('Tourism', 'Monthly', 'm11')
