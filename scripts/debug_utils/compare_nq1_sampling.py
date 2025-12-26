
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.load_data.config import DATASETS
from src.grasynda_unified import GrasyndaUnified

def compare_nq1_sampling(dataset_name='M3', group='Monthly', uid='M1165'):
    print(f"Comparing NQ=1 Sampling: Discrete vs Uniform Probability for {uid}...")
    
    data_loader = DATASETS[dataset_name]
    df, horizon, n_lags, freq_str, freq_int = data_loader.load_everything(group)
    uid_df = df[df['unique_id'] == uid].copy()
    
    # 1. Discrete Sampling Model
    discrete_model = GrasyndaUnified(
        period=freq_int,
        n_quantiles=1,
        components_to_model=['trend', 'remainder'],
        apply_differentiation=True,
        sampling_type='discrete'
    )
    
    # 2. Uniform Probability Sampling Model (Continuous Uniform)
    uniform_model = GrasyndaUnified(
        period=freq_int,
        n_quantiles=1,
        components_to_model=['trend', 'remainder'],
        apply_differentiation=True,
        sampling_type='continuous_uniform'
    )
    
    # Generate
    synth_discrete = discrete_model.transform(uid_df)
    synth_uniform = uniform_model.transform(uid_df)
    
    # --- Plotting ---
    fig, axes = plt.subplots(3, 1, figsize=(15, 18), sharex=True)
    
    # Full Series Comparison
    axes[0].plot(uid_df['ds'], uid_df['y'], color='black', label='Original', linewidth=2, alpha=0.6)
    axes[0].plot(uid_df['ds'], synth_discrete['y'], color='#3498db', label='Discrete Sampling (Shuffle)', alpha=0.8)
    axes[0].plot(uid_df['ds'], synth_uniform['y'], color='#e74c3c', label='Uniform Prob Sampling (Min-Max)', alpha=0.8)
    axes[0].set_title(f"Reconstructed Series: {uid}")
    axes[0].legend()
    
    # Component comparison (Trend Slopes)
    # We'll extract them for visual audit
    decomposed = discrete_model.decompose_tsd(uid_df, freq_int, False)
    trend_prep = discrete_model._differentiate_component(decomposed, 'trend')
    orig_slopes = trend_prep['diff_trend'].values
    
    # For Discrete: Slopes are a shuffle of original
    synth_slopes_discrete = np.random.choice(orig_slopes, len(orig_slopes))
    # For Uniform: Slopes are sampled from min-max
    s_min, s_max = orig_slopes.min(), orig_slopes.max()
    synth_slopes_uniform = np.random.uniform(s_min, s_max, len(orig_slopes))
    
    axes[1].plot(uid_df['ds'], orig_slopes, color='black', alpha=0.3, label='Original Slopes')
    axes[1].plot(uid_df['ds'], synth_slopes_discrete, color='#3498db', label='Discrete Slopes', alpha=0.7)
    axes[1].plot(uid_df['ds'], synth_slopes_uniform, color='#e74c3c', label='Uniform Prob Slopes', alpha=0.7)
    axes[1].set_title("Component Trace: Trend Slopes (Character Check)")
    axes[1].legend()
    
    # Remainder comparison
    orig_rem = decomposed['remainder'].values
    synth_rem_discrete = np.random.choice(orig_rem, len(orig_rem))
    r_min, r_max = orig_rem.min(), orig_rem.max()
    synth_rem_uniform = np.random.uniform(r_min, r_max, len(orig_rem))
    
    axes[2].plot(uid_df['ds'], orig_rem, color='black', alpha=0.3, label='Original Remainder')
    axes[2].plot(uid_df['ds'], synth_rem_discrete, color='#2ecc71', label='Discrete Remainder', alpha=0.7)
    axes[2].plot(uid_df['ds'], synth_rem_uniform, color='#f1c40f', label='Uniform Prob Remainder', alpha=0.7)
    axes[2].set_title("Component Trace: Remainder (Spike Check)")
    axes[2].legend()
    
    plt.tight_layout()
    output_dir = 'assets/nq1_comparison_detailed'
    os.makedirs(output_dir, exist_ok=True)
    save_path = f"{output_dir}/{uid}_sampling_compare.png"
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"Results saved to {save_path}")

if __name__ == "__main__":
    compare_nq1_sampling('M3', 'Monthly', 'M1165')
    compare_nq1_sampling('Tourism', 'Monthly', 'm11')
