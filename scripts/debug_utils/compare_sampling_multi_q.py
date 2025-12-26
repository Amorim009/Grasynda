
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.load_data.config import DATASETS
from src.grasynda_unified import GrasyndaUnified

def compare_sampling_q(n_q, dataset_name='M3', group='Monthly', uid='M1165'):
    print(f"Comparing NQ={n_q} Sampling: Discrete vs Uniform Probability for {uid}...")
    
    data_loader = DATASETS[dataset_name]
    df, horizon, n_lags, freq_str, freq_int = data_loader.load_everything(group)
    uid_df = df[df['unique_id'] == uid].copy()
    
    # 1. Discrete Sampling Model
    discrete_model = GrasyndaUnified(
        period=freq_int,
        n_quantiles=n_q,
        components_to_model=['trend', 'remainder'],
        apply_differentiation=True,
        sampling_type='discrete'
    )
    
    # 2. Uniform Probability Sampling Model (Continuous Uniform)
    uniform_model = GrasyndaUnified(
        period=freq_int,
        n_quantiles=n_q,
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
    axes[0].plot(uid_df['ds'], synth_discrete['y'], color='#3498db', label=f'Discrete (NQ={n_q})', alpha=0.8)
    axes[0].plot(uid_df['ds'], synth_uniform['y'], color='#e74c3c', label=f'Uniform Prob (NQ={n_q})', alpha=0.8)
    axes[0].set_title(f"Reconstructed Series: {uid} | NQ={n_q}")
    axes[0].legend()
    
    # Component comparison (Trend Slopes)
    # We'll extract them for visual audit
    decomposed = discrete_model.decompose_tsd(uid_df, freq_int, False)
    trend_prep = discrete_model._differentiate_component(decomposed, 'trend')
    orig_slopes = trend_prep['diff_trend'].values
    
    # For Discrete: Slopes are a shuffle of original (with NQ=1) or transition-based (NQ>1)
    # To be accurate, we'll extract from the actual model run if possible, 
    # but transform() doesn't expose them. I'll simulate the sampling logic here for the plot.
    
    def simulate_sampling(model, df_in, target_col, sampling_type, component):
        # Setup matrix and quantiles
        df_local = df_in.copy()
        df_local['Quantile'] = model._get_quantiles(df_local, target_col, component=component)
        model._calc_transition_matrix(df_local, component=component)
        
        # Q Path
        q_path = model._generate_quantile_series(df_local, component)[uid]
        
        # Sample
        synth_vals = np.zeros(len(df_local))
        uid_vals = df_local[target_col].values
        uid_quantiles = df_local['Quantile'].values
        
        for i in range(len(synth_vals)):
            q = q_path[i]
            bin_vals = uid_vals[uid_quantiles == q]
            if len(bin_vals) == 0:
                synth_vals[i] = synth_vals[i-1] if i > 0 else 0
                continue
                
            if sampling_type == 'discrete':
                synth_vals[i] = np.random.choice(bin_vals)
            else:
                synth_vals[i] = np.random.uniform(bin_vals.min(), bin_vals.max())
        return synth_vals

    synth_slopes_discrete = simulate_sampling(discrete_model, trend_prep, 'diff_trend', 'discrete', 'trend')
    synth_slopes_uniform = simulate_sampling(uniform_model, trend_prep, 'diff_trend', 'continuous_uniform', 'trend')
    
    axes[1].plot(uid_df['ds'], orig_slopes, color='black', alpha=0.3, label='Original Slopes')
    axes[1].plot(uid_df['ds'], synth_slopes_discrete, color='#3498db', label='Discrete Slopes', alpha=0.7)
    axes[1].plot(uid_df['ds'], synth_slopes_uniform, color='#e74c3c', label='Uniform Prob Slopes', alpha=0.7)
    axes[1].set_title(f"Component Trace: Trend Slopes (NQ={n_q})")
    axes[1].legend()
    
    # Remainder comparison
    orig_rem = decomposed['remainder'].values
    synth_rem_discrete = simulate_sampling(discrete_model, decomposed, 'remainder', 'discrete', 'remainder')
    synth_rem_uniform = simulate_sampling(uniform_model, decomposed, 'remainder', 'continuous_uniform', 'remainder')
    
    axes[2].plot(uid_df['ds'], orig_rem, color='black', alpha=0.3, label='Original Remainder')
    axes[2].plot(uid_df['ds'], synth_rem_discrete, color='#2ecc71', label='Discrete Remainder', alpha=0.7)
    axes[2].plot(uid_df['ds'], synth_rem_uniform, color='#f1c40f', label='Uniform Prob Remainder', alpha=0.7)
    axes[2].set_title(f"Component Trace: Remainder (NQ={n_q})")
    axes[2].legend()
    
    plt.tight_layout()
    output_dir = f'assets/sampling_compare_detailed_nq{n_q}'
    os.makedirs(output_dir, exist_ok=True)
    save_path = f"{output_dir}/{uid}_sampling_compare.png"
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"Results saved to {save_path}")

if __name__ == "__main__":
    for nq in [25, 50, 1000]:
        compare_sampling_q(nq, 'M3', 'Monthly', 'M1165')
        compare_sampling_q(nq, 'Tourism', 'Monthly', 'm11')
