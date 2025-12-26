
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.load_data.config import DATASETS
from src.grasynda_unified import GrasyndaUnified

def generate_component_vals(target_col, n_q, df_in, model_obj, is_diff=False):
    df_local = df_in.copy()
    df_local['Quantile'] = model_obj._get_quantiles(df_local, target_col, component=target_col.replace('diff_', ''))
    hist_vals = df_local[target_col].values
    
    synth = np.zeros(len(df_local))
    synth[0] = hist_vals[0]
    for i in range(1, len(synth)):
        synth[i] = np.random.choice(hist_vals)
        
    if is_diff:
        start = df_in['trend'].values[0]
        integrated = np.zeros(len(df_local))
        integrated[0] = start
        integrated[1:] = start + np.cumsum(synth[1:])
        return integrated, synth # Return both integrated and slopes
    return synth

def visualize_nq1_results(dataset_name='M3', group='Monthly', uid='M1165'):
    print(f"Generating 1-Quantile Results for {uid}...")
    
    data_loader = DATASETS[dataset_name]
    df, horizon, n_lags, freq_str, freq_int = data_loader.load_everything(group)
    uid_df = df[df['unique_id'] == uid].copy()
    
    grasynda = GrasyndaUnified(
        period=freq_int,
        n_quantiles=1,
        components_to_model=['trend', 'remainder'],
        apply_differentiation=True,
        sampling_type='discrete'
    )
    
    # 1. Decompose & Learn
    decomposed = grasynda.decompose_tsd(uid_df, freq_int, False)
    
    # Trend
    trend_prep = grasynda._differentiate_component(decomposed, 'trend')
    trend_prep['Quantile'] = grasynda._get_quantiles(trend_prep, 'diff_trend', component='trend')
    grasynda._calc_transition_matrix(trend_prep, 'trend')
    
    # Remainder
    rem_prep = decomposed.copy()
    rem_prep['Quantile'] = grasynda._get_quantiles(rem_prep, 'remainder', component='remainder')
    grasynda._calc_transition_matrix(rem_prep, 'remainder')
    
    # 2. Trace components
    synth_trend_integrated, synth_slopes = generate_component_vals('diff_trend', 1, trend_prep, grasynda, is_diff=True)
    synth_rem = generate_component_vals('remainder', 1, rem_prep, grasynda)
    
    # Reconstruction (Original Seasonal)
    synth_y = synth_trend_integrated + decomposed['seasonal'].values + synth_rem
    
    # 3. Plot
    fig = plt.figure(figsize=(15, 18))
    gs = fig.add_gridspec(4, 1, height_ratios=[1, 0.7, 0.7, 0.5])
    ax1, ax2, ax3, ax4 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2]), fig.add_subplot(gs[3])
    
    ax1.plot(uid_df['ds'], uid_df['y'], color='black', linewidth=3, label='Original')
    ax1.plot(uid_df['ds'], synth_y, color='#e67e22', label='Hybrid Grasynda (NQ=1)', alpha=0.8)
    ax1.set_title(f"Full Series Reconstruction: {uid} (NQ=1)")
    ax1.legend()
    
    # Show Raw Slopes as requested
    ax2.plot(uid_df['ds'], trend_prep['diff_trend'], color='black', alpha=0.5, label='Original Slopes (diff_trend)')
    ax2.plot(uid_df['ds'], synth_slopes, color='#3498db', label='Generated Slopes (Random Shuffle)')
    ax2.set_title("Component Trace: Trend Slopes (diff_trend)")
    ax2.legend()
    
    ax3.plot(uid_df['ds'], decomposed['remainder'], color='black', alpha=0.5, label='Original Remainder')
    ax3.plot(uid_df['ds'], synth_rem, color='#2ecc71', label='Generated Remainder (White Noise)')
    ax3.set_title("Component Trace: Remainder")
    ax3.legend()
    
    mat = grasynda.transition_mats['remainder'][uid]
    im = ax4.imshow(mat, cmap='YlGnBu')
    ax4.set_title("Transition Probability Matrix (N=1)")
    ax4.set_xticks([0]); ax4.set_yticks([0])
    plt.colorbar(im, ax=ax4, orientation='horizontal', shrink=0.5)
    
    plt.tight_layout()
    output_dir = 'assets/nq1_audit'
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/{uid}_nq1_viz.png", dpi=120)
    plt.close()

if __name__ == "__main__":
    visualize_nq1_results('M3', 'Monthly', 'M1165')
    visualize_nq1_results('Tourism', 'Monthly', 'm11')
