
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.load_data.config import DATASETS
from src.grasynda_unified import GrasyndaUnified

def sample_nq1(vals):
    """Simple shuffle for N=1."""
    return np.random.choice(vals, size=len(vals))

def compare_nq1_models(dataset_name='M3', group='Monthly', uid='M1165'):
    print(f"Comparing NQ=1 for {uid}...")
    
    data_loader = DATASETS[dataset_name]
    df, horizon, n_lags, freq_str, freq_int = data_loader.load_everything(group)
    uid_df = df[df['unique_id'] == uid].copy()
    
    # Standard Grasynda (NQ=1)
    # Hybrid Grasynda (NQ=1)
    
    # 1. Decomposition
    model_obj = GrasyndaUnified(period=freq_int, n_quantiles=1)
    decomposed = model_obj.decompose_tsd(uid_df, freq_int, False)
    
    # Get original components
    orig_trend = decomposed['trend'].values
    orig_seasonal = decomposed['seasonal'].values
    orig_rem = decomposed['remainder'].values
    
    # Trend Slopes
    trend_prep = model_obj._differentiate_component(decomposed, 'trend')
    orig_slopes = trend_prep['diff_trend'].values
    
    # Generate for Standard (Default)
    synth_rem_std = sample_nq1(orig_rem)
    y_std = orig_trend + orig_seasonal + synth_rem_std
    
    # Generate for Hybrid
    synth_slopes = sample_nq1(orig_slopes) # This is "trend before integration"
    # Integrate (starting from original start)
    synth_trend = np.zeros(len(synth_slopes))
    synth_trend[0] = orig_trend[0]
    synth_trend[1:] = orig_trend[0] + np.cumsum(synth_slopes[1:])
    
    synth_rem_hyb = sample_nq1(orig_rem)
    y_hyb = synth_trend + orig_seasonal + synth_rem_hyb
    
    # --- Plotting ---
    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(3, 2)
    
    # Column 0: Standard (Default)
    ax_std_y = fig.add_subplot(gs[0, 0])
    ax_std_y.plot(uid_df['ds'], uid_df['y'], color='black', label='Original', alpha=0.5)
    ax_std_y.plot(uid_df['ds'], y_std, color='#3498db', label='Standard Grasynda (NQ=1)')
    ax_std_y.set_title(f"Standard (Default): {uid}")
    ax_std_y.legend()

    ax_std_rem = fig.add_subplot(gs[1, 0])
    ax_std_rem.plot(uid_df['ds'], orig_rem, color='black', alpha=0.3, label='Orig Remainder')
    ax_std_rem.plot(uid_df['ds'], synth_rem_std, color='#2ecc71', label='Gen Remainder (White Noise)')
    ax_std_rem.set_title("Standard Generated component: Remainder")
    ax_std_rem.legend()
    
    ax_std_msg = fig.add_subplot(gs[2, 0])
    ax_std_msg.text(0.5, 0.5, "Standard Grasynda\nTrend is FIXED (original)", ha='center', va='center', fontsize=14)
    ax_std_msg.axis('off')

    # Column 1: Hybrid
    ax_hyb_y = fig.add_subplot(gs[0, 1])
    ax_hyb_y.plot(uid_df['ds'], uid_df['y'], color='black', label='Original', alpha=0.5)
    ax_hyb_y.plot(uid_df['ds'], y_hyb, color='#e67e22', label='Hybrid Grasynda (NQ=1)')
    ax_hyb_y.set_title(f"Hybrid: {uid}")
    ax_hyb_y.legend()

    ax_hyb_slope = fig.add_subplot(gs[1, 1])
    ax_hyb_slope.plot(uid_df['ds'], orig_slopes, color='black', alpha=0.3, label='Orig Slopes')
    ax_hyb_slope.plot(uid_df['ds'], synth_slopes, color='#e74c3c', label='Gen Slopes (Random Walk engine)')
    ax_hyb_slope.set_title("Hybrid Generated component: Trend Slopes")
    ax_hyb_slope.legend()

    ax_hyb_rem = fig.add_subplot(gs[2, 1])
    ax_hyb_rem.plot(uid_df['ds'], orig_rem, color='black', alpha=0.3, label='Orig Remainder')
    ax_hyb_rem.plot(uid_df['ds'], synth_rem_hyb, color='#2ecc71', label='Gen Remainder (White Noise)')
    ax_hyb_rem.set_title("Hybrid Generated component: Remainder")
    ax_hyb_rem.legend()

    plt.tight_layout()
    output_path = f"assets/nq1_audit/comparison_{uid}.png"
    plt.savefig(output_path, dpi=120)
    plt.close()
    print(f"Saved comparison to {output_path}")

if __name__ == "__main__":
    os.makedirs('assets/nq1_audit', exist_ok=True)
    compare_nq1_models('M3', 'Monthly', 'M1165')
    compare_nq1_models('Tourism', 'Monthly', 'm11')
