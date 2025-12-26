
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.load_data.config import DATASETS
from src.grasynda_unified import GrasyndaUnified

def audit_nq1_behavior(dataset_name='M3', group='Monthly', uid='M1165'):
    print(f"\n{'='*60}")
    print(f"AUDIT 1-QUANTILE: {uid}")
    print(f"{'='*60}")
    
    data_loader = DATASETS[dataset_name]
    df, horizon, n_lags, freq_str, freq_int = data_loader.load_everything(group)
    uid_df = df[df['unique_id'] == uid].copy()
    
    # 1. Initialize Grasynda with 1 Quantile
    grasynda = GrasyndaUnified(
        period=freq_int,
        n_quantiles=1,
        components_to_model=['trend', 'remainder'],
        apply_differentiation=True,
        sampling_type='discrete'
    )
    
    # Decompose
    decomposed = grasynda.decompose_tsd(uid_df, freq_int, False)
    
    # Learn Trend (diff)
    trend_df = grasynda._differentiate_component(decomposed, 'trend')
    target_trend = 'diff_trend'
    trend_df['Quantile'] = grasynda._get_quantiles(trend_df, target_trend, component='trend')
    grasynda._calc_transition_matrix(trend_df, 'trend')
    
    # Learn Remainder
    decomposed['Quantile'] = grasynda._get_quantiles(decomposed, 'remainder', component='remainder')
    grasynda._calc_transition_matrix(decomposed, 'remainder')
    
    # 2. Generate
    # We'll use the core logic directly to see what's happening
    length = len(uid_df)
    
    # Quantile paths for NQ=1 must be all 0s
    q_trend = np.zeros(length, dtype=int)
    q_rem = np.zeros(length, dtype=int)
    
    # Value mapping
    def get_synth(orig_vals, quantiles, target_q_path):
        # With NQ=1, quantiles is all 0s. 
        # bin_vals should contain ALL original values in bin 0.
        bin_0_vals = orig_vals[quantiles == 0]
        print(f"  Component Stats: Min={orig_vals.min():.2f}, Max={orig_vals.max():.2f}, Count={len(orig_vals)}")
        print(f"  Bin 0 Size: {len(bin_0_vals)}")
        
        synth = np.zeros(len(target_q_path))
        synth[0] = orig_vals[0]
        for i in range(1, len(synth)):
            synth[i] = np.random.choice(bin_0_vals)
        return synth

    print("\nTrend (diff_trend) Synthesis:")
    v_trend_diff = get_synth(trend_df[target_trend].values, trend_df['Quantile'].values, q_trend)
    # Integrate
    start_trend = decomposed['trend'].values[0]
    v_trend_integrated = np.zeros(length)
    v_trend_integrated[0] = start_trend
    v_trend_integrated[1:] = start_trend + np.cumsum(v_trend_diff[1:])
    
    print("\nRemainder Synthesis:")
    v_rem = get_synth(decomposed['remainder'].values, decomposed['Quantile'].values, q_rem)
    
    # 3. Viz
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    
    axes[0].plot(uid_df.index, decomposed['trend'], label='Original Trend', color='black', alpha=0.5)
    axes[0].plot(uid_df.index, v_trend_integrated, label='Synth Trend (NQ=1)', color='orange')
    axes[0].set_title("Trend Comparison")
    axes[0].legend()
    
    axes[1].plot(uid_df.index, decomposed['remainder'], label='Original Remainder', color='black', alpha=0.5)
    axes[1].plot(uid_df.index, v_rem, label='Synth Remainder (NQ=1)', color='green')
    axes[1].set_title("Remainder Comparison")
    axes[1].legend()
    
    synth_y = v_trend_integrated + decomposed['seasonal'].values + v_rem
    axes[2].plot(uid_df.index, uid_df['y'], label='Original Y', color='black', linewidth=2)
    axes[2].plot(uid_df.index, synth_y, label='Triple Hybrid (NQ=1)', color='red', alpha=0.7)
    axes[2].set_title("Full Series Reconstruction")
    axes[2].legend()
    
    plt.tight_layout()
    os.makedirs('scripts/debug_utils/audit_plots', exist_ok=True)
    plt.savefig('scripts/debug_utils/audit_plots/audit_nq1.png')
    
    print(f"\nAudit complete. Plot saved to scripts/debug_utils/audit_plots/audit_nq1.png")

if __name__ == "__main__":
    audit_nq1_behavior()
