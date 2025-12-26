
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from utils.load_data.config import DATASETS
from src.grasynda_unified import GrasyndaUnified

# Ensure Reproducibility
random.seed(42)
np.random.seed(42)

def generate_and_plot(dataset_name, group, uids, quantiles_list):
    data_loader = DATASETS[dataset_name]
    df, horizon, n_lags, freq_str, freq_int = data_loader.load_everything(group)
    
    output_base = f"assets/multi_quantile_audit/{dataset_name}"
    os.makedirs(output_base, exist_ok=True)
    
    for uid in uids:
        uid_df = df[df['unique_id'] == uid].copy()
        print(f"Processing {dataset_name} - {uid}...")
        
        for n_q in quantiles_list:
            # 1. Setup Standard Grasynda (Remainder ONLY)
            std_grasynda = GrasyndaUnified(
                period=freq_int,
                n_quantiles=n_q,
                components_to_model=['remainder'],
                sampling_type='discrete'
            )
            
            # 2. Setup Hybrid Grasynda (Trend + Remainder)
            hyb_grasynda = GrasyndaUnified(
                period=freq_int,
                n_quantiles=n_q,
                components_to_model=['trend', 'remainder'],
                apply_differentiation=True,
                sampling_type='discrete'
            )
            
            # Decompose once (both use STL)
            decomposed = std_grasynda.decompose_tsd(uid_df, freq_int, False)
            orig_trend = decomposed['trend'].values
            orig_seasonal = decomposed['seasonal'].values
            orig_rem = decomposed['remainder'].values
            
            # Differentiate Trend for extraction
            trend_prep = hyb_grasynda._differentiate_component(decomposed, 'trend')
            orig_slopes = trend_prep['diff_trend'].values
            
            # Run models to get synth series
            # To get components, we'll manually use the logic inside transform/create_synthetic
            # but using the actual model objects to ensure "correct methodology"
            
            # --- Standard Synthesis ---
            # Remainder
            rem_df_std = decomposed.copy()
            rem_df_std['Quantile'] = std_grasynda._get_quantiles(rem_df_std, 'remainder', component='remainder')
            std_grasynda._calc_transition_matrix(rem_df_std, 'remainder')
            synth_rem_dict_std = std_grasynda._create_synthetic_ts_quantile(rem_df_std, 'remainder', 'remainder', 'discrete')
            synth_rem_std = synth_rem_dict_std[uid].values
            y_std = orig_trend + orig_seasonal + synth_rem_std
            
            # --- Hybrid Synthesis ---
            # Trend
            # Use differentiated trend
            trend_df_hyb = trend_prep.copy()
            trend_df_hyb['Quantile'] = hyb_grasynda._get_quantiles(trend_df_hyb, 'diff_trend', component='trend')
            hyb_grasynda._calc_transition_matrix(trend_df_hyb, 'trend')
            synth_slopes_dict = hyb_grasynda._create_synthetic_ts_quantile(trend_df_hyb, 'trend', 'diff_trend', 'discrete')
            synth_slopes = synth_slopes_dict[uid].values
            
            # Reconstruct Trend
            synth_trend = np.zeros(len(synth_slopes))
            synth_trend[0] = orig_trend[0]
            synth_trend[1:] = orig_trend[0] + np.cumsum(synth_slopes[1:])
            
            # Remainder
            rem_df_hyb = decomposed.copy()
            rem_df_hyb['Quantile'] = hyb_grasynda._get_quantiles(rem_df_hyb, 'remainder', component='remainder')
            hyb_grasynda._calc_transition_matrix(rem_df_hyb, 'remainder')
            synth_rem_dict_hyb = hyb_grasynda._create_synthetic_ts_quantile(rem_df_hyb, 'remainder', 'remainder', 'discrete')
            synth_rem_hyb = synth_rem_dict_hyb[uid].values
            
            y_hyb = synth_trend + orig_seasonal + synth_rem_hyb
            
            # --- Plotting 3x2 ---
            fig = plt.figure(figsize=(18, 14))
            gs = fig.add_gridspec(3, 2, height_ratios=[1, 0.8, 0.8])
            
            # Row 0: Full Series
            ax00 = fig.add_subplot(gs[0, 0])
            ax00.plot(uid_df['ds'], uid_df['y'], color='black', label='Original', alpha=0.5, linewidth=2)
            ax00.plot(uid_df['ds'], y_std, color='#3498db', label='Standard Grasynda')
            ax00.set_title(f"Standard (Default): {uid} | NQ={n_q}")
            ax00.legend()
            
            ax01 = fig.add_subplot(gs[0, 1])
            ax01.plot(uid_df['ds'], uid_df['y'], color='black', label='Original', alpha=0.5, linewidth=2)
            ax01.plot(uid_df['ds'], y_hyb, color='#e67e22', label='Hybrid Grasynda')
            ax01.set_title(f"Hybrid (T+R): {uid} | NQ={n_q}")
            ax01.legend()
            
            # Row 1: Standard Component vs Hybrid Trend Component
            ax10 = fig.add_subplot(gs[1, 0])
            ax10.plot(uid_df['ds'], orig_rem, color='black', alpha=0.3, label='Orig Rem')
            ax10.plot(uid_df['ds'], synth_rem_std, color='#2ecc71', label='Synth Rem (Standard)')
            ax10.set_title("Standard Component: Remainder")
            ax10.legend()
            
            ax11 = fig.add_subplot(gs[1, 1])
            ax11.plot(uid_df['ds'], orig_slopes, color='black', alpha=0.3, label='Orig Slopes')
            ax11.plot(uid_df['ds'], synth_slopes, color='#e74c3c', label='Synth Trend Slopes (Hybrid)')
            ax11.set_title("Hybrid Component: Trend Slopes (Before Integration)")
            ax11.legend()
            
            # Row 2: Hybrid Remainder
            ax21 = fig.add_subplot(gs[2, 1])
            ax21.plot(uid_df['ds'], orig_rem, color='black', alpha=0.3, label='Orig Rem')
            ax21.plot(uid_df['ds'], synth_rem_hyb, color='#2ecc71', label='Synth Rem (Hybrid)')
            ax21.set_title("Hybrid Component: Remainder")
            ax21.legend()
            
            ax20 = fig.add_subplot(gs[2, 0])
            ax20.axis('off')
            ax20.text(0.5, 0.5, f"Decomposition: STL\nQuantiles: {n_q}\nSampling: Discrete\n\nStandard: Orig Trend + Orig Season + Synth Rem\nHybrid: Synth Trend + Orig Season + Synth Rem", ha='center', va='center', fontsize=12)
            
            plt.tight_layout()
            save_path = f"{output_base}/{uid}_nq{n_q}.png"
            plt.savefig(save_path, dpi=100)
            plt.close()

if __name__ == "__main__":
    n_qs = [3, 10, 25, 100]
    
    # M3 Random 5
    m3_uids = ['M1165', 'M1065', 'M1200', 'M1300', 'M1400'] # Pre-picked for variety or random?
    # Actually let's just pick 5 from the dataset to be "random" as requested
    
    def get_random_uids(dataset_name, group, k=5):
        data_loader = DATASETS[dataset_name]
        df, _, _, _, _ = data_loader.load_everything(group)
        all_uids = df['unique_id'].unique().tolist()
        return random.sample(all_uids, k)

    m3_random = get_random_uids('M3', 'Monthly')
    tourism_random = get_random_uids('Tourism', 'Monthly')
    
    generate_and_plot('M3', 'Monthly', m3_random, n_qs)
    generate_and_plot('Tourism', 'Monthly', tourism_random, n_qs)
