"""
Core Experiment: Standard vs Hybrid Grasynda Comparison
Focus: Default stochastic sampling, varying quantiles, Low vs High Seasonality.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict
from utils.load_data.config import DATASETS
from src.grasynda_unified import GrasyndaUnified

# Series from pick_extreme_series.py
SERIES_MAP = {
    'M3': {
        'Low Seasonality': ['M1165', 'M1169'],
        'High Seasonality': ['M1065', 'M1066']
    },
    'Tourism': {
        'Low Seasonality': ['m11', 'm9'],
        'High Seasonality': ['m197', 'm1']
    }
}

def generate_synth_values(model: GrasyndaUnified, df: pd.DataFrame, uid: str, component: str) -> np.ndarray:
    """Helper to generate values using the model's default stochastic path."""
    uid_df = df[df['unique_id'] == uid]
    # 1. Generate stochastic quantile path
    q_dict = model._generate_quantile_series(df, component)
    q_path = q_dict[uid]
    
    # 2. Map to values (discrete sampling)
    target_col = f'diff_{component}' if f'diff_{component}' in df.columns else component
    vals_all = df[target_col].values
    quants_all = model._get_quantiles(df, target_col)
    
    bin_vals = {q: vals_all[quants_all == q] for q in range(model.n_quantiles)}
    
    synth_comp = np.zeros(len(uid_df))
    synth_comp[0] = uid_df[target_col].values[0]
    for i in range(1, len(synth_comp)):
        q = q_path[i]
        choices = bin_vals.get(q)
        if choices is not None and len(choices) > 0:
            synth_comp[i] = np.random.choice(choices)
        else:
            synth_comp[i] = synth_comp[i-1]
            
    # Integrate if needed
    if target_col.startswith('diff_'):
        start_orig = uid_df[component].values[0]
        integrated = np.zeros(len(synth_comp))
        integrated[0] = start_orig
        integrated[1:] = start_orig + np.cumsum(synth_comp[1:])
        return integrated
    return synth_comp

def run_core_comparison(dataset_name='M3', group='Monthly'):
    print(f"\n--- {dataset_name} {group} ---")
    data_loader = DATASETS[dataset_name]
    df, horizon, n_lags, freq_str, freq_int = data_loader.load_everything(group)
    
    all_uids = SERIES_MAP[dataset_name]['Low Seasonality'] + SERIES_MAP[dataset_name]['High Seasonality']
    df_sample = df[df['unique_id'].isin(all_uids)].copy()
    
    quantiles = [25, 3, 1]
    
    for n_q in quantiles:
        print(f"  Quantiles: {n_q}")
        output_dir = f'assets/core_comparison/{dataset_name}_{group}/nq{n_q}'
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Standard Grasynda (Remainder synth, Trend original)
        std_model = GrasyndaUnified(period=freq_int, n_quantiles=n_q, components_to_model=['remainder'])
        df_std = std_model.decompose_tsd(df_sample, freq_int, False)
        # Learn components
        df_std['Quantile'] = std_model._get_quantiles(df_std, 'remainder')
        std_model._calc_transition_matrix(df_std, 'remainder')
        
        # 2. Hybrid Grasynda (Remainder synth, Trend synth)
        hyb_model = GrasyndaUnified(period=freq_int, n_quantiles=n_q, components_to_model=['trend', 'remainder'], apply_differentiation=True)
        df_hyb = hyb_model.decompose_tsd(df_sample, freq_int, False)
        # Learn Trend
        tr_df = hyb_model._differentiate_component(df_hyb, 'trend')
        tr_df['Quantile'] = hyb_model._get_quantiles(tr_df, 'diff_trend')
        hyb_model._calc_transition_matrix(tr_df, 'trend')
        # Learn Remainder
        df_hyb['Quantile'] = hyb_model._get_quantiles(df_hyb, 'remainder')
        hyb_model._calc_transition_matrix(df_hyb, 'remainder')
        
        for uid in all_uids:
            cat = 'Low' if uid in SERIES_MAP[dataset_name]['Low Seasonality'] else 'High'
            orig_uid_df = df_hyb[df_hyb['unique_id'] == uid]
            
            # Generate Standard
            v_rem_std = generate_synth_values(std_model, df_std, uid, 'remainder')
            y_std = orig_uid_df['trend'].values + orig_uid_df['seasonal'].values + v_rem_std
            
            # Generate Hybrid
            v_trend_hyb = generate_synth_values(hyb_model, tr_df, uid, 'trend')
            v_rem_hyb = generate_synth_values(hyb_model, df_hyb, uid, 'remainder')
            y_hyb = v_trend_hyb + orig_uid_df['seasonal'].values + v_rem_hyb
            
            # Plot
            plt.figure(figsize=(15, 6))
            plt.plot(orig_uid_df['ds'], orig_uid_df['y'], color='black', linewidth=3, label='Original')
            plt.plot(orig_uid_df['ds'], y_std, label='Standard Grasynda (Trend Fixed)', alpha=0.7, color='#3498db')
            plt.plot(orig_uid_df['ds'], y_hyb, label='Hybrid Grasynda (Trend Synth)', alpha=0.7, color='#e67e22')
            
            plt.title(f"{cat} Seasonality: {uid} | Q={n_q} | Comparison", fontsize=14, fontweight='bold')
            plt.legend(loc='best')
            plt.grid(alpha=0.3)
            plt.savefig(f"{output_dir}/{uid}_comparison.png", dpi=120)
            plt.close()

if __name__ == "__main__":
    run_core_comparison('M3')
    run_core_comparison('Tourism')
