
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
from utils.load_data.config import DATASETS
from src.grasynda_unified import GrasyndaUnified

def analyze_and_visualize_dominance(dataset_name='M3', group='Monthly', n_samples=200):
    print(f"Analyzing {dataset_name} Dominance (checking {n_samples} samples)...")
    
    data_loader = DATASETS[dataset_name]
    df, horizon, n_lags, freq_str, freq_int = data_loader.load_everything(group)
    
    # Random sample for analysis
    all_uids = df['unique_id'].unique()
    if len(all_uids) > n_samples:
        np.random.seed(42)  # Fixed seed for reproducibility
        sample_uids = np.random.choice(all_uids, n_samples, replace=False)
    else:
        sample_uids = all_uids
    
    # 1. Identify Dominant Series
    dominant_map = {'Seasonal': [], 'Trend': [], 'Remainder': []}
    
    for uid in sample_uids:
        series = df[df['unique_id'] == uid]['y'].values
        if len(series) < 2 * freq_int: continue
        
        # Quick STL
        try:
            res = STL(series, period=freq_int, robust=False).fit()
            sd_y = np.std(series)
            if sd_y == 0: continue
            
            sd_t = np.std(res.trend)
            sd_s = np.std(res.seasonal)
            sd_r = np.std(res.resid)
            
            rel_t = sd_t / sd_y
            rel_s = sd_s / sd_y
            rel_r = sd_r / sd_y
            
            # Simple max rule
            if rel_s >= rel_t and rel_s >= rel_r:
                dominant_map['Seasonal'].append((uid, rel_s))
            elif rel_t >= rel_s and rel_t >= rel_r:
                dominant_map['Trend'].append((uid, rel_t))
            else:
                dominant_map['Remainder'].append((uid, rel_r))
        except:
            continue
            
    # Select best candidates (highest relative std)
    candidates = {}
    for dom_type in dominant_map:
        dominant_map[dom_type].sort(key=lambda x: x[1], reverse=True)
        # Take up to 5 top candidates
        candidates[dom_type] = [x[0] for x in dominant_map[dom_type][:5]]
        print(f"Top {dom_type} dominant series: {candidates[dom_type]}")
        
    uids_to_plot = []
    for ulist in candidates.values():
        uids_to_plot.extend(ulist)
        
    df_plot = df[df['unique_id'].isin(uids_to_plot)].copy()
    
    # 2. Configure Models
    common_params = {
        'period': freq_int,
        'n_quantiles': 25,
        'components_to_model': ['trend', 'remainder'],
        'component_params': {
            'trend': {
                'sampling_type': 'continuous_uniform',
                'apply_differentiation': True
            },
            'remainder': {
                'sampling_type': 'discrete'
            }
        },
        'ensemble_transitions': False
    }
    
    # Model A: Hybrid Quantile
    print("Generating Hybrid Quantile...")
    model_quant = GrasyndaUnified(**common_params, graph_type='quantile')
    synth_quant, comps_quant = model_quant.transform(df_plot, return_components=True)
    
    # Model B: Hybrid Visibility
    print("Generating Hybrid Visibility...")
    # For Visibility, differentiation defaults to specific graph type logic if not careful
    # We use global valid defaults for visibility
    model_vis = GrasyndaUnified(**common_params, graph_type='visibility', visibility_type='horizontal')
    synth_vis, comps_vis = model_vis.transform(df_plot, return_components=True)
    
    # Get original trend for comparison (decompose_tsd)
    decomposed_orig = model_quant.decompose_tsd(df_plot, freq_int, False)
    
    # 3. Plotting
    output_dir = 'assets/hybrid_dominance_audit_fixed'
    os.makedirs(output_dir, exist_ok=True)
    
    for dom_type, uids in candidates.items():
        for uid in uids:
            print(f"Plotting {uid} ({dom_type} Dominant)...")
            
            orig = df_plot[df_plot['unique_id'] == uid]
            sq = synth_quant[synth_quant['unique_id'] == f'GrasyndaUnified_{uid}']
            sv = synth_vis[synth_vis['unique_id'] == f'GrasyndaUnified_{uid}']
            
            if sq.empty or sv.empty:
                print(f"Skipping {uid} due to generation error.")
                continue
                
            fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
            
            # Plot 1: Full Series
            axes[0].plot(orig['ds'], orig['y'], color='black', label='Original', lw=2, alpha=0.7)
            axes[0].plot(sq['ds'], sq['y'], color='#3498db', label='Hybrid Quantile', lw=1.5, alpha=0.8)
            axes[0].plot(sv['ds'], sv['y'], color='#e74c3c', label='Hybrid Visibility', lw=1.5, alpha=0.8)
            axes[0].set_title(f"Comparison: {uid} | Dominance: {dom_type}\nTrend=Uniform | Remainder=Discrete")
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # Plot 2: Trend Component
            orig_trend = decomposed_orig[decomposed_orig['unique_id'] == uid]['trend']
            
            # Extract synthetic trends from component dicts
            # Note: The keys in comps_quant['trend'] are original UIDs
            t_quant = comps_quant['trend'][uid]
            t_vis = comps_vis['trend'][uid]
            
            axes[1].plot(orig['ds'], orig_trend, color='black', label='Original Trend', lw=2, alpha=0.6, linestyle='--')
            axes[1].plot(orig['ds'], t_quant.values, color='#3498db', label='Syn Trend (Quantile)', lw=2, alpha=0.9)
            axes[1].plot(orig['ds'], t_vis.values, color='#e74c3c', label='Syn Trend (VisGraph)', lw=2, alpha=0.9)
            axes[1].set_title(f"Trend Component: {uid}")
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            save_path = f"{output_dir}/{dom_type}_{uid}.png"
            plt.savefig(save_path, dpi=120)
            plt.close()
            print(f"Saved to {save_path}")

if __name__ == "__main__":
    analyze_and_visualize_dominance()
