
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.load_data.config import DATASETS
from src.grasynda_unified import GrasyndaUnified

# Ensure Reproducibility
random.seed(42)
np.random.seed(42)

def get_random_uids(dataset_name, group, k=10):
    data_loader = DATASETS[dataset_name]
    df, _, _, _, _ = data_loader.load_everything(group)
    all_uids = df['unique_id'].unique().tolist()
    return random.sample(all_uids, min(k, len(all_uids)))

def plot_triple_comparison_with_components(dataset_name, group, uids, n_q=25):
    print(f"Generating Triple Comparisons for {dataset_name} ({group}) (NQ={n_q})...")
    data_loader = DATASETS[dataset_name]
    df, _, _, _, freq_int = data_loader.load_everything(group)
    
    clean_group = group.replace(":", "_").replace("/", "_")
    output_base = f"assets/sampling_audit_components/{dataset_name}_{clean_group}"
    os.makedirs(output_base, exist_ok=True)
    
    for uid in uids:
        print(f"  Processing {uid}...")
        uid_df = df[df['unique_id'] == uid].copy()
        
        # 1. Discrete Hybrid
        discrete_model = GrasyndaUnified(
            period=freq_int,
            n_quantiles=n_q,
            components_to_model=['trend', 'remainder'],
            component_params={
                'trend': {'apply_differentiation': True, 'sampling_type': 'discrete'},
                'remainder': {'apply_differentiation': False, 'sampling_type': 'discrete'}
            }
        )
        
        # 2. Continuous Uniform Hybrid
        uniform_model = GrasyndaUnified(
            period=freq_int,
            n_quantiles=n_q,
            components_to_model=['trend', 'remainder'],
            component_params={
                'trend': {'apply_differentiation': True, 'sampling_type': 'continuous_uniform'},
                'remainder': {'apply_differentiation': False, 'sampling_type': 'continuous_uniform'}
            }
        )
        
        # 3. Visibility Horizontal Hybrid
        vis_model = GrasyndaUnified(
            period=freq_int,
            components_to_model=['trend', 'remainder'],
            component_params={
                'trend': {'graph_type': 'visibility', 'visibility_type': 'horizontal', 'apply_differentiation': True, 'sampling_type': 'discrete'},
                'remainder': {'graph_type': 'visibility', 'visibility_type': 'horizontal', 'apply_differentiation': False, 'sampling_type': 'discrete'}
            }
        )
        
        try:
            # Transform and return components
            s_d_y, c_d = discrete_model.transform(uid_df, return_components=True)
            s_u_y, c_u = uniform_model.transform(uid_df, return_components=True)
            s_v_y, c_v = vis_model.transform(uid_df, return_components=True)
            
            # Decompose original for component ground truth
            orig_decomposed = GrasyndaUnified.decompose_tsd(uid_df, freq_int, False)
            orig_trend_diff = orig_decomposed['trend'].diff().fillna(0)
            orig_rem = orig_decomposed['remainder']
            
        except Exception as e:
            print(f"    Error transforming {uid}: {e}")
            import traceback
            traceback.print_exc()
            continue
            
        # --- Plotting ---
        fig, axes = plt.subplots(3, 3, figsize=(24, 18), sharex=True)
        
        models_data = [
            ('Discrete', '#3498db', s_d_y, c_d),
            ('Uniform', '#e74c3c', s_u_y, c_u),
            ('Visibility', '#2ecc71', s_v_y, c_v)
        ]
        
        for col, (title, color, s_y, components) in enumerate(models_data):
            # Row 0: Full Reconstruction
            axes[0, col].plot(uid_df['ds'], uid_df['y'], color='black', label='Original', alpha=0.35, linewidth=2.5)
            axes[0, col].plot(uid_df['ds'], s_y['y'], color=color, label=f'Synthetic {title}', linewidth=1.2)
            axes[0, col].set_title(f"RECONSTRUCTION: {title}")
            axes[0, col].legend(loc='upper left', fontsize='small')
            
            # Row 1: Trend Component (Slopes)
            s_trend = components['trend'][uid]
            s_slopes = s_trend.diff().fillna(0)
            
            axes[1, col].plot(uid_df['ds'], orig_trend_diff, color='black', alpha=0.35, label='Orig Slopes')
            axes[1, col].plot(uid_df['ds'], s_slopes, color=color, label='Synth Slopes', linewidth=1)
            axes[1, col].set_title(f"COMPONENT: Trend Slopes ({title})")
            axes[1, col].legend(loc='upper left', fontsize='small')
            
            # Row 2: Remainder
            s_rem = components['remainder'][uid]
            axes[2, col].plot(uid_df['ds'], orig_rem, color='black', alpha=0.35, label='Orig Rem')
            axes[2, col].plot(uid_df['ds'], s_rem, color=color, label='Synth Rem', linewidth=1)
            axes[2, col].set_title(f"COMPONENT: Remainder ({title})")
            axes[2, col].legend(loc='upper left', fontsize='small')
            
        for ax in axes.flatten():
            ax.grid(alpha=0.2)

        plt.suptitle(f"Grasynda Triple Component Audit | {dataset_name} - {uid} | Hybrid (NQ={n_q})", fontsize=20)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        save_path = f"{output_base}/{uid}_audit_components.png"
        plt.savefig(save_path, dpi=100)
        plt.close()
        print(f"    Saved: {uid}")

if __name__ == "__main__":
    n_q = 25
    configs = [
        ('M3', 'Monthly'),
        ('Tourism', 'Monthly'),
        ('Gluonts', 'm1_monthly')
    ]
    
    for ds, group in configs:
        uids = get_random_uids(ds, group, k=4)
        plot_triple_comparison_with_components(ds, group, uids, n_q=n_q)
