
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Robust project root detection
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
sys.path.insert(0, project_root)

from src.grasynda_unified import GrasyndaUnified
from utils.load_data.config import DATASETS, DATA_GROUPS

OUTPUT_DIR = os.path.join(project_root, "assets", "results", "visualizations", "ensemble_scaling")

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.random.seed(42)
    
    # Use M3 Monthly for a good variety of shapes
    ds_name, grp = 'M3', 'Monthly'
    print(f"Loading {ds_name} ({grp})...")
    loader = DATASETS[ds_name]
    df_all, _, _, _, freq_int = loader.load_everything(grp)
    
    # Pick 20 random series
    all_uids = df_all['unique_id'].unique()
    sample_uids = np.random.choice(all_uids, 20, replace=False)
    df_sample = df_all[df_all['unique_id'].isin(sample_uids)].copy()
    
    # Standard Hybrid Settings
    # Using 50 quantiles and discrete sampling as "Standard" base
    common_params = {
        'period': freq_int,
        'n_quantiles': 25,
        'components_to_model': ['trend', 'remainder'],
        'component_params': {
            'trend': {'sampling_type': 'continuous_uniform', 'apply_differentiation': True},
            'remainder': {'sampling_type': 'continuous_uniform', 'apply_differentiation': False}
        }
    }
    
    print("\n--- GENERATING VARIANTS ---")
    
    print("[1/3] No Ensemble...")
    model0 = GrasyndaUnified(**common_params, ensemble_transitions=False)
    df0 = model0.transform(df_sample)
    
    print("[2/3] Ensemble Size 5...")
    model5 = GrasyndaUnified(**common_params, ensemble_transitions=True, ensemble_size=5)
    df5 = model5.transform(df_sample)
    
    print("[3/3] Ensemble Size 10...")
    model10 = GrasyndaUnified(**common_params, ensemble_transitions=True, ensemble_size=10)
    df10 = model10.transform(df_sample)
    
    # Plotting
    rows_per_page = 10
    pages = 2
    
    colors = {
        'Original': 'black',
        'No Ensemble': '#FF7F0E', # Orange
        'Ens 5': '#2196F3',       # Blue
        'Ens 10': '#D32F2F'       # Red
    }
    
    for p in range(pages):
        start = p * rows_per_page
        end = start + rows_per_page
        page_uids = sample_uids[start:end]
        
        fig, axes = plt.subplots(len(page_uids), 1, figsize=(16, 4 * len(page_uids)))
        
        for i, uid in enumerate(page_uids):
            ax = axes[i]
            real = df_sample[df_sample['unique_id'] == uid]
            x = np.arange(len(real))
            
            # Sub-sample names to sync with transform IDs
            g_uid = f"GrasyndaUnified_{uid}"
            
            # Plots
            ax.plot(x, real['y'], color=colors['Original'], label='Original', lw=3.5, alpha=0.15)
            
            # No Ensemble
            y0 = df0[df0['unique_id'] == g_uid]['y'].values
            ax.plot(x[:len(y0)], y0, color=colors['No Ensemble'], label='No Ensemble', lw=1.2, alpha=0.8)
            
            # Ens 5
            y5 = df5[df5['unique_id'] == g_uid]['y'].values
            ax.plot(x[:len(y5)], y5, color=colors['Ens 5'], label='Ensemble (Size 5)', lw=1.2, alpha=0.8)
            
            # Ens 10
            y10 = df10[df10['unique_id'] == g_uid]['y'].values
            ax.plot(x[:len(y10)], y10, color=colors['Ens 10'], label='Ensemble (Size 10)', lw=1.5)
            
            ax.set_title(f"Series: {uid} (M3 Monthly)", loc='left', fontsize=12)
            ax.legend(loc='upper right', ncol=4, frameon=True)
            ax.grid(True, alpha=0.2)
            ax.set_facecolor('#F8F9FA')

        plt.suptitle(f"Ensemble Scaling Comparison: {p+1}/2 (No Ens vs 5 vs 10)", fontsize=18, y=0.99)
        plt.tight_layout(rect=[0, 0, 1, 0.98])
        
        out_path = os.path.join(OUTPUT_DIR, f"scaling_comparison_page_{p+1}.png")
        plt.savefig(out_path, dpi=180)
        plt.close()
        print(f"Saved: {out_path}")

    print("\nVisual verification complete.")

if __name__ == "__main__":
    main()
