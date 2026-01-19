
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

MODELS = {
    'Hybrid (No Ens)': 'G_NoEns_Hybrid',
    'Hybrid Ensemble (Q12)': 'G_Q12_Uniform',
    'Hybrid Visibility (Q25)': 'G_Vis_Q25_Uniform'
}

OUTPUT_DIR = os.path.join(project_root, "assets", "results", "visualizations", "focused_comparisons")

def generate_grasyndas(df_real, freq_int, methods):
    model_dict = {}
    
    # 1. Hybrid (No Ensemble)
    if 'Hybrid (No Ens)' in methods:
        model = GrasyndaUnified(period=freq_int, components_to_model=['trend', 'remainder'],
                                     ensemble_size=1, 
                                     component_params={
                                         'trend': {'sampling_type': 'discrete', 'apply_differentiation': True},
                                         'remainder': {'sampling_type': 'discrete'}
                                     })
        model_dict['Hybrid (No Ens)'] = model.transform(df_real)

    # 2. Hybrid Ensemble (Q12 Uniform)
    if 'Hybrid Ensemble (Q12)' in methods:
        model = GrasyndaUnified(period=freq_int, n_quantiles=12,
                                   components_to_model=['trend', 'remainder'],
                                   ensemble_size=50, ensemble_transitions=True,
                                   component_params={
                                       'trend': {'sampling_type': 'continuous_uniform', 'apply_differentiation': True},
                                       'remainder': {'sampling_type': 'continuous_uniform'}
                                   })
        model_dict['Hybrid Ensemble (Q12)'] = model.transform(df_real)

    # 3. Hybrid Visibility (Q25)
    if 'Hybrid Visibility (Q25)' in methods:
        model = GrasyndaUnified(period=freq_int, n_quantiles=25,
                                   graph_type='visibility', visibility_type='horizontal',
                                   components_to_model=['trend', 'remainder'],
                                   ensemble_size=50, ensemble_transitions=True,
                                   component_params={
                                       'trend': {'sampling_type': 'continuous_uniform', 'apply_differentiation': True},
                                       'remainder': {'sampling_type': 'continuous_uniform'}
                                   })
        model_dict['Hybrid Visibility (Q25)'] = model.transform(df_real)
    
    return model_dict

def plot_focused_comparison(dataset_name, group, uids, df_real, models):
    fig, axes = plt.subplots(len(uids), 1, figsize=(15, 3 * len(uids)))
    if len(uids) == 1: axes = [axes]
    
    colors = {
        'Original': 'black',
        'Hybrid (No Ens)': '#FF7F0E',       # Orange
        'Hybrid Ensemble (Q12)': '#D62728', # Red
        'Hybrid Visibility (Q25)': '#1F77B4' # Blue
    }

    for i, uid in enumerate(uids):
        ax = axes[i]
        real_series = df_real[df_real['unique_id'] == uid]
        x_vals = np.arange(len(real_series))
        
        # Original
        ax.plot(x_vals, real_series['y'], color=colors['Original'], label='Original', linewidth=2.5, alpha=0.5)
        
        # Variants
        for name in MODELS.keys():
            df_g = models.get(name)
            if df_g is not None:
                g_series = df_g[df_g['unique_id'] == f"GrasyndaUnified_{uid}"]
                if not g_series.empty:
                    ax.plot(x_vals[:len(g_series)], g_series['y'], color=colors[name], label=name, linewidth=1.2)
        
        ax.set_title(f"Series: {uid}", loc='left', fontsize=10)
        ax.legend(loc='upper right', fontsize='small', ncol=4)
        ax.grid(True, alpha=0.2)
        ax.set_facecolor('#F9F9F9')

    plt.suptitle(f"Grasynda Evolution: {dataset_name} ({group})", fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    
    fname = os.path.join(OUTPUT_DIR, f"{dataset_name}_{group}_evolution.png")
    plt.savefig(fname, dpi=200)
    plt.close()
    print(f"  Saved: {fname}")

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.random.seed(42)
    
    # Just run on 2 representative groups to be fast and high-quality
    TARGET_GROUPS = [('Gluonts', 'm1_monthly'), ('M3', 'Monthly')]
    
    for ds_name, grp in TARGET_GROUPS:
        print(f"Generating evolution plots for {ds_name} ({grp})...")
        try:
            loader = DATASETS[ds_name]
            df_all, _, _, _, freq_int = loader.load_everything(grp)
            
            all_uids = df_all['unique_id'].unique()
            sample_uids = np.random.choice(all_uids, min(5, len(all_uids)), replace=False)
            df_sample = df_all[df_all['unique_id'].isin(sample_uids)].copy()
            
            models = generate_grasyndas(df_sample, freq_int, MODELS.keys())
            plot_focused_comparison(ds_name, grp, sample_uids, df_sample, models)
            
        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
