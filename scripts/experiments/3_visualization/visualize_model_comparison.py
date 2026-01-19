
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL

# Robust project root detection
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
sys.path.insert(0, project_root)

from src.grasynda_unified import GrasyndaUnified
from utils.load_data.config import DATASETS, DATA_GROUPS

MODELS = {
    'G_Standard': 'G_Standard',
    'G_NoEns_Hybrid': 'G_NoEns_Hybrid',
    'G_Ens100_Mixed_Hybrid': 'G_Ens100_Mixed_Hybrid',
    'G_Q12_Uniform': 'G_Q12_Uniform',
    'G_Q8_Uniform': 'G_Q8_Uniform',
    'G_Vis_Q25_Uniform': 'G_Vis_Q25_Uniform',
    'TSMixup': 'TSMixup',
    'DBA': 'DBA',
    'MagnitudeWarping': 'MagnitudeWarping'
}

BASE_DIR = os.path.join(project_root, "assets", "results", "training_sets")
OUTPUT_DIR = os.path.join(project_root, "assets", "results", "visualizations", "model_comparisons")

def load_baseline(dataset_name, group, method):
    fpath = os.path.join(BASE_DIR, f"{dataset_name}_{group}_{method}.csv")
    if os.path.exists(fpath):
        return pd.read_csv(fpath)
    return None

def get_baseline_id_map(dataset_name, group, method, df_sample, df_full_real):
    """
    Create a mapping from original UID to baseline synthetic UID.
    """
    df_b = load_baseline(dataset_name, group, method)
    if df_b is None: return {}, None
    
    sample_uids = sorted(df_sample['unique_id'].unique())
    all_orig_uids = sorted(df_full_real['unique_id'].unique())
    
    patterns = {
        'MagnitudeWarping': '_MWARP',
        'TSMixup': 'TSMixup_',
        'DBA': 'DBA_'
    }
    p = patterns.get(method)
    
    id_map = {}
    all_b_uids = df_b['unique_id'].unique()
    
    if method == 'MagnitudeWarping':
        for u in sample_uids:
            prefix = f"{u}{p}"
            matches = [str(x) for x in all_b_uids if str(x).startswith(prefix)]
            if matches: id_map[u] = matches[0]
    
    if not id_map:
        # Index-based mapping using global positions
        b_synth_uids = sorted([str(x) for x in all_b_uids if str(x).startswith(p)])
        
        for u in sample_uids:
            if u in all_orig_uids:
                idx = all_orig_uids.index(u)
                if idx < len(b_synth_uids):
                    id_map[u] = b_synth_uids[idx]
                 
    return id_map, df_b

def generate_grasyndas(df_real, freq_int, methods):
    model_dict = {}
    
    # 1. Standard
    if 'G_Standard' in methods:
        model_std = GrasyndaUnified(period=freq_int, components_to_model=['remainder'])
        model_dict['G_Standard'] = model_std.transform(df_real)

    # 2. Hybrid (No Ensemble)
    if 'G_NoEns_Hybrid' in methods:
        model_noens = GrasyndaUnified(period=freq_int, components_to_model=['trend', 'remainder'],
                                     ensemble_size=1, # No ensemble
                                     component_params={
                                         'trend': {'sampling_type': 'discrete', 'apply_differentiation': True},
                                         'remainder': {'sampling_type': 'discrete'}
                                     })
        model_dict['G_NoEns_Hybrid'] = model_noens.transform(df_real)

    # 3. Mixed Hybrid (Ens 100)
    if 'G_Ens100_Mixed_Hybrid' in methods:
        model_mixed = GrasyndaUnified(period=freq_int, components_to_model=['trend', 'remainder'],
                                      ensemble_size=100, ensemble_transitions=True,
                                      component_params={
                                          'trend': {'sampling_type': 'continuous_uniform', 'apply_differentiation': True},
                                          'remainder': {'sampling_type': 'discrete'}
                                      })
        model_dict['G_Ens100_Mixed_Hybrid'] = model_mixed.transform(df_real)

    # 4. Q12 Uniform (Authenticity Peak)
    if 'G_Q12_Uniform' in methods:
        model_q12 = GrasyndaUnified(period=freq_int, n_quantiles=12,
                                   components_to_model=['trend', 'remainder'],
                                   ensemble_size=50, ensemble_transitions=True,
                                   component_params={
                                       'trend': {'sampling_type': 'continuous_uniform', 'apply_differentiation': True},
                                       'remainder': {'sampling_type': 'continuous_uniform'}
                                   })
        model_dict['G_Q12_Uniform'] = model_q12.transform(df_real)

    # 5. Q8 Uniform
    if 'G_Q8_Uniform' in methods:
        model_q8 = GrasyndaUnified(period=freq_int, n_quantiles=8,
                                   components_to_model=['trend', 'remainder'],
                                   ensemble_size=50, ensemble_transitions=True,
                                   component_params={
                                       'trend': {'sampling_type': 'continuous_uniform', 'apply_differentiation': True},
                                       'remainder': {'sampling_type': 'continuous_uniform'}
                                   })
        model_dict['G_Q8_Uniform'] = model_q8.transform(df_real)

    # 6. Visibility Q25
    if 'G_Vis_Q25_Uniform' in methods:
        model_vis = GrasyndaUnified(period=freq_int, n_quantiles=25,
                                   graph_type='visibility', visibility_type='horizontal',
                                   components_to_model=['trend', 'remainder'],
                                   ensemble_size=50, ensemble_transitions=True,
                                   component_params={
                                       'trend': {'sampling_type': 'continuous_uniform', 'apply_differentiation': True},
                                       'remainder': {'sampling_type': 'continuous_uniform'}
                                   })
        model_dict['G_Vis_Q25_Uniform'] = model_vis.transform(df_real)
    
    return model_dict

def plot_comparison(dataset_name, group, uids, df_real, grasynda_models, baseline_dfs, baseline_id_maps):
    # Scale: 1 row per series, containing original + 6 models
    fig, axes = plt.subplots(len(uids), 1, figsize=(15, 3 * len(uids)), sharex=False)
    if len(uids) == 1: axes = [axes]
    
    colors = {
        'Original': 'black',
        'G_Q12_Uniform': '#D62728',       # Red (Peak Auth)
        'G_Q8_Uniform': '#E377C2',         # Pink
        'G_Vis_Q25_Uniform': '#1F77B4',     # Blue
        'G_NoEns_Hybrid': '#FFBB78',       # Light Orange
        'G_Ens100_Mixed_Hybrid': '#AEC7E8', # Light Blue
        'G_Standard': '#2CA02C',           # Green
        'TSMixup': '#9467BD',              # Purple
        'DBA': '#8C564B',                  # Brown
        'MagnitudeWarping': '#FF7F0E'      # Orange
    }

    for i, uid in enumerate(uids):
        try:
            ax = axes[i]
            real_series = df_real[df_real['unique_id'] == uid]
            x_vals = np.arange(len(real_series))
            ax.plot(x_vals, real_series['y'], color=colors['Original'], label='Original', linewidth=2, alpha=0.6)
            
            # Grasyndas
            g_names = ['G_Standard', 'G_NoEns_Hybrid', 'G_Ens100_Mixed_Hybrid', 
                       'G_Q12_Uniform', 'G_Q8_Uniform', 'G_Vis_Q25_Uniform']
            for name in g_names:
                df_g = grasynda_models.get(name)
                if df_g is not None:
                    g_uid = f"GrasyndaUnified_{uid}"
                    g_series = df_g[df_g['unique_id'] == g_uid]
                    if not g_series.empty:
                        ax.plot(x_vals[:len(g_series)], g_series['y'], color=colors[name], label=name, linewidth=1)
            
            # Baselines
            for name in ['TSMixup', 'DBA', 'MagnitudeWarping']:
                df_b = baseline_dfs.get(name)
                id_map = baseline_id_maps.get(name, {})
                if df_b is not None:
                    b_uid = id_map.get(uid)
                    if b_uid:
                        b_series = df_b[df_b['unique_id'] == b_uid]
                        if not b_series.empty:
                            ax.plot(x_vals[:len(b_series)], b_series['y'], color=colors[name], label=name, linewidth=1)
            
            ax.set_title(f"Series: {uid} (Verified Mapping)")
            ax.legend(loc='upper right', fontsize='x-small', ncol=3)
            ax.grid(True, alpha=0.3)
        except Exception as e:
            print(f"      Error plotting series {uid}: {e}")

    plt.suptitle(f"Comparison: {dataset_name} ({group})", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    fname = os.path.join(OUTPUT_DIR, f"{dataset_name}_{group}_comparison.png")
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"  Saved comparison grid: {fname}")

def plot_single_series_breakdown(dataset_name, group, uid, df_real, grasynda_models, baseline_dfs, baseline_id_maps):
    """
    Plot a single series with each model in its own subplot to see the shape clearly.
    """
    models_to_show = [
        ('Original', 'black'),
        ('G_Q12_Uniform', '#D62728'),
        ('G_Vis_Q25_Uniform', '#1F77B4'),
        ('G_Standard', '#2CA02C'),
        ('TSMixup', '#9467BD'),
        ('DBA', '#8C564B'),
        ('MagnitudeWarping', '#FF7F0E')
    ]
    
    fig, axes = plt.subplots(len(models_to_show), 1, figsize=(12, 18), sharex=True)
    real_series = df_real[df_real['unique_id'] == uid]
    x_vals = np.arange(len(real_series))

    for i, (name, color) in enumerate(models_to_show):
        ax = axes[i]
        # Always background the original for reference
        ax.plot(x_vals, real_series['y'], color='black', alpha=0.1, label='Reference')
        
        if name == 'Original':
            ax.plot(x_vals, real_series['y'], color=color, linewidth=2)
        elif name.startswith('G_'):
            df_g = grasynda_models.get(name)
            if df_g is not None:
                g_uid = f"GrasyndaUnified_{uid}"
                g_series = df_g[df_g['unique_id'] == g_uid]
                if not g_series.empty:
                    ax.plot(x_vals[:len(g_series)], g_series['y'], color=color, linewidth=1.5)
        else:
            df_b = baseline_dfs.get(name)
            id_map = baseline_id_maps.get(name, {})
            b_uid = id_map.get(uid)
            if df_b is not None and b_uid:
                b_series = df_b[df_b['unique_id'] == b_uid]
                if not b_series.empty:
                    ax.plot(x_vals[:len(b_series)], b_series['y'], color=color, linewidth=1.5)
        
        ax.set_title(f"{name} | {uid}")
        ax.grid(True, alpha=0.2)
        ax.set_ylabel("Value")

    plt.tight_layout()
    fname = os.path.join(OUTPUT_DIR, f"{dataset_name}_{group}_{uid}_breakdown.png")
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"  Saved detailed breakdown: {fname}")

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.random.seed(42)
    
    for ds_name, grp in DATA_GROUPS:
        print(f"Processing {ds_name} ({grp})...")
        try:
            loader = DATASETS[ds_name]
            # Use load_everything (already has the right filters for these analysis runs)
            df_all, _, _, _, freq_int = loader.load_everything(grp)
            
            all_uids = df_all['unique_id'].unique()
            sample_uids = np.random.choice(all_uids, min(10, len(all_uids)), replace=False)
            
            df_sample = df_all[df_all['unique_id'].isin(sample_uids)].copy()
            
            # Generate Grasyndas
            g_methods = ['G_Standard', 'G_NoEns_Hybrid', 'G_Ens100_Mixed_Hybrid',
                         'G_Q12_Uniform', 'G_Q8_Uniform', 'G_Vis_Q25_Uniform']
            g_models = generate_grasyndas(df_sample, freq_int, g_methods)
            
            # Load Baselines and build ID maps using full real data for global indices
            b_dfs = {}
            b_id_maps = {}
            for b_name in ['TSMixup', 'DBA', 'MagnitudeWarping']:
                id_map, df_b = get_baseline_id_map(ds_name, grp, b_name, df_sample, df_all)
                if df_b is not None:
                    b_dfs[b_name] = df_b
                    b_id_maps[b_name] = id_map
            
            # Plot main comparison grid
            plot_comparison(ds_name, grp, sample_uids, df_sample, g_models, b_dfs, b_id_maps)
            
            # Plot 5 detailed breakdowns per dataset group
            for j in range(min(5, len(sample_uids))):
                plot_single_series_breakdown(ds_name, grp, sample_uids[j], df_sample, g_models, b_dfs, b_id_maps)
            
        except Exception as e:
            print(f"  Error processing {ds_name}_{grp}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
