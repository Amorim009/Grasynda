import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Robust project root detection
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
sys.path.insert(0, project_root)

# Attempt imports
try:
    from src.grasynda_unified import GrasyndaUnified
    from utils.load_data.config import DATASETS, DATA_GROUPS
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

# --- CONFIGURATION: ONLY NOISE TEST ---
MODELS = {
    'Noise Test': 'Noise_test'
}

OUTPUT_DIR = os.path.join(project_root, "assets", "results", "visualizations", "noise_tests")

def generate_grasyndas(df_real, freq_int, models_dict):
    """
    Generates synthetic data specifically for the requested models.
    """
    model_results = {}

    for label, method_code in models_dict.items():
        print(f"  > Generating {label}...")
        
        try:
            # Logic specifically for the Noise Test configuration
            if method_code == 'Noise_test':
                model = GrasyndaUnified(
                    period=freq_int,
                    n_quantiles=1,                  # Single quantile often approximates noise/global distribution
                    sampling_type='continuous_uniform',
                    graph_type='quantile',
                    components_to_model=['y']
                )
            else:
                # Fallback or future expansion
                model = GrasyndaUnified(period=freq_int, components_to_model=['y'])

            # Generate the data
            # .transform() usually returns the synthetic dataframe
            df_syn = model.transform(df_real.copy())
            model_results[label] = df_syn
            
        except Exception as e:
            print(f"    Failed to generate {label}: {e}")
            model_results[label] = None

    return model_results

def plot_focused_comparison(dataset_name, group, uids, df_real, models_data):
    # Dynamic plot height
    fig, axes = plt.subplots(len(uids), 1, figsize=(15, 3.5 * len(uids)))
    if len(uids) == 1: axes = [axes]
    
    colors = {
        'Original': 'black',
        'Noise Test': '#2CA02C',  # Green for Noise Test
    }

    for i, uid in enumerate(uids):
        ax = axes[i]
        real_series = df_real[df_real['unique_id'] == uid].sort_values('ds')
        x_vals = np.arange(len(real_series))
        
        # 1. Plot Original
        ax.plot(x_vals, real_series['y'].values, color=colors['Original'], label='Original', linewidth=2.0, alpha=0.6)
        
        # 2. Plot Noise Test
        for name, df_g in models_data.items():
            if df_g is not None and not df_g.empty:
                # Try to find the synthetic series for this UID
                syn_uid = f"GrasyndaUnified_{uid}"
                g_series = df_g[df_g['unique_id'] == syn_uid]
                
                # Fallback search if exact ID match fails
                if g_series.empty:
                     g_series = df_g[df_g['unique_id'].astype(str).str.contains(str(uid))]

                if not g_series.empty:
                    y_syn = g_series['y'].values
                    # Match lengths for plotting
                    x_syn = x_vals[:len(y_syn)] if len(y_syn) <= len(x_vals) else np.arange(len(y_syn))
                    
                    c = colors.get(name, 'red') # Default to red if key missing
                    ax.plot(x_syn, y_syn, color=c, label=name, linewidth=1.2, alpha=0.9)
        
        ax.set_title(f"Series: {uid}", loc='left', fontsize=12, fontweight='bold')
        
        # Unique legend
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize='small')
        
        ax.grid(True, alpha=0.25, linestyle='--')
        ax.set_facecolor('#FAFAFA')

    plt.suptitle(f"Grasynda Noise Test: {dataset_name} ({group})", fontsize=16, y=0.99)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    fname = os.path.join(OUTPUT_DIR, f"{dataset_name}_{group}_noise_test.png")
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved plot: {fname}")

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.random.seed(42)
    
    # Define targets
    TARGET_GROUPS = [('Gluonts', 'm1_monthly'), ('M3', 'Monthly')]
    
    for ds_name, grp in TARGET_GROUPS:
        print(f"Processing {ds_name} ({grp})...")
        try:
            if ds_name not in DATASETS:
                print(f"  Skipping {ds_name}: Dataset not found in config.")
                continue

            # Load Data
            loader = DATASETS[ds_name]
            load_result = loader.load_everything(grp)
            df_all = load_result[0]
            freq_int = load_result[4]
            
            # Select random samples
            all_uids = df_all['unique_id'].unique()
            sample_uids = np.random.choice(all_uids, min(5, len(all_uids)), replace=False)
            df_sample = df_all[df_all['unique_id'].isin(sample_uids)].copy()
            
            # Generate ONLY Noise Test
            models_data = generate_grasyndas(df_sample, freq_int, MODELS)
            
            # Plot
            plot_focused_comparison(ds_name, grp, sample_uids, df_sample, models_data)
            
        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()