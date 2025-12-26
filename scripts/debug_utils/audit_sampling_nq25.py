
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

def get_random_uids(dataset_name, group, k=5):
    data_loader = DATASETS[dataset_name]
    df, _, _, _, _ = data_loader.load_everything(group)
    all_uids = df['unique_id'].unique().tolist()
    return random.sample(all_uids, k)

def plot_comparison_nq25(dataset_name, group, uids, n_q=25):
    print(f"Generating NQ={n_q} Sampling Comparisons for {dataset_name}...")
    data_loader = DATASETS[dataset_name]
    df, _, _, _, freq_int = data_loader.load_everything(group)
    
    output_base = f"assets/sampling_audit_nq{n_q}/{dataset_name}"
    os.makedirs(output_base, exist_ok=True)
    
    for uid in uids:
        uid_df = df[df['unique_id'] == uid].copy()
        print(f"  Processing {uid}...")
        
        # 1. Discrete Hybrid
        discrete_model = GrasyndaUnified(
            period=freq_int,
            n_quantiles=n_q,
            components_to_model=['trend', 'remainder'],
            apply_differentiation=True,
            sampling_type='discrete'
        )
        
        # 2. Continuous Uniform Hybrid
        uniform_model = GrasyndaUnified(
            period=freq_int,
            n_quantiles=n_q,
            components_to_model=['trend', 'remainder'],
            apply_differentiation=True,
            sampling_type='continuous_uniform'
        )
        
        # Decompose & Transform
        synth_discrete = discrete_model.transform(uid_df)
        synth_uniform = uniform_model.transform(uid_df)
        
        # Extraction for visualization (Recalculate internal traces for plotting)
        decomposed = discrete_model.decompose_tsd(uid_df, freq_int, False)
        orig_rem = decomposed['remainder'].values
        trend_prep = discrete_model._differentiate_component(decomposed, 'trend')
        orig_slopes = trend_prep['diff_trend'].values
        
        # --- Plotting ---
        fig = plt.figure(figsize=(18, 14))
        gs = fig.add_gridspec(3, 2)
        
        # Row 0: Full Reconstruction
        ax00 = fig.add_subplot(gs[0, 0])
        ax00.plot(uid_df['ds'], uid_df['y'], color='black', label='Original', alpha=0.5, linewidth=2)
        ax00.plot(uid_df['ds'], synth_discrete['y'], color='#3498db', label='Hybrid Discrete')
        ax00.set_title(f"Hybrid: Discrete Sampling (N={n_q})")
        ax00.legend()
        
        ax01 = fig.add_subplot(gs[0, 1])
        ax01.plot(uid_df['ds'], uid_df['y'], color='black', label='Original', alpha=0.5, linewidth=2)
        ax01.plot(uid_df['ds'], synth_uniform['y'], color='#e74c3c', label='Hybrid Cont. Uniform')
        ax01.set_title(f"Hybrid: Continuous Uniform Sampling (N={n_q})")
        ax01.legend()
        
        # Row 1: Distribution of Residuals (Character Check)
        # We'll just plot the first gen of each to show visual texture
        # To get the actual traces used in the transform above, we'd need to mock the sampler, 
        # but for visualization, re-running a local sample is sufficient to show "texture".
        
        # Local re-sampling to show texture
        def local_sample_discrete(vals, n_q):
            # Simple simulation of what happened inside
            return np.random.choice(vals, size=len(vals)) # Note: N=25 would be per bin, this is simplified for visual comparison
        
        # Correctly simulated component comparison
        ax10 = fig.add_subplot(gs[1, 0])
        ax10.plot(uid_df['ds'], orig_rem, color='black', alpha=0.3, label='Orig Rem')
        # We'll just show the generated remainder from the actual transform
        # We need to hack a bit to extract them or just re-run the logic
        # Actually, let's just use the integrated trend from the synth_df to show divergence
        ax10.set_title("Resulting Diversity: Discrete")
        ax10.text(0.5, 0.5, "Uses exact\nhistorical points", ha='center', va='center', fontsize=14, color='blue')
        
        ax11 = fig.add_subplot(gs[1, 1])
        ax11.set_title("Resulting Diversity: Continuous Uniform")
        ax11.text(0.5, 0.5, "Samples across\nthe bin interval", ha='center', va='center', fontsize=14, color='red')
        
        # Row 2: Remainder Contrast
        # We'll manually sample to show what "Continuous Uniform N=25" looks like vs "N=1"
        # Logic: find bins, sample min-max of those bins
        quantiles = pd.qcut(orig_rem, n_q, labels=False, duplicates='drop')
        synth_rem_uni = np.zeros(len(orig_rem))
        for i in range(len(orig_rem)):
            q = quantiles[i]
            bin_vals = orig_rem[quantiles == q]
            synth_rem_uni[i] = np.random.uniform(bin_vals.min(), bin_vals.max())
            
        ax20 = fig.add_subplot(gs[2, 0])
        ax20.plot(uid_df['ds'], orig_rem, color='black', alpha=0.3)
        ax20.plot(uid_df['ds'], np.random.choice(orig_rem, len(orig_rem)), color='#2ecc71', alpha=0.7, label='Discrete Reshuffle')
        ax20.set_title("Discrete Remainder (Values from History)")
        ax20.legend()

        ax21 = fig.add_subplot(gs[2, 1])
        ax21.plot(uid_df['ds'], orig_rem, color='black', alpha=0.3)
        ax21.plot(uid_df['ds'], synth_rem_uni, color='#2ecc71', alpha=0.7, label='Cont. Uniform N=25')
        ax21.set_title("Continuous Uniform Remainder (Fills Bin Ranges)")
        ax21.legend()

        plt.tight_layout()
        save_path = f"{output_base}/{uid}_nq{n_q}.png"
        plt.savefig(save_path, dpi=100)
        plt.close()

if __name__ == "__main__":
    m3_uids = get_random_uids('M3', 'Monthly')
    tourism_uids = get_random_uids('Tourism', 'Monthly')
    
    plot_comparison_nq25('M3', 'Monthly', m3_uids, n_q=25)
    plot_comparison_nq25('Tourism', 'Monthly', tourism_uids, n_q=25)
