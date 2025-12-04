"""
Comprehensive Visibility Graph Comparison - 10 Series with Full Transition Matrices
All methods overlaid on same axis
"""

import sys
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from utils.load_data.config import DATASETS
from src.qgraph_ts import Grasynda
from src.grasynda_visibility import GrasyndaVisibilityGraph

def print_transition_matrix(name: str, gen_instance, uid: str):
    """Print full transition matrix."""
    print(f"\n{'='*80}")
    print(f"TRANSITION MATRIX: {name} | Series: {uid}")
    print(f"{'='*80}")
    
    if hasattr(gen_instance, 'degree_transitions'):
        # Visibility graph
        trans_info = gen_instance.degree_transitions[uid]
        trans_probs = trans_info['transition_probs']
        unique_degrees = trans_info['unique_degrees']
        
        print(f"Total degrees: {len(unique_degrees)}")
        print(f"Degrees: {sorted(unique_degrees)}\n")
        
        print("Transition probabilities:")
        for deg_from in sorted(unique_degrees):
            if deg_from in trans_probs:
                print(f"\n  From {deg_from}:")
                sorted_trans = sorted(trans_probs[deg_from].items(), 
                                    key=lambda x: x[1], reverse=True)
                for deg_to, prob in sorted_trans:
                    print(f"    → {deg_to}: {prob:.4f}")
                    
    elif hasattr(gen_instance, 'transition_matrix'):
        # Quantile
        trans_matrix = gen_instance.transition_matrix[uid]
        print(f"Shape: {trans_matrix.shape}")
        print(f"\nFull matrix:")
        print(trans_matrix)
    else:
        print("No matrix")

def visualize_all_methods():
    print("="*80)
    print("COMPREHENSIVE COMPARISON - 10 SERIES")
    print("="*80)
    
    # Load data
    data_loader = DATASETS['M3']
    group = 'Monthly'
    min_samples = data_loader.min_samples[group]
    
    df, horizon, n_lags, freq_str, freq_int = data_loader.load_everything(
        group, min_n_instances=min_samples
    )
    
    # Pick 10 series
    all_uids = df['unique_id'].unique()
    np.random.seed(42)
    sample_uids = np.random.choice(all_uids, size=min(10, len(all_uids)), replace=False)
    
    print(f"\nSelected {len(sample_uids)} series\n")
    
    # Output folder
    output_dir = 'assets/plots/comprehensive_comparison'
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each series
    for series_idx, uid in enumerate(sample_uids, 1):
        print(f"\n{'='*80}")
        print(f"SERIES {series_idx}/10: {uid}")
        print(f"{'='*80}")
        
        df_sample = df[df['unique_id'] == uid].copy()
        print(f"Length: {len(df_sample)}")
        
        generators = {}
        synthetic_data = {}
        
        print("\nGenerating...")
        
        # 1. Quantile Uniform
        print("  1/5: Quantile Uniform")
        gen = Grasynda(n_quantiles=25, quantile_on='remainder', period=freq_int, 
                       ensemble_transitions=False)
        synthetic_data['Quantile Uniform'] = gen.transform(df_sample)
        generators['Quantile Uniform'] = gen
        
        # 2. Horizontal VG (Remainder)
        print("  2/5: Horizontal VG (Remainder)")
        gen = GrasyndaVisibilityGraph(period=freq_int, visibility_type='horizontal',
                                      quantile_on='remainder', use_decomposition=True)
        synthetic_data['Horizontal VG (Remainder)'] = gen.transform(df_sample)
        generators['Horizontal VG (Remainder)'] = gen
        
        # 3. Natural VG (Remainder)
        print("  3/5: Natural VG (Remainder)")
        gen = GrasyndaVisibilityGraph(period=freq_int, visibility_type='natural',
                                      quantile_on='remainder', use_decomposition=True)
        synthetic_data['Natural VG (Remainder)'] = gen.transform(df_sample)
        generators['Natural VG (Remainder)'] = gen
        
        # 4. Natural VG (Trend)
        print("  4/5: Natural VG (Trend)")
        gen = GrasyndaVisibilityGraph(period=freq_int, visibility_type='natural',
                                      quantile_on='trend', use_decomposition=True)
        synthetic_data['Natural VG (Trend)'] = gen.transform(df_sample)
        generators['Natural VG (Trend)'] = gen
        
        # 5. Natural VG (No STL)
        print("  5/5: Natural VG (No STL)")
        gen = GrasyndaVisibilityGraph(period=freq_int, visibility_type='natural',
                                      use_decomposition=False)
        synthetic_data['Natural VG (No STL)'] = gen.transform(df_sample)
        generators['Natural VG (No STL)'] = gen
        
        # Print matrices
        print(f"\n{'='*80}")
        print(f"TRANSITION MATRICES: {uid}")
        print(f"{'='*80}")
        
        for name, gen in generators.items():
            print_transition_matrix(name, gen, uid)
        
        # Overlay plot
        print(f"\nCreating overlay plot...")
        
        plt.figure(figsize=(16, 8))
        
        # Original
        plt.plot(df_sample['ds'], df_sample['y'], 
                color='black', linewidth=3, alpha=0.9, label='Original', zorder=10)
        
        # Synthetic
        colors = ['blue', 'purple', 'green', 'orange', 'red']
        alphas = [0.6, 0.6, 0.6, 0.6, 0.5]
        linestyles = ['--', '--', '--', '-.', ':']
        
        for idx, (name, synth_df) in enumerate(synthetic_data.items()):
            if 'Quantile' in name:
                synth_uid = f'Grasynda_{uid}'
            else:
                synth_uid = f'GrasyndaVG_{uid}'
            
            synth_series = synth_df[synth_df['unique_id'] == synth_uid]
            
            plt.plot(synth_series['ds'], synth_series['y'], 
                    color=colors[idx], linewidth=2, alpha=alphas[idx],
                    linestyle=linestyles[idx], label=name)
        
        plt.title(f"All Methods - {uid}", fontsize=14, fontweight='bold')
        plt.xlabel("Date", fontsize=12)
        plt.ylabel("Value", fontsize=12)
        plt.legend(loc='best', fontsize=10, framealpha=0.9)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        save_path = os.path.join(output_dir, f'overlay_{uid}.png')
        plt.savefig(save_path, dpi=150)
        plt.close()
        
        print(f"Saved: {save_path}")
    
    print(f"\n{'='*80}")
    print("COMPLETE!")
    print(f"Results in: {output_dir}")
    print(f"{'='*80}")

if __name__ == "__main__":
    visualize_all_methods()
