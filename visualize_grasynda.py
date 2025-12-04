"""
Consolidated Grasynda Visualization Script

This script contains all visualization functions for comparing different Grasynda variants.
Use the command-line argument to specify which visualization to generate:
  - 'all': Generate all visualizations (default)
  - 'variants': Compare Uniform, KDE, and Visibility variants
  - 'decomp': Compare raw vs decomposed visibility graph
  - 'tstr': Visualize TSTR experiment results
"""

import sys
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from utils.load_data.config import DATASETS
from src.qgraph_ts import Grasynda, GrasyndaKDE
from src.grasynda_visibility import GrasyndaVisibilityGraph


def visualize_variants():
    """Compare Original + 4 synthetic variants: Uniform, KDE, Vis+STL, Vis-STL."""
    print("\n" + "=" * 80)
    print("VISUALIZING ALL GRASYNDA VARIANTS")
    print("=" * 80)

    # Load M3 Monthly dataset
    data_loader = DATASETS['M3']
    group = 'Monthly'
    min_samples = data_loader.min_samples[group]
    
    df, horizon, n_lags, freq_str, freq_int = data_loader.load_everything(
        group, min_n_instances=min_samples
    )
    
    # Select 3 random series
    np.random.seed(42)
    sample_uids = np.random.choice(df['unique_id'].unique(), size=3, replace=False)
    df_sample = df[df['unique_id'].isin(sample_uids)].copy()
    
    # Generate ALL Synthetic Variants
    print("Generating Grasynda Uniform...")
    gen_uniform = Grasynda(n_quantiles=25, quantile_on='remainder', period=freq_int, ensemble_transitions=False)
    synth_uniform = gen_uniform.transform(df_sample)
    
    print("Generating Grasynda KDE...")
    gen_kde = GrasyndaKDE(n_quantiles=25, quantile_on='remainder', period=freq_int, ensemble_transitions=False)
    synth_kde = gen_kde.transform(df_sample)
    
    print("Generating Visibility WITH STL (Decomposition)...")
    gen_vis_decomp = GrasyndaVisibilityGraph(period=freq_int, visibility_type='horizontal', 
                                             generation_method='degree_matching', use_decomposition=True)
    synth_vis_decomp = gen_vis_decomp.transform(df_sample)
    
    print("Generating Visibility WITHOUT STL (Raw)...")
    gen_vis_raw = GrasyndaVisibilityGraph(period=freq_int, visibility_type='horizontal', 
                                         generation_method='degree_matching', use_decomposition=False)
    synth_vis_raw = gen_vis_raw.transform(df_sample)
    
    # Plotting
    output_dir = 'assets/plots'
    os.makedirs(output_dir, exist_ok=True)
    
    for uid in sample_uids:
        print(f"Plotting series: {uid}")
        
        orig = df_sample[df_sample['unique_id'] == uid]
        
        def get_synth(synth_df, alias):
            return synth_df[synth_df['unique_id'] == f'{alias}_{uid}']
        
        s_uniform = get_synth(synth_uniform, 'Grasynda')
        s_kde = get_synth(synth_kde, 'GrasyndaKDE')
        s_vis_decomp = get_synth(synth_vis_decomp, 'GrasyndaVG')
        s_vis_raw = get_synth(synth_vis_raw, 'GrasyndaVG')
        
        # Create ONE plot with all methods overlaid
        plt.figure(figsize=(16, 8))
        
        # Plot all methods on the same axes
        plt.plot(orig['ds'], orig['y'], label='Original', color='black', linewidth=2.5, alpha=0.9)
        
        if not s_uniform.empty:
            plt.plot(s_uniform['ds'], s_uniform['y'], label='Grasynda Uniform', 
                    color='blue', linewidth=2, alpha=0.7, linestyle='--')
        
        if not s_kde.empty:
            plt.plot(s_kde['ds'], s_kde['y'], label='Grasynda KDE', 
                    color='green', linewidth=2, alpha=0.7, linestyle='--')
        
        if not s_vis_decomp.empty:
            plt.plot(s_vis_decomp['ds'], s_vis_decomp['y'], label='Visibility WITH STL', 
                    color='purple', linewidth=2, alpha=0.7, linestyle='--')
        
        if not s_vis_raw.empty:
            plt.plot(s_vis_raw['ds'], s_vis_raw['y'], label='Visibility WITHOUT STL (Noisy)', 
                    color='red', linewidth=2, alpha=0.7, linestyle=':')
        
        plt.title(f"Series {uid}: All Synthetic Generation Methods Comparison", fontsize=14, fontweight='bold')
        plt.xlabel("Date", fontsize=12)
        plt.ylabel("Value", fontsize=12)
        plt.legend(loc='best', fontsize=11)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(output_dir, f'all_variants_{uid}.png')
        plt.savefig(save_path, dpi=150)
        plt.close()
        
    print(f"All variants plots saved to {output_dir}")


def visualize_decomp_comparison():
    """Compare raw vs decomposed visibility graph generation."""
    print("\n" + "=" * 80)
    print("COMPARING RAW VS DECOMPOSED VISIBILITY GRAPH")
    print("=" * 80)

    data_loader = DATASETS['M3']
    group = 'Monthly'
    min_samples = data_loader.min_samples[group]
    
    df, horizon, n_lags, freq_str, freq_int = data_loader.load_everything(
        group, min_n_instances=min_samples
    )
    
    target_uids = ['M884', 'M851']
    df_sample = df[df['unique_id'].isin(target_uids)].copy()
    
    if df_sample.empty:
        target_uids = df['unique_id'].unique()[:2]
        df_sample = df[df['unique_id'].isin(target_uids)].copy()
    
    # Generate with/without decomposition
    print("Generating: Raw Data Mode...")
    gen_raw = GrasyndaVisibilityGraph(period=freq_int, visibility_type='horizontal',
                                     generation_method='degree_matching', use_decomposition=False)
    synth_raw = gen_raw.transform(df_sample)
    
    print("Generating: Decomposed Mode...")
    gen_decomp = GrasyndaVisibilityGraph(period=freq_int, visibility_type='horizontal',
                                        generation_method='degree_matching', use_decomposition=True)
    synth_decomp = gen_decomp.transform(df_sample)
    
    output_dir = 'assets/plots'
    os.makedirs(output_dir, exist_ok=True)
    
    for uid in target_uids:
        print(f"Plotting series: {uid}")
        
        orig = df_sample[df_sample['unique_id'] == uid]
        
        def get_synth(df, alias):
            return df[df['unique_id'] == f'{alias}_{uid}']
            
        s_raw = get_synth(synth_raw, 'GrasyndaVG')
        s_decomp = get_synth(synth_decomp, 'GrasyndaVG')
        
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 1, 1)
        plt.plot(orig['ds'], orig['y'], label='Original', color='black', linewidth=2)
        if not s_raw.empty:
            plt.plot(s_raw['ds'], s_raw['y'], label='VG (Raw Mode)', color='red', alpha=0.7)
        plt.title(f"Series {uid}: Original vs VG Raw Mode (No Decomposition)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 1, 2)
        plt.plot(orig['ds'], orig['y'], label='Original', color='black', linewidth=2)
        if not s_decomp.empty:
            plt.plot(s_decomp['ds'], s_decomp['y'], label='VG (Decomposed Mode)', color='green', alpha=0.7)
        plt.title(f"Series {uid}: Original vs VG Decomposed Mode (With STL)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'decomp_comparison_{uid}.png'))
        plt.close()
        
    print(f"Decomposition comparison plots saved to {output_dir}")


def visualize_tstr_methods():
    """Visualize TSTR experiment results."""
    print("\n" + "=" * 80)
    print("VISUALIZING TSTR METHODS")
    print("=" * 80)

    data_loader = DATASETS['M3']
    group = 'Monthly'
    min_samples = data_loader.min_samples[group]
    
    df, horizon, n_lags, freq_str, freq_int = data_loader.load_everything(
        group, min_n_instances=min_samples
    )
    
    np.random.seed(42)
    sample_uids = np.random.choice(df['unique_id'].unique(), size=3, replace=False)
    df_sample = df[df['unique_id'].isin(sample_uids)].copy()
    
    # Generate all TSTR variants
    print("Generating Grasynda Uniform...")
    gen_uniform = Grasynda(n_quantiles=25, quantile_on='remainder', period=freq_int, ensemble_transitions=False)
    synth_uniform = gen_uniform.transform(df_sample)
    
    print("Generating Visibility Raw...")
    gen_vis_raw = GrasyndaVisibilityGraph(period=freq_int, visibility_type='horizontal',
                                         generation_method='degree_matching', use_decomposition=False)
    synth_vis_raw = gen_vis_raw.transform(df_sample)
    
    print("Generating Visibility Decomposed...")
    gen_vis_decomp = GrasyndaVisibilityGraph(period=freq_int, visibility_type='horizontal',
                                            generation_method='degree_matching', use_decomposition=True)
    synth_vis_decomp = gen_vis_decomp.transform(df_sample)
    
    output_dir = 'assets/plots'
    os.makedirs(output_dir, exist_ok=True)
    
    for uid in sample_uids:
        print(f"Plotting series: {uid}")
        
        orig = df_sample[df_sample['unique_id'] == uid]
        
        def get_synth(synth_df, alias):
            return synth_df[synth_df['unique_id'] == f'{alias}_{uid}']
        
        s_uniform = get_synth(synth_uniform, 'Grasynda')
        s_vis_raw = get_synth(synth_vis_raw, 'GrasyndaVG')
        s_vis_decomp = get_synth(synth_vis_decomp, 'GrasyndaVG')
        
        fig, axes = plt.subplots(4, 1, figsize=(15, 12))
        
        # Real Baseline
        axes[0].plot(orig['ds'], orig['y'], label='Real Baseline', color='black', linewidth=2)
        axes[0].set_title(f"Series {uid}: Real Baseline (MASE: 1.129)")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Uniform TSTR
        axes[1].plot(orig['ds'], orig['y'], label='Original', color='black', alpha=0.3, linewidth=1)
        if not s_uniform.empty:
            axes[1].plot(s_uniform['ds'], s_uniform['y'], label='Uniform Synthetic', color='blue', linewidth=2)
        axes[1].set_title(f"Uniform TSTR (MASE: 1.138)")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # VisRaw TSTR
        axes[2].plot(orig['ds'], orig['y'], label='Original', color='black', alpha=0.3, linewidth=1)
        if not s_vis_raw.empty:
            axes[2].plot(s_vis_raw['ds'], s_vis_raw['y'], label='VisRaw Synthetic', color='red', linewidth=2)
        axes[2].set_title(f"VisRaw TSTR (MASE: 2.064 - Looks like NOISE!)")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        # VisDecomp TSTR
        axes[3].plot(orig['ds'], orig['y'], label='Original', color='black', alpha=0.3, linewidth=1)
        if not s_vis_decomp.empty:
            axes[3].plot(s_vis_decomp['ds'], s_vis_decomp['y'], label='VisDecomp Synthetic', color='green', linewidth=2)
        axes[3].set_title(f"VisDecomp TSTR (MASE: 1.137 - FIXED with Decomposition!)")
        axes[3].legend()
        axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(output_dir, f'tstr_comparison_{uid}.png')
        plt.savefig(save_path)
        plt.close()
        
    print(f"TSTR plots saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate Grasynda visualizations')
    parser.add_argument('mode', nargs='?', default='all', 
                       choices=['all', 'variants', 'decomp', 'tstr'],
                       help='Visualization mode (default: all)')
    
    args = parser.parse_args()
    
    if args.mode == 'all':
        visualize_variants()
        visualize_decomp_comparison()
        visualize_tstr_methods()
    elif args.mode == 'variants':
        visualize_variants()
    elif args.mode == 'decomp':
        visualize_decomp_comparison()
    elif args.mode == 'tstr':
        visualize_tstr_methods()
    
    print("\n" + "=" * 80)
    print("ALL VISUALIZATIONS COMPLETE")
    print("=" * 80)
