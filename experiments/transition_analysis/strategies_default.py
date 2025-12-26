"""
Experiment: Impact of Transition Matrix Sampling Strategies

This script tests 4 different strategies for generating synthetic quantile paths:
1. Max Probability: Always choose the highest probability transition (greedy/deterministic)
2. Min Probability: Always choose the lowest non-zero probability transition
3. Default Grasynda: Use stochastic sampling with probabilities (original method)
4. Zero Probability: Always choose a transition that never occurred in training (0 probability)

WITHOUT modifying the original Grasynda code.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict
from utils.load_data.config import DATASETS
from src.grasynda_unified import GrasyndaUnified


class TransitionStrategyExperiment:
    """Custom class to test different transition matrix strategies."""
    
    def __init__(self, grasynda_model: GrasyndaUnified):
        """
        Args:
            grasynda_model: Trained GrasyndaUnified instance with transition_mats already learned
        """
        self.model = grasynda_model
        self.n_quantiles = grasynda_model.n_quantiles
        self.transition_mats = grasynda_model.transition_mats
    
    def generate_quantile_series_max_prob(self, df: pd.DataFrame, uid: str) -> np.ndarray:
        """
        Strategy 1: Always choose the HIGHEST probability transition.
        Improved: Handles ties by picking the closest quantile to current state.
        Handles unobserved states (uniform probs) by staying in place.
        """
        mat = self.transition_mats[uid]
        uid_df = df.query(f'unique_id=="{uid}"')
        
        q_series = np.zeros(len(uid_df), dtype=int)
        q_series[0] = uid_df['Quantile'].values[0]
        
        for t in range(1, len(q_series)):
            current_q = q_series[t-1]
            probs = mat[current_q]
            
            if np.all(probs == probs[0]): # Uniform/No data
                q_series[t] = current_q
            else:
                max_val = np.max(probs)
                max_indices = np.where(probs == max_val)[0]
                # Tie-break: pick closest to current
                q_series[t] = max_indices[np.argmin(np.abs(max_indices - current_q))]
        
        return q_series
    
    def generate_quantile_series_min_prob(self, df: pd.DataFrame, uid: str) -> np.ndarray:
        """
        Strategy 2: Always choose from the LOWEST non-zero probability transitions 
        starting from the quantile we are NOW in.
        """
        mat = self.transition_mats[uid]
        uid_df = df.query(f'unique_id=="{uid}"')
        
        q_series = np.zeros(len(uid_df), dtype=int)
        q_series[0] = uid_df['Quantile'].values[0]
        
        for t in range(1, len(q_series)):
            current_q = q_series[t-1]
            probs = mat[current_q]
            
            # 1. Find indices with prob > 0
            non_zero_indices = np.where(probs > 0)[0]
            
            # 2. Find the minimum value among those probabilities
            if len(non_zero_indices) > 0:
                min_val = np.min(probs[non_zero_indices])
                # 3. Randomly select from ALL quantiles that share this minimum probability
                candidates = non_zero_indices[np.where(probs[non_zero_indices] == min_val)[0]]
                q_series[t] = np.random.choice(candidates)
            else:
                # Should not happen as probabilities sum to 1, but for safety:
                q_series[t] = current_q
        
        return q_series
    
    def generate_quantile_series_default(self, df: pd.DataFrame, uid: str) -> np.ndarray:
        """
        Strategy 3: Default Grasynda - stochastic sampling using probabilities.
        This calls the original method on REMAINDER component.
        """
        # Use the model's original method (on remainder for Standard Grasynda)
        quantile_series_dict = self.model._generate_quantile_series(df, 'remainder')
        return quantile_series_dict[uid]
    
    def generate_quantile_series_zero_prob(self, df: pd.DataFrame, uid: str) -> np.ndarray:
        """
        Strategy 4: Always choose a ZERO probability transition.
        This explores impossible/never-seen transitions.
        """
        mat = self.transition_mats[uid]
        uid_df = df.query(f'unique_id=="{uid}"')
        
        q_series = np.zeros(len(uid_df), dtype=int)
        q_series[0] = uid_df['Quantile'].values[0]
        
        for t in range(1, len(q_series)):
            current_q = q_series[t-1]
            probs = mat[current_q]
            
            # 1. Find ALL targets with ZERO transition probability from HERE
            zero_indices = np.where(probs == 0)[0]
            
            if len(zero_indices) > 0:
                # 2. Randomly select among them to "respect the choice"
                q_series[t] = np.random.choice(zero_indices)
            else:
                # 3. If NO zero probabilities exist (row is uniform or saturated),
                # use the absolute minimum probability available
                min_val = np.min(probs)
                candidates = np.where(probs == min_val)[0]
                q_series[t] = np.random.choice(candidates)
        
        return q_series
    
    def generate_synthetic_values(self, df: pd.DataFrame, uid: str, 
                                   quantile_series: np.ndarray, 
                                   sampling_type: str = 'discrete') -> np.ndarray:
        """
        Convert quantile series to actual values using the specified sampling type.
        For Default Grasynda: uses REMAINDER component, then reconstructs.
        """
        uid_df = df.query(f'unique_id=="{uid}"')
        uid_vals = uid_df['remainder']  # Use remainder for Default Grasynda
        uid_quantiles = uid_df['Quantile']
        
        # Pre-calculate bin properties
        from scipy.stats import gaussian_kde
        
        bin_props = {}
        for q in range(self.n_quantiles):
            vals = uid_vals[uid_quantiles == q].values
            if len(vals) > 0:
                bin_props[q] = {
                    'vals': vals,
                    'min': vals.min(),
                    'max': vals.max(),
                    'kde': gaussian_kde(vals) if len(vals) > 1 and sampling_type == 'kde' else None
                }
            else:
                bin_props[q] = None
        
        synth_remainder = np.zeros(len(uid_vals))
        synth_remainder[0] = uid_vals.values[0]
        
        for i in range(1, len(uid_vals)):
            q = quantile_series[i]
            props = bin_props.get(q)
            
            if props is None:
                synth_remainder[i] = synth_remainder[i-1]
                continue
            
            if sampling_type == 'discrete':
                synth_remainder[i] = np.random.choice(props['vals'])
            elif sampling_type == 'continuous_uniform':
                if props['min'] == props['max']:
                    synth_remainder[i] = props['min']
                else:
                    synth_remainder[i] = np.random.uniform(props['min'], props['max'])
            elif sampling_type == 'kde':
                if props['kde']:
                    synth_remainder[i] = props['kde'].resample(1)[0][0]
                else:
                    synth_remainder[i] = np.random.choice(props['vals'])
        
        # RECONSTRUCT: Add synthetic remainder to original trend + seasonality
        orig_remainder = uid_df['remainder'].values
        mae_rem = np.mean(np.abs(orig_remainder - synth_remainder))
        std_y = np.std(uid_df['y'])
        std_rem = np.std(orig_remainder)
        
        print(f"    Diagnostic [{uid}]:")
        print(f"      - Remainder STD: {std_rem:.4f} vs Series STD: {std_y:.4f} (Ratio: {std_rem/std_y:.2%})")
        print(f"      - Synthetic Remainder MAE from Original: {mae_rem:.4f}")
        
        synth_vals = synth_remainder + uid_df['trend'].values + uid_df['seasonal'].values
        return synth_vals


def run_experiment(dataset_name='M3', group='Monthly', n_samples=5, n_quantiles=25):
    """
    Run the transition strategy experiment.
    
    Args:
        dataset_name: Dataset to use
        group: Data group
        n_samples: Number of time series to test
        n_quantiles: Number of quantiles for discretization
    """
    print(f"="*80)
    print(f"Transition Matrix Strategy Experiment")
    print(f"Dataset: {dataset_name} - {group}")
    print(f"Samples: {n_samples}, Quantiles: {n_quantiles}")
    print(f"="*80)
    
    # Load data
    data_loader = DATASETS[dataset_name]
    min_samples = data_loader.min_samples[group]
    df, horizon, n_lags, freq_str, freq_int = data_loader.load_everything(
        group, min_n_instances=min_samples
    )
    
    # Sample series
    np.random.seed(42)
    sample_uids = np.random.choice(
        df['unique_id'].unique(), 
        size=min(n_samples, len(df['unique_id'].unique())), 
        replace=False
    )
    df_sample = df[df['unique_id'].isin(sample_uids)].copy()
    
    # Initialize Grasynda model (STANDARD/DEFAULT variant)
    # STL decomposition + discrete remainder sampling
    print("\nInitializing Grasynda model...")
    grasynda = GrasyndaUnified(
        period=freq_int,
        n_quantiles=n_quantiles,
        components_to_model=['remainder'],  # Model remainder after STL
        sampling_type='discrete'              # Discrete sampling
    )
    
    # Prepare data: decompose and add quantiles for remainder
    print("Decomposing series with STL...")
    df_prepared = grasynda.decompose_tsd(df_sample, freq_int, False)
    print("Learning transition patterns from remainder...")
    df_prepared['Quantile'] = grasynda._get_quantiles(df_prepared, 'remainder')
    grasynda._calc_transition_matrix(df_prepared)
    
    # Initialize experiment
    experiment = TransitionStrategyExperiment(grasynda)
    
    # Output directory
    output_dir = f'assets/transition_experiment/{dataset_name}_{group}'
    os.makedirs(output_dir, exist_ok=True)
    
    # Run experiment for each series
    for uid in sample_uids:
        print(f"\n{'='*60}")
        print(f"Processing Series: {uid}")
        print(f"{'='*60}")
        
        uid_df = df_prepared[df_prepared['unique_id'] == uid]
        
        # Generate quantile series with each strategy
        strategies = {
            'Max Probability': experiment.generate_quantile_series_max_prob,
            'Min Probability': experiment.generate_quantile_series_min_prob,
            'Default Grasynda': experiment.generate_quantile_series_default,
            'Zero Probability': experiment.generate_quantile_series_zero_prob
        }
        
        results = {}
        for name, strategy_func in strategies.items():
            print(f"  - {name}...")
            q_series = strategy_func(df_prepared, uid)
            synth_vals = experiment.generate_synthetic_values(
                df_prepared, uid, q_series, sampling_type='discrete'
            )
            results[name] = {
                'quantile_series': q_series,
                'synthetic_values': synth_vals
            }
        
        # Visualize results
        visualize_comparison(uid, uid_df, results, output_dir, grasynda.transition_mats[uid])
    
    print(f"\n{'='*80}")
    print(f"Experiment complete! Results saved to: {output_dir}")
    print(f"{'='*80}")


def visualize_comparison(uid: str, original_df: pd.DataFrame, 
                         results: Dict, output_dir: str, matrix: np.ndarray):
    """
    Create visualization with 3 panels: series, quantiles, and transition matrix heatmap.
    """
    fig = plt.figure(figsize=(16, 14))
    gs = plt.GridSpec(3, 1, height_ratios=[1, 0.7, 0.7])
    
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])
    
    # Colors for each strategy
    colors = {
        'Max Probability': '#e74c3c',  # Red
        'Min Probability': '#3498db',  # Blue
        'Default Grasynda': '#2ecc71', # Green
        'Zero Probability': '#9b59b6'  # Purple
    }
    
    # ========== PANEL 1: Time Series Comparison ==========
    ax1.plot(original_df['ds'], original_df['y'], 
             label='Original', color='black', linewidth=2.5, alpha=0.8, zorder=5)
    
    for name, data in results.items():
        ax1.plot(original_df['ds'], data['synthetic_values'],
                 label=name, color=colors[name], linewidth=1.5, alpha=0.6)
    
    ax1.set_title(f'Series {uid}: Transition Strategy Comparison', 
                  fontsize=16, fontweight='bold')
    ax1.set_ylabel('Value', fontsize=12)
    ax1.legend(loc='best', fontsize=10, framealpha=0.9, ncol=3)
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # ========== PANEL 2: Quantile Path Comparison ==========
    ax2.plot(original_df['ds'], original_df['Quantile'], 
             label='Original Quantile Path', color='black', 
             linewidth=2.5, alpha=0.6, linestyle=':', zorder=5)
    
    for name, data in results.items():
        ax2.plot(original_df['ds'], data['quantile_series'],
                 label=name, color=colors[name], linewidth=1.8, alpha=0.7)
    
    ax2.set_title('Quantile Series (Discretized Remainder)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Time', fontsize=11)
    ax2.set_ylabel('Quantile Bin', fontsize=11)
    ax2.legend(loc='best', fontsize=10, framealpha=0.9, ncol=3)
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    # ========== PANEL 3: Transition Matrix Heatmap ==========
    # Mask zeros for better visibility of faint probabilities
    masked_matrix = np.ma.masked_where(matrix == 0, matrix)
    im = ax3.imshow(masked_matrix, cmap='viridis', interpolation='nearest', origin='lower')
    ax3.set_title(f'Original Transition Matrix (Learned from {len(original_df)} points)', 
                  fontsize=14, fontweight='bold')
    ax3.set_xlabel('To Quantile', fontsize=11)
    ax3.set_ylabel('From Quantile', fontsize=11)
    
    # Add colorbar
    fig.colorbar(im, ax=ax3, label='Probability', pad=0.01, aspect=30)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'comparison_{uid}.png'), dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    print("\n" + "="*80)
    print("TRANSITION STRATEGY EXPERIMENT - MULTI-DATASET RUN")
    print("="*80)
    
    # Define datasets to test
    datasets_to_test = [
        ('M3', 'Monthly'),
        ('M3', 'Quarterly'),
        ('Tourism', 'Monthly'),
        ('Tourism', 'Quarterly')
    ]
    
    # Run experiment on each dataset
    for dataset_name, group in datasets_to_test:
        print(f"\n{'#'*80}")
        print(f"# Starting: {dataset_name} - {group}")
        print(f"{'#'*80}")
        
        try:
            run_experiment(
                dataset_name=dataset_name,
                group=group,
                n_samples=5,  # 5 random series per dataset
                n_quantiles=25
            )
        except Exception as e:
            print(f"\n[ERROR] Failed on {dataset_name} {group}: {e}")
            continue
    
    print("\n" + "="*80)
    print("ALL EXPERIMENTS COMPLETE!")
    print("="*80)
    print("\nResults saved to: assets/transition_experiment/")
    print("  - M3_Monthly/")
    print("  - M3_Quarterly/")
    print("  - Tourism_Monthly/")
    print("  - Tourism_Quarterly/")
    print("\nKey Questions Answered:")
    print("1. Max Probability: Does greedy selection produce smooth but unrealistic patterns?")
    print("2. Min Probability: Does exploring rare transitions create erratic behavior?")
    print("3. Default Grasynda: Baseline stochastic sampling")
    print("4. Zero Probability: Does violating learned patterns produce nonsensical data?")
