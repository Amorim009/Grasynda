import pandas as pd
import numpy as np
import os

def analyze_all_results():
    print("=" * 80)
    print("FINAL COMPREHENSIVE MODEL EVALUATION")
    print("=" * 80)

    # 1. Define result files
    files = {
        'Grasynda': 'assets/results/neural_sampler_mase_results.csv',
        'OtherModels': 'assets/results/other_models_mase_results.csv',
        'Visibility': 'assets/results/visibility_graph_mase_results.csv'
    }

    # 2. Load and Merge Data
    dfs = []
    for source, filepath in files.items():
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            # Standardize column names if necessary (though they should be consistent)
            dfs.append(df)
        else:
            print(f"Warning: File not found: {filepath}")

    if not dfs:
        print("No result files found!")
        return

    # Merge on dataset and group
    master_df = dfs[0]
    for i in range(1, len(dfs)):
        # Merge with suffixes to handle overlapping columns (like 'original', 'SeasonalNaive')
        master_df = pd.merge(master_df, dfs[i], on=['dataset', 'group'], how='outer', suffixes=('', f'_{i}'))

    # 3. Define Methods to Compare
    # We prioritize the primary instance of each method and ignore duplicate columns from merges
    all_methods = [
        # Baselines
        'SeasonalNaive', 
        'original',
        
        # Grasynda (Quantile/Neural)
        'grasynda_uniform',
        'grasynda_kde',
        # 'grasynda_neural', # Include if available
        
        # Grasynda (Visibility Graph)
        'grasynda_vis_degree',
        'grasynda_vis_hybrid',
        
        # Other Augmentation Models
        'SeasonalMBB',
        'Jittering',
        'Scaling',
        'TimeWarping',
        'MagnitudeWarping',
        'TSMixup',
        'DBA'
    ]

    # Filter for methods that actually exist in the merged dataframe
    # We check for the exact name, or the name with a suffix if the primary one is missing
    final_methods = []
    for method in all_methods:
        if method in master_df.columns:
            final_methods.append(method)
        else:
            # Check for suffixed versions if original is missing (unlikely for primary keys)
            # But 'original' might be 'original' in first file and 'original_1' in second
            # We trust the first file's version (Grasynda file) as the canonical one for baselines
            pass

    print(f"Comparing {len(final_methods)} methods across {len(master_df)} datasets.")
    print("-" * 80)

    # 4. Calculate Metrics
    
    # A. Average MASE
    avg_mase = master_df[final_methods].mean().sort_values()
    
    # B. Average Rank
    # Rank per dataset (row), then average the ranks
    # method='min' means ties get the same lower rank
    ranks_per_dataset = master_df[final_methods].rank(axis=1, method='min', ascending=True)
    avg_rank = ranks_per_dataset.mean().sort_values()

    # 5. Create Summary Table
    summary = pd.DataFrame({
        'Avg MASE': avg_mase,
        'Avg Rank': avg_rank
    })

    # Add a "Score" metric? (Rank of MASE + Rank of Rank)? 
    # For now, just sort by Avg Rank as it's the most robust metric
    summary = summary.sort_values('Avg Rank')

    # Round for display
    summary_display = summary.round(3)

    print("\nFINAL LEADERBOARD (Sorted by Average Rank)")
    print("(Lower is Better for both metrics)")
    print("-" * 80)
    print(summary_display)
    print("-" * 80)

    # 6. Save Results
    output_path = 'assets/results/final_model_comparison.csv'
    summary.to_csv(output_path)
    print(f"\nSaved detailed summary to: {output_path}")

if __name__ == "__main__":
    analyze_all_results()
