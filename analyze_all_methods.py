import pandas as pd
import numpy as np

# Read the main results files (using M3-Monthly NHITS as example to get column structure)
datasets_groups = [
    ('Gluonts', 'm1_monthly'),
    ('Gluonts', 'm1_quarterly'),
    ('M3', 'Monthly'),
    ('M3', 'Quarterly'),
    ('Tourism', 'Monthly'),
    ('Tourism', 'Quarterly'),
]

# Read neural sampler results
neural_results = pd.read_csv('assets/results/neural_sampler_mase_results.csv')
print("Neural Sampler Results:")
print(neural_results)
print("\n")

# Aggregate results from main_results files
all_results = []

for dataset, group in datasets_groups:
    # Read the NHITS results file
    filename = f'assets/results/main_results/{dataset}-{group},NHITS.csv'
    try:
        df = pd.read_csv(filename)
        # Filter to only MASE metric
        mase_df = df[df['metric'] == 'mase']
        
        # Calculate mean MASE for each method
        methods = ['SeasonalMBB', 'Jittering', 'Scaling', 'TimeWarping', 
                   'MagnitudeWarping', 'TSMixup', 'DBA', 'original', 
                   'derived', 'derived_ensemble', 'QGTS', 'QGTSE', 'SeasonalNaive']
        
        result_row = {'dataset': dataset, 'group': group}
        for method in methods:
            if method in mase_df.columns:
                result_row[method] = mase_df[method].mean()
        
        all_results.append(result_row)
        print(f"{dataset} - {group}: MASE averages calculated")
    except Exception as e:
        print(f"Error reading {dataset} - {group}: {e}")

# Create comparison dataframe
comparison_df = pd.DataFrame(all_results)
print("\n" + "="*80)
print("ALL AUGMENTATION METHODS - MASE COMPARISON")
print("="*80)
print(comparison_df.to_string(index=False))

# Calculate overall averages
print("\n" + "="*80)
print("AVERAGE MASE ACROSS ALL 6 DATASETS:")
print("="*80)

numeric_cols = ['SeasonalMBB', 'Jittering', 'Scaling', 'TimeWarping', 
                'MagnitudeWarping', 'TSMixup', 'DBA', 'original', 
                'derived', 'derived_ensemble', 'QGTS', 'QGTSE', 'SeasonalNaive']

avg_results = comparison_df[numeric_cols].mean().sort_values()
print(avg_results)

# Save combined results
comparison_df.to_csv('assets/results/all_augmentation_methods_comparison.csv', index=False)
print("\n\nResults saved to: assets/results/all_augmentation_methods_comparison.csv")

# Combine with neural sampler results for complete view
print("\n" + "="*80)
print("COMPLETE COMPARISON INCLUDING NEURAL SAMPLER VARIANTS")
print("="*80)

# Merge neural sampler results with main results
complete_comparison = []
for idx, row in comparison_df.iterrows():
    dataset = row['dataset']
    group = row['group']
    
    # Find matching neural result
    neural_row = neural_results[(neural_results['dataset'] == dataset) & 
                                (neural_results['group'] == group)]
    
    if not neural_row.empty:
        combined = {
            'dataset': dataset,
            'group': group,
            'original': row['original'],
            'grasynda_neural': neural_row['grasynda_neural'].values[0],
            'grasynda_kde': neural_row['grasynda_kde'].values[0],
            'grasynda_uniform': neural_row['grasynda_uniform'].values[0],
            'derived': row['derived'],
            'QGTS': row['QGTS'],
            'SeasonalMBB': row['SeasonalMBB'],
            'Jittering': row['Jittering'],
            'Scaling': row['Scaling'],
            'TimeWarping': row['TimeWarping'],
            'MagnitudeWarping': row['MagnitudeWarping'],
            'TSMixup': row['TSMixup'],
            'DBA': row['DBA'],
            'SeasonalNaive': row['SeasonalNaive'],
        }
        complete_comparison.append(combined)

complete_df = pd.DataFrame(complete_comparison)
print(complete_df.to_string(index=False))

print("\n" + "="*80)
print("OVERALL AVERAGE RANKING (Lower MASE is better):")
print("="*80)
all_methods_avg = complete_df.drop(['dataset', 'group'], axis=1).mean().sort_values()
print(all_methods_avg)

complete_df.to_csv('assets/results/complete_augmentation_comparison.csv', index=False)
print("\n\nComplete results saved to: assets/results/complete_augmentation_comparison.csv")
