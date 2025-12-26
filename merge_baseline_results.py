"""
Merge baseline results with augmentation results and create final consolidated Excel file.
"""

import pandas as pd
import numpy as np

print("="*80)
print("MERGING BASELINE RESULTS WITH AUGMENTATION RESULTS")
print("="*80)

# Load augmentation results
print("\n1. Loading augmentation results...")
df_aug = pd.read_csv('assets/results/universal_experiments/universal_experiment_results_complete.csv')
print(f"   Augmentation experiments: {len(df_aug)}")

# Load baseline results
print("\n2. Loading baseline results...")
df_baseline = pd.read_csv('assets/results/universal_experiment_results_missing.csv')
print(f"   Baseline experiments: {len(df_baseline)}")
print(f"   Baseline methods: {df_baseline['Augmentation_Method'].unique()}")
print(f"   Baseline status: {df_baseline['Status'].value_counts().to_dict()}")

# Combine
print("\n3. Merging datasets...")
df_all = pd.concat([df_aug, df_baseline], ignore_index=True)
print(f"   Total experiments: {len(df_all)}")
print(f"   Total methods: {df_all['Augmentation_Method'].nunique()}")

# Save combined CSV
output_csv = 'assets/results/universal_experiments/ALL_RESULTS_COMPLETE.csv'
df_all.to_csv(output_csv, index=False)
print(f"\n✓ Saved: {output_csv}")

# Create consolidated Excel with baseline
print("\n4. Creating comprehensive Excel file...")
output_excel = 'assets/results/Universal_Experiments_COMPLETE.xlsx'

# Create combined columns
df_all['Dataset_Full'] = df_all['Dataset'] + ' - ' + df_all['Group']

with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
    
    # Sheet 1: All Results
    print("   Sheet 1: All Results (with Baseline)...")
    
    # Prepare data for Excel: Treat Baseline as a method available in both modes
    # 1. Get non-baseline results
    df_aug_only = df_all[df_all['Augmentation_Method'] != 'Baseline'].copy()
    
    # 2. Get baseline results and duplicate for both modes
    df_baseline = df_all[df_all['Augmentation_Method'] == 'Baseline'].copy()
    
    # Baseline for Train+Real
    df_baseline_tr = df_baseline.copy()
    df_baseline_tr['Training_Mode'] = 'Train+Real'
    
    # Baseline for TSTR
    df_baseline_tstr = df_baseline.copy()
    df_baseline_tstr['Training_Mode'] = 'TSTR'
    
    # 3. Combine everything for Excel
    df_excel = pd.concat([df_aug_only, df_baseline_tr, df_baseline_tstr], ignore_index=True)
    
    all_results = df_excel[['Dataset_Full', 'Augmentation_Method', 'Forecasting_Model', 
                           'Training_Mode', 'MASE', 'Train_Size', 'Test_Size', 'Status']].copy()
    all_results = all_results.sort_values(['Dataset_Full', 'Forecasting_Model', 'Training_Mode', 'Augmentation_Method'])
    all_results['MASE'] = all_results['MASE'].round(6)
    all_results.to_excel(writer, sheet_name='All Results', index=False)
    
    # Sheets 2-7: By Forecasting Architecture
    for model in ['NHITS', 'MLP', 'KAN']:
        print(f"   Sheets: {model} Results...")
        model_df = df_excel[df_excel['Forecasting_Model'] == model]
        
        for mode in ['TSTR', 'Train+Real']:
            mode_df = model_df[model_df['Training_Mode'] == mode]
            if len(mode_df) == 0:
                continue
                
            pivot = mode_df.pivot_table(
                values='MASE',
                index=['Dataset_Full'],
                columns=['Augmentation_Method'],
                aggfunc='mean'
            ).round(4)
            
            # Reorder columns: Baseline first, then Grasynda, then others
            cols = list(pivot.columns)
            baseline_cols = [c for c in cols if 'Baseline' in c]
            grasynda_cols = sorted([c for c in cols if 'Grasynda' in c])
            other_cols = sorted([c for c in cols if 'Baseline' not in c and 'Grasynda' not in c])
            pivot = pivot[baseline_cols + grasynda_cols + other_cols]
            
            sheet_name = f"{model}_{mode}"
            pivot.to_excel(writer, sheet_name=sheet_name)
    
    # NEW SHEET: Method Rankings by Mean MASE for each Architecture
    print("   Sheet: Method Rankings by Architecture...")
    rankings_by_arch = []
    
    for model in ['NHITS', 'MLP', 'KAN']:
        model_df = df_all[df_all['Forecasting_Model'] == model].copy()
        
        # Calculate mean MASE for each method across all datasets
        method_means = model_df.groupby('Augmentation_Method')['MASE'].agg(['mean', 'std', 'count']).reset_index()
        method_means.columns = ['Method', 'Mean_MASE', 'Std_MASE', 'N_Datasets']
        method_means = method_means.sort_values('Mean_MASE')
        method_means['Rank'] = range(1, len(method_means) + 1)
        method_means['Architecture'] = model
        method_means = method_means[['Architecture', 'Rank', 'Method', 'Mean_MASE', 'Std_MASE', 'N_Datasets']]
        method_means = method_means.round(4)
        
        rankings_by_arch.append(method_means)
    
    # Combine all rankings
    all_rankings = pd.concat(rankings_by_arch, ignore_index=True)
    all_rankings.to_excel(writer, sheet_name='Rankings by Architecture', index=False)
    
    # Sheet: Overall Method Rankings (average across all architectures)
    print("   Sheet: Overall Method Rankings...")
    overall_ranks = df_all.groupby('Augmentation_Method')['MASE'].agg(['mean', 'std', 'count']).reset_index()
    overall_ranks.columns = ['Method', 'Mean_MASE', 'Std_MASE', 'N_Experiments']
    overall_ranks = overall_ranks.sort_values('Mean_MASE')
    overall_ranks['Rank'] = range(1, len(overall_ranks) + 1)
    overall_ranks = overall_ranks[['Rank', 'Method', 'Mean_MASE', 'Std_MASE', 'N_Experiments']]
    overall_ranks = overall_ranks.round(4)
    overall_ranks.to_excel(writer, sheet_name='Overall Rankings', index=False)
    
    # Sheet: Top 3 per Dataset/Model
    print("   Sheet: Top 3 per Dataset/Model...")
    top3_results = []
    for (dataset, model), group in df_all[df_all['Status'] == 'Success'].groupby(['Dataset_Full', 'Forecasting_Model']):
        top3 = group.nsmallest(3, 'MASE')[['Augmentation_Method', 'Training_Mode', 'MASE']].copy()
        top3['Dataset'] = dataset
        top3['Model'] = model
        top3['Rank'] = [1, 2, 3]
        top3_results.append(top3)
    
    top3_df = pd.concat(top3_results, ignore_index=True)
    top3_df = top3_df[['Dataset', 'Model', 'Rank', 'Augmentation_Method', 'Training_Mode', 'MASE']]
    top3_df['MASE'] = top3_df['MASE'].round(4)
    top3_df.to_excel(writer, sheet_name='Top 3 per Dataset', index=False)
    
    # Sheet: Method Summary Statistics
    print("   Sheet: Method Summary...")
    method_stats = df_all[df_all['Status'] == 'Success'].groupby('Augmentation_Method').agg({
        'MASE': ['mean', 'std', 'min', 'max', 'count'],
    }).round(4)
    method_stats.columns = ['Mean_MASE', 'Std_MASE', 'Min_MASE', 'Max_MASE', 'N_Experiments']
    method_stats = method_stats.sort_values('Mean_MASE')
    method_stats.to_excel(writer, sheet_name='Method Summary')
    
    # Sheet: Baseline vs Best Augmentation
    print("   Sheet: Baseline Comparison...")
    # Get baseline results
    baseline_results = df_all[df_all['Augmentation_Method'] == 'Baseline'].copy()
    baseline_results = baseline_results.pivot_table(
        values='MASE',
        index=['Dataset_Full'],
        columns=['Forecasting_Model'],
        aggfunc='mean'
    ).round(4)
    
    # Get best augmentation per dataset/model
    aug_results = df_all[df_all['Augmentation_Method'] != 'Baseline'].copy()
    best_aug = aug_results.loc[aug_results.groupby(['Dataset_Full', 'Forecasting_Model'])['MASE'].idxmin()]
    best_aug_pivot = best_aug.pivot_table(
        values='MASE',
        index=['Dataset_Full'],
        columns=['Forecasting_Model'],
        aggfunc='mean'  
    ).round(4)
    
    # Combine
    comparison = pd.DataFrame({
        ('NHITS', 'Baseline'): baseline_results.get('NHITS', np.nan),
        ('NHITS', 'Best Augmentation'): best_aug_pivot.get('NHITS', np.nan),
        ('MLP', 'Baseline'): baseline_results.get('MLP', np.nan),
        ('MLP', 'Best Augmentation'): best_aug_pivot.get('MLP', np.nan),
        ('KAN', 'Baseline'): baseline_results.get('KAN', np.nan),
        ('KAN', 'Best Augmentation'): best_aug_pivot.get('KAN', np.nan),
    })
    comparison.to_excel(writer, sheet_name='Baseline vs Best Aug')

print(f"\n✓ Created: {output_excel}")
print("\n" + "="*80)
print("COMPLETE!")
print("="*80)
print(f"\nFinal dataset:")
print(f"  Total experiments: {len(df_all)}")
print(f"  - Baseline: {len(df_baseline)}")
print(f"  - Augmentation: {len(df_aug)}")
print(f"  Methods: {df_all['Augmentation_Method'].nunique()}")
print(f"  Datasets: {df_all['Dataset'].nunique()}")
print(f"  Models: {df_all['Forecasting_Model'].nunique()}")
print("\nFiles:")
print(f"  1. {output_csv}")
print(f"  2. {output_excel}")
print("="*80)
