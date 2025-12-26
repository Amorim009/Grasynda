"""
Create a comprehensive consolidated results file from universal experiments.

This script creates a single Excel file with multiple sheets showing:
1. All Results - Complete raw data table WITH baseline (no augmentation)
2. By Architecture - Separate sheets for NHITS, MLP, KAN
3. Summary Stats - Summary statistics
"""

import pandas as pd
import numpy as np
import os

# Load the complete augmentation results
print("Loading augmentation results...")
df_aug = pd.read_csv('assets/results/universal_experiment_results_complete.csv')

print(f"Augmentation experiments: {len(df_aug)}")
print(f"All successful: {(df_aug['Status'] == 'Success').all()}")

# Load baseline results (no augmentation)
print("\nLoading baseline results (no augmentation)...")
baseline_files = [
    'assets/results/universal_experiment_results_raw.csv',
    'assets/results/universal_experiment_results_missing.csv'
]

# Try to find baseline results in the experiment files
# The baseline should be in the raw CSV with Method='None' or similar
# Let's check what we have
baseline_rows = []

# Check if there's a separate baseline file or if it's embedded
print("Searching for baseline (no augmentation) results...")
for dataset in ['Gluonts', 'M3', 'Tourism']:
    for group in df_aug[df_aug['Dataset'] == dataset]['Group'].unique():
        for model in ['NHITS', 'MLP', 'KAN']:
            for mode in ['TSTR', 'Train+Real']:
                # For now, mark baseline as "None" - you'll need to run baseline experiments
                baseline_rows.append({
                    'Dataset': dataset,
                    'Group': group,
                    'Augmentation_Method': 'Baseline (No Augmentation)',
                    'Forecasting_Model': model,
                    'Training_Mode': mode,
                    'MASE': np.nan,  # Will need actual baseline results
                    'Train_Size': np.nan,
                    'Test_Size': np.nan,
                    'Status': 'MISSING - Need to run baseline'
                })

df_baseline = pd.DataFrame(baseline_rows)

# Combine augmentation and baseline
print(f"\nCombining {len(df_aug)} augmentation + {len(df_baseline)} baseline experiments...")
df = pd.concat([df_baseline, df_aug], ignore_index=True)

# Create combined columns for better readability
df['Dataset_Full'] = df['Dataset'] + ' - ' + df['Group']
df['Model_Mode'] = df['Forecasting_Model'] + ' (' + df['Training_Mode'] + ')'

# Create Excel writer
output_file = 'assets/results/Universal_Experiments_COMPLETE.xlsx'
print(f"\nCreating {output_file}...")

with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    
    # Sheet 1: All Results (sorted and cleaned)
    print("  Sheet 1: All Results (with Baseline)...")
    all_results = df[['Dataset_Full', 'Augmentation_Method', 'Forecasting_Model', 
                      'Training_Mode', 'MASE', 'Train_Size', 'Test_Size', 'Status']].copy()
    all_results = all_results.sort_values(['Dataset_Full', 'Forecasting_Model', 'Training_Mode', 'Augmentation_Method'])
    all_results['MASE'] = all_results['MASE'].round(6)
    all_results.to_excel(writer, sheet_name='All Results', index=False)
    
    # Sheets 2-4: By Forecasting Architecture (NHITS, MLP, KAN)
    for model in ['NHITS', 'MLP', 'KAN']:
        print(f"  Sheet: {model} Results...")
        model_df = df[df['Forecasting_Model'] == model]
        
        # Create pivot: Datasets as rows, Methods as columns, split by Training Mode
        for mode in ['TSTR', 'Train+Real']:
            mode_df = model_df[model_df['Training_Mode'] == mode]
            pivot = mode_df.pivot_table(
                values='MASE',
                index=['Dataset_Full'],
                columns=['Augmentation_Method'],
                aggfunc='mean'
            ).round(4)
            
            sheet_name = f"{model}_{mode}"
            pivot.to_excel(writer, sheet_name=sheet_name)
    
    # Sheet: Method Rankings (Top 3 per combination)
    print("  Sheet: Method Rankings...")
    rankings = df[df['Status'] == 'Success'].groupby(['Dataset_Full', 'Forecasting_Model', 'Training_Mode']).apply(
        lambda x: x.nsmallest(3, 'MASE')[['Augmentation_Method', 'MASE']]
    ).reset_index()
    rankings['MASE'] = rankings['MASE'].round(4)
    rankings.to_excel(writer, sheet_name='Method Rankings', index=False)
    
    # Sheet: Method Summary Statistics
    print("  Sheet: Method Summary...")
    method_stats = df[df['Status'] == 'Success'].groupby('Augmentation_Method').agg({
        'MASE': ['mean', 'std', 'min', 'max', 'count'],
    }).round(4)
    method_stats.columns = ['Mean_MASE', 'Std_MASE', 'Min_MASE', 'Max_MASE', 'N_Experiments']
    method_stats = method_stats.sort_values('Mean_MASE')
    method_stats.to_excel(writer, sheet_name='Method Summary')
    
    # Sheet: Grasynda Comparison Only
    print("  Sheet: Grasynda Comparison...")
    grasynda_df = df[df['Augmentation_Method'].str.contains('Grasynda|Baseline')]
    grasynda_pivot = grasynda_df.pivot_table(
        values='MASE',
        index=['Dataset_Full'],
        columns=['Augmentation_Method', 'Forecasting_Model', 'Training_Mode'],
        aggfunc='mean'
    ).round(4)
    grasynda_pivot.to_excel(writer, sheet_name='Grasynda vs Baseline')

print(f"\n✅ Created: {output_file}")
print(f"\nSheets included:")
print("  1. All Results - Complete dataset with BASELINE placeholders")
print("  2-7. By Architecture - NHITS/MLP/KAN × TSTR/Train+Real (6 sheets)")
print("  8. Method Rankings - Top 3 methods for each combination")
print("  9. Method Summary - Overall performance statistics")
print("  10. Grasynda vs Baseline - Comparison with baseline")
print(f"\n⚠️  NOTE: Baseline results are marked as MISSING - you need to run baseline experiments!")
print(f"\n📂 To consult: {output_file}")
