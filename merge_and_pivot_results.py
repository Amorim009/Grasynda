"""
Merge original and missing experiment results, then generate pivot tables.

This script:
1. Merges universal_experiment_results_raw.csv with universal_experiment_results_missing.csv
2. Saves the complete results to universal_experiment_results_complete.csv
3. Generates pivot table CSV files for each model and training mode
"""

import pandas as pd
import os

# Configuration (from run_universal_experiments.py)
FORECASTING_MODELS = ['NHITS', 'MLP', 'KAN']
TRAINING_MODES = ['Train+Real', 'TSTR']

GRASYNDA_METHODS = [
    'Grasynda_Uniform',
    'Grasynda_Vis_Horizontal',
    'Grasynda_Vis_Natural',
]

OTHER_METHODS = [
    'SeasonalMBB',
    'Jittering',
    'Scaling',
    'TimeWarping',
    'MagnitudeWarping',
    'TSMixup',
    'DBA',
]

print("=" * 100)
print("MERGING EXPERIMENT RESULTS AND GENERATING PIVOT TABLES")
print("=" * 100)

# Step 1: Load and merge results
print("\nStep 1: Loading results files...")

raw_path = 'assets/results/universal_experiment_results_raw.csv'
missing_path = 'assets/results/universal_experiment_results_missing.csv'

if not os.path.exists(raw_path):
    print(f"ERROR: {raw_path} not found!")
    exit(1)

if not os.path.exists(missing_path):
    print(f"ERROR: {missing_path} not found!")
    print("Make sure the missing experiments have completed successfully.")
    exit(1)

df_raw = pd.read_csv(raw_path)
df_missing = pd.read_csv(missing_path)

print(f"  Original results: {len(df_raw)} experiments")
print(f"  Missing results:  {len(df_missing)} experiments")

# Merge
df_complete = pd.concat([df_raw, df_missing], ignore_index=True)
print(f"  Combined results: {len(df_complete)} experiments")

# Save complete results
complete_path = 'assets/results/universal_experiment_results_complete.csv'
df_complete.to_csv(complete_path, index=False)
print(f"\n  ✅ Saved: {complete_path}")

# Step 2: Verify coverage
print("\nStep 2: Verifying dataset coverage...")

expected_datasets = [
    ('Gluonts', 'm1_monthly'),
    ('Gluonts', 'm1_quarterly'),
    ('M3', 'Monthly'),
    ('M3', 'Quarterly'),
    ('Tourism', 'Monthly'),
    ('Tourism', 'Quarterly'),
]

all_good = True
for dataset, group in expected_datasets:
    count = len(df_complete[(df_complete['Dataset'] == dataset) & (df_complete['Group'] == group)])
    expected = 60  # 3 models × 10 methods × 2 modes
    status = "✅" if count == expected else f"❌ ({count}/{expected})"
    print(f"  {dataset:8} {group:10}: {status}")
    if count != expected:
        all_good = False

total_expected = 6 * 60  # 360
print(f"\n  Total: {len(df_complete)}/{total_expected} experiments")

if not all_good:
    print("\n  ⚠️  WARNING: Some datasets are incomplete!")

# Step 3: Generate pivot tables
print("\nStep 3: Generating pivot tables (Methods as Columns, Datasets as Rows)...")

pivot_files_created = []

for model_name in FORECASTING_MODELS:
    for mode in TRAINING_MODES:
        # Filter
        filtered = df_complete[
            (df_complete['Forecasting_Model'] == model_name) &
            (df_complete['Training_Mode'] == mode)
        ]
        
        if len(filtered) == 0:
            print(f"  ⚠️  No data for {model_name} - {mode}")
            continue
        
        # Dataset
        filtered['Dataset_Full'] = filtered['Dataset'] + ' - ' + filtered['Group']
        
        # Pivot
        pivot = filtered.pivot_table(
            index='Dataset_Full',
            columns='Augmentation_Method',
            values='MASE',
            aggfunc='first'  # Should be only one value per combination
        )
        
        # Reorder columns: Grasynda methods first, then others
        grasynda_cols = [c for c in pivot.columns if c in GRASYNDA_METHODS]
        other_cols = [c for c in pivot.columns if c in OTHER_METHODS]
        ordered_cols = grasynda_cols + other_cols
        pivot = pivot[ordered_cols]
        
        # Save
        filename = f'assets/results/{model_name}_{mode}_Results.csv'
        pivot.to_csv(filename)
        pivot_files_created.append(filename)
        print(f"  ✅ {model_name}_{mode}_Results.csv")

# Summary
print("\n" + "=" * 100)
print("COMPLETE!")
print("=" * 100)
print(f"\nFiles created:")
print(f"  1. {complete_path}")
for i, file in enumerate(pivot_files_created, 2):
    print(f"  {i}. {file}")

print("\n" + "=" * 100)
