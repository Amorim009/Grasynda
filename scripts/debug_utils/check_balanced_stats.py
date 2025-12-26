
import numpy as np
import pandas as pd
from utils.load_data.config import DATASETS
from src.grasynda_unified import GrasyndaUnified
from scripts.debug_utils.demo_balanced_input import create_balanced_input

# Load Data
data_loader = DATASETS['M3']
df, _, _, _, freq_int = data_loader.load_everything('Monthly')
df_series = df[df['unique_id'] == 'M806'].copy()

# Configure Model
config = {
    'period': freq_int,
    'n_quantiles': 25,
    'components_to_model': ['trend', 'remainder'],
    'component_params': {
        'trend': {'sampling_type': 'continuous_uniform', 'apply_differentiation': True},
        'remainder': {'sampling_type': 'discrete'}
    }
}
model = GrasyndaUnified(**config)

# Create balanced input
df_balanced = create_balanced_input(df_series, freq_int)

# Generate
print("Generating baseline...")
synth_base = model.transform(df_series)

print("Generating balanced...")
synth_balanced = model.transform(df_balanced)

# Extract
base_ts = synth_base[synth_base['unique_id'] == 'GrasyndaUnified_M806']
bal_ts = synth_balanced[synth_balanced['unique_id'] == 'GrasyndaUnified_M806']

# Print stats
print("\n=== DETAILED STATISTICS ===")
print(f"Original: min={df_series['y'].min():.2f}, max={df_series['y'].max():.2f}, mean={df_series['y'].mean():.2f}")
print(f"Balanced Input: min={df_balanced['y'].min():.2f}, max={df_balanced['y'].max():.2f}, mean={df_balanced['y'].mean():.2f}")
print(f"Baseline Synth: min={base_ts['y'].min():.2f}, max={base_ts['y'].max():.2f}, mean={base_ts['y'].mean():.2f}, len={len(base_ts)}")
print(f"Balanced Synth: min={bal_ts['y'].min():.2f}, max={bal_ts['y'].max():.2f}, mean={bal_ts['y'].mean():.2f}, len={len(bal_ts)}")

print(f"\nBalanced Synth first 10: {bal_ts['y'].head(10).values}")
print(f"Balanced Synth last 10: {bal_ts['y'].tail(10).values}")

# Check if they're the same
if len(bal_ts) > 0 and len(base_ts) > 0:
    diff = np.abs(bal_ts['y'].values - base_ts['y'].values).sum()
    print(f"\nDifference between Baseline and Balanced: {diff:.2f}")
    if diff < 1:
        print("WARNING: Baseline and Balanced are nearly IDENTICAL!")
