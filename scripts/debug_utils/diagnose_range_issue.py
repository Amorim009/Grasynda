
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL
from utils.load_data.config import DATASETS
from src.grasynda_unified import GrasyndaUnified

def analyze_range_shift(uid, df_series, period):
    """Analyze why balanced series has different range."""
    series = df_series['y'].values
    
    if len(series) < 2 * period:
        return None
    
    # Decompose
    res = STL(series, period=period, robust=False).fit()
    trend = res.trend
    seasonal = res.seasonal
    remainder = res.resid
    
    # Component stats
    stats = {
        'uid': uid,
        'original_mean': series.mean(),
        'original_std': series.std(),
        'original_range': series.max() - series.min(),
        'trend_mean': trend.mean(),
        'trend_std': np.std(trend),
        'seasonal_mean': seasonal.mean(),
        'seasonal_std': np.std(seasonal),
        'remainder_mean': remainder.mean(),
        'remainder_std': np.std(remainder),
    }
    
    # Balancing calculation
    std_t = np.std(trend)
    std_s = np.std(seasonal)
    target_std = (std_t + std_s) / 2
    
    scale_t = target_std / std_t if std_t > 0 else 1.0
    scale_s = target_std / std_s if std_s > 0 else 1.0
    
    stats['scale_t'] = scale_t
    stats['scale_s'] = scale_s
    
    # What happens to trend mean when scaled?
    trend_balanced = trend * scale_t
    seasonal_balanced = seasonal * scale_s
    
    stats['trend_balanced_mean'] = trend_balanced.mean()
    stats['seasonal_balanced_mean'] = seasonal_balanced.mean()
    
    # Predicted balanced series stats
    y_balanced = trend_balanced + seasonal_balanced + remainder
    stats['balanced_mean'] = y_balanced.mean()
    stats['balanced_std'] = y_balanced.std()
    stats['balanced_range'] = y_balanced.max() - y_balanced.min()
    
    # Range shift
    stats['mean_shift'] = stats['balanced_mean'] - stats['original_mean']
    stats['range_ratio'] = stats['balanced_range'] / stats['original_range']
    
    return stats

# Load data
data_loader = DATASETS['M3']
df, _, _, _, freq_int = data_loader.load_everything('Monthly')

# Analyze multiple series
all_uids = df['unique_id'].unique()[:20]  # First 20 series
results = []

print("Analyzing component scaling effects...\n")
for uid in all_uids:
    df_series = df[df['unique_id'] == uid].copy()
    stats = analyze_range_shift(uid, df_series, freq_int)
    if stats:
        results.append(stats)

# Convert to DataFrame for easy analysis
results_df = pd.DataFrame(results)

print("="*80)
print("SERIES WITH LARGE RANGE SHIFTS (>50% change)")
print("="*80)
large_shifts = results_df[abs(results_df['range_ratio'] - 1.0) > 0.5]
print(large_shifts[['uid', 'original_range', 'balanced_range', 'range_ratio', 'mean_shift', 'trend_mean', 'scale_t']].to_string())

print("\n" + "="*80)
print("SERIES WITH LARGE MEAN SHIFTS (>10% of original mean)")
print("="*80)
large_mean_shifts = results_df[abs(results_df['mean_shift'] / results_df['original_mean']) > 0.1]
print(large_mean_shifts[['uid', 'original_mean', 'balanced_mean', 'mean_shift', 'trend_mean', 'scale_t']].to_string())

print("\n" + "="*80)
print("ROOT CAUSE ANALYSIS")
print("="*80)
print("\nWhen Trend has a non-zero mean and we scale it:")
print("  Original Trend: T (mean=μ, std=σ)")
print("  Scaled Trend: T' = T * scale_factor")
print("  New Mean: μ' = μ * scale_factor")
print("\nIf scale_factor > 1 (boosting trend), the mean INCREASES.")
print("If scale_factor < 1 (reducing trend), the mean DECREASES.")
print("\nThis shifts the entire series up or down!")

print("\n" + "="*80)
print("SOLUTION: Center components before scaling, then add back the mean.")
print("="*80)
