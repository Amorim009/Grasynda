"""
Grasynda Synthesis Evaluation using PyMDMA TIME SERIES Metrics

This script evaluates synthetic time series against real ones using:
  - PyMDMA Time Series Measures: DTW, SpectralCoherence, Authenticity, etc.
  
KEY DIFFERENCE FROM run_pymdma_eval.py:
  - This evaluates RAW TIME SERIES DATA, not extracted features
  - More appropriate for evaluating synthetic time series quality
  - Measures like DTW, SpectralCoherence work on (N, L) arrays where:
    * N = number of series
    * L = length of each series

PyMDMA Time Series Measures:
  - DTW (Dynamic Time Warping): Distance between time series
  - SpectralCoherence: Frequency-domain similarity
  - SpectralWassersteinDistance: Distribution of spectral properties
  - CrossCorrelation: Temporal correlation patterns
  - Authenticity: How realistic the synthetic data is
  - Density: Coverage of the feature space
  - Coverage: How well synthetic data covers real data

Usage:
    python run_pymdma_ts_eval.py

Outputs:
    - assets/results/pymdma_metrics/ts_<dataset>_pymdma_results.csv
    - assets/results/pymdma_metrics/OVERALL_TS_PYMDMA_RESULTS.json
"""

import os
import glob
import argparse
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / 'src'))

OUT_DIR = "assets/results/pymdma_metrics"
os.makedirs(OUT_DIR, exist_ok=True)

try:
    from pymdma.time_series.measures.synthesis_val import (
        DTW, SpectralCoherence, SpectralWassersteinDistance,
        CrossCorrelation, Authenticity, Density, Coverage
    )
    HAVE_PYMDMA_TS = True
    print("[INFO] PyMDMA time series measures successfully imported")
except Exception as e:
    HAVE_PYMDMA_TS = False
    print(f"[WARNING] PyMDMA time series import failed: {e}")


def df_to_array(df: pd.DataFrame) -> np.ndarray:
    """Convert long dataframe to array shape (n_series, series_length)."""
    pivot = df.pivot(index="ds", columns="unique_id", values="y")
    pivot = pivot.ffill().bfill()
    return pivot.T.values


def df_to_list(df: pd.DataFrame) -> list:
    """Convert long dataframe to list of time series (for some PyMDMA measures)."""
    time_series_list = []
    for unique_id, group in df.groupby('unique_id'):
        group = group.sort_values('ds')
        ts_values = group['y'].values.reshape(-1, 1)  # Shape (L, 1) for single channel
        time_series_list.append(ts_values)
    return time_series_list


def compute_ts_metrics(real_data: np.ndarray, synth_data: np.ndarray) -> dict:
    """
    Compute PyMDMA time series metrics.
    
    Args:
        real_data: shape (n_series, series_length) - real time series array
        synth_data: shape (n_series, series_length) - synthetic time series array
    
    Returns:
        dict with metric names and values
    """
    if not HAVE_PYMDMA_TS:
        raise RuntimeError("PyMDMA time series measures not available")

    results = {}
    
    print(f"    Real data: {real_data.shape}, Synth data: {synth_data.shape}")
    
    # DTW requires list format
    try:
        real_list = df_to_list(pd.DataFrame({
            'unique_id': np.repeat(np.arange(real_data.shape[0]), real_data.shape[1]),
            'ds': np.tile(np.arange(real_data.shape[1]), real_data.shape[0]),
            'y': real_data.flatten()
        }))
        synth_list = df_to_list(pd.DataFrame({
            'unique_id': np.repeat(np.arange(synth_data.shape[0]), synth_data.shape[1]),
            'ds': np.tile(np.arange(synth_data.shape[1]), synth_data.shape[0]),
            'y': synth_data.flatten()
        }))
        
        dtw = DTW()
        dtw_result = dtw.compute(real_list, synth_list)
        results['DTW'] = float(dtw_result.value[0]) if isinstance(dtw_result.value, tuple) else float(dtw_result.value)
        print(f"      DTW: {results['DTW']:.4f}")
    except Exception as e:
        print(f"      DTW computation failed: {e}")
        results['DTW'] = np.nan
    
    # SpectralCoherence works on 2D arrays
    try:
        sc = SpectralCoherence()
        sc_result = sc.compute(real_data, synth_data)
        results['SpectralCoherence'] = float(sc_result.value[0]) if isinstance(sc_result.value, tuple) else float(sc_result.value)
        print(f"      SpectralCoherence: {results['SpectralCoherence']:.4f}")
    except Exception as e:
        print(f"      SpectralCoherence computation failed: {e}")
        results['SpectralCoherence'] = np.nan
    
    # CrossCorrelation
    try:
        cc = CrossCorrelation()
        cc_result = cc.compute(real_data, synth_data)
        results['CrossCorrelation'] = float(cc_result.value[0]) if isinstance(cc_result.value, tuple) else float(cc_result.value)
        print(f"      CrossCorrelation: {results['CrossCorrelation']:.4f}")
    except Exception as e:
        print(f"      CrossCorrelation computation failed: {e}")
        results['CrossCorrelation'] = np.nan
    
    # Authenticity (how realistic synthetic is)
    try:
        auth = Authenticity()
        auth_result = auth.compute(real_data, synth_data)
        results['Authenticity'] = float(auth_result.value[0]) if isinstance(auth_result.value, tuple) else float(auth_result.value)
        print(f"      Authenticity: {results['Authenticity']:.4f}")
    except Exception as e:
        print(f"      Authenticity computation failed: {e}")
        results['Authenticity'] = np.nan
    
    return results


def evaluate_all():
    """Evaluate all synthetic datasets using pymdma time series metrics."""
    TRAIN_DIR = "assets/results/training_sets"

    original_files = sorted([f for f in os.listdir(TRAIN_DIR) if f.endswith('_original.csv')])
    if not original_files:
        print('[ERROR] No original files found')
        return

    all_results = {}
    for orig in original_files:
        base = orig.replace('_original.csv', '')
        print(f"\n{'='*70}")
        print(f"Dataset: {base} (TIME SERIES EVALUATION)")
        print(f"{'='*70}")
        
        real_df = pd.read_csv(os.path.join(TRAIN_DIR, orig))
        real_data = df_to_array(real_df)
        print(f"[Real] Data shape: {real_data.shape} (n_series={real_data.shape[0]}, series_len={real_data.shape[1]})")

        # Find and evaluate synthetic files
        synth_files = sorted([f for f in os.listdir(TRAIN_DIR) 
                             if f.startswith(f"{base}_") and not f.endswith('_original.csv')])
        results = {}
        
        for f in synth_files:
            name = f.replace(f"{base}_", '').replace('.csv', '')
            synth_df = pd.read_csv(os.path.join(TRAIN_DIR, f))
            synth_data = df_to_array(synth_df)

            print(f"\n  [{name:20s}]")

            # Compute time series metrics
            if HAVE_PYMDMA_TS:
                metrics = compute_ts_metrics(real_data, synth_data)
            else:
                print("[ERROR] PyMDMA time series measures not available")
                continue

            results[name] = {
                'DTW': round(float(metrics['DTW']), 4) if not np.isnan(metrics['DTW']) else None,
                'SpectralCoherence': round(float(metrics['SpectralCoherence']), 4) if not np.isnan(metrics['SpectralCoherence']) else None,
                'CrossCorrelation': round(float(metrics['CrossCorrelation']), 4) if not np.isnan(metrics['CrossCorrelation']) else None,
                'Authenticity': round(float(metrics['Authenticity']), 4) if not np.isnan(metrics['Authenticity']) else None,
            }

        # Save per-dataset results
        if results:
            out_csv = os.path.join(OUT_DIR, f"ts_{base}_pymdma_results.csv")
            pd.DataFrame.from_dict(results, orient='index').to_csv(out_csv)
            all_results[base] = results
            print(f"\n  ✓ Results saved to {out_csv}")

    # Save aggregate results
    import json
    agg_path = os.path.join(OUT_DIR, 'OVERALL_TS_PYMDMA_RESULTS.json')
    with open(agg_path, 'w') as fh:
        json.dump(all_results, fh, indent=2)
    
    print(f"\n{'='*70}")
    print("✓ Time Series Evaluation Complete")
    print(f"  - Results: {agg_path}")
    print(f"  - Per-dataset: {OUT_DIR}/ts_<dataset>_pymdma_results.csv")
    print(f"\nMetrics Explanation:")
    print(f"  DTW:                  Distance between time series (lower = more similar)")
    print(f"  SpectralCoherence:    Frequency-domain similarity (higher = more similar)")
    print(f"  CrossCorrelation:     Temporal patterns similarity (higher = better)")
    print(f"  Authenticity:         How realistic synthetic data is (higher = better)")
    print(f"{'='*70}")


if __name__ == '__main__':
    evaluate_all()
