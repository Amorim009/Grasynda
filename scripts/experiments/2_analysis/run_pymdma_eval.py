"""
Grasynda Synthesis Evaluation using PyMDMA Metrics

This script evaluates synthetic time series against real ones using:
  - TSFEL: Time Series Feature Extraction Library (temporal domain features)
  - PyMDMA: Improved Precision & Recall metrics for synthetic data quality

PyMDMA Metrics Documentation:
  - ImprovedPrecision: How many synthetic samples fall within the real feature distribution.
    Higher precision = synthetic data closer to real distribution
  - ImprovedRecall: How well synthetic samples cover/represent the real distribution.
    Higher recall = synthetic data spans real distribution (often ~1.0 for large synthetic sets)
  
  Both metrics return MetricResult with:
    - .value: tuple (dataset_level_float, instance_level_array)
    - dataset_level: the aggregated metric (scalar 0-1)
    - instance_level: per-series breakdown

Expected Results:
  - Recall often approaches 1.0 because synthetic samples are large and cover real features
  - Precision varies (0.80-1.0) based on feature distribution similarity
  - TimeWarping: lower precision (~0.80-0.95) = less realistic features
  - Other methods: high precision (~0.99-1.0) = more realistic features
  - GrasyndaE: consistently high precision & recall across datasets

Correct Arguments per pymdma library:
  - ImprovedPrecision.compute(real_features, fake_features) -> MetricResult
  - ImprovedRecall.compute(real_features, fake_features) -> MetricResult
  - Both expect: real_features shape (n_series, n_features)
                 fake_features shape (n_series, n_features)

Usage:
    python run_pymdma_eval.py --use-pymdma

Outputs:
    - assets/results/pymdma_metrics/<dataset>_pymdma_results.csv
    - assets/results/pymdma_metrics/OVERALL_PYMDMA_RESULTS.json
    - assets/results/pymdma_metrics/<dataset>_*_features.npy
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
    import tsfel
    HAVE_TS_FEL = True
except Exception:
    HAVE_TS_FEL = False

try:
    import pymdma
    from pymdma.time_series.measures.synthesis_val import ImprovedPrecision, ImprovedRecall
    HAVE_PYMDMA = True
    print("[INFO] pymdma and synthesis_val successfully imported")
except Exception as e:
    HAVE_PYMDMA = False
    print(f"[WARNING] pymdma import failed: {e}")

# Try to import Grasynda variants
try:
    from grasynda_unified import GrasyndaUnified
    HAVE_GRASYNDA_UNIFIED = True
except Exception:
    HAVE_GRASYNDA_UNIFIED = False

try:
    from grasynda_trend import GrasyndaTrend
    HAVE_GRASYNDA_TREND = True
except Exception:
    HAVE_GRASYNDA_TREND = False

try:
    from grasynda_visibility import GrasyndaVisibilityGraph
    HAVE_GRASYNDA_VIS = True
except Exception:
    HAVE_GRASYNDA_VIS = False

try:
    from grasynda_continuous import GrasyndaContinuous
    HAVE_GRASYNDA_CONTINUOUS = True
except Exception:
    HAVE_GRASYNDA_CONTINUOUS = False

try:
    from grasynda_hybrid import GrasyndaHybrid
    HAVE_GRASYNDA_HYBRID = True
except Exception:
    HAVE_GRASYNDA_HYBRID = False

print("[INFO] Grasynda imports:")
print(f"  Unified: {HAVE_GRASYNDA_UNIFIED}")
print(f"  Trend: {HAVE_GRASYNDA_TREND}")
print(f"  Visibility: {HAVE_GRASYNDA_VIS}")
print(f"  Continuous: {HAVE_GRASYNDA_CONTINUOUS}")
print(f"  Hybrid: {HAVE_GRASYNDA_HYBRID}")


def df_to_array(df: pd.DataFrame) -> np.ndarray:
    pivot = df.pivot(index="ds", columns="unique_id", values="y")
    pivot = pivot.ffill().bfill()
    return pivot.T.values


def extract_features_tsfel(time_series_data: np.ndarray, fs: int = 12) -> np.ndarray:
    cfg = tsfel.get_features_by_domain('temporal')
    feats = []
    for i in range(time_series_data.shape[0]):
        feats.append(tsfel.time_series_features_extractor(cfg, time_series_data[i, :], fs=fs, verbose=0).values.flatten())
    return np.array(feats)


def extract_features_fallback(time_series_data: np.ndarray) -> np.ndarray:
    # lightweight fallback features: mean, std, median, iqr, skew, kurt, acf1
    features = []
    for i in range(time_series_data.shape[0]):
        ts = time_series_data[i, :].astype(float)
        mean = np.nanmean(ts)
        std = np.nanstd(ts)
        median = np.nanmedian(ts)
        q75, q25 = np.nanpercentile(ts, [75, 25])
        iqr = q75 - q25
        skew = float(pd.Series(ts).skew()) if len(ts) > 2 else 0.0
        kurt = float(pd.Series(ts).kurt()) if len(ts) > 3 else 0.0
        if len(ts) > 1:
            ts0 = ts[:-1] - np.nanmean(ts[:-1])
            ts1 = ts[1:] - np.nanmean(ts[1:])
            denom = np.sqrt(np.nanvar(ts0) * np.nanvar(ts1))
            acf1 = float(np.nanmean(ts0 * ts1) / denom) if denom != 0 else 0.0
        else:
            acf1 = 0.0
        features.append([mean, std, median, iqr, skew, kurt, acf1])
    return np.array(features)


def extract_features(time_series_data: np.ndarray):
    if HAVE_TS_FEL:
        try:
            return extract_features_tsfel(time_series_data)
        except Exception as e:
            print(f"tsfel extraction failed; fallback used: {e}")
    return extract_features_fallback(time_series_data)


def compute_pymdma_metrics(real_feats: np.ndarray, synth_feats: np.ndarray):
    """
    Compute ImprovedPrecision and ImprovedRecall using pymdma.
    
    Args:
        real_feats: shape (n_real_series, n_features) - real feature vectors
        synth_feats: shape (n_synth_series, n_features) - synthetic feature vectors
    
    Returns:
        dict with 'precision' and 'recall' (both floats in [0, 1])
    
    Note:
        pymdma returns MetricResult.value as tuple: (dataset_level, instance_level)
        We extract dataset_level which is the aggregated metric.
    """
    if not HAVE_PYMDMA:
        raise RuntimeError("pymdma not available in the environment")

    precision = ImprovedPrecision()
    recall = ImprovedRecall()

    # compute() signature: compute(real_features, fake_features) -> MetricResult
    pval_result = precision.compute(real_feats, synth_feats)
    rval_result = recall.compute(real_feats, synth_feats)

    # Extract dataset_level metric (first element of .value tuple)
    def extract_dataset_level(metric_result):
        if hasattr(metric_result, 'value'):
            val = metric_result.value
            if isinstance(val, tuple):
                return val[0]  # dataset_level metric
            return val
        return metric_result

    return {
        'precision': float(extract_dataset_level(pval_result)),
        'recall': float(extract_dataset_level(rval_result))
    }


def compute_knn_approx(real_feats: np.ndarray, synth_feats: np.ndarray, k: int = 5):
    # fallback approx using kNN-ball overlap (same as run_synthesis_evaluation)
    from sklearn.neighbors import KDTree
    tree_real = KDTree(real_feats)
    tree_synth = KDTree(synth_feats)

    dists_real, _ = tree_real.query(real_feats, k=k+1)
    r_real = dists_real[:, -1]

    dists_synth, _ = tree_synth.query(synth_feats, k=k+1)
    r_synth = dists_synth[:, -1]

    dist_synth_to_real, idx = tree_real.query(synth_feats, k=1)
    inside_real_ball = (dist_synth_to_real.flatten() <= r_real[idx.flatten()])
    precision = float(np.mean(inside_real_ball))

    dist_real_to_synth, idx2 = tree_synth.query(real_feats, k=1)
    inside_synth_ball = (dist_real_to_synth.flatten() <= r_synth[idx2.flatten()])
    recall = float(np.mean(inside_synth_ball))

    return {'precision': precision, 'recall': recall}


def evaluate_all(use_pymdma: bool = True):
    """Evaluate all synthetic datasets against their real counterparts using pymdma metrics."""
    TRAIN_DIR = "assets/results/training_sets"

    original_files = sorted([f for f in os.listdir(TRAIN_DIR) if f.endswith('_original.csv')])
    if not original_files:
        print('[ERROR] No original files found')
        return

    all_results = {}
    for orig in original_files:
        base = orig.replace('_original.csv', '')
        print(f"\n{'='*70}")
        print(f"Dataset: {base}")
        print(f"{'='*70}")
        
        real_df = pd.read_csv(os.path.join(TRAIN_DIR, orig))
        real_data = df_to_array(real_df)
        print(f"[Real] Data shape: {real_data.shape} (n_series={real_data.shape[0]}, series_len={real_data.shape[1]})")
        
        real_feats = extract_features(real_data)
        print(f"[Real] Features shape: {real_feats.shape} (n_series={real_feats.shape[0]}, n_features={real_feats.shape[1]})")

        # Save features for inspection
        feat_out = os.path.join(OUT_DIR, f"{base}_features.npy")
        np.save(feat_out, real_feats)

        # Find and evaluate synthetic files
        synth_files = sorted([f for f in os.listdir(TRAIN_DIR) 
                             if f.startswith(f"{base}_") and not f.endswith('_original.csv')])
        results = {}
        
        for f in synth_files:
            name = f.replace(f"{base}_", '').replace('.csv', '')
            synth_df = pd.read_csv(os.path.join(TRAIN_DIR, f))
            synth_data = df_to_array(synth_df)
            synth_feats = extract_features(synth_data)

            print(f"\n  [{name:20s}] Data: {synth_data.shape}, Features: {synth_feats.shape}")

            # Save synthetic features
            np.save(os.path.join(OUT_DIR, f"{base}_{name}_features.npy"), synth_feats)

            # Compute metrics
            if use_pymdma and HAVE_PYMDMA:
                metrics = compute_pymdma_metrics(real_feats, synth_feats)
                metric_source = "[pymdma]"
            else:
                # Fallback: simple distance metric
                from sklearn.neighbors import KDTree
                tree_real = KDTree(real_feats)
                tree_synth = KDTree(synth_feats)
                dists_real, _ = tree_real.query(real_feats, k=6)
                r_real = dists_real[:, -1]
                dists_synth, _ = tree_synth.query(synth_feats, k=6)
                r_synth = dists_synth[:, -1]
                dist_synth_to_real, idx = tree_real.query(synth_feats, k=1)
                prec = float(np.mean(dist_synth_to_real.flatten() <= r_real[idx.flatten()]))
                dist_real_to_synth, idx2 = tree_synth.query(real_feats, k=1)
                rec = float(np.mean(dist_real_to_synth.flatten() <= r_synth[idx2.flatten()]))
                metrics = {'precision': prec, 'recall': rec}
                metric_source = "[kNN approx]"

            print(f"      Precision: {metrics['precision']:7.4f} | Recall: {metrics['recall']:7.4f} {metric_source}")

            results[name] = {
                'precision': round(float(metrics['precision']), 3),
                'recall': round(float(metrics['recall']), 3)
            }

        # Save per-dataset results
        out_csv = os.path.join(OUT_DIR, f"{base}_pymdma_results.csv")
        pd.DataFrame.from_dict(results, orient='index').to_csv(out_csv)
        all_results[base] = results

    # Save aggregate results
    import json
    agg_path = os.path.join(OUT_DIR, 'OVERALL_PYMDMA_RESULTS.json')
    with open(agg_path, 'w') as fh:
        json.dump(all_results, fh, indent=2)
    
    print(f"\n{'='*70}")
    print("✓ Evaluation complete. Results saved to:")
    print(f"  - Per-dataset CSVs: {OUT_DIR}/<dataset>_pymdma_results.csv")
    print(f"  - Overall JSON: {agg_path}")
    print(f"  - Feature arrays: {OUT_DIR}/<dataset>_*_features.npy")
    print(f"\nMetrics Interpretation:")
    print(f"  Precision:  How many synthetic samples match real distribution (higher = better)")
    print(f"  Recall:     How well synthetic samples cover real distribution (often ~1.0)")
    print(f"{'='*70}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--use-pymdma', action='store_true', default=True,
                        help='Use pymdma metrics if available (default: True)')
    args = parser.parse_args()

    if not HAVE_PYMDMA:
        print("[WARNING] pymdma not available; falling back to kNN approximation")
    
    evaluate_all(use_pymdma=args.use_pymdma)
