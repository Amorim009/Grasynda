"""
Grasynda Synthesis Evaluation - CORRECTED

This script evaluates synthetic time series properly by:
1. Loading the training set (which contains original + synthetic)
2. Removing the original data to get ONLY the synthetic
3. Computing metrics on purely synthetic data vs. real data

This avoids the bug where recall=1.0 due to original data being included.

Usage:
    python run_pymdma_eval_corrected.py --use-pymdma
"""

import os
import glob
import argparse
import numpy as np
import pandas as pd
import sys
from pathlib import Path

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


def df_to_array(df: pd.DataFrame) -> np.ndarray:
    """Convert long dataframe to array shape (n_series, series_length)."""
    pivot = df.pivot(index="ds", columns="unique_id", values="y")
    pivot = pivot.ffill().bfill()
    return pivot.T.values


def extract_features_tsfel(time_series_data: np.ndarray, fs: int = 12) -> np.ndarray:
    """Extract temporal domain features using TSFEL."""
    cfg = tsfel.get_features_by_domain('temporal')
    feats = []
    for i in range(time_series_data.shape[0]):
        feats.append(tsfel.time_series_features_extractor(cfg, time_series_data[i, :], fs=fs, verbose=0).values.flatten())
    return np.array(feats)


def extract_features_fallback(time_series_data: np.ndarray) -> np.ndarray:
    """Lightweight fallback feature extractor."""
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
    """Extract features using TSFEL or fallback."""
    if HAVE_TS_FEL:
        try:
            return extract_features_tsfel(time_series_data)
        except Exception as e:
            print(f"tsfel extraction failed; fallback used: {e}")
    return extract_features_fallback(time_series_data)


def get_only_synthetic(synth_df: pd.DataFrame, real_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract ONLY the synthetic series from the mixed dataset.
    
    The saved 'synthetic' files contain: original_data + synthetic_data
    We need to remove the original data to get pure synthetic.
    """
    real_ids = set(real_df['unique_id'].unique())
    
    # Filter to keep only rows that don't have the exact original values
    # Strategy: Keep rows with unique_ids NOT in real data
    synth_only = synth_df[~synth_df['unique_id'].isin(real_ids)].copy()
    
    print(f"  [Data Separation]")
    print(f"    Real IDs: {len(real_ids)}")
    print(f"    Total in synth file: {synth_df['unique_id'].nunique()}")
    print(f"    Pure synthetic only: {synth_only['unique_id'].nunique()}")
    
    return synth_only


def compute_pymdma_metrics(real_feats: np.ndarray, synth_feats: np.ndarray):
    """Compute ImprovedPrecision and ImprovedRecall using pymdma."""
    if not HAVE_PYMDMA:
        raise RuntimeError("pymdma not available in the environment")

    precision = ImprovedPrecision()
    recall = ImprovedRecall()

    pval_result = precision.compute(real_feats, synth_feats)
    rval_result = recall.compute(real_feats, synth_feats)

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


def evaluate_all(use_pymdma: bool = True):
    """Evaluate synthetic datasets using ONLY the pure synthetic portions."""
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
        print(f"[Real] Features shape: {real_feats.shape}")

        # Find and evaluate synthetic files
        synth_files = sorted([f for f in os.listdir(TRAIN_DIR) 
                             if f.startswith(f"{base}_") and not f.endswith('_original.csv')])
        results = {}
        
        for f in synth_files:
            name = f.replace(f"{base}_", '').replace('.csv', '')
            synth_df = pd.read_csv(os.path.join(TRAIN_DIR, f))
            
            # ⭐ KEY FIX: Extract ONLY the synthetic data, remove original
            synth_only_df = get_only_synthetic(synth_df, real_df)
            
            if synth_only_df.empty:
                print(f"\n  [{name:20s}] ⚠️  No purely synthetic data found (all data is original)")
                continue
            
            synth_data = df_to_array(synth_only_df)
            synth_feats = extract_features(synth_data)

            print(f"\n  [{name:20s}]")
            print(f"    Synth data shape: {synth_data.shape}")
            print(f"    Synth features shape: {synth_feats.shape}")

            # Compute metrics
            if use_pymdma and HAVE_PYMDMA:
                metrics = compute_pymdma_metrics(real_feats, synth_feats)
                print(f"    Precision: {metrics['precision']:7.4f} | Recall: {metrics['recall']:7.4f}")
            else:
                # Fallback kNN
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
                print(f"    Precision: {metrics['precision']:7.4f} | Recall: {metrics['recall']:7.4f} [kNN fallback]")

            results[name] = {
                'precision': round(float(metrics['precision']), 3),
                'recall': round(float(metrics['recall']), 3)
            }

        # Save per-dataset results
        if results:
            out_csv = os.path.join(OUT_DIR, f"{base}_pymdma_results_corrected.csv")
            pd.DataFrame.from_dict(results, orient='index').to_csv(out_csv)
            all_results[base] = results
            print(f"\n  Results saved to {out_csv}")

    # Save aggregate results
    import json
    agg_path = os.path.join(OUT_DIR, 'OVERALL_PYMDMA_RESULTS_CORRECTED.json')
    with open(agg_path, 'w') as fh:
        json.dump(all_results, fh, indent=2)
    
    print(f"\n{'='*70}")
    print("✓ Evaluation complete (CORRECTED - original data removed)")
    print(f"  Results: {agg_path}")
    print(f"  Per-dataset: {OUT_DIR}/<dataset>_pymdma_results_corrected.csv")
    print(f"{'='*70}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--use-pymdma', action='store_true', default=True,
                        help='Use pymdma metrics if available (default: True)')
    args = parser.parse_args()

    if not HAVE_PYMDMA:
        print("[WARNING] pymdma not available; falling back to kNN approximation")
    
    evaluate_all(use_pymdma=args.use_pymdma)
