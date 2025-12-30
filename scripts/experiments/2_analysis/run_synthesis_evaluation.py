"""Synthesis evaluation (from scratch)

This script evaluates synthetic training sets against real ones using a
lightweight feature extractor (tsfel optional) and kNN-based precision/recall
metrics (no dependency on `pymdma`).

Usage:
    python run_synthesis_evaluation.py

Outputs CSVs to `assets/results/synthesis_eval/` with columns: precision, recall.
"""

import os
import glob
import argparse
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree


TRAIN_DIR = "assets/results/training_sets"
OUT_DIR = "assets/results/synthesis_eval"
os.makedirs(OUT_DIR, exist_ok=True)


try:
    import tsfel
    HAVE_TS_FEL = True
except Exception:
    HAVE_TS_FEL = False


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


def extract_features(time_series_data: np.ndarray) -> np.ndarray:
    if HAVE_TS_FEL:
        try:
            return extract_features_tsfel(time_series_data)
        except Exception:
            pass
    return extract_features_fallback(time_series_data)


def compute_knn_precision_recall(real_feats: np.ndarray, synth_feats: np.ndarray, k: int = 5) -> Tuple[float, float]:
    """Compute kNN-ball precision and recall.

    Precision: fraction of synthetic samples that fall within at least one real-sample kNN ball.
    Recall: fraction of real samples that fall within at least one synthetic-sample kNN ball.
    """
    # Build KD trees
    tree_real = KDTree(real_feats)
    tree_synth = KDTree(synth_feats)

    # Compute radii: distance from each point to its k-th nearest neighbor (exclude itself)
    # For real
    dists_real, _ = tree_real.query(real_feats, k=k+1)  # includes self at distance 0
    r_real = dists_real[:, -1]

    # For synth
    dists_synth, _ = tree_synth.query(synth_feats, k=k+1)
    r_synth = dists_synth[:, -1]

    # Precision: proportion of synth points within ANY real ball
    # For each synth point, find nearest real point and check distance <= that real's radius
    dist_synth_to_real, idx = tree_real.query(synth_feats, k=1)
    inside_real_ball = (dist_synth_to_real.flatten() <= r_real[idx.flatten()])
    precision = float(np.mean(inside_real_ball))

    # Recall: proportion of real points within ANY synth ball
    dist_real_to_synth, idx2 = tree_synth.query(real_feats, k=1)
    inside_synth_ball = (dist_real_to_synth.flatten() <= r_synth[idx2.flatten()])
    recall = float(np.mean(inside_synth_ball))

    return precision, recall


def evaluate_dataset_files(base: str):
    orig = f"{base}_original.csv"
    real_df = pd.read_csv(os.path.join(TRAIN_DIR, orig))
    real_data = df_to_array(real_df)
    real_feats = extract_features(real_data)

    results = {}
    for path in sorted(glob.glob(os.path.join(TRAIN_DIR, f"{base}_*.csv"))):
        fname = os.path.basename(path)
        if fname.endswith('_original.csv'):
            continue
        synth_name = fname.replace(f"{base}_", '').replace('.csv', '')
        synth_df = pd.read_csv(path)
        synth_data = df_to_array(synth_df)
        synth_feats = extract_features(synth_data)

        precision, recall = compute_knn_precision_recall(real_feats, synth_feats, k=5)
        results[synth_name] = {'precision': round(precision, 3), 'recall': round(recall, 3)}

    out_path = os.path.join(OUT_DIR, f"{base}_knn_results.csv")
    pd.DataFrame.from_dict(results, orient='index').to_csv(out_path)
    return results


def main():
    # find all _original files and derive bases
    originals = sorted([os.path.basename(p).replace('_original.csv', '') for p in glob.glob(os.path.join(TRAIN_DIR, '*_original.csv'))])
    if not originals:
        print('No original training sets found.')
        return

    all_results = {}
    for base in originals:
        print(f"Evaluating: {base}")
        res = evaluate_dataset_files(base)
        all_results[base] = res
        print(f" -> Saved to {os.path.join(OUT_DIR, f'{base}_knn_results.csv')}")

    # Save aggregated JSON
    try:
        import json
        with open(os.path.join(OUT_DIR, 'OVERALL_KNN_RESULTS.json'), 'w') as fh:
            json.dump(all_results, fh, indent=2)
    except Exception:
        pass


if __name__ == '__main__':
    main()
