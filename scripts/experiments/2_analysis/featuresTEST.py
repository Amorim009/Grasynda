import os
import glob
import pandas as pd
import numpy as np
from typing import List

TRAIN_DIR = "assets/results/training_sets"
PYMDMA_DIR = "assets/results/pymdma_metrics"
os.makedirs(PYMDMA_DIR, exist_ok=True)

try:
    import tsfel
    HAVE_TS_FEL = True
except Exception:
    HAVE_TS_FEL = False

try:
    import pymdma
    from pymdma.time_series.measures.synthesis_val import ImprovedPrecision, ImprovedRecall
    HAVE_PYMDMA = True
except Exception:
    HAVE_PYMDMA = False


def df_to_array(df: pd.DataFrame) -> np.ndarray:
    """Convert long dataframe to array shape (n_series, series_length).
    Uses pivot on `ds` x `unique_id` and forward/backward fills missing values.
    """
    pivot = df.pivot(index="ds", columns="unique_id", values="y")
    pivot = pivot.ffill().bfill()
    return pivot.T.values


def extract_features_tsfel(time_series_data: np.ndarray, fs: int = 12) -> np.ndarray:
    cfg = tsfel.get_features_by_domain('temporal')
    features_list = []
    for i in range(time_series_data.shape[0]):
        single_ts = time_series_data[i, :]
        features = tsfel.time_series_features_extractor(cfg, single_ts, fs=fs, verbose=0)
        features_list.append(features.values.flatten())
    return np.array(features_list)


def extract_features_fallback(time_series_data: np.ndarray) -> np.ndarray:
    """Fallback lightweight feature extractor in case `tsfel` is missing.
    Produces features: mean, std, median, iqr, skew, kurtosis, acf1
    """
    from math import isfinite
    features = []
    for i in range(time_series_data.shape[0]):
        ts = time_series_data[i, :].astype(float)
        # basic stats
        mean = np.nanmean(ts)
        std = np.nanstd(ts)
        median = np.nanmedian(ts)
        q75, q25 = np.nanpercentile(ts, [75, 25])
        iqr = q75 - q25

        # skew/kurtosis using pandas if available
        try:
            ser = pd.Series(ts)
            skew = float(ser.skew())
            kurt = float(ser.kurt())
        except Exception:
            skew = 0.0
            kurt = 0.0

        # acf1 (lag-1 autocorrelation)
        if len(ts) > 1:
            ts0 = ts[:-1] - np.nanmean(ts[:-1])
            ts1 = ts[1:] - np.nanmean(ts[1:])
            denom = np.sqrt(np.nanvar(ts0) * np.nanvar(ts1))
            if denom == 0:
                acf1 = 0.0
            else:
                acf1 = float(np.nanmean(ts0 * ts1) / denom)
        else:
            acf1 = 0.0

        feat = [mean, std, median, iqr, skew, kurt, acf1]
        feat = [f if (isinstance(f, float) and isfinite(f)) else 0.0 for f in feat]
        features.append(feat)

    return np.array(features)


def extract_features(time_series_data: np.ndarray, fs: int = 12) -> np.ndarray:
    if HAVE_TS_FEL:
        try:
            return extract_features_tsfel(time_series_data, fs=fs)
        except Exception as e:
            print(f"tsfel extraction failed, falling back: {e}")
    return extract_features_fallback(time_series_data)


def extract_main_value(metric_result):
    if hasattr(metric_result, 'value'):
        val = metric_result.value
        if isinstance(val, tuple):
            return val[0]
        return val
    return metric_result


def evaluate_all_training_sets():
    original_files = sorted([f for f in os.listdir(TRAIN_DIR) if f.endswith("_original.csv")])

    if not original_files:
        print("No original training set files found in TRAIN_DIR")
        return

    all_results = {}

    for orig in original_files:
        base = orig.replace("_original.csv", "")
        # attempt to split dataset and group by last underscore
        if "_" in base:
            dataset_name, group = base.rsplit("_", 1)
        else:
            dataset_name, group = base, ""

        print(f"\nProcessing dataset: {dataset_name} group: {group}")

        real_path = os.path.join(TRAIN_DIR, orig)
        real_df = pd.read_csv(real_path)
        real_data = df_to_array(real_df)
        real_features = extract_features(real_data)

        # find synthetic files for this dataset/group
        syn_pattern = os.path.join(TRAIN_DIR, f"{base}_*.csv")
        synthetic_files = [os.path.basename(p) for p in glob.glob(syn_pattern) if not p.endswith("_original.csv")]

        results = {}

        for fname in synthetic_files:
            synth_name = fname.replace(f"{base}_", "").replace('.csv', '')
            print(f"  - Evaluating: {synth_name}")
            synth_df = pd.read_csv(os.path.join(TRAIN_DIR, fname))
            synth_data = df_to_array(synth_df)
            synth_features = extract_features(synth_data)

            if not HAVE_PYMDMA:
                print("pymdma not available in this environment. Skipping metric computation.")
                results[synth_name] = {'precision': None, 'recall': None}
                continue

            precision = ImprovedPrecision()
            recall = ImprovedRecall()

            try:
                pval = extract_main_value(precision.compute(real_features, synth_features))
            except Exception as e:
                print(f"    precision compute failed: {e}")
                pval = None

            try:
                rval = extract_main_value(recall.compute(real_features, synth_features))
            except Exception as e:
                print(f"    recall compute failed: {e}")
                rval = None

            results[synth_name] = {'precision': None if pval is None else float(round(pval, 3)),
                                   'recall': None if rval is None else float(round(rval, 3))}

        all_results[f"{dataset_name}_{group}"] = results

        # Save per-dataset results
        out_path = os.path.join(PYMDMA_DIR, f"{dataset_name}_{group}_pymdma_results.csv")
        pd.DataFrame.from_dict(results, orient='index').to_csv(out_path)
        print(f"  -> Saved results to {out_path}")

    # Optionally write aggregated results
    agg_path = os.path.join(PYMDMA_DIR, "OVERALL_PYMDMA_RESULTS.json")
    try:
        import json
        with open(agg_path, 'w') as fh:
            json.dump(all_results, fh, indent=2)
        print(f"Overall results written to {agg_path}")
    except Exception:
        pass


if __name__ == '__main__':
    evaluate_all_training_sets()
