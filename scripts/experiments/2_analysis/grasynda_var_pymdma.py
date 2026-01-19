import os
import sys
import pandas as pd
import numpy as np
import warnings
import argparse
import time
import traceback
from typing import List

# Robust project root detection
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
sys.path.insert(0, project_root)

# Try imports
try:
    from src.grasynda_unified import GrasyndaUnified
    from utils.load_data.config import DATASETS, DATA_GROUPS
    from pymdma.time_series.measures.synthesis_val import (
        ImprovedPrecision, ImprovedRecall, Authenticity,
        Density, Coverage, CosineSimilarity,
        SpectralCoherence, CrossCorrelation, SpectralWassersteinDistance,
        MMD, WassersteinDistance, FrechetDistance
    )
    print("Imports successful.", flush=True)
except Exception as e:
    print(f"Import error: {e}", flush=True)
    sys.exit(1)

# Suppress warnings
warnings.filterwarnings("ignore")

OUT_DIR = os.path.join("assets", "results", "grasynda_variants_audit")
os.makedirs(OUT_DIR, exist_ok=True)

# -------------------------------------------------------------------------
# Robust Helpers from compute_pymdma_metrics.py
# -------------------------------------------------------------------------

def df_to_list(df: pd.DataFrame) -> List[np.ndarray]:
    lst = []
    uids = df['unique_id'].unique()
    for uid in uids:
        group = df[df['unique_id'] == uid]
        lst.append(group['y'].values.reshape(-1, 1))
    return lst

def to_3d_array(lst):
    if not lst: return np.array([])
    m_len = min(len(x) for x in lst)
    truncated = [x[:m_len] for x in lst]
    arr_3d = np.array(truncated)
    if arr_3d.ndim == 2:
        arr_3d = arr_3d[..., np.newaxis]
    return arr_3d

def extract_features(data_list: List[np.ndarray], fs: int = 12):
    if not data_list: return np.array([])
    try:
        import tsfel
        cfg = tsfel.get_features_by_domain('temporal')
        cfg.update(tsfel.get_features_by_domain('statistical'))
        feats = []
        for series in data_list:
            row = np.nan_to_num(series.flatten())
            f = tsfel.time_series_features_extractor(cfg, row, fs=fs, verbose=0)
            feats_val = np.nan_to_num(f.values.flatten())
            feats.append(feats_val)
        return np.array(feats)
    except Exception as e:
        print(f"Error extracting features: {e}", flush=True)
        return np.array([])
def get_grasynda_variants(freq_int):
    return {
        'Grasynda_Auth_Q12_Uniform': GrasyndaUnified(
            period=freq_int,
            n_quantiles=12,
            components_to_model=['trend', 'remainder'],
            ensemble_size=50,
            ensemble_transitions=True,
            component_params={
                'trend': {'sampling_type': 'continuous_uniform', 'apply_differentiation': True},
                'remainder': {'sampling_type': 'continuous_uniform', 'apply_differentiation': False}
            }
        ),
        'Grasynda_Auth_Q8_Uniform': GrasyndaUnified(
            period=freq_int,
            n_quantiles=8,
            components_to_model=['trend', 'remainder'],
            ensemble_size=50,
            ensemble_transitions=True,
            component_params={
                'trend': {'sampling_type': 'continuous_uniform', 'apply_differentiation': True},
                'remainder': {'sampling_type': 'continuous_uniform', 'apply_differentiation': False}
            }
        ),
        'Grasynda_Auth_Visibility_Q25_Uniform': GrasyndaUnified(
            period=freq_int,
            n_quantiles=25,
            components_to_model=['trend', 'remainder'],
            ensemble_size=50,
            ensemble_transitions=True,
            graph_type='visibility',
            visibility_type='horizontal',
            component_params={
                'trend': {'sampling_type': 'continuous_uniform', 'apply_differentiation': True},
                'remainder': {'sampling_type': 'continuous_uniform', 'apply_differentiation': False}
            }
        )
    }


# -------------------------------------------------------------------------
# Evaluation Engine
# -------------------------------------------------------------------------

def run_dataset_audit(dataset_name, group, methods_to_run=None):
    print(f"\n--- AUDIT: {dataset_name} ({group}) ---", flush=True)
    data_loader = DATASETS[dataset_name]
    df_real, _, _, _, freq_int = data_loader.load_everything(group)
    
    unique_ids = df_real['unique_id'].unique()
    print(f"  Real series total: {len(unique_ids)}, FreqInt: {freq_int}", flush=True)
    
    models = get_grasynda_variants(freq_int)
    results = []
    
    real_list = df_to_list(df_real)
    print("  Extracting real features...", flush=True)
    v_real = extract_features(real_list, fs=freq_int)
    
    m_prec, m_rec = ImprovedPrecision(), ImprovedRecall()
    m_auth, m_dens, m_cov, m_cos = Authenticity(), Density(), Coverage(), CosineSimilarity()
    m_mmd, m_wass, m_frech = MMD(), WassersteinDistance(), FrechetDistance()
    m_sc, m_cc, m_spec_wass = SpectralCoherence(), CrossCorrelation(), SpectralWassersteinDistance()

    for name, model in models.items():
        if methods_to_run and name not in methods_to_run: continue
        print(f"    Evaluating {name}...", flush=True)
        try:
            synth_df = model.transform(df_real)
            synth_list = df_to_list(synth_df)
            v_synth = extract_features(synth_list, fs=freq_int)
            
            real_arr_3d = to_3d_array(real_list)
            synth_arr_3d = to_3d_array(synth_list)
            
            start = time.time()
            metrics = {
                'Dataset': dataset_name, 'Group': group, 'Method': name, 'N': len(unique_ids),
                'Precision': m_prec.compute(v_real, v_synth).dataset_level.value,
                'Recall': m_rec.compute(v_real, v_synth).dataset_level.value,
                'Authenticity': m_auth.compute(v_real, v_synth).dataset_level.value,
                'Density': m_dens.compute(v_real, v_synth).dataset_level.value,
                'Coverage': m_cov.compute(v_real, v_synth).dataset_level.value,
                'CosineSim': m_cos.compute(v_real, v_synth).dataset_level.value,
                'MMD': m_mmd.compute(v_real, v_synth).dataset_level.value,
                'Wasserstein': m_wass.compute(v_real, v_synth).dataset_level.value,
                'Frechet': m_frech.compute(v_real, v_synth).dataset_level.value,
                'SpectralCoh': m_sc.compute(real_arr_3d, synth_arr_3d).dataset_level.value,
                'CrossCorr': m_cc.compute(real_arr_3d, synth_arr_3d).dataset_level.value,
                'SpecWasserstein': m_spec_wass.compute(real_arr_3d, synth_arr_3d).dataset_level.value
            }
            results.append(metrics)
            print(f"      Done in {time.time() - start:.2f}s", flush=True)
        except Exception as e:
            print(f"      ERROR evaluating {name}: {e}", flush=True)
            traceback.print_exc()
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--methods', nargs='+', help='Subset of variants to run')
    args = parser.parse_args()
    all_results = []
    for ds_name, grp in DATA_GROUPS:
        res = run_dataset_audit(ds_name, grp, methods_to_run=args.methods)
        all_results.extend(res)
        pd.DataFrame(all_results).to_csv(os.path.join(OUT_DIR, "details.csv"), index=False)
    
    if not all_results:
        print("\n" + "!"*50 + "\nNO RESULTS COMPUTED. CHECK FOR ERRORS ABOVE.\n" + "!"*50)
        return
    
    df = pd.DataFrame(all_results)
    summary = df.groupby('Method').mean(numeric_only=True).round(3)
    summary.to_csv(os.path.join(OUT_DIR, "summary.csv"))
    print("\n" + "="*50 + "\nGRASYNDA VARIANTS ANALYSIS COMPLETE\n" + "="*50)
    print(summary)

if __name__ == "__main__":
    main()
