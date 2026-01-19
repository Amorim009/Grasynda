
import os
import sys
import numpy as np
import pandas as pd
import re
import argparse
from statsmodels.tsa.seasonal import STL
from pathlib import Path
from typing import Dict, List, Union

# Robust project root detection - script is at scripts/experiments/2_analysis/
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
sys.path.insert(0, project_root)

print(f"Project root: {project_root}", flush=True)

try:
    from src.grasynda_unified import GrasyndaUnified
    from utils.load_data.config import DATASETS, DATA_GROUPS
    print("Imports successful.", flush=True)
except Exception as e:
    print(f"Import error: {e}", flush=True)
    sys.exit(1)

# Try metrics imports
try:
    from pymdma.time_series.measures.synthesis_val import (
        ImprovedPrecision, ImprovedRecall, Authenticity,
        Density, Coverage, CosineSimilarity,
        DTW, SpectralCoherence, CrossCorrelation,
        MMD, WassersteinDistance, FrechetDistance
    )
    HAVE_PYMDMA = True
except ImportError as e:
    print(f"pyMDMA import error: {e}", flush=True)
    HAVE_PYMDMA = False

# =============================================================================
# UTILS
# =============================================================================

def df_to_array(df: pd.DataFrame) -> List[np.ndarray]: # Changed return type to List[np.ndarray]
    if df.empty: return [] # Return empty list for consistency
    # Iterate and collect arrays without truncation
    lst = []
    uids = df['unique_id'].unique() # Ensure stable ordering
    for uid in uids:
        group = df[df['unique_id'] == uid]
        lst.append(group['y'].values) # No truncation, no reshape
    return lst

def df_to_list(df: pd.DataFrame) -> List[np.ndarray]:
    lst = []
    # Ensure stable ordering
    uids = df['unique_id'].unique()
    for uid in uids:
        group = df[df['unique_id'] == uid]
        lst.append(group['y'].values.reshape(-1, 1))
    return lst

def extract_features(data_list: List[np.ndarray], fs: int = 12):
    if not data_list: return np.array([])
    try:
        import tsfel
        cfg = tsfel.get_features_by_domain('temporal')
        cfg.update(tsfel.get_features_by_domain('statistical'))
        feats = []
        for series in data_list:
            # series is (len, 1) or (len,)
            row = np.nan_to_num(series.flatten())
            f = tsfel.time_series_features_extractor(cfg, row, fs=fs, verbose=0)
            feats_val = np.nan_to_num(f.values.flatten())
            feats.append(feats_val)
        return np.array(feats)
    except Exception as e:
        print(f"Error extracting features: {e}", flush=True)
        return np.array([])

# =============================================================================
# SINGLE AUDIT ENGINE
# =============================================================================

def run_dataset_audit(dataset_name, group, methods_to_run=None):
    """
    Run audit for a specific dataset and group.
    
    Args:
        dataset_name: Name of the dataset
        group: Group within the dataset
        methods_to_run: List of method names to evaluate. If None, runs all methods.
    """
    print(f"\n--- AUDIT: {dataset_name} ({group}) ---", flush=True)
    if methods_to_run:
        print(f"  Methods to evaluate: {methods_to_run}", flush=True)
    
    # 1. Load Real Data
    data_loader = DATASETS[dataset_name]
    df_all, _, _, _, freq_int = data_loader.load_everything(group)
    
    uids = [str(x) for x in df_all['unique_id'].unique()]
    df_real = df_all[df_all['unique_id'].astype(str).isin(uids)].copy()
    
    real_list = df_to_list(df_real)
    print(f"  Real series total: {len(uids)}", flush=True)
    
    real_feats = extract_features(real_list, fs=freq_int)
    
    real_var = np.var(real_feats, axis=0)
    common_valid_idx = real_var > 0
    
    # 2. Variants Loop
    results = []
    
    # Init Metrics
    m_prec, m_rec = ImprovedPrecision(), ImprovedRecall()
    m_auth, m_dens, m_cov, m_cos = Authenticity(), Density(), Coverage(), CosineSimilarity()
    m_mmd, m_wass, m_frec = MMD(), WassersteinDistance(), FrechetDistance()
    m_sc, m_cc = SpectralCoherence(), CrossCorrelation()

    # -- Grasynda Generation --
    model_dict = {}
    
    # Only generate Grasynda models if requested
    grasynda_variants = [
        'Grasynda_Standard', 'Grasynda_Hybrid', 'Grasynda_Visibility', 
        'Grasynda_Ensemble_Hybrid', 'Grasynda_Ensemble_25', 
        'Grasynda_Ensemble_50', 'Grasynda_Ensemble_100'
    ]
    if methods_to_run is None or any(m in methods_to_run for m in grasynda_variants):
        print("  Generating Grasynda models...", flush=True)
        
        if methods_to_run is None or 'Grasynda_Standard' in methods_to_run:
            print("  Generating Grasynda_Standard (Remainder)...", flush=True)
            model_std = GrasyndaUnified(period=freq_int, components_to_model=['remainder'])
            g_std_df = model_std.transform(df_real)
            model_dict['Grasynda_Standard'] = g_std_df

        if methods_to_run is None or 'Grasynda_Hybrid' in methods_to_run:
            print("  Generating Grasynda_Hybrid...", flush=True)
            model_hybrid = GrasyndaUnified(period=freq_int, components_to_model=['trend', 'remainder'],
                                          component_params={'trend': {'sampling_type': 'discrete', 'apply_differentiation': True}})
            g_hyb_df = model_hybrid.transform(df_real)
            model_dict['Grasynda_Hybrid'] = g_hyb_df

        if methods_to_run is None or 'Grasynda_Visibility' in methods_to_run:
             print("  Generating Grasynda_Visibility...", flush=True)
             model_vis = GrasyndaUnified(period=freq_int, graph_type='visibility', components_to_model=['remainder','trend'])
             g_vis_df = model_vis.transform(df_real)
             model_dict['Grasynda_Visibility'] = g_vis_df

        # Ensemble Helpers
        def get_ensemble_model(ens_size):
            return GrasyndaUnified(
                 period=freq_int,
                 n_quantiles=50,
                 sampling_type='discrete',
                 ensemble_transitions=True,
                 ensemble_size=ens_size,
                 components_to_model=['trend', 'remainder'],
                 component_params={
                     'trend': {'apply_differentiation': True},
                     'remainder': {'apply_differentiation': False}
                 }
             )

        if methods_to_run is None or 'Grasynda_Ensemble_Hybrid' in methods_to_run:
             print("  Generating Grasynda_Ensemble_Hybrid (50q, Ens10, DiffTrend)...", flush=True)
             model_adv = get_ensemble_model(10)
             g_adv_df = model_adv.transform(df_real)
             model_dict['Grasynda_Ensemble_Hybrid'] = g_adv_df

        if methods_to_run is None or 'Grasynda_Ensemble_25' in methods_to_run:
             print("  Generating Grasynda_Ensemble_25 (50q, Ens25, DiffTrend)...", flush=True)
             model_25 = get_ensemble_model(25)
             g_25_df = model_25.transform(df_real)
             model_dict['Grasynda_Ensemble_25'] = g_25_df

        if methods_to_run is None or 'Grasynda_Ensemble_50' in methods_to_run:
             print("  Generating Grasynda_Ensemble_50 (50q, Ens50, DiffTrend)...", flush=True)
             model_50 = get_ensemble_model(50)
             g_50_df = model_50.transform(df_real)
             model_dict['Grasynda_Ensemble_50'] = g_50_df

        if methods_to_run is None or 'Grasynda_Ensemble_100' in methods_to_run:
             print("  Generating Grasynda_Ensemble_100 (50q, Ens100, DiffTrend)...", flush=True)
             model_100 = get_ensemble_model(100)
             g_100_df = model_100.transform(df_real)
             model_dict['Grasynda_Ensemble_100'] = g_100_df
    
    # -- Load Baselines --
    BASE_DIR = os.path.join(project_root, "assets", "results", "training_sets")
    baselines = ['Scaling', 'Jittering', 'SeasonalMBB', 'MagnitudeWarping', 'TimeWarping', 'TSMixup', 'DBA']
    baseline_patterns = {
        'Scaling': '_SCALE', 'Jittering': '_JITTER', 'SeasonalMBB': '_MBB',
        'MagnitudeWarping': '_MWARP', 'TimeWarping': '_TWARP', 'TSMixup': 'TSMixup_', 'DBA': 'DBA_'
    }

    for method in baselines:

        if methods_to_run is not None and method not in methods_to_run:
            continue
            
        fpath = os.path.join(BASE_DIR, f"{dataset_name}_{group}_{method}.csv")
        if os.path.exists(fpath):
            print(f"  Loading baseline: {method}", flush=True)
            model_dict[method] = pd.read_csv(fpath)

    # 3. Evaluate each model
    for name, synth_df in model_dict.items():
        print(f"    Evaluating {name}...", flush=True)
        
    
        synth_pure_df = pd.DataFrame()
        
        if name.startswith('Grasynda'):
            # GrasyndaUnified already returns 1 series per original as f'{alias}_{uid}'
            # We just filter for that alias
            alias = 'GrasyndaUnified' # default in class
            target_ids = [f"{alias}_{u}" for u in uids]
            synth_pure_df = synth_df[synth_df['unique_id'].astype(str).isin(target_ids)].copy()
        
        elif name in ['TSMixup', 'DBA']:
            # Sequential prefixes
            pattern = baseline_patterns[name]
            # Try to find exactly len(uids) series starting with pattern
            all_synth_ids = [str(x) for x in synth_df['unique_id'].unique() if str(x).startswith(pattern)]
            if len(all_synth_ids) > len(uids):
                all_synth_ids = all_synth_ids[:len(uids)]
            synth_pure_df = synth_df[synth_df['unique_id'].astype(str).isin(all_synth_ids)].copy()
            
        else:
            # Suffix patterns (Scaling, etc.)
            p = baseline_patterns.get(name)
            # Robust matching: The suffix might be _SCALE0, _SCALE1, or _SCALE{global_idx}
            # We need ONE synthetic series per real series.
            
            # Create a set of all synthetic IDs for fast lookup
            syn_id_list = sorted([str(x) for x in synth_df['unique_id'].unique() if p in str(x)])
            
            # Map clean UID -> matched synthetic ID
            matched_ids = []
            
            for u in uids:
                prefix = f"{u}{p}"
                
                # Check for direct prefix match (e.g. M1_SCALE...)
                matches = [s for s in syn_id_list if s.startswith(prefix)]
                
                if matches:
                    matched_ids.append(matches[0]) # Take the first one found
            
            synth_pure_df = synth_df[synth_df['unique_id'].isin(matched_ids)].copy()

        # Data Check
        synth_list = df_to_list(synth_pure_df)
        if not synth_list:
            print(f"      Warning: No synthetic data filtered for {name} (N_synth=0)", flush=True)
            continue
            
        print(f"      Check: N_real={len(real_list)}, N_synth={len(synth_list)}", flush=True)
        
        # Features & Metric Compute
        synth_feats_raw = extract_features(synth_list, fs=freq_int)
        synth_var = np.var(synth_feats_raw, axis=0) if synth_feats_raw.size > 0 else np.array([])
        final_valid_idx = common_valid_idx & (synth_var > 0)
        
        v_real_feats = real_feats[:, final_valid_idx] if final_valid_idx.any() else real_feats
        v_synth_feats = synth_feats_raw[:, final_valid_idx] if final_valid_idx.any() else synth_feats_raw
        
        # Explicitly confirm feature consistency for the user
        print(f"      Feature consistency check: Real={v_real_feats.shape[1]} features, Synth={v_synth_feats.shape[1]} features", flush=True)
        
        # Spectral metrics need 3D array form: (n_samples, n_timesteps, n_channels)
        # We handle variable-length series by truncating to minimum length
        # to create uniform arrays for spectral metrics.
        
        def to_3d_array(lst):
            """
            Convert list of 2D arrays (n_timesteps, 1) to 3D array (n_samples, n_timesteps, 1).
            Truncates all series to minimum length to create uniform array.
            """
            if not lst:
                return np.array([])
            
            # Find minimum length
            m_len = min(len(x) for x in lst)
            
            # Truncate and stack
            # Each x has shape (n_timesteps, 1), truncate to (m_len, 1)
            truncated = [x[:m_len] for x in lst]
            
            # Stack to create (n_samples, m_len, 1)
            arr_3d = np.array(truncated)
            
            # Ensure 3D shape
            if arr_3d.ndim == 2:
                # If somehow we got 2D, add channel dimension
                arr_3d = arr_3d[..., np.newaxis]
            
            return arr_3d

        # Convert to 3D arrays for spectral metrics
        real_arr_3d = to_3d_array(real_list)
        synth_arr_3d = to_3d_array(synth_list)

        # Validate data shapes for debugging
        print(f"      Data shapes: real_list[0]={real_list[0].shape}, synth_list[0]={synth_list[0].shape}", flush=True)
        
        # Check for overlap/leakage
        # Flatten arrays to compares content
        real_flat = [x.flatten().tobytes() for x in real_list]
        synth_flat = [x.flatten().tobytes() for x in synth_list]
        
        # Intersection
        overlap_count = len(set(real_flat) & set(synth_flat))
        if overlap_count > 0:
            print(f"      WARNING: Found {overlap_count} synthetic series that are EXACTLY identical to real series!", flush=True)
        else:
            print(f"      Overlap Check: OK (0 identical series)", flush=True)
            
        print(f"      Features: real={v_real_feats.shape}, synth={v_synth_feats.shape}", flush=True)
        print(f"      3D arrays: real={real_arr_3d.shape}, synth={synth_arr_3d.shape}", flush=True)
        
        try:
            import time
            start_total = time.time()
            
            print("      Computing Precision...", flush=True)
            t0 = time.time()
            v_prec = m_prec.compute(v_real_feats, v_synth_feats).dataset_level.value
            print(f"      > Done ({time.time()-t0:.2f}s)", flush=True)

            print("      Computing Recall...", flush=True)
            t0 = time.time()
            v_rec = m_rec.compute(v_real_feats, v_synth_feats).dataset_level.value
            print(f"      > Done ({time.time()-t0:.2f}s)", flush=True)

            print("      Computing Authenticity...", flush=True)
            t0 = time.time()
            v_auth = m_auth.compute(v_real_feats, v_synth_feats).dataset_level.value
            print(f"      > Done ({time.time()-t0:.2f}s)", flush=True)

            print("      Computing Density...", flush=True)
            t0 = time.time()
            v_dens = m_dens.compute(v_real_feats, v_synth_feats).dataset_level.value
            print(f"      > Done ({time.time()-t0:.2f}s)", flush=True)

            print("      Computing Coverage...", flush=True)
            t0 = time.time()
            v_cov = m_cov.compute(v_real_feats, v_synth_feats).dataset_level.value
            print(f"      > Done ({time.time()-t0:.2f}s)", flush=True)

            print("      Computing CosineSim...", flush=True)
            t0 = time.time()
            v_cos = m_cos.compute(v_real_feats, v_synth_feats).dataset_level.value
            print(f"      > Done ({time.time()-t0:.2f}s)", flush=True)

            print("      Computing MMD...", flush=True)
            t0 = time.time()
            v_mmd = m_mmd.compute(v_real_feats, v_synth_feats).dataset_level.value
            print(f"      > Done ({time.time()-t0:.2f}s)", flush=True)

            print("      Computing Wasserstein...", flush=True)
            t0 = time.time()
            v_wass = m_wass.compute(v_real_feats, v_synth_feats).dataset_level.value
            print(f"      > Done ({time.time()-t0:.2f}s)", flush=True)

            print("      Computing Frechet...", flush=True)
            t0 = time.time()
            v_frec = m_frec.compute(v_real_feats, v_synth_feats).dataset_level.value
            print(f"      > Done ({time.time()-t0:.2f}s)", flush=True)

            print("      Computing SpectralCoherence...", flush=True)
            t0 = time.time()
            v_sc = m_sc.compute(real_arr_3d, synth_arr_3d).dataset_level.value
            print(f"      > Done ({time.time()-t0:.2f}s)", flush=True)

            print("      Computing CrossCorrelation...", flush=True)
            t0 = time.time()
            v_cc = m_cc.compute(real_arr_3d, synth_arr_3d).dataset_level.value
            print(f"      > Done ({time.time()-t0:.2f}s)", flush=True)


            row = {
                'Dataset': dataset_name, 'Group': group, 'Method': name,
                'N': len(real_list),
                'Precision': v_prec,
                'Recall': v_rec,
                'Authenticity': v_auth,
                'Density': v_dens,
                'Coverage': v_cov,
                'CosineSim': v_cos,
                'MMD': v_mmd,
                'Wasserstein': v_wass,
                'Frechet': v_frec,
                'SpectralCoh': v_sc,
                'CrossCorr': v_cc
            }
            print(f"      Total metrics computed in {time.time()-start_total:.2f}s", flush=True)
            results.append(row)
        except Exception as e:
            print(f"      Error in evaluation for {name}: {e}", flush=True)

    return results

# =============================================================================
# OUTPUT HELPERS
# =============================================================================

def save_summary_and_partition(results_list, out_dir, details_file):
    """
    Save the detailed result list, partition it by model, and generate an aggregated summary.
    """
    if not results_list:
        return

    full_df = pd.DataFrame(results_list)
    
    # 1. Save Detailed CSV (Checkpoint)
    full_df.to_csv(details_file, index=False)
    
    # 2. Split Results by Method (Individual Files)
    models_dir = os.path.join(out_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    for method in full_df['Method'].unique():
        method_df = full_df[full_df['Method'] == method]
        safe_name = "".join([c if c.isalnum() or c in ('_', '-') else "_" for c in method])
        method_file = os.path.join(models_dir, f"{safe_name}.csv")
        method_df.to_csv(method_file, index=False)

    # 3. Aggregated Summary (Average across datasets)
    numeric_cols = [c for c in full_df.columns if c not in ['Dataset', 'Group', 'Method', 'N']]
    summary_df = full_df.groupby('Method')[numeric_cols].mean().reset_index()
    
    # Round to 3 decimal places
    summary_df = summary_df.round(3)
    
    summary_file = os.path.join(out_dir, "grasynda_comprehensive_audit_summary.csv")
    summary_df.to_csv(summary_file, index=False)
    
    return summary_df

# =============================================================================
# MAIN ORCHESTRATOR
# =============================================================================

def run_full_experiment(methods_to_run=None):
    """
    Run full experiment across all datasets.
    
    Args:
        methods_to_run: List of method names to evaluate. If None, runs all methods.
    """
    print("="*60, flush=True)
    if methods_to_run:
        print(f"PYMDMA METRICS FOR SELECTED METHODS: {', '.join(methods_to_run)}", flush=True)
    else:
        print("COMPREHENSIVE REALISM AUDIT: ALL DATASETS, ALL METHODS", flush=True)
        
    print("="*60, flush=True)
    
    out_dir = os.path.join(project_root, "assets", "results", "pymdma_metrics")
    os.makedirs(out_dir, exist_ok=True)
    details_file = os.path.join(out_dir, "grasynda_comprehensive_audit_details.csv")
    
    # Load existing results if any to allow resume
    if os.path.exists(details_file):
        try:
            existing_df = pd.read_csv(details_file)
            all_results = existing_df.to_dict('records')
            completed_combos = set(zip(existing_df['Dataset'], existing_df['Group']))
            print(f"Found {len(all_results)} existing records. Resuming...", flush=True)
        except:
            all_results = []
            completed_combos = set()
    else:
        all_results = []
        completed_combos = set()
    
    for ds_name, grp in DATA_GROUPS:
        # Check if we should skip this combo based on what methods are requested
        # Ideally we'd check if this exact (dataset, group, method) is done, but simplifying here
        
        try:
            res = run_dataset_audit(ds_name, grp, methods_to_run=methods_to_run)
            all_results.extend(res)
            
            # Checkpoint: Save details, split models, and update summary after each dataset
            save_summary_and_partition(all_results, out_dir, details_file)
            print(f">>> Checkpoint saved: {len(all_results)} total records.", flush=True)
            
        except Exception as e:
            import traceback
            print(f"  MAJOR ERROR auditing {ds_name}_{grp}: {e}", flush=True)
            traceback.print_exc()

    if not all_results:
        print("No results collected. Exiting.", flush=True)
        return

    # Final Save and Report
    summary_df = save_summary_and_partition(all_results, out_dir, details_file)
    summary_file = os.path.join(out_dir, "grasynda_comprehensive_audit_summary.csv")
    
    print("\n" + "="*60, flush=True)
    print(f"GLOBAL SUMMARY: {summary_file}", flush=True)
    print("="*60, flush=True)
    if summary_df is not None:
        print(summary_df.to_string(index=False), flush=True)
    print("="*60, flush=True)
    print(f"Detailed logs in: {details_file}", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute PyMDMA metrics for synthetic data')
    parser.add_argument('--methods', nargs='+', default=None,
                        help='Specific methods to evaluate (e.g., --methods Scaling Jittering). '
                             'Available: Scaling, Jittering, SeasonalMBB, MagnitudeWarping, TimeWarping, '
                             'TSMixup, DBA, Grasynda_Standard, Grasynda_Hybrid. '
                             'If not specified, all methods are evaluated.')
    
    args = parser.parse_args()
    
    if args.methods:
        print(f"Running with selected methods: {args.methods}", flush=True)
    
    run_full_experiment(methods_to_run=args.methods)
