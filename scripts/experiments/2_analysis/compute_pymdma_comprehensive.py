"""
PyMDMA Comprehensive Synthesis Validation Metrics.

Computes Authenticity, Fidelity, Diversity, and Privacy for all models.
Excludes Grasynda RAW Y variants.
Uses optimized parameters from universal experiments.
Enforces synthetic-only data for metric calculation.
"""

import os
import sys
import numpy as np
import pandas as pd
import time
import warnings

warnings.filterwarnings("ignore")

# ================= PROJECT SETUP =================

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
sys.path.insert(0, project_root)

# Project imports
from src.grasynda_unified import GrasyndaUnified
from src.grasynda_sax import GrasyndaSAX
from utils.load_data.config import DATASETS
from utils.config import SYNTH_METHODS
from src.workflow import ExpWorkflow

try:
    from pymdma.tabular.data.load import TabularDataset
    from pymdma.tabular.measures.synthesis_val import (
        Authenticity, ImprovedPrecision, ImprovedRecall, DCRPrivacy
    )
except ImportError as e:
    print(f"PyMDMA import error: {e}")
    sys.exit(1)


# ================= CONFIGURATION =================

# Baseline methods
BASELINES = ['Exact_Copy', 'Random_Noise']

# Grasynda variants to run for the full evaluation sweep.
GRASYNDA_VARIANTS_TO_RUN = [
    'Hybrid_SAX_Q10_Ensemble5_Continuous',
    'Hybrid_SAX_Q10_NoEnsemble_Continuous',
]

# Other DA methods to run for the full evaluation sweep.
OTHER_DA_METHODS = [
    'SeasonalMBB',
    'Jittering',
    'Scaling',
    'TimeWarping',
    'MagnitudeWarping',
    'TSMixup',
    'DBA',
    'TimeVAE',
]

# Full list of methods to run.
METHODS_TO_RUN = BASELINES + GRASYNDA_VARIANTS_TO_RUN + OTHER_DA_METHODS

DATASETS_TO_TEST = [
    ('M3', 'Monthly'),
    ('M3', 'Quarterly'),
    ('Gluonts', 'm1_monthly'),
    ('Gluonts', 'm1_quarterly'),
    ('NN3', 'Monthly'),
    ('Tourism', 'Monthly'),
    ('Tourism', 'Quarterly'),
]

# Number of synthetic series to generate per original series
N_SYNTH_PER_UID = 1 

# Output files
out_dir = os.path.join(
    project_root, "assets", "results", "pymdma_metrics", "all_datasets_all_methods_no_tsdiff_20260323"
)
os.makedirs(out_dir, exist_ok=True)

CHECKPOINT_FILENAME = os.path.join(out_dir, "checkpoint_comprehensive_pymdma.csv")
FINAL_RESULTS_FILENAME = os.path.join(out_dir, "final_comprehensive_pymdma_results.csv")
SUMMARY_FILENAME = os.path.join(out_dir, "summary_comprehensive_pymdma.csv")


# ================= OPTIMIZED PARAMETERS =================

OPTIMAL_PARAMS = {
    "TSMixup": {
        ("M3", "Monthly"):    {"max_n_uids": 3, "dirichlet_alpha": 2.0},
        ("M3", "Quarterly"): {"max_n_uids": 3, "dirichlet_alpha": 2.0},
        ("Tourism", "Monthly"):    {"max_n_uids": 3, "dirichlet_alpha": 5.0},
        ("Tourism", "Quarterly"): {"max_n_uids": 3, "dirichlet_alpha": 0.5},
    },
    "TimeVAE": {
        ("M3", "Monthly"):    {"latent_dim": 16, "reconstruction_wt": 5.0, "max_epochs": 100},
        ("M3", "Quarterly"): {"latent_dim": 4, "reconstruction_wt": 5.0, "max_epochs": 100},
        ("Tourism", "Monthly"):    {"latent_dim": 16, "reconstruction_wt": 5.0, "max_epochs": 100},
        ("Tourism", "Quarterly"): {"latent_dim": 16, "reconstruction_wt": 5.0, "max_epochs": 50},
    },
    "DBA": {
        ("M3", "Monthly"):    {"max_n_uids": 2, "dirichlet_alpha": 1.0, "max_iter": 10},
        ("M3", "Quarterly"): {"max_n_uids": 2, "dirichlet_alpha": 3.0, "max_iter": 10},
        ("Tourism", "Monthly"):    {"max_n_uids": 2, "dirichlet_alpha": 2.5, "max_iter": 5},
    },
    "MagnitudeWarping": {
        ("M3", "Monthly"):    {"sigma": 0.02, "knot": 3},
        ("M3", "Quarterly"): {"sigma": 0.05, "knot": 3},
        ("Tourism", "Monthly"):    {"sigma": 0.02, "knot": 4},
        ("Tourism", "Quarterly"): {"sigma": 0.08, "knot": 3},
    }
}

DEFAULT_PARAMS = {
    "TSMixup": {"max_n_uids": 3, "dirichlet_alpha": 1.0},
    "TimeVAE": {"latent_dim": 8, "reconstruction_wt": 3.0, "max_epochs": 100},
    "DBA": {"max_n_uids": 2, "dirichlet_alpha": 1.0, "max_iter": 10},
    "MagnitudeWarping": {"sigma": 0.2, "knot": 4},
}


# ================= HELPER FUNCTIONS =================

def load_series_as_lists(df: pd.DataFrame):
    groups = df.groupby('unique_id')
    series_list = []
    series_ids = []

    for uid, group_df in groups:
        values = group_df.sort_values('ds')['y'].values
        series_list.append(values)
        series_ids.append(uid)

    return series_list, series_ids


def truncate_to_length(series_list, T: int) -> np.ndarray:
    return np.asarray([series[-T:] for series in series_list])


def lists_to_tabular(series_list, T: int, series_ids=None) -> pd.DataFrame:
    values = truncate_to_length(series_list, T=T)
    col_names = [f't_{i}' for i in range(T)]
    tabular_df = pd.DataFrame(values, columns=col_names, index=series_ids)
    return tabular_df


def extract_synthetic_only(augmented_df: pd.DataFrame, real_df: pd.DataFrame) -> pd.DataFrame:
    real_ids = set(real_df['unique_id'].unique())
    synth_only = augmented_df[~augmented_df['unique_id'].isin(real_ids)].copy()
    
    if not synth_only.empty:
        overlap = set(synth_only['unique_id'].unique()) & real_ids
        if overlap:
            synth_only = synth_only[~synth_only['unique_id'].isin(overlap)]
            
    return synth_only


def get_grasynda_instance(method_name, freq_int):
    if method_name == 'Hybrid_VisH_Ensemble5':
        return GrasyndaUnified(period=freq_int, sampling_type='continuous_uniform', graph_type='visibility', visibility_type='horizontal', ensemble_transitions=True, ensemble_size=5, components_to_model=['trend', 'remainder'], component_params={'trend': {'apply_differentiation': True}})
    elif method_name == 'Hybrid_Q10_Ensemble5_Continuous':
        return GrasyndaUnified(period=freq_int, n_quantiles=10, sampling_type='continuous_uniform', graph_type='quantile', ensemble_transitions=True, ensemble_size=5, components_to_model=['trend', 'remainder'], component_params={'trend': {'apply_differentiation': True}})
    elif method_name == 'Hybrid_Q10_NoEnsemble_Continuous':
        return GrasyndaUnified(period=freq_int, n_quantiles=10, sampling_type='continuous_uniform', graph_type='quantile', ensemble_transitions=False, components_to_model=['trend', 'remainder'], component_params={'trend': {'apply_differentiation': True}})
    elif method_name == 'Hybrid_SAX_Q10_Ensemble5_Continuous':
        return GrasyndaSAX(period=freq_int, n_symbols=10, n_sax_windows=1, sax_normalize=True, sampling_type='continuous_uniform', graph_type='quantile', ensemble_transitions=True, ensemble_size=5, components_to_model=['trend', 'remainder'], component_params={'trend': {'apply_differentiation': True}})
    elif method_name == 'Hybrid_SAX_Q10_NoEnsemble_Continuous':
        return GrasyndaSAX(period=freq_int, n_symbols=10, n_sax_windows=1, sax_normalize=True, sampling_type='continuous_uniform', graph_type='quantile', ensemble_transitions=False, components_to_model=['trend', 'remainder'], component_params={'trend': {'apply_differentiation': True}})
    else:
        raise ValueError(f"Unknown Grasynda method: {method_name}")


# ================= EVALUATION LOOP =================

def run_evaluation():
    all_results = []

    auth = Authenticity()
    imp_prec = ImprovedPrecision()
    imp_rec = ImprovedRecall()
    dcr_priv = DCRPrivacy()

    for ds_name, group in DATASETS_TO_TEST:
        print(f"\n### Evaluating {ds_name} - {group} ###")
        data_loader = DATASETS[ds_name]
        df_real, horizon, _, freq_str, freq_int = data_loader.load_everything(group)
        real_list, real_ids = load_series_as_lists(df_real)
        
        if 'Monthly' in group: freq_int = 12
        elif 'Quarterly' in group: freq_int = 4
        
        # Determine constraints for DA
        max_len = df_real['unique_id'].value_counts().max() - (2 * horizon)
        min_len = df_real['unique_id'].value_counts().min() - (2 * horizon)
        n_uids = df_real['unique_id'].nunique()
        max_n_uids = max(2, int(np.round(np.log(n_uids), 0)))

        for method in METHODS_TO_RUN:
            print(f"  -> Method: {method}")
            start_time = time.time()
            synth_df = None
            synth_list = None

            if method == 'Exact_Copy':
                lengths = [len(series) for series in real_list]
                T = int(np.min(lengths))
                real_eval = lists_to_tabular(real_list, T=T, series_ids=real_ids)
                synth_eval = real_eval.copy()
            elif method == 'Random_Noise':
                lengths = [len(series) for series in real_list]
                T = int(np.min(lengths))
                real_eval = lists_to_tabular(real_list, T=T, series_ids=real_ids)
                real_std = real_eval.std(ddof=0).replace(0, 1.0)
                synth_eval = pd.DataFrame(
                    np.random.randn(*real_eval.shape) * real_std.to_numpy() + real_eval.mean().to_numpy(),
                    columns=real_eval.columns,
                )
            elif method in SYNTH_METHODS:
                params = {'seas_period': freq_int, 'freq': freq_str, 'max_n_uids': max_n_uids, 'min_len': min_len, 'max_len': max_len}
                if method in DEFAULT_PARAMS: params.update(DEFAULT_PARAMS[method])
                if method in OPTIMAL_PARAMS and (ds_name, group) in OPTIMAL_PARAMS[method]: params.update(OPTIMAL_PARAMS[method][(ds_name, group)])
                
                aug_df = ExpWorkflow.get_offline_augmented_data(df_real, method, params, n_series_by_uid=N_SYNTH_PER_UID)
                synth_df = extract_synthetic_only(aug_df, df_real)
            elif method in GRASYNDA_VARIANTS_TO_RUN:
                generator = get_grasynda_instance(method, freq_int)
                synth_df = generator.transform(df_real).copy()
                synth_df = extract_synthetic_only(synth_df, df_real)

            if method not in BASELINES:
                if synth_df is None or synth_df.empty:
                    print(f"     [SKIP] Empty synthetic data for {method}")
                    continue
                synth_list, synth_ids = load_series_as_lists(synth_df)
                lengths = [len(series) for series in real_list]
                lengths.extend(len(series) for series in synth_list)
                T = int(np.min(lengths))
                if T <= 1:
                    raise ValueError(
                        f"Common truncation length too small (T={T}) for {ds_name} {group}"
                    )

                real_eval = lists_to_tabular(real_list, T=T, series_ids=real_ids)
                synth_eval = lists_to_tabular(synth_list, T=T, series_ids=synth_ids)

            # Scaling and Data Prep
            eval_loader = TabularDataset(file_path=None, data=real_eval, tag_name='real', scaler='standard', imputer='knn', with_onehot=False)
            real_scaled = eval_loader.data_s
            
            _, _, synth_scaled, _ = eval_loader.transform(data=synth_eval, scale_fit=False, meta_fit=False)

  
            auth_s = auth.compute(real_scaled, synth_scaled).value[0]
            fid_s = imp_prec.compute(real_scaled, synth_scaled).value[0]
            div_s = imp_rec.compute(real_scaled, synth_scaled).value[0]
            priv_s = dcr_priv.compute(real_scaled, synth_scaled).value[0]['privacy'] / 100
            
            res = {
                'Dataset': ds_name, 'Group': group, 'Method': method,
                'Authenticity': auth_s, 'Fidelity': fid_s, 'Diversity': div_s, 'Privacy': priv_s,
                'Time_Sec': time.time() - start_time, 'N_Synth': len(synth_eval)
            }
            all_results.append(res)
            print(f"     [OK] Auth: {auth_s:.3f}, Fid: {fid_s:.3f}, Priv: {priv_s:.3f}, Div: {div_s:.3f}, Time: {res['Time_Sec']:.1f}s, N_Synth: {res['N_Synth']}")
            

            pd.DataFrame(all_results).to_csv(CHECKPOINT_FILENAME, index=False)

    if all_results:
        final_df = pd.DataFrame(all_results)
        final_df.to_csv(FINAL_RESULTS_FILENAME, index=False)
        summary = final_df.groupby('Method')[['Authenticity', 'Fidelity', 'Diversity', 'Privacy']].mean()
        summary.to_csv(SUMMARY_FILENAME)
        print(f"\n### DONE ###\nResults: {FINAL_RESULTS_FILENAME}\nSummary: {SUMMARY_FILENAME}")

if __name__ == "__main__":
    run_evaluation()
