"""
PyMDMA Synthesis Validation Metrics.

Computes Authenticity, Fidelity, Diversity, and Privacy using pymdma.tabular.
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

print(f"Project root: {project_root}")

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

DATASETS_TO_TEST = [
    ('Gluonts', 'm1_monthly'),
    ('Gluonts', 'm1_quarterly'),
    ('M3', 'Monthly'),
    ('M3', 'Quarterly'),
    ('NN3', 'Monthly'),
    ('Tourism', 'Monthly'),
    ('Tourism', 'Quarterly'),
]


GRASYNDA_VARIANTS = [
    'Hybrid_VisH_Ensemble5',
    # 'Hybrid_Q10_Ensemble5_Continuous',
    # 'Hybrid_Q10_NoEnsemble_Continuous',
    'Hybrid_Q10_NoEnsemble_NoDiff',
    'Hybrid_Q10_NoEnsemble_NoDiff',
]

GRASYNDA_SAX_VARIANTS = [
    'Hybrid_SAX_Q10_Ensemble5_Continuous',
    'Hybrid_SAX_Q10_NoEnsemble_Continuous',
    'Hybrid_SAX_Q10_Ensemble5_RawY',
    'Hybrid_SAX_Q10_NoEnsemble_RawY',
]

# Single-run method list: comment out what you do NOT want to run.
METHODS_TO_RUN = [
    # 'Exact_Copy',
    # 'Random_Noise',
    # 'SeasonalMBB',
    # 'Jittering',
    'Scaling',
    # 'TimeWarping',
    # 'MagnitudeWarping',
    # 'TSMixup',
    # 'DBA',
    # 'Hybrid_VisH_Ensemble5',
    # 'Hybrid_Q10_Ensemble5_Continuous',
    # 'Hybrid_Q10_NoEnsemble_Continuous',
    # 'Hybrid_Q10_NoEnsemble_NoDiff',
    # 'Hybrid_SAX_Q10_Ensemble5_Continuous',
    # 'Hybrid_SAX_Q10_NoEnsemble_Continuous',
    # 'Hybrid_SAX_Q10_Ensemble5_RawY',
    # 'Hybrid_SAX_Q10_NoEnsemble_RawY',
]

# Number of synthetic series to generate per original series for DA methods.
N_SYNTH_PER_UID = 1

# Output files
CHECKPOINT_FILENAME = "checkpoint_scaling_default.csv"
FINAL_RESULTS_FILENAME = "final_results_scaling_default.csv"


# ================= DATA HANDLING =================

def timeseries_to_tabular(df: pd.DataFrame) -> pd.DataFrame:
    groups = df.groupby('unique_id')
    min_len = groups['y'].count().min()

    rows = []
    for _, group_df in groups:
        values = group_df.sort_values('ds')['y'].values
        values = values[-min_len:]
        rows.append(values)

    col_names = [f't_{i}' for i in range(min_len)]
    tabular_df = pd.DataFrame(rows, columns=col_names)

    return tabular_df


def extract_synthetic_only(augmented_df: pd.DataFrame, real_df: pd.DataFrame) -> pd.DataFrame:
    real_ids = real_df['unique_id'].unique()
    synth_only = augmented_df[~augmented_df['unique_id'].isin(real_ids)].copy()
    
    return synth_only


# ================= GRASYNDA FACTORY =================

def get_grasynda_instance(method_name, freq_int):
    """Return configured GrasyndaUnified instance for the given method."""
    
    if method_name == 'Hybrid_VisH_Ensemble5':
        return GrasyndaUnified(
            period=freq_int,
            sampling_type='continuous_uniform',
            graph_type='visibility',
            visibility_type='horizontal',
            ensemble_transitions=True,
            ensemble_size=5,
            components_to_model=['trend', 'remainder'],
            component_params={
                'trend': {'apply_differentiation': True}
            }
        )
    
    elif method_name == 'Hybrid_Q10_Ensemble5_Continuous':
        return GrasyndaUnified(
            period=freq_int,
            n_quantiles=10,
            sampling_type='continuous_uniform',
            graph_type='quantile',
            ensemble_transitions=True,
            ensemble_size=5,
            components_to_model=['trend', 'remainder'],
            component_params={
                'trend': {'apply_differentiation': True}
            }
        )
    
    elif method_name == 'Hybrid_Q10_NoEnsemble_Continuous':
        return GrasyndaUnified(
            period=freq_int,
            n_quantiles=10,
            sampling_type='continuous_uniform',
            graph_type='quantile',
            ensemble_transitions=False,
            components_to_model=['trend', 'remainder'],
            component_params={
                'trend': {'apply_differentiation': True}
            }
        )


    elif method_name == 'Hybrid_Q10_NoEnsemble_NoDiff':
        return GrasyndaUnified(
            period=freq_int,
            n_quantiles=10,
            sampling_type='continuous_uniform',
            graph_type='quantile',
            ensemble_transitions=False,
            components_to_model=['trend', 'remainder'],
            component_params={
                'trend': {'apply_differentiation': False}
            }
        )

    elif method_name == 'Hybrid_SAX_Q10_Ensemble5_Continuous':
        return GrasyndaSAX(
            period=freq_int,
            n_symbols=10,
            n_sax_windows=1,
            sax_normalize=True,
            sampling_type='continuous_uniform',
            graph_type='quantile',
            ensemble_transitions=True,
            ensemble_size=5,
            components_to_model=['trend', 'remainder'],
            component_params={
                'trend': {'apply_differentiation': True}
            }
        )

    elif method_name == 'Hybrid_SAX_Q10_NoEnsemble_Continuous':
        return GrasyndaSAX(
            period=freq_int,
            n_symbols=10,
            n_sax_windows=1,
            sax_normalize=True,
            sampling_type='continuous_uniform',
            graph_type='quantile',
            ensemble_transitions=False,
            components_to_model=['trend', 'remainder'],
            component_params={
                'trend': {'apply_differentiation': True}
            }
        )

    elif method_name == 'Hybrid_SAX_Q10_Ensemble5_RawY':
        return GrasyndaSAX(
            period=freq_int,
            n_symbols=10,
            n_sax_windows=1,
            sax_normalize=True,
            sampling_type='continuous_uniform',
            graph_type='quantile',
            ensemble_transitions=True,
            ensemble_size=5,
            components_to_model=['y'],
        )

    elif method_name == 'Hybrid_SAX_Q10_NoEnsemble_RawY':
        return GrasyndaSAX(
            period=freq_int,
            n_symbols=10,
            n_sax_windows=1,
            sax_normalize=True,
            sampling_type='continuous_uniform',
            graph_type='quantile',
            ensemble_transitions=False,
            components_to_model=['y'],
        )
    
    else:
        raise ValueError(f"Unknown Grasynda method: {method_name}")


# ================= EVALUATION =================

def run_dataset_evaluation(dataset_name, group, methods_to_run):
    print(f"\nEvaluating {dataset_name} ({group})")

    data_loader = DATASETS[dataset_name]
    min_samples_cfg = data_loader.min_samples.get(group, 10)
    df_all, horizon, _, _, freq_int = data_loader.load_everything(
        group, min_n_instances=min_samples_cfg
    )
    
    if 'Monthly' in group or 'm1_monthly' in group:
        freq_int = 12

    df_real = df_all.copy()
    real_tabular = timeseries_to_tabular(df_real)
    loader_cache = {}

    max_len = df_real['unique_id'].value_counts().max() - (2 * horizon)
    min_len = df_real['unique_id'].value_counts().min() - (2 * horizon)
    max_len = max(10, max_len)
    min_len = max(10, min_len)
    
    n_uids = df_real['unique_id'].nunique()
    max_n_uids = int(np.round(np.log(n_uids), 0))
    max_n_uids = 2 if max_n_uids < 2 else max_n_uids
    
    auth = Authenticity()
    imp_prec = ImprovedPrecision()
    imp_rec = ImprovedRecall()
    dcr_priv = DCRPrivacy()
    
    results = []

    for method_name in methods_to_run:
        print(f"  -> {method_name}")
        start = time.time()
        if method_name == 'Exact_Copy':
            synth_tabular = real_tabular.copy()

        elif method_name == 'Random_Noise':
            synth_tabular = pd.DataFrame(
                np.random.randn(*real_tabular.shape),
                columns=real_tabular.columns
            )

        elif method_name in SYNTH_METHODS:
            augmentation_params = {
                'seas_period': freq_int,
                'max_n_uids': max_n_uids,
                'min_len': min_len,
                'max_len': max_len,
            }
            augmented_df = ExpWorkflow.get_offline_augmented_data(
                df_real, method_name, augmentation_params, n_series_by_uid=N_SYNTH_PER_UID
            )
            synth_df = extract_synthetic_only(augmented_df, df_real)

        elif method_name in (GRASYNDA_VARIANTS + GRASYNDA_SAX_VARIANTS):
            generator = get_grasynda_instance(method_name, freq_int)
            synth_list = []
            for i in range(N_SYNTH_PER_UID):
                synth_i = generator.transform(df_real).copy()
                if N_SYNTH_PER_UID > 1:
                    synth_i['unique_id'] = synth_i['unique_id'].astype(str) + f"_r{i}"
                synth_list.append(synth_i)
            synth_df = pd.concat(synth_list, ignore_index=True)


        if method_name not in ('Exact_Copy', 'Random_Noise'):
            if synth_df is None or synth_df.empty:
                print("     Failed: empty synthetic data.")
                continue
            synth_tabular = timeseries_to_tabular(synth_df)

        n_syn_cols = synth_tabular.shape[1]
        n_real_cols = real_tabular.shape[1]
        min_cols = min(n_syn_cols, n_real_cols)
        if min_cols <= 0:
            print("     Failed: tabular conversion produced zero columns.")
            continue

        cols = [f't_{i}' for i in range(min_cols)]
        real_eval = real_tabular.loc[:, cols]
        synth_eval = synth_tabular.loc[:, cols]

        if min_cols not in loader_cache:
            eval_loader = TabularDataset(
                file_path=None,
                data=real_eval,
                tag_name='real',
                scaler='standard',
                scaler_kwargs={},
                embed=None,
                embed_kwargs={},
                imputer='knn',
                imputer_kwargs={},
                with_onehot=False
            )
            loader_cache[min_cols] = (eval_loader, eval_loader.data_s)

        eval_loader, real_scaled_use = loader_cache[min_cols]
        _, _, syn_scaled, _ = eval_loader.transform(
            data=synth_eval, scale_fit=False, meta_fit=False
        )

        auth_score = auth.compute(real_scaled_use, syn_scaled).value[0]
        fidelity_score = imp_prec.compute(real_scaled_use, syn_scaled).value[0]
        diversity_score = imp_rec.compute(real_scaled_use, syn_scaled).value[0]
        privacy_score = dcr_priv.compute(real_scaled_use, syn_scaled).value[0]['privacy'] / 100
        elapsed = time.time() - start
        print(
            f"     ok privacy={privacy_score:.4f} "
            f"fidelity={fidelity_score:.4f} "
            f"time={elapsed:.1f}s"
        )

        results.append({
            'Dataset': dataset_name,
            'Group': group,
            'Method': method_name,
            'Authenticity': auth_score,
            'Fidelity': fidelity_score,
            'Diversity': diversity_score,
            'Privacy': privacy_score,
            'Time_Sec': elapsed
        })

    return results


# ================= MAIN =================
if __name__ == "__main__":
    out_dir = os.path.join(project_root, "assets", "results", "pymdma_metrics", "corrected")
    os.makedirs(out_dir, exist_ok=True)

    known_methods = (
        {'Exact_Copy', 'Random_Noise'}
        | set(SYNTH_METHODS.keys())
        | set(GRASYNDA_VARIANTS)
        | set(GRASYNDA_SAX_VARIANTS)
    )
    unknown_methods = [m for m in METHODS_TO_RUN if m not in known_methods]
    if unknown_methods:
        raise ValueError(f"Unknown methods in METHODS_TO_RUN: {unknown_methods}")
    
    all_results = []
    
    for ds_name, group in DATASETS_TO_TEST:
        res = run_dataset_evaluation(ds_name, group, METHODS_TO_RUN)
        all_results.extend(res)
        
        # Incremental checkpoint
        pd.DataFrame(all_results).to_csv(
            os.path.join(out_dir, CHECKPOINT_FILENAME), index=False
        )
    
    if all_results:
        final_df = pd.DataFrame(all_results)
        final_path = os.path.join(out_dir, FINAL_RESULTS_FILENAME)
        final_df.to_csv(final_path, index=False)

        metrics = ['Authenticity', 'Fidelity', 'Diversity', 'Privacy']

        summary = final_df.groupby('Method')[metrics].mean()
        summary_path = os.path.join(out_dir, "summary_scaling_default.csv")
        summary.to_csv(summary_path)

        final_df['DS'] = final_df['Dataset'] + '_' + final_df['Group']
        rank_dfs = []
        for ds_key, ds_df in final_df.groupby('DS'):
            ds_pivot = ds_df.set_index('Method')[metrics]
            ds_ranks = ds_pivot.rank(ascending=False, method='min')
            rank_dfs.append(ds_ranks)

        avg_ranks = pd.concat(rank_dfs).groupby(level=0).mean()
        avg_ranks['Avg_Rank'] = avg_ranks.mean(axis=1)
        avg_ranks = avg_ranks.sort_values('Avg_Rank')
        rankings_path = os.path.join(out_dir, "rankings_scaling_default.csv")
        avg_ranks.to_csv(rankings_path)
        print(f"Saved results: {final_path}")
        print(f"Saved summary: {summary_path}")
        print(f"Saved rankings: {rankings_path}")
    else:
        print("No results found.")

