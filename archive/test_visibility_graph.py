"""
Test script for Grasynda Visibility Graph variant.

This script demonstrates how to use the new visibility graph-based
Grasynda implementation and compare it with the standard quantile-based approach.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
from functools import partial

from neuralforecast import NeuralForecast
from statsforecast.models import SeasonalNaive
from statsforecast import StatsForecast
from utilsforecast.losses import mase, smape
from utilsforecast.evaluation import evaluate

from utils.load_data.config import DATASETS
from utils.config import MODEL_CONFIG, MODELS
from utils.load_data.base import LoadDataset
from src.qgraph_ts import Grasynda  # Original quantile-based
from src.grasynda_visibility import GrasyndaVisibilityGraph

# Configuration
MODEL = 'NHITS'
DATA_GROUPS = [
    ('Gluonts', 'm1_monthly'),
    ('Gluonts', 'm1_quarterly'),
    ('M3', 'Monthly'),
    ('M3', 'Quarterly'),
    ('Tourism', 'Monthly'),
    ('Tourism', 'Quarterly'),
]

print("=" * 80)
print("RUNNING GRASYNDA VISIBILITY GRAPH EXPERIMENTS ON ALL DATASETS")
print("=" * 80)

all_results = []

for idx, (data_name, group) in enumerate(DATA_GROUPS):
    print(f"\n{'=' * 80}")
    print(f"DATASET {idx+1}/6: {data_name} - {group}")
    print(f"{'=' * 80}")

    # Load data
    data_loader = DATASETS[data_name]
    min_samples = data_loader.min_samples[group]
    print(f'Min samples: {min_samples}')

    df, horizon, n_lags, freq_str, freq_int = data_loader.load_everything(
        group, min_n_instances=min_samples
    )

    print(f"Data shape: {df.shape}")
    print(f"Unique IDs: {df['unique_id'].nunique()}")
    print(f"Horizon: {horizon}, Lags: {n_lags}, Freq: {freq_str}")

    # Split data
    train, test = LoadDataset.train_test_split(df, horizon)

    # Prepare training sets
    training_sets = {}

    # 1. Original (baseline)
    training_sets['original'] = train.copy()

    # 2. Grasynda Uniform (Original quantile-based)
    print("\n--- Generating Grasynda Uniform (Baseline) ---")
    grasynda_uniform = Grasynda(
        n_quantiles=25,
        quantile_on='remainder',
        period=freq_int,
        ensemble_transitions=False
    )
    synth_uniform = grasynda_uniform.transform(train)
    train_uniform = pd.concat([train, synth_uniform]).reset_index(drop=True)
    training_sets['grasynda_uniform'] = train_uniform
    
    # 3. Grasynda Visibility - Degree Matching (Raw Data)
    print("\n--- Generating Grasynda Visibility (Degree Matching) ---")
    grasynda_vis_degree = GrasyndaVisibilityGraph(
        period=freq_int,
        visibility_type='horizontal',
        generation_method='degree_matching',
        use_decomposition=False
    )
    synth_vis_degree = grasynda_vis_degree.transform(train)
    train_vis_degree = pd.concat([train, synth_vis_degree]).reset_index(drop=True)
    training_sets['grasynda_vis_degree'] = train_vis_degree

    # 4. Grasynda Visibility - Hybrid (Raw Data)
    print("\n--- Generating Grasynda Visibility (Hybrid) ---")
    grasynda_vis_hybrid = GrasyndaVisibilityGraph(
        period=freq_int,
        visibility_type='horizontal',
        generation_method='hybrid',
        use_decomposition=False
    )
    synth_vis_hybrid = grasynda_vis_hybrid.transform(train)
    train_vis_hybrid = pd.concat([train, synth_vis_hybrid]).reset_index(drop=True)
    training_sets['grasynda_vis_hybrid'] = train_vis_hybrid

    # Train and evaluate models
    print("\n--- Training NHITS Models ---")
    
    input_data = {'input_size': n_lags, 'h': horizon}
    test_with_fcst = test.copy()

    for tsgen, train_df_ in training_sets.items():
        print(f"Training with: {tsgen}")
        model_params = MODEL_CONFIG.get(MODEL)
        model_conf = {**input_data, **model_params}
        
        nf = NeuralForecast(
            models=[MODELS[MODEL](**model_conf, alias=tsgen)],
            freq=freq_str
        )
        nf.fit(df=train_df_, val_size=horizon)
        
        fcst = nf.predict()
        test_with_fcst = test_with_fcst.merge(
            fcst.reset_index(), on=['unique_id', 'ds'], how="left"
        )

    # Baseline
    print("Training SeasonalNaive baseline...")
    stats_models = [SeasonalNaive(season_length=freq_int)]
    sf = StatsForecast(models=stats_models, freq=freq_str, n_jobs=1)
    sf.fit(train)
    sf_fcst = sf.predict(h=horizon)
    test_with_fcst = test_with_fcst.merge(
        sf_fcst.reset_index(), on=['unique_id', 'ds'], how="left"
    )

    # Evaluation
    print("\n--- Evaluating ---")
    evaluation_df = evaluate(
        test_with_fcst,
        [partial(mase, seasonality=freq_int), smape],
        train_df=train
    )

    # Extract MASE results
    mase_results = evaluation_df.query('metric=="mase"').mean(numeric_only=True)
    
    print(f"\nMASE Results for {data_name} - {group}:")
    print(mase_results)

    # Store results
    result_row = {
        'dataset': data_name,
        'group': group,
        'SeasonalNaive': mase_results.get('SeasonalNaive', np.nan),
        'original': mase_results.get('original', np.nan),
        'grasynda_uniform': mase_results.get('grasynda_uniform', np.nan),
        'grasynda_vis_degree': mase_results.get('grasynda_vis_degree', np.nan),
        'grasynda_vis_hybrid': mase_results.get('grasynda_vis_hybrid', np.nan),
    }
    all_results.append(result_row)

# FINAL SUMMARY
print("\n" + "=" * 80)
print("FINAL SUMMARY - MASE ACROSS ALL 6 DATASETS")
print("=" * 80)

results_df = pd.DataFrame(all_results)
print(results_df.to_string(index=False))

print("\n" + "=" * 80)
print("AVERAGE MASE ACROSS ALL DATASETS:")
print("=" * 80)
numeric_cols = ['SeasonalNaive', 'original', 'grasynda_uniform', 'grasynda_vis_degree', 'grasynda_vis_hybrid']
avg_results = results_df[numeric_cols].mean()
print(avg_results)

# Save results
output_path = 'assets/results/visibility_graph_mase_results.csv'
results_df.to_csv(output_path, index=False)
print(f"\nResults saved to: {output_path}")

print("\n" + "=" * 80)
print("EXPERIMENT COMPLETE")
print("=" * 80)
