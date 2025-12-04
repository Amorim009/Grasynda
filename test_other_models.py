import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import numpy as np
import pandas as pd
from functools import partial

from neuralforecast import NeuralForecast
from statsforecast.models import SeasonalNaive
from statsforecast import StatsForecast
from utilsforecast.losses import mase, smape
from utilsforecast.evaluation import evaluate

from utils.load_data.config import DATASETS
from utils.config import MODEL_CONFIG, MODELS, SYNTH_METHODS
from utils.load_data.base import LoadDataset
from src.workflow import ExpWorkflow

MODEL = 'NHITS'

# All 6 datasets
DATA_GROUPS = [
    ('Gluonts', 'm1_monthly'),
    ('Gluonts', 'm1_quarterly'),
    ('M3', 'Monthly'),
    ('M3', 'Quarterly'),
    ('Tourism', 'Monthly'),
    ('Tourism', 'Quarterly'),
]

print("=" * 80)
print("RUNNING BENCHMARK FOR OTHER AUGMENTATION MODELS")
print("=" * 80)

all_results = []

for idx, (data_name, group) in enumerate(DATA_GROUPS):
    print(f"\n{'=' * 80}")
    print(f"DATASET {idx+1}/6: {data_name} - {group}")
    print(f"{'=' * 80}")
    
    # LOADING DATA AND SETUP
    data_loader = DATASETS[data_name]
    min_samples = data_loader.min_samples[group]
    print(f'Min samples: {min_samples}')
    
    df, horizon, n_lags, freq_str, freq_int = data_loader.load_everything(
        group, min_n_instances=min_samples
    )
    
    print(f"Data shape: {df.shape}")
    print(f"Unique IDs: {df['unique_id'].nunique()}")
    print(f"Horizon: {horizon}, Lags: {n_lags}, Freq: {freq_str}")
    
    # DATA SPLITS
    train, test = LoadDataset.train_test_split(df, horizon)
    
    # AUGMENTATION PARAMETERS
    max_len = df['unique_id'].value_counts().max() - (2 * horizon)
    min_len = df['unique_id'].value_counts().min() - (2 * horizon)
    n_uids = df['unique_id'].nunique()
    max_n_uids = int(np.round(np.log(n_uids), 0))
    max_n_uids = 2 if max_n_uids < 2 else max_n_uids
    
    augmentation_params = {
        'seas_period': freq_int,
        'max_n_uids': max_n_uids,
        'max_len': max_len,
        'min_len': min_len,
    }
    
    # PREPARE TRAINING SETS
    training_sets = {}
    
    # 1. Original (baseline)
    training_sets['original'] = train.copy()
    
    # 2. Other Augmentation Methods
    print("\n--- Generating Augmented Data ---")
    for method_name in SYNTH_METHODS:
        print(f"Generating data for: {method_name}")
        try:
            train_aug = ExpWorkflow.get_offline_augmented_data(
                train_=train,
                generator_name=method_name,
                augmentation_params=augmentation_params,
                n_series_by_uid=1
            )
            training_sets[method_name] = train_aug
        except Exception as e:
            print(f"Failed to generate data for {method_name}: {e}")
    
    # MODELING
    input_data = {'input_size': n_lags, 'h': horizon}
    test_with_fcst = test.copy()
    
    print("\n--- Training NHITS Models ---")
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
    
    # BASELINE
    print("Training SeasonalNaive baseline...")
    stats_models = [SeasonalNaive(season_length=freq_int)]
    sf = StatsForecast(models=stats_models, freq=freq_str, n_jobs=1)
    sf.fit(train)
    sf_fcst = sf.predict(h=horizon)
    test_with_fcst = test_with_fcst.merge(
        sf_fcst.reset_index(), on=['unique_id', 'ds'], how="left"
    )
    
    # EVALUATION
    print("\n--- Evaluating ---")
    evaluation_df = evaluate(
        test_with_fcst,
        [partial(mase, seasonality=freq_int), smape],
        train_df=train
    )
    
    # Extract MASE results
    mase_results = evaluation_df.query('metric=="mase"').mean(numeric_only=True)
    
    print(f"\n{'=' * 60}")
    print(f"MASE RESULTS FOR {data_name} - {group}:")
    print(f"{'=' * 60}")
    print(mase_results)
    
    # Store results
    result_row = {
        'dataset': data_name,
        'group': group,
        'SeasonalNaive': mase_results.get('SeasonalNaive', np.nan),
        'original': mase_results.get('original', np.nan),
    }
    
    for method_name in SYNTH_METHODS:
        result_row[method_name] = mase_results.get(method_name, np.nan)
        
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
numeric_cols = ['SeasonalNaive', 'original'] + list(SYNTH_METHODS.keys())
avg_results = results_df[numeric_cols].mean()
print(avg_results)

# Save results
output_path = 'assets/results/other_models_mase_results.csv'
results_df.to_csv(output_path, index=False)
print(f"\nResults saved to: {output_path}")
