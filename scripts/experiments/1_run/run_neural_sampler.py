import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import numpy as np
import pandas as pd

from neuralforecast import NeuralForecast
from statsforecast.models import SeasonalNaive
from statsforecast import StatsForecast
from utilsforecast.losses import mase, smape
from utilsforecast.evaluation import evaluate
from functools import partial

from utils.load_data.config import DATASETS
from utils.config import MODEL_CONFIG, MODELS
from utils.load_data.base import LoadDataset
from src.qgraph_ts import Grasynda, GrasyndaKDE, GrasyndaNeuralSampler

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
print("RUNNING NEURAL SAMPLER EXPERIMENTS ON ALL 6 DATASETS")
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
    
    # PREPARE TRAINING SETS
    training_sets = {}
    
    # 1. Original (baseline)
    training_sets['original'] = train.copy()
    
    # 2. Grasynda with Neural Sampler
    print("\n--- Training Neural Sampler ---")
    grasynda_neural = GrasyndaNeuralSampler(
        n_quantiles=25,
        quantile_on='remainder',
        period=freq_int,
        ensemble_transitions=False
    )
    
    # Decompose and prepare data for training
    train_decomp = grasynda_neural.decompose_tsd(train, period=freq_int, robust=False)
    train_decomp['Quantile'] = grasynda_neural._get_quantiles(train_decomp)
    
    # Train the neural sampler on REAL data
    print("Training neural sampler (10 epochs)...")
    grasynda_neural.train_neural_sampler(train_decomp, epochs=10, batch_size=32)
    
    # Generate synthetic data using neural sampler
    # Generate synthetic data using neural sampler
    print("Generating synthetic data with neural sampler...")
    synth_neural = grasynda_neural.transform(train)
    train_neural = pd.concat([train, synth_neural]).reset_index(drop=True)
    training_sets['grasynda_neural'] = train_neural
    
    # 3. Grasynda with KDE Sampling (Original)
    print("\n--- Running Grasynda (KDE) ---")
    grasynda_kde = GrasyndaKDE(
        n_quantiles=25,
        quantile_on='remainder',
        period=freq_int,
        ensemble_transitions=False
    )
    synth_kde = grasynda_kde.transform(train)
    train_kde = pd.concat([train, synth_kde]).reset_index(drop=True)
    training_sets['grasynda_kde'] = train_kde

    # 4. Grasynda with Uniform Sampling (Default)
    print("\n--- Running Grasynda (Uniform) ---")
    grasynda_uniform = Grasynda(
        n_quantiles=25,
        quantile_on='remainder',
        period=freq_int,
        ensemble_transitions=False
    )
    synth_uniform = grasynda_uniform.transform(train)
    train_uniform = pd.concat([train, synth_uniform]).reset_index(drop=True)
    training_sets['grasynda_uniform'] = train_uniform
    
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
        'original': mase_results.get('original', np.nan),
        'grasynda_neural': mase_results.get('grasynda_neural', np.nan),
        'grasynda_kde': mase_results.get('grasynda_kde', np.nan),
        'grasynda_uniform': mase_results.get('grasynda_uniform', np.nan),
        'SeasonalNaive': mase_results.get('SeasonalNaive', np.nan),
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
avg_results = results_df[['original', 'grasynda_neural', 'grasynda_kde', 'grasynda_uniform', 'SeasonalNaive']].mean()
print(avg_results)

# Save results
output_path = 'assets/results/neural_sampler_mase_results.csv'
results_df.to_csv(output_path, index=False)
print(f"\nResults saved to: {output_path}")
