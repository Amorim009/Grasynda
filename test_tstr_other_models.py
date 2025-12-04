"""
TSTR (Train on Synthetic, Test on Real) Experiment Script for OTHER Models

Purpose:
Evaluate the quality of synthetic data from other augmentation methods by 
training models PURELY on synthetic data and testing them on real held-out data.

Methods Compared:
- SeasonalMBB
- Jittering
- Scaling
- TimeWarping
- MagnitudeWarping
- TSMixup
- DBA
"""

import sys
import os
import pandas as pd
import numpy as np
from functools import partial

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from neuralforecast import NeuralForecast
from utilsforecast.losses import mase
from utilsforecast.evaluation import evaluate

from utils.load_data.config import DATASETS
from utils.config import MODEL_CONFIG, MODELS, SYNTH_METHODS
from utils.load_data.base import LoadDataset
from src.workflow import ExpWorkflow

# Configuration
MODEL = 'NHITS'
EPOCHS = 10  # Keep low for speed
DATA_GROUPS = [
    ('Gluonts', 'm1_monthly'),
    ('Gluonts', 'm1_quarterly'),
    ('M3', 'Monthly'),
    ('M3', 'Quarterly'),
    ('Tourism', 'Monthly'),
    ('Tourism', 'Quarterly'),
]

def run_tstr_other_models():
    print("=" * 80)
    print("RUNNING TSTR FOR OTHER AUGMENTATION MODELS")
    print(f"Model: {MODEL} | Epochs: {EPOCHS}")
    print("=" * 80)

    all_results = []

    for idx, (data_name, group) in enumerate(DATA_GROUPS):
        print(f"\n{'=' * 80}")
        print(f"DATASET {idx+1}/6: {data_name} - {group}")
        print(f"{'=' * 80}")

        # 1. Load Data
        data_loader = DATASETS[data_name]
        min_samples = data_loader.min_samples[group]
        
        df, horizon, n_lags, freq_str, freq_int = data_loader.load_everything(
            group, min_n_instances=min_samples
        )
        
        # 2. Split Data
        train_real, test_real = LoadDataset.train_test_split(df, horizon)
        
        print(f"Train shape: {train_real.shape}")
        print(f"Test shape: {test_real.shape}")
        
        # 3. Generate Synthetic Training Sets
        training_sets = {}
        
        # Augmentation parameters
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
        
        print("\n--- Generating Augmented Data ---")
        for method_name in SYNTH_METHODS:
            print(f"Generating data for: {method_name}")
            try:
                # Generate synthetic data
                # Note: get_offline_augmented_data typically returns concatenated real+synthetic
                # We need to extract ONLY the synthetic part for TSTR
                # The synthetic IDs usually have a suffix or prefix
                
                train_aug_full = ExpWorkflow.get_offline_augmented_data(
                    train_=train_real,
                    generator_name=method_name,
                    augmentation_params=augmentation_params,
                    n_series_by_uid=1
                )
                
                # Filter to keep ONLY synthetic data
                # Synthetic data usually has different IDs
                real_ids = train_real['unique_id'].unique()
                train_synth = train_aug_full[~train_aug_full['unique_id'].isin(real_ids)].copy()
                
                if len(train_synth) == 0:
                    print(f"Warning: No synthetic data found for {method_name}. Using full augmented set.")
                    train_synth = train_aug_full
                else:
                    print(f"  -> Extracted {len(train_synth)} synthetic rows")
                
                training_sets[method_name] = train_synth
                
            except Exception as e:
                print(f"Failed to generate data for {method_name}: {e}")

        # 4. Train & Evaluate
        print("\n--- Training & Evaluating (TSTR) ---")
        
        input_data = {'input_size': n_lags, 'h': horizon}
        dataset_results = {}
        
        for name, train_data in training_sets.items():
            print(f"Training TSTR: {name} (Size: {len(train_data)})")
            
            try:
                # Configure model
                model_params = MODEL_CONFIG.get(MODEL)
                # Override epochs for speed
                model_params['max_steps'] = 100 * EPOCHS
                
                model_conf = {**input_data, **model_params}
                
                nf = NeuralForecast(
                    models=[MODELS[MODEL](**model_conf, alias=name)],
                    freq=freq_str
                )
                
                # Train on SYNTHETIC data
                nf.fit(df=train_data, val_size=horizon)
                
                # Predict on REAL test set
                # Pass real training history for context
                fcst = nf.predict(df=train_real)
                
                # Merge with ground truth
                test_with_fcst = test_real.merge(
                    fcst.reset_index(), on=['unique_id', 'ds'], how="left"
                )
                
                # Evaluate
                eval_df = evaluate(
                    test_with_fcst,
                    [partial(mase, seasonality=freq_int)],
                    train_df=train_real
                )
                
                score = eval_df.query('metric=="mase"')[name].mean()
                dataset_results[name] = score
                print(f"  -> MASE: {score:.4f}")
                
            except Exception as e:
                print(f"  -> Failed: {e}")
                dataset_results[name] = np.nan

        # Store results
        result_row = {
            'dataset': data_name,
            'group': group,
            **dataset_results
        }
        all_results.append(result_row)

    # 5. Summary
    print("\n" + "=" * 80)
    print("TSTR RESULTS FOR OTHER MODELS")
    print("=" * 80)
    
    results_df = pd.DataFrame(all_results)
    print(results_df.to_string(index=False))
    
    print("\nAverage MASE:")
    print(results_df.mean(numeric_only=True))
    
    # Save
    output_path = 'assets/results/tstr_other_models_results.csv'
    os.makedirs('assets/results', exist_ok=True)
    results_df.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")

if __name__ == "__main__":
    run_tstr_other_models()
