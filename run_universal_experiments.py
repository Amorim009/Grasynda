"""
Universal.

Results: assets/results/universal_experiment_results_{MODEL}.csv
"""


import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
from functools import partial
from datetime import datetime

from neuralforecast import NeuralForecast
from utilsforecast.losses import mase
from utilsforecast.evaluation import evaluate

from utils.load_data.config import DATASETS
from utils.config import MODEL_CONFIG, MODELS, SYNTH_METHODS
from utils.load_data.base import LoadDataset
from src.qgraph_ts import Grasynda
from src.grasynda_visibility import GrasyndaVisibilityGraph
from src.workflow import ExpWorkflow

# Configuration

# Datasets

DATASETS_TO_TEST = [
    # Already completed (commented out to run only missing datasets):
    # ('Gluonts', 'm1_monthly'),
    # ('Gluonts', 'm1_quarterly'),
    # Missing datasets to run:
    ('M3', 'Monthly'),
    ('M3', 'Quarterly'),
    ('Tourism', 'Monthly'),
    ('Tourism', 'Quarterly'),
]

# Grasynda

GRASYNDA_METHODS = [
    'Grasynda_Uniform',
    'Grasynda_Vis_Horizontal',
    'Grasynda_Vis_Natural',
]

# Others

OTHER_METHODS = [
    'SeasonalMBB',
    'Jittering',
    'Scaling',
    'TimeWarping',
    'MagnitudeWarping',
    'TSMixup',
    'DBA',
]

# Models

FORECASTING_MODELS = [
    'NHITS',
    'MLP',
    'KAN',
]

# Modes

TRAINING_MODES = [
    'Train+Real',  # Augmented
    'TSTR',        # Synthetic
]

# Parameters

EPOCHS = 10
MAX_STEPS = 100 * EPOCHS

# ============ HELPER FUNCTIONS ============

def generate_grasynda_data(method_name, train_df, freq_int):
    """Grasynda"""
    
    if method_name == 'Grasynda_Uniform':
        generator = Grasynda(
            n_quantiles=25,
            quantile_on='remainder',
            period=freq_int,
            ensemble_transitions=False
        )
        synth = generator.transform(train_df)
        return synth
    
    elif method_name == 'Grasynda_Vis_Horizontal':
        generator = GrasyndaVisibilityGraph(
            period=freq_int,
            visibility_type='horizontal',
            quantile_on='remainder',
            use_decomposition=True,
            robust=False
        )
        synth = generator.transform(train_df)
        return synth
    
    elif method_name == 'Grasynda_Vis_Natural':
        generator = GrasyndaVisibilityGraph(
            period=freq_int,
            visibility_type='natural',
            quantile_on='remainder',
            use_decomposition=True,
            robust=False
        )
        synth = generator.transform(train_df)
        return synth
    
    else:
        raise ValueError(f"Unknown Grasynda method: {method_name}")

def generate_other_augmentation_data(method_name, train_df, augmentation_params, n_series=1):
    """Others"""
    
    train_aug = ExpWorkflow.get_offline_augmented_data(
        train_=train_df,
        generator_name=method_name,
        augmentation_params=augmentation_params,
        n_series_by_uid=n_series
    )
    
    return train_aug

def extract_synthetic_only(train_aug, real_train):
    """Extract"""
    real_ids = real_train['unique_id'].unique()
    synth_only = train_aug[~train_aug['unique_id'].isin(real_ids)].copy()
    
    if len(synth_only) == 0:
        print("  Warning: No synthetic IDs found, using full augmented set")
        return train_aug
    
    return synth_only

# ============ MAIN EXPERIMENT LOOP ============

def run_universal_experiments():
    print("=" * 100)
    print("UNIVERSAL TIME SERIES FORECASTING EXPERIMENT SUITE")
    print("=" * 100)
    print(f"\nConfiguration:")
    print(f"  Datasets: {len(DATASETS_TO_TEST)}")
    print(f"  Grasynda Methods: {len(GRASYNDA_METHODS)}")
    print(f"  Other Methods: {len(OTHER_METHODS)}")
    print(f"  Forecasting Models: {len(FORECASTING_MODELS)}")
    print(f"  Training Modes: {len(TRAINING_MODES)}")
    print(f"\n  Total Experiments: {len(DATASETS_TO_TEST) * (len(GRASYNDA_METHODS) + len(OTHER_METHODS)) * len(FORECASTING_MODELS) * len(TRAINING_MODES)}")
    print("=" * 100)
    
    all_results = []
    experiment_count = 0
    start_time = datetime.now()
    
    for dataset_idx, (data_name, group) in enumerate(DATASETS_TO_TEST):
        print(f"\n{'=' * 100}")
        print(f"DATASET {dataset_idx+1}/{len(DATASETS_TO_TEST)}: {data_name} - {group}")
        print(f"{'=' * 100}")
        
        # Load data
        data_loader = DATASETS[data_name]
        min_samples = data_loader.min_samples[group]
        
        df, horizon, n_lags, freq_str, freq_int = data_loader.load_everything(
            group, min_n_instances=min_samples
        )
        
        print(f"  Data: {df.shape}, Unique IDs: {df['unique_id'].nunique()}")
        print(f"  Horizon: {horizon}, Lags: {n_lags}, Frequency: {freq_str}")
        
        # Split data
        train, test = LoadDataset.train_test_split(df, horizon)
        
        # Prepare augmentation parameters for other methods
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
        
        # Test all augmentation methods
        all_methods = GRASYNDA_METHODS + OTHER_METHODS
        
        for method_idx, method_name in enumerate(all_methods):
            print(f"\n  [{method_idx+1}/{len(all_methods)}] Method: {method_name}")
            
            try:
                # Generate synthetic data
                if method_name in GRASYNDA_METHODS:
                    print(f"    Grasynda...")
                    synth = generate_grasynda_data(method_name, train, freq_int)
                else:
                    print(f"    Augmenting...")
                    synth = generate_other_augmentation_data(
                        method_name, train, augmentation_params
                    )
                
                # Prepare training sets for both modes
                if method_name in GRASYNDA_METHODS:
                    # Grasynda
                    training_sets = {
                        'Train+Real': pd.concat([train, synth]).reset_index(drop=True),
                        'TSTR': synth
                    }
                else:
                    # Others
                    training_sets = {
                        'Train+Real': synth,  # Combined
                        'TSTR': extract_synthetic_only(synth, train)
                    }
                
                # Test all forecasting models
                for model_name in FORECASTING_MODELS:
                    # Test all training modes
                    for mode in TRAINING_MODES:
                        experiment_count += 1
                        train_data = training_sets[mode]
                        
                        print(f"      [{experiment_count}] {model_name} - {mode} (n={len(train_data)})")
                        
                        try:
                            # Configure model
                            model_params = MODEL_CONFIG.get(model_name)
                            model_params['max_steps'] = MAX_STEPS
                            
                            model_conf = {
                                'input_size': n_lags,
                                'h': horizon,
                                **model_params
                            }
                            
                            # Train
                            nf = NeuralForecast(
                                models=[MODELS[model_name](**model_conf, alias=f"{method_name}_{mode}")],
                                freq=freq_str
                            )
                            
                            nf.fit(df=train_data, val_size=horizon)
                            
                            # Predict
                            if mode == 'TSTR':
                                fcst = nf.predict(df=train)
                            else:
                                fcst = nf.predict()
                            
                            # Evaluate
                            test_with_fcst = test.merge(
                                fcst.reset_index(), on=['unique_id', 'ds'], how="left"
                            )
                            
                            eval_df = evaluate(
                                test_with_fcst,
                                [partial(mase, seasonality=freq_int)],
                                train_df=train
                            )
                            
                            mase_score = eval_df.query('metric=="mase"')[f"{method_name}_{mode}"].mean()
                            
                            # Store result
                            all_results.append({
                                'Dataset': data_name,
                                'Group': group,
                                'Augmentation_Method': method_name,
                                'Forecasting_Model': model_name,
                                'Training_Mode': mode,
                                'MASE': mase_score,
                                'Train_Size': len(train_data),
                                'Test_Size': len(test),
                                'Status': 'Success'
                            })
                            
                            print(f"        → MASE: {mase_score:.4f}")
                            
                        except Exception as e:
                            print(f"        → FAILED: {str(e)[:100]}")
                            all_results.append({
                                'Dataset': data_name,
                                'Group': group,
                                'Augmentation_Method': method_name,
                                'Forecasting_Model': model_name,
                                'Training_Mode': mode,
                                'MASE': np.nan,
                                'Train_Size': len(train_data) if 'train_data' in locals() else 0,
                                'Test_Size': len(test),
                                'Status': f'Error: {str(e)[:50]}'
                            })
                
            except Exception as e:
                print(f"    Failed: {str(e)[:100]}")
                # Record
                for model_name in FORECASTING_MODELS:
                    for mode in TRAINING_MODES:
                        all_results.append({
                            'Dataset': data_name,
                            'Group': group,
                            'Augmentation_Method': method_name,
                            'Forecasting_Model': model_name,
                            'Training_Mode': mode,
                            'MASE': np.nan,
                            'Train_Size': 0,
                            'Test_Size': len(test),
                            'Status': f'Generation Error: {str(e)[:50]}'
                        })
        
        # Incremental
        results_df = pd.DataFrame(all_results)
        results_df.to_csv('assets/results/universal_experiment_results_missing.csv', index=False)
        print(f"\n  Incremental save: {len(results_df)} experiments")
    
    # Pivots
    results_df = pd.DataFrame(all_results)
    
    print("\n" + "=" * 100)
    print("CREATING PIVOT TABLES (Methods as Columns, Datasets as Rows)")
    print("=" * 100)
    
    for model_name in FORECASTING_MODELS:
        for mode in TRAINING_MODES:
            # Filter
            filtered = results_df[
                (results_df['Forecasting_Model'] == model_name) &
                (results_df['Training_Mode'] == mode)
            ]
            
            if len(filtered) == 0:
                continue
            
            # Dataset
            filtered['Dataset_Full'] = filtered['Dataset'] + ' - ' + filtered['Group']
            
            # Pivot
            pivot = filtered.pivot_table(
                index='Dataset_Full',
                columns='Augmentation_Method',
                values='MASE',
                aggfunc='first'  # Should be only one value per combination
            )
            
            # Reorder
            grasynda_cols = [c for c in pivot.columns if c in GRASYNDA_METHODS]
            other_cols = [c for c in pivot.columns if c in OTHER_METHODS]
            ordered_cols = grasynda_cols + other_cols
            pivot = pivot[ordered_cols]
            
            # Save
            filename = f'assets/results/{model_name}_{mode}_Results.csv'
            pivot.to_csv(filename)
            print(f"  Saved: {filename}")
    
    # Summary
    elapsed = datetime.now() - start_time
    print("\n" + "=" * 100)
    print("COMPLETE")
    print("=" * 100)
    print(f"Experiments: {len(all_results)}")
    print(f"Success: {sum(1 for r in all_results if r['Status'] == 'Success')}")
    print(f"Failed: {sum(1 for r in all_results if r['Status'] != 'Success')}")
    print(f"Time: {elapsed}")
    print(f"\nFiles created:")
    print(f"  - universal_experiment_results_raw.csv (all data)")
    for model_name in FORECASTING_MODELS:
        for mode in TRAINING_MODES:
            print(f"  - {model_name}_{mode}_Results.csv (pivot table)")
    print("=" * 100)
    
    return results_df

if __name__ == "__main__":
    os.makedirs('assets/results', exist_ok=True)
    results = run_universal_experiments()
