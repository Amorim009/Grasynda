"""
Baseline Experiments - No Data Augmentation

This script runs forecasting experiments WITHOUT any data augmentation
to establish baseline performance for comparison with augmentation methods.

Results: assets/results/baseline_experiment_results.csv
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
from datetime import datetime
from functools import partial

from neuralforecast import NeuralForecast
from utilsforecast.losses import mase
from utilsforecast.evaluation import evaluate

from utils.load_data.config import DATASETS
from utils.load_data.base import LoadDataset
from utils.config import MODEL_CONFIG, MODELS

# Configuration - MUST MATCH run_universal_experiments.py
DATASETS_TO_TEST = [
    ('Gluonts', 'm1_monthly'),
    ('Gluonts', 'm1_quarterly'),
    ('M3', 'Monthly'),
    ('M3', 'Quarterly'),
    ('Tourism', 'Monthly'),
    ('Tourism', 'Quarterly'),
]

FORECASTING_MODELS = ['NHITS', 'MLP', 'KAN']

# Parameters - MUST MATCH universal experiments
EPOCHS = 10
MAX_STEPS = 100 * EPOCHS  # Same as universal

def run_baseline_experiment(train, test, horizon, freq_str, freq_int, n_lags, model_name):
    """
    Run baseline experiment following EXACT methodology from run_universal_experiments.py
    
    Args:
        train: Training DataFrame
        test: Test DataFrame  
        horizon: Forecast horizon
        freq_str: Frequency string
        freq_int: Frequency integer (for MASE seasonality)
        n_lags: Number of lags (input_size)
        model_name: 'NHITS', 'MLP', or 'KAN'
    
    Returns:
        MASE score, train_size, test_size
    """
    # Get model configuration - EXACTLY as in universal experiments
    model_params = MODEL_CONFIG.get(model_name).copy()
    model_params['max_steps'] = MAX_STEPS  # CRITICAL: Must match universal
    
    model_conf = {
        'input_size': n_lags,
        'h': horizon,
        **model_params
    }
    
    # Create model - EXACTLY as in universal experiments  
    model = MODELS[model_name](**model_conf, alias=f"Baseline_{model_name}")
    
    # Train - EXACTLY as in universal experiments
    nf = NeuralForecast(
        models=[model],
        freq=freq_str
    )
    
    nf.fit(df=train, val_size=horizon)  # CRITICAL: Must include val_size
    
    # Predict - EXACTLY as in universal experiments
    fcst = nf.predict()
    
    # Evaluate - EXACTLY as in universal experiments
    test_with_fcst = test.merge(
        fcst.reset_index(), on=['unique_id', 'ds'], how="left"
    )
    
    eval_df = evaluate(
        test_with_fcst,
        [partial(mase, seasonality=freq_int)],  # CRITICAL: Must include seasonality
        train_df=train
    )
    
    mase_score = eval_df.query('metric=="mase"')[f"Baseline_{model_name}"].mean()
    
    return mase_score, len(train), len(test)


# Main execution
if __name__ == "__main__":
    print("=" * 100)
    print("BASELINE EXPERIMENTS - NO DATA AUGMENTATION")
    print("=" * 100)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nDatasets: {len(DATASETS_TO_TEST)}")
    print(f"Models: {FORECASTING_MODELS}")
    print(f"Training Modes: {TRAINING_MODES}")
    print(f"Total experiments: {len(DATASETS_TO_TEST) * len(FORECASTING_MODELS) * len(TRAINING_MODES)}")
    print("=" * 100)
    
    results = []
    experiment_num = 0
    total_experiments = len(DATASETS_TO_TEST) * len(FORECASTING_MODELS) * len(TRAINING_MODES)
    
    for dataset_name, group in DATASETS_TO_TEST:
        print(f"\n{'='*100}")
        print(f"Dataset: {dataset_name} - {group}")
        print(f"{'='*100}")
        
        # Load dataset - EXACTLY as in universal experiments
        data_loader = DATASETS[dataset_name]
        min_samples = data_loader.min_samples[group]
        
        df, horizon, n_lags, freq_str, freq_int = data_loader.load_everything(
            group, min_n_instances=min_samples
        )
        
        print(f"Loaded: {len(df['unique_id'].unique())} series, {len(df)} rows")
        print(f"Horizon: {horizon}, Lags: {n_lags}, Frequency: {freq_str}")
        
        # Split data - EXACTLY as in universal experiments
        train, test = LoadDataset.train_test_split(df, horizon)
        print(f"Train: {len(train)} rows, Test: {len(test)} rows")
        
        for model_name in FORECASTING_MODELS:
            experiment_num += 1
            print(f"\n[{experiment_num}/{total_experiments}] {dataset_name}-{group} | {model_name}")
            
            try:
                mase_score, train_size, test_size = run_baseline_experiment(
                    train=train,
                    test=test,
                    horizon=horizon,
                    freq_str=freq_str,
                    freq_int=freq_int,
                    n_lags=n_lags,
                    model_name=model_name
                )
                
                result = {
                    'Dataset': dataset_name,
                    'Group': group,
                    'Augmentation_Method': 'Baseline (No Augmentation)',
                    'Forecasting_Model': model_name,
                    'Training_Mode': 'Baseline',
                    'MASE': mase_score,
                    'Train_Size': train_size,
                    'Test_Size': test_size,
                    'Status': 'Success'
                }
                
                results.append(result)
                print(f"✓ MASE: {mase_score:.6f} | Train: {train_size} | Test: {test_size}")
                
                # Save incrementally
                results_df = pd.DataFrame(results)
                results_df.to_csv('assets/results/baseline_experiment_results.csv', index=False)
                
            except Exception as e:
                print(f"✗ Error: {str(e)}")
                import traceback
                traceback.print_exc()
                result = {
                    'Dataset': dataset_name,
                    'Group': group,
                    'Augmentation_Method': 'Baseline (No Augmentation)',
                    'Forecasting_Model': model_name,
                    'Training_Mode': 'Baseline',
                    'MASE': np.nan,
                    'Train_Size': np.nan,
                    'Test_Size': np.nan,
                    'Status': f'Error: {str(e)}'
                }
                results.append(result)
    
    # Final save
    results_df = pd.DataFrame(results)
    results_df.to_csv('assets/results/baseline_experiment_results.csv', index=False)
    
    print("\n" + "=" * 100)
    print("BASELINE EXPERIMENTS COMPLETE")
    print("=" * 100)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Results saved to: assets/results/baseline_experiment_results.csv")
    print(f"Total experiments: {len(results)}")
    print(f"Successful: {sum(1 for r in results if r['Status'] == 'Success')}")
    print(f"Failed: {sum(1 for r in results if r['Status'] != 'Success')}")
    print("=" * 100)
