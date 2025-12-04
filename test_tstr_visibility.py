"""
TSTR (Train on Synthetic, Test on Real) Experiment Script

Purpose:
Evaluate the quality of synthetic data by training models PURELY on synthetic data
and testing them on real held-out data.

Variants:
1. Real Baseline (Train Real -> Test Real)
2. Uniform TSTR (Train Synth Uniform -> Test Real)
3. Vis Raw TSTR (Train Synth Vis Raw -> Test Real)
4. Vis Decomp TSTR (Train Synth Vis Decomp -> Test Real)
"""

import sys
import os
import pandas as pd
import numpy as np
from functools import partial

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from neuralforecast import NeuralForecast
from statsforecast.models import SeasonalNaive
from statsforecast import StatsForecast
from utilsforecast.losses import mase, smape
from utilsforecast.evaluation import evaluate

from utils.load_data.config import DATASETS
from utils.config import MODEL_CONFIG, MODELS
from utils.load_data.base import LoadDataset
from src.qgraph_ts import Grasynda
from src.grasynda_visibility import GrasyndaVisibilityGraph

# Configuration
MODEL = 'NHITS'
EPOCHS = 10  # Keep low for speed as requested
DATA_GROUPS = [
    ('Gluonts', 'm1_monthly'),
    ('Gluonts', 'm1_quarterly'),
    ('M3', 'Monthly'),
    ('M3', 'Quarterly'),
    ('Tourism', 'Monthly'),
    ('Tourism', 'Quarterly'),
]

def run_tstr_experiments():
    print("=" * 80)
    print("RUNNING TSTR (TRAIN SYNTHETIC, TEST REAL) EXPERIMENTS")
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
        
        # 2. Split Data (Standard Train/Test Split)
        # We only forecast the 'horizon' (test set), not the whole series
        train_real, test_real = LoadDataset.train_test_split(df, horizon)
        
        print(f"Train shape: {train_real.shape}")
        print(f"Test shape: {test_real.shape}")
        print(f"Horizon: {horizon}")

        # 3. Generate Synthetic Training Sets
        # IMPORTANT: TSTR means we replace the real training data with synthetic data
        training_sets = {}

        # A. Real Baseline (Control)
        training_sets['Real_Baseline'] = train_real.copy()

        # B. Grasynda Uniform (TSTR)
        print("\nGenerating Synthetic: Grasynda Uniform...")
        gen_uniform = Grasynda(
            n_quantiles=25,
            quantile_on='remainder',
            period=freq_int,
            ensemble_transitions=False
        )
        # Transform returns ONLY the synthetic data (usually)
        # But Grasynda.transform implementation in this codebase might return just the synthetic part
        # Let's verify: yes, it returns generated_time_series dict or df
        # We need to make sure we format it correctly for NeuralForecast
        synth_uniform = gen_uniform.transform(train_real)
        
        # CRITICAL: For TSTR, we DO NOT concat with train_real. We use ONLY synth_uniform.
        # We must ensure synth_uniform has the same schema as train_real
        # And we need to make sure the unique_ids match or are handled correctly
        # NeuralForecast needs 'unique_id', 'ds', 'y'
        
        # The transform method usually returns a DataFrame with 'unique_id' modified (e.g. 'Grasynda_uid')
        # For TSTR to work on the SAME test set, we usually need the model to learn patterns 
        # that generalize to the real test set IDs. 
        # However, NeuralForecast models like NHITS are global models. They learn from ALL series.
        # So having different IDs in train (synthetic) vs test (real) is FINE for global models,
        # AS LONG AS we provide the 'real' history during inference (predict step).
        
        training_sets['Uniform_TSTR'] = synth_uniform

        # C. Vis Degree Raw (TSTR)
        print("Generating Synthetic: Vis Degree (Raw)...")
        gen_vis_raw = GrasyndaVisibilityGraph(
            period=freq_int,
            visibility_type='horizontal',
            generation_method='degree_matching',
            use_decomposition=False
        )
        synth_vis_raw = gen_vis_raw.transform(train_real)
        training_sets['VisRaw_TSTR'] = synth_vis_raw

        # D. Vis Degree Decomp (TSTR)
        print("Generating Synthetic: Vis Degree (Decomposed)...")
        gen_vis_decomp = GrasyndaVisibilityGraph(
            period=freq_int,
            visibility_type='horizontal',
            generation_method='degree_matching',
            use_decomposition=True
        )
        synth_vis_decomp = gen_vis_decomp.transform(train_real)
        training_sets['VisDecomp_TSTR'] = synth_vis_decomp

        # 4. Train & Evaluate
        print("\n--- Training & Evaluating ---")
        
        input_data = {'input_size': n_lags, 'h': horizon}
        
        # We evaluate all models on the SAME real test set
        # But we need to be careful: nf.predict() usually uses the training set passed to fit() 
        # to construct the lags for the first prediction.
        # If we trained on synthetic data (which might have different IDs or values), 
        # we MUST pass the REAL training history to predict() so it has the correct context 
        # for forecasting the real test set.
        
        dataset_results = {}
        
        for name, train_data in training_sets.items():
            print(f"Training: {name} (Size: {len(train_data)})")
            
            # Configure model
            model_params = MODEL_CONFIG.get(MODEL)

          
            model_conf = {**input_data, **model_params}
            
            nf = NeuralForecast(
                models=[MODELS[MODEL](**model_conf, alias=name)],
                freq=freq_str
            )
            
            # Train on the specific training set (Real or Synthetic)
            nf.fit(df=train_data, val_size=horizon)
            
            # Predict on REAL test set
            # We pass the REAL training history (train_real) to predict()
            # This ensures the model uses real recent values to forecast the future
            # even if it was trained on synthetic data.
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
            
            # Extract MASE score for the specific model alias
            # evaluate returns a df with columns: unique_id, metric, <model_alias>
            score = eval_df.query('metric=="mase"')[name].mean()
            dataset_results[name] = score
            print(f"  -> MASE: {score:.4f}")

        # Store results
        result_row = {
            'dataset': data_name,
            'group': group,
            **dataset_results
        }
        all_results.append(result_row)

    # 5. Summary
    print("\n" + "=" * 80)
    print("TSTR EXPERIMENT RESULTS (MASE)")
    print("=" * 80)
    
    results_df = pd.DataFrame(all_results)
    print(results_df.to_string(index=False))
    
    print("\nAverage MASE:")
    print(results_df.mean(numeric_only=True))
    
    # Save
    os.makedirs('assets/results', exist_ok=True)
    results_df.to_csv('assets/results/tstr_mase_results.csv', index=False)
    print("\nSaved to assets/results/tstr_mase_results.csv")

if __name__ == "__main__":
    run_tstr_experiments()
