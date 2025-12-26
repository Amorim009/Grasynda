
import os
import numpy as np
import pandas as pd
from utils.load_data.config import DATASETS
from src.grasynda_unified import GrasyndaUnified

def audit_alignment(n_q=25):
    print(f"Auditing Reconstruction Alignment (NQ={n_q})...")
    
    # Use a specific dataset and series
    dataset_name = 'M3'
    group = 'Monthly'
    uid = 'M1165'
    
    data_loader = DATASETS[dataset_name]
    df, horizon, n_lags, freq_str, freq_int = data_loader.load_everything(group)
    uid_df = df[df['unique_id'] == uid].copy()
    
    # Initialize Hybrid Grasynda (Trend + Remainder)
    model = GrasyndaUnified(
        period=freq_int,
        n_quantiles=n_q,
        components_to_model=['trend', 'remainder'],
        component_params={'trend': {'apply_differentiation': True}},
        sampling_type='discrete'
    )
    
    # 1. Generate with return_components=True
    synth_df, synth_components = model.transform(uid_df, return_components=True)
    
    # Extract components
    # Notice: GrasyndaUnified adds 'GrasyndaUnified_' prefix to unique_id in synth_df
    # In synth_components, the keys are components, and values are dicts {uid: Series}
    gen_trend = synth_components['trend'][uid].values
    gen_rem = synth_components['remainder'][uid].values
    
    # We also need the original seasonal component (unmodified)
    decomposed_orig = model.decompose_tsd(uid_df, freq_int, False)
    orig_seasonal = decomposed_orig['seasonal'].values
    
    # Calculate reconstructed Y manually
    manual_y = gen_trend + orig_seasonal + gen_rem
    
    # Get y from the model result
    processed_uid = f"GrasyndaUnified_{uid}"
    model_y = synth_df[synth_df['unique_id'] == processed_uid]['y'].values
    
    print(f"\nAudit results for {uid}:")
    print(f"Manual Reconstruction Min: {manual_y.min():.4f}")
    print(f"Model Result Y Min:        {model_y.min():.4f}")
    
    discrepancy = np.abs(manual_y - model_y).max()
    print(f"Max Discrepancy:           {discrepancy:.4e}")
    
    # Check individual components mins
    print(f"\nGenerated Component Mins:")
    print(f"Trend:                     {gen_trend.min():.4f}")
    print(f"Remainder:                 {gen_rem.min():.4f}")
    print(f"Orig Seasonal:             {orig_seasonal.min():.4f}")
    
    # Audit _integrate_component logic
    print(f"\nChecking Integration Logic for Trend...")
    trend_prep = model._differentiate_component(decomposed_orig, 'trend')
    orig_diffs = trend_prep['diff_trend'].values
    # Just a mock check
    print(f"Original Trend points:     {len(decomposed_orig)}")
    print(f"Synthetic Trend points:    {len(gen_trend)}")
    
    if len(gen_trend) != len(decomposed_orig):
        print("WARNING: Length mismatch in trend!")
    
    return manual_y, model_y

if __name__ == "__main__":
    audit_alignment()
