
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
from utils.load_data.config import DATASETS
from src.grasynda_unified import GrasyndaUnified

def component_balance_generate(uid, df_series, model, balance_strategy='boost_remainder'):
    """
    Implements the component rescaling strategy.
    
    Logic:
    1. Decompose (STL)
    2. Rescale components based on strategy
    3. Generate synthetic version of the *modified* components
    4. Reconstruct
    """
    # 1. Decompose
    series = df_series['y'].values
    period = model.period
    if len(series) < 2 * period:
        return df_series.copy() # fallback
        
    res = STL(series, period=period, robust=False).fit()
    trend = res.trend
    seasonal = res.seasonal
    remainder = res.resid
    
    std_s = np.std(seasonal)
    std_r = np.std(remainder)
    
    if std_s == 0 or std_r == 0:
        return model.transform(df_series)
        
    # 2. Rescaling Logic
    # User Goal: "scaled to the same magnitude"
    # Let's define target magnitude as the mean of the two stds
    target_std = (std_s + std_r) / 2
    
    # Scale factors
    scale_s = target_std / std_s
    scale_r = target_std / std_r
    
    print(f"[{uid}] Std(S)={std_s:.2f}, Std(R)={std_r:.2f} -> Target={target_std:.2f}")
    print(f"[{uid}] Scaling S by {scale_s:.2f}, R by {scale_r:.2f}")
    
    # B. Generate Remainder (On BOOSTED remainder)
    remainder_boosted = remainder * scale_r
    print(f"[{uid}] Boosted Remainder Range: {remainder_boosted.min():.2f} to {remainder_boosted.max():.2f}")
    if np.any(np.isinf(remainder_boosted)) or np.any(np.isnan(remainder_boosted)):
        print("WARNING: Boosted Remainder contains Inf/NaN")
    
    # We create a dummy DF for the remainder generation
    df_rem = df_series.copy()
    df_rem['remainder'] = remainder_boosted
    # Trick: Grasynda needs 'y' usually, but if we call internal methods we can pass specific cols
    # Or we can create an instance that processes 'remainder' without decomposition if we hack it.
    # Actually, GrasyndaUnified.transform checks "skip_decomposition = 'y' in components_to_model".
    # It ALWAYS decomposes if 'remainder' is target and 'y' is not.
    # We need to Subclass or call internal method    # Let's use the internal method for precision
    # Calculate Quantiles first!
    df_rem['Quantile'] = model._get_quantiles(df_rem, 'remainder', component='remainder')
    model._calc_transition_matrix(df_rem, component='remainder')
    
    st_rem = model._get_param('remainder', 'sampling_type')
    print(f"[{uid}] Resolved Remainder Sampling Type: {st_rem}")
    
    gen_rem_dict = model._create_synthetic_ts_quantile(
        df_rem, 'remainder', 'remainder', 
        sampling_type='discrete' # FORCE DISCRETE to avoid overflow risk/debug param issue
    )
    syn_rem_boosted = gen_rem_dict[uid].values
    
    # C. Generate Trend (Standard)
    # We can rely on Model for this, or just use original trend if only remainder was target
    # Let's assume Hybrid: we want Synthetic Trend too
    # We'll use the proper model flow for Trend
    # But since we can't easily inject just one component into .transform(), let's generate Trend separately
    # using a Trend-only model config on the original data (since trend wasn't scaled)
    df_trend = df_series.copy()
    df_trend['y'] = trend # Treat trend as Y? No
    df_trend['trend'] = trend # Prepare column
    # Use internal method for Trend
    # Note: Trend usually requires differentiation
    trend_diff = pd.Series(trend).diff().fillna(0)
    df_trend['diff_trend'] = trend_diff
    
    # Calculate Quantiles for Trend
    # Need to config model to know trend params if checking 'trend'
    df_trend['Quantile'] = model._get_quantiles(df_trend, 'diff_trend', component='trend')
    model._calc_transition_matrix(df_trend, component='trend')

    gen_trend_dict = model._create_synthetic_ts_quantile(
        df_trend, 'trend', 'diff_trend',
        sampling_type='discrete' # FORCE DISCRETE for demo stability
    ) 
    # Integrate
    syn_trend_diff = gen_trend_dict[uid]
    syn_trend = model._integrate_component(df_trend, 'trend', {uid: syn_trend_diff})[uid].values

    # 4. Reconstruction
    # "add the scaled down seasonality to the synthetic remainder (generated on the boosted remainder) and the trend"
    # Scaled down seasonality = S * scale_s (which might be < 1 if S was dominant)
    
    seasonality_scaled = seasonal * scale_s
    
    # If we sum them directly:
    # y_new = Syn_Trend + Scaled_Seas + Syn_Rem_Boosted
    # Note: Syn_Rem_Boosted has magnitude of Target_Std.
    # Scaled_Seas has magnitude of Target_Std.
    # So they are balanced.
    
    y_final = syn_trend + seasonality_scaled + syn_rem_boosted
    
    # Pack into DataFrame
    df_final = df_series.copy()
    df_final['unique_id'] = f"{uid}_Balanced"
    df_final['y'] = y_final
    return df_final

def demonstrate_scaling(dataset_name='M3', group='Monthly', uid='M806'):
    print(f"Demonstrating Component Scaling on {uid}...")
    
    data_loader = DATASETS[dataset_name]
    df, _, _, _, freq_int = data_loader.load_everything(group)
    df_series = df[df['unique_id'] == uid].copy()
    
    # Config (Hybrid Fixed)
    config = {
        'period': freq_int,
        'n_quantiles': 25,
        'components_to_model': ['trend', 'remainder'],
        'component_params': {
            'trend': {'sampling_type': 'continuous_uniform', 'apply_differentiation': True},
            'remainder': {'sampling_type': 'discrete'}
        }
    }
    model = GrasyndaUnified(**config)
    
    # 1. Baseline
    synth_base = model.transform(df_series)
    
    # 2. Balanced Generation
    df_balanced = component_balance_generate(uid, df_series, model)
    
    # 3. Plotting
    output_dir = 'assets/scaling_demo'
    os.makedirs(output_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Original
    ax.plot(df_series['ds'], df_series['y'], color='black', label='Original', lw=2, alpha=0.8)
    
    # Baseline
    base_ts = synth_base[synth_base['unique_id'] == f'GrasyndaUnified_{uid}']
    ax.plot(base_ts['ds'], base_ts['y'], color='gray', linestyle='--', label='Baseline Hybrid', alpha=0.5)
    
    # Balanced
    ax.plot(df_balanced['ds'], df_balanced['y'], color='#27ae60', label='Balanced Scaling (Boosted Rem, Reduced Seas)', lw=1.5)
    
    ax.set_title(f"Component Scaling Strategy ({uid})\nSeasonality & Remainder Scaled to Average Magnitude")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = f"{output_dir}/{uid}_balanced.png"
    plt.savefig(save_path, dpi=120)
    print(f"Saved to {save_path}")

if __name__ == "__main__":
    demonstrate_scaling()
