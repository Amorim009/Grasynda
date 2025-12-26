
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
from utils.load_data.config import DATASETS
from src.grasynda_unified import GrasyndaUnified

def trend_transfer_generate(uid, df_series, model, transfer_ratio=0.5):
    """
    Implements 'Seasonality to Trend' transfer strategy.
    
    Logic:
    1. Split S into S_remain and S_transfer.
    2. Add S_transfer to Trend -> T_boosted.
    3. Generate synthetic version of T_boosted.
    4. Reconstruct: Syn_T_boosted + S_remain + Syn_R.
    """
    series = df_series['y'].values
    period = model.period
    if len(series) < 2 * period:
        return model.transform(df_series)

    # 1. Decompose
    res = STL(series, period=period, robust=False).fit()
    orig_trend = res.trend
    orig_seasonal = res.seasonal
    orig_remainder = res.resid
    
    # 2. Transfer Logic
    # We move 'transfer_ratio' of the Seasonality into the Trend
    # S = S_remain + S_trans
    s_transfer = orig_seasonal * transfer_ratio
    s_remain = orig_seasonal * (1 - transfer_ratio)
    
    t_boosted = orig_trend + s_transfer
    
    # 3. Component Generation
    
    # A. Generate Trend on 't_boosted'
    # Grasynda Trend Gen: Diff -> Quantize -> Gen -> Integrate
    df_trend = df_series.copy()
    diff_t_boosted = pd.Series(t_boosted).diff().fillna(0)
    
    # Prepare dummy DF for internal call
    df_trend['diff_trend'] = diff_t_boosted
    df_trend['trend'] = t_boosted # Used for base integration? No, integrate uses first value.
    
    # Calc Quantiles
    df_trend['Quantile'] = model._get_quantiles(df_trend, 'diff_trend', component='trend')
    model._calc_transition_matrix(df_trend, component='trend')
    
    # Generate
    gen_trend_dict = model._create_synthetic_ts_quantile(
        df_trend, 'trend', 'diff_trend',
        sampling_type='discrete' # Use discrete for stability
    )
    
    # Integrate (CAREFUL: integrate usually starts from df[component].iloc[0])
    # We need to ensure we start from t_boosted[0]
    # We can manually integrate
    syn_diff = gen_trend_dict[uid].values
    syn_boosted_trend = np.zeros_like(syn_diff)
    syn_boosted_trend[0] = t_boosted[0]
    for i in range(1, len(syn_diff)):
        syn_boosted_trend[i] = syn_boosted_trend[i-1] + syn_diff[i]
        
    # B. Generate Remainder (Standard, unscaled)
    # We assume standard generation for remainder
    # Use internal call for speed
    df_rem = df_series.copy()
    df_rem['remainder'] = orig_remainder
    df_rem['Quantile'] = model._get_quantiles(df_rem, 'remainder', component='remainder')
    model._calc_transition_matrix(df_rem, component='remainder')
    
    gen_rem_dict = model._create_synthetic_ts_quantile(
        df_rem, 'remainder', 'remainder',
        sampling_type='discrete'
    )
    syn_rem = gen_rem_dict[uid].values
    
    # 4. Reconstruct
    # y_new = Syn_T_boosted + S_remain + Syn_R
    y_final = syn_boosted_trend + s_remain + syn_rem
    
    df_final = df_series.copy()
    df_final['unique_id'] = f"{uid}_TrendTransfer"
    df_final['y'] = y_final
    
    print(f"DEBUG: T_boosted range: {t_boosted.min():.2f} to {t_boosted.max():.2f}")
    print(f"DEBUG: Syn_T_boosted range: {syn_boosted_trend.min():.2f} to {syn_boosted_trend.max():.2f}")
    print(f"DEBUG: Final Y range: {y_final.min():.2f} to {y_final.max():.2f}")
    if np.all(syn_boosted_trend == 0):
        print("DEBUG ALERT: Synthetic Trend is all ZEROS.")
    if np.any(np.isnan(syn_boosted_trend)):
        print("DEBUG ALERT: Synthetic Trend contains NaNs.")
    
    return df_final, t_boosted, syn_boosted_trend

def demonstrate_trend_transfer(dataset_name='M3', group='Monthly', uid='M806'):
    print(f"Demonstrating Seasonality->Trend Transfer on {uid}...")
    
    data_loader = DATASETS[dataset_name]
    df, _, _, _, freq_int = data_loader.load_everything(group)
    df_series = df[df['unique_id'] == uid].copy()
    
    # Config
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
    
    # 2. Transfer Generation (50% Transfer)
    df_trans, t_boosted, syn_t_boosted = trend_transfer_generate(uid, df_series, model, transfer_ratio=0.5)
    
    # 3. Plotting
    output_dir = 'assets/trend_transfer_demo'
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    # Top: Final Series
    axes[0].plot(df_series['ds'], df_series['y'], color='black', label='Original', lw=1, alpha=0.5)
    
    # Baseline removed to reduce clutter
    
    # Trend Transfer - HIGH VISIBILITY
    axes[0].plot(df_trans['ds'], df_trans['y'], color='red', label='Trend Transfer (50% Seas -> Trend)', lw=2.5, linestyle='-')
    axes[0].set_title(f"Seasonality to Trend Transfer Strategy ({uid})")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Bottom: What happened to the Trend?
    # Compare "Boosted Trend" vs "Synthetic Boosted Trend"
    axes[1].plot(df_series['ds'], t_boosted, color='green', linestyle='--', label='Input Boosted Trend (Target)', lw=2)
    axes[1].plot(df_series['ds'], syn_t_boosted, color='red', label='Synthetic Generated Trend', lw=2)
    
    axes[1].set_title("Mechanism: The 'Trend' Component absorbs seasonality and gets regenerated")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = f"{output_dir}/{uid}_transfer.png"
    plt.savefig(save_path, dpi=120)
    print(f"Saved to {save_path}")

if __name__ == "__main__":
    demonstrate_trend_transfer()
