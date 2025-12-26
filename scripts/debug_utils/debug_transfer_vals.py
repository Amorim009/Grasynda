
import numpy as np
import pandas as pd
from scripts.debug_utils.demo_trend_transfer import trend_transfer_generate
from utils.load_data.config import DATASETS
from src.grasynda_unified import GrasyndaUnified

# Load Data
data_loader = DATASETS['M3']
df, _, _, _, freq_int = data_loader.load_everything('Monthly')
df_series = df[df['unique_id'] == 'M806'].copy()

# Configure Model
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

# Run Generation
print("Running Generation...")
# Validation Logic Inline
try:
    # 1. Decompose
    series = df_series['y'].values
    period = model.period
    from statsmodels.tsa.seasonal import STL
    res = STL(series, period=period, robust=False).fit()
    orig_trend = res.trend
    orig_seasonal = res.seasonal
    orig_remainder = res.resid
    
    # 2. Transfer
    transfer_ratio = 0.5
    s_transfer = orig_seasonal * transfer_ratio
    t_boosted = orig_trend + s_transfer
    
    # 3. Trend Gen Prep
    df_trend = df_series.copy()
    diff_t_boosted = pd.Series(t_boosted).diff().fillna(0)
    df_trend['diff_trend'] = diff_t_boosted
    df_trend['trend'] = t_boosted 
    
    print(f"Diff Trend Min/Max: {diff_t_boosted.min()}, {diff_t_boosted.max()}")
    if diff_t_boosted.isna().any(): print("ALERT: Diff Trend has NaNs!")
    
    # Calc Quantiles
    n_q = model._get_param('trend', 'n_quantiles')
    print(f"DEBUG: n_quantiles for trend: {n_q}")
    
    print("Calculating Quantiles...")
    df_trend['Quantile'] = model._get_quantiles(df_trend, 'diff_trend', component='trend')
    print(f"Quantiles Unique: {df_trend['Quantile'].unique()}")
    if df_trend['Quantile'].isna().any(): print("ALERT: Quantiles contains NaNs!")
    
    model._calc_transition_matrix(df_trend, component='trend')
    mat = model.transition_mats['trend']
    print(f"Transition Matrix Shape: {mat.shape}")
    print(f"Matrix Row 0 Sum: {mat[0].sum()}")
    print(f"Matrix Row 0 Non-Zero Idx: {np.where(mat[0] > 0)[0]}")
    
    # Generate
    print("Generating Dictionary...")
    # Trace _generate_quantile_series locally if possible? No, hidden.
    # We rely on output analysis.
    
    gen_trend_dict = model._create_synthetic_ts_quantile(
        df_trend, 'trend', 'diff_trend',
        sampling_type='discrete' 
    )
    
    syn_diff = gen_trend_dict['M806'].values
    print(f"Syn Diff Example: {syn_diff[:10]}")
    print(f"Syn Diff NaNs: {np.isnan(syn_diff).sum()}")
    
    syn_boosted_trend = np.zeros_like(syn_diff)
    syn_boosted_trend[0] = t_boosted[0]
    for i in range(1, len(syn_diff)):
        syn_boosted_trend[i] = syn_boosted_trend[i-1] + syn_diff[i]
        
    print(f"Final Syn Trend NaNs: {np.isnan(syn_boosted_trend).sum()}")

except Exception as e:
    print(f"CRASH: {e}")
    import traceback
    traceback.print_exc()
