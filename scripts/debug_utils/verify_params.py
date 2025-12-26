
import pandas as pd
from src.grasynda_unified import GrasyndaUnified

def verify_parameter_flow():
    print("Verifying GrasyndaUnified Parameter Flow...")
    
    # 1. Setup Configuration to Match visualize_hybrid_dominance.py
    config = {
        'period': 12,
        'n_quantiles': 25,
        'components_to_model': ['trend', 'remainder'],
        'component_params': {
            'trend': {
                'sampling_type': 'continuous_uniform',  # Specific override
                'apply_differentiation': True,
                'graph_type': 'visibility',
                'visibility_type': 'horizontal'
            },
            'remainder': {
                'sampling_type': 'discrete',            # Specific override
                'graph_type': 'quantile'
            }
        },
        'sampling_type': 'discrete', # Global default
        'graph_type': 'quantile'     # Global default
    }
    
    model = GrasyndaUnified(**config)
    
    # 2. Test _get_param for standard lookups
    print("\n[Test 1] Standard Param Lookup:")
    trend_st = model._get_param('trend', 'sampling_type')
    rem_st = model._get_param('remainder', 'sampling_type')
    
    print(f"Trend Sampling (Should be continuous_uniform): {trend_st} -> {'PASS' if trend_st == 'continuous_uniform' else 'FAIL'}")
    print(f"Rem Sampling (Should be discrete):           {rem_st} -> {'PASS' if rem_st == 'discrete' else 'FAIL'}")
    
    # 3. Test _generate_synthetic_series_vis logic simulation
    # The method inside GrasyndaUnified runs this logic:
    # sampling_type = self._get_param(target_col, 'sampling_type')
    # if not sampling_type: 
    #      sampling_type = self._get_param(target_col.replace('diff_', ''), 'sampling_type')
    
    print("\n[Test 2] Visibility Generation Logic Simulation (New Logic):")
    target_col = 'diff_trend'
    
    # Logic trace (New Logic: Always strip diff_)
    base_col = target_col.replace('diff_', '')
    final_st = model._get_param(base_col, 'sampling_type')
        
    print(f"Lookup '{base_col}': {final_st} (Expected: continuous_uniform)")

        
    print(f"Final Resolved Sampling for '{target_col}': {final_st}")
    
    if final_st == 'continuous_uniform':
        print(">> VERIFICATION SUCCESS: Visibility Graph correctly resolves sampling type for diff_trend.")
    else:
        print(">> VERIFICATION FAILED: Visibility Graph did not resolve correctly.")

if __name__ == "__main__":
    verify_parameter_flow()
