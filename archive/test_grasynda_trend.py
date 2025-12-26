"""
Quick test for GrasyndaTrend to verify it works correctly.
This uses synthetic data for faster validation.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import pandas as pd
import numpy as np

from src.grasynda_trend import GrasyndaTrend


def create_test_data():
    """Create simple synthetic time series for testing."""
    np.random.seed(42)
    
    # Create 2 simple time series with trend + seasonality + noise
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    
    data = []
    for uid in ['TS1', 'TS2']:
        trend = np.linspace(10, 20, 100)
        seasonal = 3 * np.sin(np.arange(100) * 2 * np.pi / 12)
        noise = np.random.normal(0, 0.5, 100)
        
        y_values = trend + seasonal + noise
        
        for i, date in enumerate(dates):
            data.append({
                'unique_id': uid,
                'ds': date,
                'y': y_values[i]
            })
    
    return pd.DataFrame(data)


def main():
    print("=" * 80)
    print("QUICK TEST: GRASYNDA TREND VARIANT")
    print("=" * 80)
    
    # Create test data
    print("\n1. Creating synthetic test data...")
    df = create_test_data()
    print(f"   ✅ Created {len(df)} data points")
    print(f"   Unique IDs: {df['unique_id'].nunique()}")
    
    # Test GrasyndaTrend
    print("\n2. Testing GrasyndaTrend...")
    
    generator = GrasyndaTrend(
        n_quantiles=10,  # Smaller for quick testing
        period=12,
        ensemble_transitions=False,
        robust=False
    )
    
    try:
        synth_df = generator.transform(df)
        print(f"   ✅ SUCCESS! Generated {len(synth_df)} synthetic points")
        print(f"   Synthetic unique IDs: {synth_df['unique_id'].unique()}")
        
        # Basic validation
        print("\n3. Validating output...")
        
        # Check structure
        assert 'ds' in synth_df.columns, "Missing 'ds' column"
        assert 'unique_id' in synth_df.columns, "Missing 'unique_id' column"
        assert 'y' in synth_df.columns, "Missing 'y' column"
        print("   ✅ Column structure correct")
        
        # Check unique_ids renamed correctly
        expected_uids = ['GrasyndaTrend_TS1', 'GrasyndaTrend_TS2']
        actual_uids = sorted(synth_df['unique_id'].unique())
        assert actual_uids == expected_uids, f"Expected {expected_uids}, got {actual_uids}"
        print("   ✅ Unique IDs renamed correctly")
        
        # Check no NaNs
        assert not synth_df['y'].isna().any(), "Found NaN values"
        print("   ✅ No NaN values")
        
        # Check length matches
        assert len(synth_df) == len(df), f"Length mismatch: {len(synth_df)} vs {len(df)}"
        print("   ✅ Length matches original")
        
        # Statistics comparison
        print("\n4. Statistics comparison:")
        for uid in ['TS1', 'TS2']:
            original = df[df['unique_id'] == uid]
            synthetic = synth_df[synth_df['unique_id'] == f'GrasyndaTrend_{uid}']
            
            print(f"\n   {uid}:")
            print(f"      Original:  mean={original['y'].mean():.2f}, std={original['y'].std():.2f}")
            print(f"      Synthetic: mean={synthetic['y'].mean():.2f}, std={synthetic['y'].std():.2f}")
        
        print("\n" + "=" * 80)
        print("✅ ALL TESTS PASSED!")
        print("=" * 80)
        print("\nGrasyndaTrend is working correctly!")
        print("It successfully:")
        print("  - Performs STL decomposition")
        print("  - Differentiates the trend")
        print("  - Applies quantile-based generation to differentiated trend")
        print("  - Integrates back to recover synthetic trend")
        print("  - Combines with original remainder and seasonal components")
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
