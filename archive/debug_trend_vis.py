
import pandas as pd
import numpy as np
import traceback
import sys
from src.grasynda_trend_visibility import GrasyndaTrendVisibility

def test_debug():
    print("Creating dummy data...")
    n = 100
    ds = pd.date_range(start='2020-01-01', periods=n, freq='D')
    y = np.linspace(0, 10, n) + np.sin(np.linspace(0, 10, n)) + np.random.normal(0, 0.1, n)
    
    df = pd.DataFrame({
        'unique_id': 'test_series',
        'ds': ds,
        'y': y
    })
    
    print("Initializing GrasyndaTrendVisibility...")
    model = GrasyndaTrendVisibility(period=10, visibility_type='horizontal')
    
    print("Running transform...")
    try:
        synth_df = model.transform(df)
        print("Transform successful!")
        print(synth_df.head())
    except Exception as e:
        print(f"\nERROR: {type(e).__name__}: {e}")
        print("\nFull traceback:")
        traceback.print_exc()
        
        # Add more debugging
        print("\n" + "="*60)
        print("DEBUG INFO:")
        print("="*60)
        
        # Try to show what went wrong
        import sys
        exc_type, exc_value, exc_tb = sys.exc_info()
        import traceback
        for frame_summary in traceback.extract_tb(exc_tb):
            print(f"File: {frame_summary.filename}")
            print(f"Line: {frame_summary.lineno}")
            print(f"Function: {frame_summary.name}")
            print(f"Code: {frame_summary.line}")
            print()

if __name__ == "__main__":
    test_debug()
