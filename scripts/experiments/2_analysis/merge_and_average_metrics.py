
import pandas as pd
import os
import glob

# Paths
MODELS_DIR = r'assets/results/pymdma_metrics/models'
GLOBAL_SUMMARY_FILE = r'assets/results/pymdma_metrics/global_pymdma_summary.csv'

def merge_and_average():
    # 1. Collect all CSV files
    all_files = glob.glob(os.path.join(MODELS_DIR, "*.csv"))
    if not all_files:
        print(f"No CSV files found in {MODELS_DIR}")
        return

    print(f"Merging {len(all_files)} files...")
    
    list_df = []
    for f in all_files:
        try:
            df_temp = pd.read_csv(f)
            list_df.append(df_temp)
            print(f"  Loaded: {os.path.basename(f)}")
        except Exception as e:
            print(f"  Error loading {f}: {e}")

    if not list_df:
        return

    # 2. Concat all data
    full_df = pd.concat(list_df, ignore_index=True)
    
    # 3. Aggregated Summary (Average across datasets)
    # Exclude metadata columns and SpecWasserstein from the mean calculation
    metadata_cols = ['Dataset', 'Group', 'Method', 'N']
    numeric_cols = [c for c in full_df.columns if c not in metadata_cols and c != 'SpecWasserstein']
    
    # Group by Method and Calculate Mean
    summary_df = full_df.groupby('Method')[numeric_cols].mean().reset_index()
    
    # Round to 3 decimal places
    summary_df = summary_df.round(3)
    
    # Save Summary
    summary_df.to_csv(GLOBAL_SUMMARY_FILE, index=False)
    print(f"\nGlobal summary saved to {GLOBAL_SUMMARY_FILE}")
    print("-" * 60)
    print(summary_df.to_string(index=False))
    print("-" * 60)

if __name__ == "__main__":
    merge_and_average()
