
import os
import sys
import pandas as pd
import numpy as np

# Robust project root detection
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
sys.path.insert(0, project_root)

from src.grasynda_unified import GrasyndaUnified
from utils.load_data.config import DATASETS

def main():
    np.random.seed(42)
    
    # Load M3 Monthly
    ds_name, grp = 'M3', 'Monthly'
    print(f"Loading {ds_name} ({grp})...")
    loader = DATASETS[ds_name]
    df_all, _, _, _, freq_int = loader.load_everything(grp)
    
    # Pick 5 random series for the experiment pool
    all_uids = df_all['unique_id'].unique()
    sample_uids = np.random.choice(all_uids, 50, replace=False) # Large pool to find neighbors
    df_sample = df_all[df_all['unique_id'].isin(sample_uids)].copy()
    
    target_uid = sample_uids[0]
    print(f"\nEvaluating Target Series: {target_uid}")
    
    # Initialize Grasynda with Ensemble Size 5
    # Use 12 quantiles to make the matrices easier to visualize in text if needed
    model = GrasyndaUnified(
        period=freq_int,
        n_quantiles=12,
        ensemble_transitions=True,
        ensemble_size=5,
        components_to_model=['remainder']
    )
    
    # Trigger graph learning and ensembling
    _ = model.transform(df_sample)
    
    # Extract Matrices
    mats_orig = model.transition_mats['remainder']
    mats_ens = model.ensemble_transition_mats['remainder']
    
    if target_uid not in mats_orig or target_uid not in mats_ens:
        print("Error: Matrix not found in model state.")
        return
        
    m_orig = mats_orig[target_uid]
    m_ens = mats_ens[target_uid]
    
    # 1. Transition Counts (Non-zero entries)
    nz_orig = np.count_nonzero(m_orig)
    nz_ens = np.count_nonzero(m_ens)
    
    print("\n--- Transition Matrix Statistics ---")
    print(f"Original Non-zero Transitions: {nz_orig}")
    print(f"Ensembled Non-zero Transitions: {nz_ens}")
    print(f"Increase in Path Diversity: {((nz_ens - nz_orig) / nz_orig * 100):.1f}%")
    
    # 2. Sum Check (Probabilistic validity)
    sum_orig = np.sum(m_orig, axis=1)
    sum_ens = np.sum(m_ens, axis=1)
    print(f"\nNormalization Check (Original): Min Row Sum: {sum_orig.min():.4f}, Max Row Sum: {sum_orig.max():.4f}")
    print(f"Normalization Check (Ensemble): Min Row Sum: {sum_ens.min():.4f}, Max Row Sum: {sum_ens.max():.4f}")
    
    # 3. Inspect specific row (Middle of the range)
    mid_q = 6
    print(f"\n--- Probabilities for Quantile {mid_q} ---")
    orig_row = m_orig[mid_q]
    ens_row = m_ens[mid_q]
    
    print(f"Original Row {mid_q}: {orig_row}")
    print(f"Ensembled Row {mid_q}: {ens_row}")
    
    # Find active transitions for this row
    orig_active = np.where(orig_row > 0)[0]
    ens_active = np.where(ens_row > 0)[0]
    new_paths = [q for q in ens_active if q not in orig_active]
    
    print(f"\nOriginal active transitions: {orig_active}")
    print(f"Ensembled active transitions: {ens_active}")
    print(f"New structural paths discovered via partners: {new_paths}")

if __name__ == "__main__":
    main()
