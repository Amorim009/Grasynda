"""
Plot: Average Rank in MASE (TSTR) vs Average Rank in Privacy (PyMDMA)
X-axis: Average MASE Rank (lower = better forecasting) - averaged across NHITS, MLP, KAN
Y-axis: Average Privacy Rank (lower = more private)
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
sys.path.insert(0, project_root)

# ---- Load PyMDMA Privacy results ----
pymdma_path = os.path.join(project_root, "assets", "results", "pymdma_metrics", "corrected", "final_results.csv")
pymdma_df = pd.read_csv(pymdma_path)
pymdma_df = pymdma_df[~pymdma_df['Method'].isin(['Exact_Copy', 'Random_Noise', 'Jittering', 'SeasonalMBB'])]
pymdma_df['DS'] = pymdma_df['Dataset'] + '_' + pymdma_df['Group']

privacy_ranks = []
for ds_key, ds_df in pymdma_df.groupby('DS'):
    ds_pivot = ds_df.set_index('Method')[['Privacy']]
    ds_rank = ds_pivot.rank(ascending=False, method='min')
    privacy_ranks.append(ds_rank)
avg_privacy_rank = pd.concat(privacy_ranks).groupby(level=0).mean()
avg_privacy_rank.columns = ['Privacy_Rank']

# ---- Load TSTR MASE results for NHITS, MLP, KAN ----
tstr_datasets = ['M3 - Monthly', 'M3 - Quarterly', 'Tourism - Monthly', 'Tourism - Quarterly']
model_names = ['NHITS', 'MLP', 'KAN']

all_model_ranks = []

for model_name in model_names:
    tstr_path = os.path.join(project_root, "assets", "results", "universal_experiments", f"{model_name}_TSTR_Results.csv")
    tstr_df = pd.read_csv(tstr_path)
    
    tstr_df = tstr_df[tstr_df['Dataset_Full'].isin(tstr_datasets)]
    tstr_df = tstr_df.drop(columns=['Jittering', 'SeasonalMBB'], errors='ignore')
    
    tstr_long = tstr_df.melt(id_vars='Dataset_Full', var_name='Method', value_name='MASE')
    
    # Rank MASE per dataset (lower MASE = rank 1)
    mase_ranks = []
    for ds, ds_df in tstr_long.groupby('Dataset_Full'):
        ds_pivot = ds_df.set_index('Method')[['MASE']]
        ds_rank = ds_pivot.rank(ascending=True, method='min')
        ds_rank.columns = ['MASE_Rank']
        mase_ranks.append(ds_rank)
    
    model_avg_rank = pd.concat(mase_ranks).groupby(level=0).mean()
    model_avg_rank.columns = [f'{model_name}_MASE_Rank']
    all_model_ranks.append(model_avg_rank)

# Average across the 3 models
combined_ranks = pd.concat(all_model_ranks, axis=1)
combined_ranks['MASE_Rank'] = combined_ranks.mean(axis=1)
avg_mase_rank = combined_ranks[['MASE_Rank']]

print("MASE Ranks by Model:")
print(combined_ranks.round(2).to_string())
print()

# ---- Map method names between TSTR and PyMDMA ----
name_map = {
    'DBA': 'DBA',
    'TSMixup': 'TSMixup',
    'TimeWarping': 'TimeWarping',
    'MagnitudeWarping': 'MagnitudeWarping',
    'Scaling': 'Scaling',
    'Grasynda_Vis_Horizontal': 'Hybrid_VisH_Ensemble10',
    'Grasynda_Uniform': 'Hybrid_Q10_Ensemble10_Continuous',
    'Grasynda_Vis_Natural': 'Hybrid_Q10_NoEnsemble_Continuous',
}

avg_mase_rank['PyMDMA_Method'] = avg_mase_rank.index.map(lambda x: name_map.get(x, None))
avg_mase_rank = avg_mase_rank.dropna(subset=['PyMDMA_Method'])
avg_mase_rank = avg_mase_rank.set_index('PyMDMA_Method')

# ---- Merge ----
merged = avg_mase_rank.join(avg_privacy_rank, how='inner')
print("Merged data:")
print(merged.round(2).to_string())
print()

# ---- Define colors ----
grasynda_methods = ['Hybrid_VisH_Ensemble10', 'Hybrid_Q10_Ensemble10_Continuous', 'Hybrid_Q10_NoEnsemble_Continuous']
colors = []
for m in merged.index:
    if m in grasynda_methods:
        colors.append('#E63946')
    else:
        colors.append('#457B9D')

# ---- Plot ----
fig, ax = plt.subplots(figsize=(10, 7))

ax.scatter(merged['MASE_Rank'], merged['Privacy_Rank'], 
           c=colors, s=150, zorder=5, edgecolors='white', linewidth=1.5)

for method, row in merged.iterrows():
    label = method.replace('Hybrid_', '').replace('_Continuous', '').replace('_', ' ')
    ax.annotate(label, (row['MASE_Rank'], row['Privacy_Rank']),
                textcoords="offset points", xytext=(10, 5),
                fontsize=9, fontweight='bold')

ax.set_xlabel('Average MASE Rank (TSTR) →  lower = better forecasting\n(averaged across NHITS, MLP, KAN)', fontsize=12)
ax.set_ylabel('Average Privacy Rank (PyMDMA) →  lower = more private', fontsize=12)
ax.set_title('MASE Forecasting Performance vs Privacy\n(Average Rank across M3 & Tourism datasets)', fontsize=14, fontweight='bold')

ax.invert_xaxis()
ax.invert_yaxis()

from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#E63946', edgecolor='white', label='Grasynda'),
    Patch(facecolor='#457B9D', edgecolor='white', label='Baseline'),
]
ax.legend(handles=legend_elements, loc='upper left', fontsize=11)

ax.grid(True, alpha=0.3)
ax.set_axisbelow(True)

plt.tight_layout()
out_path = os.path.join(project_root, "assets", "results", "pymdma_metrics", "corrected", "mase_vs_privacy_rank.png")
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"Saved plot to: {out_path}")
plt.close()
