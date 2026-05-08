"""
Save PyMDMA results to comprehensive CSV files.
"""

import os
import sys
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
sys.path.insert(0, project_root)

results_path = os.path.join(project_root, "assets", "results", "pymdma_metrics", "corrected", "final_results.csv")
out_dir = os.path.join(project_root, "assets", "results", "pymdma_metrics", "corrected", "tables")
os.makedirs(out_dir, exist_ok=True)

df = pd.read_csv(results_path)
df = df[~df['Method'].isin(['Jittering'])]

metrics = ['Authenticity', 'Fidelity', 'Diversity', 'Privacy']
df['DS'] = df['Dataset'] + '_' + df['Group']

# 1. Per-metric scores across datasets (one CSV per metric)
for metric in metrics:
    pivot = df.pivot(index='Method', columns='DS', values=metric)
    pivot['Average'] = pivot.mean(axis=1)
    pivot = pivot.sort_values('Average', ascending=False).round(4)
    path = os.path.join(out_dir, f"{metric.lower()}_scores_per_dataset.csv")
    pivot.to_csv(path)
    print(f"Saved: {path}")

# 2. All metrics for each dataset (one CSV per dataset)
for ds_key, ds_df in df.groupby('DS'):
    pivot = ds_df.set_index('Method')[metrics].round(4)
    path = os.path.join(out_dir, f"all_metrics_{ds_key}.csv")
    pivot.to_csv(path)
    print(f"Saved: {path}")

# 3. Average ranks per metric (one CSV)
df_rank = df[~df['Method'].isin(['Exact_Copy', 'Random_Noise'])]
rank_dfs = []
for ds_key, ds_df in df_rank.groupby('DS'):
    ds_pivot = ds_df.set_index('Method')[metrics]
    ds_ranks = ds_pivot.rank(ascending=False, method='min')
    rank_dfs.append(ds_ranks)

avg_ranks = pd.concat(rank_dfs).groupby(level=0).mean()
avg_ranks = avg_ranks.sort_values('Privacy').round(2)
path = os.path.join(out_dir, "average_ranks_per_metric.csv")
avg_ranks.to_csv(path)
print(f"Saved: {path}")

# 4. Average across all datasets (one CSV)
avg_scores = df.groupby('Method')[metrics].mean().round(4)
avg_scores = avg_scores.sort_values('Privacy', ascending=False)
path = os.path.join(out_dir, "average_scores_all_datasets.csv")
avg_scores.to_csv(path)
print(f"Saved: {path}")

print(f"\nAll files saved to: {out_dir}")
