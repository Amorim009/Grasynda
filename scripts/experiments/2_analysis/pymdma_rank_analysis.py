"""
All metric scores per dataset and average rank per metric across datasets.
Includes Exact_Copy and Random_Noise in score tables.
Excludes Jittering from rankings. Sorted by Privacy rank.
"""

import os
import sys
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
sys.path.insert(0, project_root)

results_path = os.path.join(project_root, "assets", "results", "pymdma_metrics", "corrected", "final_results.csv")
df = pd.read_csv(results_path)

metrics = ['Authenticity', 'Fidelity', 'Diversity', 'Privacy']
df['DS'] = df['Dataset'] + '_' + df['Group']

# ---- All metrics per dataset (with Exact_Copy and Random_Noise) ----
for metric in metrics:
    print(f"\n{'=' * 80}")
    print(f"{metric.upper()} SCORES PER DATASET")
    print(f"{'=' * 80}")
    pivot = df.pivot(index='Method', columns='DS', values=metric)
    pivot['Average'] = pivot.mean(axis=1)
    pivot = pivot.sort_values('Average', ascending=False)
    print(pivot.round(4).to_string())

# ---- Average rank per metric (exclude sanity tests + Jittering) ----
df_rank = df[~df['Method'].isin(['Exact_Copy', 'Random_Noise', 'Jittering'])]

rank_dfs = []
for ds_key, ds_df in df_rank.groupby('DS'):
    ds_pivot = ds_df.set_index('Method')[metrics]
    ds_ranks = ds_pivot.rank(ascending=False, method='min')
    rank_dfs.append(ds_ranks)

avg_ranks = pd.concat(rank_dfs).groupby(level=0).mean()

for metric in metrics:
    print(f"\n{'=' * 60}")
    print(f"AVERAGE RANK: {metric} (1=best, lower is better)")
    print(f"{'=' * 60}")
    col = avg_ranks[[metric]].sort_values(metric)
    print(col.round(2).to_string())

# ---- Combined rank table sorted by Privacy ----
print(f"\n{'=' * 80}")
print("ALL AVERAGE RANKS (sorted by Privacy)")
print(f"{'=' * 80}")
avg_ranks = avg_ranks.sort_values('Privacy')
print(avg_ranks.round(2).to_string())
