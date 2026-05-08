"""
Verify Q10 Ensemble10 MASE ranks and show detailed breakdown
"""
import os
import pandas as pd

project_root = r'c:\Users\lhenr\Desktop\Grasynda'
model_names = ['NHITS', 'MLP', 'KAN']
tstr_datasets = ['M3 - Monthly', 'M3 - Quarterly', 'Tourism - Monthly', 'Tourism - Quarterly']

all_ranks = []

for model_name in model_names:
    tstr_path = os.path.join(project_root, 'assets', 'results', 'universal_experiments', f'{model_name}_TSTR_Results.csv')
    tstr_df = pd.read_csv(tstr_path)
    
    tstr_df = tstr_df[tstr_df['Dataset_Full'].isin(tstr_datasets)]
    tstr_df = tstr_df.drop(columns=['Jittering', 'SeasonalMBB'], errors='ignore')
    
    tstr_long = tstr_df.melt(id_vars='Dataset_Full', var_name='Method', value_name='MASE')
    
    mase_ranks = []
    for ds, ds_df in tstr_long.groupby('Dataset_Full'):
        ds_pivot = ds_df.set_index('Method')[['MASE']]
        ds_rank = ds_pivot.rank(ascending=True, method='min')
        mase_ranks.append(ds_rank)
    
    model_avg = pd.concat(mase_ranks).groupby(level=0).mean()
    model_avg.columns = [f'{model_name}_Rank']
    all_ranks.append(model_avg)

combined = pd.concat(all_ranks, axis=1)
combined['Avg_Rank'] = combined.mean(axis=1)

print("TSTR MASE Ranks by Model:")
print(combined.round(2).to_string())
print()
print(f"\nGrasynda_Uniform average rank: {combined.loc['Grasynda_Uniform', 'Avg_Rank']:.2f}")
