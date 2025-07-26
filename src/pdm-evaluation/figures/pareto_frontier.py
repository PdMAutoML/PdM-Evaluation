import random
import sys

import matplotlib as mpl
import numpy as np
import pandas as pd
import paretoset
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches

RANDOM_STATE = 42

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
mpl.rcParams['svg.hashsalt'] = str(RANDOM_STATE)

flavors = [
    "Auto profile ",
    "Incremental ",
    "Semisupervised ",
    "Unsupervised "

]
formal_flavor_name_map = {
    'Auto profile ': 'online',
    'Incremental ': 'sliding',
    'Semisupervised ': 'historical',
    'Unsupervised ': 'unsupervised'
}

flavor_color_map = {
    "Auto profile ": "#D9534F",
    "Unsupervised ": "#5DA5DA",
    "Incremental ": "#60BD68",
    "Semisupervised ": "#F28E2B",
}

print(sys.argv)
df = pd.read_csv(f'data_analysis_runtime.csv')

df.drop_duplicates(inplace=True, ignore_index=True)

df['Technique'] = df.apply(lambda row: 'KNN' if 'Distance' in row['Technique'] else row['Technique'], axis=1)

df['Technique'] = df.apply(lambda row: row['Technique'].lower().replace('unsupervised', '').capitalize() if 'unsupervised' in row['Technique'].lower() else row['Technique'], axis=1)

df['Technique'] = df.apply(lambda row: row['Technique'].lower().replace('(uns)', '').capitalize() if '(uns)' in row['Technique'].lower() else row['Technique'], axis=1)

df['Technique'] = df.apply(lambda row: row['Technique'].lower().replace('(semi)', '').capitalize() if '(semi)' in row['Technique'].lower() else row['Technique'], axis=1)

df['Technique'] = df.apply(lambda row: row['Technique'].lower().replace('semi', '').capitalize() if 'semi' in row['Technique'].lower() else row['Technique'], axis=1)

df['Technique'] = df.apply(lambda row: 'IsolationForest' if 'Isolation' in row['Technique'] else row['Technique'], axis=1)

df['Technique'] = df.apply(lambda row: 'LocalOutlierFactor' if 'Local' in row['Technique'] else row['Technique'], axis=1)

df.loc[len(df)] = ['Auto profile ', 'Chronos', 'CMAPSS', 259200, 0.200]

df['is_max'] = (
    df['Metric']
      .eq(df.groupby(['Dataset','Technique','Flavor'])['Metric'].transform('max'))
      .astype(int)
)

df = df[df['is_max'] > 0]

df = df[~df['Dataset'].isin(['AI4I', 'IMS', 'AZURE', 'FEMTO'])]

df['Duration'] = np.log10(df['Duration'])

FONT_SIZE = 60

plt.rcParams.update({
    'font.size': FONT_SIZE,
    'axes.titlesize': FONT_SIZE,
    'axes.labelsize': FONT_SIZE,
    'xtick.labelsize': FONT_SIZE,
    'ytick.labelsize': FONT_SIZE,
    'legend.fontsize': FONT_SIZE
})


datasets = list(df['Dataset'].unique())
n_datasets = len(datasets)


fig, axes = plt.subplots(2, 3, figsize=(40, 30), sharex=True, sharey=True)
axes = axes.flatten()

pareto_rows = []

for idx, (current_dataset, grouped_df) in enumerate(df.groupby('Dataset')):
    ax = axes[idx]

    tick_dict = {
        0: '10⁰',
        1: '10¹',
        2: '10²',
        3: '10³',
        4: '10⁴',
        5: '10⁵',
        6: '10⁶',
    }

    ax.set_xticks(list(tick_dict.keys()))
    ax.set_xticklabels([tick_dict[i] for i in tick_dict.keys()])

    mask = paretoset.paretoset(
        grouped_df[['Duration', 'Metric']].values,
        sense=['min', 'max']
    )
    pareto_df = grouped_df[mask]
    pareto_rows.append(pareto_df)

    color_list = pareto_df["Flavor"].map(flavor_color_map)

    ax.scatter(grouped_df['Duration'], grouped_df['Metric'], c='grey', alpha=0.25, s=450)
    ax.scatter(pareto_df['Duration'], pareto_df['Metric'], c=color_list, s=450)
    frontier = pareto_df.sort_values('Duration')
    ax.plot(frontier['Duration'], frontier['Metric'], color='darkblue', linewidth=0.25)
    ax.set_xlabel('Duration (s)')
    ax.set_ylabel('AD1 AUC')
    ax.set_title(f'Dataset: {current_dataset}', pad=30)
    # ax.legend()

handles, labels = axes[0].get_legend_handles_labels()

patches = [
    mpatches.Patch(color=flavor_color_map[name], label=formal_flavor_name_map[name] + ' flavor')
    for name in flavor_color_map.keys()
]

patches.append(mpatches.Patch(color='grey', label='Dominated configuration'))
patches.append(mpatches.Patch(color='darkblue', label='Pareto frontier'))

fig.legend(
    handles=patches,
    # labels,
    loc='center',
    ncol=3,
    fontsize=FONT_SIZE,
)

# plt.tight_layout()
plt.tight_layout(h_pad=5)
plt.show()

fig.savefig("pareto_frontiers.pdf", format='pdf', bbox_inches='tight')

pareto_all = pd.concat(pareto_rows)

print("\nFlavor distribution on Pareto frontier:")
print(pareto_all['Flavor'].value_counts(normalize=True) * 100)

print("\nTechnique distribution on Pareto frontier:")
print(pareto_all['Technique'].value_counts(normalize=True) * 100)
