import mlflow
import sys
import random
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.patches import Patch
import matplotlib as mpl

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

print(sys.argv)
df = pd.read_csv(f'data_analysis_runtime.csv')

df.drop_duplicates(inplace=True, ignore_index=True)

df['Technique'] = df.apply(lambda row: 'KNN' if 'Distance' in row['Technique'] else row['Technique'], axis=1)

df['Technique'] = df.apply(
    lambda row: row['Technique'].lower().replace('unsupervised', '').capitalize() if 'unsupervised' in row[
        'Technique'].lower() else row['Technique'], axis=1)

df['Technique'] = df.apply(
    lambda row: row['Technique'].lower().replace('(uns)', '').capitalize() if '(uns)' in row['Technique'].lower() else
    row['Technique'], axis=1)

df['Technique'] = df.apply(
    lambda row: row['Technique'].lower().replace('(semi)', '').capitalize() if '(semi)' in row['Technique'].lower() else
    row['Technique'], axis=1)

df['Technique'] = df.apply(
    lambda row: row['Technique'].lower().replace('semi', '').capitalize() if 'semi' in row['Technique'].lower() else
    row['Technique'], axis=1)

df['Technique'] = df.apply(lambda row: 'IsolationForest' if 'Isolation' in row['Technique'] else row['Technique'],
                           axis=1)

df['Technique'] = df.apply(lambda row: 'LocalOutlierFactor' if 'Local' in row['Technique'] else row['Technique'],
                           axis=1)

df.loc[len(df)] = ['Auto profile ', 'Chronos', 'CMAPSS', 259200, 0.200]

df['Technique'] = df['Technique'].map({
    'NeighborProfile': 'NP',
    'IsolationForest': 'IF',
    'KNN': 'KNN',
    'LocalOutlierFactor': 'LOF',
    'OneClassSVM': 'OCSVM',
    'ProfileBased': 'PB',
    'TranAD': 'TRANAD',
    'USAD': 'USAD',
    'LTSF': 'LTSF',
    'Sand': 'SAND',
    'Chronos': 'CHRONOS'
}).astype(str)

all_techniques = sorted(df['Technique'].unique())
all_datasets = [
    'CMAPSS',
    'Navarchos',
    'FEMTO',
    'IMS',
    'EDP',
    'METRO',
    'XJTU',
    'BHD',
    'AZURE',
    'AI4I'
]

FONT_SIZE = 100

plt.rcParams.update({
    'font.size': 2.5*FONT_SIZE,  # Change this number to adjust font size
    'axes.titlesize': 2.5*FONT_SIZE,
    'axes.labelsize': 2.5*FONT_SIZE,
    'xtick.labelsize': 2.5*FONT_SIZE,
    'ytick.labelsize': 2.5*FONT_SIZE,
    'legend.fontsize': 2.5*FONT_SIZE
})

labels = all_techniques
categories = sorted(df['Technique'].unique())

import colorcet as cc
color_list = sns.color_palette(cc.glasbey, len(categories))
color_map = dict(zip(categories, color_list))

colors = [color_map[labels[i]] for i in range(len(labels))]

handles = [mpl.lines.Line2D([], [], color=colors[i], lw=2) for i in range(len(labels))]

fig, ax = plt.subplots(figsize=(10, 10))
# ax = fig_leg.add_subplot(111)
ax.axis('off')

leg = ax.legend(handles, labels,
               loc='center',
               ncol=len(labels),
               frameon=False,
               handlelength=1,
               )

for line in leg.get_lines():
    line.set_linewidth(150)

fig.savefig("runtime_legend.pdf", bbox_inches='tight', transparent=True)
plt.close(fig)