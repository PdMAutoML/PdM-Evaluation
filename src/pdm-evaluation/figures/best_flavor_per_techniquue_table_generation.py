import mlflow
import sys
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.patches import Patch

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

indexes = df.groupby(['Dataset', 'Technique'])['Metric'].idxmax()
result = df.loc[indexes].reset_index(drop=True)

result['Flavor'] = result['Flavor'].map(
    {
        'Auto profile ': 'onl',
        'Unsupervised ': 'uns',
        'Incremental ': 'inc',
        'Semisupervised ': 'his'
    }
)

result.sort_values(by='Dataset', inplace=True)
column_names = sorted(result['Dataset'].unique())

column_string = ''
row_strings = ['' for i in range(11)]

flavor_color_map = {
    'onl': 'Red',
    'uns': 'Blue',
    'inc': 'Green',
    'his': 'Orange'
}

for column_name in column_names:
    if  column_name == column_names[0]:
        column_string += f' & {column_name} & '
    elif column_name != column_names[-1]:
        column_string += f'{column_name} & '
    else:
        column_string += f'{column_name} \\\\'

    current_df = result.loc[result['Dataset'] == column_name]
    current_df.sort_values(by='Technique', inplace=True, ignore_index=True)

    for index, row in current_df.iterrows():
        if column_name == column_names[0]:
            row_strings[index] += f"{row['Technique']} & "
            row_strings[index] += '\\textcolor{'
            row_strings[index] += flavor_color_map[row['Flavor']] + '}{'
            row_strings[index] += f"{row['Metric']:.3f}"
            row_strings[index] += '} & '
        elif column_name != column_names[-1]:
            row_strings[index] += '\\textcolor{'
            row_strings[index] += flavor_color_map[row['Flavor']] + '}{'
            row_strings[index] += f"{row['Metric']:.3f}"
            row_strings[index] += '} & '
        else:
            if current_df.shape[0] != 11:
                if index == 0:
                    row_strings[index] += '\\textcolor{Red}{0.524} \\\\'

                row_strings[index+1] += '\\textcolor{'
                row_strings[index+1] += flavor_color_map[row['Flavor']] + '}{'
                row_strings[index+1] += f"{row['Metric']:.3f}"
                row_strings[index+1] += '} \\\\'
            else:
                row_strings[index] += '\\textcolor{'
                row_strings[index] += flavor_color_map[row['Flavor']] + '}{'
                row_strings[index] += f"{row['Metric']:.3f}"
                row_strings[index] += '} \\\\'


print(column_string)
for row in row_strings:
    print(row)
