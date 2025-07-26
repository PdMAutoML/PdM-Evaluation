import pandas as pd
import itertools
import networkx as nx
import numpy as np
import choix
import matplotlib.pyplot as plt

def compare_rows(df):
    results = []
    # Iterate all unique row-pairs (i, j) with i < j
    for i, j in itertools.combinations(df.index, 2):
        # Subset to the two rows
        a, b = df.loc[i], df.loc[j]
        # Compare elementwise across all columns
        for col in df.columns:
            if col == 'Dataset':
                continue

            if a[col] > b[col]:
                results.append((i, j))
            elif a[col] < b[col]:
                results.append((j, i))
            # if equal, do nothing
    return results

# vus_df = pd.read_csv(f'vus_results.csv')

ad1_df = pd.read_csv('ad1_auc_results.csv').T
ad1_df.columns = ad1_df.iloc[0]
ad1_df = ad1_df.reset_index()
ad1_df = ad1_df[1:]
ad1_df.columns.values[0] = 'Dataset'
ad1_df.reset_index(inplace=True, drop=True)

results = compare_rows(ad1_df)

n_items = 10

params = choix.ilsr_pairwise(n_items, results)
print(params)
print("ranking (worst to best):", np.argsort(params))
