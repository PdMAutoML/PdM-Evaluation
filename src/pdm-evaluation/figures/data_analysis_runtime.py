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
FONT_SIZE = 100

plt.rcParams.update({
    'font.size': FONT_SIZE,  # Change this number to adjust font size
    'axes.titlesize': FONT_SIZE,
    'axes.labelsize': 2*FONT_SIZE,
    'xtick.labelsize': 1.5*FONT_SIZE,
    'ytick.labelsize': 2*FONT_SIZE,
    'legend.fontsize': FONT_SIZE
})

if len(sys.argv) <= 1:
    mlflow_server_url_list = ["http://127.0.0.1:8080/", "http://155.207.202.68:8081/"]

    current_df = pd.DataFrame([], columns=['Flavor', 'Technique', 'Dataset', 'Duration', 'Metric'])
    for current_url in mlflow_server_url_list:
        mlflow.set_tracking_uri(current_url)
        client = mlflow.tracking.MlflowClient()

        rules=[("params.postprocessor","Default")]
        metric="metrics.AD1_AUC"

        # techniques = ["PB", "KNN", "IF", "LOF", "NP", "SAND", "OCSVM", "DISTANCE BASED", "LTSF", "TRANAD", "USAD","CHRONOS"]
        datasets = [
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

        for dataset in datasets:
            print(dataset)
            experiments = client.search_experiments(filter_string=f"name LIKE '%{dataset}%'")

            for experiment in experiments:
                if "correlated" in experiment.name:
                    continue

                print(experiment.name)

                if ('XJTU' in experiment.name and 'XJTU PH' not in experiment.name) or 'TSB' in experiment.name:
                        print(f'Skipping {experiment.name}')
                        continue

                exp_id = experiment.experiment_id

                if len(sys.argv) == 1:
                    runs = mlflow.search_runs(exp_id)
                else:
                    runs = mlflow.search_runs(exp_id, filter_string=f'attributes.created > {sys.argv[1]}')

                max_value = None

                for index, run in runs.iterrows():
                    skip = False
                    for rule in rules:
                        if rule[0] not in run.keys():
                            skip = True
                            break

                        if run[rule[0]] is None:
                            skip = True
                            break

                        elif rule[1] in str(run[rule[0]]):
                            continue

                        else:
                            skip = True
                            break

                    if skip:
                        continue


                    duration = run["end_time"] - run['start_time']
                    value_duration_seconds = duration.seconds

                    if metric not in run.keys():
                        print(f"Not such metric, available: {run.keys()}")
                        continue
                    else:
                        value_metric = run[metric]

                    new_df = pd.DataFrame([[experiment.name.split(dataset)[0], run.loc['params.method'], dataset, value_duration_seconds, value_metric]], columns=current_df.columns)
                    current_df = pd.concat([current_df, new_df], ignore_index=True)


    current_df.to_csv(f'data_analysis_runtime.csv', index=False)
else:
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
    total = 0

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

    for flavor in flavors:
        current_df = df[df['Flavor'] == flavor].copy()

        if flavor == 'Auto profile ':
            current_df.loc[len(current_df)] = ['Auto profile ', 'Chronos', 'CMAPSS', 259200, 0.200]


        print(current_df.shape)
        total += current_df.shape[0]

        # current_df['DurationScaled'] = (df['Duration'] - df['Duration'].mean()) / df['Duration'].std()
        # current_df = current_df[current_df['Duration'] <= 60]

        current_df['log_Duration'] = np.log10(current_df['Duration'])

        max_metric_per_dataset_dict = current_df.groupby('Dataset')['Metric'].max().to_dict()
        min_metric_per_dataset_dict = current_df.groupby('Dataset')['Metric'].min().to_dict()

        current_df['NormalizedMetric'] = current_df.apply(
            lambda row: (row['Metric'] - min_metric_per_dataset_dict[row['Dataset']])
                            /
                        (max_metric_per_dataset_dict[row['Dataset']] - min_metric_per_dataset_dict[row['Dataset']]),
            axis=1)


        fig, ax = plt.subplots(figsize=(100, 100))

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

        ax.tick_params(axis='both', pad=30)

        sns.violinplot(data=current_df, x='log_Duration', y='Dataset', color=".8", orient='h', order=all_datasets)

        # sns.stripplot(data=current_df, x='Dataset', y='log_Duration', hue='Technique', alpha='', ax=ax)
        # plt.title('Log-scaled duration by Dataset and Technique')
        # plt.xlabel('Dataset')
        # plt.ylabel('Log-scaled Duration')
        # plt.legend(title='Technique')
        #
        # plt.tight_layout()
        # plt.show()

        # if 'Semi' not in flavor:
        x_positions = current_df['Dataset'].map({
            'CMAPSS': 0,
            'Navarchos': 1,
            'FEMTO': 2,
            'IMS': 3,
            'EDP': 4,
            'METRO': 5,
            'XJTU': 6,
            'BHD': 7,
            'AZURE': 8,
            'AI4I': 9
        }
        )
        # else:
        #     x_positions = current_df['Dataset'].map({
        #         'CMAPSS': 0,
        #         'FEMTO': 1,
        #     })

        x_jittered = x_positions + np.random.uniform(-0.1, 0.1, size=len(current_df))

        categories = sorted(df['Technique'].unique())

        import colorcet as cc
        color_list = sns.color_palette(cc.glasbey, len(categories))
        color_map = dict(zip(categories, color_list))

        for i in range(len(current_df)):
            plt.scatter(
                current_df['log_Duration'].iloc[i],
                x_jittered.iloc[i],
                alpha=current_df['NormalizedMetric'].iloc[i],
                color=color_map[current_df['Technique'].iloc[i]],
                s=4000
            )

        # plt.tick_params(axis='both', labelsize=FONT_SIZE)

        plt.yticks([i for i in range(10)], [
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
        # if 'Semi' not in flavor else [
        #     'CMAPSS',
        #     'FEMTO'
        # ]
        , ha='right', fontsize=1.65*FONT_SIZE)

        plt.ylabel('')
        plt.xlabel('Execution time (s)', fontsize=2.5*FONT_SIZE)
        # plt.title(f'Log-scaled duration by dataset and technique for the {formal_flavor_name_map[flavor]} flavor', fontsize=FONT_SIZE)

        # legend_handles = [Patch(color=color_map[cat], label=cat) for cat in categories]

        # plt.legend(handles=legend_handles, fontsize=30)

        plt.show()
        fig.savefig(f"{formal_flavor_name_map[flavor]}_log_duration.pdf", format='pdf', bbox_inches='tight')

        # break

    assert total == len(df)