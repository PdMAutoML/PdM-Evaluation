from utils import loadDataset
import matplotlib.pyplot as plt
import pandas as pd

dataset = loadDataset.get_dataset("edp-wt")

index_of_turbine_of_interest = dataset['target_sources'].index('T07') # T06
turbine_of_interest_data = dataset['target_data'][index_of_turbine_of_interest]
failure_occurence_index = 41706 # T07 - 24101  # TO6 - 33005

turbine_of_interest_data = turbine_of_interest_data.loc[:failure_occurence_index]

hyd_oil_temp_avg = turbine_of_interest_data['Hyd_Oil_Temp_Avg']

min_value = hyd_oil_temp_avg.min()
max_value = hyd_oil_temp_avg.max()

fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(50,10), sharex=True)

axes[0].plot(hyd_oil_temp_avg, label='Hyd Oil Temp Avg', linewidth=1, color='blue')


for ax in axes[:2]:
    ax.fill_between(turbine_of_interest_data.index[-289:-1], min_value-1, [max_value + 1 for i in range(len(turbine_of_interest_data.index[-289:-1]))], color='grey', alpha=0.3)
    ax.fill_between(turbine_of_interest_data.index[-8929:-289], min_value-1, [max_value + 1 for i in range(len(turbine_of_interest_data.index[-8929:-289]))], color='red', alpha=0.3)
    ax.axvline(x=failure_occurence_index, color='red', linewidth=3)

# pb_scores = pd.read_csv('ProfileBased.csv')
# axes[2].plot(pb_scores['0'], label='Profile based', linewidth=1, color='purple')
# axes[2].fill_between(turbine_of_interest_data.index[-289:-1], pb_scores['0'].min(), [pb_scores['0'].max() for i in range(len(turbine_of_interest_data.index[-289:-1]))], color='grey', alpha=0.3)
# axes[2].fill_between(turbine_of_interest_data.index[-8929:-289], pb_scores['0'].min(), [pb_scores['0'].max() for i in range(len(turbine_of_interest_data.index[-8929:-289]))], color='red', alpha=0.3)
# axes[2].axvline(x=failure_occurence_index, color='red', linewidth=4)
#
# ltsf_scores = pd.read_csv('LTSFLinear.csv')
# axes[3].plot(ltsf_scores['0'], label='LTSF', linewidth=1, color='black')
# axes[3].fill_between(turbine_of_interest_data.index[-289:-1], ltsf_scores['0'].min(), [ltsf_scores['0'].max() for i in range(len(turbine_of_interest_data.index[-289:-1]))], color='grey', alpha=0.3)
# axes[3].fill_between(turbine_of_interest_data.index[-8929:-289], ltsf_scores['0'].min(), [ltsf_scores['0'].max() for i in range(len(turbine_of_interest_data.index[-8929:-289]))], color='red', alpha=0.3)
# axes[3].axvline(x=failure_occurence_index, color='red', linewidth=4)
#
handles, labels = [], []
for ax in axes:
    h, l = ax.get_legend_handles_labels()
    handles.extend(h)
    labels.extend(l)



fig.legend(handles, labels, loc='upper center', ncol=2)

plt.legend()
plt.show()
