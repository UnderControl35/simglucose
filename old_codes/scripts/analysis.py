import numpy as np
import pandas as pd
import pickle as pkl
import seaborn as sns
import matplotlib.pyplot as plt
from simglucose.analysis.risk import risk_index

with open('/home/berk/VS_Project/simglucose/examples/trajectories/DATA_eps_11-2022-11-21 20:20:00.pkl', 'rb') as handle:
    data = pkl.load(handle)

col = []
for key in data[0].keys(): col.append(key)
df = pd.DataFrame(columns=col)
df2 = pd.DataFrame(columns=col)

chunks = []
risk = []
for i,traj in enumerate(data):
    for key in traj.keys():
        risk_i, lbgi, hbgi = [], [], []
        df[key] = pd.Series(np.squeeze(traj[key]))
    for idx in range(len(traj['observations'])):
        x,y,z = risk_index(traj['observations'][idx], horizon=1)
        lbgi.append(x)
        hbgi.append(y)
        risk_i.append(z)


    hypo_percent = (traj['observations'] < 70).sum() / len(traj['observations'])
    hyper_percent = (traj['observations'] > 180).sum() / len(traj['observations'])
    event = hypo_percent + hyper_percent

    df['risk_i'], df['LBGI'], df['HBGI'] = risk_i, lbgi, hbgi
    risk.append(risk_i)

    #TODO: Seperate trajectories into different segments..
    print('HYPO: %{:.3f}, HYPER: %{:.3f}, EVENT: %{:.3f}'.format(hypo_percent, hyper_percent, event))
    df['hypo_percent'], df['hyper_percent'], df['event'] = hypo_percent, hyper_percent, event
    df['traj_id'] = i
    chunks.append(df)

result = pd.concat(chunks)
pd.set_option('float_format', '{:f}'.format)
pd.set_option("display.max_rows", len(result))
#print(result)
print(result.describe())

fig = plt.figure(figsize=(12,10), dpi=80)
hm_df = pd.pivot_table(result, index='actions', columns='observations', values='rewards')
result = result.drop(columns='terminals', axis=1)

#print(hm_df)
#print(result.columns)
mode = 'error'

if mode == 'corr':
    corr = result.corr()
    plt.title("Correlation Table")
    plt.yticks(rotation=0)
    sns.heatmap(corr, annot=True, cmap="viridis")
elif mode == 'error':
    ###### https://seaborn.pydata.org/examples/wide_form_violinplot.html #####
    ###### https://jakevdp.github.io/PythonDataScienceHandbook/04.03-errorbars.html ####
    sns.set_theme(style="whitegrid")
    

    # "Melt" the dataset to "long-form" or "tidy" representation
    result = pd.melt(result, "risk_i", var_name="measurement")

    # Initialize the figure
    f, ax = plt.subplots()
    sns.despine(bottom=True, left=True)

    # Show each observation with a scatterplot
    sns.stripplot(
        data=result, x="value", y="measurement", hue="traj_id",
        dodge=True, alpha=.25, zorder=1, legend=False
    )

    # Show the conditional means, aligning each pointplot in the
    # center of the strips by adjusting the width allotted to each
    # category (.8 by default) by the number of hue levels
    sns.pointplot(
        data=result, x="value", y="measurement", hue="traj_id",
        join=False, dodge=.8 - .8 / 3, palette="dark",
        markers="d", scale=.75, errorbar=None
    )

    # Improve the legend
    sns.move_legend(
        ax, loc="lower right", ncol=3, frameon=True, columnspacing=1, handletextpad=0
    )
elif mode == 'test':
    print(result['risk_i'])


plt.show()