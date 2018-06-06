import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_excel('Table8_DMSO_Ken_Confluency_Raw.xlsx', header=7)
value_cols = df.columns.tolist()[2:]
dfo = pd.melt(df, id_vars='Elapsed (hr)', value_vars=value_cols)
dfo['Elapsed (hr)'] = dfo['Elapsed (hr)'].astype(int)
dfo['passage_number'] = [s.split('_')[1]
                         for s in dfo['variable'].tolist()]
dfo['agent'] = [s.split('_')[0]
                for s in dfo.variable.tolist()]

dfo = dfo.replace(['Ken'], 'kenpaullone')
color_dict = {'DMSO': (0, 0, 0),
              'mevastatin': (0.13672, 0.23438, 0.99609),
              'kenpaullone': (0.97656, 0.19531, 0.19531)}


fig, ax = plt.subplots()
ax = sns.tsplot(data=dfo, time='Elapsed (hr)', value='value',
                condition='agent', unit='passage_number',
                ci='sd', color=color_dict)
ax.set_xlabel('Elapsed time (hours)', fontweight='bold')
ax.set_ylabel('confluency (%)', fontweight='bold')
plt.savefig('confluency.pdf', dpi=300)
