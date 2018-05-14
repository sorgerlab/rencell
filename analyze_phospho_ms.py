import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from synapseclient import Synapse
from msda import process_phospho_ms as pm
from msda import preprocessing as pr
from msda import pca


# Load individual 10-plexes, process and
# normalize data into a single dataset.
# --------------------------------------
syn = Synapse()
syn.login()

set1_ids = ['syn10534323', 'syn10534325', 'syn10534331', 'syn10534329']
set1_df_list = [pd.read_excel((syn.get(id).path)) for id in set1_ids]
# no recorded value for syn1053432
df_set1, _ = pm.merge(set1_df_list[:-1])
# Filter peptides with localization score less than 13
df_set1 = pm.filter_max_score(df_set1, max_score_cutoff=13.0)
set1_columns = [str(s).replace('default', 'Set1')
                for s in df_set1.columns.tolist()]
set1_columns = [s.replace('max_score', 'set1_max_score') for s in set1_columns]
df_set1.columns = set1_columns

set2_ids = ['syn10534326', 'syn10534328', 'syn10534332', 'syn10534333']
set2_df_list = [pd.read_excel((syn.get(id).path)) for id in set2_ids]
df_set2, _ = pm.merge(set2_df_list)
# Filter peptides with localization score less than 13
df_set2 = pm.filter_max_score(df_set2, max_score_cutoff=13.0)
set2_columns = [str(s).replace('default', 'Set2')
                for s in df_set2.columns.tolist()]
set2_columns = [s.replace('max_score', 'set2_max_score') for s in set2_columns]
df_set2.columns = set2_columns

df_meta = pd.read_csv(open(syn.get('syn11025099').path))
samples = df_meta['Sample'].tolist()
df_merged = pr.merge_batches([df_set1, df_set2],
                             df_meta, pMS=True, norm=False)
df_merged[samples] = df_merged[samples].replace([''], np.nan)
df_merged[samples] = df_merged[samples].astype(float)
df_merged.to_csv('pMS.csv')



# Plot PCA
# --------
color_dict = {'Set1': (0.97656, 0.19531, 0.19531),
              'Set2': (0.13672, 0.23438, 0.99609)}

df_meta.columns = ['tmt_label', 'sample']
df_meta['set'] = [s.split('_')[0] for s in df_meta['sample'].tolist()]
df_meta['day'] = [s[5:] for s in df_meta['sample'].tolist()]
df_pca, ev = pca.compute_pca(dfpms, df_meta)
dfp = pca.plot_scatter(df_pca, ev, color_col='set',
                       color_dict=color_dict)  # annotate_points='day')
