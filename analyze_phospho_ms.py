import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from synapseclient import Synapse
from msda import process_phospho_ms as pm
from msda import preprocessing as pr
from msda import pca
from msda import kmeans
from msda.clustering import plot_clustermap as pc
from msda import enrichr_api as ai
from msda import mapping

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
df_meta = df_meta.replace(['Set1_Day_0_5'], 'Set1_Day_0.25')
df_meta = df_meta.replace(['Set2_Day_0_5'], 'Set2_Day_0.25')
samples = df_meta['Sample'].tolist()
df_merged = pr.merge_batches([df_set1, df_set2],
                             df_meta, pMS=True, norm=False)
df_merged[samples] = df_merged[samples].replace([''], np.nan)
df_merged[samples] = df_merged[samples].astype(float)
df_merged.to_csv('pMS.csv')
dfp = df_merged.copy()
dfp['Uniprot_Id'] = [s.split('|')[1] for s in dfp.Uniprot_Id.tolist()]
dfp = pr.correct_gene_names(dfp)
dfp.insert(1, 'HGNC_Gene_Name', [mapping.get_name_from_symbol(sy)
                                 for sy in dfp.Gene_Symbol.tolist()])

# Plot PCA
# --------
color_dict = {'Set1': (0.97656, 0.19531, 0.19531),
              'Set2': (0.13672, 0.23438, 0.99609)}

df_meta.columns = ['tmt_label', 'sample']
df_meta['set'] = [s.split('_')[0] for s in df_meta['sample'].tolist()]
df_meta['day'] = [s[5:] for s in df_meta['sample'].tolist()]
df_pca, ev = pca.compute_pca(dfp, df_meta)
dfpca = pca.plot_scatter(df_pca, ev, color_col='set',
                       color_dict=color_dict)  # annotate_points='day')
ax = plt.gca()
ax.invert_yaxis()

# Plot kmeans
# -----------
dfc = dfp.replace([0], np.nan).dropna()
dfk = kmeans.cluster(dfc, samples, num_clusters=11)

cluster_map = {2: 'upregulated',
               6: 'upregulated mid to late stage',
               4: 'transiently upregulated mid-stage',
               8: 'downregulated',
               9: 'upregulated late',
               0: 'downregulated early',
               5: 'unperturbed',
               1: 'transiently upregulated early',
               10: 'transiently upregulated early',
               3: 'transiently upregulated early',
               7: 'downregulated early'}
dfk['kmeans_cluster_name'] = dfk['kmeans_cluster_number'].map(
    cluster_map)
dfcl = dfk['kmeans_cluster_name'].copy()
dfg = pd.concat([dfp, dfcl], axis=1)
dfg.to_csv('phosphoMS_baseline.csv')

dfk2 = dfk[dfk.kmeans_cluster_name != 'unperturbed'].copy()
fig = kmeans.plot(dfk2, df_meta, 'day',
                  cluster_id_col='kmeans_cluster_name',
                  ftr='phosphopeptide')
xcols = [0, 0.25, 1, 2, 3, 4, 7, 10, 14, 15]
plt.setp(fig.axes, xticks=range(10), xticklabels=xcols)
plt.subplots_adjust(wspace=0.4, bottom=0.06, hspace=0.4, top=0.95, left=0.05)
plt.savefig('pms_kmeans.pdf', dpi=300)


# Perform enrichment analysis from custom neuronal gene set library
# -----------------------------------------------------------------
def get_neuronal_enrichment(df):
    clusters = df['kmeans_cluster_name'].unique()

    dfl = []
    for cl in clusters:
        dfc = df[df['kmeans_cluster_name'] == cl].copy()
        dfe = ai.get_enrichment(dfc.index.tolist(),
                                'neuronal_gene_set_library')
        dfe2 = dfe[dfe['Adjusted P-value'] < 0.2].copy()
        dfe2['cluster_name'] = [cl]*len(dfe2)
        dfl.append(dfe2)
    dfn = pd.concat(dfl)
    dfp = dfn.pivot(index='Term', columns='cluster_name',
                    values='Adjusted P-value')
    return dfp

dfk.index = dfk.Gene_Symbol.tolist()

dfp = get_neuronal_enrichment(dfk)
dfp = dfp.apply(np.log10).multiply(-1)
pc(dfp.fillna(0), 'pMS_neuronal_enrichment.pdf',
   xticklabels=True)


def plot_enrichment(dfi, col):
    dfs = dfi.dropna(subset=[col])
    dfs = dfs[dfs[col] > 2].copy()
    dfs.index = [s.replace('GO_', '') for s in dfs.index.tolist()]
    dfs.index = [s.replace('CTRL_VS_WEST_EQUINE_ENC_VIRUS', '')
                 for s in dfs.index.tolist()]
    dfs.index = [s.replace('MOLENAAR_', '') for s in dfs.index.tolist()]
    dfs.index = [s.replace('LE_', '') for s in dfs.index.tolist()]
    dfs.index = [s.replace('WEST_EQUINE_ENC_VIRUS', '')
                 for s in dfs.index.tolist()]
    dfs.index = [s.replace('_', ' ') for s in dfs.index.tolist()]
    num_terms = len(dfs)
    height = 1.66 * num_terms
    fig, ax = plt.subplots(figsize=(12, height))
    try:
        dfs.plot(kind='barh', x=dfs.index, y=col,
                 color='b', alpha=0.8, legend=False, ax=ax)
        ax.tick_params(labelsize=18)
        plt.xlabel('- log10(p-value)', fontweight='bold', fontsize=24)
        plt.ylabel('Neuronal differentiation related GO Terms',
                   fontweight='bold',
                   rotation=90, fontsize=24)
        plt.title(col, fontweight='bold', fontsize=24)
        plt.subplots_adjust(left=0.7, right=0.9)
        col = col.replace(' ', '_')
        plt.savefig("pMS_enrichment_%s.pdf" % col, dpi=300)
    except TypeError:
        pass
