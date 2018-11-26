# Load python dependencies
# ------------------------
import pandas as pd
from msda import kmeans
from synapseclient import Synapse
from msda import pca
import matplotlib.pyplot as plt
import numpy as np
from msda import enrichr_api as ai
from msda.clustering import plot_clustermap as pc
from msda import mapping
from msda import preprocessing as pr

# Load files from synapse
# -----------------------
syn = Synapse()
syn.login()

df = pd.read_csv('total_MS.csv')

# Wrangle data
# -----------
cols = df.columns.tolist()
new_cols = [c.replace('0p5', '0.25') for c in cols]
col_map = {o: n for o, n in zip(cols, new_cols)}
df = df.rename(columns=col_map)
df.insert(2, 'HGNC_Gene_Name', [mapping.get_name_from_symbol(sy)
                                 for sy in df.Gene_Symbol.tolist()])
dfc = df.replace([0], np.nan).dropna().copy()
samples = dfc.columns.tolist()[6:]


days = [s.split('_')[-1] for s in samples]
set_ = [s.split('_')[0] for s in samples]

# Create metadata specs from wrangeled data
df_meta = pd.DataFrame(list(zip(samples, set_, days)),
                       columns=['sample', 'set', 'day'])


# Compute PCA
# -----------
color_dict = {'Set1': (0.97656, 0.19531, 0.19531),
              'Set2': (0.13672, 0.23438, 0.99609)}

df_pca, ev = pca.compute_pca(dfc, df_meta)
dfp = pca.plot_scatter(df_pca, ev, color_col='set',
                       color_dict=color_dict) #  annotate_points='day')
plt.savefig('ms_PCA.pdf', dpi=300)


# Perform Kmeans clustering
# -------------------------
dfk = kmeans.cluster(dfc, samples, num_clusters=11)
cluster_map = {9: 'transiently upregulated early',
               2: 'transiently upregulated mid-stage',
               4: 'downregulated mid-stage',
               0: 'upregulated mid to late stages',
               7: 'upregulated late',
               6: 'upregulated',
               1: 'downregulated',
               5: 'downregulated',
               3: 'upregulated',
               8: 'unperturbed',
               10: 'downregulated'}
dfk['kmeans_cluster_name'] = dfk['kmeans_cluster_number'].map(
    cluster_map)
dfcl = dfk['kmeans_cluster_name'].copy()
dfg = pd.concat([df, dfcl], axis=1)

# Compute anova
# -------------
samples = [s for s in dfg.columns.tolist() if 'Day_' in s]
grp_index = np.arange(0, 20, 2)
groups = []
for i in gr_index:
    groups.append(samples[i:i+2])
dfg = pr.compute_anove(dfg, groups)    
dfg.to_csv('total_proteome_baseline.csv', index=False)

# dfk.insert(2, 'HGNC_Gene_Name', [mapping.get_name_from_symbol(sy)
#                                 for sy in dfk.Gene_Symbol.tolist()])

prtns = ['CDK1', 'CDK2', 'CDK4', 'CDK6',
         'CDK7', 'CDK8', 'CDK9', 'CDK12', 'CDK13',
         'CDK5',
         'SEMA5B', 'NLGN1', 'NLGN2', 'DCX',
         'PLXNB1', 'NLGN3', 'ROBO2',
         'SYP', 'MAPT', 'SLC1A2', 'GLUD1',
         'SLC6A1', 'SLC6A11',
         'REST', 'PTB2']

dfk2 = dfk[dfk.kmeans_cluster_name != 'unperturbed'].copy()
fig = kmeans.plot(dfk2, df_meta, 'day',
                  cluster_id_col='kmeans_cluster_name')
xcols = [0, 0.25, 1, 2, 3, 4, 7, 10, 14, 15]
plt.setp(fig.axes, xticks=range(10), xticklabels=xcols)
plt.subplots_adjust(wspace=0.4, bottom=0.06, hspace=0.4, top=0.95)
plt.savefig('ms_kmeans.pdf', dpi=300)


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
pc(dfp.fillna(0), 'MS_neuronal_enrichment.pdf',
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


def plot_barplots(df, df_meta, ftr):
    sd = {s: d for s, d in zip(df_meta['sample'].tolist(),
                               df_meta['day'].tolist())}
    df2 = df.copy()
    df2 = df2.rename(columns=sd)
    df2.index = df2.Gene_Symbol.tolist()
    df3 = df2[samples].copy()
    df3 = df2[samples]
    df4 = df3[samples].mean(axis=1, level=0)
    df4 = df4.loc[ftr]
    if type(df4) == pd.core.frame.DataFrame:
        df4 = df4.mean(axis=0)
    fig, ax = plt.subplots(figsize=(2, 5))
    df4.plot(kind='barh', color='blue', alpha=0.7, ax=ax)
    ax.set_title(ftr, fontweight='bold', fontsize=18)
    ax.tick_params(labelsize=18)
    ax = plt.gca()
    ax.invert_yaxis()
    plt.subplots_adjust(left=0.25)
    plt.savefig('mean_barplot_%s.pdf' % ftr, dpi=300)

ftrs = ['TUBB3', 'GFAP', 'PLP1', 'SYP',
        'GAP43', 'MAP1B', 'MAP2', 'DCX',
        'MKI67', 'MAPT']

for ft in ftrs:
    plot_barplots(df, df_meta, ft)
