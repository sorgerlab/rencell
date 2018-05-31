import synapseclient
import pandas as pd
import matplotlib.pyplot as plt
from msda import preprocessing as pr
from msda import pca
from matplotlib.gridspec import GridSpec
import numpy as np
import seaborn as sns
from msda import enrichr_api as ai

syn = synapseclient.Synapse()
syn.login()


def normalize(df):
    del df['Number of peptides']
    treatment = df.columns.tolist()[0].split('_')[0]
    df2 = df.copy()
    ctrl = "%s_Day_0" % treatment
    df2 = df2.div(df2[ctrl], axis=0)
    return df2


def change_op(df):
    cols = df.columns.tolist()
    new_cols = [c.replace('0p5', '0.25') for c in cols]
    col_map = {o: n for o, n in zip(cols, new_cols)}
    df = df.rename(columns=col_map).copy()
    return df

df_dmso = pd.read_csv(open(syn.get('syn11701294').path))
df_dmso = df_dmso.set_index('Gene_Symbol', drop=False)
df_dmso = change_op(df_dmso)
df_dmso = pr.merge_duplicate_features(df_dmso)
df_dmso = normalize(df_dmso)

df_ken = pd.read_csv(open(syn.get('syn11701292').path))
df_ken = df_ken.set_index('Gene_Symbol', drop=False)
df_ken = change_op(df_ken)
df_ken = pr.merge_duplicate_features(df_ken)
df_ken = normalize(df_ken)

df_mev = pd.read_csv(open(syn.get('syn11701293').path))
df_mev = df_mev.set_index('Gene_Symbol', drop=False)
df_mev = change_op(df_mev)
df_mev = pr.merge_duplicate_features(df_mev)
df_mev = normalize(df_mev)

dfc = pd.concat([df_dmso, df_ken, df_mev], axis=1)
samples = [c for c in dfc.columns.tolist() if 'peptides' not in c]

color_dict = {'DMSO': (0, 0, 0),
              'mevastatin': (0.13672, 0.23438, 0.99609),
              'kenpaullone': (0.97656, 0.19531, 0.19531)}

df_meta = pd.read_csv('perturbation_metadata.csv')
dfc2 = dfc.apply(np.log2)
df_pca, ev = pca.compute_pca(dfc2, df_meta)
dfp = pca.plot_scatter(df_pca, ev, color_col='drug',
                       color_dict=color_dict)
#  annotate_points='sample')
ax = plt.gca()
ax.invert_yaxis()


def make_long_table(dfc, samples, prtns):
    dfc2 = dfc[samples].copy()
    dfc2[samples] = dfc2[samples].div(dfc2['DMSO_Day_0'],
                                      axis=0)
    index_ = dfc2.loc[prtns[0]].index
    ind_split = [s.split('_') for s in index_]
    dfm = pd.DataFrame(ind_split, columns=['drug', 'd', 'day'])
    del dfm['d']
    dfm.index = index_
    dfcc = pd.concat([dfm, dfc2.loc[prtns].T], axis=1)
    dfcc[prtns] = dfcc[prtns].apply(np.log2)
    dfcc = dfcc.replace(['Mev'], 'mevastatin')
    dfcc = dfcc.replace(['Ken'], 'kenpaullone')
    dfcc['day'] = dfcc['day'].astype(float)
    return dfcc


prtns = ['NTN1', 'DCX', 'MAPT', 'HMGCR', 'GSK3A', 'GSK3B',
         'CDK1', 'CDK2',  'CDK4', 'CDK6', 'CDK5', 'CDK7',
         'CDK12', 'CDK9', 'CDK13']

genes = ['ROBO3', 'RGS10', 'SLC6A6', 'HMGCR', 'KRT17',
         'CCNA2', 'TK1', 'DCX', 'GSK3A', 'GSK3B',
         'MAPT', 'NTN1', 'CDK1', 'CDK6', 'CDK12']


def plot_long_table(dfcc, prtns):

    # Define plot properties
    # ----------------------
    grid_height = int(np.ceil(len(prtns) / 5))
    grid_dims = (grid_height, 5)
    fig = plt.figure(figsize=(35, 7 * grid_height), dpi=100)
    GridSpec(*grid_dims)
    color_dict = {'DMSO': (0, 0, 0),
                  'kenpaullone': (0.97656, 0.19531, 0.19531),
                  'mevastatin': (0.13672, 0.23438, 0.99609)}

    # Loop over and plot each cluster as a subplot
    # --------------------------------------------
    for ain, cl in enumerate(prtns):
        ax_loc = np.unravel_index(ain, grid_dims)
        ax = plt.subplot2grid(grid_dims, ax_loc)
        sns.pointplot(data=dfcc, x='day', y=cl, hue='drug',
                      ax=ax, palette=color_dict, scale=0.5,
                      legend=False)
        ax.set_title(cl, fontweight='bold', fontsize=12)
        ylims = ax.get_ylim()
        ymax = np.max((2, ylims[1]))
        ymin = np.min((-2, ylims[0]))
        xlims = ax.get_xlim()
        ax.plot(xlims, [0, 0], '--k', alpha=0.5)
        ax.set_ylim((ymin, ymax))
        ax.set_ylabel('')
        ax.set_xlabel('')
        ax.legend([])
        ax.set_xticklabels([])
        if ain < len(prtns)-5:
            ax.set_xticks([])
    plt.subplots_adjust(right=0.9, left=0.1,
                        wspace=0.3, hspace=0.5)    
    return fig


def get_delta(dfc):
    mev_late = ['Mev_Day_7', 'Mev_Day_10', 'Mev_Day_14', 'Mev_Day_15']
    ken_late = ['Ken_Day_7', 'Ken_Day_10', 'Ken_Day_14', 'Ken_Day_15']
    dmso_late = ['DMSO_Day_7', 'DMSO_Day_10',
                 'DMSO_Day_14', 'DMSO_Day_15']
    dfc2 = dfc.copy()
    dfc2['dmso_late_mean'] = dfc2[dmso_late].mean(axis=1)
    dfc2['ken_late_mean'] = dfc2[ken_late].mean(axis=1)
    dfc2['mev_late_mean'] = dfc2[mev_late].mean(axis=1)
    dfc2['ken_delta'] = dfc2['ken_late_mean'] - dfc2['dmso_late_mean']
    dfc2['mev_delta'] = dfc2['mev_late_mean'] - dfc2['dmso_late_mean']
    dfc2['ken_delta'] = dfc2['ken_delta'].apply(np.abs)
    dfc2['mev_delta'] = dfc2['mev_delta'].apply(np.abs)
    return dfc2


def get_delta_enrichment(df, cond='mev_delta',
                         delta_thresh=2):
    df2 = df[df[cond] > delta_thresh].copy()
    genes = df2.index.tolist()
    dfe = ai.get_enrichment(genes,
                            'neuronal_gene_set_library')
    return dfe


dfcc = make_long_table(dfc, samples, genes)
fig = plot_long_table(dfcc, genes)
