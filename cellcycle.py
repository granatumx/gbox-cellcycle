#!/usr/bin/env python

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statistics as st
import scipy as sp
import time
from random import randrange
import re
import sys
import scanpy as sc

from granatum_sdk import Granatum


def parse(st):
    return list(map(lambda s: s.strip(), list(filter(lambda s: s != "", st.split(',')))))


def configure_plotting(dpi=100, fontsize=10, dpi_save=300, figsize=(5,4)):
    sc.settings.set_figure_params(dpi=dpi, fontsize=fontsize, dpi_save=dpi_save, figsize=figsize, format='png')


# Create a dataframe from an ann
def ann_to_df(adata):
    if sp.sparse.issparse(adata.X):
        return pd.DataFrame.sparse.from_spmatrix(data=adata.X, index=adata.obs_names.to_list(), columns=adata.var_names.to_list())
    return pd.DataFrame(data=adata.X, index=adata.obs_names.to_list(), columns=adata.var_names.to_list())

def main():
    tic = time.perf_counter()

    gn = Granatum()

    # We assume the filtering has been applied before entry
    # We are producing a cell cycle phase label from the expression matrix
    assay = gn.pandas_from_assay(gn.get_import('assay')).T
    adata = sc.AnnData(assay, dtype=np.float64)

    configure_plotting()

    sgenes = [x.strip() for x in open('./data/sgenes.txt')]
    g2mgenes = [x.strip() for x in open('./data/g2mgenes.txt')]
    cellcyclegenes = list(set(sgenes + g2mgenes).intersection(set(adata.var_names)))
    
    sc.pp.scale(adata)
    sc.tl.score_genes_cell_cycle(adata, s_genes=sgenes, g2m_genes=g2mgenes)
 
    adata_cc_genes = adata[:, cellcyclegenes]
    sc.tl.pca(adata_cc_genes)
    sc.pl.pca_scatter(adata_cc_genes, color='phase')
    plt.legend(loc="upper center")
    gn.add_current_figure_to_results('PCA Scatter Plot of Cell Cycle Phases')

    reporting_df = pd.concat([adata.obs["phase"], ann_to_df(adata_cc_genes)], axis=1)
    gn.add_pandas_df(reporting_df.reset_index())

    gn.export_statically(dict(zip(adata.obs["phase"].index, adata.obs["phase"].tolist())), "Cell cycle phase labels")
    gn.export(adata.obs["phase"].to_csv(), "CellCyclePhase.csv", kind='raw', meta=None, raw=True)
    gn.export(reporting_df.to_csv(), "CellCycleGeneMatrix.csv", kind='raw', meta=None, raw=True)
    

    # Append timing information
    toc = time.perf_counter()
    time_passed = round(toc - tic, 2)
    gn.add_result("* Finished in {} seconds*".format(time_passed), "markdown")

    gn.commit()


if __name__ == "__main__":
    main()
