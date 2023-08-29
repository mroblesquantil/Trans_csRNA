import os
import pickle
import sys

import h5py
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import seaborn as sns
from matplotlib.gridspec import SubplotSpec
from scipy.spatial.distance import euclidean
from scipy.stats import pearsonr, spearmanr
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from tqdm import tqdm

##################################################################################
###################################### Funciones auxiliares
##################################################################################

def log_1(x):
    return np.log10(x+1)

def get_distances_matrix(X: np.array, tipo: str = 'similitud') -> np.array:
    """
    Computes distance matrix between pair of points.

    input:
    - X: np.array with coordinates of each sample point
    - tipo: str representing the kind of operation to perform in each pair of cells. 

    output: 
    - distancias: np.array with pairwise distance between points
    """
    num_cells = X.shape[0]
    distancias = np.zeros((num_cells, num_cells))

    for c1 in tqdm(range(num_cells)):
        celula1 = X[c1,:]
        indices_nocero1 = celula1.nonzero()[1]
        for c2 in range(c1, num_cells):
            celula2 = X[c2,:]
            indices_nocero2 = celula2.nonzero()[1]

            if tipo == 'similitud':
                indices_comunes = set(indices_nocero1).intersection(set(indices_nocero2))
                distancias[c1][c2] = len(indices_comunes)
                distancias[c2][c1] = len(indices_comunes)

            elif tipo == 'euclidea':
                dist = euclidean(np.array(celula1.todense())[0,:], np.array(celula2.todense())[0,:])
                distancias[c1][c2] = dist
                distancias[c2][c1] = dist

            elif tipo == 'correlacion_pearson':
                indices_comunes = list(set(indices_nocero1).intersection(set(indices_nocero2)))
                x1 = np.array(celula1.todense())[0,indices_comunes]
                x2 = np.array(celula2.todense())[0,indices_comunes]
                corr = pearsonr(x1, x2)
                distancias[c1][c2] = corr.statistic
                distancias[c2][c1] = corr.statistic

            elif tipo == 'rango_spearman':
                indices_comunes = list(set(indices_nocero1).intersection(set(indices_nocero2)))
                x1 = np.array(celula1.todense())[0,indices_comunes]
                x2 = np.array(celula2.todense())[0,indices_comunes]
                corr = spearmanr(x1, x2)
                distancias[c1][c2] = corr.correlation 
                distancias[c2][c1] = corr.correlation 
            
    return distancias

##################################################################################
###################################### Lectura de datos
##################################################################################

def leer_datos(path: str) -> sc.AnnData:
    """
    Reads the .h5 data from path

    input: 
    - path: str

    output:
    - adata: AnnData
    """
    data = h5py.File(path)
    x = np.array(data['X'])

    adata = sc.AnnData(x)

    return adata


def crear_grafico_conteos(adata: sc.AnnData, path_save: str) -> plt.Figure:
    """
    Creates a count graph

    input:
    - adata: AnnData with single cell counts
    - path_save: str where the graph will be saved

    output:
    - p: Figure
    """
    sc.pp.calculate_qc_metrics(adata, inplace=True)

    p = sns.jointplot(
        data=adata.obs,
        x="log1p_total_counts",
        y="log1p_n_genes_by_counts",
        kind="hex",
        xlim = (4,11), ylim = (3.5,9)
    )
    p.fig.suptitle('Conteos')
    p.savefig(path_save)

    return p

def normalizar_filtrar(adata: sc.AnnData)-> sc.AnnData:
    """
    Selects only the 10% more represented cells after applying a log transformation to counts.

    input:
    - adata: AnnData with count information

    output:
    - adata: AnnData with log transformed counts and with filtered cells
    """
    adata.X = np.array(list(map(log_1, adata.X)))

    quantile = pd.qcut(adata.obs['total_counts'], 10, labels = False) 
    adata = adata[quantile == 9]

    print(f'Tamaño final dataset: {adata.shape}')

    return adata

def compute_simmilarity_matrices(adata: sc.AnnData) -> dict:
    """
    Computed similarity matrices for similarity (number of shared genes), euclidean distance, spearman and pearson correlation

    input:
    - adata: AnnData filtered with count information

    output:
    - similitud: Dictionary with a matrix for each similarity function 
    """
    mat = sp.csr_matrix(adata.X)
    similitud = {}

    for kind in ['similitud', 'euclidea', 'correlacion_pearson', 'rango_spearman']:
        print('\n--->', kind)
        similitud[kind] = get_distances_matrix(mat, tipo = kind)

    return similitud

def plot_similarity_matrices(similitud: dict, path_save: str)-> plt.Figure:
    """
    Plots similarity matrices

    input:
    - similitud: Dictionary with a matrix for each similarity function 
    - path_save: str direction to save the plot

    output:
    - p: Figure with heatmaps
    """
    p = sns.color_palette("ch:s=-.2,r=.6", as_cmap=True)

    fig, axes = plt.subplots(nrows = 4, figsize = (10,20))

    for ax, (key, distancias) in zip(axes,similitud.items()):
        sns.heatmap(distancias, cmap=p, ax = ax).set(title = key)

    fig.savefig(path_save)

    return fig


def main():
    archivo = sys.argv[1]
    carpeta_guardar = sys.argv[2]

    adata = leer_datos(archivo)
    crear_grafico_conteos(adata, carpeta_guardar + '/distribucion_conteos.png')

    adata = normalizar_filtrar(adata)

    similitud = compute_simmilarity_matrices(adata)

    plot_similarity_matrices(similitud, carpeta_guardar + '/heatmaps.png')

if __name__ == "__main__":
    main()