import h5py
import networkx as nx
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, Birch, KMeans
from sklearn.datasets import make_blobs 
from sklearn.model_selection import GridSearchCV
from scipy.spatial.distance import euclidean
from sklearn.neighbors import kneighbors_graph
from scipy.optimize import linear_sum_assignment as linear_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size

    y_true = y_true - 1

    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_assignment(w.max() - w)

    w_order = np.zeros((D, D), dtype=np.int64)
    for i in range(D):
        for j in range(D):
            w_order[i,j] = w[i, ind[1][j]]

    return sum([w[i, j] for i, j in zip(ind[0], ind[1])]) * 1.0 / y_pred.size


def model_DBSCAN(X, y, 
                 eps_possible = np.linspace(0.01,100,100), 
                 min_samples_possible = [4*i for i in range(1,10)]):
    max_acc = 0
    max_acc_e_m = 0,0
    best_model = None 

    for e in eps_possible:
        for m in min_samples_possible:
            dbscan = DBSCAN(eps=e, min_samples=m)

            # Realizar el clustering
            labels = dbscan.fit_predict(X)
            acc = cluster_acc(y, labels)
            
            if acc > max_acc: 
                max_acc = acc
                max_acc_e_m = e, m
                best_model = dbscan
    
    ari = adjusted_rand_score(y, best_model.predict(X))
    nmi = normalized_mutual_info_score(y, best_model.predict(X)) 
    
    metrics = {'Accuracy': max_acc, 'ARI': ari, 'NMI': nmi}

    return best_model, max_acc_e_m[0], max_acc_e_m[1], metrics


def model_BIRCH(X, y, 
                threshold: [0.1, 0.5, 1.0, 2.0, 5.0, 10.0], 
                branching_factor: [10, 50, 100, 150]
                ):
    # Define los hiperparámetros a ajustar
    param_grid = {
        'threshold': threshold,
        'branching_factor': branching_factor
    }

    max_acc = 0
    params = 0, 0 
    best_model = None 
    for t in param_grid['threshold']:
        for b in param_grid['branching_factor']:
            birch_model = Birch(n_clusters=8, threshold = t, branching_factor = b)
            birch_model.fit(X)

            labels = birch_model.predict(X)

            acc = cluster_acc(y, labels)
            if acc > max_acc:
                max_acc = acc
                params = t, b 
                best_model = birch_model

    ari = adjusted_rand_score(y, best_model.predict(X))
    nmi = normalized_mutual_info_score(y, best_model.predict(X)) 
    
    metrics = {'Accuracy': max_acc, 'ARI': ari, 'NMI': nmi}

    return best_model, params[0], params[1], metrics 


def model_KMeans(X, y):
    n_clusters = len(set(y))
    
    kmeans_model = KMeans(n_clusters=n_clusters)
    kmeans_model.fit(X)
    labels = kmeans_model.predict(X)

    acc = cluster_acc(y, labels)
    ari = adjusted_rand_score(y, labels)
    nmi = normalized_mutual_info_score(y, labels) 
    
    metrics = {'Accuracy': acc, 'ARI': ari, 'NMI': nmi}

    return kmeans_model, metrics