import pickle
import warnings

import h5py
import numpy as np
from scipy.optimize import linear_sum_assignment as linear_assignment
from sklearn.cluster import DBSCAN, Birch, KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

warnings.filterwarnings('ignore')



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

def model_BIRCH(X, y, 
                threshold = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0], 
                branching_factor =  [10, 50, 100, 150]
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

# BIRCH 
def run_birch(raw_data, path_resultados):
    """
    Ejecuta Birch
    """
    for d, data in raw_data.items():
        y = data['y']
        z = data['X']

        model, _, _, metrics = model_BIRCH(z, y)
        acc, nmi, ari = metrics['Accuracy'], metrics['NMI'], metrics['ARI']
        print(f'\nModelo {d} ---- ACC {acc}. NMI {nmi}. ARI {ari}.')

        name_model = path_resultados + d + "/model_completedata_birch.pkl"   
        name_predictions = path_resultados + d + "/predictions_completedata_birch.pkl"
        
        with open(name_model, "wb") as f:
            pickle.dump(model, f)
        with open(name_predictions, "wb") as f:
            pickle.dump(model.predict(z), f)

# KMEANS 
def run_kmeans(raw_data, path_resultados):
    """
    Ejecuta Birch
    """
    for d, data in raw_data.items():
        y = data['y']
        z = data['X']

        model, metrics = model_KMeans(z, y)
        acc, nmi, ari = metrics['Accuracy'], metrics['NMI'], metrics['ARI']
        print(f'\nModelo {d} ---- ACC {acc}. NMI {nmi}. ARI {ari}.')

        name_model = path_resultados + d + "/model_completedata_kmeans.pkl"   
        name_predictions = path_resultados + d + "/predictions_completedata_kmeans.pkl"
        
        with open(name_model, "wb") as f:
            pickle.dump(model, f)
        with open(name_predictions, "wb") as f:
            pickle.dump(model.predict(z), f)


def main():
    raw_data = {}

    data10x = h5py.File('../Tests-GMM/data/Small_Datasets/' + '10X_PBMC_select_2100_top2000.h5')
    raw_data['10PBMC'] = {'X': np.array(data10x['X']), 'y': np.array(data10x['Y'])}

    dataLiver = h5py.File('../Tests-GMM/data/Small_Datasets/' + 'HumanLiver_counts_top5000.h5')
    raw_data['HumanLiver'] = {'X': np.array(dataLiver['X']), 'y': np.array(dataLiver['Y'])}

    dataMacosko = h5py.File('../Tests-GMM/data/Small_Datasets/' + 'Macosko_mouse_retina.h5')
    raw_data['Macosko'] = {'X': np.array(dataMacosko['X']), 'y': np.array(dataMacosko['Y'])}

    run_kmeans(raw_data, "results_models/")
    run_birch(raw_data, "results_models/")

if __name__ == "__main__":
    main()

