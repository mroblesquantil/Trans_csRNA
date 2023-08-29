# Librerías 

import pickle
import sys
from glob import glob

import h5py
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.metrics import (calinski_harabasz_score, davies_bouldin_score,
                             silhouette_score)


def log_1(x):
    return np.log10(x+1)

# Lectura de datos 
def leer_modelos(path: str)-> dict:

    data = {}
    datasets = [f.split('\\')[-1] for f in glob(path + '/*')]

    for d in datasets:
        predictions = {}
        fn = glob(path + f'/{d}/predictions_*.pkl')

        for f in fn:
            with open(f, "rb") as file: prediction = pickle.load(file)
            model_name = f.split('.')[-2].split('predictions_')[1]
            predictions[model_name] = prediction
        
        data[d] = predictions

    return data

def leer_labels_reales() -> dict:
    raw_data = {}

    data10x = h5py.File('../Tests-GMM/data/Small_Datasets/' + '10X_PBMC_select_2100_top2000.h5')
    X  = np.array(list(map(log_1, np.array(data10x['X']))))
    adata = sc.AnnData(X)
    sc.pp.normalize_total(adata)
    X = adata.X
    raw_data['10PBMC'] = {'X': X, 'y': np.array(data10x['Y'])}

    dataLiver = h5py.File('../Tests-GMM/data/Small_Datasets/' + 'HumanLiver_counts_top5000.h5')
    X  = np.array(list(map(log_1, np.array(dataLiver['X']))))
    adata = sc.AnnData(X)
    sc.pp.normalize_total(adata)
    X = adata.X
    raw_data['HumanLiver'] = {'X':X, 'y': np.array(dataLiver['Y'])}

    dataMacosko = h5py.File('../Tests-GMM/data/Small_Datasets/' + 'Macosko_mouse_retina.h5')
    X  = np.array(list(map(log_1, np.array(dataMacosko['X']))))
    adata = sc.AnnData(X)
    sc.pp.normalize_total(adata)
    X = adata.X
    raw_data['Macosko'] = {'X': X, 'y': np.array(dataMacosko['Y'])}

    return raw_data

# Cálculo de métricas
def calcular_metricas(real_labels, predictions):
    
    rta = pd.DataFrame(columns = ['dataset', 'model', 'SS', 'DBS', 'CHS'])
    for name, dict in real_labels.items():
        X = dict['X']

        for pred_name, pred in predictions[name].items():
            print('-->', name, pred_name)
            ss = silhouette_score(X, pred)
            dbs = davies_bouldin_score (X, pred)
            chs = calinski_harabasz_score (X, pred)

            rta.loc[name + '_' + pred_name] = [name, pred_name, ss, dbs,chs]
    
    return rta

# Main 
def main():
    carpeta_lectura = sys.argv[1]
    carpeta_guardar = sys.argv[2]

    predictions = leer_modelos(carpeta_lectura)
    real_labels = leer_labels_reales()

    metrics = calcular_metricas(real_labels, predictions)
    name = carpeta_guardar + '/resultados_modelos_procesado.csv'
    metrics.to_csv(name, index = None)

    print(f'Se guardó correctamente el resultado en {name}')


if __name__ == "__main__":
    main()