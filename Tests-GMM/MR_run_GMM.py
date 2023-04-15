import  os
import h5py
import torch
import pickle
import argparse
import numpy as np
import scanpy as sc
import pandas as pd 
from time import time
from sklearn import metrics
import torch.nn.functional as F

from MR_GMM import scDCC
from utils import cluster_acc
from preprocess import read_dataset, normalize


if __name__ == "__main__":

    # setting the hyper parameters
    parser = argparse.ArgumentParser(description='train',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--n_clusters', default=8, type=int)
    parser.add_argument('--label_cells', default=0.1, type=float)
    parser.add_argument('--label_cells_files', default='label_selected_cells_1.txt')
    parser.add_argument('--n_pairwise', default=0, type=int)
    parser.add_argument('--n_pairwise_error', default=0, type=float)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--data_file', default='data/Small_Datasets/10X_PBMC_select_2100.h5')
    parser.add_argument('--maxiter', default=200, type=int) ############## 200
    parser.add_argument('--pretrain_epochs', default=1, type=int) ############# 300
    parser.add_argument('--gamma', default=1., type=float,
                        help='coefficient of clustering loss')
    parser.add_argument('--ml_weight', default=1., type=float,
                        help='coefficient of must-link loss')
    parser.add_argument('--cl_weight', default=1., type=float,
                        help='coefficient of cannot-link loss')
    parser.add_argument('--update_interval', default=1, type=int)
    parser.add_argument('--tol', default=0.001, type=float)
    parser.add_argument('--ae_weights', default=None)
    parser.add_argument('--save_dir', default='results/scDCC_p0_1/')
    parser.add_argument('--ae_weight_file', default='AE_weights_p0_1.pth.tar')

    # Argumentos de GMM
    cov_identidad = False # Se mantiene la covarianza como la identidad en todos los casos
    parser.add_argument('--cov_identidad', default = cov_identidad)
    if cov_identidad:
        parser.add_argument('--path_results', default='results_MR_COVIdentidad/')
    else:
        parser.add_argument('--path_results', default='results_MR_COVDiagonal/') #  default='results_MR_COVTry/')

    args = parser.parse_args()

    # Lectura de datos
    data_mat = h5py.File(args.data_file)
    x = np.array(data_mat['X'])
    y = np.array(data_mat['Y'])
    data_mat.close()

    # preprocesamiento scRNA-seq read counts matrix
    adata = sc.AnnData(x)
    adata.obs['Group'] = y

    adata = read_dataset(adata,
                     transpose=False,
                     test_split=False,
                     copy=True)

    adata = normalize(adata,
                      size_factors=True,
                      normalize_input=True,
                      logtrans_input=True)

    input_size = adata.n_vars

    print(args)

    print(adata.X.shape)
    print(y.shape)

    if not os.path.exists(args.label_cells_files):
        indx = np.arange(len(y))
        np.random.shuffle(indx)
        label_cell_indx = indx[0:int(np.ceil(args.label_cells*len(y)))]
    else:
        label_cell_indx = np.loadtxt(args.label_cells_files, dtype=np.int)

    x_sd = adata.X.std(0)
    x_sd_median = np.median(x_sd)
    print("median of gene sd: %.5f" % x_sd_median)
    sd = 2.5

    # Creación del modelo (sin el valor final de y, sin most links)
    model = scDCC(input_dim=adata.n_vars, z_dim=32, n_clusters=args.n_clusters, 
                encodeLayer=[256, 64], decodeLayer=[64, 256], sigma=sd, gamma=args.gamma,
                cov_identidad = args.cov_identidad, path = args.path_results) 
    
    print(str(model))

    # Entrenar el autoencoder (ZINB Loss)
    t0 = time()
    if args.ae_weights is None:
        model.pretrain_autoencoder(x=adata.X, X_raw=adata.raw.X, size_factor=adata.obs.size_factors, 
                                batch_size=args.batch_size, epochs=args.pretrain_epochs, ae_weights=args.ae_weight_file)
    else:
        if os.path.isfile(args.ae_weights):
            print("==> loading checkpoint '{}'".format(args.ae_weights))
            checkpoint = torch.load(args.ae_weights)
            model.load_state_dict(checkpoint['ae_state_dict'])
        else:
            print("==> no checkpoint found at '{}'".format(args.ae_weights))
            raise ValueError
    
    print('Pretraining time: %d seconds.' % int(time() - t0))

    if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)

    # Segundo entrenamiento: clustering loss + ZINB loss
    y_pred,  mu, pi, cov, z, epochs = model.fit(X=adata.X, X_raw=adata.raw.X, sf=adata.obs.size_factors,  
                                    batch_size=args.batch_size,  num_epochs=args.maxiter,
                                    update_interval=args.update_interval, tol=args.tol, save_dir=args.save_dir, lr = 0.001)
    
    # Se guardan los resultados
    pd.DataFrame(z.detach().numpy()).to_csv(args.path_results + 'Z.csv')
    pd.DataFrame(mu.detach().numpy()).to_csv(args.path_results + 'Mu.csv')
    pd.DataFrame(pi.detach().numpy()).to_csv(args.path_results + 'Pi.csv')
    pd.DataFrame(cov).to_csv(args.path_results + 'Cov.csv')

    print("Se guardan los resultados de las predicciones")
    
    #############################################################
    print("------------- SE GUARDAN LOS DATOS GMM")
    with open(args.path_results + '/prediccion.pickle', 'wb') as handle:
        pickle.dump(y_pred, handle)

    #############################################################
    print('Total time: %d seconds.' % int(time() - t0))
    
    # Evaluación final de resultados: métricas comparando con los clusters reales
    eval_cell_y_pred = np.delete(y_pred, label_cell_indx)
    eval_cell_y = np.delete(y, label_cell_indx)
    acc = np.round(cluster_acc(eval_cell_y, eval_cell_y_pred), 5)
    nmi = np.round(metrics.normalized_mutual_info_score(eval_cell_y, eval_cell_y_pred), 5)
    ari = np.round(metrics.adjusted_rand_score(eval_cell_y, eval_cell_y_pred), 5)
    print('Evaluating cells: ACC= %.4f, NMI= %.4f, ARI= %.4f' % (acc, nmi, ari))

    if not os.path.exists(args.label_cells_files):
        np.savetxt(args.label_cells_files, label_cell_indx, fmt="%i")