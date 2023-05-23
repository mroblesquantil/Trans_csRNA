import torch
import pickle
import math, os
import numpy as np
import torch.nn as nn
from sklearn import metrics
import torch.optim as optim
from torch.nn import Parameter
import torch.nn.functional as F
from sklearn.cluster import KMeans
from torch.autograd import Variable
from scipy.stats import multivariate_normal
from layers import ZINBLoss, ClusteringLoss, MeanAct, DispAct
from torch.utils.data import DataLoader, TensorDataset

from utils import * 

def buildNetwork(layers, type, activation="relu"):
    net = []
    for i in range(1, len(layers)):
        net.append(nn.Linear(layers[i-1], layers[i]))
        if activation=="relu":
            net.append(nn.ReLU())
        elif activation=="sigmoid":
            net.append(nn.Sigmoid())
    return nn.Sequential(*net)


class scDCC(nn.Module):
    def __init__(self, input_dim, z_dim, n_clusters, path, encodeLayer=[], decodeLayer=[], 
            activation="relu", sigma=1., alpha=1., gamma=1., ml_weight=1., cl_weight=1.,
            cov_identidad = True):
        super(scDCC, self).__init__()
        
        # Inicialización de valores para el autoencoder
        self.z_dim = z_dim
        self.n_clusters = n_clusters
        self.activation = activation
        self.sigma = sigma
        self.alpha = alpha
        self.gamma = gamma
        self.ml_weight = ml_weight
        self.cl_weight = cl_weight
        self.encoder = buildNetwork([input_dim]+encodeLayer, type="encode", activation=activation)
        self.decoder = buildNetwork([z_dim]+decodeLayer, type="decode", activation=activation)
        self._enc_mu = nn.Linear(encodeLayer[-1], z_dim)
        self._dec_mean = nn.Sequential(nn.Linear(decodeLayer[-1], input_dim), MeanAct())
        self._dec_disp = nn.Sequential(nn.Linear(decodeLayer[-1], input_dim), DispAct())
        self._dec_pi = nn.Sequential(nn.Linear(decodeLayer[-1], input_dim), nn.Sigmoid())
        
        # Inicialización de parámetros para el clustering
        self.mu = Parameter(torch.rand(n_clusters, z_dim, dtype=torch.float32))
        self.pi = Parameter(torch.rand(n_clusters, 1, dtype=torch.float32))
        self.diag_cov = Parameter(torch.rand(n_clusters, z_dim, dtype=torch.float32))

        # Funciones auxiliares: cálculo del ZINB loss, Softmax y Clustering Loss
        self.zinb_loss = ZINBLoss()
        self.softmax = nn.Softmax(dim=1)
        self.clustering_loss = ClusteringLoss()

        # Se guardan las covarianzas
        self.cov = torch.Tensor([np.identity(self.z_dim)]*self.n_clusters)
        self.cov_identidad = cov_identidad

        # Directorio en donde se guardan los resultados
        self.path = path 

    
    def forward(self, x):
        h = self.encoder(x+torch.randn_like(x) * self.sigma)
        z = self._enc_mu(h)
        h = self.decoder(z)
        _mean = self._dec_mean(h)
        _disp = self._dec_disp(h)
        _pi = self._dec_pi(h)

        h0 = self.encoder(x)
        z0 = self._enc_mu(h0)

        prob_matrix = self.find_probabilities(z0)
        return z0, _mean, _disp, _pi, prob_matrix
    
    def encodeBatch(self, X, batch_size=256):
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            self.cuda()
        
        encoded = []
        num = X.shape[0]
        num_batch = int(math.ceil(1.0*X.shape[0]/batch_size))
        for batch_idx in range(num_batch):
            xbatch = X[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)]
            inputs = Variable(xbatch)
            z, _, _, _, _ = self.forward(inputs)
            encoded.append(z.data)

        encoded = torch.cat(encoded, dim=0)
        return encoded

    def find_probabilities(self, Z):
        """
        Encuentra las probabilidades de cada punto a cada cluster a partir de las medias y las covarianzas.
        """
        
        try: proba = torch.exp(torch.distributions.MultivariateNormal(
                    self.mu, self.cov).log_prob(Z.unsqueeze(1)))
        except: breakpoint()
                
        # Normalizamos
        proba = torch.div(proba,proba.sum(1).unsqueeze(-1))   
        
        # Multiplicamos por pi
        proba = torch.multiply(proba, nn.Softmax(dim=0)(self.pi).squeeze(1))

        # Normalizamos 
        proba = torch.where(proba < 0.00001, 0.00001, proba.double())


        return proba
    
    def find_probabilities_identity(self, Z):
        """
        Encuentra las probabilidades de cada punto a cada cluster a partir de las medias y las covarianzas.
        """
        
        cov =  torch.Tensor([np.identity(32)]*8)
        try: proba = torch.exp(torch.distributions.MultivariateNormal(self.mu, 
                            cov).log_prob(Z.unsqueeze(1)))
        except: breakpoint()
                
        # Normalizamos
        proba = torch.div(proba,proba.sum(1).unsqueeze(-1))   
        
        # Multiplicamos por pi
        proba = torch.multiply(proba, nn.Softmax(dim=0)(self.pi).squeeze(1))

        # Normalizamos 
        proba = torch.where(proba < 0.00001, 0.00001, proba.double())

        return proba


    def find_covariance(self, Z, mu, phi):
        """"
        Args:
            phi: Matriz (n_puntos x n_clusters) donde phi[i,k] representa la probabilidad de que el punto i esté en el cluster k.
            X: Matriz (n_puntos x d) con los puntos
            mu: Matriz (n_clusters x d) con las medias de cada cluster
        Returns:
            cov_mats: Lista (n_clusters) con una matriz de covarianza por cada cluster 
        """
        n_clus = self.n_clusters
        Z = Z.detach().numpy()
        mu = mu.detach().numpy()

        cov_mats = []
        for k in range(n_clus):
            nk = np.sum(phi[:,k])

            vects = []
            for i in range(self.z_dim):
                r =  np.matrix(Z[i,:] - mu[k,:])
                v = phi[i,k]*np.matmul( r.transpose(), r )
                vects.append(v)
            
            m = 1/nk*np.sum(vects, axis = 0)
            if nk == 0: m =  np.identity(self.z_dim)
            cov_mats.append(m)
        
        return cov_mats

    def save_checkpoint(self, state, index, filename):
        newfilename = os.path.join(filename, 'FTcheckpoint_%d.pth.tar' % index)
        torch.save(state, newfilename)

    def pretrain_autoencoder(self, x, X_raw, size_factor, batch_size=256, lr=0.0001, epochs=400, ae_save=True, ae_weights='AE_weights.pth.tar'):
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            self.cuda()
        dataset = TensorDataset(torch.Tensor(x), torch.Tensor(X_raw), torch.Tensor(size_factor))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        print("Pretraining stage")
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=lr, amsgrad=True)
        for epoch in range(epochs):
            for batch_idx, (x_batch, x_raw_batch, sf_batch) in enumerate(dataloader):
                x_tensor = Variable(x_batch)#.cuda()
                x_raw_tensor = Variable(x_raw_batch)#.cuda()
                sf_tensor = Variable(sf_batch)#.cuda()
                _, mean_tensor, disp_tensor, pi_tensor, prob_matrix = self.forward(x_tensor)
                loss = self.zinb_loss(x=x_raw_tensor, mean=mean_tensor, disp=disp_tensor, pi=pi_tensor, scale_factor=sf_tensor)
                
                #temp = self.mu.detach().numpy().copy()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                print('Pretrain epoch [{}/{}], ZINB loss:{:.4f}'.format(batch_idx+1, epoch+1, loss.item()))
        
        if ae_save:
            torch.save({'ae_state_dict': self.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()}, ae_weights)

    def save_checkpoint(self, state, index, filename):
        newfilename = os.path.join(filename, 'FTcheckpoint_%d.pth.tar' % index)
        torch.save(state, newfilename)

    def fit(self, X, X_raw, sf, lr=0.1, batch_size=256, num_epochs=10, update_interval=1, tol=1e-3, save_dir='', y = None):
        '''X: tensor data'''
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            self.cuda()

        save_dir = self.path +'/'
        print("Clustering stage")
        X = torch.tensor(X)#.cuda()
        X_raw = torch.tensor(X_raw)#.cuda()
        sf = torch.tensor(sf)#.cuda()
        
        optimizer = optim.Adadelta(filter(lambda p: p.requires_grad, self.parameters()), lr=lr, rho=.95)

        print("Initializing cluster centers with kmeans.")
        kmeans = KMeans(self.n_clusters, n_init=20)
        data = self.encodeBatch(X)

        print("------------- SE GUARDAN LOS DATOS ANTES DE KMEANS (datos pre-train)")
        with open(self.path + '/DATOS_ANTES_KMEANS.pickle', 'wb') as handle:
            pickle.dump( data.data.cpu().numpy(), handle)

        self.y_pred = kmeans.fit_predict(data.data.cpu().numpy())
        self.y_pred_last = self.y_pred

        self.mu.data.copy_(torch.Tensor(kmeans.cluster_centers_))
        
        self.train()
        num = X.shape[0]
        num_batch = int(math.ceil(1.0*X.shape[0]/batch_size))

        clustering_metrics = {'ac': [], 'nmi': [], 'ari': []}
        clustering_metrics_id = {'ac': [], 'nmi': [], 'ari': []}
        losses = {'zinb': [], 'gmm': []}

        for epoch in range(num_epochs):
            print(f"---> Epoca {epoch}")

            if epoch%update_interval == 0:
                latent = self.encodeBatch(X)
                
                z = self.encodeBatch(X)        

                    
                if self.cov_identidad == False:
                    diag = torch.where(self.diag_cov.double() <= 0, 1/2100, self.diag_cov.double())
                    x = [torch.diag(diag.detach()[i]) for i in range(self.n_clusters)]
                    self.cov = torch.stack(x)

                distr = self.find_probabilities(z)
                self.y_pred = torch.argmax(torch.tensor(distr), dim=1).data.cpu().numpy()

                # check stop criterion
                delta_label = np.sum(self.y_pred != self.y_pred_last).astype(np.float32) / num
                
                # save current model
                if (epoch>0 and delta_label < tol) or epoch%10 == 0:
                    self.save_checkpoint({'epoch': epoch+1,
                            'state_dict': self.state_dict(),
                            'mu': self.mu,
                            'y_pred': self.y_pred,
                            'z': z,
                            'pi': self.pi,
                            'cov': self.cov,
                            }, epoch+1, filename=save_dir)
                    
                self.y_pred_last = self.y_pred
                # if epoch>0 and delta_label < tol:
                    
                #     with open(f'{self.path}/DATOS_DESPUES_KMEANS{epoch}.pickle', 'wb') as handle:
                #         pickle.dump( latent, handle)

                #     print('delta_label ', delta_label, '< tol ', tol)
                #     print("Reach tolerance threshold. Stopping training.")
                #     break
            
            cluster_loss_val = 0
            recon_loss_val = 0
            train_loss = 0
            # train 1 epoch for clustering loss
            for batch_idx in range(num_batch):
                xbatch = X[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)]
                xrawbatch = X_raw[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)]
                sfbatch = sf[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)]

                inputs = Variable(xbatch)
                rawinputs = Variable(xrawbatch)
                sfinputs = Variable(sfbatch)

                z, meanbatch, dispbatch, pibatch, prob_matrixbatch = self.forward(inputs) 


                if self.cov_identidad == False:
                    diag = torch.where(self.diag_cov.double() <= 0, 1/2100, self.diag_cov.double())
                    self.cov = torch.stack([torch.diag(diag.detach()[i]) for i in range(self.n_clusters)]) 

                cluster_loss = self.clustering_loss(prob_matrixbatch)
                recon_loss = self.zinb_loss(rawinputs, meanbatch, dispbatch, pibatch, sfinputs)
                loss = cluster_loss + recon_loss
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                cluster_loss_val += cluster_loss * len(inputs)
                recon_loss_val += recon_loss * len(inputs)
                train_loss = cluster_loss_val + recon_loss_val
                
                
            print("#Epoch %3d: Total: %.4f Clustering Loss: %.9f ZINB Loss: %.4f" % (
                epoch + 1, train_loss / num, cluster_loss_val / num, recon_loss_val / num))      

            losses['zinb'].append(recon_loss_val / num)
            losses['gmm'].append(cluster_loss_val / num)

            
            if epoch == num_epochs - 1: 
                with open(f'{self.path}/DATOS_DESPUES_KMEANS{epoch}.pickle', 'wb') as handle:
                    pickle.dump( latent, handle)

            if not y is None:
                z = self.encodeBatch(X)        
                distr = self.find_probabilities(z)
                self.y_pred = torch.argmax(torch.tensor(distr), dim=1).data.cpu().numpy()

                accuracy = np.round(cluster_acc(y_true = y, y_pred = self.y_pred), 5)
                nmi = np.round(metrics.normalized_mutual_info_score(y, self.y_pred), 5)
                ari = np.round(metrics.adjusted_rand_score(y, self.y_pred), 5)

                clustering_metrics['ac'].append(accuracy)
                clustering_metrics['nmi'].append(nmi)
                clustering_metrics['ari'].append(ari)

                distr = self.find_probabilities_identity(z)
                y_pred_identity = torch.argmax(torch.tensor(distr), dim=1).data.cpu().numpy()

                accuracy = np.round(cluster_acc(y_true = y, y_pred = y_pred_identity), 5)
                nmi = np.round(metrics.normalized_mutual_info_score(y, y_pred_identity), 5)
                ari = np.round(metrics.adjusted_rand_score(y, y_pred_identity), 5)

                clustering_metrics_id['ac'].append(accuracy)
                clustering_metrics_id['nmi'].append(nmi)
                clustering_metrics_id['ari'].append(ari)
        
            with open(self.path + '/clustering_metrics.pickle', 'wb') as handle:
                pickle.dump( clustering_metrics, handle)

            with open(self.path + '/clustering_metrics_id.pickle', 'wb') as handle:
                pickle.dump( clustering_metrics_id, handle)

            with open(self.path + '/losses.pickle', 'wb') as handle:
                pickle.dump( losses, handle)

        return self.y_pred, self.mu, self.pi, self.cov, z, epoch, clustering_metrics, clustering_metrics_id, losses
