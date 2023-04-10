import torch
import pickle
import math, os
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.nn import Parameter
import torch.nn.functional as F
from sklearn.cluster import KMeans
from torch.autograd import Variable
from scipy.stats import multivariate_normal
from layers import ZINBLoss, MeanAct, DispAct
from torch.utils.data import DataLoader, TensorDataset

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
        self.mu = Parameter(torch.Tensor(n_clusters, z_dim))
        self.pi = Parameter(torch.Tensor(n_clusters, 1))

        # Los parámetros del clustering no se se actualizan en el pre train
        self.pi.requires_grad = False 
        self.mu.requires_grad = False 

        # Funciones auxiliares: cálculo del ZINB loss y Softmax
        self.zinb_loss = ZINBLoss()
        self.softmax = nn.Softmax(dim=1)

        # Se guardan las covarianzas
        self.cov = [np.identity(self.z_dim)]*self.n_clusters
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
        return z0, _mean, _disp, _pi 
    
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
            z, _, _, _ = self.forward(inputs)
            encoded.append(z.data)

        encoded = torch.cat(encoded, dim=0)
        return encoded

    def find_probabilities(self, Z, mu, cov):
        """
        Encuentra las probabilidades de cada punto a cada cluster a partir de las medias y las covarianzas.
        """
        n_clus = self.n_clusters
        Z = Z.detach().numpy()
        mu = mu.detach().numpy()
        
        probab = []
        # Itero sobre los puntos
        for i in range(Z.shape[0]):
            Z_i = Z[i,:]
            probab_i = []

            # Itero sobre los cluster
            for k in range(n_clus):
                mean_vec = mu[k,:]
                cov_cluster = cov[k]

                try: pdf = multivariate_normal(mean_vec, cov_cluster, allow_singular = True).pdf(Z_i)
                except: pdf = multivariate_normal(mean_vec, np.identity(self.z_dim), allow_singular = True).pdf(Z_i)
                probab_i.append(pdf)

            probab.append(probab_i)
        
        return probab


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
    
    def clustering_GMM_loss(self, Z, pi, mu, cov):
        """
        Args: 
            X: Matriz (n_puntos x d) con los puntos
            pi: Vector (n_clusters) con los pesos de cada cluster
            mu: Matriz (n_clusters x d) con las medias de cada cluster
            cov_mats: Lista (n_clusters) con una matriz de covarianza por cada cluster 
        Returns:
            GMM loss
        """
        Z = Z.detach().numpy()
        pi = pi.detach().numpy()
        mu = mu.detach().numpy()

        n_clus = self.n_clusters

        # Itero sobre los puntos
        res = []
        for i in range(Z.shape[0]):
            # Itero sobre los clusters
            res_i = []
            for k in range(n_clus):
                cov_cluster = cov[k]
                mean_vec = mu[k,:]
                Z_i = Z[i,:]
                
                try: pdf = multivariate_normal(mean_vec, cov_cluster, allow_singular = True).pdf(Z_i)
                except: 
                    pdf = multivariate_normal(mean_vec, np.identity(self.z_dim), allow_singular = True).pdf(Z_i)

                res_i.append(pi[k]*pdf)
            
            res.append(-np.log(np.sum(res_i) + 0.00001)) # La suma del epsilon es porque a veces da 0 y el loss queda en inf
        
        return np.sum(res)

    def find_phi(self, Z, mu, cov, pi):
        """
        Calcula la matriz de phi
        """
        Z = Z.detach().numpy()
        mu = mu.detach().numpy()
        pi = pi.detach().numpy()

        phi = np.zeros((Z.shape[0], self.n_clusters))
        for i in range(Z.shape[0]):
            for k in range(self.n_clusters):
                denom = 0
                for j in range(self.n_clusters):
                    cov_mat = cov[j] 
                    mean_vec = mu[j,:]
                    Z_i = Z[i,:]
                    try:
                        pdf = multivariate_normal(mean_vec, cov_mat, allow_singular = True).pdf(Z_i)
                    except:
                        pdf = multivariate_normal(mean_vec, np.identity(self.z_dim), allow_singular = True).pdf(Z_i)

                    denom += pi[j][0] * pdf

                cov_mat = cov[k] 
                mean_vec = mu[k,:]
                Z_i = Z[i,:]
                try:
                    pdf = multivariate_normal(mean_vec, cov_mat, allow_singular = True).pdf(Z_i)
                except:
                    pdf = multivariate_normal(mean_vec, np.identity(self.z_dim), allow_singular = True).pdf(Z_i)


                num = pi[k][0]*pdf

                phi[i,k] = num/denom

        return phi 

    def save_checkpoint(self, state, index, filename):
        newfilename = os.path.join(filename, 'FTcheckpoint_%d.pth.tar' % index)
        torch.save(state, newfilename)

    def pretrain_autoencoder(self, x, X_raw, size_factor, batch_size=256, lr=0.001, epochs=400, ae_save=True, ae_weights='AE_weights.pth.tar'):
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
                _, mean_tensor, disp_tensor, pi_tensor = self.forward(x_tensor)
                loss = self.zinb_loss(x=x_raw_tensor, mean=mean_tensor, disp=disp_tensor, pi=pi_tensor, scale_factor=sf_tensor)
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

    def fit(self, X, X_raw, sf, lr=1., batch_size=256, num_epochs=10, update_interval=1, tol=1e-3, save_dir=''):
        '''X: tensor data'''
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            self.cuda()

        save_dir = self.path +'/'
        print("Clustering stage")
        X = torch.tensor(X)#.cuda()
        X_raw = torch.tensor(X_raw)#.cuda()
        sf = torch.tensor(sf)#.cuda()

        self.pi.requires_grad = True 
        self.mu.requires_grad = True 
        
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

        for epoch in range(num_epochs):
            print(f"---> Epoca {epoch}")

            if epoch%update_interval == 0:
                latent = self.encodeBatch(X)
                
                z = self.encodeBatch(X)
                pi = torch.div(self.pi, torch.sum(self.pi)) # No se si esto está bien ponerlo acá
                    
                if self.cov_identidad == False:
                    phi = self.find_phi(z,self.mu,self.cov, pi)
                    self.cov = self.find_covariance(z, self.mu, phi)

                distr = self.find_probabilities(z,self.mu, self.cov)
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
                            'pi': pi,
                            'cov': self.cov,
                            }, epoch+1, filename=save_dir)
                    
                self.y_pred_last = self.y_pred
                if epoch>0 and delta_label < tol:
                    
                    with open(f'{self.path}/DATOS_DESPUES_KMEANS{epoch}.pickle', 'wb') as handle:
                        pickle.dump( latent, handle)

                    print('delta_label ', delta_label, '< tol ', tol)
                    print("Reach tolerance threshold. Stopping training.")
                    break
            
            cluster_loss_val = 0
            recon_loss_val = 0
            train_loss = 0
            # train 1 epoch for clustering loss
            for batch_idx in range(num_batch):
                xbatch = X[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)]
                xrawbatch = X_raw[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)]
                sfbatch = sf[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)]

                optimizer.zero_grad()
                inputs = Variable(xbatch)
                rawinputs = Variable(xrawbatch)
                sfinputs = Variable(sfbatch)

                z, meanbatch, dispbatch, pibatch = self.forward(inputs)

                # Se normalizan los valores para pi
                pi = torch.div(self.pi, torch.sum(self.pi)) # No se si esto está bien ponerlo acá

                if self.cov_identidad == False:
                    phi = self.find_phi(z,self.mu,self.cov, pi)
                    self.cov = self.find_covariance(z, self.mu, phi)

                cluster_loss = self.clustering_GMM_loss(z, cov=self.cov, mu=self.mu, pi = pi)
                recon_loss = self.zinb_loss(rawinputs, meanbatch, dispbatch, pibatch, sfinputs)
                loss = cluster_loss + recon_loss
                
                cluster_loss_val += cluster_loss * len(inputs)
                recon_loss_val += recon_loss * len(inputs)
                train_loss = cluster_loss_val + recon_loss_val
                
                loss.backward()
                optimizer.step()
                
            print("#Epoch %3d: Total: %.4f Clustering Loss: %.4f ZINB Loss: %.4f" % (
                epoch + 1, train_loss / num, cluster_loss_val / num, recon_loss_val / num))            
            
            if epoch == num_epochs - 1: 
                with open(f'{self.path}/DATOS_DESPUES_KMEANS{epoch}.pickle', 'wb') as handle:
                    pickle.dump( latent, handle)
               
        return self.y_pred, self.mu, self.pi, self.cov, z, epoch
