import pandas as pd
import numpy as np
from numpy import matmul
from scipy import stats
from scipy import special
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from numpy.linalg import inv
import seaborn as sns
import math
import random
from pandas.core.frame import DataFrame
from pandas.core.series import Series
from typing import Any
from numpy import dtype, floating, ndarray
import time







def vectorized_gaussian_logpdf(X, means, covariances):
    # """
    # Compute log N(x_i; mu_i, sigma_i) for each x_i, mu_i, sigma_i
    # Args:
    #     X : shape (d)
    #         Data points
    #     means : shape (self.N, d)
    #         Mean vectors
    #     covariances : shape (self.N, d)
    #         Diagonal covariance matrices
    # Returns:
    #     logpdfs : shape (n,)
    #         Log probabilities
    # """
        d = X.shape[0]
        constant = d * np.log(2 * np.pi)
        log_determinants = np.log(np.prod(covariances, axis=1))
        deviations = X - means
        inverses = 1 / covariances
        return -0.5 * (constant + log_determinants + np.sum(deviations * inverses * deviations, axis=1))

def vectorized_norm_logsf(X, means, covariances):
    #  '''
    #  Compute log Norm(X_i, mu_i, sigma_i) for each x_i, mu_i, sigma_i
    #  Args:
    #      X: shape (n,)
    #      means: shape (n,d)
    #      covariances (d,)
    #  Returns: logsf: shape (d,)
    #  '''
    sf = [stats.norm.sf(x=X, loc=means[:,i], scale=np.sqrt(covariances[i])) for i in range(means.shape[1])]
    sf = np.array(sf).prod(axis=1)
    return sf

def vectorized_norm_logpdf(X, means, covariances):
    #  '''
    #  Compute log Norm(X_i, mu_i, sigma_i) for each x_i, mu_i, sigma_i
    #  Args:
    #      X: shape (n,)
    #      means: shape (n,d)
    #      covariances (d,)
    #  Returns: logsf: shape (d,)
    #  '''
    sf = [stats.norm.pdf(x=X, loc=means[:,i], scale=np.sqrt(covariances[i])) for i in range(means.shape[1])]
    sf = np.array(sf).prod(axis=1)
    return sf

class Gibbs_Survival_DP_subset:
        def __init__(self, N, M, scalability_amplifier=1, burnin=5000, thinning=50, B=100, a_3=1, b_3=0.001, a_4=1, b_4=0.001, random_state=555) -> None:
                
                self.burnin: int = burnin
                # Number of iterations to be dropped
                self.thinning: int = thinning
                # Number of iterations out of which one iteration is recorded
                self.B: int = B
                # Number of iterations to be recorded
                self.A: float = scalability_amplifier
                # Amplifier due to scalability
                self.a_3: float = a_3
                # Shape parameter of gamma distribution of mu_theta
                self.b_3: float = b_3
                # Scale parameter of gamma distribution of mu_theta
                self.a_4: float = a_4
                # Shape parameter of gamma distribution of mu_psi
                self.b_4: float = b_4
                # Scale parameter of gamma distribution of mu_psi
                self.N: int = N
                # Number of predefined theta clusters
                self.n_N: ndarray[int] = None
                # Number of observations in each theta cluster
                self.M: int = M
                # Number of predefined psi clusters
                self.n_M: ndarray[int] = None
                # Number of observations in each psi cluster
                self.random_state: int = random_state
                self.data: DataFrame = None
                # Data to be analyzed by this model
                self.beta_y: ndarray = None
                # Collection of theta
                self.beta_x: ndarray = None
                # Collection of psi
                self.response_name: str = None
                # Column name of time-to-event variable
                self.censoring_name: str = None
                # Column name of event variable
                self.predictor_name: ndarray|list = None
                # Column name of predictors
                self.predictor = None
                # Data of covariates
                self.n: int = None
                # Number of observations in the data
                self.mu_theta: float = None
                # Concentration parameter of the theta-layer DP
                self.mu_psi: float = None
                # Concentration parameter of the psi-layer DP
                self.concentration_parameter_records: DataFrame = pd.DataFrame() 
                # Records of concentration parameters in each collected iteration
                self.cluster_records: list = [] 
                # Records of clustering allocatn in each collected iteration
                self.data_records = None 
                # Records of data in each collected iteration
                self.K = None 
                # Clustering allocation of y
                self.J = None 
                # Clustering allocation of X
                self.p: int = None 
                # Dimensionality of covariates
                self.a_y = None
                # Shape parameter of inverse-gamma distribution of partial coeff in covariance matrix in the base measure of theta
                self.b_y = None 
                # Scale parameter of inverse-gamma distribution of partial coeff in covariance matrix in the base measure of theta
                self.a_x = 1 
                # Shape parameter of inverse-gamma distribution of partial coeff in covariance matrix in the base measure of psi
                self.b_x = None 
                # Scale parameter of inverse-gamma distribution of partial coeff in covariance matrix in the base measure of psi
                self.tau_y = None 
                # Partial coeff in covariance matrix in theta
                self.tau_x = None 
                # Partial coeff in covariance matrix in psi
                self.C_y = None 
                # Partial inv-covariance matrix in the base measure of theta
                self.c_X = None 
                # Partial inv-covariance matrix in the base measure of psi
                self.mu_0_theta = None 
                # Mean vector in the base measure of theta
                self.mu_0_psi = None 
                # Mean vector in the base measure of psi
                self.p_theta = None
                # Weights in theta-cluster, (N,)
                self.p_psi = None
                # Weights in psi-cluster, (N,M)
                self.posterior_samples = None
        
        def initialize(self) -> None:

            # Initialize the theta, psi, and corresponding clustering allocations based on selected N,M:

            # 1) Initialize the basic properties of training data
            np.random.seed(self.random_state)
            self.predictor: DataFrame = self.data.drop(columns=[self.response_name, self.censoring_name, 'z_x', 'z_y', 'y_mu', 'y_sigma'])
            self.n, self.p = self.predictor.shape
            self.predictor_name = self.predictor.columns

            # 2) Initialize the theta clustering allocations using KMeans
            kmeans_y = KMeans(n_clusters=self.N, random_state=self.random_state).fit(self.data[[self.response_name, self.censoring_name]])
            self.K: ndarray[Any, dtype[int]] = kmeans_y.labels_
            self.data['z_y'] = self.K

            # 3) Prepare for the initialization of theta, psi, and psi-allocations
            group = self.data.groupby('z_y')
            kmeans_x = KMeans(n_clusters=self.M, random_state=self.random_state).fit(self.predictor)
            self.data['z_x'] = kmeans_x.labels_
            self.n_N = group.size()
            self.n_M = []
            beta_y = []
            tau_y = []
            self.beta_x = []
            self.tau_x = []
            col_names_x_mu: list[str] = [i+"_mu" for i in self.predictor_name]
            col_names_x_sigma: list[str] = [i + "_sigma" for i in self.predictor_name]

                # 3.1) Initialize theta in each theta-allocation ignoring the censorship
            for name, dat in group:
                y = dat[self.response_name]
                X = dat[self.predictor_name]
                sigma_y = y.std()**2
                reg = LinearRegression().fit(X, y)
                tau_y.append(sigma_y)
                beta_y.append(reg.coef_)
                group_x = dat.groupby('z_x')
                nnn_M = []
                for i in range(self.M):
                    nnn_M.append(((dat['z_y'] == name)&(dat['z_x'] == i)).sum())
                self.n_M.append(nnn_M)

                # 3.2) Initialize psi and its allocation
            group_x = self.data.groupby('z_x')
            for name_x, dat_xx in group_x:
                dat_xx = dat_xx[self.predictor_name]
                beta_x = dat_xx.mean(axis=0)
                if dat_xx.shape[0] == 1:
                    sigma_x = abs(dat_xx.values.reshape((self.p-1,))/20)
                else:
                    sigma_x = dat_xx.std(axis=0)
                sigma_x = sigma_x + np.random.uniform(0.0001,0.001,self.p)
                self.beta_x.append(beta_x)
                self.tau_x.append(sigma_x**2)
            self.J: DataFrame = self.data[['z_y', 'z_x']]
            self.beta_y = np.array(beta_y) # shape (N, p)
            self.tau_y = np.array(tau_y) # shape(N,)
            print(tau_y)
            self.beta_x = np.array(self.beta_x) # shape (N,M,p)
            self.tau_x = np.array(self.tau_x) # shape (N,M,p)
            self.n_M = np.array(self.n_M) # shape (N,M)

        def squared_SBP(self):
            # '''
            # Squared stick-breaking process:
            # '''
            # print('z_y', self.n_N)
            x_theta = self.n_M.copy()
            a_theta = self.n_N.copy() + 1
            b_theta = self.mu_theta + self.n_N.sum() - np.cumsum(self.n_N)
            # print('b_theta', b_theta)
            v_theta = stats.beta.rvs(a=a_theta[:-1], b=b_theta[:-1], random_state=self.n+self.random_state)
            v_theta = np.append(v_theta, 1)
            x = np.roll(1-v_theta, 1, axis=0)
            x[0] = 1
            p_theta = v_theta*np.cumprod(x)
            p_psi = []
            V_psi = []
            for i in range(self.n_N.shape[0]):
                  nkj = x_theta[i]
                  a_psi = nkj + 1
                  b_psi = a_psi + nkj.sum() - np.cumsum(nkj)
                  v_psi = stats.beta.rvs(a=a_psi, b=b_psi, random_state=self.N+self.M+self.random_state)
                  v_psi[-1] = 1
                  V_psi.append(v_psi)
                  x = np.roll(1-v_psi, 1, axis=0)
                  x[0] = 1
                  ppsi = v_psi*np.cumprod(x)
                  p_psi.append(ppsi)
            p_psi = np.array(p_psi)
            V_psi = np.array(V_psi)
            return p_theta, p_psi, v_theta, V_psi

        def prior_specification(self) -> None:
            # '''
            # Specify the hyper-parameters in the joint base measure.
            # '''
            X: DataFrame = self.data[self.predictor_name]
            self.c_X = X.std(axis=0).values
            self.C_y = inv(matmul(X.values.T, X.values))#1*np.diag(np.ones(self.p))#np.matmul(X.T.values, X.values)
            y: Series = self.data[self.response_name]
            reg = LinearRegression(fit_intercept=False).fit(X,y)
            self.mu_0_theta = matmul(inv(matmul(X.values.T,X.values)), matmul(X.values.T, y.values))
            self.mu_0_psi = self.predictor.mean(axis=0).values
            self.a_y = 2
            self.b_y = 10#self.n/self.p
            self.b_x = 0.25
            self.mu_theta = self.a_3*self.b_3
            self.mu_psi = self.a_4*self.b_4
            # self.n_N = []
            # self.n_M = []
            # for i in range(self.N):
            #     self.n_N.append(self.data['z_y'] == i+1).sum()
            #     n_M_N = []
            #     for j in range(self.M):
            #         n_M_N.append((self.data['z_y'] == i+1)&(self.data['z_x'] == j+1)).sum()
            #     self.n_M.append(n_M_N)
            # self.n_N = np.array(self.n_N)
            # self.n_M = np.array(self.n_M)
            self.p_theta, self.p_psi,a,b = self.squared_SBP()

        def fit(self, data, response_name, censoring_name):
            self.data = data
            self.response_name = response_name
            self.censoring_name = censoring_name
            self.data_records = data.copy()
            self.initialize()
            self.prior_specification()
            self.data[['z_y', 'z_x', self.censoring_name]] = self.data[['z_y', 'z_x', self.censoring_name]].astype(int)
            mu_theta_recording=[]
            mu_psi_recording=[]
            z_y_recording=[]
            z_x_recording=[]
            beta_y_recording=[]
            tau_y_recording=[]
            beta_x_recording=[]
            tau_x_recording=[]
            loglikelihood_y_recording = []
            log_z_y_trace = []

            start_0 = time.time()
            ### BG sampler begins:
            for j in range(1, self.burnin+self.B*self.thinning+1):
                print(j)
                start = time.time()
                ### Step 1.1: update cluster allocations
                # if j == 142:
                #     break
                for i in range(self.n):
                    # if j >= 141:
                    #     print(i)
                    y, delta, z_y, z_x = self.data.loc[i, [self.response_name, self.censoring_name, 'z_y', 'z_x']]
                    # if j >= 141:
                    #     print(y, z_y, z_x)
                    self.n_N[int(z_y)] += -1
                    # print(z_y)
                    self.n_M[int(z_y)][int(z_x)] += -1
                    x = self.predictor.iloc[i,].values
                    mean_y = matmul(x, self.beta_y.T)
                    # if j >= 141:
                    #     print(mean_y)
                    # print(matmul(self.C_y, self.mu_0_theta).shape)
                    c_Y = (self.A*matmul(x.reshape(-1,1), x.reshape(1,-1))+self.C_y)
                    new_beta_mean = matmul(inv(c_Y), (self.A*x*y).reshape(-1,1)+matmul(self.C_y, self.mu_0_theta.reshape(-1,1)))
                    new_tau_shape = self.a_y+self.A/2
                    new_tau_scale = self.b_y+0.5*(y**2 + matmul(matmul(self.mu_0_theta.reshape(1,-1), self.C_y), self.mu_0_theta.reshape(-1,1))-matmul(matmul(new_beta_mean.reshape(1,-1), c_Y), new_beta_mean.reshape(-1,1)))
                    # print(y**2, matmul(matmul(self.mu_0_theta.T, self.C_y), self.mu_0_theta), matmul(matmul(new_beta_mean.T, c_Y), new_beta_mean))
                    # print(new_tau_scale)
                    new_tau = stats.invgamma.rvs(a=new_tau_shape, scale=new_tau_scale)
                    
                    new_beta_tau = new_tau*inv(c_Y)
                    
                    new_beta = stats.multivariate_normal.rvs(mean=new_beta_mean.reshape(1,-1)[0], cov=new_beta_tau, size=100)
                    new_y_mean = matmul(x, new_beta.T)
                    # print(y, new_y_mean)
                    
                    # print(self.tau_y[int(z_y)], new_tau)
                    new_likelihood_y = ((stats.norm.pdf(y, loc=new_y_mean, scale=np.sqrt(new_tau))**(self.A*delta))*(stats.norm.sf(y, loc=new_y_mean, scale=np.sqrt(new_tau))**((1-delta)*self.A))).mean()
                    likelihood_y = (stats.norm.pdf(y, loc=mean_y, scale=np.sqrt(self.tau_y))**(self.A*delta))*(stats.norm.sf(y, loc=mean_y, scale=np.sqrt(self.tau_y))**((1-delta)*self.A))
                    likelihood_y = np.append(likelihood_y, new_likelihood_y)
                    # if j >= 141:
                    #     print(likelihood_y/likelihood_y.sum())
                    new_mean_x = (1/(self.A/self.tau_x[int(z_x)] + self.c_X/self.tau_x[int(z_x)]))*(self.A*x/self.tau_x[int(z_x)] + self.mu_0_psi*self.c_X/self.tau_x[int(z_x)])
                    new_sigma_x = 1/(self.A/self.tau_x[int(z_x)] + self.c_X/self.tau_x[int(z_x)])
                    new_beta_x = stats.multivariate_normal.rvs(mean=new_mean_x, cov=np.diag(new_sigma_x), size=100)
                    new_shape_x  = self.a_x + self.A/2
                    new_scale_x = self.b_x + self.A*0.5*((x - new_beta_x) ** 2)
                    sigma_x_new = stats.invgamma.rvs(a=new_shape_x, scale=new_scale_x)
                    new_likelihood_x = np.exp(vectorized_gaussian_logpdf(x, new_beta_x, (sigma_x_new/self.c_X))).mean()
                    likelihood_x = np.exp(vectorized_gaussian_logpdf(x, self.beta_x, self.tau_x/self.c_X))
                    likelihood_x = np.append(likelihood_x, new_likelihood_x)
                    n_N = self.n_N.copy()
                    n_N = np.append(n_N, self.mu_theta)
                    n_M = self.n_M.copy()
                    n_M = np.append(n_M.T, [self.mu_psi*np.ones(self.n_N.shape[0])], axis=0).T
                    # if j >= 141:
                    #     print(likelihood_x, n_M)
                    n_M = (n_M.T/n_M.sum(axis=1)).T
                    ppx_in_cluster_y = matmul(n_M, likelihood_x)
                    ppy_given_x = np.append(ppx_in_cluster_y, new_likelihood_x)
                    # if j >= 141:
                    #     print(ppx_in_cluster_y)
                    ppy = ppy_given_x*n_N*likelihood_y
                    F_y = np.append(range(self.n_N.shape[0]), self.n_N.shape[0])
                    # print(F_y)
                    z_y = random.choices(F_y, weights=ppy/ppy.sum(), k=1)[0]
                    # if j >= 20 and (z_y == 1 or z_y==2):
                    #     print(ppy/ppy.sum())
                    #     print(likelihood_y)
                    #     print(mean_y, self.data.loc[i,'y_mu'], y, delta)
                    # if j >= 141:
                    #     print(z_y)
                    # print((pp_y/pp_y.sum()).max())
                    # print((pp_y/pp_y.sum())[z_y])
                    if z_y == (n_N.shape[0]-1):
                        z_beta_y = random.choice(range(100))
                        self.beta_y = np.append(self.beta_y, [new_beta[z_beta_y]], axis=0)
                        self.tau_y = np.append(self.tau_y, new_tau)
                        self.n_N.loc[self.n_N.index.max()+1] = 1
                        self.n_M = np.append(self.n_M, [np.zeros(self.n_M.shape[1])], axis=0)
                    else:
                        self.n_N[z_y] +=1
                    self.data.loc[i, 'z_y'] = z_y
                    pp_x = np.append(self.n_M.sum(axis=0), self.mu_psi)
                    pp_x = pp_x*likelihood_x
                    z_x = random.choices(np.arange(pp_x.shape[0]), weights=pp_x/pp_x.sum(), k=1)[0]
                    if z_x == (pp_x.shape[0] - 1):
                        self.beta_x = np.append(self.beta_x, [new_beta_x[0]], axis=0)
                        self.tau_x = np.append(self.tau_x, [sigma_x_new[0]], axis=0)
                        self.n_M = np.append(self.n_M.T, [np.zeros(self.n_M.shape[0])], axis=0)
                        self.n_M = self.n_M.T
                        self.n_M[int(z_y)][int(z_x)] = 1
                    else:
                        self.n_M[int(z_y)][int(z_x)] += 1
                    self.data.loc[i, 'z_x'] = z_x
                
                ### Step 1.2: M-H
                # ### Move 1:
                # n_N = self.n_N.copy()
                # n_M = self.n_M.copy()
                # population_move_3 = []
                # population_move_1 = []
                # for i in range(self.n_N.shape[0]):
                #     a = self.n_M[i]
                #     if (a!=0).sum() == 1:
                #         a = np.argwhere(a)[0][0]
                #         ind = [i, a]
                #         population_move_3.append(ind)
                #     elif (a!=0).sum() > 1:
                #         a = np.argwhere(a)
                #         for r in range((a!=0).sum()):
                #             population_move_1.append([i,a[r][0]])
                # n_1_ = len(population_move_3)
                # n_2_plus = len(population_move_1)
                # if n_2_plus >=1:
                #     move1 = random.choice(population_move_1)
                #     population_k_move_1 = np.arange(n_N.shape[0])[n_N!=0]
                #     population_k_move_1 = population_k_move_1[population_k_move_1!=move1[0]]
                #     move1_target = random.choice(population_k_move_1)
                #     if [move1_target, move1[1]] in population_move_1:
                #         n_2_plus_new = n_2_plus - 1
                #     elif [move1_target, move1[1]] in population_move_3:
                #         n_2_plus_new = n_2_plus-1
                #     else:
                #         n_2_plus_new = n_2_plus
                #     dat = self.data[(self.data['z_y'] == move1[0])&(self.data['z_x'] == move1[1])]
                #     y_mean_in_move = matmul(dat[self.predictor_name], self.beta_y[move1[0]])
                #     y_mean_in_target = matmul(dat[self.predictor_name], self.beta_y[move1_target])
                #     likelihood_in_move = np.prod(stats.norm.pdf(dat[self.response_name], loc=y_mean_in_move, scale=np.sqrt(self.tau_y[move1[0]])))
                #     likelihood_in_target = np.prod(stats.norm.pdf(dat[self.response_name], loc=y_mean_in_target, scale=np.sqrt(self.tau_y[move1_target])))
                #     if (likelihood_in_move !=0) & (n_2_plus_new != 0):
                #         p = ((special.gamma(n_N[move1[0]]- n_M[move1[0]][move1[1]]) * special.gamma(n_N[move1_target] + n_M[move1[0]][move1[1]]))/(special.gamma(n_N[move1[0]]) * special.gamma(n_N[move1_target])))*(
                #             ((special.gamma(n_N[move1[0]]+self.mu_psi) * special.gamma(n_N[move1_target] + self.mu_psi))/(special.gamma(n_N[move1[0]]+self.mu_psi-n_M[move1[0]][move1[1]]) * special.gamma(n_N[move1_target]+self.mu_psi+n_M[move1[0]][move1[1]]))))*(
                #             likelihood_in_target/likelihood_in_move)*(n_2_plus/n_2_plus_new)
                #         p = min(1,p)
                #         u = stats.uniform.rvs(0,1)
                #         print(u, p)
                #         if u < p:
                #             self.data.loc[(self.data['z_y'] == move1[0])&(self.data['z_x'] == move1[1]), 'z_y'] = move1_target
                #             self.n_M[move1_target, move1[1]] = self.n_M[move1_target, move1[1]]+self.n_M[move1[0]][move1[1]]
                #             self.n_N[move1[0]] = self.n_N[move1[0]] - self.n_M[move1[0]][move1[1]]
                #             self.n_N[move1_target] = self.n_N[move1_target] + self.n_M[move1[0]][move1[1]]
                #             self.n_M[move1[0]][move1[1]] = 0
                #             n_2_plus = n_2_plus_new
                #             if [move1_target, move1[1]] in population_move_3:
                #                 n_1_ += -1
                #                 population_move_3.remove([move1_target, move1[1]])
                # n_N = self.n_N.copy()
                # n_M = self.n_M.copy()
                # u = stats.uniform.rvs(0,1)
                # if u>=0.5:
                #     if len(population_move_3) >= 1:
                #         move3 = random.choice(population_move_3)
                #         population_k_move_3 = np.arange(n_N.shape[0])[n_N!=0]
                #         population_k_move_3 = population_k_move_3[population_k_move_3!=move3[0]]
                #         move3_target = random.choice(population_k_move_3)
                #         if [move3_target, move3[1]] in population_move_1:
                #             n_2_plus_new = n_2_plus
                #             n_1_new = n_1_-1
                #         elif [move3_target, move3[1]] in population_move_3:
                #             n_2_plus_new = n_2_plus
                #             n_1_new = n_1_
                #         else:
                #             n_2_plus_new = n_2_plus + 1
                #             n_1_new = n_1_ -1
                #         dat = self.data[(self.data['z_y'] == move3[0])&(self.data['z_x'] == move3[1])]
                #         y_mean_in_move = matmul(dat[self.predictor_name], self.beta_y[move3[0]])
                #         y_mean_in_target = matmul(dat[self.predictor_name], self.beta_y[move3_target])
                #         likelihood_in_move = np.prod(stats.norm.pdf(dat[self.response_name], loc=y_mean_in_move, scale=np.sqrt(self.tau_y[move3[0]])))
                #         likelihood_in_target = np.prod(stats.norm.pdf(dat[self.response_name], loc=y_mean_in_target, scale=np.sqrt(self.tau_y[move3_target])))
                #         if (likelihood_in_move !=0)&(n_2_plus_new!=0):
                #             p = ((special.gamma(n_N[move3_target] + n_M[move3[0]][move3[1]]))/(special.gamma(n_N[move3[0]]) * special.gamma(n_N[move3_target])))*(
                #                 ((special.gamma(n_N[move3[0]]+self.mu_psi) * special.gamma(n_N[move3_target] + self.mu_psi))/(special.gamma(self.mu_psi) * special.gamma(n_N[move3_target]+self.mu_psi+n_M[move3[0]][move3[1]]))))*(
                #                 likelihood_in_target/likelihood_in_move)*((n_1_*len(population_k_move_3) - 1)/(self.mu_theta*n_2_plus_new))
                #             p = min(1,p)
                #             u = stats.uniform.rvs(0,1)
                #             if u < p:
                #                 self.data.loc[(self.data['z_y'] == move3[0])&(self.data['z_x'] == move3[1]), 'z_y'] = move1_target
                #                 self.n_M[move3_target, move3[1]] = self.n_M[move3_target, move3[1]]+self.n_M[move3[0]][move3[1]]
                #                 self.n_N[move3[0]] = self.n_N[move3[0]] - self.n_M[move3[0]][move3[1]]
                #                 self.n_N[move3_target] = self.n_N[move3_target] + self.n_M[move3[0]][move3[1]]
                #                 self.n_M[move3[0]][move3[1]] = 0


                ### Step 1.2: Relist the clustering
                unorder_z_y = np.sort(self.data['z_y'].unique())
                self.beta_y = self.beta_y[unorder_z_y]
                # print(unorder_z_y)
                self.tau_y = self.tau_y[unorder_z_y]
                # print(self.data['z_y'].unique())
                self.data['z_y'].replace(unorder_z_y, np.arange(self.beta_y.shape[0]), inplace=True)

                unorder_z_x = np.sort(self.data['z_x'].unique())
                self.beta_x = self.beta_x[unorder_z_x]
                self.tau_x = self.tau_x[unorder_z_x]
                self.data['z_x'].replace(unorder_z_x, np.arange(self.beta_x.shape[0]), inplace=True)

                ### Step 2.1: Update theta-cluster parameters:
                self.n_N = self.data['z_y'].value_counts(sort=False).sort_index()
                # self.n_N = 
                print(self.n_N)
                # if j >= 8000:
                #     print(self.n_N)
                for i in range(self.beta_y.shape[0]):
                    dat = self.data.drop(columns=['y_mu', 'y_sigma'])
                    dat = dat[dat['z_y'] == i]
                    dat_1 = dat.drop(columns=['z_y', 'z_x'])[dat[self.censoring_name] == 1]
                    dat_0 = dat.drop(columns=['z_y', 'z_x'])[dat[self.censoring_name] == 0]
                    y_uncensored = dat_1[self.response_name].values
                    y_censored = dat_0[self.response_name].values
                    X_uncensored = dat_1[self.predictor_name].values
                    X_censored = dat_0[self.predictor_name].values
                    c_Y = self.A*matmul(X_uncensored.T, X_uncensored)+self.C_y
                    print(X_uncensored.shape)
                    mean = matmul(inv(c_Y), (self.A*matmul(X_uncensored.T, y_uncensored)).reshape(-1,1)+matmul(self.C_y, self.mu_0_theta.reshape(-1,1)))                    
                    sigma = self.tau_y[i]*inv(c_Y)
                    beta_new = stats.multivariate_normal.rvs(mean=mean.reshape(1,-1)[0], cov=sigma, size=5)
                    mean_y = matmul(X_uncensored, beta_new.T)
                    # print(y_uncensored, mean_y)
                    if X_uncensored.shape[0] <= self.p/2:
                        shape = self.a_y+self.A*self.p/2
                    else:
                        shape = self.a_y+self.A*dat_1.shape[0]/2
                    scale = self.b_y+0.5*((y_uncensored**2).sum() + matmul(matmul(self.mu_0_theta.reshape(1,-1), self.C_y), self.mu_0_theta.reshape(-1,1))-matmul(matmul(mean.reshape(1,-1), c_Y), mean.reshape(-1,1)))#self.b_y+self.A*((y_uncensored.reshape(-1,1) - mean_y)**2).sum(axis=0)/2
                    print('x_hyper:', shape, scale)
                    sigma_new = stats.invgamma.rvs(a=shape, scale=scale, size=5)
                    # self.beta_y[i] = beta_new
                    # self.tau_y[i] = sigma_new
                    if dat_0.shape[0] == 0:
                        self.beta_y[i] = beta_new[0]
                        self.tau_y[i] = sigma_new[0]
                    else:
                        mean_censored_y = matmul(X_censored, beta_new.T)
                        sf = vectorized_norm_logsf(X=y_censored, means=mean_censored_y, covariances=sigma_new)
                        # pdf = vectorized_norm_logpdf(X=y_censored, means=mean_censored_y, covariances=sigma_new)
                        # likeli = sf*pdf
                        self.beta_y[i] = beta_new[sf.argmax()]
                        self.tau_y[i] = sigma_new[sf.argmax()]
                print(self.tau_y)


                ### Step 2.2: Update psi-cluster parameters:
                self.n_M = []
                for i in range(self.beta_x.shape[0]):
                    dat = self.data[self.data['z_x'] == i]
                    group = dat.groupby('z_y')
                    n_m = np.zeros(self.n_N.shape[0])
                    n_m_effective = group.size()
                    n_m[n_m_effective.index] = n_m_effective
                    self.n_M.append(n_m)
                    X = dat[self.predictor_name]
                    mean = (1/(self.A*X.shape[0]/self.tau_x[i] + self.c_X/self.tau_x[i]))*(self.A*X.sum(axis=0)/self.tau_x[i] + self.mu_0_psi*self.c_X/self.tau_x[i])
                    sigma_sq = 1/(self.A*X.shape[0]/self.tau_x[i] + self.c_X/self.tau_x[i])
                    new_beta = stats.multivariate_normal.rvs(mean=mean, cov=np.diag(sigma_sq), size=1)
                    shape = (self.a_x + self.A*X.shape[0]/2)
                    scale = self.b_x + self.A*0.5*((X - new_beta) ** 2).sum()
                    # if j >=20:
                    #     print(new_beta)
                    sigma_x_new = stats.invgamma.rvs(a=shape, scale=scale)
                    self.beta_x[i] = new_beta
                    self.tau_x[i] = sigma_x_new
                self.n_M = np.array(self.n_M).T
                # if j >= 8000:
                #     print(self.n_M)
                print(self.n_M)

                ### Step 3.1: Update theta-concentration parameter:
                u = stats.beta.rvs(self.mu_theta + 1, self.data.shape[0])
                p_mu = (self.a_3 + self.beta_y.shape[0] - 1) / (self.data.shape[0] * (self.b_3 - math.log(u)) + self.a_3 + self.beta_y.shape[0] - 1)
                # print(p_mu)
                p_pgamma = [p_mu, 1 - p_mu]
                k = random.choices(range(2), p_pgamma)[0]
                if k != 0:
                    self.mu_theta = stats.gamma.rvs(self.a_3 + self.beta_y.shape[0] - 1, 1/(self.b_3 - math.log(u)))
                else:
                    self.mu_theta = stats.gamma.rvs(self.a_3 + self.beta_y.shape[0], 1/(self.b_3 - math.log(u)))

                ### Step 3.2: Update psi-concentration parameter:
                u = stats.beta.rvs(self.mu_psi + 1, self.data.shape[0])
                p_mu = (self.a_4 + self.beta_x.shape[0] - 1) / (self.data.shape[0] * (self.b_4 - math.log(u)) + self.a_4 + self.beta_x.shape[0] - 1)
                p_pgamma = [p_mu, 1 - p_mu]
                k = random.choices(range(2), p_pgamma)[0]
                if k != 0:
                    self.mu_psi = stats.gamma.rvs(self.a_4 + self.beta_x.shape[0] - 1, 1/(self.b_4 - math.log(u)))
                else:
                    self.mu_psi = stats.gamma.rvs(self.a_4 + self.beta_x.shape[0], 1/(self.b_4 - math.log(u)))
                
                print(self.mu_theta)
                end = time.time()
                print(end-start)

                if (j >= self.burnin - 1 + self.thinning) & ((j + 1 - self.burnin) % self.thinning == 0):
                    self.p_theta, self.p_psi, v_theta, v_psi = self.squared_SBP()
                    LIKE = []
                    for i in range(self.beta_y.shape[0]):
                        dat = self.data.drop(columns=['y_mu', 'y_sigma'])
                        dat = dat[dat['z_y'] == i]
                        dat_1 = dat.drop(columns=['z_y', 'z_x'])[dat[self.censoring_name] == 1]
                        dat_0 = dat.drop(columns=['z_y', 'z_x'])[dat[self.censoring_name] == 0]
                        y_uncensored = dat_1[self.response_name].values
                        # y_censored = dat_0[self.response_name].values
                        X_uncensored = dat_1[self.predictor_name].values
                        # X_censored = dat_0[self.predictor_name].values
                        mean_y = matmul(X_uncensored, self.beta_y.T)
                        # mean_censored_y = matmul(X_censored, self.beta_y.T)
                        pdf = vectorized_norm_logpdf(X=y_uncensored, means=mean_y, covariances=self.tau_y)[i]
                        # sf = vectorized_norm_logsf(X=y_censored, means=mean_censored_y, covariances=self.tau_y)[i]
                        likelihood_y = pdf
                        # likelihood_y = sf*pdf
                        LIKE.append(likelihood_y)
                    loglikelihood_y = np.log(LIKE).sum()
                    log_z_y_trace.append(np.log(self.data['z_y'].sum()))
                    loglikelihood_y_recording.append(loglikelihood_y)
                    mu_theta_recording.append(self.mu_theta)
                    mu_psi_recording.append(self.mu_psi)
                    z_y_recording.append(self.data['z_y'])
                    z_x_recording.append(self.data['z_x'])
                    beta_y_recording.append(self.beta_y)
                    tau_y_recording.append(self.tau_y)
                    beta_x_recording.append(self.beta_x)
                    tau_x_recording.append(self.tau_x)

            self.posterior_samples = {'mu_theta': mu_theta_recording, 'mu_psi': mu_psi_recording, 'z_y': z_y_recording, 
                                                                      'z_x': z_x_recording, 'beta_y': beta_y_recording, 'tau_y': tau_y_recording, 
                                                                      'beta_x': beta_x_recording, 'tau_x': tau_x_recording, 'loglikelihood_y': loglikelihood_y_recording,
                                                                      'log_z_y_trace': log_z_y_trace}
            end_0 = time.time()
            print(end_0-start_0)

np.random.seed(555)
random.seed(555)
dat = pd.read_csv('''C:/Users/ndong/OneDrive - Texas Tech University/Documents/Scalable Survival-DP Package/Simulation/Original Copy of Simulation/Simulation 3/simulated data 3_3.csv''', index_col=0)
EDP = Gibbs_Survival_DP_subset(N=5, M=5, burnin=2000, thinning=2, B=50, random_state=818)
EDP.fit(data=dat, response_name='Y', censoring_name='c_y')
records = EDP.posterior_samples
z_y_name = []
z_x_name = []
for i in range(50):
    z_y_name.append(f'z_y_{i+1}_iteration')
    z_x_name.append(f'z_x_{i+1}_iteration')
Z_Y = pd.DataFrame(np.array(records['z_y']).T, columns=z_y_name)
Z_X = pd.DataFrame(np.array(records['z_x']).T, columns=z_x_name)
Z_Y.to_csv('''simulation_3_3_Z_Y_records.csv''')
Z_X.to_csv('''simulation_3_3_Z_X_records.csv''')
beta_y_name = []
tau_y_name = []
for i in range(50):
    beta_y__name = []
    tau_y__name = []
    for j in range(len(records['beta_y'][i])):
        beta_y__name.append(f'beta_y_cluster{j+1}_in_{i+1}_iteration')
        tau_y__name.append(f'tau_y_cluster{j+1}_in_{i+1}_iteration')
    beta_y_name.append(beta_y__name)
    tau_y_name.append(tau_y__name)
def flatten_extend(matrix):
    flat_list = []
    for row in matrix:
        flat_list.extend(row)
    return flat_list
flatten_beta_y = flatten_extend(records['beta_y'])
flatten_beta_y_name = flatten_extend(beta_y_name)
BETA_Y = pd.DataFrame(np.array(flatten_beta_y).T, columns=flatten_beta_y_name)
BETA_Y.to_csv('''simulation_3_3_BETA_Y_records.csv''')
flatten_tau_y = flatten_extend(records['tau_y'])
flatten_tau_y_name = flatten_extend(tau_y_name)
TAU_Y = pd.DataFrame(np.array(flatten_tau_y).reshape(1,-1), columns=flatten_tau_y_name)
TAU_Y.to_csv('''simulation_3_3_TAU_Y_records.csv''')
beta_x_name = []
tau_x_name = []
for i in range(50):
    beta_x__name = []
    tau_x__name = []
    for j in range(len(records['beta_x'][i])):
        beta_x__name.append(f'beta_x_cluster{j+1}_in_{i+1}_iteration')
        tau_x__name.append(f'tau_x_cluster{j+1}_in_{i+1}_iteration')
    beta_x_name.append(beta_x__name)
    tau_x_name.append(tau_x__name)
flatten_beta_x = flatten_extend(records['beta_x'])
flatten_beta_x_name = flatten_extend(beta_x_name)
BETA_X = pd.DataFrame(np.array(flatten_beta_x).T, columns=flatten_beta_x_name)
BETA_X.to_csv('''simulation_3_3_BETA_X_records.csv''')
flatten_tau_x = flatten_extend(records['tau_x'])
flatten_tau_x_name = flatten_extend(tau_x_name)
TAU_X = pd.DataFrame(np.array(flatten_tau_x).T, columns=flatten_tau_x_name)
TAU_X.to_csv('''simulation_3_3_TAU_X_records.csv''')
Log_z_y_trace = pd.DataFrame(records['log_z_y_trace'])
Log_likelihood = pd.DataFrame(records['loglikelihood_y'])
Log_z_y_trace.to_csv('''simulation_3_3_log_z_y_trace.csv''')
Log_likelihood.to_csv('''simulation_3_3_loglikelihood.csv''')




                    
                    








                    