import numpy as np
import pandas as pd
import scipy.stats as spst
import math

def synthetic_covariates_from_mixtures(weights, N, mean, cov, random_state):
    '''
    weights: np.array
    N: int
    mean: list of np.arrays
    cov: list of np.arrays(or int)

    X: pd.Dataframe
    '''
    N_j = N*weights
    N_j = N_j.astype(int)
    M = weights.shape[0]
    p = mean[0].shape[0]
    col_name = [f'x{i}' for i in range(1, p + 1)]
    X = []
    for j in range(M):
        X_j = spst.multivariate_normal.rvs(mean=mean[j], cov=cov[j], random_state=random_state, size=int(N_j[j]))
        X_j = pd.DataFrame(X_j, columns=col_name)
        Z_i = np.ones(N_j[j])*j
        X_j['Z'] = Z_i
        X.append(X_j)
    X = pd.concat(X, ignore_index=True)
    return X

def X_weights_mean_cov_generator(num_cluster, mean_loc, mean_scale, cov_scale, p, random_state):
    '''
    num_cluster: int
    mean_loc: float
    mean_scale: float
    cov_scale: float
    p: int
    random_state: int

    weights: np.array
    mean: list
    cov: list
    '''
    weights = np.ones(num_cluster)/num_cluster
    mean = []
    cov = []
    for i in range(num_cluster):
        mean_i = spst.uniform.rvs(loc=mean_loc, scale=mean_scale, size=p, random_state=random_state+i)
        cov_i = (i+1)*cov_scale/num_cluster
        mean.append(mean_i)
        cov.append(cov_i)
    return weights, mean, cov

def Beta_and_Std_generator(X_mean, X_cov, desired_mean):
    '''
    X_mean: list
    X_cov: list
    desired_mean: list

    Beta: list
    Std: list
    '''
    num_cluster = len(X_mean)
    Beta = []
    Std = []
    for i in range(num_cluster):
        beta_i = X_mean[i] * desired_mean[i]/np.dot(X_mean[i], X_mean[i])
        Beta.append(beta_i)
        std_i = X_cov[i]
        Std.append(std_i)
    return Beta, Std

def synthetic_survival_data_generator(mean, std, censor_rate, X, random_state):
    '''
    mean: list
    std: list
    censor_rate: np.array
    X: pd.DataFrame
    random_state: int

    data: pd.DataFrame
    '''
    N_j = X['Z'].value_counts(sort=False)
    X_raw = X.drop(columns='Z')
    N_j = N_j.astype(int)
    N_j_0 = N_j*censor_rate
    N_j_0 = N_j_0.astype(int)
    N_j_1 = N_j - N_j_0
    C = []
    SF_0 = []
    SF_1 = []
    for j in range(len(mean)):
        SF_0_j = spst.uniform.rvs(loc=0.7, scale=0.29, size=N_j[j], random_state=random_state+j)
        SF_1_j = spst.uniform.rvs(loc=0, scale=0.7, size=N_j[j], random_state=random_state+j)
        C_j_0 = np.ones(N_j_0[j])*0
        C_j_1 = np.ones(N_j_1[j])*1
        C_j = np.append(C_j_0, C_j_1)
        C.append(C_j)
        SF_0.append(SF_0_j)
        SF_1.append(SF_1_j)
    C = np.hstack(C)
    X['C'] = C
    Y = []
    True_likeli = []
    y_mean = []
    for j in range(len(mean)):
        X_j = X_raw.loc[X['Z'] == j,].values
        C_j = X.loc[X['Z'] == j, 'C'].values
        mean_y = np.matmul(X_j, mean[j])
        print(mean_y.max(), mean_y.min())
        std_j = std[j]
        Y_j_1 = spst.lognorm.isf(SF_1[j], s=std_j, scale=np.exp(mean_y))
        true_likeli_j = SF_1[j] * C_j + SF_0[j] * (1 - C_j)
        Y_j_0 = spst.lognorm.isf(SF_0[j], s=std_j, scale=np.exp(mean_y))
        Y_j = (Y_j_1*C_j) + (Y_j_0 * (1-C_j))
        Y.append(Y_j)
        True_likeli.append(true_likeli_j)
        y_mean.append(mean_y)
    Y = np.hstack(Y)
    True_likeli = np.hstack(True_likeli)
    Y_mean = np.hstack(y_mean)
    X['y'] = Y

    return X, True_likeli, Y_mean


np.random.seed(555)
weights, mean, cov = X_weights_mean_cov_generator(num_cluster=3, mean_loc=-3, mean_scale=5.75, cov_scale=0.1, p=50, random_state=999)
XX = []
for i in range(20):
    print(i)
    X = synthetic_covariates_from_mixtures(weights=weights, N=500, mean=mean, cov=cov, random_state=i+3)
    beta, std = Beta_and_Std_generator(X_mean=mean, X_cov=cov, desired_mean=[3,5,8])
    print(beta)
    data, true_sf, y_mean = synthetic_survival_data_generator(mean=beta, std=std, censor_rate=np.array([0.4,0.4,0.4]), X=X, random_state=i+3)
    data.to_csv(f'Simulation_3_cluster_subset_{i+1}.csv')


    