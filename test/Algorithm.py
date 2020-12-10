import networkx as nx
import matplotlib.pyplot as plt
from statistics import mean
import numpy as np
import scipy.io as spio
import pywt

import torch
import syft as sy


# the following is purely for the purpose of pretty printing matrices
from IPython.display import display
import sympy; sympy.init_printing()


def local_degree(P, eps_deg):
    n = P.shape[0]
    W = np.zeros((n, n))
    deg = P.sum(axis=1)
    for i in range(0, n):
        for j in range(0, n):
            if P[i, j] == 1:
                W[i, j] = 1.0/(max(deg[i], deg[j])+eps_deg)
    W = np.diag(np.ones(n)-(W.sum(axis=1))) + W
    W = (W+np.transpose(W)+2*np.eye(n, n))/4
    return W


def loadData():
    mat = spio.loadmat('../data/dataset3.mat', squeeze_me=True)
    xtrain = mat['xtrain']
    ytrain = mat['ytrain']
    xtest = mat['xtest']
    ytest = mat['ytest']
    # permute order of datapoints in training set to mix {+1}, {-1} examples
    random_idx = np.random.rand(xtrain.shape[0]).argsort()
    np.take(xtrain, random_idx, axis=0, out=xtrain)
    np.take(ytrain, random_idx, axis=0, out=ytrain)
    return xtrain, ytrain, xtest, ytest


def getSplitIndices(m, xtrain):
    # assign training datapoints to agents
    n = xtrain.shape[0]  # number of training datapoints
    nppa = int(n / m)  # number of (data)points per agent
    n_j_start = np.arange(start=0, stop=n, step=nppa, dtype=np.int)
    n_j_stop = np.arange(start=nppa - 1, stop=n, step=nppa, dtype=np.int)
    return n_j_start, n_j_stop


def initFederatedDataVar(m, xtrain):
    n = np.zeros((m,), dtype=np.object)
    gammas = np.zeros((m,), dtype=np.object)
    data = np.zeros((m,), dtype=np.object)
    labels = np.zeros((m,), dtype=np.object)
    return n, gammas, data, labels


def federatedData(m, xtrain, ytrain):
    n_j_start, n_j_stop = getSplitIndices(m, xtrain)
    n, gammas, data, labels = initFederatedDataVar(m, xtrain)
    for j in range(m):
        n[j] = n_j_stop[j] - n_j_start[j] + 1
        gammas[j] = np.ones((1, n_j_stop[j] - n_j_start[j] + 1))
        data[j] = xtrain[n_j_start[j]:n_j_stop[j] + 1, :]  # size n_j x d+1
        labels[j] = ytrain[n_j_start[j]:n_j_stop[j] + 1]  # size n_j x 1

    return n, gammas, data, labels


def initcPDSVar(m, xtrain, gammas, n, data, labels):
    x = np.random.normal(0, 1, (m, xtrain.shape[1] + 1))  # miu=0, sigma=1, size: m x p
    y = np.zeros((m,), dtype=np.object)
    q_kminus1 = np.zeros((m,), dtype=np.object)
    q = np.zeros((m,), dtype=np.object)
    for j in range(m):
        y[j] = np.random.normal(0, 1, (1, n[j]))
        q_kminus1[j] = np.zeros((1, n[j]))
        q[j] = gammas[j] * (np.einsum('ij,ij->i',
                                      np.concatenate([np.diag(labels[j]) @ data[j], labels[j].reshape(n[j], 1)],
                                                     axis=1),
                                      np.tile(x[j, :], (n[j], 1))) - y[j])

    return x, y, q_kminus1, q


def loadDataCentralized():
    # Load optimal solution computed by centralized approach
    mat = spio.loadmat('../data/sSVM_gurobi.mat', squeeze_me=True)
    x_opt = mat['theta_opt_SSVM']
    # x_opt = np.array([0.7738, 0.7131, 0.0000, 0.0433, -0.0112, 0.0462])
    w_SSVM = x_opt[:-1]
    b_SSVM = x_opt[-1]
    return x_opt, w_SSVM, b_SSVM


class cPDS:
    def __init__(self, S, L_p, theta, gammas, data, labels, q, n, x):
        self.tau = 10
        self.rho = 10
        self.S = S
        self.L_p = L_p
        self.theta = theta  # theta[j] -> 1xm
        self.gammas = gammas  # gammas[j] -> 1x250
        self.data = data  # data[j] -> 250x5
        self.labels = labels  # labels[j] -> 250x1
        self.q = q  # q[j] -> 1x250
        self.q_kminus1 = np.zeros((1, n))  # 1x250 init:0
        self.x = x  # x[j] -> 1xp (p=n°features+1)
        self.lamda_kminus1 = np.zeros((4, 6))  # # 1x6 init:0
        self.n = n

    def compute(self, lambdaa):
        # x-Update
        beta_k_j = self.x[:-1]
        beta_k_j0 = self.x[-1]
        lambda_d_k = lambdaa[:, -1]
        lambda_d_kminus1 = self.lamda_kminus1[:, -1]
        lamda_dplus1_k = lambdaa[:, -1]
        lamda_dplus1_kminus1 = self.lamda_kminus1[:, -1]

        #lambda_d_k = lambdaa[:-1]
        #lambda_d_kminus1 = self.lamda_kminus1[:-1]
        #lamda_dplus1_k = lambdaa[-1]
        #lamda_dplus1_kminus1 = self.lamda_kminus1[-1]

        # Beta_k+1_jt
        mu = (2 * self.rho) / (self.tau + self.theta)
        #v1 = sum(self.gammas * self.labels * (2 * self.q - self.q_kminus1) @ self.data)
        v1 = np.sum(np.diag(self.gammas * self.labels * (2 * self.q - self.q_kminus1)) * self.data, axis=0)
        #v2 = 2 * lambda_d_k - lambda_d_kminus1
        v2 = self.L_p @ self.S @ (2 * lambda_d_k - lambda_d_kminus1)
        v3 = self.theta * beta_k_j
        u = ((v1 + v2 - v3) / (self.tau + self.theta)) * -1
        beta_kplus1_jt = pywt.threshold(u, mu, 'soft')

        # Beta_k+1_j0
        v1_0 = np.sum(self.gammas * self.labels * (2 * self.q - self.q_kminus1), axis=1)
        #v2_0 = 2 * lamda_dplus1_k - lamda_dplus1_kminus1
        v2_0 = self.L_p @ self.S @ (2 * lamda_dplus1_k - lamda_dplus1_kminus1)
        v3_0 = self.theta * beta_k_j0
        beta_kplus1_j0 = ((-1 * v1_0) - v2_0 + v3_0) / self.theta

        self.x = np.append([beta_kplus1_jt], beta_kplus1_j0)

        # y-Update
        t0 = np.transpose(1/(self.gammas ** 2))
        t1 = np.ones((self.n, 1))
        t2 = np.transpose(self.gammas * self.q)
        t3 = np.einsum('ij,ij->i', np.concatenate(
            [
                np.diag(np.ravel(self.gammas ** 2) * self.labels) @ self.data,
                np.transpose((self.gammas ** 2) * self.labels)
            ], axis=1), np.tile(self.x, (self.n, 1))).reshape(self.n, 1)

        y_kplus1_j_temp1 = t0 * (t1 + t2 + t3)
        y_kplus1_j_temp2 = t0 * (t2 + t3)

        y_kplus1_j = np.ones((self.n, 1))
        y_kplus1_j[1 > y_kplus1_j_temp1] = y_kplus1_j_temp1[1 > y_kplus1_j_temp1]
        y_kplus1_j[1 < y_kplus1_j_temp2] = y_kplus1_j_temp2[1 < y_kplus1_j_temp2]

        # q-Update
        q_kplus1_j = self.q + self.gammas * np.transpose(
            np.transpose(np.einsum('ij,ij->i', np.concatenate(
                [
                    np.diag(self.labels) @ self.data,
                    np.transpose(self.labels).reshape(self.n, 1)
                ], axis=1), np.tile(self.x, (self.n, 1)))).reshape(self.n, 1) - y_kplus1_j)

        self.q = q_kplus1_j
        self.lamda_kminus1 = lambdaa

        return self.x


def plot(residuals_x, x, xtrain, xtest, ytrain, ytest, w_SSVM, b_SSVM):
    # plot residuals
    plt.plot(residuals_x)
    plt.ylabel('Residuals x')
    plt.xlabel('Iterations')
    plt.show()
    # plot residuals on the log scale
    plt.semilogy(residuals_x)
    plt.ylabel('Residuals x in logscale')
    plt.xlabel('Iterations')
    plt.show()

    x_return = np.mean(x, 0)
    w_cPDS = x_return[:-1]
    b_cPDS = x_return[-1]

    x1 = np.arange(-2 + np.min(np.concatenate((xtrain[:, 0], xtest[:, 0]), axis=0)),
                   np.max(np.concatenate((xtrain[:, 0], xtest[:, 0]), axis=0)) + 2, 0.1)
    x2_cPDS = (-w_cPDS[0] / w_cPDS[1]) * x1 - b_cPDS / w_cPDS[1]
    x2_SSVM = (-w_SSVM[0] / w_SSVM[1]) * x1 - b_SSVM / w_SSVM[1]

    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(111)
    ax.scatter(xtrain[ytrain == -1, 0], xtrain[ytrain == -1, 1], c='red', marker='o', label='training class -1')
    ax.scatter(xtrain[ytrain == 1, 0], xtrain[ytrain == 1, 1], c='blue', marker='x', label='training class +1')
    ax.scatter(xtest[ytest == -1, 0], xtest[ytest == -1, 1], c='green', marker='s', label='test class -1')
    ax.scatter(xtest[ytest == 1, 0], xtest[ytest == 1, 1], c='yellow', marker='v', label='test class +1')
    ax.plot(x1, x2_SSVM, linewidth=2, markersize=12, label='SSVM')
    ax.plot(x1, x2_cPDS, linewidth=2, markersize=12, label='cPDS')
    ax.plot()
    plt.legend(loc='lower left')
    plt.grid()
    plt.show()

    # calculate AUC by SSVM
    pred_vals_SSVM = (xtest @ w_SSVM) + b_SSVM
    thresholds = np.sort(pred_vals_SSVM)
    miss = np.zeros(thresholds.size)
    false_alarm = np.zeros(thresholds.size)
    for i_thr in range(len(thresholds)):
        ypred = (pred_vals_SSVM <= thresholds[i_thr]) + 0
        ypred[ypred == 0] = -1
        miss[i_thr] = np.sum(np.logical_and(ypred == -1, ytest == 1)) / np.sum(ytest == 1)
        false_alarm[i_thr] = np.sum(np.logical_and(ypred == 1, ytest == -1)) / np.sum(ytest == -1)

    AUC_SSVM = np.abs(np.trapz(false_alarm, 1 - miss))
    print("AUC SSVM: ", AUC_SSVM)

    # calculate AUC by cPDS
    pred_vals_cPDS = xtest @ w_cPDS + b_cPDS
    thresholds = np.sort(pred_vals_cPDS)
    miss = np.zeros(thresholds.size)
    false_alarm = np.zeros(thresholds.size)
    for i_thr in range(thresholds.size):
        ypred = (pred_vals_cPDS <= thresholds[i_thr]) + 0
        ypred[ypred == 0] = -1
        miss[i_thr] = np.sum(np.logical_and(ypred == -1, ytest == 1)) / np.sum(ytest == 1)
        false_alarm[i_thr] = np.sum(np.logical_and(ypred == 1, ytest == -1)) / np.sum(ytest == -1)

    AUC_cPDS = np.abs(np.trapz(false_alarm, 1 - miss))
    print("AUC cPDS: ", AUC_cPDS)


def __main__():
    m = 4

    adj = nx.adjacency_matrix(nx.erdos_renyi_graph(m, 0.5))
    L = np.eye(m) - local_degree(adj, 0.1)

    xtrain, ytrain, xtest, ytest = loadData()
    x_opt, w_SSVM, b_SSVM = loadDataCentralized()
    n, gammas, data, labels = federatedData(m, xtrain, ytrain)
    x, y, q_kminus1, q = initcPDSVar(m, xtrain, gammas, n, data, labels)

    # define parameters
    t = 5
    theta = t * np.eye(m) + np.diag(np.random.uniform(0, 1, m))  # size: m x m
    S = np.eye(m)
    L_p = np.eye(m)

    max_iters = 100
    residuals_x = np.zeros(max_iters, dtype=np.double)

    cPDSs = []
    for j in range(m):
        cPDSs.append(cPDS(S, L_p[j, :], theta[j][j], gammas[j], data[j], labels[j], q[j], n[j], x[j]))

    lambdaa = S @ L @ x
    for iter in range(max_iters):
        for j in range(m):
            #x[j] = cPDSs[j].compute(lambdaa[j])
            x[j] = cPDSs[j].compute(lambdaa)

        lambdaa = lambdaa + S @ L @ x
        residuals_x[iter] = np.linalg.norm(x - (np.ones((m, 1)) * x_opt))

    print(lambdaa)
    plot(residuals_x, x, xtrain, xtest, ytrain, ytest, w_SSVM, b_SSVM)


__main__()
