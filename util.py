import numpy as np
import scipy.io as spio
import os
import pandas as pd


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
    mat = spio.loadmat('data' + os.sep + 'dataset3.mat', squeeze_me=True)
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


def initFederatedDataVar(m):
    n = np.zeros((m,), dtype=np.object)
    gammas = np.zeros((m,), dtype=np.object)
    data = np.zeros((m,), dtype=np.object)
    labels = np.zeros((m,), dtype=np.object)
    return n, gammas, data, labels


def federatedData(m, xtrain, ytrain):
    n_j_start, n_j_stop = getSplitIndices(m, xtrain)
    n, gammas, data, labels = initFederatedDataVar(m)
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
    mat = spio.loadmat('data' + os.sep + 'sSVM_gurobi.mat', squeeze_me=True)
    x_opt = mat['theta_opt_SSVM']
    # x_opt = np.array([0.7738, 0.7131, 0.0000, 0.0433, -0.0112, 0.0462])
    w_SSVM = x_opt[:-1]
    b_SSVM = x_opt[-1]
    return x_opt, w_SSVM, b_SSVM


def writeIntoCSV(m, folder_name, file_name, row):
    path = 'logs' + os.sep + folder_name + os.sep + str(m) + '_agents'
    if not os.path.exists(path):
        os.makedirs(path)

    with open(path + os.sep + file_name + '.csv', 'a') as fd:
        fd.write(row + '\n')


def computeAgentsMean(m, graph_param):
    folder = ['enc_' + str(graph_param), 'not_' + str(graph_param)]
    for file in folder:
        path_file = 'logs' + os.sep + file + os.sep
        first_line = "iteration n°,"
        mean = pd.DataFrame()
        for i in m:
            first_line = first_line + str(i) + " Agents,,,,,,,"
            path = path_file + str(i) + "_agents" + os.sep
            data = {}
            for j in range(i+1):
                data[j] = pd.read_csv(path + "agent_" + str(j) + ".csv", header=None)

            df_agent = pd.concat(data, axis=1).iloc[:, :-1]
            df_agent_max = df_agent.max(axis=1).to_frame()
            df_agent_max.columns = ['Encryption time max - Agents (s)']
            df_agent_min = df_agent.min(axis=1).to_frame()
            df_agent_min.columns = ['Encryption time min - Agents (s)']
            df_agent_mean = df_agent.mean(axis=1).to_frame()
            df_agent_mean.columns = ['Encryption time mean - Agents (s)']

            df_lambda = pd.concat(data, axis=1).iloc[:, -1]
            df_lambda.columns = ['Encryption time lambda (s)']

            aggr = pd.read_csv(path + "aggregator.csv", header=None)
            aggr.columns = ['Sum time (s)']

            main = pd.read_csv(path + "main.csv", header=None)
            main.columns = ['Decryption time (s)']

            iteration = pd.read_csv(path + "iteration_time.csv", header=None)
            iteration.columns = ['Iteration time (s)']

            mean = pd.concat([mean, df_agent_max, df_agent_min, df_agent_mean, df_lambda, aggr, main, iteration], axis=1)

        with open(path_file + "time.csv", 'w') as fd:
            fd.write(first_line + '\n')
        mean = mean.round(3)
        mean.to_csv(path_file + "time.csv", mode='a')
