import os
import numpy as np


def getSplitIndices(m, xtrain):
    # assign training datapoints to agents
    n = xtrain.shape[0]  # number of training datapoints
    nppa = int(n / m)  # number of (data)points per agent
    n_j_start = np.arange(start=0, stop=n, step=nppa, dtype=np.int)
    n_j_stop = np.arange(start=nppa - 1, stop=n, step=nppa, dtype=np.int)
    return n_j_start, n_j_stop


def initFederatedDataVar(m):
    n = np.empty((m,), dtype=np.object)
    gammas = np.empty((m,), dtype=np.object)
    data = np.empty((m,), dtype=np.object)
    labels = np.empty((m,), dtype=np.object)
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
    q = np.zeros((m,), dtype=np.object)
    for j in range(m):
        y[j] = np.random.normal(0, 1, (1, n[j]))
        q[j] = gammas[j] * (np.einsum('ij,ij->i',
                                      np.concatenate((np.diag(labels[j]) @ data[j], labels[j].reshape(n[j], 1)), axis=1),
                                      np.tile(x[j, :], (n[j], 1))) - y[j])

    return x, q


def compute_auc(w_param, b_param, xtest, ytest, classes):
    pred_vals_cPDS = xtest @ w_param + b_param
    thresholds = np.sort(pred_vals_cPDS, axis=0)
    miss = np.zeros(thresholds.size)
    false_alarm = np.zeros(thresholds.size)
    for i_thr in range(thresholds.size):
        ypred = (pred_vals_cPDS <= thresholds[i_thr]) + 0
        ypred[ypred == 0] = classes[0]
        miss[i_thr] = np.sum(np.logical_and(ypred == classes[0], ytest == classes[1])) / np.sum(ytest == classes[1])
        false_alarm[i_thr] = np.sum(np.logical_and(ypred == classes[1], ytest == classes[0])) / np.sum(ytest == classes[0])

    return np.abs(np.trapz(false_alarm, 1 - miss))


def save_auc(m, gp_param, auc_cPDS):
    with open('logs' + os.sep + "auc.csv", 'a') as fd:
        fd.write(str(m) + ',' + str(gp_param) + ',' + str(auc_cPDS) + '\n')