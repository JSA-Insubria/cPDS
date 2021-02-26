import numpy as np
from sklearn.model_selection import train_test_split

import util
import cPDS


def tuning(n_agent, max_iters, L, data, labels, classes):
    # return 5, 10, 10

    ts = [1, 1, 1, 5, 5, 5, 10, 10, 10, 25, 25, 25]
    taus = [0.1, 1, 10, 0.1, 1, 10, 0.1, 1, 10, 0.1, 1, 10]
    rhos = [0.1, 1, 10, 0.1, 1, 10, 0.1, 1, 10, 0.1, 1, 10]

    xtrain, xtest, ytrain, ytest = train_test_split(data, labels)
    n, gammas, data, labels = util.federatedData(n_agent, xtrain, ytrain)
    x_init, q = util.initcPDSVar(n_agent, xtrain, gammas, n, data, labels)

    auc = np.zeros(len(ts))
    for param_idx in range(len(ts)):
        t = ts[param_idx]
        tau = taus[param_idx]
        rho = rhos[param_idx]

        w_cPDS, b_cPDS = train(n_agent, max_iters, L, t, tau, rho, n, gammas, data, labels, x_init, q)
        auc[param_idx] = util.compute_auc(w_cPDS, b_cPDS, xtest, ytest, classes)
        print('AUC: ', auc[param_idx], ', t: ', ts[param_idx], ', tau: ', taus[param_idx], ', rho: ', rhos[param_idx])

    max_auc = auc.argmax()
    print('Optimal cPDS parameters: ', 't: ', ts[max_auc], ', tau: ', taus[max_auc], ', rho: ', rhos[max_auc],
          '. Maximum training AUC=', auc[max_auc])
    return ts[max_auc], taus[max_auc], rhos[max_auc]


def train(n_agent, max_iters, L, t, tau, rho, n, gammas, data, labels, x, q):
    m = n_agent
    theta = t + np.random.uniform(0, 1, m)

    cPDSs = []
    for j in range(m):
        cPDSs.append(cPDS.cPDS(j, tau, rho, theta[j], gammas[j], data[j], labels[j], q[j], n[j], x[j], L[j]))

    lambdaa = L @ x

    for i in range(max_iters):
        x = np.asarray([cPDSs[node].compute(lambdaa[node]) for node in range(m)])

        # encrypt for node
        lambdaa_kplus1 = np.empty(shape=lambdaa.shape)
        for node in range(m):
            lambdaa_kplus1[node] = aggregator_sum(L[node], lambdaa[node], x)

        lambdaa = lambdaa_kplus1

    x_return = np.mean(x, axis=0)
    w_cPDS = x_return[:-1]
    b_cPDS = x_return[-1]
    return w_cPDS, b_cPDS


def aggregator_sum(L, lambdaa_k, x):
    tmp_sum = np.zeros(shape=lambdaa_k.shape)
    for j in range(len(L)):
        if L[j] != 0:
            v = np.asarray([[x_i * L_i for x_i in x[j]] for L_i in L])
            tmp_sum += v[j]

    lambdaa_kplus1 = lambdaa_k + tmp_sum
    return lambdaa_kplus1
