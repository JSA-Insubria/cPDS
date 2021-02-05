import numpy as np
import cPDS
import util
import plot


def tuning(n_agent, L):
    return 5, 10, 10

    ts = [1, 1, 1, 5, 5, 5, 10, 10, 10, 25, 25, 25]
    taus = [0.1, 1, 10, 0.1, 1, 10, 0.1, 1, 10, 0.1, 1, 10]
    rhos = [0.1, 1, 10, 0.1, 1, 10, 0.1, 1, 10, 0.1, 1, 10]

    max_iters = 100
    xtrain, ytrain, xtest, ytest = util.loadData()

    auc = np.zeros(len(ts))

    for param_idx in range(len(ts)):
        t = ts[param_idx]
        tau = taus[param_idx]
        rho = rhos[param_idx]

        x = train(n_agent, max_iters, L, t, tau, rho, xtrain, ytrain)
        auc[param_idx] = plot.compute_error(xtest, ytest, x)

    max_auc = auc.argmax()
    print('t: ', ts[max_auc], ', tau: ', taus[max_auc], ', rho: ', rhos[max_auc])
    return ts[max_auc], taus[max_auc], rhos[max_auc]


def train(n_agent, max_iters, L, t, tau, rho, xtrain, ytrain):
    m = n_agent

    n, gammas, data, labels = util.federatedData(m, xtrain, ytrain)
    x, y, q_kminus1, q = util.initcPDSVar(m, xtrain, gammas, n, data, labels)
    theta = t * np.random.uniform(0, 1, m)

    cPDSs = []
    for j in range(m):
        cPDSs.append(cPDS.cPDS(j, tau, rho, theta[j], gammas[j], data[j], labels[j], q[j], n[j], x[j]))

    lambdaa = L @ x

    for i in range(max_iters):
        x = np.asarray([cPDSs[node].compute(lambdaa[node]) for node in range(m)])

        # encrypt for node
        lambdaa_kplus1 = np.empty(shape=lambdaa.shape)
        for node in range(m):
            lambdaa_kplus1[node] = aggregator_sum(node, L[node], lambdaa[node], x)

        lambdaa = lambdaa_kplus1

    return x


def aggregator_sum(node, L, lambdaa_k, x):
    tmp_sum = np.zeros(shape=lambdaa_k.shape)
    for j in range(len(L)):
        if L[j] != 0:
            v = np.asarray([[x_i * L_i for x_i in x[j]] for L_i in L])
            tmp_sum += v[j]

    lambdaa_kplus1 = lambdaa_k + tmp_sum
    return lambdaa_kplus1