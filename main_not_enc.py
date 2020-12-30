import numpy as np

import util as util
import graph_util as graph_util
import cPDS as cPDS
import plot as plot

import datetime


def save_time(m, file, time_pre):
    time_post = datetime.datetime.now()
    util.writeIntoCSV(m, file, str((time_post - time_pre).total_seconds()))


def aggregator_sum(lambdaa, S, L, x):
    lambdaa_k_plus_1 = lambdaa + S @ L @ x
    return lambdaa_k_plus_1


def agent_encrypt(cPDSs, lambdaa):
    x_tmp = cPDSs.compute(lambdaa)
    return x_tmp


def __main__(m):

    adj = graph_util.get_graph(m, 0.5)
    L = np.eye(m) - util.local_degree(adj, 1)

    # define parameters
    t = 5

    xtrain, ytrain, xtest, ytest = util.loadData()
    x_opt, w_SSVM, b_SSVM = util.loadDataCentralized()

    n, gammas, data, labels = util.federatedData(m, xtrain, ytrain)
    x, y, q_kminus1, q = util.initcPDSVar(m, xtrain, gammas, n, data, labels)

    # define parameters
    theta = t * np.diag(np.random.uniform(0, 1, m))
    S = np.eye(m)
    L_p = L

    max_iters = 100
    residuals_x = np.zeros(max_iters, dtype=np.double)

    cPDSs = []
    for j in range(m):
        cPDSs.append(cPDS.cPDS(j, S[j], L_p[j], theta[j][j], gammas[j], data[j], labels[j], q[j], n[j], x[j]))

    lambdaa = S @ L @ x
    total_time_pre = datetime.datetime.now()

    for i in range(max_iters):
        iteration_time_pre = datetime.datetime.now()
        for j in range(m):
            x[j] = agent_encrypt(cPDSs[j], lambdaa[j])

        # sum lambdaa
        lambdaa = aggregator_sum(lambdaa, S, L, x)
        #residuals_x[iter] = np.linalg.norm(msk.decryptMatrix(x) - (np.ones((m, 1)) * x_opt))

        save_time(m, 'iteration_time', iteration_time_pre)

    save_time(m, 'execution_time', total_time_pre)

    plot.plot(residuals_x, x, xtrain, xtest, ytrain, ytest, w_SSVM, b_SSVM)


#m = [10, 30, 60, 100]
m = [10]
for i in m:
    __main__(i)

