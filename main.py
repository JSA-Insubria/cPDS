import networkx as nx
import numpy as np
import os

import phe.paillier as paillier

import util as util
import cPDS as cPDS
import plot as plot
import Aggregator as Aggr

import datetime


def __main__(m):

    adj = nx.adjacency_matrix(nx.erdos_renyi_graph(m, 0.5))
    L = np.eye(m) - util.local_degree(adj, 0.1)

    mpk, msk, pk_list, sk_list = paillier.generate_braghin_keypair(m+1)

    xtrain, ytrain, xtest, ytest = util.loadData()
    x_opt, w_SSVM, b_SSVM = util.loadDataCentralized()
    n, gammas, data, labels = util.federatedData(m, xtrain, ytrain)
    x, y, q_kminus1, q = util.initcPDSVar(m, xtrain, gammas, n, data, labels)

    # define parameters
    t = 5
    theta = t * np.eye(m) + np.diag(np.random.uniform(0, 1, m))  # size: m x m
    S = np.eye(m)
    L_p = np.eye(m)

    max_iters = 100
    residuals_x = np.zeros(max_iters, dtype=np.double)

    cPDSs = []
    for j in range(m):
        cPDSs.append(cPDS.cPDS(j, pk_list[j], S, L_p[j, :], theta[j][j], gammas[j], data[j], labels[j], q[j], n[j], x[j]))

    #lambdaa = S @ L @ x
    lambdaa = x - (L @ x)
    aggregator = Aggr.Aggregator(pk_list[-1], L)

    for i in range(max_iters):
        for j in range(m):
            x[j] = cPDSs[j].compute(m, lambdaa)

        # agents send x to the aggregator
        aggregator.send_encrypted_x(x)
        # sum encrypted values
        lambdaa_encrypted = aggregator.sum_lambaa_x(m, lambdaa)

        # decrypt lambdaa
        time_pre = datetime.datetime.now()
        lambdaa = msk.decryptMatrix(lambdaa_encrypted)
        time_post = datetime.datetime.now()
        util.writeIntoCSV(m, 'main', str((time_post - time_pre).total_seconds()))
        #residuals_x[iter] = np.linalg.norm(x - (np.ones((m, 1)) * x_opt))

    print(lambdaa)
    plot.plot(residuals_x, x, xtrain, xtest, ytrain, ytest, w_SSVM, b_SSVM)


m = [10, 30, 60, 100]
for i in m:
    __main__(i)
