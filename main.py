import numpy as np

import phe.paillier as paillier

import util as util
import graph_util as graph_util
import cPDS as cPDS
import plot as plot
import extra as extra
import not_enc as not_enc

import datetime


gp_param = 0
m = 0


def save_time(file, time_pre):
    time_post = datetime.datetime.now()
    util.writeIntoCSV(m, 'enc_' + str(gp_param), file, str((time_post - time_pre).total_seconds()))


def aggregator_sum(L, S, lambdaa_k, x):
    lambdaa_kplus1 = np.empty(shape=lambdaa_k.shape, dtype=object)
    for i in range(L.shape[0]):
        time_pre = datetime.datetime.now()
        for j in range(L.shape[1]):
            if L[i][j] != 0:
                v = L[j].reshape(-1, 1) * x[j]
                lambdaa_kplus1[i] = lambdaa_k[i] + v[j]
        save_time('agent_sum_' + str(i), time_pre)

    return lambdaa_kplus1


def agent_encrypt(cPDSs, lambdaa, pk, j):
    x_tmp = cPDSs.compute(lambdaa)
    time_pre = datetime.datetime.now()
    x_enc = pk.encryptMatrix(x_tmp)
    save_time('agent_enc_' + str(j), time_pre)
    return x_enc


def main_decrypt(msk, lambdaa_encrypted):
    time_pre = datetime.datetime.now()
    lambdaa = msk.decryptMatrix(lambdaa_encrypted)
    save_time('decrypt', time_pre)
    return lambdaa


def main_iter_error(msk, x_opt,xtrain, ytrain, x):
    x_dec = msk.decryptMatrix(x)
    residuals_x = np.linalg.norm(x_dec - (np.ones((m, 1)) * x_opt))

    error_x = (1 - plot.compute_error(xtrain, ytrain, x_dec))
    #error_x = (1 - extra.compute_error_extra(xtrain, ytrain, x_dec))
    return residuals_x, error_x


def startcPDS(n_agent, graph_param):

    global m, gp_param
    m = n_agent
    gp_param = graph_param

    adj = graph_util.get_graph(m, gp_param)
    L = np.eye(m) - util.local_degree(adj, 0.1)

    mpk, msk, pk_list, sk_list = paillier.generate_cPDS_keypair(m+1)

    # define parameters
    t = 5

    xtrain, ytrain, xtest, ytest = util.loadData()
    #xtrain, ytrain, xtest, ytest = extra.loadData_extra()

    x_opt, w_SSVM, b_SSVM = util.loadDataCentralized()

    n, gammas, data, labels = util.federatedData(m, xtrain, ytrain)
    x_init, y, q_kminus1, q = util.initcPDSVar(m, xtrain, gammas, n, data, labels)

    # define parameters
    theta = t * np.eye(m) + np.diag(np.random.uniform(0, 1, m))  # size: m x m
    S = np.eye(m)
    L_p = L

    max_iters = 100
    residuals_x = np.zeros(max_iters, dtype=np.double)
    error_x = np.zeros(max_iters, dtype=np.double)
    x = np.empty(shape=x_init.shape, dtype=object)

    cPDSs = []
    for j in range(m):
        cPDSs.append(cPDS.cPDS(j, S[j], L_p[j], theta[j][j], gammas[j], data[j], labels[j], q[j], n[j], x_init[j]))

    lambdaa = S @ L @ x_init

    for i in range(max_iters):
        iteration_time_pre = datetime.datetime.now()

        # compute and encrypt x
        x = np.asarray([agent_encrypt(cPDSs[j], lambdaa[j], pk_list[j], j) for j in range(m)])

        # sum and decrypt lambdaa
        lambdaa_encrypted = aggregator_sum(L, S, lambdaa, x)
        lambdaa = main_decrypt(msk, lambdaa_encrypted)

        save_time('iteration_time', iteration_time_pre)

        # compute residual and error
        residuals_x[i], error_x[i] = main_iter_error(msk, x_opt, xtrain, ytrain, x)

    x_dec = msk.decryptMatrix(x)
    plot.plot_error('enc', m, gp_param, error_x, max_iters)
    plot.plot('enc', m, gp_param, residuals_x, x_dec, xtrain, xtest, ytrain, ytest, w_SSVM, b_SSVM)
    #extra.plot_extra(x_dec, xtrain, xtest, ytrain, ytest)

    not_enc.main_not_enc(m, graph_param, max_iters, w_SSVM, b_SSVM, x_opt, xtrain, ytrain, xtest, ytest, S, L, L_p,
                         theta, gammas, data, labels, q, n, x_init)


if __name__ == "__main__":
    gp = [0.1, 0.5, 1]
    for j in gp:
        agents = [5, 10, 20, 30]
        for i in agents:
            startcPDS(i, j)

        util.computeAgentsMean(agents, j)
