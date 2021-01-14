import numpy as np

import phe.paillier as paillier

import util as util
import graph_util as graph_util
import cPDS as cPDS
import plot as plot
import extra as extra

import datetime


gp_param = 0
m = 0


def save_time(file, time_pre):
    time_post = datetime.datetime.now()
    util.writeIntoCSV(m, 'not_' + str(gp_param), file, str((time_post - time_pre).total_seconds()))


def aggregator_sum(L, S, lambdaa, x):
    for i in range(L.shape[0]):
        time_pre = datetime.datetime.now()
        for j in range(L.shape[1]):
            if (i != j) & (L[i][j] != 0):
                x[i] = x[i] + x[j]
        save_time('agent_sum_' + str(i), time_pre)

    time_pre = datetime.datetime.now()
    lambdaa = lambdaa + L @ S @ x
    save_time('lambda_sum', time_pre)
    return lambdaa


def agent_encrypt(cPDSs, lambdaa, j):
    x_tmp = cPDSs.compute(lambdaa)
    time_pre = datetime.datetime.now()
    save_time('agent_enc_' + str(j), time_pre)
    return x_tmp


def main_decrypt(lambdaa_encrypted):
    time_pre = datetime.datetime.now()
    lambdaa = lambdaa_encrypted
    save_time('decrypt', time_pre)
    return lambdaa


def main_iter_error(x_opt,xtrain, ytrain, x):
    residuals_x = np.linalg.norm(x - (np.ones((m, 1)) * x_opt))

    error_x = (1 - plot.compute_error(xtrain, ytrain, x))
    #error_x = (1 - extra.compute_error_extra(xtrain, ytrain, x_dec))
    return residuals_x, error_x


def startcPDS(n_agent, graph_param):

    global m, gp_param
    m = n_agent
    gp_param = graph_param

    #adj = graph_util.get_graph(m, 0.1)
    #L = np.eye(m) - util.local_degree(adj, 1)

    L = graph_util.read_graph(m)

    # define parameters
    t = 5

    xtrain, ytrain, xtest, ytest = util.loadData()
    #xtrain, ytrain, xtest, ytest = extra.loadData_extra()

    x_opt, w_SSVM, b_SSVM = util.loadDataCentralized()

    n, gammas, data, labels = util.federatedData(m, xtrain, ytrain)
    x, y, q_kminus1, q = util.initcPDSVar(m, xtrain, gammas, n, data, labels)

    # define parameters
    theta = t * np.eye(m) + np.diag(np.random.uniform(0, 1, m))  # size: m x m
    S = np.eye(m)
    L_p = L

    max_iters = 100
    residuals_x = np.zeros(max_iters, dtype=np.double)
    error_x = np.zeros(max_iters, dtype=np.double)

    cPDSs = []
    for j in range(m):
        cPDSs.append(cPDS.cPDS(j, S[j], L_p[j], theta[j][j], gammas[j], data[j], labels[j], q[j], n[j], x[j]))

    lambdaa = S @ L @ x

    for i in range(max_iters):
        iteration_time_pre = datetime.datetime.now()

        x = np.asarray([agent_encrypt(cPDSs[j], lambdaa[j], j) for j in range(m)])

        # sum x between nodes
        lambdaa = aggregator_sum(L, S, lambdaa, x)

        save_time('iteration_time', iteration_time_pre)

        # compute residual and error
        residuals_x[i], error_x[i] = main_iter_error(x_opt, xtrain, ytrain, x)

    plot.plot_error('not', m, gp_param, error_x, max_iters)
    plot.plot('not', m, gp_param, residuals_x, x, xtrain, xtest, ytrain, ytest, w_SSVM, b_SSVM)
    #extra.plot_extra(x, xtrain, xtest, ytrain, ytest)


if __name__ == "__main__":
    #gp = [0.1, 0.5, 1]
    gp = [0.1]
    for j in gp:
        #agents = [5, 10, 20, 30]
        agents = [5]
        for i in agents:
            startcPDS(i, j)

        #util.computeAgentsMean(agents, j)
