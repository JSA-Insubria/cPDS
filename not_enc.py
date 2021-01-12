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


def aggregator_sum(lambdaa, S, L, x):
    time_pre = datetime.datetime.now()
    save_time('agent_' + str(m), time_pre)

    time_pre = datetime.datetime.now()
    lambdaa_k_plus_1 = lambdaa + S @ L @ x
    save_time('aggregator', time_pre)
    return lambdaa_k_plus_1


def agent_encrypt(cPDSs, lambdaa, j):
    x_tmp = cPDSs.compute(lambdaa)

    time_pre = datetime.datetime.now()
    save_time('agent_' + str(j), time_pre)
    return x_tmp


def main_decrypt(lambdaa_encrypted):
    time_pre = datetime.datetime.now()
    lambdaa = lambdaa_encrypted
    save_time('main', time_pre)
    return lambdaa


def main_iter_error(x_opt, xtrain, ytrain, x):
    residuals_x = np.linalg.norm(x - (np.ones((m, 1)) * x_opt))

    error_x = (1 - plot.compute_error(xtrain, ytrain, x))
    #error_x = (1 - extra.compute_error_extra(xtrain, ytrain, x))
    return residuals_x, error_x


def main_not_enc(n_agent, graph_param, max_iters,w_SSVM, b_SSVM, x_opt, xtrain, ytrain, xtest, ytest, S, L, L_p, theta, gammas, data, labels, q, n, x):

    global m, gp_param
    m = n_agent
    gp_param = graph_param

    residuals_x = np.zeros(max_iters, dtype=np.double)
    error_x = np.zeros(max_iters, dtype=np.double)

    cPDSs = []
    for j in range(m):
        cPDSs.append(cPDS.cPDS(j, S[j], L_p[j], theta[j][j], gammas[j], data[j], labels[j], q[j], n[j], x[j]))

    lambdaa = S @ L @ x

    for i in range(max_iters):
        iteration_time_pre = datetime.datetime.now()

        x = np.asarray([agent_encrypt(cPDSs[j], lambdaa[j], j) for j in range(m)])

        # sum lambdaa
        lambdaa = aggregator_sum(lambdaa, S, L, x)

        # decrypt lambdaa
        lambdaa = main_decrypt(lambdaa)

        save_time('iteration_time', iteration_time_pre)

        # compute residual and error
        residuals_x[i], error_x[i] = main_iter_error(x_opt, xtrain, ytrain, x)

    plot.plot_error(error_x, max_iters)
    plot.plot(m, gp_param, residuals_x, x, xtrain, xtest, ytrain, ytest, w_SSVM, b_SSVM)
    # extra.plot_extra(x, xtrain, xtest, ytrain, ytest)