import numpy as np

import util as util
import graph_util as graph_util
import cPDS as cPDS
import plot as plot
import extra as extra
import not_enc as not_enc
import key_gen_util as key_gen_util
import param_tuning

import datetime


gp_param = 0
m = 0


def save_time(file, time_pre):
    time_post = datetime.datetime.now()
    util.writeIntoCSV(m, 'enc_' + str(gp_param), file, str((time_post - time_pre).total_seconds()))


def save_time_enc(file, time):
    util.writeIntoCSV(m, 'enc_' + str(gp_param), file, str(time))


def compute_Lx(L, x):
    res = np.zeros(shape=x.shape)
    for i in range(len(L)):
        if L[i] != 0:
            res[i] = L[i] * x[i]

    return res


def aggregator_sum(node, L, lambdaa_k, x):
    time_pre = datetime.datetime.now()
    tmp_sum = np.zeros(shape=lambdaa_k.shape, dtype=object)
    for j in range(len(L)):
        if L[j] != 0:
            tmp_sum += x[j]

    lambdaa_kplus1 = lambdaa_k + tmp_sum
    save_time('agent_sum_' + str(node), time_pre)

    return lambdaa_kplus1


def agent_encrypt(keys_dict, L, x, node, enc_time_nodes):
    x_enc_node = np.zeros(shape=x.shape, dtype=object)
    key = 0
    for other_node in range(len(L)):
        if L[other_node] != 0:
            if other_node != node:
                time_pre = datetime.datetime.now()
                x_enc_node[other_node] = keys_dict['pk_list' + str(node)][key].encryptMatrix(x[other_node])
                enc_time_nodes[other_node] += (datetime.datetime.now() - time_pre).total_seconds()
                key += 1
            else:
                time_pre = datetime.datetime.now()
                x_enc_node[other_node] = keys_dict['pk_list' + str(node)][-1].encryptMatrix(x[other_node])
                enc_time_nodes[other_node] += (datetime.datetime.now() - time_pre).total_seconds()

    return x_enc_node, enc_time_nodes


def main_decrypt(keys_dict, lambdaa_encrypted):
    lambdaa = np.empty(lambdaa_encrypted.shape)
    for node in range(m):
        time_pre = datetime.datetime.now()
        lambdaa[node] = keys_dict['msk' + str(node)].decryptMatrix(lambdaa_encrypted[node])
        save_time('agent_dec_' + str(node), time_pre)

    return lambdaa


def main_iter_error(x_opt, xtrain, ytrain, x):
    residuals_x = np.linalg.norm(x - (np.ones((m, 1)) * x_opt))

    error_x = (1 - plot.compute_error(xtrain, ytrain, x))
    #error_x = (1 - extra.compute_error_extra(xtrain, ytrain, x))
    return residuals_x, error_x


def startcPDS(n_agent, graph_param, L, t, tau, rho):

    global m, gp_param
    m = n_agent
    gp_param = graph_param

    keys_dict = key_gen_util.gen_keys(L)

    xtrain, ytrain, xtest, ytest = util.loadData()
    #xtrain, ytrain, xtest, ytest = extra.loadData_extra()

    x_opt, w_SSVM, b_SSVM = util.loadDataCentralized()

    n, gammas, data, labels = util.federatedData(m, xtrain, ytrain)
    x_init, y, q_kminus1, q = util.initcPDSVar(m, xtrain, gammas, n, data, labels)

    # define parameters
    theta = t + np.random.uniform(0, 1, m)

    max_iters = 100
    residuals_x = np.zeros(max_iters, dtype=np.double)
    error_x = np.zeros(max_iters, dtype=np.double)
    x = np.empty(shape=x_init.shape, dtype=object)

    cPDSs = []
    for j in range(m):
        cPDSs.append(cPDS.cPDS(j, tau, rho, theta[j], gammas[j], data[j], labels[j], q[j], n[j], x_init[j]))

    lambdaa = L @ x_init

    for i in range(max_iters):
        iteration_time_pre = datetime.datetime.now()

        # compute x for all nodes
        x = np.asarray([cPDSs[node].compute(lambdaa[node]) for node in range(m)])

        # encrypt for node
        lambdaa_kplus1 = np.zeros(shape=lambdaa.shape, dtype=object)
        enc_time_nodes = np.zeros(shape=m)
        for node in range(m):
            res = compute_Lx(L[node], x)
            x_enc, enc_time_nodes = agent_encrypt(keys_dict, L[node], res, node, enc_time_nodes)
            lambdaa_kplus1[node] = aggregator_sum(node, L[node], lambdaa[node], x_enc)

        # save agent time
        [save_time_enc('agent_enc_' + str(node), enc_time_nodes[node]) for node in range(m)]

        lambdaa = main_decrypt(keys_dict, lambdaa_kplus1)
        save_time('iteration_time', iteration_time_pre)

        # compute residual and error
        residuals_x[i], error_x[i] = main_iter_error(x_opt, xtrain, ytrain, x)

    plot.plot_error('enc', m, gp_param, error_x, max_iters)
    plot.plot('enc', m, gp_param, residuals_x, x, xtrain, xtest, ytrain, ytest, w_SSVM, b_SSVM)
    #extra.plot_extra(x, xtrain, xtest, ytrain, ytest)

    not_enc.main_not_enc(m, graph_param, max_iters, w_SSVM, b_SSVM, x_opt, xtrain, ytrain, xtest, ytest, L, tau, rho,
                         theta, gammas, data, labels, q, n, x_init)


if __name__ == "__main__":
    gp = [0.1, 0.5, 1]
    for j in gp:
        agents = [5, 10]
        for i in agents:
            adj = graph_util.get_graph(i, j)
            L = np.eye(i) - util.local_degree(adj, 0.1)
            t, tau, rho = param_tuning.tuning(i, L)
            startcPDS(i, j, L, t, tau, rho)

        util.computeAgentsMean(agents, j)
