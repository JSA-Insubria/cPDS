import datetime
import numpy as np

import load_save_data
import cPDS


gp_param = 0
m = 0


def save_time(file, time_pre):
    time_post = datetime.datetime.now()
    load_save_data.writeIntoCSV(m, 'not_' + str(gp_param), file, str((time_post - time_pre).total_seconds()))


def save_time_enc(file, time):
    load_save_data.writeIntoCSV(m, 'not_' + str(gp_param), file, str(time))


def compute_Lx(L, x):
    res = np.zeros(shape=x.shape)
    for i in range(len(L)):
        if L[i] != 0:
            res[i] = L[i] * x[i]

    return res


def aggregator_sum(node, L, lambdaa_k, x):
    time_pre = datetime.datetime.now()
    tmp_sum = np.zeros(shape=lambdaa_k.shape)
    for j in range(len(L)):
        if L[j] != 0:
            tmp_sum += x[j]

    lambdaa_kplus1 = lambdaa_k + tmp_sum
    save_time('agent_sum_' + str(node), time_pre)

    return lambdaa_kplus1


def agent_encrypt(L, x, node, enc_time_nodes):
    x_enc_node = np.empty(shape=x.shape)
    key = 0
    for other_node in range(len(L)):
        if L[other_node] != 0:
            if other_node != node:
                time_pre = datetime.datetime.now()
                #x_enc_node[other_node] = keys_dict['pk_list' + str(node)][key].encryptMatrix(x[other_node])
                enc_time_nodes[other_node] += (datetime.datetime.now() - time_pre).total_seconds()
                key += 1
            else:
                time_pre = datetime.datetime.now()
                #x_enc_node[other_node] = keys_dict['pk_list' + str(node)][-1].encryptMatrix(x[other_node])
                enc_time_nodes[other_node] += (datetime.datetime.now() - time_pre).total_seconds()

    return x_enc_node, enc_time_nodes


def main_decrypt(lambdaa_encrypted):
    lambdaa = np.empty(lambdaa_encrypted.shape)
    for node in range(m):
        time_pre = datetime.datetime.now()
        #lambdaa[node] = keys_dict['msk' + str(node)].decryptMatrix(lambdaa_encrypted[node])
        save_time('agent_dec_' + str(node), time_pre)

    return lambdaa


def train_cPDS_not_enc(n_agent, graph_param, max_iters, L, tau, rho, n, gammas, data, labels, x, q, theta):

    global m, gp_param
    m = n_agent
    gp_param = graph_param

    cPDSs = []
    for j in range(m):
        cPDSs.append(cPDS.cPDS(j, tau, rho, theta[j], gammas[j], data[j], labels[j], q[j], n[j], x[j], L[j]))

    lambdaa = L @ x

    for i in range(max_iters):
        iteration_time_pre = datetime.datetime.now()

        x = np.asarray([cPDSs[node].compute(lambdaa[node]) for node in range(m)])

        # encrypt for node
        lambdaa_kplus1 = np.empty(shape=lambdaa.shape)
        enc_time_nodes = np.zeros(shape=m)
        for node in range(m):
            res = compute_Lx(L[node], x)
            x_enc, enc_time_nodes = agent_encrypt(L[node], res, node, enc_time_nodes)
            lambdaa_kplus1[node] = aggregator_sum(node, L[node], lambdaa[node], res)

        # save agent time
        [save_time_enc('agent_enc_' + str(node), enc_time_nodes[node]) for node in range(m)]

        lambdaa = main_decrypt(lambdaa_kplus1)
        save_time('iteration_time', iteration_time_pre)

    x_return = np.mean(x, axis=0)
    w_cPDS = x_return[:-1]
    b_cPDS = x_return[-1]
    return w_cPDS, b_cPDS
