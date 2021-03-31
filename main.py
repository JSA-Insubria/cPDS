import os
import numpy as np

import util
import load_save_data
import graph_util
import param_tuning
import train_cPDS
import train_cPDS_not_enc
import matplotlib.pyplot as plt


def load_data():
    xtrain, ytrain, xtest, ytest = load_save_data.loadData()
    x_opt, w_SSVM, b_SSVM = load_save_data.loadDataCentralized()
    classes = np.unique(ytrain)
    return xtrain, ytrain, xtest, ytest, classes, x_opt, w_SSVM, b_SSVM


def load_cPDS_parameters(m):
    n, gammas, data, labels = util.federatedData(m, xtrain, ytrain)
    x_init, q = util.initcPDSVar(m, xtrain, gammas, n, data, labels)
    return n, gammas, data, labels, x_init, q


def plot(m, p, fpr, tpr, roc_auc, fpr1, tpr1, roc_auc1):
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label='AUC enc= %0.2f' % roc_auc)
    plt.plot(fpr1, tpr1, 'g', label='AUC= %0.2f' % roc_auc1)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')

    path = 'logs' + os.sep + 'graph' + os.sep + 'auc_graph'
    if not os.path.exists(path):
        os.makedirs(path)

    plt.savefig(path + os.sep + 'auc_graph_' + str(m) + '_' + str(p) + '.png')
    plt.show()


if __name__ == "__main__":
    xtrain, ytrain, xtest, ytest, classes, x_opt, w_SSVM, b_SSVM = load_data()
    max_iters = 100

    gp = [0.2, 0.5, 1]
    for j in gp:
        agents = [5, 10, 20, 30]
        for i in agents:
            L = graph_util.get_graph(i, j)
            n, gammas, data, labels, x_init, q = load_cPDS_parameters(i)

            # run parameters tuning
            t, tau, rho = param_tuning.tuning(i, 200, L, xtrain, ytrain, classes)
            theta = t + np.random.uniform(0, 1, i)

            # run cPDS without encryption
            w_cPDS_not_enc, b_cPDS_not_enc = train_cPDS_not_enc.train_cPDS_not_enc(i, j, max_iters, L, tau, rho, n,
                                                                                   gammas, data, labels, x_init, q,
                                                                                   theta)
            auc1, fpr1, tpr1 = util.compute_auc(w_cPDS_not_enc, b_cPDS_not_enc, xtest, ytest, classes)
            print('cPDS AUC not_enc: ', auc1)
            util.save_auc(i, j, auc1)

            # run cPDS with encryption
            w_cPDS, b_cPDS = train_cPDS.train_cPDS(i, j, max_iters, L, tau, rho, n, gammas, data, labels, x_init,
                                                   q, theta)
            auc, fpr, tpr = util.compute_auc(w_cPDS, b_cPDS, xtest, ytest, classes)
            print('cPDS AUC enc: ', auc)
            util.save_auc(i, j, auc)

            plot(i, j, fpr, tpr, auc, fpr1, tpr1, auc1)

        load_save_data.compute_time_csv(agents, j)


'''# compute auc mean
def save_mean_auc(m, gp_param, auc_cPDS):
    with open('logs' + os.sep + "auc_mean.csv", 'a') as fd:
        fd.write(str(m) + ',' + str(gp_param) + ',' + str(auc_cPDS) + '\n')


if __name__ == "__main__":
    xtrain, ytrain, xtest, ytest, classes, x_opt, w_SSVM, b_SSVM = load_data()
    max_iters = 100

    gp = [0.2, 0.5, 1]
    for j in gp:
        agents = [5, 10, 20, 30]
        for i in agents:
            for time in range(10):

                L = graph_util.get_graph(i, j)
                n, gammas, data, labels, x_init, q = load_cPDS_parameters(i)

                # run parameters tuning
                t, tau, rho = param_tuning.tuning(i, 200, L, xtrain, ytrain, classes)
                theta = t + np.random.uniform(0, 1, i)

                # run cPDS without encryption
                w_cPDS_not_enc, b_cPDS_not_enc = train_cPDS_not_enc.train_cPDS_not_enc(i, j, max_iters, L, tau, rho, n,
                                                                                       gammas, data, labels, x_init, q,
                                                                                       theta)
                auc1, fpr1, tpr1 = util.compute_auc(w_cPDS_not_enc, b_cPDS_not_enc, xtest, ytest, classes)
                print('cPDS AUC not_enc: ', auc1)

                save_mean_auc(i, j, auc1)'''
