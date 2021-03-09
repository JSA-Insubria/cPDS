import numpy as np

import util
import load_save_data
import graph_util
import param_tuning
import train_cPDS
import train_cPDS_not_enc
import plotly.graph_objects as go


def load_data():
    #xtrain, ytrain, xtest, ytest = load_save_data.loadData()
    #xtrain, ytrain, xtest, ytest = load_save_data.loadData_extra()
    xtrain, ytrain, xtest, ytest = load_save_data.loadData_extra_mat()

    x_opt, w_SSVM, b_SSVM = load_save_data.loadDataCentralized()
    classes = np.unique(ytrain)
    return xtrain, ytrain, xtest, ytest, classes, x_opt, w_SSVM, b_SSVM


def load_cPDS_parameters(m):
    n, gammas, data, labels = util.federatedData(m, xtrain, ytrain)
    x_init, q = util.initcPDSVar(m, xtrain, gammas, n, data, labels)
    return n, gammas, data, labels, x_init, q


def compute_auc(w_cPDS, b_cPDS, xtest, ytest, classes):
    return util.compute_auc(w_cPDS, b_cPDS, xtest, ytest, classes)


def plot(w_cPDS, b_cPDS, w_SSVM, b_SSVM, xtrain, xtest, ytrain, ytest, classes):
    x1 = np.arange(-2 + np.min(np.concatenate((xtrain[:, 9], xtest[:, 9]), axis=0)),
                   np.max(np.concatenate((xtrain[:, 9], xtest[:, 9]), axis=0)) + 2, 0.1)
    x2_cPDS = (-w_cPDS[9] / w_cPDS[14]) * x1 - b_cPDS / w_cPDS[14]
    x2_SSVM = (-w_SSVM[0] / w_SSVM[1]) * x1 - b_SSVM / w_SSVM[1]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xtrain[ytrain == classes[0], 9], y=xtrain[ytrain == classes[0], 14], mode='markers',
                             name='training class 0'))
    fig.add_trace(go.Scatter(x=xtrain[ytrain == classes[1], 9], y=xtrain[ytrain == classes[1], 14], mode='markers',
                             name='training class 1'))
    fig.add_trace(go.Scatter(x=xtest[ytest == classes[0], 9], y=xtest[ytest == classes[0], 14], mode='markers',
                             name='test class 0'))
    fig.add_trace(go.Scatter(x=xtest[ytest == classes[1], 9], y=xtest[ytest == classes[1], 14], mode='markers',
                             name='test class 1'))
    fig.add_trace(go.Scatter(x=x1, y=x2_SSVM, mode='lines', name='SSVM'))
    fig.add_trace(go.Scatter(x=x1, y=x2_cPDS, mode='lines', name='cPDS'))
    fig.update_layout(title='AUC')
    fig.show()


if __name__ == "__main__":
    xtrain, ytrain, xtest, ytest, classes, x_opt, w_SSVM, b_SSVM = load_data()
    max_iters = 1000

    #gp = [0.2, 0.5, 1]
    gp = [0.1]
    for j in gp:
        #agents = [5, 10, 20, 30]
        agents = [10]
        for i in agents:
            L = graph_util.get_graph(i, j)
            n, gammas, data, labels, x_init, q = load_cPDS_parameters(i)

            # run parameters tuning
            t, tau, rho = param_tuning.tuning(i, 2000, L, xtrain, ytrain, classes)
            theta = t + np.random.uniform(0, 1, i)

            # run cPDS with encryption
            #w_cPDS, b_cPDS = train_cPDS.train_cPDS(i, j, max_iters, L, tau, rho, n, gammas, data, labels, x_init,
                                                   #q, theta)
            #auc = compute_auc(w_cPDS, b_cPDS, xtest, ytest, classes)
            #print('cPDS AUC enc: ', auc)
            #util.save_auc(i, j, auc)

            # run cPDS without encryption
            w_cPDS_not_enc, b_cPDS_not_enc = train_cPDS_not_enc.train_cPDS_not_enc(i, j, max_iters, L, tau, rho, n,
                                                                                   gammas, data, labels, x_init, q, theta)
            auc = compute_auc(w_cPDS_not_enc, b_cPDS_not_enc, xtest, ytest, classes)
            print('cPDS AUC not_enc: ', auc)
            util.save_auc(i, j, auc)

            #plot(w_cPDS_not_enc, b_cPDS_not_enc, w_SSVM, b_SSVM, xtrain, xtest, ytrain, ytest, classes)

        #load_save_data.compute_time_csv(agents, j)
