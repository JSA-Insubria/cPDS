import numpy as np

import util
import train_cPDS_not_enc


def tuning(n_agent, max_iters, L, xtrain, ytrain, xtest, ytest, classes):
    ts = [1, 1, 1, 5, 5, 5, 10, 10, 10, 25, 25, 25]
    taus = [0.1, 1, 10, 0.1, 1, 10, 0.1, 1, 10, 0.1, 1, 10]
    rhos = [0.1, 1, 10, 0.1, 1, 10, 0.1, 1, 10, 0.1, 1, 10]

    #xtrain, xtest, ytrain, ytest = train_test_split(data_train, labels)
    n, gammas, data, labels = util.federatedData(n_agent, xtrain, ytrain)
    x_init, q = util.initcPDSVar(n_agent, xtrain, gammas, n, data, labels)

    auc = np.zeros(len(ts))
    for param_idx in range(len(ts)):
        t = ts[param_idx]
        tau = taus[param_idx]
        rho = rhos[param_idx]

        theta = t + np.random.uniform(0, 1, n_agent)

        w_cPDS, b_cPDS = train_cPDS_not_enc.train_cPDS_not_enc(n_agent, 'tuning', max_iters, L, tau, rho, n, gammas, data, labels, x_init, q, theta)
        auc[param_idx], _, _ = util.compute_auc(w_cPDS, b_cPDS, xtest, ytest, classes)
        print('AUC: ', auc[param_idx], ', t: ', ts[param_idx], ', tau: ', taus[param_idx], ', rho: ', rhos[param_idx])

    max_auc = auc.argmax()
    print('Optimal cPDS parameters: ', 't: ', ts[max_auc], ', tau: ', taus[max_auc], ', rho: ', rhos[max_auc],
          '. Maximum training AUC=', auc[max_auc])
    return ts[max_auc], taus[max_auc], rhos[max_auc]