import numpy as np
import matplotlib.pyplot as plt


def plot(residuals_x, x, xtrain, xtest, ytrain, ytest, w_SSVM, b_SSVM):
    # plot residuals
    plt.plot(residuals_x)
    plt.ylabel('Residuals x')
    plt.xlabel('Iterations')
    plt.show()
    # plot residuals on the log scale
    plt.semilogy(residuals_x)
    plt.ylabel('Residuals x in logscale')
    plt.xlabel('Iterations')
    plt.show()

    x_return = np.mean(x, axis=0)
    w_cPDS = x_return[:-1]
    b_cPDS = x_return[-1]

    x1 = np.arange(-2 + np.min(np.concatenate((xtrain[:, 0], xtest[:, 0]), axis=0)),
                   np.max(np.concatenate((xtrain[:, 0], xtest[:, 0]), axis=0)) + 2, 0.1)
    x2_cPDS = (-w_cPDS[0] / w_cPDS[1]) * x1 - b_cPDS / w_cPDS[1]
    x2_SSVM = (-w_SSVM[0] / w_SSVM[1]) * x1 - b_SSVM / w_SSVM[1]

    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(111)
    ax.scatter(xtrain[ytrain == -1, 0], xtrain[ytrain == -1, 1], c='red', marker='o', label='training class -1')
    ax.scatter(xtrain[ytrain == 1, 0], xtrain[ytrain == 1, 1], c='blue', marker='x', label='training class +1')
    ax.scatter(xtest[ytest == -1, 0], xtest[ytest == -1, 1], c='green', marker='s', label='test class -1')
    ax.scatter(xtest[ytest == 1, 0], xtest[ytest == 1, 1], c='yellow', marker='v', label='test class +1')
    ax.plot(x1, x2_SSVM, linewidth=2, markersize=12, label='SSVM')
    ax.plot(x1, x2_cPDS, linewidth=2, markersize=12, label='cPDS')
    ax.plot()
    plt.legend(loc='lower left')
    plt.grid()
    plt.show()

    # calculate AUC by SSVM
    pred_vals_SSVM = (xtest @ w_SSVM) + b_SSVM
    thresholds = np.sort(pred_vals_SSVM)
    miss = np.zeros(thresholds.size)
    false_alarm = np.zeros(thresholds.size)
    for i_thr in range(len(thresholds)):
        ypred = (pred_vals_SSVM <= thresholds[i_thr]) + 0
        ypred[ypred == 0] = -1
        miss[i_thr] = np.sum(np.logical_and(ypred == -1, ytest == 1)) / np.sum(ytest == 1)
        false_alarm[i_thr] = np.sum(np.logical_and(ypred == 1, ytest == -1)) / np.sum(ytest == -1)

    AUC_SSVM = np.abs(np.trapz(false_alarm, 1 - miss))
    print("AUC SSVM: ", AUC_SSVM)

    # calculate AUC by cPDS
    pred_vals_cPDS = xtest @ w_cPDS + b_cPDS
    thresholds = np.sort(pred_vals_cPDS)
    miss = np.zeros(thresholds.size)
    false_alarm = np.zeros(thresholds.size)
    for i_thr in range(thresholds.size):
        ypred = (pred_vals_cPDS <= thresholds[i_thr]) + 0
        ypred[ypred == 0] = -1
        miss[i_thr] = np.sum(np.logical_and(ypred == -1, ytest == 1)) / np.sum(ytest == 1)
        false_alarm[i_thr] = np.sum(np.logical_and(ypred == 1, ytest == -1)) / np.sum(ytest == -1)

    AUC_cPDS = np.abs(np.trapz(false_alarm, 1 - miss))
    print("AUC cPDS: ", AUC_cPDS)