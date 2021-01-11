import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go


def compute_error(xtrain, ytrain, x):
    x_return = np.mean(x, axis=0)
    w_cPDS = x_return[:-1]
    b_cPDS = x_return[-1]

    pred_vals_cPDS = xtrain @ w_cPDS + b_cPDS
    thresholds = np.sort(pred_vals_cPDS, axis=0)
    miss = np.zeros(thresholds.size)
    false_alarm = np.zeros(thresholds.size)
    for i_thr in range(thresholds.size):
        ypred = (pred_vals_cPDS <= thresholds[i_thr]) + 0
        ypred[ypred == 0] = -1
        miss[i_thr] = np.sum(np.logical_and(ypred == -1, ytrain == 1)) / np.sum(ytrain == 1)
        false_alarm[i_thr] = np.sum(np.logical_and(ypred == 1, ytrain == -1)) / np.sum(ytrain == -1)

    return np.abs(np.trapz(false_alarm, 1 - miss))


def plot_error(error, max_iter):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.arange(max_iter)+1, y=error, mode='lines+markers', name='error'))
    fig.update_layout(title='Error', xaxis_title='Iter', yaxis_title='AUC')
    fig.show()


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

    print("w_SSVM: ", w_SSVM)
    print("w_cPDS: ", w_cPDS)
    print("b_SSVM: ", b_SSVM)
    print("b_cPDS: ", b_cPDS)

    x1 = np.arange(-2 + np.min(np.concatenate((xtrain[:, 0], xtest[:, 0]), axis=0)),
                   np.max(np.concatenate((xtrain[:, 0], xtest[:, 0]), axis=0)) + 2, 0.1)
    x2_cPDS = (-w_cPDS[0] / w_cPDS[1]) * x1 - b_cPDS / w_cPDS[1]
    x2_SSVM = (-w_SSVM[0] / w_SSVM[1]) * x1 - b_SSVM / w_SSVM[1]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xtrain[ytrain == -1, 0], y=xtrain[ytrain == -1, 1], mode='markers', name='training class -1'))
    fig.add_trace(go.Scatter(x=xtrain[ytrain == 1, 0], y=xtrain[ytrain == 1, 1], mode='markers', name='training class +1'))
    fig.add_trace(go.Scatter(x=xtest[ytest == -1, 0], y=xtest[ytest == -1, 1], mode='markers', name='test class -1'))
    fig.add_trace(go.Scatter(x=xtest[ytest == 1, 0], y=xtest[ytest == 1, 1], mode='markers', name='test class +1'))
    fig.add_trace(go.Scatter(x=x1, y=x2_SSVM, mode='lines', name='SSVM'))
    fig.add_trace(go.Scatter(x=x1, y=x2_cPDS, mode='lines', name='cPDS'))
    fig.update_layout(title='AUC')
    fig.show()

    # calculate AUC by SSVM
    pred_vals_SSVM = (xtest @ w_SSVM) + b_SSVM
    thresholds = np.sort(pred_vals_SSVM, axis=0)
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
    thresholds = np.sort(pred_vals_cPDS, axis=0)
    miss = np.zeros(thresholds.size)
    false_alarm = np.zeros(thresholds.size)
    for i_thr in range(thresholds.size):
        ypred = (pred_vals_cPDS <= thresholds[i_thr]) + 0
        ypred[ypred == 0] = -1
        miss[i_thr] = np.sum(np.logical_and(ypred == -1, ytest == 1)) / np.sum(ytest == 1)
        false_alarm[i_thr] = np.sum(np.logical_and(ypred == 1, ytest == -1)) / np.sum(ytest == -1)

    AUC_cPDS = np.abs(np.trapz(false_alarm, 1 - miss))
    print("AUC cPDS: ", AUC_cPDS)