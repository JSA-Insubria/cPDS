import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.graph_objects as go


# --- util ---
def loadData_extra():
    dataset = pd.read_csv('data/HIGGS.csv', nrows=10000)
    y = dataset.iloc[:, 0].to_numpy()
    x = dataset.iloc[:, 1:].to_numpy()

    x = StandardScaler().fit_transform(x)
    x = PCA(n_components=2).fit_transform(x)

    xtrain, xtest, ytrain, ytest = train_test_split(x, y)
    return xtrain, ytrain, xtest, ytest


# --- plot ---
def plot_extra(x, xtrain, xtest, ytrain, ytest):
    x_return = np.mean(x, axis=0)
    w_cPDS = x_return[:-1]
    b_cPDS = x_return[-1]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=xtrain[ytrain == 0, 0], y=xtrain[ytrain == 0, 1], mode='markers', name='training class 0'))
    fig.add_trace(
        go.Scatter(x=xtrain[ytrain == 1, 0], y=xtrain[ytrain == 1, 1], mode='markers', name='training class +1'))
    fig.add_trace(go.Scatter(x=xtest[ytest == 0, 0], y=xtest[ytest == 0, 1], mode='markers', name='test class 0'))
    fig.add_trace(go.Scatter(x=xtest[ytest == 1, 0], y=xtest[ytest == 1, 1], mode='markers', name='test class +1'))
    fig.show()


    # calculate AUC by cPDS
    pred_vals_cPDS = xtest @ w_cPDS + b_cPDS
    thresholds = np.sort(pred_vals_cPDS, axis=0)
    miss = np.zeros(thresholds.size)
    false_alarm = np.zeros(thresholds.size)
    for i_thr in range(thresholds.size):
        ypred = (pred_vals_cPDS <= thresholds[i_thr]) + 0
        ypred[ypred == 0] = 0
        miss[i_thr] = np.sum(np.logical_and(ypred == 0, ytest == 1)) / np.sum(ytest == 1)
        false_alarm[i_thr] = np.sum(np.logical_and(ypred == 1, ytest == 0)) / np.sum(ytest == 0)

    AUC_cPDS = np.abs(np.trapz(false_alarm, 1 - miss))
    print("AUC cPDS: ", AUC_cPDS)


# --- main ---
def compute_error_extra(xtrain, ytrain, x):
    x_return = np.mean(x, axis=0)
    w_cPDS = x_return[:-1]
    b_cPDS = x_return[-1]

    pred_vals_cPDS = xtrain @ w_cPDS + b_cPDS
    thresholds = np.sort(pred_vals_cPDS, axis=0)
    miss = np.zeros(thresholds.size)
    false_alarm = np.zeros(thresholds.size)
    for i_thr in range(thresholds.size):
        ypred = (pred_vals_cPDS <= thresholds[i_thr]) + 0
        ypred[ypred == 0] = 0
        miss[i_thr] = np.sum(np.logical_and(ypred == 0, ytrain == 1)) / np.sum(ytrain == 1)
        false_alarm[i_thr] = np.sum(np.logical_and(ypred == 1, ytrain == 0)) / np.sum(ytrain == 0)

    return np.abs(np.trapz(false_alarm, 1 - miss))