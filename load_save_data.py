import os
import pandas as pd
import numpy as np

from scipy.io import loadmat


# ------- load data -------
def loadData():
    mat = loadmat('data' + os.sep + 'dataset3_10000_20.mat', squeeze_me=True)
    xtrain = mat['xtrain']
    ytrain = mat['ytrain']
    xtest = mat['xtest']
    ytest = mat['ytest']
    random_idx = np.random.rand(xtrain.shape[0]).argsort()
    np.take(xtrain, random_idx, axis=0, out=xtrain)
    np.take(ytrain, random_idx, axis=0, out=ytrain)
    return xtrain, ytrain, xtest, ytest


def loadDataCentralized():
    # Load optimal solution computed by centralized approach
    mat = loadmat('data' + os.sep + 'sSVM_gurobi.mat', squeeze_me=True)
    x_opt = mat['theta_opt_SSVM']
    # x_opt = np.array([0.7738, 0.7131, 0.0000, 0.0433, -0.0112, 0.0462])
    w_SSVM = x_opt[:-1]
    b_SSVM = x_opt[-1]
    return x_opt, w_SSVM, b_SSVM


# ------- save logs -------
def writeIntoCSV(m, folder_name, file_name, row):
    path = 'logs' + os.sep + folder_name + os.sep + str(m) + '_agents'
    if not os.path.exists(path):
        os.makedirs(path)

    with open(path + os.sep + file_name + '.csv', 'a') as fd:
        fd.write(row + '\n')


def compute_time_csv(m, graph_param):
    folder = ['enc_' + str(graph_param), 'not_' + str(graph_param)]
    for file in folder:
        path_file = 'logs' + os.sep + file + os.sep
        first_line = "iteration n°,"
        mean = pd.DataFrame()
        for i in m:
            first_line = first_line + str(i) + " Agents,,,,,,,,,,"
            path = path_file + str(i) + "_agents" + os.sep
            data_enc = {}
            data_sum = {}
            data_dec = {}
            for j in range(i):
                data_enc[j] = pd.read_csv(path + "agent_enc_" + str(j) + ".csv", header=None)
                data_sum[j] = pd.read_csv(path + "agent_sum_" + str(j) + ".csv", header=None)
                data_dec[j] = pd.read_csv(path + "agent_dec_" + str(j) + ".csv", header=None)

            df_agent_enc = pd.concat(data_enc, axis=1)
            df_agent_enc_max = df_agent_enc.max(axis=1).to_frame()
            df_agent_enc_max.columns = ['Encryption time max - Agents (s)']
            df_agent_enc_min = df_agent_enc.min(axis=1).to_frame()
            df_agent_enc_min.columns = ['Encryption time min - Agents (s)']
            df_agent_enc_mean = df_agent_enc.mean(axis=1).to_frame()
            df_agent_enc_mean.columns = ['Encryption time mean - Agents (s)']

            df_agent_sum = pd.concat(data_sum, axis=1)
            df_agent_sum_max = df_agent_sum.max(axis=1).to_frame()
            df_agent_sum_max.columns = ['Sum time max - Agents (s)']
            df_agent_sum_min = df_agent_sum.min(axis=1).to_frame()
            df_agent_sum_min.columns = ['Sum time min - Agents (s)']
            df_agent_sum_mean = df_agent_sum.mean(axis=1).to_frame()
            df_agent_sum_mean.columns = ['Sum time mean - Agents (s)']

            df_agent_dec = pd.concat(data_dec, axis=1)
            df_agent_dec_max = df_agent_dec.max(axis=1).to_frame()
            df_agent_dec_max.columns = ['Decryption time max - Agents (s)']
            df_agent_dec_min = df_agent_dec.min(axis=1).to_frame()
            df_agent_dec_min.columns = ['Decryption time min - Agents (s)']
            df_agent_dec_mean = df_agent_dec.mean(axis=1).to_frame()
            df_agent_dec_mean.columns = ['Decryption time mean - Agents (s)']

            iteration = pd.read_csv(path + "iteration_time.csv", header=None)
            iteration.columns = ['Iteration time (s)']

            mean = pd.concat([mean,
                              df_agent_enc_max, df_agent_enc_min, df_agent_enc_mean,
                              df_agent_sum_max, df_agent_sum_min, df_agent_sum_mean,
                              df_agent_dec_max, df_agent_dec_min, df_agent_dec_mean,
                              iteration], axis=1)

        with open(path_file + "time.csv", 'w') as fd:
            fd.write(first_line + '\n')
        mean = mean.round(3)
        mean.to_csv(path_file + "time.csv", mode='a')
