import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def local_degree(P, eps_deg):
    n = P.shape[0]
    W = np.zeros((n, n))
    deg = P.sum(axis=1)
    for i in range(0, n):
        for j in range(0, n):
            if P[i, j] == 1:
                W[i, j] = 1.0/(max(deg[i], deg[j])+eps_deg)
    W = np.diag(np.ones(n)-(W.sum(axis=1))) + W
    W = (W+np.transpose(W)+2*np.eye(n, n))/4
    return W


def get_graph(m, p):
    nc = 0
    while nc != 1:
        G = nx.erdos_renyi_graph(m, p)
        nc = nx.number_connected_components(G)

    nx.draw(G, with_labels=True)

    path = 'logs' + os.sep + 'graph'
    if not os.path.exists(path):
        os.makedirs(path)

    plt.savefig(path + os.sep + 'graph_' + str(m) + '_' + str(p) + '.png')
    plt.show()

    adj = nx.adjacency_matrix(G)
    return np.eye(m) - local_degree(adj, 0.1)

