import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def get_graph(m, p):
    nc = 0
    while nc != 1:
        G = nx.erdos_renyi_graph(m, p)
        nc = nx.number_connected_components(G)

    nx.draw(G, with_labels=True)
    plt.show()

    path = 'logs' + os.sep + 'graph'
    if not os.path.exists(path):
        os.makedirs(path)

    plt.savefig(path + os.sep + 'graph_' + str(m) + '_' + str(p) + '.png')

    return nx.adjacency_matrix(G)


def save_graph(m, L):
    path = 'logs' + os.sep + 'graph'
    if not os.path.exists(path):
        os.makedirs(path)

    np.savetxt(path + os.sep + 'graph_' + str(m) + '.txt', L, delimiter=",", fmt="%1.11f")


def read_graph(m):
    path = 'logs' + os.sep + 'graph'
    L = np.loadtxt(path + os.sep + 'graph_' + str(m) + '.txt', delimiter=",")
    return L
