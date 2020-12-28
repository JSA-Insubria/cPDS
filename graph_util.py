import networkx as nx
import matplotlib.pyplot as plt


def get_graph(m, p):
    nc = 0
    while nc != 1:
        G = nx.erdos_renyi_graph(m, p)
        nc = nx.number_connected_components(G)

    nx.draw(G, with_labels=True)
    plt.show()
    return nx.adjacency_matrix(G)
