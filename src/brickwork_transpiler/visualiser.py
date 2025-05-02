import networkx as nx
from matplotlib import pyplot as plt

def create_tuple_node_graph(rows, cols):
    G = nx.Graph()
    for i in range(rows):
        for j in range(cols):
            G.add_node((i, j))
    for i in range(rows):
        for j in range(cols-1):
            G.add_edge((i, j), (i, j+1))
    return G

def plot_graph(G):
    pos = {node: (node[1], -node[0]) for node in G.nodes()}
    plt.figure(figsize=(5,5))
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=600)
    plt.axis('equal'); plt.show()