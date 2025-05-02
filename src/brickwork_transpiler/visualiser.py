import networkx as nx
from matplotlib import pyplot as plt


def plot_graph(G):
    pos = {node: (node[1], -node[0]) for node in G.nodes()}
    plt.figure(figsize=(5,5))
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=600)
    plt.axis('equal'); plt.show()