import networkx as nx

# Generates a single brick row graph
def create_tuple_node_graph(rows, cols):

    G = nx.Graph()

    for i in range(rows):
        for j in range(cols):
            G.add_node((i, j))
    for i in range(rows):
        for j in range(cols-1):
            G.add_edge((i, j), (i, j+1))

    return G
