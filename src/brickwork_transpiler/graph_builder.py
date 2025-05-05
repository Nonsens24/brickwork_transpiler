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


def generate_brickwork_graph_tuples(n_rows, n_cols):
    nodes = [(y, x) for y in range(n_rows) for x in range(n_cols)]
    edges = []

    for y in range(n_rows):
        for x in range(n_cols - 1):
            edges.append(((y, x), (y, x + 1)))

    for x in range(n_cols):
        if x % 8 == 3:
            for y in range(0, n_rows, 2):
                if y + 1 < n_rows:
                    edges.append(((y, x), (y + 1, x)))
                    edges.append(((y, x + 2), (y+1, x + 2)))
        elif x % 8 == 7:
            for y in range(1, n_rows, 2):
                if y + 1 < n_rows:
                    edges.append(((y, x), (y + 1, x)))
                    edges.append(((y, x + 2), (y + 1, x + 2)))

    return {'nodes': nodes, 'edges': edges, 'width': n_cols, 'height': n_rows}