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


def generate_brickwork_graph_from_instruction_matrix(matrix):
    """
    Brickwork graph on n_rows × (n_cols-1) sites,
    with nodes indexed as (row, col), skipping column 0.

    Returns a dict with:
      • 'nodes': list of (row, col) pairs, col in 0..n_cols-1
      • 'edges': list of ((row1, col1), (row2, col2))
      • 'width':  n_cols
      • 'height': n_rows
    """
    n_rows = len(matrix)                          # number of rows
    n_cols = len(matrix[0]) if matrix else 0      # number of columns
    n_cols = (n_cols * 4) + 1                     # scale to bricks + output column (+correction) +1+1
    # start_col = 1                                 # need this since mod 8 doesn't account for 0

    # all sites (row, col)
    nodes = [(row, col)
             for row in range(n_rows)
             for col in range(n_cols)]

    edges = []

    # 1) horizontal chain: (row, col) — (row, col+1)
    for row in range(n_rows):
        for col in range(n_cols - 1):
            edges.append(((row, col), (row, col + 1)))

    # 2) brickwork vertical links with proper bounds checks
    for col in range(n_cols):
        if (col+1) % 8 == 3:    # col + 1 since indexing has to start at 0 but mod 8 has to
            # connect even‐indexed rows at this column (and at col+2 if in range)
            for row in range(0, n_rows - 1, 2):
                edges.append(((row, col),     (row + 1, col)))
                if col + 2 < n_cols:
                    edges.append(((row, col + 2), (row + 1, col + 2)))
        elif (col+1) % 8 == 7:
            # connect odd‐indexed rows at this column (and at col+2 if in range)
            for row in range(1, n_rows - 1, 2):
                edges.append(((row, col),     (row + 1, col)))
                if col + 2 < n_cols:
                    edges.append(((row, col + 2), (row + 1, col + 2)))

    return {
        'nodes':  nodes,
        'edges':  edges,
        'width':  n_cols,
        'height': n_rows
    }



def to_networkx_graph(brickwork):
    G = nx.Graph()
    G.add_nodes_from(brickwork['nodes'])
    G.add_edges_from(brickwork['edges'])

    return G