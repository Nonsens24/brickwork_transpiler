from fractions import Fraction
import matplotlib.pyplot as plt
import networkx as nx


def plot_graph(G, cell_size=0.5, margin=0.5):
    """
    Draw G so that each adjacent node pair is spaced by `cell_size` inches.

    Parameters
    ----------
    G : networkx.Graph
        Graph whose nodes are (row, col) tuples.
    cell_size : float, optional
        Size in inches of each grid cell (distance between adjacent nodes).
    margin : float, optional
        Extra padding around the grid, in grid‐cells.
    """
    # 1) figure out grid dimensions
    rows = [r for (r, _) in G.nodes()]
    cols = [c for (_, c) in G.nodes()]
    n_rows = max(rows) + 1
    n_cols = max(cols) + 1

    # 2) build pos mapping (x = col, y = row)
    pos = {(r, c): (c, r) for (r, c) in G.nodes()}

    # 3) compute figure size so each cell is cell_size inches
    fig_w = cell_size * (n_cols + 2 * margin)
    fig_h = cell_size * (n_rows + 2 * margin)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # 4) draw with constant node spacing
    nx.draw(
        G,
        pos=pos,
        with_labels=True,
        node_color='lightblue',
        node_size=600,
        font_size=8,
        ax=ax
    )

    # 5) square aspect & invert so row=0 is at top
    ax.set_aspect('equal')
    ax.invert_yaxis()

    # 6) pad the axes so nodes aren’t flush to the border
    ax.set_xlim(-margin, n_cols - 1 + margin)
    ax.set_ylim(n_rows - 1 + margin, -margin)

    plt.tight_layout()
    plt.show()


def print_matrix(matrix):
    """
    Print the qubit-major matrix as a table:
      rows = qubits, columns = grouped stages
    """
    if not matrix:
        print("No data to display.")
        return
    num_qubits = len(matrix)
    num_cols   = len(matrix[0])

    # build header
    header = ["Qubit \\ Gate Group"] + [f"Group {c}" for c in range(num_cols)]

    # build rows
    rows = []
    for q in range(num_qubits):
        row = [f"Qubit {q}"]
        for c in range(num_cols):
            ops = matrix[q][c]
            # convert any Gate obj → its name
            row.append(", ".join(op.name if hasattr(op, "name") else str(op)
                                 for op in ops)
                       if ops else "-")
        rows.append(row)

    # compute column widths
    cols = [header] + rows
    col_widths = [max(len(r[i]) for r in cols) for i in range(len(header))]

    # print header
    sep_line = "|-" + "-|-".join("-" * w for w in col_widths) + "-|"
    header_line = "| " + " | ".join(header[i].ljust(col_widths[i])
                                   for i in range(len(header))) + " |"
    print(header_line)
    print(sep_line)

    # print each row
    for row in rows:
        print("| " + " | ".join(row[i].ljust(col_widths[i])
                                for i in range(len(row))) + " |")


import networkx as nx
import matplotlib.pyplot as plt

def plot_brickwork_graph_from_pattern(bw_pattern,
                                      show_angles=True,
                                      node_size=600,
                                      node_color='skyblue',
                                      edge_color='gray',
                                      font_size=8,
                                      figsize=None,
                                      cell_size=0.5,
                                      margin=0.5,
                                      title="Brickwork Graph"):

    # 1. Extract graph structure
    graph = bw_pattern.get_graph()
    nodes, edges = graph[0], graph[1]

    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    # 2. Get angles if desired
    angles = {}
    if show_angles:
        angles = bw_pattern.get_angles()  # expects {(r,c): float, ...}

    # 3. Figure out grid size
    rows = [r for (r, _) in nodes]
    cols = [c for (_, c) in nodes]
    n_rows = max(rows) + 1
    n_cols = max(cols) + 1

    # 4. Compute positions and optional figure size
    pos = {(r, c): (c, r) for (r, c) in nodes}
    if figsize is None:
        figsize = (cell_size * (n_cols + 2 * margin),
                   cell_size * (n_rows + 2 * margin))

    fig, ax = plt.subplots(figsize=figsize)

    # 5. Draw nodes and edges
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=edge_color)
    nx.draw_networkx_nodes(G, pos,
                           ax=ax,
                           node_size=node_size,
                           node_color=node_color)

    # 6. Build and draw labels: “(r,c)\nθπ”
    labels = {}
    for node in nodes:
        r, c = node
        if show_angles and node in angles:
            # angles are in multiples of π; format to 2 decimals
            θ = angles[node]
            labels[node] = f"({r},{c})\n{θ:.2f}π"
        else:
            labels[node] = f"({r},{c})"
    nx.draw_networkx_labels(G, pos,
                            labels=labels,
                            font_size=font_size,
                            ax=ax)

    # 7. Final plot tweaks
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.set_xlim(-margin, n_cols - 1 + margin)
    ax.set_ylim(n_rows - 1 + margin, -margin)
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()



def format_angle(angle, denom=4):
    """Return a string representing angle as a multiple of π/denom."""
    frac = Fraction(angle / np.pi).limit_denominator(denom)
    n = frac.numerator
    d = frac.denominator

    if n == 0:
        return "0"
    elif abs(n) == 1 and d == 1:
        return "π" if n > 0 else "-π"
    elif d == 1:
        return f"{n}π"
    elif abs(n) == 1:
        return f"π/{d}" if n > 0 else f"-π/{d}"
    else:
        return f"{n}π/{d}"