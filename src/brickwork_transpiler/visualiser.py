import networkx as nx
from matplotlib import pyplot as plt


def plot_graph(G):

    pos = {node: (node[1], -node[0]) for node in G.nodes()}
    plt.figure(figsize=(5,5))
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=600)
    plt.axis('equal'); plt.show()


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
    header = ["Qubit \\ Column"] + [f"Column {c}" for c in range(num_cols)]

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