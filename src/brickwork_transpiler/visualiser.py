from fractions import Fraction
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


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
import matplotlib.patches as mpatches

def plot_brickwork_graph_from_pattern(
        bw_pattern,
        show_angles: bool = True,
        node_size: float = 600,
        node_color: str = 'skyblue',
        edge_color: str = 'gray',
        font_size: int = 8,
        figsize: tuple = None,
        cell_size: float = 0.5,
        node_spacing: float = 1.2,     # ← new: how many “cell_sizes” between nodes
        margin: float = 0.5,
        title: str = "Brickwork Graph",
        use_node_colours: bool = False,
        node_colours: dict = None
    ):
    if node_colours is None:
        node_colours = {}

    # 1. Build graph
    nodes, edges = bw_pattern.get_graph()
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    # 2. Angles
    angles = bw_pattern.get_angles() if show_angles else {}

    # 3. Grid dims
    rows = [r for (r, _) in nodes]
    cols = [c for (_, c) in nodes]
    n_rows, n_cols = max(rows) + 1, max(cols) + 1

    # 4. Figure & layout
    legend_height = 0.5   # inches for the legend row
    if figsize is None:
        # total spacing per “cell” in inches
        spacing = cell_size * node_spacing
        total_width  = spacing * (n_cols + 2*margin)
        total_height = spacing * (n_rows + 2*margin) + legend_height
        figsize = (total_width, total_height)

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(
        2, 1,
        height_ratios=[n_rows*cell_size*node_spacing, legend_height],
        hspace=0.0
    )
    ax        = fig.add_subplot(gs[0, 0])
    legend_ax = fig.add_subplot(gs[1, 0])
    legend_ax.axis('off')

    # 5. Compute positions with uniform spacing
    pos = {
        (r, c): (c * node_spacing, r * node_spacing)
        for (r, c) in nodes
    }

    # 6. Draw edges
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=edge_color)

    # 7. Draw nodes
    colours = (
        [node_colours.get(node, node_color) for node in G.nodes()]
        if use_node_colours else
        node_color
    )
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_size=node_size,
        node_color=colours
    )

    # 8. Draw labels
    labels = {}
    for node in nodes:
        r, c = node
        if show_angles and node in angles:
            θ = angles[node]
            labels[node] = f"({r},{c})\n{θ:.2f}π"
        else:
            labels[node] = f"({r},{c})"
    nx.draw_networkx_labels(
        G, pos,
        labels=labels,
        font_size=font_size,
        ax=ax
    )

    # 9. Final tweaks on main axes
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.set_xlim(-margin * node_spacing,
                (n_cols - 1 + margin) * node_spacing)
    ax.set_ylim((n_rows - 1 + margin) * node_spacing,
                -margin * node_spacing)
    ax.set_title(title)
    ax.axis('off')

    # 10. Legend below
    if use_node_colours:
        handles = [
            mpatches.Patch(color='lightcoral',  label='CX gate - ⊕ top'),
            mpatches.Patch(color='lightsalmon', label='CX gate - ⊕ bot'),   # ● ctrl / ⊕ targ
            mpatches.Patch(color='lightblue',   label='Identity'),
            mpatches.Patch(color='lightgreen',  label='Euler rotation'),
            mpatches.Patch(color='skyblue',     label='Output qubits'),
        ]
        legend_ax.legend(
            handles=handles,
            title="Gate types",
            loc='center',
            ncol=4,
            frameon=True
        )

    plt.show()
    #Save vector graphics img
    fig.savefig(f"images/graphs/{title}.pdf", format="pdf", bbox_inches="tight")
    fig.savefig(f"images/graphs/{title}.png", format="png", dpi=300, bbox_inches="tight")

# plot_graphix_pattern_scalar_index.py

# plot_graphix_pattern_scalar_index.py

def plot_graphix_noise_graph(
        pattern,
        show_angles: bool = True,
        node_size: float = 600,
        edge_color: str = 'gray',
        font_size: int = 8,
        figsize: tuple = None,
        cell_size: float = 0.5,
        node_spacing: float = 1.2,
        margin: float = 0.5,
        title: str = "Pauli Noise Graph",
        cmap_name: str = 'viridis',  # dark-blue to dark-red continuous spectrum
        vmin: float = None,
        vmax: float = None,
        show_colorbar: bool = True,
        save: bool = False,
        filename_prefix: str = "pattern_graph"
):
    """
    Plot a Graphix Pattern object, using measurement angles to color-code nodes as a continuous heatmap.

    The colormap 'jet' spans dark blue → light blue → yellow → light red → dark red.
    The colorbar shows only start/end ticks labeled in π units for clarity.
    """
    import networkx as nx
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.colors import Normalize

    # Build graph
    nodes, edges = pattern.get_graph()
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    # Measurement angles (multiples of π)
    angles = pattern.get_angles()

    # Prepare heatmap values and normalization
    values = [angles.get(n, 0.0) for n in G.nodes()]
    vmin_val = min(values) if vmin is None else vmin
    vmax_val = max(values) if vmax is None else vmax
    norm = Normalize(vmin=vmin_val, vmax=vmax_val)
    cmap = cm.get_cmap(cmap_name)
    node_colors = [cmap(norm(val)) for val in values]

    # Layout: compute layers
    depth, layers = pattern.get_layers()
    node_layer = {n: layer for layer, nset in layers.items() for n in nset}
    for o in getattr(pattern, 'output_nodes', []):
        node_layer[o] = depth + 1
    max_width = max(
        len([n for n, l in node_layer.items() if l == layer])
        for layer in range(depth + 2)
    )

    # Setup figure and gridspec
    legend_height = 0.6 if show_colorbar else 0.0
    if figsize is None:
        total_width = cell_size * node_spacing * (depth + 1 + 2 * margin)
        total_height = cell_size * node_spacing * (max_width - 1 + 2 * margin) + legend_height
        figsize = (total_width, total_height)
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(
        2, 1,
        height_ratios=[
            cell_size * node_spacing * (max_width - 1 + 2 * margin),
            legend_height
        ],
        hspace=0.05
    )
    ax = fig.add_subplot(gs[0, 0])
    cax = fig.add_subplot(gs[1, 0]) if show_colorbar else None

    # Compute positions
    pos = {}
    for layer in range(depth + 2):
        layer_nodes = sorted([n for n, l in node_layer.items() if l == layer])
        for idx, n in enumerate(layer_nodes):
            pos[n] = (layer * node_spacing * cell_size,
                       idx * node_spacing * cell_size)

    # Draw edges and nodes
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=edge_color)
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_size=node_size,
        node_color=node_colors
    )

    # Draw labels
    labels = {}
    for n in G.nodes():
        if show_angles and n in angles:
            labels[n] = f"{n}\n{angles[n]:.2f}π"
        else:
            labels[n] = str(n)
    nx.draw_networkx_labels(G, pos, labels, font_size=font_size, ax=ax)

    # Axis tweaks
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.set_xlim(-margin * node_spacing * cell_size,
                (depth + margin + 1) * node_spacing * cell_size)
    ax.set_ylim(
        (max_width - 1 + margin) * node_spacing * cell_size,
        -margin * node_spacing * cell_size
    )
    ax.set_title(title)
    ax.axis('off')

    # Colorbar: continuous with start/end ticks
    if show_colorbar and cax:
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array(values)
        cbar = plt.colorbar(sm, cax=cax, orientation='horizontal')
        cbar.set_label('Measurement angle noise', labelpad=8)
        # Show only start and end
        cbar.set_ticks([vmin_val, vmax_val])
        cbar.set_ticklabels([f"{vmin_val:.2f}π", f"{vmax_val:.2f}π"])
        # Ensure axis line is visible
        cbar.ax.xaxis.set_ticks_position('bottom')
        cbar.ax.xaxis.set_label_position('bottom')

    # Save if requested
    if save:
        fig.savefig("Pauli_noise_graph.pdf", bbox_inches='tight')
        fig.savefig("Pauli_noise_graph.png", dpi=300, bbox_inches='tight')

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



import os

def plot_depths(
    depths: list[int],
    title: str = "Circuit Depth vs. Input Size",
    subtitle: str = "",
    layout_method: str = "trivial",
    routing_method: str = "Sabre"
) -> None:
    """
    Plots a list of circuit depths and saves the figure as PDF and PNG.

    Args:
        depths:         List of depth values. The x-axis will be 0..len(depths)-1.
        title:          Plot title (also used in filenames—avoid slashes!).
        layout_method:  Name of the layout method (for filename).
        routing_method: Name of the routing method (for filename).
    """
    # Prepare data
    x = list(range(len(depths)))
    y = depths

    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, y, marker='o', linestyle='-')
    ax.set_xlabel("Number of Qubits")
    ax.set_ylabel("Circuit Depth")
    ax.set_title(f"{title}: {subtitle}")
    ax.grid(True)
    plt.tight_layout()

    # Ensure output directory exists
    out_dir = "images/plots"
    os.makedirs(out_dir, exist_ok=True)

    # Build a safe base filepath (no double dots)
    safe_title = title.replace(" ", "_")
    base = f"{out_dir}/{safe_title}_layout_{layout_method}_routing_{routing_method}"

    # Save to PDF and PNG
    fig.savefig(f"{base}.pdf", format="pdf", bbox_inches="tight")
    fig.savefig(f"{base}.png", format="png", dpi=300, bbox_inches="tight")

    # Finally show on screen
    plt.show()

#
# def plot_qft_complexity(n_max=8):
#     """
#     Plots the space and time complexity of an n-qubit QFT circuit
#     from n = 1 up to n = n_max.
#     """
#     n = np.arange(1, n_max + 1)
#     space = n
#     time = n + (n * (n - 1)) // 2 + (n // 2)  # H gates + controlled-phase + swaps
#
#     # Create the plot
#     fig, ax = plt.subplots(figsize=(8, 5))
#
#     plt.figure()
#     plt.plot(n, time, marker='o', label='Time Complexity (Gate Count)')
#     plt.plot(n, space, marker='s', label='Space Complexity (Qubit Count)')
#     plt.xlabel('Number of Qubits (n)')
#     plt.ylabel('Count')
#     plt.title('QFT Circuit Complexity vs Number of Qubits')
#     plt.xticks(n)
#     plt.grid(True)
#     plt.legend()
#
#     # Save to PDF and PNG
#     fig.savefig("images/plots/QFT_space_time_complexity.pdf", format="pdf", bbox_inches="tight")
#     fig.savefig("images/plots/QFT_space_time_complexity.png", format="png", dpi=300, bbox_inches="tight")
#
#     plt.show()



def plot_qft_complexity(n_max=8, brickwork_points=None):
    """
    Plots the space and time complexity of an n-qubit QFT circuit
    from n = 1 up to n = n_max.

    Parameters:
    - n_max (int): Maximum number of qubits to plot.
    - brickwork_points (list or array of length n_max, optional):
        If provided, will be plotted as "Brickwork Scaling".
    """
    # x-axis: qubit counts
    n = np.arange(1, n_max + 1)
    # space = n qubits
    space = n
    # time = n H-gates + n(n−1)/2 CP-gates + floor(n/2) swaps
    time = n + (n * (n - 1)) // 2 + (n // 2)

    # n Hadamards at 3 elementary gates each,
    # n(n–1)/2 controlled-phase rotations,
    # and (n/2) swaps at 3 CX each:

    # Hadamards → Rz–Rx–Rz
    # controlled-phase gates
    # swaps → 3 CX per swap
    time_adjusted = (3 * n)  + (n * (n - 1)) // 2 + (3 * (n // 2))
    time_adjusted_bricks = n + (n * (n - 1)) // 2 + (6 * (n // 2)) # 6 bricks per swap / 1 brick per H / 1 brick per CX (deep)


    upper_bound_bricks = 3 * time_adjusted_bricks

    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 5))

    plt.plot(n, time, marker='o', label='Time Complexity QFT (Non-decomposed Gate Count)')
    plt.plot(n, time_adjusted_bricks, marker='^', label='Time Complexity QFT LB (Brick Count)')
    plt.plot(n, upper_bound_bricks, marker='v', label='Theoretical UB (Brick Count)')
    # plt.plot(n, space, marker='s', label='Space Complexity (Qubit Count)')

    if brickwork_points is not None:
        # divide each entry by a constant and truncate to int
        # brickwork_points = [int(p / 2) for p in brickwork_points]
        if len(brickwork_points) != len(n):
            raise ValueError(
                f"brickwork_points length ({len(brickwork_points)}) "
                f"must equal n_max ({n_max})"
            )
        plt.plot(
            n,
            brickwork_points,
            marker='s',
            linestyle='--',
            label='Brickwork Scaling (Brick Count)'
        )

    plt.xlabel('Number of Qubits (n)')
    plt.ylabel('Count')
    plt.title('QFT Circuit Complexity vs Number of Qubits')
    plt.xticks(n)
    plt.grid(True)
    plt.legend()

    # Save to PDF and PNG
    fig.savefig("images/plots/QFT_space_time_complexity.pdf", format="pdf", bbox_inches="tight")
    fig.savefig("images/plots/QFT_space_time_complexity.png", format="png", dpi=300, bbox_inches="tight")

    plt.show()






if __name__ == "__main__":
    plot_qft_complexity()

