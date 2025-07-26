import math
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

def plot_brickwork_graph_from_pattern22(
        bw_pattern,
        show_angles: bool = True,
        node_size: float = 1200,  # increased
        node_color: str = 'skyblue',
        edge_color: str = 'gray',
        font_size: int = 8,  # small labels
        figsize: tuple = None,
        cell_size: float = 0.6,  # slightly bigger cells
        node_spacing: float = 1.5,  # more spacing between nodes
        margin: float = 1.0,  # more margin around the graph
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
    legend_height = 0.6  # a bit more space for legend
    if figsize is None:
        spacing = cell_size * node_spacing
        total_width = spacing * (n_cols + 2 * margin)
        total_height = spacing * (n_rows + 2 * margin) + legend_height
        figsize = (total_width, total_height)

    fig = plt.figure(figsize=figsize, dpi=150)  # higher dpi for clean look
    gs = fig.add_gridspec(
        2, 1,
        height_ratios=[n_rows * cell_size * node_spacing, legend_height],
        hspace=0.0
    )
    ax = fig.add_subplot(gs[0, 0])
    legend_ax = fig.add_subplot(gs[1, 0])
    legend_ax.axis('off')

    # 5. Compute positions with uniform spacing
    pos = {
        (r, c): (c * node_spacing, r * node_spacing)
        for (r, c) in nodes
    }

    # 6. Draw edges
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=edge_color, width=1.5)

    # 7. Draw nodes
    colours = (
        [node_colours.get(node, node_color) for node in G.nodes()]
        if use_node_colours else
        node_color
    )
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_size=node_size,
        node_color=colours,
        edgecolors='black', linewidths=0.5  # thin black border around nodes
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
        font_family='DejaVu Sans',  # modern clean font
        ax=ax
    )

    # 9. Final tweaks on main axes
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.set_xlim(-margin * node_spacing,
                (n_cols - 1 + margin) * node_spacing)
    ax.set_ylim((n_rows - 1 + margin) * node_spacing,
                -margin * node_spacing)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axis('off')

    # 10. Legend below
    if use_node_colours:
        handles = [
            mpatches.Patch(color='lightcoral',  label='CX gate - ⊕ top'),
            mpatches.Patch(color='lightsalmon', label='CX gate - ⊕ bot'),
            mpatches.Patch(color='lightblue',   label='Identity'),
            mpatches.Patch(color='lightgreen',  label='Euler rotation'),
            mpatches.Patch(color='skyblue',     label='Output qubits'),
        ]
        legend_ax.legend(
            handles=handles,
            title="Gate types",
            loc='center',
            ncol=4,
            frameon=False,  # remove frame for cleaner look
            fontsize=9,
            title_fontsize=10
        )

    plt.show()

    # Save vector graphics
    fig.savefig(f"images/graphs/{title}.pdf", format="pdf", bbox_inches="tight")
    fig.savefig(f"images/graphs/{title}.png", format="png", dpi=300, bbox_inches="tight")


def plot_brickwork_graph_from_pattern(
        bw_pattern,
        show_angles: bool = True,
        node_size: float = 1500,
        node_color: str = 'skyblue',
        edge_color: str = 'gray',
        font_size: int = 9,
        figsize: tuple = None,
        cell_size: float = 0.7,
        node_spacing: float = 1.8,
        margin: float = 1.2,
        title: str = "Brickwork Graph",
        use_node_colours: bool = False,
        node_colours: dict = None
    ):
    if node_colours is None:
        node_colours = {}

    nodes, edges = bw_pattern.get_graph()
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    angles = bw_pattern.get_angles() if show_angles else {}

    rows = [r for (r, _) in nodes]
    cols = [c for (_, c) in nodes]
    n_rows, n_cols = max(rows) + 1, max(cols) + 1

    legend_height = 0.8
    if figsize is None:
        spacing = cell_size * node_spacing
        total_width = spacing * (n_cols + 2 * margin)
        total_height = spacing * (n_rows + 2 * margin) + legend_height
        figsize = (total_width, total_height)

    fig = plt.figure(figsize=figsize, dpi=200)
    gs = fig.add_gridspec(
        2, 1,
        height_ratios=[n_rows * cell_size * node_spacing, legend_height],
        hspace=0.1
    )
    ax = fig.add_subplot(gs[0, 0])
    legend_ax = fig.add_subplot(gs[1, 0])
    legend_ax.axis('off')

    pos = {
        (r, c): (c * node_spacing, r * node_spacing)
        for (r, c) in nodes
    }

    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=edge_color, width=3.0)

    colours = [node_colours.get(node, node_color) for node in G.nodes()] if use_node_colours else node_color
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_size=node_size,
        node_color=colours,
        edgecolors='black', linewidths=2.0
    )

    if show_angles:
        labels = {node: f"({r},{c})\n{angles[node]:.2f}π" if show_angles and node in angles else f"({r},{c})" for node, (r, c) in zip(nodes, nodes)}
    else:
        labels = {}

    nx.draw_networkx_labels(
        G, pos, labels=labels,
        font_size=font_size,
        font_family='DejaVu Sans',
        ax=ax,
        font_weight='medium'
    )

    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.set_xlim(-margin * node_spacing, (n_cols - 1 + margin) * node_spacing)
    ax.set_ylim((n_rows - 1 + margin) * node_spacing, -margin * node_spacing)
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.axis('off')

    if use_node_colours:
        handles = [
            mpatches.Patch(color='lightcoral', label='CX gate - ⊕ top'),
            mpatches.Patch(color='lightsalmon', label='CX gate - ⊕ bot'),
            mpatches.Patch(color='lightblue', label='Identity'),
            mpatches.Patch(color='lightgreen', label='Euler rotation'),
            mpatches.Patch(color='skyblue', label='Output qubits'),
        ]
        legend_ax.legend(
            handles=handles,
            title="Gate types",
            loc='center',
            ncol=3,
            frameon=False,
            fontsize=11,
            title_fontsize=12
        )

    # plt.tight_layout()
    plt.show()

    fig.savefig(f"images/graphs/{title}.pdf", format="pdf", bbox_inches="tight")
    fig.savefig(f"images/graphs/{title}.png", format="png", dpi=300, bbox_inches="tight")


def plot_brickwork_graph_from_pattern33(
        bw_pattern,
        show_angles: bool = True,
        node_size: float = 2500,
        node_color: str = 'skyblue',
        edge_color: str = 'dimgray',
        font_size: int = 10,
        figsize: tuple = None,
        cell_size: float = 1.0,
        node_spacing: float = 2.5,
        margin: float = 1.5,
        title: str = "Brickwork Graph",
        use_node_colours: bool = False,
        node_colours: dict = None
    ):
    if node_colours is None:
        node_colours = {}

    nodes, edges = bw_pattern.get_graph()
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    angles = bw_pattern.get_angles() if show_angles else {}

    rows = [r for (r, _) in nodes]
    cols = [c for (_, c) in nodes]
    n_rows, n_cols = max(rows) + 1, max(cols) + 1

    legend_height = 1.0
    if figsize is None:
        spacing = cell_size * node_spacing
        total_width = spacing * (n_cols + 2 * margin)
        total_height = spacing * (n_rows + 2 * margin) + legend_height
        figsize = (total_width, total_height)

    fig = plt.figure(figsize=figsize, dpi=200)
    gs = fig.add_gridspec(
        2, 1,
        height_ratios=[n_rows * cell_size * node_spacing, legend_height],
        hspace=0.15
    )
    ax = fig.add_subplot(gs[0, 0])
    legend_ax = fig.add_subplot(gs[1, 0])
    legend_ax.axis('off')

    pos = {
        (r, c): (c * node_spacing, r * node_spacing)
        for (r, c) in nodes
    }

    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=edge_color, width=4.0)

    colours = [node_colours.get(node, node_color) for node in G.nodes()] if use_node_colours else node_color
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_size=node_size,
        node_color=colours,
        edgecolors='black', linewidths=2.0
    )

    labels = {node: f"({r},{c})\n{angles[node]:.2f}π" if show_angles and node in angles else f"({r},{c})" for node, (r, c) in zip(nodes, nodes)}

    nx.draw_networkx_labels(
        G, pos, labels=labels,
        font_size=font_size,
        font_family='DejaVu Sans',
        ax=ax,
        font_weight='medium'
    )

    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.set_xlim(-margin * node_spacing, (n_cols - 1 + margin) * node_spacing)
    ax.set_ylim((n_rows - 1 + margin) * node_spacing, -margin * node_spacing)
    ax.set_title(title, fontsize=18, fontweight='bold')
    ax.axis('off')

    if use_node_colours:
        handles = [
            mpatches.Patch(color='lightcoral', label='CX gate - ⊕ top'),
            mpatches.Patch(color='lightsalmon', label='CX gate - ⊕ bot'),
            mpatches.Patch(color='lightblue', label='Identity'),
            mpatches.Patch(color='lightgreen', label='Euler rotation'),
            mpatches.Patch(color='skyblue', label='Output qubits'),
        ]
        legend_ax.legend(
            handles=handles,
            title="Gate types",
            loc='center',
            ncol=3,
            frameon=False,
            fontsize=12,
            title_fontsize=13
        )

    plt.tight_layout()
    plt.show()

    fig.savefig(f"images/graphs/{title}.pdf", format="pdf", bbox_inches="tight")
    fig.savefig(f"images/graphs/{title}.png", format="png", dpi=300, bbox_inches="tight")

from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx

def plot_brickwork_graph_locked2(
        bw_pattern,
        node_image_path: str='images/lock_delta_nobg_2.png',
        edge_color: str = 'gray',
        figsize: tuple = None,
        cell_size: float = 0.7,
        node_spacing: float = 1.8,
        margin: float = 1.2,
        title: str = "UBQC structure"
    ):
    # Get the graph
    nodes, edges = bw_pattern.get_graph()
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    # Grid size
    rows = [r for (r, _) in nodes]
    cols = [c for (_, c) in nodes]
    n_rows, n_cols = max(rows) + 1, max(cols) + 1

    # Figure size
    legend_height = 0.8
    if figsize is None:
        spacing = cell_size * node_spacing
        total_width = spacing * (n_cols + 2 * margin)
        total_height = spacing * (n_rows + 2 * margin) + legend_height
        figsize = (total_width, total_height)

    # Create figure
    fig = plt.figure(figsize=figsize, dpi=200)
    gs = fig.add_gridspec(
        2, 1,
        height_ratios=[n_rows * cell_size * node_spacing, legend_height],
        hspace=0.1
    )
    ax = fig.add_subplot(gs[0, 0])
    legend_ax = fig.add_subplot(gs[1, 0])
    legend_ax.axis('off')

    # Compute positions
    pos = {
        (r, c): (c * node_spacing, r * node_spacing)
        for (r, c) in nodes
    }

    # Draw edges
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=edge_color, width=2.0)

    # Load the image
    img = mpimg.imread(node_image_path)
    imagebox = OffsetImage(img, zoom=0.09)  # adjust zoom as needed

    # Define vertical offset (example: 20% of node spacing)
    y_offset = (-0.1* node_spacing)  # you can fine-tune this factor

    # Add image with vertical shift
    for node, (x, y) in pos.items():
        ab = AnnotationBbox(imagebox, (x, y + y_offset), frameon=False)
        ax.add_artist(ab)

    # Final plot settings
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.set_xlim(-margin * node_spacing, (n_cols - 1 + margin) * node_spacing)
    ax.set_ylim((n_rows - 1 + margin) * node_spacing, -margin * node_spacing)
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.axis('off')

    plt.tight_layout()
    plt.show()

    # Save the figure
    fig.savefig(f"images/graphs/{title}.pdf", format="pdf", bbox_inches="tight")
    fig.savefig(f"images/graphs/{title}.png", format="png", dpi=300, bbox_inches="tight")


def plot_brickwork_graph_from_pattern_old_style(
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
    fig.savefig(f"images/graphs/{title}_old_style.pdf", format="pdf", bbox_inches="tight")
    fig.savefig(f"images/graphs/{title}_old_style.png", format="png", dpi=300, bbox_inches="tight")


import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import networkx as nx

def plot_brickwork_graph_locked(
        bw_pattern,
        node_image_path: str = 'images/lock_delta_nobg_2.png',
        use_locks: bool = True,
        edge_color: str = 'gray',
        figsize: tuple = None,
        cell_size: float = 0.7,
        node_spacing: float = 1.8,
        margin: float = 1.2,
        title: str = "UBQC structure"
    ):
    # Get the graph
    nodes, edges = bw_pattern.get_graph()
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    # Grid size
    rows = [r for (r, _) in nodes]
    cols = [c for (_, c) in nodes]
    n_rows, n_cols = max(rows) + 1, max(cols) + 1

    # Figure size
    legend_height = 0.8
    if figsize is None:
        spacing = cell_size * node_spacing
        total_width = spacing * (n_cols + 2 * margin)
        total_height = spacing * (n_rows + 2 * margin) + legend_height
        figsize = (total_width, total_height)

    # Create figure
    fig = plt.figure(figsize=figsize, dpi=200)
    gs = fig.add_gridspec(
        2, 1,
        height_ratios=[n_rows * cell_size * node_spacing, legend_height],
        hspace=0.1
    )
    ax = fig.add_subplot(gs[0, 0])
    legend_ax = fig.add_subplot(gs[1, 0])
    legend_ax.axis('off')

    # Compute positions
    pos = {
        (r, c): (c * node_spacing, r * node_spacing)
        for (r, c) in nodes
    }

    # Draw edges
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=edge_color, width=3.0)

    plt.rcParams['mathtext.fontset'] = 'cm'  # Use Computer Modern
    plt.rcParams['mathtext.rm'] = 'serif'  # Use serif font for normal text

    if use_locks:
        # Load the image
        img = mpimg.imread(node_image_path)
        imagebox = OffsetImage(img, zoom=0.09)  # adjust zoom as needed

        # Define vertical offset
        y_offset = (-0.1 * node_spacing)  # adjust if needed

        # Add image with vertical shift
        for node, (x, y) in pos.items():
            ab = AnnotationBbox(imagebox, (x, y + y_offset), frameon=False)
            ax.add_artist(ab)

    else:

        pos = {
            (r, c): (c * node_spacing, r * node_spacing)
            for (r, c) in nodes
        }

        nx.draw_networkx_edges(G, pos, ax=ax, edge_color=edge_color, width=4.0)


        nx.draw_networkx_nodes(
            G, pos, ax=ax,
            node_size=1500,
            node_color='#e9e9f9',
            edgecolors='black', linewidths=2.0
        )

        # labels = {node: rf"$\delta$" for node, (r, c) in zip(nodes, nodes)}
        # labels = {}
        labels = {node: rf"$\theta_{{{node[0]},{node[1]}}}$" for node in nodes}
        nx.draw_networkx_labels(
            G, pos, labels=labels,
            font_size=20,
            font_family='serif',
            ax=ax,
            font_weight='medium'
        )


    # Final plot settings
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.set_xlim(-margin * node_spacing, (n_cols - 1 + margin) * node_spacing)
    ax.set_ylim((n_rows - 1 + margin) * node_spacing, -margin * node_spacing)
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.axis('off')

    # Add LaTeX equation to the legend
    equation = (r"$\{|+_{\delta_{x,y}}\rangle, |-_{\delta_{x,y}}\rangle\},$" r"$\quad$" 
                r"$\delta_{x,y} = \phi'_{x,y} + \theta_{x,y} + \pi r_{x,y}$")
    legend_ax.text(
        0.5, 1, equation,
        fontsize=25,  # <-- constant size for readability
        ha='center', va='center',
    bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5')
    )

    # plt.tight_layout()
    plt.show()

    # Save the figure
    fig.savefig(f"images/graphs/{title}_enc.pdf", format="pdf", bbox_inches="tight")
    fig.savefig(f"images/graphs/{title}_enc.png", format="png", dpi=300, bbox_inches="tight")

# plot_graphix_pattern_scalar_index.py

# plot_graphix_pattern_scalar_index.py

def plot_graphix_noise_graph(
        pattern,
        show_angles: bool = True,
        node_size: float = 1500,
        edge_color: str = 'gray',
        font_size: int = 9,
        figsize: tuple = None,
        cell_size: float = 0.7,
        node_spacing: float = 1.8,
        margin: float = 1.2,
        title: str = "Shaded Noise Graph",
        cmap_name: str = 'viridis',  # dark-blue to dark-red continuous spectrum
        vmin: float = None,
        vmax: float = None,
        show_colorbar: bool = True,
        save: bool = False,
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

    unique_layers = sorted(set(node_layer.values()))
    layer_to_idx = {layer: i for i, layer in enumerate(unique_layers)}

    pos = {}
    for layer in unique_layers:
        nodes_in_layer = sorted(n for n, l in node_layer.items() if l == layer)
        for idx, n in enumerate(nodes_in_layer):
            x = layer_to_idx[layer] * node_spacing * cell_size
            y = idx * node_spacing * cell_size
            pos[n] = (x, y)

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
    # nx.draw_networkx_labels(G, pos, labels, font_size=font_size, ax=ax)

    from matplotlib.colors import rgb_to_hsv

    # Assign dynamic font colors based on background brightness
    font_colors = []
    for color in node_colors:
        r, g, b, _ = color
        luminance = 0.299 * r + 0.587 * g + 0.114 * b
        font_colors.append('white' if luminance < 0.5 else 'black')

    # Draw labels with adjusted font color
    for idx, (n, label) in enumerate(labels.items()):
        ax.text(pos[n][0], pos[n][1], label,
                fontsize=font_size,
                ha='center', va='center',
                color=font_colors[idx])

    # Axis tweaks
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.set_xlim(-margin * node_spacing * cell_size,
                (depth + margin) * node_spacing * cell_size)
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
        cbar = plt.colorbar(
            sm,
            cax=cax,
            orientation='horizontal',

            # alternatively:
        )
        cbar.set_label('Measurement angle noise', labelpad=8)
        # Show only start and end
        cbar.set_ticks([vmin_val, vmax_val])
        cbar.set_ticklabels([f"{vmin_val:.2f}π", f"{vmax_val:.2f}π"])
        # Ensure axis line is visible

        # --- shrink the colour‑bar’s width to a % and keep it centred ----
        box = cax.get_position()  # current [x0, y0, width, height] in figure‑coords
        new_width = box.width * 0.80  # % of the present length
        dx = (box.width - new_width) / 2  # equal margin left & right
        cax.set_position([box.x0 + dx, box.y0, new_width, box.height])

        cbar.ax.xaxis.set_ticks_position('bottom')
        cbar.ax.xaxis.set_label_position('bottom')

    # Save if requested
    if save:
        fig.savefig("noise_graph.pdf", bbox_inches='tight')
        fig.savefig("noise_graph.png", dpi=300, bbox_inches='tight')

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


def plot_qrs_bw_scaling(user_counts: list[int],
                    input_brick_depths: list[int],
                    aligned_brick_depths: list[int],
                    feature_length: int):
    """
    Plot three curves (aligned_counts, input_counts, and their theoretical upper bound)
    against a specified list of user_counts on the x-axis.

    Parameters
    ----------
    user_counts : list[int]
        The number of users for each data point (e.g. [4, 8, 16, 32, 64, 128]).
    input_counts : list[int]
        The measured “input brick” counts for each user count.
    aligned_counts : list[int]
        The measured “aligned brick” counts for each user count.
    """

    # 1) Sanity check: all three lists must have the same length
    n = len(user_counts)
    if len(input_brick_depths) != n or len(aligned_brick_depths) != n:
        raise ValueError(
            f"All input lists must have the same length: "
            f"user_counts={n}, input_brick_depths={len(input_brick_depths)}, aligned_brick_depths={len(aligned_brick_depths)}"
        )

    # 2) Compute theoretical upper bound: 3× input_brick_depths
    theoretical_ub = [3 * x for x in input_brick_depths]

    # 3) Create the figure and axes
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))

    # 4) Plot each series vs. user_counts
    ax.plot(user_counts, aligned_brick_depths, marker='o',
            label='BQRS brickwork scaling (Brick Count)')
    ax.plot(user_counts, input_brick_depths, marker='^',
            label='Time Complexity BQRS LB (Brick Count)')
    ax.plot(user_counts, theoretical_ub, marker='v',
            label='Theoretical UB (Brick Count)')

    # 5) Labeling
    ax.set_xlabel('Number of Users')
    ax.set_ylabel('Count')
    ax.set_title(f'BQRS Circuit Complexity vs Number of Users (feature length = {feature_length})')

    # 6) Ensure x-axis ticks are exactly the user_counts values
    ax.set_xticks(user_counts)
    ax.set_xticklabels([str(u) for u in user_counts], rotation=45)

    ax.grid(True)
    ax.legend()

    # 7) Save & show
    fig.savefig("images/plots/BQRS_space_time_scaling_by_users.pdf",
                format="pdf", bbox_inches="tight")
    fig.savefig("images/plots/BQRS_space_time_scaling_by_users.png",
                format="png", dpi=300, bbox_inches="tight")

    plt.show()

    import math
    import matplotlib.pyplot as plt

def decomposition_cost(l: int) -> int:
    """
    Estimate the two-qubit (“CX”) gate count needed to decompose an l-controlled
    gate into {Rz, Rx, CX, I}. We use a simple linear model:
        c = 2*l - 1

    In other words, each l-controlled CZ or CX is assumed to expand into (2*l - 1) CXs
    plus single-qubit rotations (Rz, Rx) and identity wires.
    """
    return 2 * l - 1

def plot_time_complexity(user_counts: list[int], feature_widths: list[int]) -> None:
    """
    For each (user_count, feature_width) pair, invoke `time_complexity(q, l, c)`
    to compute gate-counts, then plot the resulting metrics against user_counts.

    We assume:
      •  `q = log2(user_count)` must be an integer (so user_count is a power of two).
      •  `l = feature_width` is the width (in qubits) of each feature vector.
      •  `c = decomposition_cost(l)` is the two-qubit cost of each multi-controlled gate,
         under the basis {Rz, Rx, CX, I}.

    Each call to `time_complexity` returns a dict with keys:
      - "Database_creation"  : int (≈ 2^l + (2^q·(2^q − 1)//2))
      - "kNN_distance"       : int (≡ 3·l + 2)
      - "Grover_amplify"     : int (≡ 7·l + 2·c + 3)
      - "Total"              : int (sum of the above)

    The function then produces a single 2D plot (gate-count vs. number of users) showing:
      •  Total complexity
      •  Database creation cost
      •  k-NN distance cost
      •  Grover amplification cost

    Parameters
    ----------
    user_counts : list[int]
        A list of “number of users” values (each must be a power of two).
        Internally we take q = log2(user_count) to pass into `time_complexity`.

    feature_widths : list[int]
        A list of feature-vector widths (l). Must have the same length as user_counts.

    Raises
    ------
    ValueError
        If the two input lists differ in length, or if any user_count is not a power of two.
    """

    if len(user_counts) != len(feature_widths):
        raise ValueError(
            f"user_counts length ({len(user_counts)}) must equal "
            f"feature_widths length ({len(feature_widths)})"
        )

    # Containers for each metric
    total_complexities = []
    db_creations = []
    kNNs = []
    grovers = []

    for uc, l in zip(user_counts, feature_widths):
        # Ensure user_count is a power of two so q = integer log2(uc)
        if uc <= 0 or (uc & (uc - 1)) != 0:
            raise ValueError(f"User count {uc} is not a positive power of two.")
        q = int(math.log2(uc))

        # Compute c from our decomposition model
        c = decomposition_cost(l)

        # Invoke the time_complexity function defined earlier
        metrics = time_complexity(q=q, l=l, c=c)

        db_creations.append(metrics["Database_creation"])
        kNNs.append(metrics["kNN_distance"])
        grovers.append(metrics["Grover_amplify"])
        total_complexities.append(metrics["Total"])

    # --- Create the plot ---
    fig, ax = plt.subplots(figsize=(8, 5))

    # Total complexity vs. number of users
    ax.plot(user_counts, total_complexities, marker='o', linestyle='-',
            label='Total Gate Count')

    # Breakdown curves
    ax.plot(user_counts, db_creations, marker='s', linestyle='--',
            label='Database Creation')
    ax.plot(user_counts, kNNs, marker='^', linestyle='-.',
            label='kNN Distance')
    ax.plot(user_counts, grovers, marker='v', linestyle=':',
            label='Grover Amplify')

    # Label axes
    ax.set_xlabel('Number of Users\n(q = log₂(user_count))')
    ax.set_ylabel('Gate Count')
    ax.set_title('QRS Circuit Time Complexity vs Number of Users')

    # Force x-ticks exactly at each user_count
    ax.set_xticks(user_counts)
    ax.set_xticklabels([str(u) for u in user_counts], rotation=45)

    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    plt.show()

import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def time_complexity(q: int, l: int, c: int) -> dict[str, int]:
    """
    As defined previously, returns a dict of gate-count metrics for given (q, l, c).
    """
    N = 2 ** q
    database_creation = 2 ** l + (N * (N - 1)) // 2
    kNN_distance = 3 * l + 2
    grover_amplify = 7 * l + 2 * c + 3
    total = database_creation + kNN_distance + grover_amplify

    return {
        "Database_creation": database_creation,
        "kNN_distance": kNN_distance,
        "Grover_amplify": grover_amplify,
        "Total": total,
    }

def plot_time_complexity_3d(user_counts: list[int], feature_widths: list[int]) -> None:
    """
    Produces a 3D surface plot of Total Gate Count as a function of:
      • x-axis: number of users (must be powers of two)
      • y-axis: feature-vector width l
      • z-axis: Total gate count from time_complexity(q, l, c)

    For each pair (u, l) in the grid formed by user_counts × feature_widths:
      1. Compute q = log2(u) (raises ValueError if u is not a power of two).
      2. Compute c = decomposition_cost(l).
      3. Compute metrics = time_complexity(q, l, c).
      4. Extract metrics["Total"] into a 2D array Z.

    Parameters
    ----------
    user_counts : list[int]
        A strictly positive list of integers, each a power of two.
    feature_widths : list[int]
        A strictly positive list of integers representing l. Can be any positive ints.

    Raises
    ------
    ValueError
        If any user_count is not a positive power of two.
    """

    # 1) Verify inputs
    for u in user_counts:
        if u <= 0 or (u & (u - 1)) != 0:
            raise ValueError(f"User count {u} is not a positive power of two.")

    user_counts = sorted(user_counts)
    feature_widths = sorted(feature_widths)

    # 2) Build meshgrid over (user_counts, feature_widths)
    U_vals = np.array(user_counts)
    L_vals = np.array(feature_widths)
    U_grid, L_grid = np.meshgrid(U_vals, L_vals, indexing='xy')  # shape: (len(L), len(U))

    # 3) Compute Total complexity for each (u, l)
    Z_total = np.zeros_like(U_grid, dtype=float)
    for i, l in enumerate(L_vals):
        c = decomposition_cost(int(l))
        for j, u in enumerate(U_vals):
            q = int(math.log2(int(u)))
            metrics = time_complexity(q=q, l=int(l), c=c)
            Z_total[i, j] = metrics["Total"]

    # 4) Plot 3D surface
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Convert U_grid and L_grid to float for plotting
    ax.plot_surface(
        U_grid.astype(float),
        L_grid.astype(float),
        Z_total,
        cmap='viridis',
        edgecolor='none',
        alpha=0.8
    )

    # 5) Labeling
    ax.set_xlabel('Number of Users (N = 2^q)')
    ax.set_ylabel('Feature Width (l)')
    ax.set_zlabel('Total Gate Count')
    ax.set_title('3D Surface of QRS Time Complexity')

    # 6) Configure ticks
    ax.set_xticks(U_vals)
    ax.set_xticklabels([str(int(u)) for u in U_vals], rotation=45, ha='right')
    ax.set_yticks(L_vals)
    ax.set_yticklabels([str(int(l)) for l in L_vals])
    # … after plotting the surface …
    ax.view_init(elev=20, azim=245)
    # plt.tight_layout()

    # Save & show
    fig.savefig(f"images/plots/BQRS_space_time_3d.pdf",
                format="pdf", bbox_inches="tight")
    fig.savefig(f"images/plots/BQRS_space_time_3d.png",
                format="png", dpi=300, bbox_inches="tight")

    plt.show()



import math
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def plot_time_complexity_with_bw_lines(
    user_counts: list[int],
    feature_widths: list[int],
    bw_in_depths: list[int],
    bw_aligned_depths: list[int],
    feature_length: int,
    elev: float = 20,
    azim: float = 240
) -> None:
    """
    3D surface of Total Gate Count vs. (Number of Users, Feature Width), with two overlay lines
    (bw_in_depths and bw_aligned_depths) at a fixed feature_width = feature_length.

    Parameters
    ----------
    user_counts : list[int]
        List of powers-of-two (e.g. [4, 8, 16, 32, 64, 128]).
    feature_widths : list[int]
        List of feature-vector widths (e.g. [4, 5, 6, 7, 8]).
    bw_in_depths : list[int]
        Z-values for the “in-depths” line corresponding to each user_count at l = feature_length.
    bw_aligned_depths : list[int]
        Z-values for the “aligned-depths” line corresponding to each user_count at l = feature_length.
    feature_length : int
        The fixed l value at which bw_in_depths and bw_aligned_depths are defined.
    elev : float, optional
        Elevation angle in degrees for the 3D view (default=30).
    azim : float, optional
        Azimuth angle in degrees for the 3D view (default=45).
    """

    # 1) Validate that each user_count is a power of two
    for u in user_counts:
        if u <= 0 or (u & (u - 1)) != 0:
            raise ValueError(f"User count {u} is not a positive power of two.")
    if len(user_counts) != len(bw_in_depths) or len(user_counts) != len(bw_aligned_depths):
        raise ValueError("Lengths of user_counts, bw_in_depths, and bw_aligned_depths must match.")
    if feature_length not in feature_widths:
        raise ValueError("feature_length must be one of the values in feature_widths.")

    # Sort inputs
    user_counts = sorted(user_counts)
    feature_widths = sorted(feature_widths)

    # 2) Build meshgrid over (user_counts, feature_widths)
    U_vals = np.array(user_counts)
    L_vals = np.array(feature_widths)
    U_grid, L_grid = np.meshgrid(U_vals, L_vals, indexing='xy')  # shape: (len(L), len(U))

    # 3) Compute Total complexity for each (u, l)
    Z_total = np.zeros_like(U_grid, dtype=float)
    for i, l in enumerate(L_vals):
        c = decomposition_cost(int(l))
        for j, u in enumerate(U_vals):
            q = int(math.log2(int(u)))
            metrics = time_complexity(q=q, l=int(l), c=c)
            Z_total[i, j] = metrics["Total"]

    # 4) Plot 3D surface
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(
        U_grid.astype(float),
        L_grid.astype(float),
        Z_total,
        cmap='viridis',
        edgecolor='none',
        alpha=0.8,
        zorder=0
    )

    # 5) Overlay the two lines at l = feature_length
    y_line = [feature_length] * len(user_counts)
    x_line = user_counts
    z_in = bw_in_depths
    z_aligned = bw_aligned_depths

    # Plot "in-depths" line
    ax.plot(
        x_line,
        y_line,
        z_in,
        color='red',
        marker='^',
        markersize=6,
        label='Decomposed BW Complexity (l={})'.format(feature_length),
        linewidth=2,
        zorder=10
    )

    # Plot "aligned-depths" line
    ax.plot(
        x_line,
        y_line,
        z_aligned,
        color='blue',
        marker='o',
        markersize=6,
        label='Aligned Decomposed BW Complexity (l={})'.format(feature_length),
        linewidth=2,
        zorder=10 # Because of bug in depth buffer force lines on top
    )

    # 6) Adjust view angle
    ax.view_init(elev=elev, azim=azim)

    # 7) Labeling
    ax.set_xlabel('Number of Users (N = 2^q)')
    ax.set_ylabel('Feature Width (l)')
    ax.set_zlabel('Gate Count / Depth')
    ax.set_title('3D Surface of QRS Time Complexity with BW Depth Lines')

    ax.set_xticks(U_vals)
    ax.set_xticklabels([str(int(u)) for u in U_vals], rotation=45, ha='right')
    ax.set_yticks(L_vals)
    ax.set_yticklabels([str(int(l)) for l in L_vals])

    ax.legend()
    # plt.tight_layout()

    # Save & show
    fig.savefig(f"images/plots/BQRS_space_time_3d_with_bw_l{feature_length}.pdf",
                format="pdf", bbox_inches="tight")
    fig.savefig(f"images/plots/BQRS_space_time_3d_with_bw_l{feature_length}.png",
                format="png", dpi=300, bbox_inches="tight")

    plt.show()




if __name__ == "__main__":
    plot_qft_complexity()

