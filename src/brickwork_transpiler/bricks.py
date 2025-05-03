import networkx as nx
from graphix.generator import generate_from_graph
from src.brickwork_transpiler import graph_builder, visualiser


#Follows a, b, c, 0 scheme
def arbitrary_brick(a, b, c, input=None):

    inside_graph = graph_builder.create_tuple_node_graph(1, 5)

    # two hadamard cancel leaving 4 required measurements instead of 6
    # Angles are negative by convention (add source)
    angles = {
        (0, 0): -a,  # Measure node (0, 0) at angle -a
        (0, 1): -b,
        (0, 2): -c,
        (0, 3): 0.0  # final teleport (J(0) Hadamard)
    }  #

    if input is None:
        inputs = [] # Initialise to |+> state
    else:
        inputs = [(0, 0)]

    outputs = [(0, 4)]

    # See Graphix generator documentation: https://graphix.readthedocs.io/en/v0.2.1/_modules/graphix/generator.html
    brick_pattern = generate_from_graph(inside_graph, angles, inputs, outputs)

    return brick_pattern


#TODO:
def CX_brick(inputs=None):

    # Layout:
    #  (0, 0) -- (0, 1) -- (0, 2) -- (0, 3) -- (0, 4)
    #                        |                   |
    #  (1, 0) -- (1, 1) -- (1, 2) -- (1, 3) -- (1, 4)
    inside_graph = graph_builder.create_tuple_node_graph(2, 5)
    inside_graph.add_edge((0, 2), (1, 2))
    inside_graph.add_edge((0, 4), (1, 4))

    visualiser.plot_graph(inside_graph)

    # invert angle sign because ... (SOURCE) graphix
    angles = {
        (0, 0): 0.0,
        (0, 1): 0.0,
        (0, 2): -1/4,
        (0, 3): 0.0,
        (1, 0): 0.0,
        (1, 1): -1/4,
        (1, 2): 0.0,
        (1, 3): 1/4
    }

    if input is None:
        inputs = [] # Initialise to |+> state
    else:
        inputs = [(0, 0), (1, 0)]

    outputs = [(0, 4), (1, 4)]

    brick_pattern = generate_from_graph(inside_graph, angles, inputs, outputs)

    return brick_pattern