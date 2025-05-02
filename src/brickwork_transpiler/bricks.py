import networkx as nx
from graphix.generator import generate_from_graph
from src.brickwork_transpiler import graph_builder


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
def CZ_brick():

    G = nx.Graph()
    # label them A, B, C, D in a square:
    #  A — B
    #  |    |
    #  C — D
    for n in ["A","B","C","D"]:
        G.add_node(n)
    G.add_edges_from([("A","B"), ("A","C"), ("B","D"), ("C","D")])

    # meas = {
    #     "B": Measurement(0, Plane.XY),
    #     "C": Measurement(0, Plane.XY),
    # }

    inputs  = ["A","D"]
    outputs = ["A","D"]   # A and D survive as the two output wires
    # og = OpenGraph(G, meas, inputs, outputs) # -------- Change this to generate_from_graph!!!
    # this is exactly a CZ between A→D (up to Pauli).
    # pattern = og.to_pattern()
    # pattern.perform_pauli_measurements()
    # return pattern