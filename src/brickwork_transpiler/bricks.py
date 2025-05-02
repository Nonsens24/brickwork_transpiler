from graphix.generator import generate_from_graph


#Follows a, b, gamma, 0 scheme
def arbitrary_brick(a, b, g, input=None):

    inside_graph = visualiser.create_tuple_node_graph(1, 5)
    # measurements = {
    #     (0, 0): Measurement(2 * a, Plane.XY),
    #     (0, 1): Measurement(2 * b, Plane.XY),
    #     (0, 2): Measurement(2 * g, Plane.XY),
    #     (0, 3): Measurement(0.0, Plane.XY)
    # }

    # Angles are negative by convention (add source)
    angles = {
        (0, 0): -a,  # at node 0
        (0, 1): -b,  # at node 1
        (0, 2): -g,  # at node 2
        (0, 3): 0.0  # final teleport (X basis)
    }  #

    if input is None:
        inputs = [] #Init in + state
    else:
        inputs = [(0, 0)]

    outputs = [(0, 4)]

    og = generate_from_graph(inside_graph, angles, inputs, outputs)
    # pattern = og.to_pattern()

    return og


def make_H_J():

    G = visualiser.create_tuple_node_graph(1, 2)       # nodes: (0,0) → (0,1)
    meas = { (0,0): Measurement(0, Plane.XY) }
    inputs  = []
    outputs = [(0,1)]
    og = OpenGraph(G, meas, inputs, outputs)

    return og.to_pattern()


import networkx as nx


def make_H_brick():
    # ——— 1×5 path subgraph ———
    # Nodes: (0,0) — (0,1) — (0,2) — (0,3) — (0,4)
    inside_graph = visualiser.create_tuple_node_graph(1, 5)

    # ——— pattern measurements on the three interior nodes ———
    # Implements J(π/2)·J(0)·J(π/2) ≃ H  (up to Pauli corrections)
    measurements = {
        (0, 0): Measurement(1/2, Plane.XY),   # φ = π/2
        (0, 1): Measurement(1/2, Plane.XY),   # φ = 0
        (0, 2): Measurement(1/2, Plane.XY),   # φ = π/2
        (0, 3): Measurement(0, Plane.XY),

        # (0, 4): Measurement(1 / 2, Plane.XY),  # φ = π/2
        # (0, 5): Measurement(1 / 2, Plane.XY),  # φ = 0
        # (0, 6): Measurement(1 / 2, Plane.XY),  # φ = π/2
        # (0, 7): Measurement(0, Plane.XY)
    }

    # Exclude endpoints from measurement
    inputs  = []
    outputs = [(0, 4)]

    og = OpenGraph(inside_graph, measurements, inputs, outputs)
    pattern = og.to_pattern()

    # Feed‑forward Pauli corrections
    pattern.perform_pauli_measurements()

    return pattern


import visualiser
from graphix.opengraph import OpenGraph, Measurement
from graphix.fundamentals import Plane


def make_T_brick():
    # ——— 1×8 path subgraph ———
    # Nodes: (0,0)—(0,1)—…—(0,7)
    G = visualiser.create_tuple_node_graph(1, 3)

    # ——— chain three J‑gadgets to realize J(π/4) ≃ T up to H‑by‑products ———
    # J(0) at (0,2)&(0,3) act as wire‑extensions; J(π/4) at (0,1) does the π/8 rotation.
    measurements = {
        (0, 0): Measurement(0.125, Plane.XY),  # π/4
        (0, 1): Measurement(0.0,  Plane.XY),  # J(0)
        # (0, 2): Measurement(0.0,  Plane.XY),  # H
        # (0, 3): Measurement(0.0,  Plane.XY),  # H

    }

    # single logical input at (0,0), output at (0,7)
    inputs  = []
    outputs = [(0, 2)]

    og = OpenGraph(G, measurements, inputs, outputs)
    pattern = og.to_pattern()                # builds via flow/gflow :contentReference[oaicite:6]{index=6}
    pattern.perform_pauli_measurements()      # apply byproduct corrections :contentReference[oaicite:7]{index=7}

    return pattern

def make_CZ_brick():

    G = nx.Graph()
    # label them A, B, C, D in a square:
    #  A — B
    #  |    |
    #  C — D
    for n in ["A","B","C","D"]:
        G.add_node(n)
    G.add_edges_from([("A","B"), ("A","C"), ("B","D"), ("C","D")])

    meas = {
        "B": Measurement(0, Plane.XY),
        "C": Measurement(0, Plane.XY),
    }
    inputs  = ["A","D"]
    outputs = ["A","D"]   # A and D survive as the two output wires
    og = OpenGraph(G, meas, inputs, outputs)
    # this is exactly a CZ between A→D (up to Pauli).
    pattern = og.to_pattern()
    pattern.perform_pauli_measurements()
    return pattern
