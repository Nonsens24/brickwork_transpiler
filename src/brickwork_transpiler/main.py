import sys

import networkx as nx
import numpy as np
# from graphix.rng import ensure_rng
# from graphix.states import BasicStates
from matplotlib import pyplot as plt
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.visualization import circuit_drawer
# sys.path.append('/Users/rexfleur/Documents/TUDelft/Master_CESE/Thesis/Code/gospel')  # Full path to the cloned repo
# from gospel.brickwork_state_transpiler import generate_random_pauli_pattern
# from gospel.brick
# from gospel.brickwork_state_transpiler import (
#     generate_random_pauli_pattern,
#     # generate_random_dephasing_pattern,
#     # generate_random_depolarising_pattern,
#     # generate_random_two_qubit_depolarising_pattern,
#     # generate_random_two_qubit_depolarising_tensor_pattern,
#     generate_random_kraus_pattern,
# )
import bricks
import utils
import visualiser
from libs.gospel.gospel.brickwork_state_transpiler.brickwork_state_transpiler import generate_random_pauli_pattern
from libs.gospel.gospel.brickwork_state_transpiler.brickwork_state_transpiler import transpile
from src.brickwork_transpiler import decomposer, graph_builder, pattern_converter
from src.brickwork_transpiler.noise import to_noisy_pattern
from src.brickwork_transpiler.visualiser import plot_graph

from graphix.pattern import Pattern
from graphix.channels import depolarising_channel




def main():
    # small_graphix_example.py
    # A minimal example using Graphix MBQC library

    from graphix.transpiler import Circuit

    # 1) Define a simple 1-qubit circuit: Hadamard followed by T gate
    circuit = Circuit(5)
    circuit.rz(0, np.pi/4)
    circuit.rx(0, np.pi / 4)
    circuit.rz(0, np.pi / 4)

    circuit.rz(1, np.pi/4)
    circuit.rx(1, np.pi / 4)
    circuit.rz(1, np.pi / 4)

    circuit.rz(2, np.pi/4)
    circuit.rx(2, np.pi / 4)
    circuit.rz(2, np.pi / 4)

    circuit.rz(3, np.pi/4)
    circuit.rx(3, np.pi / 4)
    circuit.rz(3, np.pi / 4)

    circuit.rz(4, np.pi/4)
    circuit.rx(4, np.pi / 4)
    circuit.rz(4, np.pi / 4)
    circuit.cnot(0, 1)

    circuit.rz(1, np.pi/4)
    circuit.rx(1, np.pi / 4)
    circuit.rz(1, np.pi / 4)
    circuit.cnot(1, 2)

    circuit.rz(2, np.pi/4)
    circuit.rx(2, np.pi / 4)
    circuit.rz(2, np.pi / 4)
    circuit.cnot(2, 3)

    circuit.rz(3, np.pi/4)
    circuit.rx(3, np.pi / 4)
    circuit.rz(3, np.pi / 4)
    circuit.cnot(3, 4)

    circuit.rz(4, np.pi/4)
    circuit.rx(4, np.pi / 4)
    circuit.rz(4, np.pi / 4)


    # circuit.t(0)

    # 2) Transpile the circuit to an MBQC measurement pattern (brickwork)
    # pattern = circuit.transpile()

    gospel_bw = transpile(circuit)

    # gospel_bw.print_pattern()

    visualiser.plot_graphix_noise_graph(gospel_bw)

    # 4) Simulate the pattern to obtain the final statevector
    gospel_result = gospel_bw.simulate_pattern(backend='statevector')
    print("\nFinal statevector:", gospel_result)


    # 1) Create the |++> state directly
    psi = Statevector.from_label('+')  # two-qubit plus state


    qc_init_H = QuantumCircuit(2)
    qc_init_H.h(0)
    qc_init_H.h(1)
    qc_init_H.rz(np.pi/5, 0)

    # 2) Define your 2-qubit circuit (no H gates needed)
    qc = QuantumCircuit(5)
    # qc.h(0)
    # qc.rz(np.pi/2, 0)
    # qc.h(0)
    # qc.t(0)
    # qc.t(0)
    # qc.t(1)
    qc.rz(np.pi / 4, 0)
    qc.rx(np.pi / 4, 0)
    qc.rz(np.pi / 4, 0)
    qc.cx(0, 1)

    qc.rz(np.pi / 4, 1)
    qc.rx(np.pi / 4, 1)
    qc.rz(np.pi / 4, 1)
    qc.cx(0, 1)

    qc.rz(np.pi / 4, 2)
    qc.rx(np.pi / 4, 2)
    qc.rz(np.pi / 4, 2)
    qc.cx(1, 2)

    qc.rz(np.pi / 4, 3)
    qc.rx(np.pi / 4, 3)
    qc.rz(np.pi / 4, 3)
    qc.cx(2, 3)

    qc.rz(np.pi / 4, 4)
    qc.rx(np.pi / 4, 4)
    qc.rz(np.pi / 4, 4)
    qc.cx(3, 4)
    # qc.cx(1, 0)
    # qc.rx(np.pi/3, 1)
    # qc.cx(0, 1)
    # qc.rz(np.pi/2, 2)
    # qc.rx(-np.pi/3, 2)
    # qc.rz(-np.pi/4, 2)
    # qc.cx(1, 2)
    # qc.cx(3, 2)
    # qc.rx(np.pi/3, 3)
    # qc.rz(np.pi/2, 2)
    # qc.rz(-np.pi/4, 4)
    # qc.cx(3, 4)
    # qc.rz(np.pi/2, 4)
    # qc.rz(np.pi / 2, 3)
    # qc.rz(np.pi / 2, 4)
    # qc.rx(np.pi / 2, 4)
    #
    # qc.rz(np.pi / 2, 6)
    # qc.rz(np.pi / 2, 7)
    # qc.rx(np.pi / 2, 4)
    # qc.cx(5, 6)
    # qc.rz(np.pi / 2, 3)
    # qc.rz(np.pi / 2, 6)
    # qc.rx(np.pi / 2, 5)
    # qc.cx(6, 7)
    # qc.rz(np.pi / 2, 4)
    # qc.rx(np.pi / 2, 5)
    # qc.rz(np.pi / 2, 6)
    # qc.rz(np.pi / 2, 7)

    # 2) Draw as an mpl Figure
    #    output='mpl' returns a matplotlib.figure.Figure
    # fig = circuit_drawer(qc, output='mpl', style={'dpi': 150})

    # 3) (Optional) tweak size or DPI
    # fig.set_size_inches(6, 4)  # width=6in, height=4in
    # 150 dpi × 6in = 900px wide, for instance

    # 4) Save to disk in any vector or raster format
    # fig.savefig("qc_diagram.svg", format="svg", bbox_inches="tight")  # vector
    # fig.savefig("qc_diagram.pdf", format="pdf", bbox_inches="tight")  # vector
    # fig.savefig("qc_diagram.png", format="png", dpi=300, bbox_inches="tight")  # raster

    # 5) If you just want to display in a script or notebook:
    plt.show()




    decomposed_qc = decomposer.decompose_qc_to_bricks_qiskit(qc, 3)

    qc_mat = decomposer.instructions_to_matrix_dag(decomposed_qc)
    visualiser.print_matrix(qc_mat)

    qc_mat = decomposer.align_bricks(qc_mat)
    visualiser.print_matrix(qc_mat)

    bw_graph_data = graph_builder.generate_brickwork_graph_from_instruction_matrix(qc_mat)

    bw_nx_graph = graph_builder.to_networkx_graph(bw_graph_data)

    visualiser.plot_graph(bw_nx_graph)

    bw_pattern, col_map = pattern_converter.to_pattern(qc_mat, bw_nx_graph)
    # bw_pattern.print_pattern(lim = 10000)

    print("printing my bw")
    bw_pattern.print_pattern(lim=1000)

    print("pritnign gospel")
    gospel_bw.print_pattern(lim=1000)

    # Noise
    # bw_noisy = to_noisy_pattern(bw_pattern, 0.01, 0.005)

    n_qubits = 8  # your existing brickwork graph :contentReference[oaicite:3]{index=3}
    n_layers = len(qc_mat[0]) + 2  # e.g. nx.diameter(bw_nx_graph)
    # print(f"mat len: {len(qc_mat[0]) * 4 + 1}")

    # Sample a random‐Pauli measurement pattern
    # rng = ensure_rng(42)  # reproducible RNG :contentReference[oaicite:4]{index=4}
    # noise_graph = generate_random_pauli_pattern(n_qubits, n_layers)

    import networkx as nx

    # # 1. Get graphs from patterns
    # nodes_ng, edges_ng = noise_graph.get_graph()
    # nodes_bw, edges_bw = bw_pattern.get_graph()
    #
    # # 2. Build NetworkX Graphs
    # G_ng = nx.Graph()
    # G_ng.add_nodes_from(nodes_ng)
    # G_ng.add_edges_from(edges_ng)
    #
    # G_bw = nx.Graph()
    # G_bw.add_nodes_from(nodes_bw)
    # G_bw.add_edges_from(edges_bw)
    #
    # # 3. Use VF2 isomorphism algorithm to find mapping
    # from networkx.algorithms import isomorphism
    #
    # GM = isomorphism.GraphMatcher(G_ng, G_bw)
    # if GM.is_isomorphic():
    #     node_mapping = GM.mapping  # Maps NG node ID → BW (row, col)
    #     reverse_mapping = {v: k for k, v in node_mapping.items()}  # Optional
    #     print("Node mapping:", node_mapping)
    # else:
    #     print("Graphs are not isomorphic — mapping failed.")

    # print(f"NG_rev_map: {reverse_mapping}")

    # noise_graph.print_pattern(lim = 10000)
    print(f"BW: {bw_pattern}")
    # bw_pattern.print_pattern(lim = 10000)


    # visualiser.plot_brickwork_graph_from_pattern(noise_graph,
    #                                              node_colours=col_map,
    #                                              use_node_colours=True,
    #                                              title="Brickwork Graph: main")

    visualiser.plot_brickwork_graph_from_pattern(bw_pattern,
                                                 node_colours=col_map,
                                                 use_node_colours=True,
                                                 title="Brickwork Graph: main")

    # noise_graph = generate_random_pauli_pattern(n_qubits, n_layers)
    # visualiser.plot_graphix_noise_graph(noise_graph, save=True)

    # Assume 'pattern' is your existing measurement pattern
    # Define a depolarizing channel with a probability of 0.05
    # depolarizing = depolarising_channel(prob=0.01)

    # Apply the depolarizing channel to qubit 0
    # bw_pattern.(depolarizing)

    # visualiser.plot_brickwork_graph_from_pattern(bw_pattern,
    #                                              node_colours=col_map,
    #                                              use_node_colours=True,
    #                                              title="Brickwork Graph: Noise Injected")

    # visualiser.visualize_brickwork_graph(bw_pattern)

    # visualiser.plot_brickwork_graph_from_pattern(bw_noisy,
    #                                              node_colours=col_map,
    #                                              use_node_colours=True,
    #                                              title="Brickwork Graph: main")

    print("Starting simulation of bw pattern. This might take a while...")
    # outstate = bw_pattern.simulate_pattern(backend='statevector').flatten()
    # print("Graphix simulator output:", outstate)

    bw_pattern.standardize()
    bw_pattern.shift_signals()

    visualiser.plot_brickwork_graph_from_pattern(bw_pattern,
                                                 node_colours=col_map,
                                                 use_node_colours=True,
                                                 title="Brickwork Graph: main after signal shift and standardisation")

    bw_pattern.print_pattern(lim=1000)

    outstate = bw_pattern.simulate_pattern(backend='statevector')

    # Calculate reference statevector
    # psi_out = psi.evolve(qc)
    # print("Qiskit reference state vector: ", psi_out.data)

    # sv2 = Statevector.from_instruction(qc).data
    # print("Qiskit reference output: ", sv2)

    ref_state = Statevector.from_instruction(qc_init_H).data
    print(f"Qiskit ref_state: {ref_state}")
    # if utils.assert_equal_up_to_global_phase(gospel_result.flatten(), ref_state.data):
    #     print("GOSPEL QISKIT Equal up to global phase")

    if utils.assert_equal_up_to_global_phase(gospel_result.flatten(), outstate.flatten()):
        print("GOSPEL MYTP Equal up to global phase")

    # if utils.assert_equal_up_to_global_phase(outstate, ref_state.data):
    #     print("Equal up to global phase")

    # print("Laying a brick:")
    # pattern = bricks.arbitrary_brick(1/4, 1/4, 1/4)
    # pattern.print_pattern()
    #
    # # TODO: get graph structure from pattern
    # # visualiser.plot_graph(pattern)
    #
    # # ARbitrary Rotation gate:
    # outstate = pattern.simulate_pattern(backend='statevector').flatten()
    # print("brick MBQC output:", outstate)
    #
    # qc = QuantumCircuit(1)
    #
    # qc.h(0)
    # qc.rz(np.pi * 1/4, 0)
    # qc.rx(np.pi * 1/4, 0)
    # qc.rz(np.pi * 1/4, 0)

    # CX gate:
    # print("Laying a brick:")
    # pattern = bricks.CX_bottom_target_brick()
    #
    # outstate = pattern.simulate_pattern(backend='statevector').flatten()
    # print("brick MBQC output:", outstate)
    #
    # qc = QuantumCircuit(2)
    #
    # # Initialise to |+>
    # qc.h(0)
    # qc.h(1)
    #
    # # cnot them
    # qc.cx(0, 1)
    #
    # sv2 = Statevector.from_instruction(qc).data
    # print("reference output: ", sv2)
    #
    # if utils.assert_equal_up_to_global_phase(outstate, sv2):
    #     print("Same up to global phase!")


if __name__ == "__main__":
    main()
