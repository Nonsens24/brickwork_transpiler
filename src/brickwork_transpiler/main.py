import sys

import networkx as nx
import numpy as np
# from graphix.rng import ensure_rng
# from graphix.states import BasicStates
from matplotlib import pyplot as plt
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.visualization import circuit_drawer, plot_histogram
from qiskit_aer import AerSimulator

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
import tests.system_tests.test_system_single_bricks
import utils
import visualiser
from libs.gospel.gospel.brickwork_state_transpiler.brickwork_state_transpiler import generate_random_pauli_pattern
from libs.gospel.gospel.brickwork_state_transpiler.brickwork_state_transpiler import transpile
from src.brickwork_transpiler import decomposer, graph_builder, pattern_converter, brickwork_transpiler, qrs_knn_grover
from src.brickwork_transpiler.noise import to_noisy_pattern
from src.brickwork_transpiler.visualiser import plot_graph
import src.brickwork_transpiler.circuits as circuits

from graphix.pattern import Pattern
from graphix.channels import depolarising_channel




def main():


    # Test QRS

    feature_mat = [
        [1, 0, 0, 0, 1],  # Sebastian-I
        [0, 1, 0, 1, 0],  # Tzula-C
        [1, 1, 1, 0, 1],  # Rex-E
        [0, 1, 1, 1, 1],  # Scott-T
    ]

    fm_lin = [[0, 0, 0, 0, 0],  # i = 0
        [1, 0, 1, 0, 1],  # i = 1 = g0
        [0, 1, 0, 1, 0],  # i = 2 = g1
        [1, 1, 1, 1, 1],  # i = 3 = g0 XOR g1
    ]

    # 2) database load via G by single‐CNOTs
    G = [
        [1, 0, 1, 0],  # feature qubit 0 ← index qubits 0,2
        [0, 1, 0, 1],  # feature qubit 1 ← index qubits 1,3
        [1, 1, 1, 1],  # feature qubit 2 ← index qubits 0,1,2,3
        [0, 1, 1, 1],  # feature qubit 3 ← index qubits 1,2,3
        [1, 0, 1, 0],  # feature qubit 4 ← index qubits 0,2
    ]

    # Linear lol
    feature_mat_paper = [
        [0, 0, 0, 0, 0, 0],  # 0000 → 000000
        [0, 0, 0, 1, 1, 0],  # 0001 → 000110
        [0, 0, 1, 0, 0, 1],  # 0010 → 001001
        [0, 0, 1, 1, 1, 1],  # 0011 → 001111
        [0, 0, 0, 1, 0, 0],  # 0100 → 000100
        [0, 0, 0, 0, 1, 0],  # 0101 → 000010
        [0, 0, 1, 1, 0, 1],  # 0110 → 001101
        [0, 0, 1, 0, 1, 1],  # 0111 → 001011
        [1, 0, 0, 0, 0, 0],  # 1000 → 100000
        [1, 0, 0, 1, 1, 0],  # 1001 → 100110
        [1, 0, 1, 0, 0, 1],  # 1010 → 101001
        [1, 0, 1, 1, 1, 1],  # 1011 → 101111
        [1, 0, 0, 1, 0, 0],  # 1100 → 100100
        [1, 0, 0, 0, 1, 0],  # 1101 → 100010
        [1, 0, 1, 1, 0, 1],  # 1110 → 101101
        [1, 0, 1, 0, 1, 1],  # 1111 → 101011
    ]

    from qiskit import ClassicalRegister, transpile
    import matplotlib.pyplot as plt

    user_feature = "11000"
    qrs = qrs_knn_grover.qrs(4, feature_mat, user_feature, True, grover_iterations=2)

    # 1) which qubits hold your “recommendation” bits?
    #    (your comment said 4–10 → that’s 7 qubits)
    measure_qubits = list(range(2, 7))  #
    print("Measure qubits:", measure_qubits)

    # 2) make a classical register of the same size
    cr = ClassicalRegister(len(measure_qubits), name='c')
    cr0 = ClassicalRegister(1, name='cr0')

    # 3) copy & attach
    qc_meas = qrs.copy()
    qc_meas.add_register(cr)

    # 4) measure 4→c[0], 5→c[1], …, 10→c[6]
    for i, q in enumerate(measure_qubits):
        qc_meas.measure(q, cr[i])

    qc_meas.add_register(cr0)
    qc_meas.measure(qrs.num_qubits-2, cr0)

    # 5) simulate
    print("Simulating...")
    sim = AerSimulator()
    qc_t = transpile(qc_meas, sim, optimization_level=3)
    shots = 1024
    result = sim.run(qc_t, shots=shots).result()
    counts = result.get_counts()

    # 6) sort & plot
    sorted_keys = sorted(counts.keys())  # '0000000' → '1111111'
    sorted_vals = [counts[k] for k in sorted_keys]

    plt.figure(figsize=(8, 4))
    plt.bar(sorted_keys, sorted_vals)
    plt.title(f"Recommendation for user vector: {user_feature}")
    plt.xlabel(f'Measured bitstring (qubits {measure_qubits[0]} - {measure_qubits[len(measure_qubits) - 1]})')
    plt.ylabel(f'Counts (out of {shots})')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

    # GRAPHING OF BW GROWTH:

    # circuit_depths = []
    # circuit_sizes = []

    # qc, input_vector = circuits.qft(3)
    #
    # bw_pattern, col_map = brickwork_transpiler.transpile(qc, input_vector)
    #
    # circuit_depths.append(bw_pattern.get_graph().__sizeof__())
    # print("sizeof: ", len(bw_pattern.get_angles()))


    # n = 24
    #
    # bw_depths = []
    #
    # for i in range(1, n):
    #     qc, _ = circuits.qft(i)
    #
    #     # Decompose to CX, rzrxrz, id   -   Need opt = 3 for SU(2) rotation merging
    #     decomposed_qc = decomposer.decompose_qc_to_bricks_qiskit(qc, opt=3,
    #                                                              routing_method='sabre',
    #                                                              layout_method='default')
    #
    #     # Optiise instruction matrix with dependency graph
    #     qc_mat, cx_mat = decomposer.instructions_to_matrix_dag(decomposed_qc)
    #     qc_mat_aligned = decomposer.align_bricks(cx_mat, qc_mat)
    #
    #     bw_depths.append(len(qc_mat_aligned[0]))
    #     print(f"i: {i}, bricks: {len(qc_mat_aligned[0])}")
    #
    #
    # visualiser.plot_qft_complexity(n-1, bw_depths)
    # END GRAPHING


    # n = 8
    # layout_method = "default"
    # routing_method = "stochastic"
    #
    # for i in range(1, 8):
    #     qc, input_vector = circuits.qft(i)
    #
    #     bw_pattern, col_map= brickwork_transpiler.transpile(qc, input_vector)
    #
    #     if i < 1:
    #         visualiser.plot_brickwork_graph_from_pattern(bw_pattern,
    #                                                      node_colours=col_map,
    #                                                      use_node_colours=True,
    #                                                      title=f"Brickwork Graph: QFT({i}) - "
    #                                                            f"routing method: Sabre - "
    #                                                            f"layout method: trivial")
    #
    #     # Always is an integer because the graph is divisable by the amount of nodes -- rectangle
    #     circuit_depth = int(len(bw_pattern.get_angles()) + len(bw_pattern.output_nodes) / len(bw_pattern.output_nodes))
    #     circuit_depths.append(circuit_depth)
    #     circuit_sizes.append(len(bw_pattern) + len(bw_pattern.output_nodes))
    #
    # visualiser.plot_depths(circuit_depths,
    #                        subtitle=f"QFT 1 to {n} qubits",
    #                        routing_method=routing_method,
    #                        layout_method=layout_method)
    #
    # visualiser.plot_depths(circuit_sizes,
    #                        title="Circuit Size vs. Input Size",
    #                        subtitle=f"QFT 1 to {n} qubits",
    #                        routing_method=routing_method,
    #                        layout_method=layout_method)


    # visualiser.plot_qft_complexity(n-1, circuit_depths)


    return 0

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


    # Noise
    # bw_noisy = to_noisy_pattern(bw_pattern, 0.01, 0.005)

    # n_qubits = 8  # your existing brickwork graph :contentReference[oaicite:3]{index=3}
    # n_layers = len(qc_mat[0]) + 2  # e.g. nx.diameter(bw_nx_graph)
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
    # bw_pattern.print_pattern(lim = 10000)


    bw_pattern, ref_state, col_map= brickwork_transpiler.transpile(qc, input_vector)


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

    # bw_pattern.print_pattern(lim=1000)

    outstate = bw_pattern.simulate_pattern(backend='statevector')

    # Calculate reference statevector
    # psi_out = psi.evolve(qc)
    # print("Qiskit reference state vector: ", psi_out.data)

    # sv2 = Statevector.from_instruction(qc).data
    # print("Qiskit reference output: ", sv2)

    # ref_state = Statevector.from_instruction(qc_init_H).data
    # print(f"Qiskit ref_state: {ref_state}")
    # # if utils.assert_equal_up_to_global_phase(gospel_result.flatten(), ref_state.data):
    # #     print("GOSPEL QISKIT Equal up to global phase")
    #
    # if utils.assert_equal_up_to_global_phase(gospel_result.flatten(), outstate.flatten()):
    #     print("GOSPEL MYTP Equal up to global phase")

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
