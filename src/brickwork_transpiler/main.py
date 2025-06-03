import sys
from pyexpat import features

import networkx as nx
import numpy as np
import qiskit.compiler.transpiler
# from graphix.rng import ensure_rng
# from graphix.states import BasicStates
from matplotlib import pyplot as plt
# from numba.core.cgutils import sizeof
from qiskit import QuantumCircuit, ClassicalRegister
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
from src.brickwork_transpiler import decomposer, graph_builder, pattern_converter, brickwork_transpiler, qrs_knn_grover, \
    hhl
from src.brickwork_transpiler.noise import to_noisy_pattern
from src.brickwork_transpiler.visualiser import plot_graph
import src.brickwork_transpiler.circuits as circuits
from qiskit import ClassicalRegister, transpile

from graphix.pattern import Pattern
from graphix.channels import depolarising_channel

import src.brickwork_transpiler.qrs_knn_grover_adapted as qrs_knn_adapted



def main():


    # Plot HHL
    # hhl_circ = hhl.generate_example_hhl_QC()
    qc, _ = circuits.qft(3)

    # print(hhl_circ)
    # hhl_circ.draw(output='mpl',
    #                    fold=30,
    #                    )
    # plt.savefig(f"images/qft3_example_poster_before_decomposition.png", dpi=300, bbox_inches="tight")
    # plt.show()

    print("Transpiling HHL circuit...")
    bw_pattern, col_map = brickwork_transpiler.transpile(qc)

    print("Plotting brickwork graph...")
    visualiser.plot_brickwork_graph_from_pattern(bw_pattern,
                                                 node_colours=col_map,
                                                 use_node_colours=True,
                                                 title="Brickwork graph")


    return 0
    # Test the QRS


    #Experimental data
    bw_in_depths = [973, 1374 , 1980, 4082, 8857, 18758]
    user_counts = [4, 8, 16, 32, 64, 128]
    bw_aligned_depths = [1236, 1746 , 2559, 5315, 11613, 26009]
    feature_length = 6
    feature_widths = [4, 6, 8, 10, 12]

    visualiser.plot_qrs_bw_scaling(user_counts, bw_in_depths, bw_aligned_depths, feature_length)

    visualiser.plot_time_complexity_3d(user_counts, feature_widths)

    visualiser.plot_time_complexity_with_bw_lines(user_counts, feature_widths,
                                                  bw_in_depths, bw_aligned_depths,
                                                  feature_length, azim=245)

    return 0

    # --- (2) Application‐specific code that uses qrs(...) to get counts & plot ---



    # Adapted
    # # 2) Build mapping 5-bit string → user name
    # bitstring_to_name = {}
    # feature_length = len(feature_mat[0])
    # for row_vec, person in zip(feature_mat, names):
    #     bitstr = "".join(str(bit) for bit in row_vec)
    #     bitstring_to_name[bitstr] = person
    #
    # # 3) Set up parameters and run QRS
    # user_feature = "101011"
    # grover_iterations = 4
    #
    # # n_items = 4
    # qc = qrs_knn_adapted.qrs(
    #     n_items=len(feature_mat),
    #     feature_mat=feature_mat,
    #     user_vector=user_feature,
    #     plot=True,
    #     grover_iterations=grover_iterations
    # )
    #
    # # 4) Identify which qubits hold c0 and the 5 “feature” qubits
    # q = int(np.log2(len(feature_mat)))  # q = 2
    # l = feature_length  # l = 5
    # feature_qubits = list(range(q, q + l))  # [2,3,4,5,6]
    # c0_qubit = q + 2 * l  # 2 + 2*5 = 12
    #
    # # 5) Create classical registers for measurement of [c0] + [feature_qubits]
    # cr_feature = ClassicalRegister(l, name="c_feature")
    # cr_flag = ClassicalRegister(1, name="c_flag")
    # qc_meas = qc.copy()
    # qc_meas.add_register(cr_feature)
    # qc_meas.add_register(cr_flag)
    #
    # # 6) Measure c0 into cr_flag, then feature_qubits into cr_feature
    # qc_meas.measure(c0_qubit, cr_flag[0])
    # for idx, qubit in enumerate(feature_qubits):
    #     qc_meas.measure(qubit, cr_feature[idx])
    #
    # # 7) Simulate
    # sim = AerSimulator()
    # qc_transpiled = transpile(qc_meas, sim, optimization_level=3)
    # shots = 1024
    # print("Simulating…")
    # result = sim.run(qc_transpiled, shots=shots).result()
    # counts = result.get_counts()
    #
    #
    # sorted_keys = sorted(counts.keys())
    # sorted_vals = [counts[k] for k in sorted_keys]
    #
    # xtick_labels = []
    # q = int(np.log2(len(feature_mat)))  # q = 2 in this example
    #
    # for full_bitstr in sorted_keys:
    #     tokens = full_bitstr.split()  # e.g. ['0', '01010', '10', '0']
    #
    #     c0 = tokens[0]
    #     # find the token of length q=2 → that is the index‐bits
    #     idx_bits = next(tok for tok in tokens if len(tok) == q)
    #
    #     p = int(idx_bits, 2)  # convert '10'→2
    #
    #     feature_vec = feature_mat[p]  # e.g. [1,1,1,0,1]
    #     feature_str = "".join(str(bit) for bit in feature_vec)
    #
    #     hd = sum(b1 != b2 for b1, b2 in zip(feature_str, user_feature))
    #
    #     person = bitstring_to_name.get(feature_str, feature_str)
    #     label = f"{c0} {person} ({feature_str}, hd={hd})"
    #     xtick_labels.append(label)
    #
    # # Plot
    # plt.figure(figsize=(8, 4))
    # plt.bar(sorted_keys, sorted_vals)
    # plt.title(f"Recommendation counts for user vector: {user_feature} - Grover = {grover_iterations}")
    # plt.xlabel("Measured index bits → recommended person")
    # plt.ylabel(f"Counts (out of {shots})")
    # plt.xticks(ticks=sorted_keys, labels=xtick_labels, rotation=45, ha='right')
    # plt.tight_layout()
    #
    # output_path = f"images/qrs/recommendation_plot_grover_{grover_iterations}.png"
    # plt.savefig(output_path, dpi=300, bbox_inches="tight")
    #
    # plt.show()

    # QRS Grover-Based Recommendation Script
    #
    #
    # 1) Define the feature matrix and corresponding user names
    feature_mat1 = [
        [0, 0, 0, 1, 1, 0],  # Sebastian-I
        [0, 1, 0, 1, 0, 0],  # Tzula-C
        [1, 1, 1, 0, 1, 0],  # Rex-E
        [0, 1, 1, 1, 1, 0],  # Scott-T
    ]

    names1 = [
        "Sebastian-I",
        "Tzula-C",
        "Rex-E",
        "Scott-T",
    ]

    feature_mat2 = [
        [0, 0, 0, 1, 1, 0],  # Sebastian-I
        [0, 1, 0, 1, 0, 0],  # Tzula-C
        [1, 1, 1, 0, 1, 0],  # Rex-E
        [0, 1, 1, 1, 1, 0],  # Scott-T
        [0, 0, 0, 1, 1, 0],  # Sebastian-I
        [0, 1, 0, 1, 0, 0],  # Tzula-C
        [1, 1, 1, 0, 1, 0],  # Rex-E
        [0, 1, 1, 1, 1, 0],  # Scott-T
    ]

    names2 = [
        "Sebastian-I",
        "Tzula-C",
        "Rex-E",
        "Scott-T",
    ]


    # names2=[]
    # # 17 rows, 2680 columns, 1680 cx gates (no optimisation
    # feature_mat2 = [
    #     [0, 0, 0, 0, 0, 0],  # 0000 → 000000
    #     [0, 0, 0, 1, 1, 0],  # 0001 → 000110
    #     [0, 0, 1, 0, 0, 1],  # 0010 → 001001
    #     [0, 0, 1, 1, 1, 1],  # 0011 → 001111
    #     [0, 0, 0, 1, 0, 0],  # 0100 → 000100
    #     [0, 0, 0, 0, 1, 0],  # 0101 → 000010
    #     [0, 0, 1, 1, 0, 1],  # 0110 → 001101
    #     [0, 0, 1, 0, 1, 1],  # 0111 → 001011
    #     [1, 0, 0, 0, 0, 0],  # 1000 → 100000
    #     [1, 0, 0, 1, 1, 0],  # 1001 → 100110
    #     [1, 0, 1, 0, 0, 1],  # 1010 → 101001
    #     [1, 0, 1, 1, 1, 1],  # 1011 → 101111
    #     [1, 0, 0, 1, 0, 0],  # 1100 → 100100
    #     [1, 0, 0, 0, 1, 0],  # 1101 → 100010
    #     [1, 0, 1, 1, 0, 1],  # 1110 → 101101
    #     [1, 0, 1, 0, 1, 1],  # 1111 → 101011
    # ]

    names3=[]
    # 17 rows, 2680 columns, 1680 cx gates (no optimisation
    feature_mat3 = [
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

    names4=[]
    # 17 rows, 2680 columns, 1680 cx gates (no optimisation
    feature_mat4 = [
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

    names5=[]
    # 17 rows, 2680 columns, 1680 cx gates (no optimisation
    feature_mat5 = [
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

    bw_depths_aligned = []
    bw_depths_input = []

    user_feature = "101011"
    grover_iterations = 2

    feature_mats = [feature_mat1, feature_mat2, feature_mat3, feature_mat4, feature_mat5]
    names = [names1, names2, names3, names4, names5]

    # | User/IceCream | Chocolate | Vanilla | Strawberry | Nuts | Vegan |
    # | ------------- | --------- | ------- | ---------- | ---- | ----- |
    # | Rex-I          | 1         | 0       | 1          | 0    | 1     |
    # | Tzula-C       | 0         | 1       | 0          | 1    | 0     |
    # | Rex-E         | 1         | 1       | 1          | 0    | 1     |
    # | Scot-T        | 0         | 1       | 1          | 1    | 0     |

    # 1) Define the feature matrix and corresponding user names
    # Requires 831 cx gates and 1423 bricks -- Graphix doesnt compute
    # feature_mat = [
    #     [0, 0, 0, 1, 1],  # Sebastian-I
    #     [0, 1, 0, 1, 0],  # Tzula-C
    #     [1, 0, 1, 0, 1],  # Rex-E
    #     [0, 1, 1, 1, 1],  # Scott-T
    # ]
    # names = ["Sebastian-I", "Tzula-C", "Rex-E", "Scott-T"]

    for id_fm, feature_mat in enumerate(feature_mats):


        # 2) Build a mapping from each 5-bit string → user name
        bitstring_to_name = {}
        feature_length = len(feature_mat[0])  # 5 bits per feature vector

        for row_vec, person in zip(feature_mat, names[id_fm]):
            bitstr = "".join(str(bit) for bit in row_vec)
            bitstring_to_name[bitstr] = person

        # 3) Set up parameters for Grover and run QRS


        # Build the QRS circuit (4 index qubits, feature_mat, user_feature)
        qrs_circuit = qrs_knn_grover.qrs(
            n_items=len(feature_mat),
            feature_mat=feature_mat,
            user_vector=user_feature,
            plot=False,
            grover_iterations=grover_iterations
        )


        # 4) Identify which qubits hold the “recommendation” bits
        #    Here, we assume they are qubits 2–6 (5 “feature” qubits + 1 extra control)
        # measure_qubits = list(range(2, 7))
        q = int(np.log2(len(feature_mats[id_fm])))
        l = len(feature_mat[0])
        # database features are qubits q..q+l-1
        feature_qubits = list(range(q, q + l))
        print("Measure qubits:", feature_qubits)

        # 5) Create classical registers for measurement
        cr_feature = ClassicalRegister(len(feature_qubits), name="c_feature")
        cr_flag = ClassicalRegister(1, name="c_flag")

        # 6) Copy the QRS circuit and append measurement operations
        qc_meas = qrs_circuit.copy()
        qc_meas.add_register(cr_feature)
        qc_meas.add_register(cr_flag)

        # Measure each recommendation qubit into the classical register
        for idx, qubit in enumerate(feature_qubits):
            qc_meas.measure(qubit, cr_feature[idx])

        # Measure the flag qubit (second-to-last qubit in the QRS circuit)
        qc_meas.measure(qrs_circuit.num_qubits - 2, cr_flag)

        # 7) Simulate the measured circuit -- not required for graphing
        # print("Simulating...")
        # sim = AerSimulator()
        # qc_transpiled = qiskit.compiler.transpiler.transpile(qc_meas, sim, optimization_level=3)
        #
        #
        # shots=1024
        # result = sim.run(qc_transpiled, shots=shots).result()
        # raw_counts = result.get_counts()
        #
        # # 1) Keep only c0=0 shots
        # filtered_counts = {}
        # for full_bitstr, cnt in raw_counts.items():
        #     if full_bitstr[0] == '0':  # only c0=0
        #         filtered_counts[full_bitstr] = filtered_counts.get(full_bitstr, 0) + cnt
        #
        # # 2) Build sorted lists for plotting
        # sorted_keys = sorted(filtered_counts.keys())
        # sorted_vals = [filtered_counts[k] for k in sorted_keys]
        #
        # xtick_labels = []
        # for full_bitstr in sorted_keys:
        #     # leading_bit is always '0' here
        #     leading_bit = full_bitstr[0]
        #
        #     # raw_suffix = e.g. "11000" which is [c_feat4,c_feat3,c_feat2,c_feat1,c_feat0]
        #     raw_suffix = full_bitstr[-feature_length:]
        #
        #     # reverse it so that index 0→qubit2, …, index4→qubit6
        #     true_bits = raw_suffix[::-1]
        #
        #     # Hamming distance between true_bits and user_feature
        #     hd = sum(b1 != b2 for b1, b2 in zip(true_bits, user_feature))
        #
        #     if true_bits in bitstring_to_name:
        #         person = bitstring_to_name[true_bits]
        #         label = f"{person} ({true_bits} {hd})"
        #     else:
        #         label = f"No_name ({true_bits} {hd})"
        #
        #     xtick_labels.append(label)
        #
        # # 1) Compute total count
        # total = sum(sorted_vals)
        #
        # # 2) Convert each count into a percentage
        # sorted_vals_pct = [v / total * 100 for v in sorted_vals]
        #
        # # 3) Plot using those percentages
        # plt.figure(figsize=(8, 4))
        # plt.bar(sorted_keys, sorted_vals_pct)
        # plt.title(
        #     f"Recommendation for user vector: {user_feature}  –  "
        #     f"{grover_iterations} Grover iterations (post‐selected on c0=0)"
        # )
        # plt.xlabel("Measured bitstrings")
        # plt.ylabel("Recommendation prob. (%)")
        # plt.xticks(ticks=sorted_keys, labels=xtick_labels, rotation=45)
        # plt.tight_layout()
        #
        # plt.savefig(f"images/qrs/recommendation_plot_grover_{grover_iterations}.png", dpi=300, bbox_inches="tight")
        # plt.show()

        print(f"size_check = {len(feature_mat)} x {len(feature_mat[0])}")


        # Decompose to CX, rzrxrz, id   -   Need opt = 3 for SU(2) rotation merging
        decomposed_qc = decomposer.decompose_qc_to_bricks_qiskit(qrs_circuit, opt=3,
                                                                 routing_method='sabre',
                                                                 layout_method='default')

        # Optiise instruction matrix with dependency graph
        qc_mat, cx_mat = decomposer.instructions_to_matrix_dag(decomposed_qc)
        qc_mat_aligned = decomposer.align_bricks(cx_mat, qc_mat)

        bw_depths_aligned.append(len(qc_mat_aligned[0]))
        bw_depths_input.append(len(qc_mat[0]))
        print(f"feature mat: {id_fm}, aligned depth: {len(qc_mat_aligned[0])}, input depth: {len(qc_mat[0])}")

    visualiser.plot_qrs_bw_scaling(bw_depths_input, bw_depths_aligned)

    # Saved experimental data:



    # print("Transpiling circuit...")
    # bw_pattern, col_map = brickwork_transpiler.transpile(qrs_circuit)
    #
    # print("Plotting brickwork graph...")
    # visualiser.plot_brickwork_graph_from_pattern(bw_pattern,
    #                                              node_colours=col_map,
    #                                              use_node_colours=True,
    #                                              title=f"Brickwork Graph: QRS KNN+Grover({grover_iterations}) - "
    #                                                    f"feature matrix dimension: {len(feature_mat)} x {len(feature_mat[0])} - "
    #                                                    f"routing method: Sabre - "
    #                                                    f"layout method: trivial")






    #
    # # Test QRS
    #
    # feature_mat = [
    #     [0, 0, 0, 1, 1],  # Sebastian-I
    #     [0, 1, 0, 1, 0],  # Tzula-C
    #     [1, 1, 1, 0, 1],  # Rex-E
    #     [0, 1, 1, 1, 1],  # Scott-T
    # ]
    #
    # fm_lin = [[0, 0, 0, 0, 0],  # i = 0
    #     [1, 0, 1, 0, 1],  # i = 1 = g0
    #     [0, 1, 0, 1, 0],  # i = 2 = g1
    #     [1, 1, 1, 1, 1],  # i = 3 = g0 XOR g1
    # ]
    #
    # # 2) database load via G by single‐CNOTs
    # G = [
    #     [1, 0, 1, 0],  # feature qubit 0 ← index qubits 0,2
    #     [0, 1, 0, 1],  # feature qubit 1 ← index qubits 1,3
    #     [1, 1, 1, 1],  # feature qubit 2 ← index qubits 0,1,2,3
    #     [0, 1, 1, 1],  # feature qubit 3 ← index qubits 1,2,3
    #     [1, 0, 1, 0],  # feature qubit 4 ← index qubits 0,2
    # ]
    #
    # # Linear lol
    # feature_mat_paper = [
    #     [0, 0, 0, 0, 0, 0],  # 0000 → 000000
    #     [0, 0, 0, 1, 1, 0],  # 0001 → 000110
    #     [0, 0, 1, 0, 0, 1],  # 0010 → 001001
    #     [0, 0, 1, 1, 1, 1],  # 0011 → 001111
    #     [0, 0, 0, 1, 0, 0],  # 0100 → 000100
    #     [0, 0, 0, 0, 1, 0],  # 0101 → 000010
    #     [0, 0, 1, 1, 0, 1],  # 0110 → 001101
    #     [0, 0, 1, 0, 1, 1],  # 0111 → 001011
    #     [1, 0, 0, 0, 0, 0],  # 1000 → 100000
    #     [1, 0, 0, 1, 1, 0],  # 1001 → 100110
    #     [1, 0, 1, 0, 0, 1],  # 1010 → 101001
    #     [1, 0, 1, 1, 1, 1],  # 1011 → 101111
    #     [1, 0, 0, 1, 0, 0],  # 1100 → 100100
    #     [1, 0, 0, 0, 1, 0],  # 1101 → 100010
    #     [1, 0, 1, 1, 0, 1],  # 1110 → 101101
    #     [1, 0, 1, 0, 1, 1],  # 1111 → 101011
    # ]
    #
    # from qiskit import ClassicalRegister, transpile
    # import matplotlib.pyplot as plt
    #
    # user_feature = "11000"
    # grover_iterations = 2
    # qrs = qrs_knn_grover.qrs(4, feature_mat, user_feature, True, grover_iterations=grover_iterations)
    #
    # # 1) which qubits hold your “recommendation” bits?
    # #    (your comment said 4–10 → that’s 7 qubits)
    # measure_qubits = list(range(2, 7))  #
    # print("Measure qubits:", measure_qubits)
    #
    # # 2) make a classical register of the same size
    # cr = ClassicalRegister(len(measure_qubits), name='c')
    # cr0 = ClassicalRegister(1, name='cr0')
    #
    # # 3) copy & attach
    # qc_meas = qrs.copy()
    # qc_meas.add_register(cr)
    #
    # # 4) measure 4→c[0], 5→c[1], …, 10→c[6]
    # for i, q in enumerate(measure_qubits):
    #     qc_meas.measure(q, cr[i])
    #
    # qc_meas.add_register(cr0)
    # qc_meas.measure(qrs.num_qubits-2, cr0)
    #
    #
    # # 5) simulate
    # print("Simulating...")
    # sim = AerSimulator()
    # qc_t = transpile(qc_meas, sim, optimization_level=3)
    # shots = 1024
    # result = sim.run(qc_t, shots=shots).result()
    # counts = result.get_counts()
    #
    # # Associate names:
    #
    # names = [
    #     "Sebastian-I",
    #     "Tzula-C",
    #     "Rex-E",
    #     "Scott-T",
    # ]
    #
    # # Precompute a mapping from bitstring → name.
    # #    We assume each row of feature_mat corresponds exactly (in order) to names[i].
    # bitstring_to_name = {}
    # for row_vec, person in zip(feature_mat, names):
    #     # Join row_vec into a string like "00011"
    #     bitstr = "".join(str(bit) for bit in row_vec)
    #     bitstring_to_name[bitstr] = person
    #
    #
    # # 3) Sort the bitstrings and collect their counts
    # sorted_keys = sorted(counts.keys())  # e.g. ['00000', '00001', ..., '11111']
    # sorted_vals = [counts[k] for k in sorted_keys]
    # feature_length = len(feature_mat[0])  # 5
    #
    # # 4) Build x-tick labels by stripping off the first (extra) bit.
    # xtick_labels = []
    # for full_bitstr in sorted_keys:
    #     # Take only the last 5 bits for lookup:
    #     suffix = full_bitstr[-feature_length:]
    #     if suffix in bitstring_to_name:
    #         person = bitstring_to_name[suffix]
    #         # Show: Name (suffix)  –  ignoring the extra leading bit
    #         label = f"{person} ({suffix})"
    #     else:
    #         # If the 5-bit suffix doesn’t match any feature-row, fallback to showing suffix alone:
    #         label = suffix
    #     xtick_labels.append(label)
    #
    # # 5) Plot using those labels
    # plt.figure(figsize=(8, 4))
    # plt.bar(sorted_keys, sorted_vals)
    # plt.title(f"Recommendation for user vector: {user_feature}  –  {grover_iterations} amplifications")
    # plt.xlabel(f"Measured bitstring (qubits {measure_qubits[0]} … {measure_qubits[-1]})")
    # plt.ylabel(f"Counts (out of {shots})")
    #
    # # Now we replace every raw '000000', '000001', etc. with our custom labels:
    # plt.xticks(
    #     ticks=sorted_keys,
    #     labels=xtick_labels,
    #     rotation=90
    # )
    # plt.tight_layout()
    #
    # # # 6) sort & plot
    # # sorted_keys = sorted(counts.keys())  # '0000000' → '1111111'
    # # sorted_vals = [counts[k] for k in sorted_keys]
    # #
    # # plt.figure(figsize=(8, 4))
    # # plt.bar(sorted_keys, sorted_vals)
    # # plt.title(f"Recommendation for user vector: {user_feature} - {grover_iterations} amplifications")
    # # plt.xlabel(f'Measured bitstring (qubits {measure_qubits[0]} - {measure_qubits[len(measure_qubits) - 1]})')
    # # plt.ylabel(f'Counts (out of {shots})')
    # # plt.xticks(rotation=90)
    # # plt.tight_layout()
    #
    # # Save to PNG, PDF, etc.
    # plt.savefig(f"images/qrs/recommendation_plot_grover_{grover_iterations}.png", dpi=300, bbox_inches="tight")
    #
    # plt.show()

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
