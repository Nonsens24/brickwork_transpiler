import qiskit.compiler.transpiler
# from graphix.rng import ensure_rng
# from graphix.states import BasicStates
# from numba.core.cgutils import sizeof

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
import visualiser
from libs.gospel.gospel.brickwork_state_transpiler.brickwork_state_transpiler import transpile
from src.brickwork_transpiler import brickwork_transpiler
from src.brickwork_transpiler.algorithms import qrs_knn_grover
from src.brickwork_transpiler.bfk_encoder import bfk_encoder
from src.brickwork_transpiler.noise import DepolarisingInjector
# from src.brickwork_transpiler.noise import to_noisy_pattern
import src.brickwork_transpiler.circuits as circuits

from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit import QuantumCircuit


def feature_oracle(feature_qubits, user_feature, total_qubits):
    n = len(feature_qubits)
    oracle = QuantumCircuit(total_qubits)
    for i, bit in enumerate(user_feature):
        if bit == 0:
            oracle.x(feature_qubits[i])
    oracle.h(feature_qubits[-1])
    if n == 1:
        oracle.z(feature_qubits[0])
    else:
        oracle.mcx(feature_qubits[:-1], feature_qubits[-1])
    oracle.h(feature_qubits[-1])
    for i, bit in enumerate(user_feature):
        if bit == 0:
            oracle.x(feature_qubits[i])
    return oracle

# ... after generating qrs_circ ...


def main():
    from qiskit import Aer
    from qiskit.utils import QuantumInstance
    # from qiskit.algorithms import AmplificationProblem, Grover
    from qiskit_algorithms import AmplificationProblem, Grover
    import numpy as np
    from qiskit_aer.primitives import Sampler as AerSampler

    # --- DATA ---
    feature_mat = [
        [1, 0, 1, 1, 1, 1],
        [0, 1, 0, 1, 0, 0],
        [0, 0, 1, 1, 1, 0],
        [0, 0, 0, 1, 0, 1],
        [0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1],
        [1, 1, 0, 0, 1, 0],
    ]
    user_feature = [0, 0, 0, 0, 0, 1]
    grover_iterations = None  # None for optimal, or set a specific int

    n_items = len(feature_mat)
    num_id_qubits = int(np.log2(n_items))
    num_db_feature_qubits = len(feature_mat[0])
    feature_qubits = list(range(num_id_qubits, num_id_qubits + num_db_feature_qubits))

    # --- BUILD CIRCUIT ---
    qrs_circ = qrs_knn_grover.qrs(
        n_items=n_items,
        feature_mat=feature_mat,
        user_vector=user_feature,
        plot=False,
        grover_iterations=None  # None lets Qiskit calculate the optimal amount
    )

    # --- FLAG QUBIT: Robust detection by register name ---
    flag_reg = [reg for reg in qrs_circ.qregs if reg.name == "c0"]
    assert len(flag_reg) == 1, "Flag register 'c0' not found or ambiguous!"
    flag_qubit = qrs_circ.find_bit(flag_reg[0][0]).index  # The only qubit in c0


    from matplotlib import pyplot as plt
    from qiskit import QuantumCircuit

    # ORACLE: Flag flips if feature_qubits match user_feature subset (ignores 'X')
    def feature_oracle_subset(feature_qubits, user_feature, flag_qubit, total_qubits):
        qc = QuantumCircuit(total_qubits)
        active_controls = []
        for q, bit in zip(feature_qubits, user_feature):
            if bit == 'X':
                continue
            if bit == 0:
                qc.x(q)
            active_controls.append(q)

        qc.mcx(active_controls, flag_qubit)

        for q, bit in zip(feature_qubits, user_feature):
            if bit == 'X':
                continue
            if bit == 0:
                qc.x(q)

        return qc


    feature_subset = ['X', 0, 0, 'X', 0, 'X']
    oracle_subset = feature_oracle_subset(feature_qubits, feature_subset, flag_qubit, qrs_circ.num_qubits)


    # Wrapper to match the required signature of Grover's AmplificationProblem
    def create_is_good_state(feature_qubits, user_feature):
        def is_good_state(bitstring):
            feat = ''.join(bitstring[-1 - q] for q in sorted(feature_qubits))
            for f_bit, u_bit in zip(feat, user_feature):
                if u_bit == 'X':
                    continue
                if f_bit != str(u_bit):
                    return False
            return True

        return is_good_state

    # Usage when defining your AmplificationProblem:
    problem = AmplificationProblem(
        oracle=oracle_subset,
        state_preparation=qrs_circ,
        is_good_state=create_is_good_state(feature_qubits, user_feature)
    )


    # --- RUN GROVER ---
    sampler = AerSampler(run_options={"shots": 2048})
    grover = Grover(sampler=sampler, iterations=grover_iterations)
    result = grover.amplify(problem)


    # ① Get the raw list of histograms (one dict per executed circuit)
    raw_list = result.circuit_results

    # ② Turn it into a single plain dict of counts
    if not raw_list:
        raise ValueError("No circuit results found in GroverResult")
    elif len(raw_list) == 1 and isinstance(raw_list[0], dict):
        counts = raw_list[0]  # the one histogram you want
    else:
        # If there are multiple runs, merge their histograms
        counts = {}
        for entry in raw_list:
            if not isinstance(entry, dict):
                continue
            for bitstr, c in entry.items():
                counts[bitstr] = counts.get(bitstr, 0) + c

    # ③ Now ‘counts’ is exactly {bitstring: shot_count}
    #     Proceed with your existing post‑selection:

    # … after you’ve built `counts` as a {bitstring: shots} dict …

    filtered = {}
    for bitstring, shots in counts.items():
        # 1) pull out the flag qubit value:
        flag_bit = bitstring[-1 - flag_qubit]
        if flag_bit != '0':
            continue

        # 2) extract only the feature bits, in MSB→LSB order:
        feat_bits = ''.join(
            bitstring[-1 - q]
            for q in sorted(feature_qubits)
        )

        # 3) tally
        filtered[feat_bits] = filtered.get(feat_bits, 0) + shots

    print(filtered)  # keys will now be length‑6 strings like '100001'

    # filtered now has just the post-selected counts.

    # --- PLOT ---
    if filtered:
        sorted_bits = sorted(filtered)
        sorted_counts = [filtered[b] for b in sorted_bits]
        total = sum(sorted_counts)
        user_bits = ''.join(map(str, user_feature))
        xticks = []
        for b in sorted_bits:
            hd = sum(c != u for c, u in zip(b, user_bits))
            xticks.append(f"{b} (HD={hd})")
        probs = [100 * v / total for v in sorted_counts]
        plt.figure(figsize=(10, 5))
        plt.bar(range(len(probs)), probs)
        plt.xticks(range(len(probs)), xticks, rotation=45, ha='right')
        plt.title(f"Grover-QKNN amplified | User: {user_feature}")
        plt.ylabel("Probability (%)")
        plt.tight_layout()
        plt.show()
    else:
        print("No 'good' states detected in the results.")

    # # --- Find the flag qubit index robustly ---
    # flag_reg = [reg for reg in qrs_circ.qregs if reg.name == "c0"]
    # assert len(flag_reg) == 1, "Flag register 'c0' not found or ambiguous!"
    # flag_qubit = qrs_circ.find_bit(flag_reg[0][0]).index  # The only qubit in c0
    #
    # # --- Attach classical registers and measure ---
    # cr_feat = ClassicalRegister(num_db_feature_qubits, 'c_feat')
    # cr_flag = ClassicalRegister(1, 'c_flag')
    # qrs_circ.add_register(cr_feat)
    # qrs_circ.add_register(cr_flag)
    #
    # for idx, q in enumerate(feature_qubits):
    #     qrs_circ.measure(q, cr_feat[idx])
    # qrs_circ.measure(flag_qubit, cr_flag[0])
    #
    # # --- Simulate ---
    # simulator = Aer.get_backend('aer_simulator')
    # result = execute(qrs_circ, simulator, shots=2048).result()
    # counts = result.get_counts()
    #
    # # --- Post-select on flag == '0' ---
    # filtered = {}
    # for key, val in counts.items():
    #     toks = key.split()
    #     if len(toks) == 2:  # 'flag feat'
    #         flag, feat = toks[0], toks[1]
    #     else:  # fallback
    #         flag, feat = key[-1], key[:-1]
    #     if flag != '0':
    #         continue
    #     filtered[feat] = filtered.get(feat, 0) + val
    #
    # # --- Sort, annotate, and plot ---
    # sorted_bits = sorted(filtered)
    # sorted_counts = [filtered[b] for b in sorted_bits]
    # user_bits = ''.join(map(str, user_feature))
    # xticks = []
    # for b in sorted_bits:
    #     hd = sum(c != u for c, u in zip(b, user_bits))
    #     xticks.append(f"{b} (HD={hd})")
    #
    # probs = [100 * v / sum(sorted_counts) for v in sorted_counts]
    # plt.figure(figsize=(10, 5))
    # plt.bar(range(len(probs)), probs)
    # plt.xticks(range(len(probs)), xticks, rotation=45, ha='right')
    # plt.title(f"QKNN result | User: {user_feature} | Grover iters: {grover_iterations}")
    # plt.ylabel("Probability (%)")
    # plt.tight_layout()
    # plt.show()

    return 0

    #
    # total_qubits = qrs_circ.num_qubits
    # oracle = feature_oracle(feature_qubits, user_feature, total_qubits)
    #
    # # "Good" state is the feature '100000' (if feature qubits are 3..8 and user_feature = [0,0,0,0,0,1])
    # def is_good_state(bitstring):
    #     # Extract feature bits: in Qiskit, bitstring[0] is qubit 0 (rightmost)
    #     # Feature qubits = 3-8 --> bits 3,4,5,6,7,8 in Qiskit order
    #     feature_bits = ''.join([bitstring[i] for i in range(3, 9)])
    #     return feature_bits == '100001'  # set to the bitstring for user_feature in little-endian
    #
    # print("Problem amplification statement...")
    # problem = AmplificationProblem(
    #     oracle=oracle,
    #     state_preparation=qrs_circ,
    #     is_good_state=is_good_state
    # )
    #
    # from collections import Counter
    # from qiskit.visualization import plot_histogram
    # import matplotlib.pyplot as plt
    #
    #
    #
    #
    #
    # def extract_feature_bits(bitstring, feature_qubits):
    #     # Qiskit: rightmost is qubit 0; so bitstring[-1-q] gives qubit q
    #     return ''.join(bitstring[-1 - q] for q in sorted(feature_qubits))
    #
    # print("Started simulation...")
    # backend = Aer.get_backend('qasm_simulator')
    # grover = Grover(quantum_instance=QuantumInstance(backend, shots=1024))
    # result = grover.amplify(problem)
    # # print("Number of Grover iterations performed:", result.iterations)
    #
    # qrs_circ.draw(output='mpl',
    #              fold=40,
    #              )
    # plt.savefig(f"images/qrs/recommendation_circ{grover_iterations}.png", dpi=300, bbox_inches="tight")
    # plt.show()
    #
    # # Get counts for feature bits only
    # counts = result.circuit_results
    # feature_counts = Counter()
    # print("\nDiagnostics:")
    # print("Feature qubits (indices):", feature_qubits)
    # print("User feature:", user_feature)
    # print("Expecting to amplify feature pattern (little endian):", ''.join(str(x) for x in user_feature[::-1]))
    #
    # for hist in counts:
    #     for bitstring, cnt in hist.items():
    #         print(f"Full measured bitstring: {bitstring}  Count: {cnt}")
    #         feature_bits = extract_feature_bits(bitstring, feature_qubits)
    #         print(
    #             f"  Extracted feature bits: {feature_bits} (should match {''.join(str(x) for x in user_feature[::-1])})")
    #         feature_counts[feature_bits] += cnt
    #
    # print("\nAggregated feature bitstring counts:")
    # for fb, cnt in sorted(feature_counts.items(), key=lambda x: -x[1]):
    #     print(f"{fb}: {cnt}")
    #
    # print("\nAmplified feature bitstring counts:")
    # for fb, cnt in sorted(feature_counts.items(), key=lambda x: -x[1]):
    #     print(f"{fb}: {cnt}")
    #
    # fig = plt.figure(figsize=(14, 6))  # Wider figure
    # ax = fig.add_subplot(111)
    # plot_histogram(feature_counts, ax=ax, bar_labels=False)
    #
    # plt.xticks(rotation=70, ha='right', fontsize=10)  # Rotate x labels for clarity
    # plt.tight_layout()  # Adjust layout to fit everything
    # plt.show()

    # import time
    # t0 = time.time()
    # problem = AmplificationProblem(oracle=oracle, state_preparation=qrs_circ, is_good_state=is_good_state)
    # print("AmplificationProblem constructed in", time.time() - t0, "seconds")
    # t1 = time.time()
    # sampler = Sampler(options={"shots": 64})
    # grover = Grover(sampler=sampler)
    # result = grover.amplify(problem)
    # print("Grover ran in", time.time() - t1, "seconds")
    #
    # print("Top measurement:", result.top_measurement)
    # print("Histogram:", result.circuit_results)
    # plot_histogram(result.circuit_results)
    # plt.show()

    # return 0

# if __name__ == "__main__":
#     main()



    # QRS_CHECKED
    # === 1) DATA ===
    feature_mat = [
        [1, 0, 1, 1, 1, 1],  # 000110
        [0, 1, 0, 1, 0, 0],  # 010000
        [0, 0, 1, 1, 1, 0],  # 111010
        [0, 0, 0, 1, 0, 1],  # 011110
        [1, 0, 0, 0, 1, 0],  # 000110 (duplicate)
        [1, 0, 0, 0, 0, 0],  # 000100
        [1, 1, 1, 1, 1, 1],  # 101010
        [1, 1, 0, 0, 1, 0],  # 010101 (exact match)
    ]
    # user_feature = [0, 1, 0, 1, 0, 1]  # six bits
    user_feature = [0, 0, 0, 0, 0, 1]  # six bits
    grover_iterations = 0

    # --- Build the QRS circuit (no measurements) ---
    qrs_circ = qrs_knn_grover_checked.qrs(
        n_items=len(feature_mat),
        feature_mat=feature_mat,
        user_vector=user_feature,
        plot=False,
        grover_iterations=grover_iterations
    )

    # === 2) Transpile & fetch layout ===
    sim = AerSimulator()
    qc_t = transpile(qrs_circ, sim, optimization_level=3)

    # invert mapping if needed, or identity if trivial
    layout_obj = qc_t.layout or qc_t._layout
    if layout_obj is None:
        v2p = {i: i for i in range(qc_t.num_qubits)}
    else:
        # Terra ≥0.23: layout.virtual_to_physical_map()
        v2p = layout_obj.get_virtual_bits()  # mapping: virt → phys

    # === 3) Figure out which physical wires hold the feature & flag bits ===
    q = int(np.log2(len(feature_mat)))
    l = len(feature_mat[0])
    logical_feat = list(range(q, q + l))
    feature_phys = [v2p[lq] for lq in logical_feat]
    flag_phys = v2p[qrs_circ.num_qubits - 2]

    print("physical wires for feature bits:", feature_phys)
    print("physical wire for flag bit:   ", flag_phys)

    # === 4) Attach classical registers & measure ===
    # qc_meas = qc_t.copy()
    cr_feat = ClassicalRegister(l, "c_feature")
    cr_flag = ClassicalRegister(1, "c_flag")
    qc_t.add_register(cr_feat)
    qc_t.add_register(cr_flag)

    # Measure feature qubits in ascending order 3→c[0], …, 8→c[5]
    for cb, qphys in enumerate(sorted(feature_phys)):
        qc_t.measure(qphys, cr_feat[cb])

    # Then measure the flag
    qc_t.measure(flag_phys, cr_flag[0])

    qc_t.draw(output='mpl',
                       fold=40,
                       )
    plt.savefig(f"images/qrs/recommendation_circ{grover_iterations}.png", dpi=300, bbox_inches="tight")
    plt.show()

    # === 5) Simulate & get counts ===
    shots = 2048
    raw_counts = sim.run(qc_t, shots=shots).result().get_counts()
    print(" raw keys:", list(raw_counts.keys())[:])

    # === 6) Extract & merge by token length (robust) ===
    filtered = {}
    flag_hist = {'0': 0, '1': 0}

    # --------------- extraction ----------------------------------
    filtered = {}  # pattern -> counts
    flag_hist = {'0': 0, '1': 0}

    # --------------- extraction & post‑selection ----------------------
    for key, cnt in raw_counts.items():
        toks = key.split()  # Qiskit inserts blanks between registers
        if len(toks) == 2:  # 'flag feat'   (two registers)
            flag_tok, feat_tok = (
                (toks[0], toks[1]) if len(toks[0]) == 1 else (toks[1], toks[0])
            )
        else:  # '0101010'     (concatenated)
            flag_tok, feat_tok = key[-1], key[:-1]

        flag_hist[flag_tok] += cnt
        if flag_tok != '0':  # keep only shots with c0 = 0
            continue

        row_bits = feat_tok  # already in MSB‑first order
        filtered[row_bits] = filtered.get(row_bits, 0) + cnt

    # --------------- output -------------------------------------------
    print("\nflag histogram:", flag_hist)
    print("feature pattern | counts")
    for pat in sorted(filtered):
        print(f"   {pat}   :  {filtered[pat]}")

    # === 7) Build labels & plot ===
    sorted_bits = sorted(filtered)
    counts = [filtered[b] for b in sorted_bits]
    total = sum(counts)
    user_bits = ''.join(map(str, user_feature))

    xticks = []
    for b in sorted_bits:
        hd = sum(c != u for c, u in zip(b, user_bits))
        xticks.append(f"{b} (HD={hd})")

    probs = [100 * v / total for v in counts]
    plt.figure(figsize=(8, 4))
    plt.bar(range(len(probs)), probs)
    plt.xticks(range(len(probs)), xticks, rotation=45, ha='right')
    plt.title(f"Grover ×{grover_iterations} | User Feature: {user_feature}")
    plt.ylabel("Probability %")
    plt.tight_layout()

    # Save the figure before showing it
    plt.savefig("images/plots/grover_plot.png", dpi=300)

    plt.show()


    return 0
    # Normal plotting


    # Plot HHL
    # hhl_circ = hhl.generate_example_hhl_QC()
    qc, _ = circuits.h_and_cx_circ()

    print(qc)
    qc.draw(output='mpl',
                       fold=30,
                       )
    plt.savefig(f"images/Circuits/h_cx_circ.png", dpi=300, bbox_inches="tight")
    plt.show()
    # | Layout Method    | Strategy                                               |
    # | ---------------- | ------------------------------------------------------ |
    # | `trivial`        | 1-to-1 mapping                                         |
    # | `dense`          | Densest‐subgraph heuristic                             |
    # | `noise_adaptive` | Minimize error rates (readout & 2-qubit)               |
    # | `sabre`          | Sabre seed + forwards/backwards swap refinement        |
    # | `default`        | VF2 perfect + Sabre fallback (or `trivial` at level 0) |
    # Routing: 'stochastic', #'sabre', #'lookahead', #'basic',


    print("Transpiling HHL circuit...")
    bw_pattern, col_map = brickwork_transpiler.transpile(qc, routing_method='stochastic', layout_method='trivial')
    bw_pattern.print_pattern(lim=200)

    encoded_pattern, log_alice = bfk_encoder(bw_pattern)
    encoded_pattern.print_pattern(lim=200)
    print(log_alice)

    encoded_pattern2, log_alice2 = bfk_encoder(bw_pattern, remove_dependencies=False)
    encoded_pattern2.print_pattern(lim=200)
    print(log_alice2)

    injector = DepolarisingInjector(single_prob=0.20, two_prob=0.10)

    noisy_pattern = injector.inject(bw_pattern)

    # simulate using ideal backend after injection
    # out = noisy_pattern.simulate_pattern(backend="statevector")

    # Suppose 'pattern' is your Graphix Pattern object
    noisy_pattern = injector.inject(bw_pattern)


    print("Plotting brickwork graph...")
    visualiser.plot_brickwork_graph_from_pattern(bw_pattern,
                                                 show_angles=True,
                                                 node_colours=col_map,
                                                 use_node_colours=True,
                                                 title="Brickwork graph: H + CX")

    print("Plotting brickwork graph...")
    visualiser.plot_brickwork_graph_from_pattern(encoded_pattern,
                                                 show_angles=True,
                                                 node_colours=col_map,
                                                 use_node_colours=True,
                                                 title="Brickwork graph: H + CX encoded")

    visualiser.plot_graphix_noise_graph(encoded_pattern, save=True)

    visualiser.plot_brickwork_graph_locked(encoded_pattern, use_locks=False, title="H + CX encoded")


    # return 0

    from graphix.channels import depolarising_channel
    # depolarizing_channel(probability)
    channel = depolarising_channel(0.05)  # 5% depolarizing noise

    from graphix.channels import depolarising_channel, two_qubit_depolarising_channel
    from graphix.noise_models.noise_model import NoiseModel

    class MyDepolNoise(NoiseModel):
        def __init__(self, p1, p2):
            super().__init__()
            self.p1 = p1  # single-qubit error prob
            self.p2 = p2  # two-qubit error prob

        def prepare_qubit(self):
            return depolarising_channel(self.p1)

        def measure(self):
            return depolarising_channel(self.p1)

        def byproduct_x(self):
            return depolarising_channel(self.p1)

        def byproduct_z(self):
            return depolarising_channel(self.p1)

        def clifford(self):
            return depolarising_channel(self.p1)

        def entangle(self):
            # CZ or other two-qubit gate
            return two_qubit_depolarising_channel(self.p2)

        def tick_clock(self):
            # Optional: apply idle decoherence every timestep
            return depolarising_channel(self.p1)

        def confuse_result(self, cmd):
            return cmd


    outstate = bw_pattern.simulate_pattern(backend='densitymatrix', noise_model = MyDepolNoise(p1=0.05, p2=0.02))

    print(f"Output of noisy simulation: {outstate}")



    print("Plotting brickwork graph...")
    visualiser.plot_brickwork_graph_from_pattern(bw_pattern,
                                                 show_angles=True,
                                                 node_colours=col_map,
                                                 use_node_colours=True,
                                                 title="Brickwork graph: DAG example")

    print("Plotting locked graph")
    visualiser.plot_brickwork_graph_locked(bw_pattern, use_locks=False)

    # visualiser.plot_brickwork_graph_from_pattern_old_style(bw_pattern,
    #                                              node_colours=col_map,
    #                                              use_node_colours=True,
    #                                              title="Brickwork graph: shift")


    return 0
    # Test the QRS


    #Experimental data
    # bw_in_depths = [973, 1374 , 1980, 4082, 8857, 18758]
    # user_counts = [4, 8, 16, 32, 64, 128]
    # bw_aligned_depths = [1236, 1746 , 2559, 5315, 11613, 26009]
    # feature_length = 6
    # feature_widths = [4, 6, 8, 10, 12]
    #
    # visualiser.plot_qrs_bw_scaling(user_counts, bw_in_depths, bw_aligned_depths, feature_length)
    #
    # visualiser.plot_time_complexity_3d(user_counts, feature_widths)
    #
    # visualiser.plot_time_complexity_with_bw_lines(user_counts, feature_widths,
    #                                               bw_in_depths, bw_aligned_depths,
    #                                               feature_length, azim=245)
    #
    # return 0

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
        [1, 0, 0, 1, 1, 1],  # Tzula-C
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

    feature_mats = [feature_mat3] #, feature_mat2, feature_mat3, feature_mat4, feature_mat5]
    names = [names3] #, names2, names3, names4, names5]

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
            plot=True,
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
        print("Simulating...")
        sim = AerSimulator()
        qc_transpiled = qiskit.compiler.transpiler.transpile(qc_meas, sim, optimization_level=3)


        shots=1024
        result = sim.run(qc_transpiled, shots=shots).result()
        raw_counts = result.get_counts()

        # 1) Keep only c0=0 shots
        filtered_counts = {}
        for full_bitstr, cnt in raw_counts.items():
            if full_bitstr[0] == '0':  # only c0=0
                filtered_counts[full_bitstr] = filtered_counts.get(full_bitstr, 0) + cnt

        # 2) Build sorted lists for plotting
        sorted_keys = sorted(filtered_counts.keys())
        sorted_vals = [filtered_counts[k] for k in sorted_keys]

        xtick_labels = []
        for full_bitstr in sorted_keys:
            # leading_bit is always '0' here
            leading_bit = full_bitstr[0]

            # raw_suffix = e.g. "11000" which is [c_feat4,c_feat3,c_feat2,c_feat1,c_feat0]
            raw_suffix = full_bitstr[-feature_length:]

            # reverse it so that index 0→qubit2, …, index4→qubit6
            true_bits = raw_suffix[::-1]

            # Hamming distance between true_bits and user_feature
            hd = sum(b1 != b2 for b1, b2 in zip(true_bits, user_feature))

            if true_bits in bitstring_to_name:
                person = bitstring_to_name[true_bits]
                label = f"{person} ({true_bits} {hd})"
            else:
                label = f"No_name ({true_bits} {hd})"

            xtick_labels.append(label)

        # 1) Compute total count
        total = sum(sorted_vals)

        # 2) Convert each count into a percentage
        sorted_vals_pct = [v / total * 100 for v in sorted_vals]

        # 3) Plot using those percentages
        plt.figure(figsize=(8, 4))
        plt.bar(sorted_keys, sorted_vals_pct)
        plt.title(
            f"Recommendation for user vector: {user_feature}  –  "
            f"{grover_iterations} Grover iterations (post‐selected on c0=0)"
        )
        plt.xlabel("Measured bitstrings")
        plt.ylabel("Recommendation prob. (%)")
        plt.xticks(ticks=sorted_keys, labels=xtick_labels, rotation=45)
        plt.tight_layout()

        plt.savefig(f"images/qrs/recommendation_plot_grover_{grover_iterations}.png", dpi=300, bbox_inches="tight")
        plt.show()

        print(f"size_check = {len(feature_mat)} x {len(feature_mat[0])}")


        # Decompose to CX, rzrxrz, id   -   Need opt = 3 for SU(2) rotation merging
        # decomposed_qc = decomposer.decompose_qc_to_bricks_qiskit(qrs_circuit, opt=3,
        #                                                          routing_method='sabre',
        #                                                          layout_method='default')
        #
        # # Optiise instruction matrix with dependency graph
        # qc_mat, cx_mat = decomposer.instructions_to_matrix_dag(decomposed_qc)
        # qc_mat_aligned = decomposer.align_bricks(cx_mat, qc_mat)
        #
        # bw_depths_aligned.append(len(qc_mat_aligned[0]))
        # bw_depths_input.append(len(qc_mat[0]))
        # print(f"feature mat: {id_fm}, aligned depth: {len(qc_mat_aligned[0])}, input depth: {len(qc_mat[0])}")

    # visualiser.plot_qrs_bw_scaling(bw_depths_input, bw_depths_aligned)

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
