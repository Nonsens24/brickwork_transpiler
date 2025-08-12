from matplotlib import pyplot as plt

from src.brickwork_transpiler import circuits, brickwork_transpiler, visualiser, utils
from src.brickwork_transpiler.bfk_encoder import encode_pattern


def build_graph():

    experiment, _ = circuits.minimal_qrs()

    experiment.draw(output='mpl',
                    fold=30,
                    style="iqp"
                    )
    plt.savefig(f"images/Circuits/minimal_recommendation_circuit.png", dpi=300, bbox_inches="tight")
    plt.show()

    bw_pattern, col_map = brickwork_transpiler.transpile(experiment, routing_method='sabre', layout_method='trivial',
                                                         with_ancillas=False, plot_decomposed=True)

    visualiser.plot_brickwork_graph_from_pattern(bw_pattern,
                                                 show_angles=True,
                                                 node_colours=col_map,
                                                 use_node_colours=True,
                                                 title="Brickwork graph: Minimal QRS")

    file_path = "src/brickwork_transpiler/experiments/data/output_data/"

    pattern_writer = utils.BufferedCSVWriter(file_path + "minimal_qrs_experiment_pattern.txt", ["pattern"])
    log_writer = utils.BufferedCSVWriter(file_path + "minimal_qrs_experiment_log.txt", ["log"])

    encoded_pattern, log_alice = encode_pattern(bw_pattern)
    pattern_writer.set("pattern", encoded_pattern.print_pattern(lim=2 ** 32))
    log_writer.set("log", log_alice)

    pattern_writer.flush()
    log_writer.flush()

    visualiser.plot_brickwork_graph_locked(encoded_pattern, use_locks=False,
                                           title="Brickwork Graph: Minimal QRS Encoded",
                                           show_angles=True)

    # Minimal runner: execute from given input statevector and plot filtered histogram

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit.library import Initialize

# Prefer AerSimulator (qiskit-aer); fall back to Aer
try:
    from qiskit_aer import AerSimulator
    _BACKEND = AerSimulator()
except Exception:
    from qiskit import Aer
    _BACKEND = Aer.get_backend("aer_simulator")

from qiskit import ClassicalRegister, transpile
from qiskit.circuit.library import Initialize
from src.brickwork_transpiler import circuits

# If you don't already have a backend, define one like this:
try:
    from qiskit_aer import AerSimulator
    _BACKEND = AerSimulator()
except Exception:
    from qiskit import Aer
    _BACKEND = Aer.get_backend("aer_simulator")


def run_and_plot_minimal_qrs_only_db(*,
                                     shots: int = 4096,
                                     db_qubits: list[int] | None = None,
                                     plot: bool = True,
                                     flip_plot_labels: bool = False,
                                     ) -> dict[str, int]:
    """
    Run minimal_qrs from its provided input statevector, measure ONLY c0 and db_qubits,
    post-select on c0 == '0', and print/plot the histogram over db_qubits only.

    Returns
    -------
    filtered : dict[str,int]
        Histogram over DB bits (as a bitstring) after post-selection on c0=0.
        (Rightmost printed bit corresponds to the *lowest* db_qubit index mapped to c[1].)
    """
    user_feature = [0, 0]
    # Build circuit + input state
    qc, input_state = circuits.minimal_qrs(user_feature)

    # Default DB subset if not provided (matches minimal_qrs definition)
    if db_qubits is None:
        db_qubits = [1, 2]

    # Prepare from the given Statevector
    init = Initialize(input_state.data)
    qc = qc.copy()
    qc.compose(init, qubits=qc.qubits, front=True, inplace=True)

    # Classical register: c[0] holds c0 (for post-selection), c[1:] hold DB bits
    m = 1 + len(db_qubits)
    cr = ClassicalRegister(m, "c")
    qc.add_register(cr)

    # Map qubits to classical bits:
    # - c0 (last quantum wire) -> c[0]  (rightmost)
    c0_idx = qc.num_qubits - 1
    qc.measure(c0_idx, cr[0])

    # - db_qubits -> c[1], c[2], ...
    #   (We map in ascending physical index for determinism.)
    for j, q in enumerate(sorted(db_qubits)):
        qc.measure(q, cr[1 + j])

    # Execute
    compiled = transpile(qc, _BACKEND, optimization_level=1)
    job = _BACKEND.run(compiled, shots=shots)
    result = job.result()
    counts = result.get_counts(compiled)  # dict: bitstring -> shots

    # --- Post-select on c0 == '0' and keep ONLY DB bits ---
    # Bitstrings are MSB..LSB; rightmost char corresponds to c[0] (our c0)
    def _sanitize(s: str) -> str:
        return s.replace(" ", "")

    counts = {_sanitize(k): int(v) for k, v in counts.items()}

    filtered: dict[str, int] = {}
    for bitstring, shots_ in counts.items():
        # c0 at rightmost position (c[0])
        if bitstring[-1] != '0':
            continue

        # DB bits occupy c[1..m-1]; take them in the same order we measured:
        # c[1] is next-to-rightmost, c[2] after that, etc.
        # Build a compact DB string (rightmost char = c[1]).
        db_bits = ''.join(bitstring[-1 - k] for k in range(1, m))
        filtered[db_bits] = filtered.get(db_bits, 0) + shots_

    # Print a small summary
    total_good = sum(filtered.values())
    print(f"Post-selected on c0=0 | good shots = {total_good}/{sum(counts.values())}")
    for b in sorted(filtered):
        print(f"{b}: {filtered[b]}")

    # --- Plot (DB-only) ---
    if plot:
        if filtered:
            xs = sorted(filtered)                     # canonical keys (whatever order you computed)
            ys = [filtered[b] for b in xs]
            tot = sum(ys)
            probs = [100.0 * y / tot for y in ys]

            # DISPLAY-ONLY: flip label bit order if you want MSB↔LSB swapped
            labels = [b[::-1] for b in xs] if flip_plot_labels else xs

            plt.figure(figsize=(7, 3.6))
            plt.bar(range(len(xs)), probs)
            plt.xticks(range(len(xs)), labels)        # use display labels, not xs
            plt.ylabel("Probability (%)")
            plt.title(f"Minimal QRS Results (post-selected c0=0) | feature: {user_feature}")
            plt.tight_layout()
            plt.savefig(f"images/plots/minimal_qrs_plot_{user_feature}.png", dpi=300,
                        bbox_inches="tight")
            plt.show()
        else:
            print("No shots survived post-selection (c0=0).")

    return filtered
