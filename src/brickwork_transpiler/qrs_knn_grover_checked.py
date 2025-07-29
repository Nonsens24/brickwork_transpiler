from matplotlib import pyplot as plt
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import MCXGate

# This implementation follows the quantum recommendation system as presented by Sawerwain and Wroblewski (2019)
# Link: https://sciendo.com/article/10.2478/amcs-2019-0011


def qrs(n_items, feature_mat, user_vector, plot=False, grover_iterations=1):

    num_id_qubits = int(np.log2(n_items))
    num_db_feature_qubits = len(feature_mat[0])
    num_user_qubits = len(user_vector)

    # --- 0) basic checks ---
    if n_items & (n_items - 1) != 0:
        raise ValueError("n_items must be a power of two")
    if any(len(row) != num_db_feature_qubits for row in feature_mat):
        raise ValueError("All rows of feature_mat must have length l")

    total_qubits = num_id_qubits + num_db_feature_qubits + num_user_qubits

    # define registers
    id_qubits         = list(range(num_id_qubits))               # index qubits (0..q-1)
    feature_qubits = list(range(num_id_qubits, num_id_qubits + num_db_feature_qubits))        # feature qubits (q..q+l-1)
    user_qubits       = list(range(num_id_qubits + num_db_feature_qubits, total_qubits))  # user bits (q+l..end)

    # total_qubits = q index + l database features + len(user_vector)
    qc = QuantumCircuit(total_qubits)
    # qc = QuantumCircuit(total_qubits, num_db_feature_qubits)

    print("Start database initialisation...")
    qrs_init = initialise_database(qc, id_qubits, feature_qubits, user_qubits, user_vector, feature_mat)

    print("Start KNN...")
    qrs_knn = knn(qrs_init, feature_qubits, user_qubits)

    qA = qc.qregs[-1][0] if qc.qregs and qc.qregs[-1].name == 'qA' else None
    if qA is None:
        grov = QuantumRegister(1, "qA")
        qc.add_register(grov)
        qA = grov[0]
    qc.x(qA); qc.h(qA)

    print("Start Grover...")
    # qrs_amplified = grover(qrs_knn, feature_qubits, user_vector, iterations=grover_iterations)
    # grover_amplify_feature(qrs_knn, feature_qubits, user_vector)

    # qc.h(qA)
    # qc.x(qA)  # return ancilla to |0⟩

    # Build and draw the circuit
    if plot:
        qrs_knn.draw(output='mpl',
                           fold=40,
                           )
        plt.savefig(f"images/qrs/recommendation_circ{grover_iterations}.png", dpi=300, bbox_inches="tight")
        plt.show()

    return qrs_knn


def initialise_database(qc: QuantumCircuit,
                        id_qubits: [],
                        feature_qubits: [],
                        user_qubits: [],
                        user_vector: [],
                        feature_mat: list[list[int]],) -> QuantumCircuit:
    """
    Creates a circuit that loads a database of size n_items=2^q with l-bit feature_vecs,
    plus appends a user_vector. Uses a full lookup via multi-controlled Xs (MCX).
    """


    # --- 1) uniform superposition on index register ---
    for qb in id_qubits:
        qc.h(qb)

    # --- 2) database load via MCX (full lookup) ---
    for i, bits in enumerate(feature_mat):
        # Build the q-bit binary string for i, then reverse it so that
        # pattern[0] is the LSB, pattern[q-1] is the MSB.
        pattern = format(i, f"0{len(id_qubits)}b")   # MSB..LSB
        pattern = pattern[::-1]         # LSB..MSB

        # Invert those id_qubits[k] where the k-th bit is '0',
        # so that after inversion, all index qubits are |1> exactly when the register is |i>
        for k, b in enumerate(pattern):
            if b == '0':
                qc.x(id_qubits[k])

        # For each feature bit = 1 in row i, flip that feature qubit
        # controlled on all q index qubits being |1> (i.e. MCX with q controls)
        for j, b in enumerate(bits):
            if b == 1:
                qc.append(MCXGate(len(id_qubits)), id_qubits + [feature_qubits[j]])

        for k, b in enumerate(pattern):
            if b == '0':
                qc.x(id_qubits[k])

    # -- 3) Encode user vector
    for idx, bit in enumerate(user_vector):
        if bit == 1:
            qc.x(user_qubits[idx])

    return qc


def knn(
    qc: QuantumCircuit,
    feature_qubits: list[int],
    user_qubits:    list[int]
) -> QuantumCircuit:
    """
    In-place QkNN Hamming-distance & phase-sum:
      - feature_qubits[i] holds the i-th database feature bit
      - user_qubits[i]    holds the i-th user-vector bit
    After this, the ancilla 'c0' carries amplitudes ∝ cos(π/(2l)·d).
    """
    l = len(feature_qubits)
    if l != len(user_qubits):
        raise ValueError("feature_qubits and user_qubits must have the same length")

    # 1) Add single ancilla c0 for the phase-sum
    c0 = QuantumRegister(1, name="c0")
    qc.add_register(c0)

    qc.h(c0)  # put c0 into |+> = (|0>+|1>)/√2

    for i in range(l):
        qc.cx(user_qubits[(l-1) - i], feature_qubits[i])

    # for u_q, f_q in zip(user_qubits, feature_qubits):
    #     qc.cx(u_q, f_q)

    # 3) Phase-sum those l distance bits onto c0

    for f in feature_qubits:
        # (a) P2: deposit a phase e^{+i π/l} on (f=1, c0=0)
        qc.cp(+np.pi / l, f, c0)

        # (b) P1: single-qubit phase e^{-i π/(2l)} on (f=1)
        qc.p(-np.pi / (2 * l), f)

    qc.h(c0)

    #  Uncompute QKNN -- for single pattern amp
    for i in range(l):
        qc.cx(user_qubits[(l - 1) - i], feature_qubits[i])

    return qc


# Amplifies amplitudes of recommendations of interest using grover's algorithm
# See Grover (1996) link: https://dl.acm.org/doi/10.1145/237814.237866
def grover(qc, feature_qubits, user_vector, iterations=2):
    """Amplify the exact‑match row."""



    target_bits = user_vector

    for _ in range(iterations):
        # ---------- oracle: flip phase if feature == user_vector ----
        for i, bit in enumerate(target_bits):
            if bit == 0:  # flip the 0‑bits
                qc.x(feature_qubits[i])

        qc.mcx(feature_qubits, qA)

        for i, bit in enumerate(target_bits):
            if bit == 0:
                qc.x(feature_qubits[i])

        # ---------- diffusion on the same register -----------------
        qc.h(feature_qubits)
        qc.x(feature_qubits)

        qc.h(feature_qubits[-1])
        qc.mcx(feature_qubits[:-1], feature_qubits[-1])
        qc.h(feature_qubits[-1])

        qc.x(feature_qubits)
        qc.h(feature_qubits)


    return qc

from qiskit import QuantumCircuit

def grover_amplify_feature(qc, feature_qubits, user_feature):
    """
    Appends a Grover oracle and diffuser to qc, amplifying |user_feature> on feature_qubits.
    Modifies qc in-place.
    """
    n = len(feature_qubits)
    # --- ORACLE: Flip phase of target feature state
    # Map |user_feature> -> |1..1>
    for i, bit in enumerate(user_feature):
        if bit == 0:
            qc.x(feature_qubits[i])
    # Multi-controlled Z (use MCX with ancilla method: flip last qubit's phase)
    qc.h(feature_qubits[-1])
    if n == 1:
        qc.z(feature_qubits[0])
    else:
        qc.mcx(feature_qubits[:-1], feature_qubits[-1])
    qc.h(feature_qubits[-1])
    # Uncompute mapping
    for i, bit in enumerate(user_feature):
        if bit == 0:
            qc.x(feature_qubits[i])

    # --- DIFFUSER: Inversion about mean on feature_qubits
    for qb in feature_qubits:
        qc.h(qb)
        qc.x(qb)
    qc.h(feature_qubits[-1])
    if n == 1:
        qc.z(feature_qubits[0])
    else:
        qc.mcx(feature_qubits[:-1], feature_qubits[-1])
    qc.h(feature_qubits[-1])
    for qb in feature_qubits:
        qc.x(qb)
        qc.h(qb)

