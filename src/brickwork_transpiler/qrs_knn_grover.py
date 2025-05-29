from matplotlib import pyplot as plt
from qiskit.circuit.library import XGate
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.quantum_info import Statevector

from src.brickwork_transpiler import utils
from src.brickwork_transpiler.utils import feature_to_affine_generator, feature_to_generator


# This follows the quantum recommendation system as presented by Sawerwain and Wroblewski (2019)
# Link: https://sciendo.com/article/10.2478/amcs-2019-0011


# | User/IceCream | Chocolate | Vanilla | Strawberry | Nuts | Vegan |
# | ------------- | --------- | ------- | ---------- | ---- | ----- |
# | Sebastian-I   | 1         | 0       | 1          | 0    | 1     |
# | Tzula-C       | 0         | 1       | 0          | 1    | 0     |
# | Rex-E         | 1         | 1       | 1          | 0    | 1     |
# | Scot-T        | 0         | 1       | 1          | 1    | 0     |

def qrs(n_items, feature_mat, user_vector, plot=False, grover_iterations=1):

    item_qubits = int(np.log2(n_items))

    qrs_init = initialise_database(n_items, feature_mat, user_vector)
    qrs_knn = knn(qrs_init, len(feature_mat[0]), item_qubits, user_vector)
    qrs_amplified = grover(qrs_knn, item_qubits, len(feature_mat[0]), user_vector, iterations=grover_iterations)

    # Build and draw the circuit
    if plot:
        qrs_amplified.draw(output='mpl',
                           fold=30,
                           )
        plt.show()

    return qrs


# Creates the user ID, Encodes the features, and creates a new user vector
# Upper bound: O(2l + N+(N-1)/2, from Li et al. (2013), link: https://arxiv.org/abs/1210.7366
def initialise_database(n_items: int,
                        feature_mat: list[list[int]],
                        user_vector: str) -> QuantumCircuit:
    """
    Creates a circuit that loads a database of size n_items=2^q with l-bit feature_vecs,
    plus appends a user_vector.  If feature_mat is a linear code, uses only CNOTs;
    otherwise does a full lookup via multi-controlled Xs.
    """
    # --- 0) basic checks & sizes ---
    if n_items & (n_items - 1) != 0:
        raise ValueError("n_items must be a power of two")
    q = int(np.log2(n_items))
    l = len(feature_mat[0])
    if any(len(row) != l for row in feature_mat):
        raise ValueError("All rows of feature_mat must have length l")

    # total qubits = q index + l feature + len(user_vector)
    qc = QuantumCircuit(q + l + len(user_vector))

    # --- 1) uniform superposition on index register ---
    qc.h(range(q))

    # --- 2) database load ---
    try:
        # try the fast linear/CNOT-only path
        G = feature_to_generator(feature_mat)
        for j in range(l):
            for k in range(q):
                if G[j][k]:
                    qc.cx(k, q + j)

    except ValueError:
        # not linear → fallback to general reversible lookup
        print("Not linear")
        for i, bits in enumerate(feature_mat):
            bitstr = format(i, f'0{q}b')
            # prepare controls to be |i⟩
            for k, b in enumerate(bitstr):
                if b == '0':
                    qc.x(k)
            # for each '1' in the feature row, MCX into feature qubit
            for j, b in enumerate(bits):
                if b:
                    qc.mcx(list(range(q)), q + j, None, mode='noancilla')
            # uncompute the control flips
            for k, b in enumerate(bitstr):
                if b == '0':
                    qc.x(k)

    # --- 3) append the user vector as X-flips ---
    for idx, bit in enumerate(user_vector):
        if bit == '1':
            qc.x(q + l + idx)

    # print(qc.draw())

    return qc



# Performs the QKNN algorithm by first calculating the hamming distance and then summing the distances
# See Trugenberger (2002) link: https://link.springer.com/article/10.1023/A:1024022632303
def knn(qrs: QuantumCircuit, feature_size: int, item_qubits: int, user_vector: str) -> QuantumCircuit:
    # Add ancilla qubit for KNN
    c0 = QuantumRegister(1, name='c0')
    qrs.add_register(c0)            # TODO: Move this to the top of the circuit!

    # Hamming distance calculations:
    for q in range(feature_size + item_qubits + 1, qrs.num_qubits - 1):
        qrs.cx(qrs.qubits[q], qrs.qubits[qrs.num_qubits - q])
        print(f"feature_size = {feature_size}, q = {q}, qrs.num - q = {qrs.num_qubits - q}")

    # print(qrs.draw(output='text'))

    # Quantum summing of Hamming distances:
    qrs.h(c0)

    for q in range(feature_size + item_qubits, 1, -1):
        qrs.cp(-np.pi/feature_size, qrs.qubits[q], c0) # feature size = l from the paper P2
        qrs.rz(np.pi/2*feature_size, qrs.qubits[q])     # P1

    qrs.h(c0)

    print(qrs.draw(output='text'))



    return qrs

# Amplifies amplitudes of recommendations of interest using grover's algorithm
# See Grover (1996) link: https://dl.acm.org/doi/10.1145/237814.237866
from qiskit import QuantumCircuit, QuantumRegister

def grover(
    qc: QuantumCircuit,
    q: int,
    l: int,
    user_vector: str,
    iterations: int = 1
) -> QuantumCircuit:
    """
    Append `iterations` Grover rounds that mark the basis state == user_vector
    on the l-qubit feature register [q, …, q+l-1], using one aux qubit |qA>.
    """
    # --- 1) allocate the Grover ancilla ---
    grov = QuantumRegister(1, name='qA')
    qc.add_register(grov)
    qA = grov[0]

    feature_qubits = list(range(q, q + l))

    for _ in range(iterations):
        # ───────────── Oracle ─────────────
        # a) Flip those feature qubits where user_vector bit == '0'
        for i, bit in enumerate(user_vector):
            if bit == '0':
                qc.x(feature_qubits[i])

        # b) prepare ancilla in |–> for phase kickback
        qc.x(qA)
        qc.h(qA)

        # c) multi-controlled X from all feature_qubits → qA
        qc.mcx(feature_qubits, qA)

        # d) undo ancilla prep
        qc.h(qA)
        qc.x(qA)

        # e) uncompute the feature-qubit bit-flips
        for i, bit in enumerate(user_vector):
            if bit == '0':
                qc.x(feature_qubits[i])

        # ────────── Diffusion (inversion-about-mean) ──────────
        qc.h(feature_qubits)
        qc.x(feature_qubits)

        # multi-controlled Z on feature register, via H–MCX–H on last feature qubit
        qc.h(feature_qubits[-1])
        qc.mcx(feature_qubits[:-1], feature_qubits[-1])
        qc.h(feature_qubits[-1])

        qc.x(feature_qubits)
        qc.h(feature_qubits)

    return qc
