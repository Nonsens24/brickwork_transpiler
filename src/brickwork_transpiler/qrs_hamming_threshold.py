from matplotlib import pyplot as plt
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import MCXGate

# Updated implementation to explicitly calculate numeric Hamming distances
# and to integrate coherently with qknn

def qrs_dist(n_items, feature_mat, user_vector, plot=False, grover_iterations=1):

    num_id_qubits = int(np.log2(n_items))
    num_db_feature_qubits = len(feature_mat[0])
    num_user_qubits = len(user_vector)
    num_distance_qubits = int(np.ceil(np.log2(num_db_feature_qubits + 1)))

    # --- basic checks ---
    if n_items & (n_items - 1) != 0:
        raise ValueError("n_items must be a power of two")
    if any(len(row) != num_db_feature_qubits for row in feature_mat):
        raise ValueError("All rows of feature_mat must have length l")

    total_qubits = num_id_qubits + num_db_feature_qubits + num_user_qubits + num_distance_qubits

    # define registers
    id_qubits = list(range(num_id_qubits))
    feature_qubits = list(range(num_id_qubits, num_id_qubits + num_db_feature_qubits))
    user_qubits = list(range(num_id_qubits + num_db_feature_qubits,
                             num_id_qubits + num_db_feature_qubits + num_user_qubits))
    distance_qubits = list(range(num_id_qubits + num_db_feature_qubits + num_user_qubits,
                                 total_qubits))

    qc = QuantumCircuit(total_qubits)

    print("Start database initialisation...")
    qc = initialise_database(qc, id_qubits, feature_qubits, user_qubits, user_vector, feature_mat)

    print("Start numeric KNN...")
    qc = knn(qc, feature_qubits, user_qubits, distance_qubits)

    # add Grover ancillary qubit
    qA = QuantumRegister(1, "qA")
    qc.add_register(qA)
    qc.x(qA)
    qc.h(qA)

    # Perform Grover amplification based on distances (not shown fully, depends on custom oracle)

    if plot:
        qc.draw(output='mpl', fold=40)
        plt.savefig(f"images/qrs/recommendation_circ{grover_iterations}.png", dpi=300, bbox_inches="tight")
        plt.show()

    return qc


def initialise_database(qc: QuantumCircuit,
                        id_qubits: list[int],
                        feature_qubits: list[int],
                        user_qubits: list[int],
                        user_vector: list[int],
                        feature_mat: list[list[int]],) -> QuantumCircuit:
    """
    Loads a database of size n_items=2^q with l-bit features,
    appends the user_vector. Uses multi-controlled Xs (MCX).
    """

    # 1) uniform superposition on index register
    for qb in id_qubits:
        qc.h(qb)

    # 2) database load via MCX (full lookup)
    for i, bits in enumerate(feature_mat):
        pattern = format(i, f"0{len(id_qubits)}b")[::-1]

        for k, b in enumerate(pattern):
            if b == '0':
                qc.x(id_qubits[k])

        for j, b in enumerate(bits):
            if b == 1:
                qc.append(MCXGate(len(id_qubits)), id_qubits + [feature_qubits[j]])

        for k, b in enumerate(pattern):
            if b == '0':
                qc.x(id_qubits[k])

    # 3) Encode user vector
    for idx, bit in enumerate(user_vector):
        if bit == 1:
            qc.x(user_qubits[idx])

    return qc



from qiskit import QuantumCircuit, QuantumRegister
import numpy as np

def knn(
    qc: QuantumCircuit,
    feature_qubits: list[int],
    user_qubits: list[int],
    distance_qubits: list[int]
) -> QuantumCircuit:
    """
    In-place QkNN Hamming-distance as binary integer:
    - feature_qubits[i]: i-th database feature bit
    - user_qubits[i]: i-th user-vector bit
    - distance_qubits: qubits that store binary Hamming distance

    distance_qubits must have ceil(log2(len(feature_qubits))) bits.
    """
    l = len(feature_qubits)
    num_distance_bits = len(distance_qubits)

    if l != len(user_qubits):
        raise ValueError("feature_qubits and user_qubits must have the same length")

    if num_distance_bits < int(np.ceil(np.log2(l + 1))):
        raise ValueError("distance_qubits must be sufficient to represent the max Hamming distance")

    # Temporary difference qubits (l bits)
    diff = QuantumRegister(l, name="diff")
    qc.add_register(diff)

    # 1. XOR user and feature qubits to diff qubits
    for i in range(l):
        qc.cx(user_qubits[i], diff[i])
        qc.cx(feature_qubits[i], diff[i])

    # 2. Compute Hamming weight (distance) from diff into distance_qubits using ripple-carry adder logic
    # Initialize distance_qubits to 0 explicitly if necessary

    # Simple ripple-carry addition:
    for i in range(l):
        qc.cx(diff[i], distance_qubits[0])
        carry = QuantumRegister(1, name=f"carry_{i}")
        qc.add_register(carry)
        qc.ccx(diff[i], distance_qubits[0], carry)

        for j in range(1, num_distance_bits):
            qc.cx(carry, distance_qubits[j])
            if j < num_distance_bits - 1:
                next_carry = QuantumRegister(1, name=f"carry_{i}_{j}")
                qc.add_register(next_carry)
                qc.ccx(carry, distance_qubits[j], next_carry)
                carry = next_carry

    # Optional: Uncompute diff qubits if you don't need them
    for i in range(l):
        qc.cx(feature_qubits[i], diff[i])
        qc.cx(user_qubits[i], diff[i])

    return qc



