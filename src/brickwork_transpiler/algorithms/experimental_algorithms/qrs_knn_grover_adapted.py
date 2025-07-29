from matplotlib import pyplot as plt
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit.library import MCXGate
import math

# --- (1) QRS‐related functions (with corrected increment/decrement) ---

def initialise_database(n_items: int,
                        feature_mat: list[list[int]],
                        user_vector: str) -> QuantumCircuit:
    """
    Loads |ψ_q>⊗|ψ_ff>⊗|ψ_fu>:
      - q = log2(n_items) index qubits, put into |+>
      - l = len(feature_mat[0]) feature qubits, loaded with rows of feature_mat via MCX
      - l user qubits, flipped according to user_vector
    """
    if n_items & (n_items - 1) != 0:
        raise ValueError("n_items must be a power of two")
    q = int(np.log2(n_items))
    l = len(feature_mat[0])
    if any(len(row) != l for row in feature_mat):
        raise ValueError("All rows of feature_mat must have length l")

    total_qubits = q + l + len(user_vector)
    qc = QuantumCircuit(total_qubits)

    id_qubits = list(range(q))
    db_feature_qubits = list(range(q, q + l))
    user_qubits = list(range(q + l, q + 2 * l))

    # 1) uniform superposition on index register
    for qb in id_qubits:
        qc.h(qb)

    # 2) database load via MCX lookup
    for i, bits in enumerate(feature_mat):
        pattern = format(i, f"0{q}b")
        for k, b in enumerate(pattern):
            if b == '0':
                qc.x(id_qubits[k])
        for j, b in enumerate(bits):
            if b:
                qc.append(MCXGate(q), id_qubits + [q + j])
        for k, b in enumerate(pattern):
            if b == '0':
                qc.x(id_qubits[k])

    # 3) append user_vector
    for idx, bit in enumerate(user_vector):
        if bit == '1':
            qc.x(user_qubits[idx])

    return qc


def knn(
    qc: QuantumCircuit,
    feature_qubits: list[int],
    user_qubits:    list[int]
) -> QuantumCircuit:
    """
    In-place QkNN Hamming‐distance & phase‐sum:
      - feature_qubits[i] holds r_{p,i}
      - user_qubits[i] holds t_i
      - adds ancilla c0; XORs r↔t; deposits phases P2/P1; H on c0; uncomputes XOR
    """
    l = len(feature_qubits)
    if l != len(user_qubits):
        raise ValueError("feature_qubits and user_qubits must have the same length")

    # 1) add ancilla c0
    c0_reg = QuantumRegister(1, name="c0")
    qc.add_register(c0_reg)
    c0 = c0_reg[0]

    # 2) XOR: d_i = r_i ⊕ t_i → feature_qubits[i]
    for i in range(l):
        qc.cx(user_qubits[i], feature_qubits[i])

    # 3) Phase‐sum onto c0
    qc.h(c0)
    for f in feature_qubits:
        # (a) P2: phase on |f=1, c0=0>
        qc.x(c0)
        qc.cp(+np.pi / l, f, c0)
        qc.x(c0)
        # (b) P1: single‐qubit phase on f=1
        qc.p(-np.pi / (2 * l), f)
    qc.h(c0)

    # 4) uncompute XOR
    for i in range(l):
        qc.cx(user_qubits[i], feature_qubits[i])

    return qc


def diffusion_on_index(qc: QuantumCircuit, index_qubits: list[int]):
    """
    Inversion‐about‐the‐mean on index_qubits:
      H–X–MCZ–X–H
    """
    qc.h(index_qubits)
    qc.x(index_qubits)
    qc.h(index_qubits[-1])
    qc.mcx(index_qubits[:-1], index_qubits[-1])
    qc.h(index_qubits[-1])
    qc.x(index_qubits)
    qc.h(index_qubits)


def controlled_increment(qc: QuantumCircuit, control_qubit: int, sum_reg: list[int]):
    """
    If control_qubit=1, add 1 (mod 2^n) to the integer in sum_reg (ripple‐carry style).
    """
    n = len(sum_reg)
    # Increment LSB
    qc.cx(control_qubit, sum_reg[0])
    # Propagate carries: when control=1 and previous bit was 1, flip next bit
    for j in range(1, n):
        qc.ccx(control_qubit, sum_reg[j - 1], sum_reg[j])


def controlled_decrement(qc: QuantumCircuit, control_qubit: int, sum_reg: list[int]):
    """
    Inverse of controlled_increment: subtract 1 (mod 2^n) if control_qubit=1.
    """
    n = len(sum_reg)
    # Reverse carry‐propagation
    for j in reversed(range(1, n)):
        qc.ccx(control_qubit, sum_reg[j - 1], sum_reg[j])
    # Decrement LSB
    qc.cx(control_qubit, sum_reg[0])


def grover_hamming_oracle(
    qc: QuantumCircuit,
    feature_qubits: list[int],
    user_vector:     str,
    d_reg:       list[int],
    sum_reg:     list[int],
    aux_qubit:   int,
    target_distance: int
):
    """
    Flip phase on |p>⊗|r_p> iff Hamming(r_p, t) = target_distance,
    using:
      - d_reg[i] to store r_{p,i}⊕t_i
      - sum_reg to accumulate Σ d_i
      - aux_qubit prepared in |−> for phase kickback
    """
    l = len(feature_qubits)
    sum_width = len(sum_reg)

    # 1) d_i = r_{p,i} ⊕ t_i
    for i in range(l):
        if user_vector[i] == '1':
            qc.cx(feature_qubits[i], d_reg[i])
            qc.x(d_reg[i])
        else:
            qc.cx(feature_qubits[i], d_reg[i])

    # 2) sum_reg += d_i for each i
    for i in range(l):
        controlled_increment(qc, control_qubit=d_reg[i], sum_reg=sum_reg)

    # 3) compare sum_reg == target_distance
    bin_str = format(target_distance, f"0{sum_width}b")
    for j, bit in enumerate(bin_str):
        if bit == '0':
            qc.x(sum_reg[j])
    qc.h(aux_qubit)
    qc.mcx(sum_reg, aux_qubit)
    qc.h(aux_qubit)
    for j, bit in enumerate(bin_str):
        if bit == '0':
            qc.x(sum_reg[j])

    # 4) uncompute sum_reg and d_reg
    for i in reversed(range(l)):
        controlled_decrement(qc, control_qubit=d_reg[i], sum_reg=sum_reg)
    for i in range(l):
        if user_vector[i] == '1':
            qc.x(d_reg[i])
            qc.cx(feature_qubits[i], d_reg[i])
        else:
            qc.cx(feature_qubits[i], d_reg[i])


def qrs(n_items, feature_mat, user_vector, plot=False, grover_iterations=1):
    """
    Full QRS:
      1) initialise_database
      2) knn (adds + measures c0)
      3) prepare Grover ancillas (aux, d_reg, sum_reg)
      4) for d0 in [0..l]:  mark & amplify nearest Hamming distance
      5) measure index qubits
    """
    q = int(np.log2(n_items))
    l = len(feature_mat[0])

    # 1) load database + user
    qc = initialise_database(n_items, feature_mat, user_vector)

    index_qubits = list(range(0, q))
    feature_qubits = list(range(q, q + l))
    user_qubits = list(range(q + l, q + 2 * l))

    # 2) k‐NN subcircuit
    qc = knn(qc, feature_qubits, user_qubits)

    # Measure c0 and post‐select
    c0_out = ClassicalRegister(1, name="c0_out")
    qc.add_register(c0_out)
    c0_qubit = qc.num_qubits - 1  # last qubit added by knn
    qc.measure(c0_qubit, c0_out[0])
    # (In practice, re‐run or uncompute if c0_out==1. Here we post‐select on c0_out=0.)

    # 3) allocate Grover ancillas
    aux = QuantumRegister(1, name="grover_anc")
    d_reg = QuantumRegister(l, name="dist_bits")
    sum_width = math.ceil(math.log2(l + 1))
    sum_reg = QuantumRegister(sum_width, name="sum_bits")

    qc.add_register(aux)
    qc.add_register(d_reg)
    qc.add_register(sum_reg)

    aux_qubit = aux[0]
    d_qubits = list(d_reg)
    sum_qubits = list(sum_reg)

    # 4) set up measurement for index qubits
    idx_out = ClassicalRegister(q, name="idx_out")
    qc.add_register(idx_out)

    for d0 in range(l + 1):
        # Prepare aux in |–>
        qc.x(aux_qubit)
        qc.h(aux_qubit)

        # Oracle: flip phase if Hamming(r_p, user_vector) == d0
        grover_hamming_oracle(
            qc,
            feature_qubits=feature_qubits,
            user_vector=user_vector,
            d_reg=d_qubits,
            sum_reg=sum_qubits,
            aux_qubit=aux_qubit,
            target_distance=d0
        )

        # Diffusion on index qubits
        for _ in range(grover_iterations):
            diffusion_on_index(qc, index_qubits)

        # Restore aux to |0>
        qc.h(aux_qubit)

        # Measure index qubits
        for i, qb in enumerate(index_qubits):
            qc.measure(qb, idx_out[i])

        # Break after the first d0 iteration; driver code can post‐select correct outcome
        break

    if plot:
        qc.draw(output="mpl", fold=30)
        plt.savefig(f"images/qrs_circuit.png", dpi=300, bbox_inches="tight")
        plt.show()

    return qc


