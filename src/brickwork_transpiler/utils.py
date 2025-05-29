import numpy as np
from qiskit import QuantumCircuit


def reorder_via_transpose(psi: np.ndarray) -> np.ndarray:
    """
    Swap MSB <-> LSB conventions in an n-qubit statevector.
    Given a flat array `psi` of length 2**n, returns the same amplitudes
    but with all indices bit-reversed.

    This is equivalent to reshaping to (2,)*n, transposing axes [n-1,...,0],
    then flattening, but uses an explicit, vectorized index mapping.
    """
    # 1) ensure a C-contiguous 1D array
    psi = np.asarray(psi, order='C')
    dim = psi.size

    # 2) infer n and sanity-check
    n = int(np.log2(dim))
    if 2**n != dim:
        raise ValueError(f"Length {dim} is not a power of 2; cannot infer n")

    # 3) build an array of all indices [0,1,...,2**n-1]
    idx = np.arange(dim, dtype=int)

    # 4) bit-reverse each index in a vectorized way
    rev = np.zeros_like(idx)
    for bit in range(n):
        rev = (rev << 1) | ((idx >> bit) & 1)

    # 5) apply the permutation
    return psi[rev]


def reorder_via_transpose_n(psi: np.ndarray) -> np.ndarray:
    """
    Flip MSB↔LSB in an n-qubit statevector of length 2**n.
    Works for n = 1,2,3,4,… as long as psi.size == 2**n.
    """
    psi = np.ascontiguousarray(psi)
    dim = psi.size
    n   = int(np.log2(dim))
    if 2**n != dim:
        raise ValueError(f"Length {dim} is not a power of 2")

    # view as an n-way tensor, each dim=2
    psi_tensor = psi.reshape((2,)*n)
    # reverse the axes
    psi_t = psi_tensor.transpose(list(reversed(range(n))))
    # flatten back
    return psi_t.reshape(dim)



def assert_equal_up_to_global_phase(state1, state2, tol=1e-6):
    """
    Assert that two quantum state vectors are equal up to global phase.
    I've got 99 problems but global phase aint one

    Parameters:
    - state1, state2: iterable of complex numbers (e.g., output of a simulator)
    - tol: numerical tolerance for isclose()

    Raises:
    - AssertionError with diagnostic info if the assertion fails.
    """
    state1 = np.array(state1, dtype=complex)
    state2 = np.array(state2, dtype=complex)

    # Normalize (just in case)
    norm1 = np.linalg.norm(state1)
    norm2 = np.linalg.norm(state2)

    if np.isclose(norm1, 0.0, atol=tol) or np.isclose(norm2, 0.0, atol=tol):
        raise AssertionError(f"One of the states' norms is zero (cannot compare)")

    state1 /= norm1
    state2 /= norm2

    inner_product = np.vdot(state1, state2)
    magnitude = np.abs(inner_product)

    if not np.isclose(magnitude, 1.0, atol=tol, rtol=0.0): # added rtol for border checking
        raise AssertionError(
            f"States are not equal up to global phase.\n"
            f"Inner product: {inner_product}\n"
            f"Absolute value: {magnitude:.6f} (should be close to 1)"
        )

    else: return True


def permute_qubits(circ: QuantumCircuit, perm: list[int]) -> QuantumCircuit:
    n = circ.num_qubits
    if sorted(perm) != list(range(n)):
        raise ValueError("perm must be a rearrangement of 0..n-1")

    new_circ = QuantumCircuit(n, circ.num_clbits)
    old_to_new = {
        old: new_circ.qubits[perm[idx]]
        for idx, old in enumerate(circ.qubits)
    }
    clbit_map = { old: new for old,new in zip(circ.clbits, new_circ.clbits) }

    for instr, qargs, cargs in circ.data:
        new_qargs = [old_to_new[q] for q in qargs]
        new_cargs = [clbit_map[c]  for c in cargs]
        new_circ.append(instr, new_qargs, new_cargs)

    return new_circ


def index_to_coordinates(index: int, num_cols: int) -> tuple[int, int]:
    """
    Maps a linear node index to (row, column) assuming row-major layout.

    Parameters:
    - index: int – Node index in flat list
    - num_cols: int – Number of columns (time steps)

    Returns:
    - (row, column): tuple[int, int]
    """
    row = index // num_cols
    column = index % num_cols
    return (row, column)


def map_indices_to_coordinates(indices: list[int], num_cols: int) -> dict[int, tuple[int, int]]:
    """
    Maps list of node indices to (row, column) tuples.

    Parameters:
    - indices: list[int] – Flat node indices
    - num_cols: int – Number of columns (time steps)

    Returns:
    - Dictionary of index → (row, column)
    """
    return {i: index_to_coordinates(i, num_cols) for i in indices}


# Returns a list with qubit entries to be permuted when compared with qiskit reference outputs
def get_qubit_entries(bw_pattern):
    if bw_pattern is None:
        raise AssertionError("bw_pattern is None")

    qubit_entries = [t[0] for t in bw_pattern.output_nodes]
    return qubit_entries


def calculate_qiskit_permutation(list):
    # reverse the Graphix list to go from big-endian → little-endian
    list.reverse()
    # entries_le == [1, 4, 0, 3, 2]

    # invert it: for each Qiskit qubit j, find its position in list
    perm = [list.index(j) for j in range(len(list))]
    return perm


def get_qiskit_permutation(bw_pattern):
    if bw_pattern is None:
        raise AssertionError("bw_pattern is None")

    qubit_entries = [t[0] for t in bw_pattern.output_nodes]

    # reverse the Graphix list to go from big-endian → little-endian
    qubit_entries.reverse()
    # entries_le == [1, 4, 0, 3, 2]

    # invert it: for each Qiskit qubit j, find its position in list
    perm = [qubit_entries.index(j) for j in range(len(qubit_entries))]
    return perm


def calculate_ref_state_from_qiskit_circuit(bw_pattern, qc, input_vector):    # TODO: One param when merged to computation_graph obj
    if bw_pattern is None:
        raise AssertionError("bw_pattern is None")

    qubit_entries = [t[0] for t in bw_pattern.output_nodes]

    # reverse the Graphix list to go from big-endian → little-endian
    qubit_entries.reverse()
    # entries_le == [1, 4, 0, 3, 2]

    # invert it: for each Qiskit qubit j, find its position in list
    perm = [qubit_entries.index(j) for j in range(len(qubit_entries))]

    qc_perm = permute_qubits(qc, perm=perm)
    return input_vector.evolve(qc_perm)


def feature_to_generator(feature_mat):
    """
    Given feature_mat: list of N rows (each a list of l bits),
    returns an l x q generator matrix G (as a list of l lists of length q)
    satisfying f(i) = G @ i mod 2 for all i in 0..N-1,
    or raises ValueError if feature_mat is not a linear code.
    """
    # 1) Dimensions
    N = len(feature_mat)
    if N == 0:
        raise ValueError("feature_mat must have at least one row")
    l = len(feature_mat[0])
    if any(len(row) != l for row in feature_mat):
        raise ValueError("All rows of feature_mat must have the same length l")

    # 2) Check N = 2^q
    if N & (N - 1) != 0:
        raise ValueError(f"N = {N} is not a power of 2")
    q = N.bit_length() - 1

    # 3) Build G by sampling f(e_k) for k=0..q-1
    #    e_k has index = (1 << k)
    basis_indices = [1 << k for k in range(q)]
    # G[j][k] = j-th bit of f(2^k)
    G = [[feature_mat[i][j] for i in basis_indices] for j in range(l)]

    # 4) Verify that f(i) == G @ i mod 2 for all i
    for i in range(N):
        # get binary expansion of i: bits[k]
        bits = [(i >> k) & 1 for k in range(q)]
        # compute G·bits mod2
        f_lin = [ sum(G[j][k] * bits[k] for k in range(q)) & 1
                  for j in range(l) ]
        if f_lin != feature_mat[i]:
            raise ValueError(f"feature_mat is not linear: "
                             f"mismatch at i={i}: "
                             f"expected {feature_mat[i]}, got {f_lin}")

    return G


def feature_to_affine_generator(feature_mat):
    """
    feature_mat: list of N = 2^q rows, each an l-bit list.
    Returns (G, c) where
      c = feature_mat[0]        # the affine offset
      G = l×q binary matrix     # pure linear part
    such that f(i) = (G @ i) XOR c.
    Raises ValueError if the *residual* map is not linear.
    """
    N = len(feature_mat)
    if N == 0:
        raise ValueError("Need at least one row")
    l = len(feature_mat[0])
    if any(len(r) != l for r in feature_mat):
        raise ValueError("All rows must have length l")

    # check power-of-two
    if N & (N-1) != 0:
        raise ValueError(f"N={N} not a power of 2")
    q = N.bit_length() - 1

    # 1) extract offset c = f(0)
    c = feature_mat[0].copy()

    # 2) build the *residual* table r(i) = f(i) XOR c
    residual = [
        [(bit ^ c_j) for (bit, c_j) in zip(feature_mat[i], c)]
        for i in range(N)
    ]

    # 3) now residual[0] == zero must hold
    if any(residual[0]):
        raise ValueError("Residual at i=0 not zero; can't form affine map")

    # 4) use the same linear-code routine on residual
    #    (this will check r(i+j)=r(i)+r(j) for all i,j)
    G = feature_to_generator(residual)
    return G, c
