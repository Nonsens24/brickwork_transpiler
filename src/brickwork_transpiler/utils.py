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
    if 2 ** n != dim:
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
    n = int(np.log2(dim))
    if 2 ** n != dim:
        raise ValueError(f"Length {dim} is not a power of 2")

    # view as an n-way tensor, each dim=2
    psi_tensor = psi.reshape((2,) * n)
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

    if not np.isclose(magnitude, 1.0, atol=tol, rtol=0.0):  # added rtol for border checking
        raise AssertionError(
            f"States are not equal up to global phase.\n"
            f"Inner product: {inner_product}\n"
            f"Absolute value: {magnitude:.6f} (should be close to 1)"
        )

    else:
        return True


def permute_qubits(circ: QuantumCircuit, perm: list[int]) -> QuantumCircuit:
    n = circ.num_qubits
    if sorted(perm) != list(range(n)):
        raise ValueError("perm must be a rearrangement of 0..n-1")

    new_circ = QuantumCircuit(n, circ.num_clbits)
    old_to_new = {
        old: new_circ.qubits[perm[idx]]
        for idx, old in enumerate(circ.qubits)
    }
    clbit_map = {old: new for old, new in zip(circ.clbits, new_circ.clbits)}

    for instr, qargs, cargs in circ.data:
        new_qargs = [old_to_new[q] for q in qargs]
        new_cargs = [clbit_map[c] for c in cargs]
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


def calculate_ref_state_from_qiskit_circuit(bw_pattern, qc,
                                            input_vector):  # TODO: One param when merged to computation_graph obj
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
        f_lin = [sum(G[j][k] * bits[k] for k in range(q)) & 1
                 for j in range(l)]
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
    if N & (N - 1) != 0:
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


def time_complexity_knn_grover(q: int, l: int, c: int) -> dict[str, int]:
    """
    Estimate the (asymptotic) gate‐count “time complexity” of the full QRS circuit
    as described in Sawerwain & Wróblewski (2019). All counts here are in terms of
    elementary gates (single‐ and two‐qubit gates) and follow the O(·) formulas from the paper.

    Parameters
    ----------
    q : int
        Number of qubits used to label database entries.
        •  N = 2**q  is the size of the classical database (number of rows).
        •  e.g. if q=4, then N=16 database items.

    l : int
        Width (in bits) of the feature‐vector register.
        •  Each database‐entry feature and the user‐feature both live in an l-qubit subregister.
        •  Hamming‐distance calculations, Grover oracles, etc., all scale in l.

    c : int
        “Extra” two‐qubit‐gate count arising when one decomposes multi-controlled Z (and
        other multi-controlled) gates in Grover’s diffusion/oracle into elementary CNOT+single‐qubit gates.
        •  In practice  c ≈ O(l) or O(l²) depending on your compilation strategy.
        •  Here we simply treat c as a parameter: “how many extra two-qubit gates are needed to implement each
           multi-controlled Z in the Grover step?”

    Returns
    -------
    dict[str, int]
        A dictionary with the following keys (all counts are exact, not Big-O, but they follow the
        same growth rates given in the paper):

        • "Database_creation" : int
            ⟶ O₁(l, N) = 2ˡ  +  N·(N − 1)//2
            (number of gates to initialize the full database: Hadamards for user‐feature superposition
            plus pairwise permutes to encode all N classical rows in |ψ_ff〉).

        • "kNN_distance"     : int
            ⟶ O₂(l) = 3·l + 2
            (gate‐count for computing Hamming distances + “quantum summing” of distance).

        • "Grover_amplify"   : int
            ⟶ O₃(l, c) = 7·l + 2·c + 3
            (gate‐count in the amplitude‐amplification stage: multi-controlled oracle + diffusion
            after k-NN has picked “close” entries).

        • "Total"            : int
            Sum of the above three numbers.  This is the total elementary-gate count of the
            entire circuit (once), up to constants.

    Notes
    -----
    1. The dominating term is “Database_creation” (O(2ˡ + N²))—once you build the database,
       you don’t have to rebuild it on each query.
    2. If you only care about the per‐query cost (i.e., once the database is loaded),
       then “Total_per_query” = kNN_distance + Grover_amplify = (3·l + 2) + (7·l + 2·c + 3) = 10·l + 2·c + 5.
    3. In many realistic settings, N = 2**q is very large (exponential in q), so the N·(N−1)/2 term
       is truly quadratic in N.
    4. The parameter c depends on your circuit decomposition strategy for multi-controlled Z gates
       in Grover: for instance, a naive “barenco-style” proliferation gives c = O(l), whereas more
       optimized decompositions can push it toward O(l log l).
    5. All counts here assume a standard universal gate set (Hadamard, single‐qubit Z/X, and CNOT).
    """

    N = 2 ** q

    # 1) Database creation cost: 2^l (Hadamards for user feature) + N*(N−1)/2 (permutations for |ψ_ff⟩)
    database_creation = 2 ** l + (N * (N - 1)) // 2

    # 2) k-NN Hamming-distance and summing: exactly 3*l + 2 gates
    kNN_distance = 3 * l + 2

    # 3) Grover amplitude amplification: 7*l + 2*c + 3 gates
    grover_amplify = 7 * l + 2 * c + 3

    total = database_creation + kNN_distance + grover_amplify

    return {
        "Database_creation": database_creation,
        "kNN_distance": kNN_distance,
        "Grover_amplify": grover_amplify,
        "Total": total,
    }


import csv
import os


class BufferedCSVWriter:
    def __init__(self, filename, headers):
        self.filename = filename
        self.headers = headers
        self.row = {}  # Current row buffer
        # Write header if file doesn't exist
        if not os.path.exists(filename):
            with open(filename, mode='w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writeheader()

    def set(self, key, value):
        self.row[key] = value

    def flush(self):
        # Write current row (can be partial), then clear buffer
        with open(self.filename, mode='a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.headers, restval="")
            writer.writerow(self.row)
        self.row = {}


from qiskit.quantum_info import Statevector
from qiskit.circuit.library import PermutationGate
import numpy as np
import math

# --- 1) Save the mapping (call this once, right after transpile) -----------------
def extract_logical_to_physical(qc_in, qc_out):
    """
    Returns a list `logical_to_physical` such that logical_to_physical[j]
    is the *output wire index* in qc_out where logical qubit j (of qc_in) ended up.
    """
    layout = getattr(qc_out, "layout", None)

    # Preferred: Transpiler provides a dict {Qubit_in -> int(out_index)}
    if layout is not None and getattr(layout, "input_qubit_mapping", None) is not None:
        return [layout.input_qubit_mapping[q] for q in qc_in.qubits]

    # Fallback: try to compose initial_layout with routing_permutation()
    if layout is not None and getattr(layout, "initial_layout", None) is not None:
        init = layout.initial_layout  # {Qubit_in -> int(pre-route index)}
        pre = [init[q] for q in qc_in.qubits]
        try:
            route = list(layout.routing_permutation())  # maps pre-route index -> final index
        except Exception:
            route = list(range(qc_out.num_qubits))
        return [route[p] for p in pre]

    # Last resort: assume identity (no layout/routing applied)
    return list(range(min(qc_in.num_qubits, qc_out.num_qubits)))

def undo_layout_on_state(state, logical_to_physical, total_qubits=None):
    """
    Given a state in the *transpiled circuit's wire order*, return a Statevector
    reordered so that the original logical qubits come first.

    Qiskit PermutationGate expects a source->destination map:
      perm[i] = j  means "move qubit at index i to position j".

    Here, logical_to_physical[j] = p says "logical j ended up at physical p".
    To undo, we must move physical p -> position j, i.e. perm[p] = j.

    `state` can be:
      - qiskit.quantum_info.Statevector
      - a 1D numpy array/list of amplitudes
    If you pass a numpy array, also pass `total_qubits` (or it will be inferred).
    """
    # Ensure Statevector, infer N if needed
    if isinstance(state, Statevector):
        sv = state
        N = sv.num_qubits
    else:
        arr = np.asarray(state, dtype=complex)
        N = total_qubits if total_qubits is not None else int(round(math.log2(arr.size)))
        sv = Statevector(arr, dims=[2]*N)

    logical_idx = list(logical_to_physical)
    L = len(logical_idx)

    # Put any extra qubits (ancillas) after the logical ones, preserving physical order
    anc_idx = [i for i in range(N) if i not in logical_idx]

    # Build the source->dest permutation:
    #   - logicals: physical p -> logical position j  (perm[p] = j)
    #   - ancillas: remaining physical indices -> positions L, L+1, ...
    perm = [None] * N
    for j, p in enumerate(logical_idx):
        perm[p] = j
    for k, p in enumerate(anc_idx):
        perm[p] = L + k

    # Validate we produced a complete permutation
    if any(x is None for x in perm):
        missing = [i for i, x in enumerate(perm) if x is None]
        raise ValueError(f"Incomplete permutation; missing assignments for physical indices {missing}.")
    if sorted(perm) != list(range(N)):
        raise ValueError(f"Invalid permutation {perm} (must be a permutation of 0..{N-1}).")

    return sv.evolve(PermutationGate(perm))

# # --- 2) Undo the mapping on a statevector after simulation ----------------------
# def undo_layout_on_state(state, logical_to_physical, total_qubits=None):
#     """
#     Given a state in the *transpiled circuit's wire order*, return a Statevector
#     reordered so that the original logical qubits come first.
#
#     `state` can be:
#       - qiskit.quantum_info.Statevector
#       - a 1D numpy array/list of amplitudes
#     If you pass a numpy array, also pass `total_qubits` (or it will be inferred from len(state)).
#     """
#     # Ensure Statevector, infer N if needed
#     if isinstance(state, Statevector):
#         sv = state
#         N = sv.num_qubits
#     else:
#         arr = np.asarray(state, dtype=complex)
#         N = total_qubits if total_qubits is not None else int(round(math.log2(arr.size)))
#         sv = Statevector(arr, dims=[2]*N)
#
#     logical_idx = list(logical_to_physical)
#     # Put any extra qubits (ancillas) after the logical ones, preserving their order
#     anc_idx = [i for i in range(N) if i not in logical_idx]
#
#     # PermutationGate(pattern): pattern[k] = m  means "move qubit m to position k"
#     pattern = logical_idx + anc_idx
#     return sv.evolve(PermutationGate(pattern))
