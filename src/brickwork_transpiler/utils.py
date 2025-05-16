import numpy as np

import numpy as np

import numpy as np

# def reorder_via_transpose(psi: np.ndarray) -> np.ndarray:
#     """
#     Flip MSB ↔ LSB in an n-qubit statevector of length 2**n.
#     Works for any n ≥ 1, and will error if psi.size is not a power of two.
#
#     Parameters
#     ----------
#     psi : np.ndarray
#         Flat statevector of length 2**n (dtype=complex64 or complex128).
#
#     Returns
#     -------
#     np.ndarray
#         Reordered flat statevector, same shape and dtype as input.
#     """
#     # 0) ensure C-contiguity so reshape/transpose behave predictably
#     psi = np.ascontiguousarray(psi)
#
#     # 1) basic checks
#     dim = psi.size
#     n   = int(np.log2(dim))
#     if 2**n != dim:
#         raise ValueError(f"Length {dim} is not a power of 2 (cannot infer n)")
#
#     # 2) reshape into an n-way tensor, *not* (2)*n
#     #    the comma is critical: (2,)*n → (2,2,…,2) of length n
#     shape = (2,)*n
#     psi_tensor = psi.reshape(shape)
#
#     # 3) reverse the qubit axes: [n-1, n-2, …, 0]
#     psi_t = psi_tensor.transpose(list(range(n-1, -1, -1)))
#
#     # 4) flatten back to a vector
#     return psi_t.reshape(dim)

import numpy as np

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


# def reorder_via_transpose(psi: np.ndarray) -> np.ndarray:
#     """
#     Flip MSB ↔ LSB in an n-qubit statevector of length 2**n.
#     Works for any n ≥ 1.
#     """
#     psi = np.ascontiguousarray(psi)     # enforce C-contiguity
#     dim = psi.size
#     n   = int(np.log2(dim))
#     if 2**n != dim:
#         raise ValueError(f"Length {dim} is not a power of 2")
#
#     # 1) reshape into an n-way tensor with 2 entries per qubit
#     shape = (2,)*n
#     psi_tensor = psi.reshape(shape)
#
#     # 2) reverse the qubit axes
#     psi_t = psi_tensor.transpose(list(range(n-1, -1, -1)))
#
#     # 3) flatten back to a vector
#     return psi_t.reshape(dim)


# def reorder_via_transpose(psi: np.ndarray) -> np.ndarray:
#     """
#     Given an (2**n,) statevector `psi` in Qiskit ordering (q0 is LSB),
#     returns the same amplitudes in Graphix ordering (q0 is MSB), or vice versa.
#     """
#     dim = psi.size
#     n = int(np.log2(dim))
#     # 1) view as tensor of shape (2,2,...,2)
#     psi_tensor = psi.reshape((2,)*n)
#     # 2) reverse the axis order
#     psi_t = psi_tensor.transpose(list(reversed(range(n))))
#     # 3) flatten back out
#     return psi_t.reshape(dim)


import numpy as np


import numpy as np

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


from qiskit import QuantumCircuit

def permute_qubits(circ: QuantumCircuit, permutation: list[int]) -> QuantumCircuit:
    """
    Returns a new QuantumCircuit equivalent to `circ` but with its qubits
    permuted according to `permutation`.

    Args:
        circ:           The input QuantumCircuit.
        permutation:    A list of length circ.num_qubits such that
                        new_position = permutation[old_position].

    Returns:
        QuantumCircuit  A new circuit with the same operations routed through
                        the permuted qubit ordering.
    """
    # Sanity checks
    n = circ.num_qubits
    if sorted(permutation) != list(range(n)):
        raise ValueError(f"permutation must be a rearrangement of 0..{n-1}")

    # Create an empty circuit with the same regs
    new_circ = QuantumCircuit(n, circ.num_clbits)

    # Build a map from old qubit objects to new ones
    old_to_new = { old: new_circ.qubits[permutation[idx]]
                   for idx, old in enumerate(circ.qubits) }

    # Similarly map classical bits 1:1
    clbit_map = { old: new for old, new in zip(circ.clbits, new_circ.clbits) }

    # Replay each instruction with remapped qubits/clbits
    for instr, qargs, cargs in circ.data:
        new_qargs = [ old_to_new[q] for q in qargs ]
        new_cargs = [ clbit_map[c]  for c in cargs ]
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
