import numpy as np
# import qiskit.compiler.transpiler
from matplotlib import pyplot as plt
from qiskit.quantum_info import Statevector
import os
import re

# from tensorflow.python.keras.utils.layer_utils import print_summary

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
# from libs.gospel.gospel.brickwork_state_transpiler.brickwork_state_transpiler import transpile
from src.brickwork_transpiler import brickwork_transpiler, utils
from src.brickwork_transpiler.algorithms import qrs_knn_grover
from src.brickwork_transpiler.algorithms.hhl import generate_example_hhl_QC
from src.brickwork_transpiler.bfk_encoder import encode_pattern
from src.brickwork_transpiler.circuits import minimal_qrs
from src.brickwork_transpiler.experiments import qrs_full_transpilation, plot_qrs_data, qrs_no_db_transpilation, \
    qft_transpilation, hhl_transpilation
from src.brickwork_transpiler.noise import DepolarisingInjector, SimulationNoiseModel
# from src.brickwork_transpiler.noise import to_noisy_pattern
import src.brickwork_transpiler.circuits as circuits

from qiskit import transpile, ClassicalRegister, QuantumRegister
from qiskit_aer import AerSimulator
from qiskit import QuantumCircuit
import src.brickwork_transpiler.experiments.minimal_qrs_transpilation_simulation as mqrs
# from src.brickwork_transpiler.utils import calculate_ref_state_from_qiskit_circuit, \
#     logical_to_final_physical, axes_phys_from_tn_outputs, undo_sabre_on_tn_state, get_qiskit_permutation, \
#     extract_logical_to_physical, undo_layout_on_state, extract_physical_to_logical, extract_logical_to_physical24, \
#     undo_sabre_to_preroute_physical24, permute_qubits

from typing import List, Tuple, Iterable
from qiskit.transpiler.layout import TranspileLayout, Layout
from qiskit.circuit import Qubit

from src.brickwork_transpiler.utils import extract_logical_to_physical24, undo_sabre_to_preroute_physical24, \
    calculate_ref_state_from_qiskit_circuit, extract_logical_to_physical, undo_layout_on_state


# def permutation_input_to_output(tl: TranspileLayout) -> List[int]:
#     """
#     Robustly compute π mapping input wire index -> output wire index from a TranspileLayout,
#     tolerating dynamic layouts (non-identical Qubit objects across mappings).
#
#     Strategy
#     --------
#     1) Build keys for input qubits: (reg_name, qubit_index, reg_size) -> input_index.
#     2) Extract physical->virtual from initial_layout; convert virtual to input_index.
#     3) Extract physical->virtual from final_layout; convert virtual to output_index (its .index).
#     4) Glue via physical index: for each physical p, set π[input_i_at_p] = output_j_at_p.
#     """
#     n = tl._input_qubit_count
#
#     def qkey(q: Qubit) -> Tuple[str, int, int]:
#         reg = getattr(q, "register", None)
#         name = getattr(reg, "name", None)
#         size = getattr(reg, "size", None)
#         if size is None and reg is not None:
#             try:
#                 size = len(reg)
#             except Exception:
#                 size = -1
#         return (name, getattr(q, "index", None), size)
#
#     # ---- 1) input qubit keys -> input indices
#     inkey_to_i = {qkey(q): i for q, i in tl.input_qubit_mapping.items()}
#
#     # ---- helpers to extract p->v pairs from a Layout regardless of internal orientation
#     def phys_to_virtual_pairs(layout: Layout) -> Iterable[Tuple[int, Qubit]]:
#         # Preferred: official API if available
#         try:
#             m = layout.get_physical_bits()  # recent Qiskit returns dict[int, Qubit]
#             if isinstance(m, dict):
#                 return list(m.items())
#         except Exception:
#             pass
#
#         # Try private internals (different Qiskit versions)
#         for attr in ("_p2v", "_int_to_bit", "_physical_to_virtual"):
#             d = getattr(layout, attr, None)
#             if isinstance(d, dict) and d:
#                 return list(d.items())
#
#         # Fallback: infer orientation via items()
#         pairs = []
#         try:
#             for a, b in layout.items():
#                 if isinstance(a, int):
#                     pairs.append((a, b))
#                 elif isinstance(b, int):
#                     pairs.append((b, a))
#         except Exception:
#             pass
#         if not pairs:
#             raise ValueError("Could not extract physical→virtual mapping from Layout.")
#         return pairs
#
#     # ---- 2) p -> input index i
#     p_to_i = {}
#     for p, v in phys_to_virtual_pairs(tl.initial_layout):
#         k = qkey(v)
#         if k not in inkey_to_i:
#             raise KeyError(
#                 f"Virtual {v} (key={k}) from initial_layout not found in input_qubit_mapping."
#             )
#         p_to_i[p] = inkey_to_i[k]
#
#     # ---- 3) p -> output index j
#     def out_index(vout: Qubit) -> int:
#         j = getattr(vout, "index", None)
#         if j is not None:
#             return j
#         # Fallback via output_qubit_list using structural key
#         out_list = list(getattr(tl, "_output_qubit_list", []))
#         if out_list:
#             kv = qkey(vout)
#             for idx, vv in enumerate(out_list):
#                 if qkey(vv) == kv:
#                     return idx
#         raise KeyError(f"Cannot determine output index for {vout}")
#
#     p_to_j = {p: out_index(v) for p, v in phys_to_virtual_pairs(tl.final_layout)}
#
#     # ---- 4) stitch via physical index
#     pi = [-1] * n
#     for p, i in p_to_i.items():
#         if p not in p_to_j:
#             raise KeyError(f"Physical index {p} present in initial_layout missing in final_layout.")
#         pi[i] = p_to_j[p]
#
#     if any(x < 0 for x in pi):
#         raise ValueError(f"Incomplete permutation derived: {pi}")
#     return pi
#
#
#
# def permutation_output_to_input(tl: TranspileLayout) -> List[int]:
#     """
#     Inverse permutation: maps output wire index to input wire index.
#     """
#     pi = permutation_input_to_output(tl)
#     inv = [None] * len(pi)
#     for i, j in enumerate(pi):
#         inv[j] = i
#     return inv
#
#
#
#
# import numpy as np
#
# def reorder_statevector_by_perm(psi, perm):
#     """
#     Reorder a statevector to reflect a qubit permutation π with π[i] = j
#     (input wire i becomes output wire j). Compatible with Qiskit's
#     little-endian basis ordering (qubit-0 = LSB).
#
#     Parameters
#     ----------
#     psi : (2**n,) array_like of complex
#         Statevector amplitudes in the *input* wire ordering.
#     perm : sequence of int, length n
#         Permutation π mapping input wire index -> output wire index.
#
#     Returns
#     -------
#     np.ndarray
#         Statevector amplitudes in the *output* wire ordering.
#
#     Notes
#     -----
#     Implementation reshapes to an n-index tensor and uses a single transpose.
#     With C-order flatten, axes 0..(n-1) correspond to qubits (n-1)..0.
#     """
#     psi = np.asarray(psi)
#     if psi.ndim != 1:
#         raise ValueError("psi must be a 1D statevector.")
#     # Infer n and validate length
#     n = int(round(np.log2(psi.size)))
#     if (1 << n) != psi.size:
#         raise ValueError("len(psi) must be a power of 2.")
#     if len(perm) != n:
#         raise ValueError("perm must have length n = log2(len(psi)).")
#     if sorted(perm) != list(range(n)):
#         raise ValueError("perm must be a permutation of range(n).")
#
#     perm = np.asarray(perm, dtype=int)
#
#     # inv[j] = i such that π[i] = j
#     inv = np.empty(n, dtype=int)
#     inv[perm] = np.arange(n, dtype=int)
#
#     # Build axis permutation:
#     # axis index for qubit q is (n-1-q); set output axis (n-1-j) to input axis (n-1-inv[j]).
#     axes = np.empty(n, dtype=int)
#     for j in range(n):
#         axes[n - 1 - j] = n - 1 - inv[j]
#
# #     # Reshape -> transpose -> flatten
# #     out = psi.reshape((2,) * n).transpose(axes).reshape(-1)
# #     # Make contiguous (optional), then return
# #     return np.ascontiguousarray(out)
# #
#
# # --- Drop-in pipeline (place this near your imports) ---
# import os
# import numpy as np
# import matplotlib.pyplot as plt
# from typing import List, Tuple, Iterable, Optional, Dict
# from qiskit.quantum_info import Statevector
# from qiskit.circuit import Qubit
# from qiskit.transpiler.layout import Layout, TranspileLayout
#
# # ========== Core utilities ==========
#
# def reorder_statevector_by_perm(psi: np.ndarray, perm: List[int]) -> np.ndarray:
#     """Reorder a statevector by permutation π with π[i]=j (input wire i → output wire j).
#     Qiskit is little-endian (q[0]=LSB)."""
#     psi = np.asarray(psi)
#     if psi.ndim != 1:
#         raise ValueError("psi must be 1D.")
#     n = int(round(np.log2(psi.size)))
#     if (1 << n) != psi.size:
#         raise ValueError("len(psi) must be a power of 2.")
#     if len(perm) != n or sorted(perm) != list(range(n)):
#         raise ValueError("perm must be a permutation of range(n).")
#
#     perm = np.asarray(perm, dtype=int)
#     inv = np.empty(n, dtype=int)   # inv[j] = i with π[i] = j
#     inv[perm] = np.arange(n, dtype=int)
#
#     axes = np.empty(n, dtype=int)
#     # axis for qubit q is (n-1-q). Set output axis (n-1-j) to input axis (n-1-inv[j]).
#     for j in range(n):
#         axes[n - 1 - j] = n - 1 - inv[j]
#     return np.ascontiguousarray(psi.reshape((2,) * n).transpose(axes).reshape(-1))
#
# def graphix_to_qiskit_statevector(psi: np.ndarray, graphix_is_msb0: bool = True) -> np.ndarray:
#     """Convert Graphix indexing → Qiskit (LSB-first). If Graphix is MSB-first (default),
#     this is a qubit-order reversal; otherwise it’s a no-op."""
#     psi = np.asarray(psi)
#     if psi.ndim != 1:
#         raise ValueError("psi must be 1D.")
#     n = int(round(np.log2(psi.size)))
#     if (1 << n) != psi.size:
#         raise ValueError("len(psi) must be 2**n.")
#     if not graphix_is_msb0:
#         return np.ascontiguousarray(psi)
#     return np.ascontiguousarray(psi.reshape((2,) * n).transpose(tuple(range(n - 1, -1, -1))).reshape(-1))
#
# # ========== Dynamic-safe SABRE layout extraction ==========
#
# def _qkey(q: Qubit) -> Tuple[Optional[str], Optional[int], Optional[int]]:
#     reg = getattr(q, "register", None)
#     name = getattr(reg, "name", None)
#     size = getattr(reg, "size", None)
#     if size is None and reg is not None:
#         try: size = len(reg)
#         except Exception: size = -1
#     return (name, getattr(q, "index", None), size)
#
# def _phys_to_virtual_pairs(layout: Layout) -> Iterable[Tuple[int, Qubit]]:
#     # Preferred modern API
#     try:
#         m = layout.get_physical_bits()  # dict[int -> Qubit]
#         if isinstance(m, dict):
#             return list(m.items())
#     except Exception:
#         pass
#     # Private fallbacks
#     for attr in ("_p2v", "_int_to_bit", "_physical_to_virtual"):
#         d = getattr(layout, attr, None)
#         if isinstance(d, dict) and d:
#             return list(d.items())
#     # Infer from items()
#     pairs = []
#     try:
#         for a, b in layout.items():
#             if isinstance(a, int): pairs.append((a, b))
#             elif isinstance(b, int): pairs.append((b, a))
#     except Exception:
#         pass
#     if not pairs:
#         raise ValueError("Could not extract physical→virtual mapping from Layout.")
#     return pairs
#
# def permutation_input_to_output(tl: TranspileLayout) -> List[int]:
#     """π: input wire index -> output wire index. Dynamic-layout safe."""
#     n = tl._input_qubit_count
#     inkey_to_i = {_qkey(q): i for q, i in tl.input_qubit_mapping.items()}  # input virtuals -> input indices
#     # physical -> input index
#     p_to_i: Dict[int, int] = {}
#     for p, v in _phys_to_virtual_pairs(tl.initial_layout):
#         k = _qkey(v)
#         if k not in inkey_to_i:
#             raise KeyError(f"Virtual {v} (key={k}) from initial_layout not in input_qubit_mapping.")
#         p_to_i[p] = inkey_to_i[k]
#     # physical -> output index
#     def out_index(vout: Qubit) -> int:
#         j = getattr(vout, "index", None)
#         if j is not None: return j
#         out_list = list(getattr(tl, "_output_qubit_list", []))
#         if out_list:
#             kv = _qkey(vout)
#             for idx, vv in enumerate(out_list):
#                 if _qkey(vv) == kv: return idx
#         raise KeyError(f"Cannot determine output index for {vout}")
#     p_to_j = {p: out_index(v) for p, v in _phys_to_virtual_pairs(tl.final_layout)}
#     # stitch
#     pi = [-1]*n
#     for p, i in p_to_i.items():
#         if p not in p_to_j:
#             raise KeyError(f"Physical {p} present in initial_layout missing in final_layout.")
#         pi[i] = p_to_j[p]
#     if any(x < 0 for x in pi):
#         raise ValueError(f"Incomplete permutation derived: {pi}")
#     return pi
#
# def inverse_perm(perm: List[int]) -> List[int]:
#     inv = [None]*len(perm)
#     for i, j in enumerate(perm): inv[j] = i
#     return inv
#


import numpy as np
from typing import Sequence, Literal

def undo_statevector_permutation(
    state: np.ndarray,
    mapping: Sequence[int],
    *,
    convention: Literal["src_at_dest","dest_from_src"] = "src_at_dest",
    little_endian: bool = True,
) -> np.ndarray:
    """
    Undo a qubit permutation on a 2**n statevector.

    mapping:
      - if convention == "src_at_dest": mapping[j] = old qubit at new position j
      - if convention == "dest_from_src": mapping[i] = new position of old qubit i
    """
    psi = np.asarray(state).reshape(-1)
    size = psi.size
    n = int(np.log2(size))
    if (1 << n) != size:
        raise ValueError(f"length {size} is not a power of two")
    p = list(mapping)
    if len(p) != n or sorted(p) != list(range(n)):
        raise ValueError("mapping must be a permutation of range(n)")

    # Normalize to the "src_at_dest" form expected by the transpose logic
    if convention == "dest_from_src":
        p = np.argsort(p).tolist()

    q2axis = (lambda q: n - 1 - q) if little_endian else (lambda q: q)
    axes_perm = [q2axis(q) for q in p]
    inv_axes = np.argsort(axes_perm)

    return np.transpose(psi.reshape((2,) * n), axes=inv_axes).reshape(-1)

# import numpy as np
# from typing import Sequence
#
# def undo_statevector_permutation(
#     state: np.ndarray,
#     sources_at_positions: Sequence[int],
#     *,
#     little_endian: bool = True,
# ) -> np.ndarray:
#     """
#     Undo a permutation of qubits that was applied to an n-qubit statevector.
#
#     Parameters
#     ----------
#     state : array_like, shape (2**n,)
#         The (permuted) statevector in the *current* qubit order.
#     sources_at_positions : Sequence[int], length n
#         A permutation p of [0..n-1] describing the permutation that was applied:
#             position j was replaced by the value from position p[j]
#         (i.e., the qubit now sitting at position j used to be qubit p[j]).
#         Example: p = [0, 2, 1] means positions 1 and 2 were swapped.
#     little_endian : bool, default True
#         If True (Qiskit-style), qubit 0 is the least significant bit (last axis).
#         If False, qubit 0 is the most significant bit (first axis).
#
#     Returns
#     -------
#     np.ndarray, shape (2**n,)
#         The statevector restored to the original qubit order.
#
#     Notes
#     -----
#     Time/memory are Θ(2**n). Validates that len(p) == n and p is a permutation.
#     """
#     psi = np.asarray(state)
#     if psi.ndim != 1:
#         psi = psi.reshape(-1)
#
#     size = psi.size
#     n = int(np.log2(size))
#     if (1 << n) != size:
#         raise ValueError(f"Statevector length {size} is not a power of two.")
#
#     p = list(sources_at_positions)
#     if len(p) != n or sorted(p) != list(range(n)):
#         raise ValueError("sources_at_positions must be a permutation of range(n) with length n.")
#
#     # Map qubit index -> axis index in the reshaped tensor
#     q2axis = (lambda q: n - 1 - q) if little_endian else (lambda q: q)
#
#     # Axes order that *was used* to apply the permutation
#     axes_perm = [q2axis(q) for q in p]
#
#     # To undo, transpose by the inverse of that axes permutation
#     inv_axes = np.argsort(axes_perm)
#
#     psi_t = psi.reshape((2,) * n)
#     return np.transpose(psi_t, axes=inv_axes).reshape(-1)


import numpy as np
from typing import Sequence, Literal

def _q2axis(q: int, n: int, little_endian: bool) -> int:
    return (n - 1 - q) if little_endian else q

def apply_qubit_permutation(
    state: np.ndarray,
    mapping: Sequence[int],
    *,
    convention: Literal["src_at_dest","dest_from_src"],
    little_endian: bool = True,
) -> np.ndarray:
    """
    Apply a qubit permutation to a statevector.
    mapping:
      - "src_at_dest": mapping[j] = old qubit that moves to new position j
      - "dest_from_src": mapping[i] = new position of old qubit i
    """
    psi = np.asarray(state).reshape(-1)
    n = int(np.log2(psi.size))
    if (1 << n) != psi.size: raise ValueError("length must be 2**n")
    p = list(mapping)
    if sorted(p) != list(range(n)): raise ValueError("mapping must be a permutation")

    if convention == "dest_from_src":
        p = np.argsort(p).tolist()  # convert to src_at_dest

    axes_old = list(range(n))  # axes for original tensor
    axes_new = [ _q2axis(q, n, little_endian) for q in p ]   # which old qubit sits at each new position
    # axes_new is expressed in "qubit indices"; convert to tensor-axis indices:
    # original tensor axes are ordered by qubit indices via _q2axis(q)
    qubit_axis = { q: _q2axis(q, n, little_endian) for q in range(n) }
    axes_perm = [ qubit_axis[q] for q in p ]  # order of old axes in the new tensor

    return np.transpose(psi.reshape((2,)*n), axes=axes_perm).reshape(-1)

def undo_qubit_permutation(
    state: np.ndarray,
    mapping: Sequence[int],
    *,
    convention: Literal["src_at_dest","dest_from_src"],
    little_endian: bool = True,
) -> np.ndarray:
    """Undo a previously applied permutation (inverse of apply_qubit_permutation)."""
    psi = np.asarray(state).reshape(-1)
    n = int(np.log2(psi.size))
    if (1 << n) != psi.size: raise ValueError("length must be 2**n")
    p = list(mapping)
    if sorted(p) != list(range(n)): raise ValueError("mapping must be a permutation")

    if convention == "dest_from_src":
        p = np.argsort(p).tolist()  # convert to src_at_dest

    # axes used to apply the permutation:
    axes_perm = [ _q2axis(q, n, little_endian) for q in p ]
    inv_axes = np.argsort(axes_perm)

    return np.transpose(psi.reshape((2,)*n), axes=inv_axes).reshape(-1)



# --- core helpers ---

def main():

    # input_vec = Statevector.from_label('+++++')  # three-qubit plus state
    # qc = QuantumCircuit(4)
    # qc.cx(0, 3)
    #
    # qc.draw(output='mpl',
    #                     fold=40,
    #                     style="iqp"
    #                     )
    #
    #
    #
    # bw_pattern, col_map, transpiled_qc = brickwork_transpiler.transpile(
    #     qc, input_vec,
    #     routing_method="sabre",
    #     layout_method="trivial",
    #     with_ancillas=False
    # )
    #
    # # Plot informative graph
    # visualiser.plot_brickwork_graph_from_pattern(bw_pattern,
    #                                              show_angles=True,
    #                                              node_colours=col_map,
    #                                              use_node_colours=True,
    #                                              title="Brickwork graph: sel_decomp_test",
    #                                              save_plot=True)




    # hhl_transpilation.experiment_hhl_transpilation(7)
    # hhl_transpilation.plot_single_hhl_dataset("HHL_logN_sq")
    # hhl_transpilation.plot_hhl_from_multiple_files("thesis_hhl_final_plots_unopt_dense_d_avg")
    # qft_transpilation.plot_single_qft_dataset("thesis_qft_log3_bound")
    # plot_qrs_data.plot_qrs_with_db_scaling_from_files("qrs_with_db_L_no_decomp")
    # hhl_transpilation.plot_depth_per_gate_vs_m_logfit("vs_m_plot")
    hhl_transpilation.plot_scaling_with_m_as_width("d_or_m_overlay_hhl_exp_toeplitz_raw_no_log")
    # hhl_transpilation.plot_hhl_davg_three()
    return 0

    # 1) Create the |++> state directly
    input_vec = Statevector.from_label('+++++')  # three-qubit plus state

    # 2) Define your 2-qubit circuit (no H gates needed)
    qc = QuantumCircuit(5)
    qc.cx(0, 1)
    qc.s(2)
    qc.rx(np.pi/3, 2)
    qc.t(2)
    qc.x(2)
    qc.cx(3, 4)

    qc.draw(output='mpl',
                        fold=40,
                        style="iqp"
                        )

    bw_pattern, col_map, transpiled_qc = brickwork_transpiler.transpile(qc, input_vec,
                                                         routing_method="sabre",
                                                         layout_method="sabre",
                                                         with_ancillas=False)
    # ref_state = calculate_ref_state_from_qiskit_circuit(bw_pattern, qc, input_vec)
    transpiled_qc.draw(output='mpl',
                        fold=40,
                        style="iqp"
                        )
    visualiser.plot_brickwork_graph_from_pattern(bw_pattern,
                                                 node_colours=col_map,
                                                 use_node_colours=True,
                                                 title="test_cx_from_zero_upper",
                                                 )

    # Optimise for tensornetwork backand and simulation efficiency
    bw_pattern.standardize()  # puts commands into N-E-M-(X/Z/C) order
    bw_pattern.shift_signals()  # optional but recommended; reduces feedforward

    tn = bw_pattern.simulate_pattern(backend="tensornetwork", graph_prep="parallel")
    psi = tn.to_statevector()  # state on your declared outputs

    ref_state = calculate_ref_state_from_qiskit_circuit(bw_pattern, qc, input_vec)
    # ref_state = input_vec.evolve(qc)

    # Round trip must be identity for any random state
    rng = np.random.default_rng(0)
    psi = rng.normal(size=32) + 1j * rng.normal(size=32)
    psi /= np.linalg.norm(psi)

    p2l = [3, 4, 1, 0, 2]  # physical -> logical
    # simulate a (hypothetical) conversion logical -> physical:
    l2p = np.argsort(p2l).tolist()

    psi_phys = apply_qubit_permutation(psi, l2p, convention="dest_from_src", little_endian=True)
    psi_back = undo_qubit_permutation(psi_phys, p2l, convention="src_at_dest", little_endian=True)
    assert np.allclose(psi, psi_back)

    mapping = extract_logical_to_physical(qc, transpiled_qc)
    print("mapping: ", mapping)

    # If you simulated with your MBQC engine and got a flat numpy array `psi`:
    # sv_logical_from_mbqc = undo_layout_on_state(psi, mapping, total_qubits=transpiled_qc.num_qubits)
    unmapped_sv = undo_statevector_permutation(psi, mapping)



    print("ref_state: ", ref_state)

    print("unmapped_sv: ", unmapped_sv)

    # Compare output up to global phase
    assert utils.assert_equal_up_to_global_phase(unmapped_sv, ref_state)

    return 0
    # # --- imports ---------------------------------------------------------------
    # import math, numpy as np
    # from qiskit import QuantumCircuit
    # from qiskit.quantum_info import Statevector
    # from qiskit.circuit.library import PermutationGate
    #
    # # --- helpers ---------------------------------------------------------------
    #
    # def extract_logical_to_physical(qc_in, qc_out):
    #     """
    #     l2p[j] = p1 (final physical wire) for logical qubit j of qc_in, as realized in qc_out.
    #     """
    #     layout = getattr(qc_out, "layout", None)
    #     if layout is None or getattr(layout, "initial_layout", None) is None:
    #         n = min(qc_in.num_qubits, qc_out.num_qubits)
    #         return list(range(n))
    #     init = layout.initial_layout  # virtual -> pre-route physical
    #     pre = [init[q] for q in qc_in.qubits]  # p0
    #     try:
    #         route = list(layout.routing_permutation())  # p0 -> p1
    #     except Exception:
    #         route = list(range(qc_out.num_qubits))
    #     return [route[p0] for p0 in pre]  # j -> p1
    #
    # def _perm_LE_for_route_inverse(route):
    #     """
    #     Convert wire-space inverse routing r^{-1}: p1->p0 into little-endian qubit-index
    #     permutation for PermutationGate (perm[i]=j means move qubit i to position j).
    #     """
    #     N = len(route)
    #     # r^{-1} over wire indices
    #     rinv_wire = [None] * N
    #     for p0, p1 in enumerate(route):
    #         rinv_wire[p1] = p0
    #     # wire <-> little-endian index: i_LE = N-1 - i_wire
    #     perm_LE = [None] * N
    #     for i_LE in range(N):
    #         i_wire = N - 1 - i_LE
    #         j_wire = rinv_wire[i_wire]
    #         j_LE = N - 1 - j_wire
    #         perm_LE[i_LE] = j_LE
    #     # sanity
    #     if sorted(perm_LE) != list(range(N)):
    #         raise ValueError(f"Bad LE permutation derived from route: {perm_LE}")
    #     return perm_LE
    #
    # def undo_sabre_on_state(state, qc_out, total_qubits=None):
    #     """
    #     Given 'state' in the final-physical wire order (Graphix result),
    #     undo SABRE's routing to obtain **pre-route physical** order.
    #     """
    #     # Normalize to Statevector
    #     if isinstance(state, Statevector):
    #         sv = state
    #     else:
    #         arr = np.asarray(state, dtype=complex)
    #         N = total_qubits if total_qubits is not None else int(round(math.log2(arr.size)))
    #         sv = Statevector(arr, dims=[2] * N)
    #
    #     layout = getattr(qc_out, "layout", None)
    #     if layout is None:
    #         return sv
    #
    #     try:
    #         route = list(layout.routing_permutation())  # p0 -> p1
    #     except Exception:
    #         return sv  # nothing to undo
    #
    #     perm_LE = _perm_LE_for_route_inverse(route)  # r^{-1} in LE indices
    #     return sv.evolve(PermutationGate(perm_LE))
    #
    # def calculate_ref_state_from_qiskit_circuit(
    #         bw_pattern,
    #         qc_in,
    #         transpiled_qc,
    #         input_vector,
    #         *,
    #         align_graphix_to_qiskit: bool = True,
    #         ancilla_init: str = "0",
    #         permute_qubits_fn=None,  # pass your existing permute_qubits if you have it
    # ):
    #     """
    #     Build a reference state that matches **pre-route physical** order.
    #
    #     Steps:
    #       1) (optional) Align Graphix’s big-endian labeling to Qiskit logical indices.
    #       2) Evolve 'input_vector' through the (optionally permuted) qc_in -> logical-order state.
    #       3) Permute logical j -> pre-route physical p0 using initial_layout.
    #       4) (optional) Pad ancillas if transpiled has more qubits.
    #     """
    #     if bw_pattern is None:
    #         raise AssertionError("bw_pattern is None")
    #
    #     qc_used = qc_in
    #     if align_graphix_to_qiskit:
    #         # Graphix output_nodes are big-endian; flip to little-endian
    #         entries_le = [t[0] for t in bw_pattern.output_nodes][::-1]
    #         # For each Qiskit logical j, find its position in Graphix’s LE list
    #         perm_logical_to_graphix = [entries_le.index(j) for j in range(len(entries_le))]
    #         if permute_qubits_fn is not None:
    #             # If you have a circuit-level reindexer, use it:
    #             qc_used = permute_qubits_fn(qc_in, perm=perm_logical_to_graphix)
    #         else:
    #             # Otherwise, we’ll apply this logical permutation to the STATE after evolution.
    #             pass
    #
    #     # Logical evolution on the intended input (|+++++> here)
    #     ref_state_logical = input_vector.evolve(qc_used)
    #
    #     # If we didn't reindex the circuit, apply the logical permutation to the state now
    #     if align_graphix_to_qiskit and permute_qubits_fn is None:
    #         N = qc_in.num_qubits
    #         # perm[i]=j moves logical i → logical position j
    #         perm = list(range(N))
    #         for i, j in enumerate(perm_logical_to_graphix):
    #             perm[i] = j
    #         ref_state_logical = ref_state_logical.evolve(PermutationGate(perm))
    #
    #     # Ancilla padding if needed
    #     L = qc_used.num_qubits
    #     N = transpiled_qc.num_qubits
    #     if N < L:
    #         raise ValueError(f"Transpiled has {N} qubits but qc_in has {L}.")
    #     if N > L:
    #         anc = Statevector.from_label(ancilla_init * (N - L))
    #         ref_state_logical = anc.tensor(ref_state_logical)  # ancillas on higher wires
    #
    #     # logical → pre-route physical (initial_layout)
    #     layout = getattr(transpiled_qc, "layout", None)
    #     if layout is None or getattr(layout, "initial_layout", None) is None:
    #         return ref_state_logical
    #
    #     init = layout.initial_layout  # virtual -> pre-route physical
    #     pre = [init[q] for q in qc_used.qubits]  # p0 for each logical j
    #     perm_j_to_p0 = list(range(N))
    #     for j, p0 in enumerate(pre):
    #         perm_j_to_p0[j] = p0
    #     # Fill any remaining slots bijectively (only relevant if N > L)
    #     if N > L:
    #         used = set(pre)
    #         rest = [p for p in range(N) if p not in used]
    #         for k, j in enumerate(range(L, N)):
    #             perm_j_to_p0[j] = rest[k]
    #     if sorted(perm_j_to_p0) != list(range(N)):
    #         raise ValueError(f"Invalid logical→pre-route map: {perm_j_to_p0}")
    #
    #     return ref_state_logical.evolve(PermutationGate(perm_j_to_p0))
    #
    # # --- your pipeline (cleaned) -----------------------------------------------
    #
    # # 0) Define input and circuit
    # input_vec = Statevector.from_label('+++++')
    #
    # qc_in = QuantumCircuit(5)
    # qc_in.cx(0, 1)
    # qc_in.s(2)
    # qc_in.rx(np.pi / 3, 2)
    # qc_in.t(2)
    # qc_in.x(2)
    # qc_in.cx(3, 4)
    #
    # # 1) Transpile to brickwork / SABRE
    # bw_pattern, col_map, transpiled_qc = brickwork_transpiler.transpile(
    #     qc_in, input_vec,
    #     routing_method="sabre",
    #     layout_method="sabre",
    #     with_ancillas=False
    # )
    #
    # # 2) Graphix simulation
    # bw_pattern.standardize()
    # bw_pattern.shift_signals()
    # tn = bw_pattern.simulate_pattern(backend="tensornetwork", graph_prep="parallel")
    # psi = tn.to_statevector()
    #
    # # 3) Undo SABRE on Graphix state -> pre-route physical order
    # try:
    #     route = list(transpiled_qc.layout.routing_permutation())  # p0 -> p1 (debug/use)
    # except Exception:
    #     route = list(range(transpiled_qc.num_qubits))
    # psi_preroute_physical = undo_sabre_on_state(psi, transpiled_qc, total_qubits=transpiled_qc.num_qubits)
    #
    # # (optional) also compute l2p (logical -> final physical) for consistency checks
    # l2p = extract_logical_to_physical(qc_in, transpiled_qc)  # j -> p1
    # pre = [transpiled_qc.layout.initial_layout[q] for q in qc_in.qubits]  # j -> p0
    # assert [route[p0] for p0 in pre] == l2p
    #
    # # 4) Qiskit reference in the SAME (pre-route physical) order
    # ref_state_preroute_physical = calculate_ref_state_from_qiskit_circuit(
    #     bw_pattern=bw_pattern,
    #     qc_in=qc_in,
    #     transpiled_qc=transpiled_qc,
    #     input_vector=input_vec,
    #     align_graphix_to_qiskit=True,  # set False if you already aligned elsewhere
    #     ancilla_init="0",
    #     permute_qubits_fn=None  # or None if you want state-level permutation
    # )
    #
    # # 5) Compare up to global phase
    # print(f"Qiskit ref_state: {ref_state_preroute_physical}")
    # print(f"psi_preroute_physical: {psi_preroute_physical}")
    # if utils.assert_equal_up_to_global_phase(psi_preroute_physical, ref_state_preroute_physical):
    #     print("Same up to global phase!")
    #
    # return 0

    # qc_in, input_vec = circuits.h_and_cx_circ()

    input_vec = Statevector.from_label('+++++')  # three-qubit plus state

    # 2) Define your 2-qubit circuit (no H gates needed)
    qc_in = QuantumCircuit(5)
    qc_in.cx(0, 1)
    qc_in.s(2)
    qc_in.rx(np.pi/3, 2)
    qc_in.t(2)
    qc_in.x(2)
    qc_in.cx(3, 4)

    bw_pattern, col_map, transpiled_qc = brickwork_transpiler.transpile(
        qc_in, input_vec,
        routing_method="sabre",
        layout_method="sabre",
        with_ancillas=False
    )

    ref_state = calculate_ref_state_from_qiskit_circuit(bw_pattern, qc_in, input_vec)
    # ref_state = Statevector.from_instruction(qc_in).data
    # correct: apply qc_in to the |+++++> state
    ref_state = input_vec.evolve(qc_in)  # NOT Statevector.from_instruction(qc_in)

    transpiled_qc.draw(output='mpl',
                        fold=40,
                        style="iqp"
                        )

    # Preprocess data to standard form
    bw_pattern.standardize()
    bw_pattern.shift_signals()

    # Get the output state from Graphix simulation
    tn = bw_pattern.simulate_pattern(backend="tensornetwork", graph_prep="parallel")
    psi = tn.to_statevector()

    # Plot informative graph
    visualiser.plot_brickwork_graph_from_pattern(bw_pattern,
                                                 show_angles=True,
                                                 node_colours=col_map,
                                                 use_node_colours=True,
                                                 title="Brickwork graph: H + CX")


    # After transpile
    l2p = extract_logical_to_physical24(qc_in, transpiled_qc)  # logical -> final physical
    psi_preroute_physical = undo_sabre_to_preroute_physical24(
        psi, l2p, qc_in, transpiled_qc, total_qubits=transpiled_qc.num_qubits
    )

    from qiskit.circuit.library import PermutationGate

    def logical_to_preroute_perm(qc_in, qc_out):
        """Return perm with PermutationGate semantics perm[i]=j (move i→j),
           that moves logical position j to its pre-route physical index p0."""
        layout = qc_out.layout
        init = layout.initial_layout
        pre = [init[q] for q in qc_in.qubits]  # p0 for each logical j
        N = qc_out.num_qubits
        perm = list(range(N))
        for j, p0 in enumerate(pre):
            perm[j] = p0
        return perm

    perm_j_to_p0 = logical_to_preroute_perm(qc_in, transpiled_qc)
    ref_state_preroute_physical = ref_state.evolve(PermutationGate(perm_j_to_p0))

    # Build the Graphix state, then undo SABRE on it (as you already do):
    route = list(transpiled_qc.layout.routing_permutation())  # p0 -> p1
    # psi_preroute_physical = undo_sabre_with_permgate24(psi, route, total_qubits=transpiled_qc.num_qubits)

    # Build the reference in the *same* pre-route physical order:
    ref_state_preroute_physical = calculate_ref_state_from_qiskit_circuit(
        bw_pattern, qc_in, transpiled_qc, input_vec, align_graphix_to_qiskit=True
    )

    print(f"Qiskit ref_state: {ref_state_preroute_physical}")
    print(f"psi_preroute_physical: {psi_preroute_physical}")
    # Compare output state upto global phase
    if utils.assert_equal_up_to_global_phase(psi_preroute_physical, ref_state_preroute_physical):
        print("Same up to global phase!")

    return 0

    # user_vec = [0, 0]
    # qc_in, input_vec = circuits.minimal_qrs(user_vec)
    #
    # print(qc_in)
    #
    # bw_pattern, col_map, transpiled_qc = brickwork_transpiler.transpile(
    #     qc_in, routing_method="sabre", layout_method=None, with_ancillas=False
    # )
    #
    # # (optional) keep your standardization steps
    # bw_pattern.standardize()
    # bw_pattern.shift_signals()
    #
    # print(transpiled_qc.layout)
    #
    # # If you already have a Graphix→Qiskit permutation from the pattern, keep using it.
    # # Otherwise, set `use_graphix_perm = False` to use the endian-based converter.
    # use_graphix_perm = True
    # graphix_is_msb0 = True  # set False if Graphix already outputs little-endian (q0=LSB)
    #
    # # --- Graphix simulation (TN backend as you had) ---
    # tn = bw_pattern.simulate_pattern(backend="tensornetwork", graph_prep="parallel")
    # psi = tn.to_statevector()  # Graphix indexing, 1D complex array
    #
    # # --- Qiskit reference statevector in INPUT wire order (no SABRE) ---
    # try:
    #     qc_nom = qc_in.remove_final_measurements(inplace=False)
    # except Exception:
    #     qc_nom = qc_in
    # psi_ref = np.asarray(Statevector.from_instruction(qc_nom).data, dtype=complex)
    #
    # # Sanity check sizes
    # if psi.size != psi_ref.size:
    #     raise RuntimeError(
    #         f"Graphix vector length {psi.size} differs from reference {psi_ref.size}. "
    #         "Check for extra ancilla or qubit count mismatch."
    #     )
    #
    # # --- Step 1: Graphix indexing → Qiskit indexing (no SABRE yet) ---
    # if use_graphix_perm:
    #     # Your function (assumed to map Graphix pattern qubit order to Qiskit’s logical order)
    #     perm_gx_to_qk = get_qiskit_permutation(bw_pattern)  # π: graphix_index -> qiskit_index
    #     psi_qk = reorder_statevector_by_perm(psi, perm_gx_to_qk)
    # else:
    #     # Fallback: treat Graphix as MSB-first and just reverse qubit order
    #     psi_qk = graphix_to_qiskit_statevector(psi, graphix_is_msb0=graphix_is_msb0)
    #
    # # --- Step 2: Undo SABRE wiring (output → input order) using inverse permutation ---
    # pi = permutation_input_to_output(transpiled_qc.layout)  # input -> output
    # inv = inverse_perm(pi)  # output -> input
    #
    # # Graphix sim corresponds to SABRE's *output* logical order; bring it back to *input* order:
    # psi_gx_input = reorder_statevector_by_perm(psi_qk, inv)
    #
    # # --- Step 3: Align global phase, compute metrics, report ---
    # overlap = np.vdot(psi_ref, psi_gx_input)  # <ref | graphix>
    # phase = -np.angle(overlap)
    # psi_gx_aligned = psi_gx_input * np.exp(1j * phase)
    #
    # fidelity = float(np.abs(np.vdot(psi_ref, psi_gx_aligned)) ** 2)
    # err = psi_gx_aligned - psi_ref
    # max_abs_err = float(np.max(np.abs(err)))
    # rms_err = float(np.sqrt(np.mean(np.abs(err) ** 2)))
    #
    # print("SABRE π (input→output):", pi)
    # print("Inverse (output→input):", inv)
    # print(f"Global phase (radians): {phase:.12f}")
    # print(f"Fidelity: {fidelity:.12e}")
    # print(f"max|err|: {max_abs_err:.12e}   RMS err: {rms_err:.12e}")
    #
    # # =========================
    # # === Your plotting part ==
    # # =========================
    # # You previously plotted after applying π (SABRE output basis). You can choose:
    # plot_basis = "input"  # options: "input" (default, matches qc_in) or "output"
    #
    # if plot_basis == "input":
    #     amps = np.asarray(psi_gx_input, dtype=complex)  # input wire order (matches psi_ref)
    # elif plot_basis == "output":
    #     # reproduce your previous psi_out for visualization in SABRE output order
    #     psi_out = reorder_statevector_by_perm(psi_qk, pi)
    #     amps = np.asarray(psi_out, dtype=complex)
    # else:
    #     raise ValueError("plot_basis must be 'input' or 'output'.")
    #
    # # --- normalize (safe) ---
    # norm = np.linalg.norm(amps)
    # if not np.isclose(norm, 1.0):
    #     amps = amps / norm
    #
    # # --- statevector -> probabilities (nonzero only) ---
    # sv = Statevector(amps)
    # probs_dict = sv.probabilities_dict()  # keys are bitstrings; bit[-1] is q0 (LSB) in Qiskit
    #
    # # --- post-select c0 = 0 on a chosen qubit index (in the chosen basis) ---
    # eps = 1e-12
    # flip_plot_labels = True
    # target_qubit = 5  # interpret this index in the selected 'plot_basis'
    # top_k = None  # e.g., 40
    #
    # filtered = {b: p for b, p in probs_dict.items() if b[-(target_qubit + 1)] == '0'}
    # Z = sum(filtered.values())
    # if Z == 0:
    #     raise ValueError("After conditioning on c0=0, no support remains. Check target_qubit/basis/endianness.")
    # filtered = {b: (p / Z) for b, p in filtered.items() if (p / Z) > eps}
    #
    # if top_k is not None:
    #     filtered = dict(sorted(filtered.items(), key=lambda kv: kv[1], reverse=True)[:top_k])
    #
    # xs = sorted(filtered)
    # ys = [filtered[b] for b in xs]
    # tot = sum(ys)
    # probs = [100.0 * y / tot for y in ys]
    #
    # labels = [b[::-1] for b in xs] if flip_plot_labels else xs
    #
    # os.makedirs("images/plots", exist_ok=True)
    # plt.figure(figsize=(7, 3.6))
    # plt.bar(range(len(xs)), probs)
    # plt.xticks(range(len(xs)), labels, rotation=70, ha='right')
    # plt.ylabel("Probability (%)")
    # title_basis = "Input" if plot_basis == "input" else "SABRE Output"
    # plt.title(f"Minimal QRS (post-select c0=0) | feature: {user_vec} | basis: {title_basis}")
    # plt.tight_layout()
    # plt.savefig(f"images/plots/minimal_qrs_statevector_plot_{user_vec}_{plot_basis}.png", dpi=300, bbox_inches="tight")
    # plt.show()
    #
    #
    # return 0
    user_vec = [0, 0]
    qc_in, input_vec = circuits.minimal_qrs(user_vec)

    print(qc_in)

    bw_pattern, col_map, transpiled_qc = brickwork_transpiler.transpile(
        qc_in, input_vec,
        routing_method="sabre",
        layout_method="sabre",
        with_ancillas=False
    )

    # visualiser.plot_brickwork_graph_from_pattern(bw_pattern,
    #                                              node_colours=col_map,
    #                                              use_node_colours=True,
    #                                              title="noise_test",
    #                                              )

    bw_pattern.standardize()
    bw_pattern.shift_signals()

    print(transpiled_qc.layout)


    # 0) simulate MBQC (TN backend)
    tn = bw_pattern.simulate_pattern(backend="densitymatrix", noise_model=SimulationNoiseModel(p1=0.05, p2=0.001))
    psi = tn.to_statevector()
    eps = 1e-12  # or 0.0 for no pruning
    flip_plot_labels = True
    target_qubit = 0
    top_k = None  # e.g., 40 to keep only the top-40 bars (None = keep all)


    # --- statevector -> probabilities (nonzero only) ---
    amps = np.asarray(psi, dtype=complex)
    norm = np.linalg.norm(amps)
    if not np.isclose(norm, 1.0):
        amps = amps / norm


    sv = Statevector(amps)

    shots = 20_000  # choose your shot budget
    counts = sv.sample_counts(shots=shots)  # dict[str, int]; b[-1] is q0 (little-endian)

    # (optional) prune tiny mass in *counts*
    min_count = 1
    counts = {b: c for b, c in counts.items() if c >= min_count}

    # (optional) keep only the top-k outcomes
    if top_k is not None:
        counts = dict(sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:top_k])

    # --- plotting ---
    xs = sorted(counts)  # canonical keys
    ys = [counts[b] for b in xs]  # <-- COUNTS now

    labels = [b[::-1] for b in xs] if flip_plot_labels else xs

    os.makedirs("images/plots", exist_ok=True)
    plt.figure(figsize=(7, 3.6))
    plt.bar(range(len(xs)), ys)
    plt.xticks(range(len(xs)), labels, rotation=70, ha='right')
    plt.ylabel("Counts")
    plt.title(f"Minimal QRS Results with noise (no post-selection) | feature: {user_vec}")
    plt.tight_layout()
    plt.savefig(f"images/plots/minimal_qrs_NOISE_CLIFFORD_statevector_evolution_plot_{user_vec}.png",
                dpi=300, bbox_inches="tight")
    plt.show()

    # # --- post-select c0 = 0 on target_qubit, renormalize, prune tiny mass ---
    # filtered = {b: p for b, p in probs_dict.items() }#if b[target_qubit] == '0'}
    # Z = sum(filtered.values())
    # if Z == 0:
    #     raise ValueError("After conditioning on c0=0, no support remains. Check target_qubit/endian.")
    # filtered = {b: (p / Z) for b, p in filtered.items() if (p / Z) > eps}
    #
    # # (optional) keep only the top-k outcomes
    # if top_k is not None:
    #     filtered = dict(sorted(filtered.items(), key=lambda kv: kv[1], reverse=True)[:top_k])
    #
    # # --- your plotting style ---
    # xs = sorted(filtered)  # canonical keys
    # ys = [filtered[b] for b in xs]
    #
    #
    # # DISPLAY-ONLY: flip label bit order if you want MSB↔LSB swapped
    # labels = [b[::-1] for b in xs] if flip_plot_labels else xs
    #
    # os.makedirs("images/plots", exist_ok=True)
    # plt.figure(figsize=(7, 3.6))
    # plt.bar(range(len(xs)), ys)
    # plt.xticks(range(len(xs)), labels, rotation=70, ha='right')  # readable labels
    # plt.ylabel("Counts")
    # plt.title(f"Minimal QRS Results with noise (no post-selection) | feature: {user_vec}")
    # plt.tight_layout()
    # plt.savefig(f"images/plots/minimal_qrs_NOISE_CLIFFORD_statevector_evolution_plot_{user_vec}.png", dpi=300, bbox_inches="tight")
    # plt.show()


    return 0

    # --- post-select c0 = 0 on the ancilla bit ---
    filtered = {b: p for b, p in probs_dict.items() if b[anc_pos] == '0'}

    flip_plot_labels = True  # display-only MSB↔LSB flip
    user_feature = "demo"  # customize per run
    eps = 1e-12  # drop tiny probabilities post-selection
    top_k = None  # e.g., 40 to keep only the top-40 bars (None = keep all)

    # --- statevector -> probabilities (nonzero only) ---
    amps = np.asarray(psi, dtype=complex)
    norm = np.linalg.norm(amps)
    if not np.isclose(norm, 1.0):
        amps = amps / norm

    sv = Statevector(amps)
    probs_dict = sv.probabilities_dict()  # keys: bitstrings; b[-1] is q0 (little-endian)

    # --- post-select c0 = 0 on target_qubit, renormalize, prune tiny mass ---
    filtered = {b: p for b, p in probs_dict.items() if b[-(target_qubit + 1)] == '0'}
    Z = sum(filtered.values())
    if Z == 0:
        raise ValueError("After conditioning on c0=0, no support remains. Check target_qubit/endian.")
    filtered = {b: (p / Z) for b, p in filtered.items() if (p / Z) > eps}

    # (optional) keep only the top-k outcomes
    if top_k is not None:
        filtered = dict(sorted(filtered.items(), key=lambda kv: kv[1], reverse=True)[:top_k])

    # --- your plotting style ---
    xs = sorted(filtered)  # canonical keys
    ys = [filtered[b] for b in xs]
    tot = sum(ys)
    probs = [100.0 * y / tot for y in ys]  # percentage

    # DISPLAY-ONLY: flip label bit order if you want MSB↔LSB swapped
    labels = [b[::-1] for b in xs] if flip_plot_labels else xs

    os.makedirs("images/plots", exist_ok=True)
    plt.figure(figsize=(7, 3.6))
    plt.bar(range(len(xs)), probs)
    plt.xticks(range(len(xs)), labels, rotation=70, ha='right')  # readable labels
    plt.ylabel("Probability (%)")
    plt.title(f"Minimal QRS Results (post-selected c0=0) | feature: {user_vec}")
    plt.tight_layout()
    plt.savefig(f"images/plots/minimal_qrs_NOISE_CLIFFORD_statevector_evolution_plot_{user_vec}.png", dpi=300, bbox_inches="tight")
    plt.show()



    return 0

    # # 1) SABRE mapping (original logical order → final physical)
    # L2P = logical_to_final_physical(qc_in, transpiled_qc)
    #
    # # 2) TN axis order in physical indices
    # axes_phys = axes_phys_from_tn_outputs(tn, bw_pattern)
    #
    # # 3) Put TN state back into original logical order (now ancilla aligns)
    # sv_logical_from_mbqc = undo_sabre_on_tn_state(psi, L2P, axes_phys)

    # tn = bw_pattern.simulate_pattern(backend="tensornetwork", graph_prep="parallel")#, noise_model=SimulationNoiseModel)
    # # tn = bw_pattern.simulate_pattern(backend="densitymatrix", noise_model=SimulationNoiseModel(p1=0.05, p2=0.0001))
    # psi = tn.to_statevector()
    #
    #
    # mapping = extract_logical_to_physical(qc_in, transpiled_qc)
    # # If you simulated with your MBQC engine and got a flat numpy array `psi`:
    # sv_logical_from_mbqc = undo_layout_on_state(psi, mapping)#, total_qubits=transpiled_qc.num_qubits)



    # ---- inputs / knobs ----

    # --- helpers: parse Qubit(...) safely (no deprecated attributes) ---
    _QUBIT_RE = re.compile(
        r"Qubit\(QuantumRegister\(\s*\d+\s*,\s*'([^']+)'\s*\)\s*,\s*(\d+)\s*\)\Z"
    )

    def _qb_info(qb):
        m = _QUBIT_RE.match(repr(qb))
        if not m:
            raise ValueError(f"Unexpected qubit repr: {repr(qb)}")
        name, idx = m.group(1), int(m.group(2))
        return name, idx

    # 1) Qiskit circuit index of c0[0] (used for filtering)
    def qiskit_index_of_c0(layout, circuit, reg_name="c0", reg_index=0):
        v = None
        for qb, vidx in layout.input_qubit_mapping.items():
            nm, ix = _qb_info(qb)
            if nm == reg_name and ix == reg_index:
                v = vidx
                break
        if v is None:
            raise KeyError(f"{reg_name}[{reg_index}] not found in input_qubit_mapping.")
        out_qb = layout.final_layout[v]
        return circuit.find_bit(out_qb).index

    # 2) Build Graphix-qubit index → Qiskit-circuit index map, g_to_q[g] = q
    def g_to_q_map(layout, circuit, graphix_reg_order=("q", "c0")):
        """
        Assumes Graphix enumerates qubits by register then index, e.g. q[0..N-1], then c0[0].
        If your source order differs, adjust graphix_reg_order.
        """
        # (a) determine virtual index -> circuit index
        v_to_q = {}
        for qb, v in layout.input_qubit_mapping.items():
            out_qb = layout.final_layout[v]
            q = circuit.find_bit(out_qb).index
            v_to_q[v] = q

        # (b) determine Graphix enumeration order over the *input* qubits
        items = []
        for qb, v in layout.input_qubit_mapping.items():
            nm, ix = _qb_info(qb)
            # order registers according to graphix_reg_order, unknown names go last alphabetically
            try:
                reg_rank = graphix_reg_order.index(nm)
            except ValueError:
                reg_rank = len(graphix_reg_order)
            items.append((reg_rank, nm, ix, v))
        items.sort()  # (reg_rank, name, index)

        # (c) compose g -> v -> q
        g_to_q = [v_to_q[v] for (_, _, _, v) in items]
        return g_to_q  # length n, with unique ints in [0..n-1]

    # 3) Reorder amplitudes by exact bit-permutation defined by g_to_q
    def reorder_amps_graphix_to_qiskit(psi, g_to_q, graphix_big_endian=True):
        psi = np.asarray(psi, dtype=complex).reshape(-1)
        n = len(g_to_q)
        if psi.size != (1 << n):
            raise ValueError(f"psi length {psi.size} != 2**{n}")
        I = np.arange(1 << n, dtype=np.uint64)
        J = np.zeros_like(I)
        # Graphix bit position for qubit g:
        #   big-endian: MSB at g=0 → bitpos = n-1-g
        #   little-endian: bitpos = g
        for g, q in enumerate(g_to_q):
            g_bitpos = (n - 1 - g) if graphix_big_endian else g
            J |= ((I >> g_bitpos) & 1) << q
        return psi[J]

    # 4) Full pipeline: reorder, then filter on c0=0 at the correct Qiskit index
    def filtered_probs_c0_eq_0(psi, circuit, layout, graphix_big_endian=True, eps=0.0, top_k=None):
        n = circuit.num_qubits
        q_idx = qiskit_index_of_c0(layout, circuit, "c0", 0)
        g2q = g_to_q_map(layout, circuit)  # derive full wire permutation
        amps_q = reorder_amps_graphix_to_qiskit(psi, g2q, graphix_big_endian)

        # normalize
        norm = np.linalg.norm(amps_q)
        if norm == 0:
            raise ValueError("Zero statevector.")
        if not np.isclose(norm, 1.0):
            amps_q = amps_q / norm

        probs = Statevector(amps_q).probabilities_dict()  # keys little-endian, rightmost = q0

        # diagnostics (optional)
        mass0 = sum(p for b, p in probs.items() if b[-(q_idx + 1)] == '0')
        mass1 = 1.0 - mass0
        # print(f"P(c0=0)={mass0:.6f}, P(c0=1)={mass1:.6f}")

        # post-select c0=0
        kept = {b: p for b, p in probs.items() if b[-(q_idx + 1)] == '0'}
        Z = sum(kept.values())
        if Z == 0:
            raise ValueError("Conditioning on c0=0 leaves no support — check mapping.")
        kept = {b: (p / Z) for b, p in kept.items() if (p / Z) > eps}

        if top_k is not None:
            kept = dict(sorted(kept.items(), key=lambda kv: kv[1], reverse=True)[:top_k])

        # Graphix index of c0 for display only
        g_idx = (n - 1 - q_idx) if graphix_big_endian else q_idx
        return kept, q_idx, g_idx, (mass0, mass1)


    # psi: length 2**n complex array from Graphix (g0 is MSB / leftmost)
    GRAPHIX_BIG_ENDIAN = True  # set False if psi is already Qiskit-ordered
    EPS = 1e-12  # or 0.0 for no pruning
    TOP_K = None  # or an int

    filtered, q_idx, g_idx, (p0, p1) = filtered_probs_c0_eq_0(
        psi,
        transpiled_qc,
        transpiled_qc.layout,
        graphix_big_endian=GRAPHIX_BIG_ENDIAN,
        eps=EPS,
        top_k=TOP_K
    )

    print(f"Qiskit circuit index for c0[0]: {q_idx}  → Graphix index: {g_idx}")
    print(f"P(c0=0)={p0:.6f}, P(c0=1)={p1:.6f}")
    print(f"#outcomes post c0=0: {len(filtered)}")
    # Example: inspect top outcomes
    print(sorted(filtered.items(), key=lambda kv: kv[1], reverse=True)[:10])

    target_qubit = q_idx
    flip_plot_labels = True  # display-only MSB↔LSB flip
    user_feature = "demo"  # customize per run
    eps = 1e-12  # drop tiny probabilities post-selection
    top_k = None  # e.g., 40 to keep only the top-40 bars (None = keep all)

    # # --- statevector -> probabilities (nonzero only) ---
    # amps = np.asarray(psi, dtype=complex)
    # norm = np.linalg.norm(amps)
    # if not np.isclose(norm, 1.0):
    #     amps = amps / norm
    #
    # sv = Statevector(amps)
    # probs_dict = sv.probabilities_dict()  # keys: bitstrings; b[-1] is q0 (little-endian)
    #
    # # --- post-select c0 = 0 on target_qubit, renormalize, prune tiny mass ---
    # filtered = {b: p for b, p in probs_dict.items() if b[-(target_qubit + 1)] == '0'}
    # Z = sum(filtered.values())
    # if Z == 0:
    #     raise ValueError("After conditioning on c0=0, no support remains. Check target_qubit/endian.")
    # filtered = {b: (p / Z) for b, p in filtered.items() if (p / Z) > eps}
    #
    # # (optional) keep only the top-k outcomes
    # if top_k is not None:
    #     filtered = dict(sorted(filtered.items(), key=lambda kv: kv[1], reverse=True)[:top_k])

    # --- your plotting style ---
    xs = sorted(filtered)  # canonical keys
    ys = [filtered[b] for b in xs]
    tot = sum(ys)
    probs = [100.0 * y / tot for y in ys]  # percentage

    # DISPLAY-ONLY: flip label bit order if you want MSB↔LSB swapped
    labels = [b[::-1] for b in xs] if flip_plot_labels else xs

    os.makedirs("images/plots", exist_ok=True)
    plt.figure(figsize=(7, 3.6))
    plt.bar(range(len(xs)), probs)
    plt.xticks(range(len(xs)), labels, rotation=70, ha='right')  # readable labels
    plt.ylabel("Probability (%)")
    plt.title(f"Minimal QRS Results (post-selected c0=0) | feature: {user_vec}")
    plt.tight_layout()
    plt.savefig(f"images/plots/minimal_qrs_NOISE_CLIFFORD_statevector_evolution_plot_{user_vec}.png", dpi=300, bbox_inches="tight")
    plt.show()

    # from qiskit.quantum_info import Statevector
    # from qiskit.quantum_info import Statevector
    #
    # # amps is your 1D complex statevector (sv_logical_from_mbqc)
    # amps = np.asarray(sv_logical_from_mbqc, dtype=complex)
    # if not np.isclose(np.linalg.norm(amps), 1.0):
    #     amps = amps / np.linalg.norm(amps)
    #
    # sv = Statevector(amps)
    #
    # # Get only nonzero outcomes
    # probs = sv.probabilities_dict()  # keys are bitstrings, rightmost char = qubit 0 (little-endian)
    #
    # target_qubit = 0  # "c0" is qubit 5 (LSB indexing)
    # # Keep only outcomes with that bit = '0'
    # kept = {b: p for b, p in probs.items() if b[-(target_qubit + 1)] == '0'}
    #
    # Z = sum(kept.values())
    # if Z == 0:
    #     raise ValueError("After conditioning on c0=0 (bit-5=0), no support remains (all mass had c0=1).")
    # # Renormalize and drop tiny bars
    # eps = 1e-12
    # kept = {b: p / Z for b, p in kept.items() if p / Z > eps}
    #
    # # Plot
    # try:
    #     from qiskit.visualization import plot_distribution
    #     fig = plot_distribution(kept, title="|ψ⟩ probabilities conditioned on c0=0 (bit-5=0)")
    # except ImportError:
    #     from qiskit.visualization import plot_histogram
    #     fig = plot_histogram(kept, title="|ψ⟩ probabilities conditioned on c0=0 (bit-5=0)")
    # fig.show()

    return 0


    user_vec = [0, 0]
    qc_in, input_vec = circuits.minimal_qrs(user_vec)

    bw_pattern, col_map, transpiled_qc = brickwork_transpiler.transpile(
        qc_in, input_vec,
        routing_method="sabre",
        layout_method="sabre",
        with_ancillas=False
    )
    bw_pattern.standardize()
    bw_pattern.shift_signals()

    tn = bw_pattern.simulate_pattern(backend="tensornetwork", graph_prep="parallel")
    psi = tn.to_statevector()  # Graphix statevector on declared outputs

    dists = mbqc_conditional_plot(
        qc_in,  # your logical circuit
        transpiled_qc,  # the transpiled circuit returned by your brickwork transpiler
        psi,  # MBQC state from Graphix
        measure_logical=[2, 1],
        anc_reg_name="c0",  # the ancilla register name in minimal_qrs
        anc_pos=0,  # c0[0]
        anc_value=0,  # keep c0=0 (discard c0=1)
        input_vec_for_check=input_vec  # lets it auto-pick the correct mapping by reference
    )

    print("Picked:", dists["picked"])
    print("Virtual-wire:", dists["H1_virtual_wire"])
    print("Physical:", dists["H2_physical"])


    return 0
    # ---------- End-to-end usage ----------

    # 0) Build logical circuit and input state
    qc_in, input_vec = minimal_qrs(user_feature=[1, 1])


    bw_pattern, col_map, transpiled_qc = brickwork_transpiler.transpile(qc_in, input_vec,
                                                         routing_method="sabre",
                                                         layout_method="sabre",
                                                         with_ancillas=False)


    bw_pattern.standardize()  # puts commands into N-E-M-(X/Z/C) order
    bw_pattern.shift_signals()  # optional but recommended; reduces feedforward

    tn = bw_pattern.simulate_pattern(backend="tensornetwork", graph_prep="parallel")
    psi = tn.to_statevector()  # state on your declared outputs


    # 2) Choose which logical qubits to read out (in desired MSB→LSB order)
    measure_logical = [2, 1]
    anc_logical = ancilla_logical_index(qc_in, "c0", 0)

    # === PATH A: Qiskit simulation (state in *virtual wire* order) ===
    sv_virtual = Statevector(input_vec).evolve(psi)
    L2W = logical_to_outputwire_indices(qc_in, transpiled_qc)
    measure_idxs_virtual = [L2W[j] for j in measure_logical]
    anc_idx_virtual = L2W[anc_logical]

    dist_virtual = conditional_dist_from_state(
        sv_virtual,
        measure_idxs=measure_idxs_virtual,
        anc_idx=anc_idx_virtual,
        anc_value=0,  # keep c0=0, discard c0=1
    )
    plot_distribution(dist_virtual, title="Qiskit evolve: P(q2 q1 | c0=0)")

    # === PATH B: MBQC engine (state in *physical* order) ===
    # psi = tn.to_statevector()  # your flat numpy array from MBQC
    # sv_physical = Statevector(psi, dims=[2]*transpiled_qc.num_qubits)
    # L2P = logical_to_physical_indices(qc_in, transpiled_qc)
    # measure_idxs_physical = [L2P[j] for j in measure_logical]
    # anc_idx_physical = L2P[anc_logical]
    # dist_physical = conditional_dist_from_state(
    #     sv_physical,
    #     measure_idxs=measure_idxs_physical,
    #     anc_idx=anc_idx_physical,
    #     anc_value=0,
    # )
    # plot_distribution(dist_physical, title="MBQC (physical): P(q2 q1 | c0=0)")

    # # (Optional) sanity check: the two paths should match
    # # assert all(abs(dist_virtual[k] - dist_physical[k]) < 1e-10 for k in dist_virtual)

    return 0
    # 0) Build your logical circuit + input state
    # 0) Build logical circuit + input state
    # 0) Build logical circuit + input state
    qc, input_vec = minimal_qrs(user_feature=[0, 0])

    # 1) Transpile and cache the mapping, then evolve and undo layout
    bw_pattern, col_map, transpiled_qc = brickwork_transpiler.transpile(
        qc, input_vec, routing_method="sabre", layout_method="sabre", with_ancillas=False
    )

    # 1) Transpile and cache the mapping, then evolve and undo layout
    mapping = extract_logical_to_physical(qc, transpiled_qc)

    sv_phys = Statevector(input_vec).evolve(transpiled_qc)
    sv_logical = undo_layout_on_state(sv_phys, mapping)

    # 2) Find the ancilla's logical index and define the measured qubits
    # anc_logical_idx = qc.qubits.index([q for q in qc.qubits if getattr(q.register, "name", "") == "c0"][0])
    # measure_qargs = [2, 1]  # measure q2 (MSB), q1 (LSB)

    anc_logical_idx = ancilla_logical_index(qc, "c0", 0)
    dist = postselect_ancilla_and_marginalize(
        sv_logical, anc_idx=anc_logical_idx, measure_qargs=[2, 1], anc_value=0
    )

    plot_distribution(dist, title="P(q2 q1 | c0=0) in logical order")

    return 0

    user_vec = [0, 0]
    qc, input_vec = circuits.minimal_qrs(user_vec)


    bw_pattern, col_map, transpiled_qc = brickwork_transpiler.transpile(qc, input_vec,
                                                         routing_method="sabre",
                                                         layout_method="sabre",
                                                         with_ancillas=False)

    visualiser.plot_brickwork_graph_from_pattern(bw_pattern,
                                                 node_colours=col_map,
                                                 use_node_colours=True,
                                                 title="test_cx_from_zero_upper",
                                                 )

    # Optimise for tensornetwork backand and simulation efficiency
    bw_pattern.standardize()  # puts commands into N-E-M-(X/Z/C) order
    bw_pattern.shift_signals()  # optional but recommended; reduces feedforward

    tn = bw_pattern.simulate_pattern(backend="tensornetwork", graph_prep="parallel")
    psi = tn.to_statevector()  # state on your declared outputs


    mapping = extract_logical_to_physical(qc, transpiled_qc)

    # sv_phys = Statevector(input_vec).evolve(transpiled_qc)
    # undo_layout_on_state(sv_phys, mapping)

    # If you simulated with your MBQC engine and got a flat numpy array `psi`:
    sv_logical_from_mbqc = undo_layout_on_state(psi, mapping, total_qubits=transpiled_qc.num_qubits)


    # user_vector = [0, 0]
    # qc, input_vec = circuits.minimal_qrs(user_vector)
    #
    # # Transpile!
    # bw_pattern, col_map, transpiled_qc = brickwork_transpiler.transpile(qc, input_vec,
    #                                                      routing_method="sabre",
    #                                                      layout_method="sabre",
    #                                                      with_ancillas=False)
    # # ref_state = calculate_ref_state_from_qiskit_circuit(bw_pattern, qc, input_vec)
    # visualiser.plot_brickwork_graph_from_pattern(bw_pattern,
    #                                              node_colours=col_map,
    #                                              use_node_colours=True,
    #                                              title="Brickwork_Graph_test_system_shift")
    #
    #
    # print("Standardize Pattern")
    # bw_pattern.standardize()  # puts commands into N-E-M-(X/Z/C) order
    # bw_pattern.shift_signals()  # optional but recommended; reduces feedforward
    #
    # print("Simulating Pattern")
    # tn = bw_pattern.simulate_pattern(backend="tensornetwork", graph_prep="parallel")
    # graphix_vec = tn.to_statevector()  # state on your declared outputs
    #
    # mapping = extract_logical_to_physical(qc, transpiled_qc)
    user_vector = [0, 0]
    qc, input_vec = circuits.minimal_qrs(user_vector)

    bw_pattern, col_map, transpiled_qc = brickwork_transpiler.transpile(
        qc, input_vec, routing_method="sabre", layout_method="sabre", with_ancillas=False
    )
    bw_pattern.standardize();
    bw_pattern.shift_signals()
    tn = bw_pattern.simulate_pattern(backend="tensornetwork", graph_prep="parallel")
    graphix_vec = tn.to_statevector()

    cond, mass, anc, targets, path = mqrs.plot_conditional_auto(
        qc, transpiled_qc, graphix_vec,
        save_path="images/plots/minimal_qrs.png",
        show=True
    )
    print(cond, mass, anc, targets, path)

    return 0

    user_vector = [0, 0]
    qc, input_vec = circuits.minimal_qrs(user_vector)

    # Transpile!
    bw_pattern, col_map, transpiled_qc = brickwork_transpiler.transpile(qc, input_vec,
                                                         routing_method="sabre",
                                                         layout_method="sabre",
                                                         with_ancillas=False)
    # ref_state = calculate_ref_state_from_qiskit_circuit(bw_pattern, qc, input_vec)
    visualiser.plot_brickwork_graph_from_pattern(bw_pattern,
                                                 node_colours=col_map,
                                                 use_node_colours=True,
                                                 title="Brickwork_Graph_test_system_shift")


    print("Standardize Pattern")
    bw_pattern.standardize()  # puts commands into N-E-M-(X/Z/C) order
    bw_pattern.shift_signals()  # optional but recommended; reduces feedforward

    print("Simulating Pattern")
    tn = bw_pattern.simulate_pattern(backend="tensornetwork", graph_prep="parallel")
    graphix_vec = tn.to_statevector()  # state on your declared outputs


    def _reverse_bits_int(x, n):
        y = 0
        for i in range(n):
            y |= ((x >> i) & 1) << (n - 1 - i)
        return y

    def _as_perm_list(mapping, n):
        """Accepts list/tuple or dict {logical:int physical}. Returns list p where p[j]=physical."""
        if mapping is None:
            return None
        if isinstance(mapping, dict):
            return [int(mapping[j]) for j in range(n)]
        return [int(x) for x in mapping]

    def to_logical_state(amps, n, *, mapping=None, input_is_qiskit_order=True):
        """
        Return a 1D complex array in LOGICAL Qiskit order (q0=LSB) after:
          (1) optional big-endian -> little-endian fix, then
          (2) undoing layout using mapping (logical->physical).
        """
        data = np.asarray(getattr(amps, "data", amps), dtype=complex).reshape(-1)
        assert data.size == (1 << n), "Length must be 2**n."

        # normalize
        norm = float((data.conj() * data).real.sum())
        if not np.isclose(norm, 1.0, atol=1e-12):
            data = data / np.sqrt(norm)

        # Align indexing to Qiskit little-endian on *physical* qubits
        if not input_is_qiskit_order:  # Graphix/other big-endian
            idx = np.fromiter((_reverse_bits_int(i, n) for i in range(1 << n)),
                              dtype=np.int64, count=(1 << n))
            data = data[idx]

        # Undo layout: mapping is logical->physical
        p = _as_perm_list(mapping, n)
        if p is None:
            return data

        # Build logical-order vector: for each logical index x, pick amplitude at physical index y
        out = np.empty_like(data)
        for x in range(1 << n):
            y = 0
            for j in range(n):
                y |= ((x >> j) & 1) << p[j]
            out[x] = data[y]
        return out

    def dump_full_distribution(amps, n, *, mapping=None, input_is_qiskit_order=True, sort=True):
        """
        Print all basis states in LOGICAL order with q0 shown on the *left* of the string.
        Returns (data_logical, probs_dict).
        """
        data_log = to_logical_state(amps, n, mapping=mapping, input_is_qiskit_order=input_is_qiskit_order)
        probs = (data_log.conj() * data_log).real
        rows = []
        for x, p in enumerate(probs):
            bits_q0_left = format(x, f"0{n}b")[::-1]  # q0 ... q{n-1} left→right
            rows.append((bits_q0_left, float(p), data_log[x]))
        if sort:
            rows.sort(key=lambda t: t[1], reverse=True)
        for b, p, a in rows:
            print(f"{b}  prob={p:.16f}  amp={a}")
        return data_log, {b: p for b, p, _ in rows}

    def conditional_targets_given_anc0(data_log, n, targets=(0, 1), ancilla=5):
        """P(targets | ancilla=0) from a LOGICAL-order statevector."""
        p = (data_log.conj() * data_log).real
        idx = np.arange(1 << n, dtype=np.uint64)
        keep = ((idx >> ancilla) & 1) == 0
        mass = float(p[keep].sum())
        if mass == 0.0:
            raise ValueError("Pr[ancilla=0] = 0.")
        bs = [((idx >> t) & 1)[keep] for t in targets]
        w = p[keep]

        def s(b):
            sel = np.ones_like(w, dtype=bool)
            for arr, bit in zip(bs, b): sel &= (arr == bit)
            return float(w[sel].sum()) / mass

        labels = ["00", "01", "10", "11"]
        return {lab: s(tuple(int(c) for c in lab)) for lab in labels}, mass

    mapping = extract_logical_to_physical(qc, transpiled_qc)
    data_log, full = dump_full_distribution(
        amps=graphix_vec, n=transpiled_qc.num_qubits,
        mapping=mapping, input_is_qiskit_order=True, sort=True
    )

    # 2) (Optional) Inspect ancilla and your two target qubits
    targets = (2, 1)  # or (1, 2) if that's what you intended earlier
    anc = 0
    cond, mass = conditional_targets_given_anc0(data_log, n=transpiled_qc.num_qubits,
                                                targets=targets, ancilla=anc)
    print(f"P(ancilla@q{anc}=0) = {mass:.6f};  conditional P{targets}|anc=0 = {cond}")

    # mapping: logical -> physical from extract_logical_to_physical(qc, transpiled_qc)
    probs, counts = postselect_marginal_on_ancilla_zero(
        amps=graphix_vec,  # numpy array (or Statevector)
        n=transpiled_qc.num_qubits,
        targets_logical=(2, 1),  # display order: (q2, q1)
        ancilla_logical=0,  # ancilla is the TOP logical qubit
        shots=20_000,
        mapping=mapping,  # undo physical->logical permutation
        input_is_qiskit_order=True  # set False if your sim is big-endian
    )

    # after: data_log, n = to_logical_state(...), transpiled_qc.num_qubits

    # Case A: ancilla is bottom (q5)  → what you ran; deterministic 00
    cond, mass = conditional_targets_given_anc0(data_log, transpiled_qc.num_qubits, targets=(0, 1), ancilla=5)
    print(f"ancilla=q5  mass={mass:.3f}  cond={cond}")

    # Case B: ancilla is top (q0)  → gives 2/3 vs 1/3
    cond, mass = conditional_targets_given_anc0(data_log, transpiled_qc.num_qubits, targets=(1, 2), ancilla=0)
    print(f"ancilla=q0  mass={mass:.3f}  cond={cond}")

    # If you want the 1/3 labelled as '01', swap targets:
    cond_swapped, _ = conditional_targets_given_anc0(data_log, transpiled_qc.num_qubits, targets=(2, 1), ancilla=0)
    print(f"ancilla=q0  targets=(2,1)  cond={cond_swapped}")

    title = f"Minimal QRS `Experiment | user vector: {user_vector}"
    labels = ["00","01","10","11"]
    heights = [cond.get(l, 0.0) for l in labels]
    fig, ax = plt.subplots()
    ax.bar(labels, heights)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Probability")
    if title: ax.set_title(title)
    for x, h in enumerate(heights):
        ax.text(x, h + 0.02, f"{h:.3f}", ha="center", va="bottom")

    fig.savefig(f"images/plots/thesis_minimal_qrs_uv_{user_vector}", dpi=200, bbox_inches="tight")
    plt.show()

    print(counts)
    print(probs)

    return 0

    # minimaLqrs.build_graph()
    # minimaLqrs.run_and_plot_minimal_qrs_only_db()

    # input_vec = Statevector.from_label('+++')  # three-qubit plus state

    # 2) Define your 2-qubit circuit (no H gates needed)
    qc, input_vec = circuits.minimal_qrs([0, 0])

    # Transpile!
    bw_pattern, col_map, transpiled_qc = brickwork_transpiler.transpile(qc, input_vec,
                                                         routing_method="sabre",
                                                         layout_method="sabre",
                                                         with_ancillas=False)
    # ref_state = calculate_ref_state_from_qiskit_circuit(bw_pattern, qc, input_vec)
    visualiser.plot_brickwork_graph_from_pattern(bw_pattern,
                                                 node_colours=col_map,
                                                 use_node_colours=True,
                                                 title="Brickwork_Graph_test_system_shift")

    # Simulate the generated pattern

    # outstate = bw_pattern.simulate_pattern(backend='tensornetwork')
    # outvec = np.asarray(outstate).ravel()

    print("Standardize Pattern")
    bw_pattern.standardize()  # puts commands into N-E-M-(X/Z/C) order
    bw_pattern.shift_signals()  # optional but recommended; reduces feedforward
    # (optional) aggressively prune:
    # bw_pattern.perform_pauli_measurements()

    print("Simulating Pattern")
    tn = bw_pattern.simulate_pattern(backend="tensornetwork", graph_prep="parallel")
    graphix_vec = tn.to_statevector()  # state on your declared outputs


    # --- core helpers -------------------------------------------------------------

    def _reverse_bits_int(x, n):
        y = 0
        for i in range(n):
            y |= ((x >> i) & 1) << (n - 1 - i)
        return y

    def _as_perm_list(mapping, n):
        """Accepts list/tuple or dict {logical:int physical}. Returns list p where p[j]=physical."""
        if mapping is None:
            return None
        if isinstance(mapping, dict):
            return [int(mapping[j]) for j in range(n)]
        return [int(x) for x in mapping]

    def to_logical_state(amps, n, *, mapping=None, input_is_qiskit_order=True):
        """
        Return a 1D complex array in LOGICAL Qiskit order (q0=LSB) after:
          (1) optional big-endian -> little-endian fix, then
          (2) undoing layout using mapping (logical->physical).
        """
        data = np.asarray(getattr(amps, "data", amps), dtype=complex).reshape(-1)
        assert data.size == (1 << n), "Length must be 2**n."

        # normalize
        norm = float((data.conj() * data).real.sum())
        if not np.isclose(norm, 1.0, atol=1e-12):
            data = data / np.sqrt(norm)

        # Align indexing to Qiskit little-endian on *physical* qubits
        if not input_is_qiskit_order:  # Graphix/other big-endian
            idx = np.fromiter((_reverse_bits_int(i, n) for i in range(1 << n)),
                              dtype=np.int64, count=(1 << n))
            data = data[idx]

        # Undo layout: mapping is logical->physical
        p = _as_perm_list(mapping, n)
        if p is None:
            return data

        # Build logical-order vector: for each logical index x, pick amplitude at physical index y
        out = np.empty_like(data)
        for x in range(1 << n):
            y = 0
            for j in range(n):
                y |= ((x >> j) & 1) << p[j]
            out[x] = data[y]
        return out

    def dump_full_distribution(amps, n, *, mapping=None, input_is_qiskit_order=True, sort=True):
        """
        Print all basis states in LOGICAL order with q0 shown on the *left* of the string.
        Returns (data_logical, probs_dict).
        """
        data_log = to_logical_state(amps, n, mapping=mapping, input_is_qiskit_order=input_is_qiskit_order)
        probs = (data_log.conj() * data_log).real
        rows = []
        for x, p in enumerate(probs):
            bits_q0_left = format(x, f"0{n}b")[::-1]  # q0 ... q{n-1} left→right
            rows.append((bits_q0_left, float(p), data_log[x]))
        if sort:
            rows.sort(key=lambda t: t[1], reverse=True)
        for b, p, a in rows:
            print(f"{b}  prob={p:.16f}  amp={a}")
        return data_log, {b: p for b, p, _ in rows}

    def conditional_targets_given_anc0(data_log, n, targets=(0, 1), ancilla=5):
        """P(targets | ancilla=0) from a LOGICAL-order statevector."""
        p = (data_log.conj() * data_log).real
        idx = np.arange(1 << n, dtype=np.uint64)
        keep = ((idx >> ancilla) & 1) == 0
        mass = float(p[keep].sum())
        if mass == 0.0:
            raise ValueError("Pr[ancilla=0] = 0.")
        bs = [((idx >> t) & 1)[keep] for t in targets]
        w = p[keep]

        def s(b):
            sel = np.ones_like(w, dtype=bool)
            for arr, bit in zip(bs, b): sel &= (arr == bit)
            return float(w[sel].sum()) / mass

        labels = ["00", "01", "10", "11"]
        return {lab: s(tuple(int(c) for c in lab)) for lab in labels}, mass

    # --- usage --------------------------------------------------------------------
    # n = transpiled_qc.num_qubits
    # mapping = extract_logical_to_physical(qc, transpiled_qc)  # logical -> physical
    # graphix_vec = tn.to_statevector()

    # 1) Dump the complete logical-basis table (after undoing layout)
    #    Set input_is_qiskit_order=False if your simulator is big-endian.
    #    (Try both True/False once to see which matches your expectations.)
    mapping = extract_logical_to_physical(qc, transpiled_qc)
    data_log, full = dump_full_distribution(
        amps=graphix_vec, n=transpiled_qc.num_qubits,
        mapping=mapping, input_is_qiskit_order=True, sort=True
    )

    # 2) (Optional) Inspect ancilla and your two target qubits
    targets = (0, 1)  # or (1, 2) if that's what you intended earlier
    anc = 5
    cond, mass = conditional_targets_given_anc0(data_log, n=transpiled_qc.num_qubits,
                                                targets=targets, ancilla=anc)
    print(f"P(ancilla@q{anc}=0) = {mass:.6f};  conditional P{targets}|anc=0 = {cond}")

    # from qiskit.quantum_info import Statevector
    #
    # # Suppose Graphix gave you a 1D array-like of complex amplitudes:
    # # graphix_vec: shape (2**n,)  e.g. length 64 for n=6
    # data = np.asarray(graphix_vec, dtype=complex).reshape(-1)
    # n = int(round(np.log2(data.size)))
    # assert 2 ** n == data.size, "Length must be a power of two."
    #
    # # (Optional) normalize defensively
    # norm = np.vdot(data, data).real
    # if not np.isclose(norm, 1.0, atol=1e-12):
    #     data = data / np.sqrt(norm)
    #
    # psi = Statevector(data, dims=(2,) * n)  # now it's a proper Qiskit Statevector
    #
    #
    # shots = 20_000
    # counts = psi.sample_counts(shots=shots)  # dict: bitstring -> frequency
    # memory = psi.sample_memory(shots=shots)  # list of per-shot bitstrings
    # probs = psi.probabilities_dict()  # exact Born probabilities

    # mapping: logical -> physical from extract_logical_to_physical(qc, transpiled_qc)
    probs, counts = postselect_marginal_on_ancilla_zero(
        amps=graphix_vec,  # numpy array (or Statevector)
        n=transpiled_qc.num_qubits,
        targets_logical=(2, 1),  # display order: (q2, q1)
        ancilla_logical=0,  # ancilla is the TOP logical qubit
        shots=20_000,
        mapping=mapping,  # undo physical->logical permutation
        input_is_qiskit_order=True  # set False if your sim is big-endian
    )


    # after: data_log, n = to_logical_state(...), transpiled_qc.num_qubits

    # Case A: ancilla is bottom (q5)  → what you ran; deterministic 00
    cond, mass = conditional_targets_given_anc0(data_log, transpiled_qc.num_qubits, targets=(0, 1), ancilla=5)
    print(f"ancilla=q5  mass={mass:.3f}  cond={cond}")

    # Case B: ancilla is top (q0)  → gives 2/3 vs 1/3
    cond, mass = conditional_targets_given_anc0(data_log, transpiled_qc.num_qubits, targets=(1, 2), ancilla=0)
    print(f"ancilla=q0  mass={mass:.3f}  cond={cond}")

    # If you want the 1/3 labelled as '01', swap targets:
    cond_swapped, _ = conditional_targets_given_anc0(data_log, transpiled_qc.num_qubits, targets=(2, 1), ancilla=0)
    print(f"ancilla=q0  targets=(2,1)  cond={cond_swapped}")

    print(counts)
    print(probs)

    return 0

    print("Permuting output state")
    # ref_state = calculate_ref_state_from_qiskit_circuit(bw_pattern, qc, input_vec)

    print("Evolving refrence state")
    # ref_state = input_vec.evolve(transpiled_qc)
    mapping = extract_logical_to_physical(qc, transpiled_qc)

    # If you simulated with Qiskit:
    sv_phys = Statevector(input_vec).evolve(transpiled_qc)
    undo_layout_on_state(sv_phys, mapping)

    # If you simulated with your MBQC engine and got a flat numpy array `psi`:
    sv_logical_from_mbqc = undo_layout_on_state(psi, mapping, total_qubits=transpiled_qc.num_qubits)
    # compare amplitudes (up to global phase)
    # print("ref_state = ", sv_logical_from_mbqc)
    print("Graphix output rerouted = ", sv_logical_from_mbqc)

    # Compare output up to global phase
    assert utils.assert_equal_up_to_global_phase(sv_logical_from_mbqc.data, psi)


    # experiment = circuits.cx_and_h_circ()
    #
    # experiment.draw(output='mpl',
    #                 fold=30,
    #                 style="iqp"
    #                    )
    # plt.savefig(f"images/Circuits/minimal_recommendation_circuit.png", dpi=300, bbox_inches="tight")
    # plt.show()
    #
    #
    # bw_pattern, col_map = brickwork_transpiler.transpile(experiment, routing_method='sabre', layout_method='trivial',
    #                                                      with_ancillas=False)
    #
    # visualiser.plot_brickwork_graph_from_pattern(bw_pattern,
    #                                              show_angles=True,
    #                                              node_colours=col_map,
    #                                              use_node_colours=True,
    #                                              title="Brickwork graph: Minimal QRS")
    #
    # file_path = "src/brickwork_transpiler/experiments/data/output_data/"
    #
    # pattern_writer = utils.BufferedCSVWriter(file_path + "minimal_qrs_experiment_pattern.txt", ["pattern"])
    # log_writer = utils.BufferedCSVWriter(file_path + "minimal_qrs_experiment_log.txt", ["log"])
    #
    # encoded_pattern, log_alice = encode_pattern(bw_pattern)
    # pattern_writer.set("pattern", encoded_pattern.print_pattern(lim=2**32))
    # log_writer.set("log", log_alice)
    #
    # pattern_writer.flush()
    # log_writer.flush()
    #
    # visualiser.plot_brickwork_graph_locked(encoded_pattern, use_locks=False, title="Brickwork Graph: Minimal QRS Encoded",
    #                                        show_angles=True)


    # Plot HHL
    # hhl_circ = hhl.generate_example_hhl_QC()
    # qc, _ = circuits.h_and_cx_circ()
    #
    # print(qc)
    # qc.draw(output='mpl',
    #                    fold=30,
    #                    )
    # plt.show()
    # # | Layout Method    | Strategy                                               |
    # # | ---------------- | ------------------------------------------------------ |
    # # | `trivial`        | 1-to-1 mapping                                         |
    # # | `dense`          | Densest‐subgraph heuristic                             |
    # # | `noise_adaptive` | Minimize error rates (readout & 2-qubit)               |
    # # | `sabre`          | Sabre seed + forwards/backwards swap refinement        |
    # # | `default`        | VF2 perfect + Sabre fallback (or `trivial` at level 0) |
    # # Routing: 'stochastic', #'sabre', #'lookahead', #'basic',
    #
    # print("Transpiling HHL circuit...")
    # bw_pattern, col_map = brickwork_transpiler.transpile(qc, routing_method='sabre', layout_method='trivial')
    #
    # visualiser.plot_brickwork_graph_from_pattern(bw_pattern,
    #                                              show_angles=True,
    #                                              node_colours=col_map,
    #                                              use_node_colours=True,
    #                                              title="Brickwork graph: H + CX")
    #
    # return 0

    #Test HHL

    from qiskit import transpiler

    # hhl = generate_example_hhl_QC()
    # # Decompose as much as possible (optional)
    # hhl = hhl.decompose()  # You may skip this; see below
    #
    # # Transpile to elementary gates (adjust basis_gates as needed)
    # basis_gates = ['cx', 'u3', 'u2', 'u1', 'rz', 'ry', 'rx', 'id', 'x', 'y', 'z', 'h', 's', 'sdg', 't', 'tdg', 'swap',
    #                'measure', "mcx", "cp", "cz"]
    # hhl_basis = transpile(hhl, basis_gates=basis_gates, optimization_level=0)
    #
    # # Now print the gate types
    # print("HHL gate counts:", hhl_basis.count_ops())
    # print("Total gates:", sum(hhl_basis.count_ops().values()))
    # import qiskit.circuit.library.standard_gates as sg
    # print(dir(sg))

    # qrs_no_db_transpilation.experiment_qrs_no_db_no_matching_element()
    # qrs_no_db_transpilation.experiment_qrs_no_db_subset_grover()
    # qrs_no_db_transpilation.experiment_qrs_no_db_one_match_duplicates()
    # qrs_no_db_transpilation.experiment_qrs_no_db_one_matching_element()
    # print("Full -- one match")
    # qrs_full_transpilation.experiment_qrs_full_one_matching_element()
    # print("Full -- no match")
    # qrs_full_transpilation.experiment_qrs_full_no_matching_element()
    # print("Full -- subset match")
    # qrs_full_transpilation.experiment_qrs_full_subset_grover()
    # print("Full -- one match duplicates")
    # qrs_full_transpilation.experiment_qrs_full_one_match_duplicates()

    # qrs_no_db_transpilation.experiment_qrs_no_db_one_matching_element()
    # qrs_no_db_transpilation.experiment_qrs_no_db_no_matching_element()
    # qrs_no_db_transpilation.experiment_qrs_no_db_subset_grover()
    # qrs_no_db_transpilation.experiment_qrs_no_db_one_match_duplicates()

    # plot_qrs_full.plot_grover_database_scaling()

    # qft_transpilation.experiment_qft_transpilation()
    # hhl_transpilation.experiment_hhl_transpilation(6)

    # plot_qrs_data.plot_qrs_with_db_scaling_from_files(name_of_plot="thesis_qrs_plot_test")


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

    # return 0

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


    return 0


    # QRS_CHECKED
    # ───────────────────────────────────────────────── DATA ───────────────────────────────────────────────
    feature_mat = [
        [1, 0, 1, 1, 1, 1],
        [0, 1, 0, 0, 0, 1],
        [0, 0, 1, 1, 1, 0],
        [0, 0, 0, 1, 0, 1],
        [1, 0, 0, 0, 1, 0],
        [1, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1],
        [1, 1, 0, 0, 1, 0],
    ]
    user_feature = [0, 1, 0, 0, 0, 1]
    feature_subset = [[0, 1, 0, 0, 0, 1]]  # oracle pattern
    grover_iterations = None
    g = 1

    n_items = len(feature_mat)
    num_id_qubits = int(np.log2(n_items))
    num_db_feature_qubits = len(feature_mat[0])
    feature_qubits = list(range(num_id_qubits,
                                num_id_qubits + num_db_feature_qubits))

    # ───────────────────────────────────────── BUILD QRS+GROVER CIRCUIT ────────────────────────────────────
    qrs_circ = qrs_knn_grover.qrs(
        n_items=n_items,
        feature_mat=feature_mat,
        user_vector=user_feature,
        feature_subset=feature_subset,
        g=g,
        plot_circ=False,
        plot_histogram=False,
        grover_iterations=grover_iterations,
        file_writer=None,
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
    plt.savefig(f"images/qrs/new_recommendation_circ{grover_iterations}.png", dpi=300, bbox_inches="tight")
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
    plt.savefig("images/plots/new_grover_plot.png", dpi=300)

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
    bw_pattern, col_map = brickwork_transpiler.transpile(qc, routing_method='sabre', layout_method='trivial')
    bw_pattern.print_pattern(lim=200)

    encoded_pattern, log_alice = encode_pattern(bw_pattern)
    encoded_pattern.print_pattern(lim=200)
    print(log_alice)

    encoded_pattern2, log_alice2 = encode_pattern(bw_pattern, remove_dependencies=False)
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
