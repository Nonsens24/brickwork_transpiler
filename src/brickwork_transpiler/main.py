import numpy as np
# import qiskit.compiler.transpiler
from matplotlib import pyplot as plt
from qiskit.quantum_info import Statevector

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
from src.brickwork_transpiler.noise import DepolarisingInjector
# from src.brickwork_transpiler.noise import to_noisy_pattern
import src.brickwork_transpiler.circuits as circuits

from qiskit import transpile, ClassicalRegister, QuantumRegister
from qiskit_aer import AerSimulator
from qiskit import QuantumCircuit
import src.brickwork_transpiler.experiments.minimal_qrs_transpilation_simulation as mqrs
from src.brickwork_transpiler.utils import calculate_ref_state_from_qiskit_circuit, extract_logical_to_physical, \
    undo_layout_on_state

import numpy as np
from qiskit.quantum_info import Statevector

# --- core helpers ---

# def permute_statevector_qubits(vec, p_old_to_new):
#     """
#     vec: complex array of length 2^m (little-endian; q0 = LSB).
#     p_old_to_new: list of length m; maps old bit position -> new bit position.
#     """
#     m = len(p_old_to_new)
#     out = np.empty_like(vec)
#     for i in range(1 << m):
#         j = 0
#         for old, new in enumerate(p_old_to_new):
#             j |= ((i >> old) & 1) << new
#         out[j] = vec[i]
#     return out
#
# def graphix_to_qiskit_perm_from_pattern(bw_pattern):
#     """
#     Graphix reports outputs in big-endian; Qiskit uses little-endian.
#     This returns p[g] = q meaning: move Graphix bit g -> Qiskit logical bit q.
#     """
#     return [t[0] for t in bw_pattern.output_nodes][::-1]
#
# def embed_state_into_n_qubits(vec_k, target_positions, n_total):
#     """
#     Embed a k-qubit state vec_k into an n-qubit register where the k logical
#     qubits occupy 'target_positions' (list of length k, each in [0..n_total-1]),
#     and all other qubits are |0>. Little-endian convention throughout.
#     """
#     k = len(target_positions)
#     out = np.zeros(1 << n_total, dtype=complex)
#     for i in range(1 << k):
#         J = 0
#         for g, pos in enumerate(target_positions):
#             J |= ((i >> g) & 1) << pos
#         out[J] = vec_k[i]
#     return out

#
# import numpy as np
#
# def postselect_marginal_on_ancilla_zero(
#     amps,
#     n,
#     targets_logical=(1, 2),
#     ancilla_logical=None,
#     shots=None,
#     seed=1,
#     mapping=None,                # logical -> physical; if given, undo to logical order
#     input_is_qiskit_order=True,  # set False if your input uses big-endian (q0 = MSB)
# ):
#     """
#     Returns:
#       probs:  dict over {'00','01','10','11'} for (q_targets in given order) | ancilla=0
#       counts: dict if shots is not None
#     Notes:
#       - Indices are in LOGICAL numbering after undo.
#       - Qiskit little-endian convention (q0 = LSB) is used internally.
#     """
#     data = np.asarray(getattr(amps, "data", amps), dtype=complex).reshape(-1)
#     assert data.size == (1 << n), "Length must be 2**n."
#
#     # Normalize
#     norm = float((data.conj() * data).real.sum())
#     if not np.isclose(norm, 1.0, atol=1e-12):
#         data = data / np.sqrt(norm)
#
#     # Work as an n-axis tensor; apply endian fix then undo layout to LOGICAL order
#     tens = data.reshape([2] * n)
#
#     if not input_is_qiskit_order:                 # big-endian -> little-endian
#         tens = np.transpose(tens, axes=list(range(n))[::-1])
#
#     if mapping is not None:                       # physical -> logical (undo)
#         # mapping: logical -> physical; make new axis j = old axis mapping[j]
#         axes = [mapping[j] for j in range(n)]
#         tens = np.transpose(tens, axes=axes)
#
#     data_log = tens.reshape(-1)                   # now in logical Qiskit order
#
#     # Indices in logical order
#     a = (n - 1) if (ancilla_logical is None) else int(ancilla_logical)
#     t1, t2 = map(int, targets_logical)
#
#     # Vectorized accumulation for P(t1,t2 | ancilla=0)
#     p = (data_log.conj() * data_log).real
#     idx = np.arange(p.size, dtype=np.uint64)
#     keep = ((idx >> a) & 1) == 0
#     if not np.any(keep):
#         raise ValueError("Post-selection event has zero probability (ancilla always 1).")
#
#     b1 = ((idx >> t1) & 1)[keep]
#     b2 = ((idx >> t2) & 1)[keep]
#     w  = p[keep]
#     mass = float(w.sum())
#
#     def s(bit1, bit2):
#         return float(w[(b1 == bit1) & (b2 == bit2)].sum()) / mass
#
#     probs = {"00": s(0,0), "01": s(0,1), "10": s(1,0), "11": s(1,1)}
#
#     counts = None
#     if shots is not None:
#         rng = np.random.default_rng(seed)
#         labels = ["00", "01", "10", "11"]
#         weights = np.array([probs[l] for l in labels], dtype=float)
#         counts = dict(zip(labels, rng.multinomial(int(shots), weights)))
#
#     return probs, counts
#
#
# from qiskit.quantum_info import Statevector
# import numpy as np
# import matplotlib.pyplot as plt
#
# from qiskit.quantum_info import Statevector
# import numpy as np
# import matplotlib.pyplot as plt
#
# def postselect_ancilla_and_marginalize(sv_logical: Statevector,
#                                        anc_idx: int,
#                                        measure_qargs: list[int],
#                                        anc_value: int = 0):
#     """
#     Version that does NOT use Statevector.project (works on older Qiskit).
#     Returns P(measure_qargs | anc_idx = anc_value) as a dict.
#     The first element of `measure_qargs` is the MSB of the returned keys.
#     """
#     # Build joint over [measure_qargs..., anc_idx] so the ancilla is the last bit in each key
#     qargs_joint = list(measure_qargs) + [anc_idx]
#     joint = sv_logical.probabilities_dict(qargs=qargs_joint)
#
#     # Numerator: collect all outcomes where ancilla bit == anc_value
#     num = {}
#     denom = 0.0
#     anc_char = str(int(anc_value))
#     for key, p in joint.items():
#         if key[-1] == anc_char:          # last char corresponds to ancilla because we appended it last
#             meas_key = key[:-1]          # drop ancilla bit
#             num[meas_key] = num.get(meas_key, 0.0) + p
#             denom += p
#
#     if np.isclose(denom, 0.0):
#         # No support on the requested ancilla outcome
#         return {format(i, f'0{len(measure_qargs)}b'): 0.0 for i in range(2**len(measure_qargs))}
#
#     # Normalize to get the conditional
#     out = {k: v/denom for k, v in num.items()}
#
#     # Stable ordering of keys
#     ordered = sorted(out.keys())
#     return {k: out[k] for k in ordered}
#
# def plot_distribution(dist: dict, title: str = "Conditional distribution"):
#     labels = list(dist.keys())
#     values = [dist[k] for k in labels]
#     plt.figure()
#     plt.bar(labels, values)
#     plt.xlabel("Measured bitstring (order = measure_qargs)")
#     plt.ylabel("Probability")
#     plt.title(title)
#     plt.tight_layout()
#     plt.show()
#
# def ancilla_logical_index(qc, reg_name="c0", pos=0):
#     """
#     Return the integer index (in qc.qubits order) of bit `pos` in register named `reg_name`.
#     Uses QuantumCircuit.find_bit when available; falls back to list index otherwise.
#     """
#     try:
#         qr = next(r for r in qc.qregs if r.name == reg_name)
#     except StopIteration:
#         raise ValueError(f"No quantum register named {reg_name!r} in circuit.")
#     qbit = qr[pos]
#     try:
#         return qc.find_bit(qbit).index  # preferred, no deprecation
#     except AttributeError:
#         return qc.qubits.index(qbit)    # older Terra fallback
#
#
# from qiskit import transpile
# from qiskit.quantum_info import Statevector
# import numpy as np
# import matplotlib.pyplot as plt
#
# # ---------- Robust helpers ----------
#
# def ancilla_logical_index(qc, reg_name="c0", pos=0):
#     """Index (in qc.qubits order) of bit `pos` in register `reg_name`."""
#     try:
#         qr = next(r for r in qc.qregs if r.name == reg_name)
#     except StopIteration:
#         raise ValueError(f"No quantum register named {reg_name!r} in circuit.")
#     qbit = qr[pos]
#     try:
#         return qc.find_bit(qbit).index  # preferred (no deprecation)
#     except AttributeError:
#         return qc.qubits.index(qbit)    # fallback for very old Terra
#
# def logical_to_outputwire_indices(qc_in, qc_out):
#     """
#     L2W[j] = index in qc_out.qubits (virtual wire order) where logical j (qc_in.qubits[j]) ended up.
#     Works across Terra versions.
#     """
#     out = []
#     for q in qc_in.qubits:
#         try:
#             out.append(qc_out.find_bit(q).index)
#         except Exception:
#             # Fallback: identity if transpile preserved bit objects in order
#             out.append(qc_out.qubits.index(q))
#     return out
#
# def logical_to_physical_indices(qc_in, qc_out):
#     """
#     L2P[j] = final *physical* index for logical j.
#     Use when your state is in physical-qubit order (e.g., MBQC/hardware).
#     """
#     lay = getattr(qc_out, "layout", None)
#     if lay is None:
#         raise ValueError("Transpiled circuit has no layout; cannot map to physical indices.")
#
#     # Modern: input_qubit_mapping (VirtualBit -> physical int)
#     if hasattr(lay, "input_qubit_mapping") and lay.input_qubit_mapping is not None:
#         return [int(lay.input_qubit_mapping[q]) for q in qc_in.qubits]
#
#     # Older: final_layout mapping VirtualBit -> PhysicalQubit
#     if hasattr(lay, "final_layout") and lay.final_layout is not None:
#         fl = lay.final_layout
#         phys = []
#         for q in qc_in.qubits:
#             pq = fl[q]  # PhysicalQubit or int
#             phys.append(int(getattr(pq, "index", pq)))
#         return phys
#
#     # As a last resort try initial_layout + routing_permutation (not ideal but better than nothing)
#     if hasattr(lay, "initial_layout") and lay.initial_layout is not None:
#         init = lay.initial_layout
#         pre = [int(init[q]) for q in qc_in.qubits]  # pre-route virtual indices
#         try:
#             route = list(lay.routing_permutation())  # virtual->virtual; NOT physical, but keep as fallback
#         except Exception:
#             route = list(range(qc_out.num_qubits))
#         return [route[p] for p in pre]
#
#     raise ValueError("Cannot extract logical→physical indices from this layout.")
#
# def conditional_dist_from_state(state, measure_idxs, anc_idx, anc_value=0):
#     """
#     `state`: qiskit Statevector (or array convertible to one) in *its native qubit order*.
#     `measure_idxs`: indices (in this state's order) of qubits to keep; first is MSB in keys.
#     `anc_idx`: index (in this state's order) of the ancilla.
#     Returns dict bitstring -> probability of P(measure_idxs | anc=anc_value).
#     """
#     sv = state if isinstance(state, Statevector) else Statevector(state)
#     qargs_joint = list(measure_idxs) + [anc_idx]  # ancilla last -> easy slice
#     joint = sv.probabilities_dict(qargs=qargs_joint)
#
#     num, denom = {}, 0.0
#     keep = str(int(anc_value))
#     for key, p in joint.items():        # key is "<bits of measure_idxs><anc>"
#         if key[-1] == keep:
#             k = key[:-1]                # drop ancilla bit
#             num[k] = num.get(k, 0.0) + p
#             denom += p
#
#     if np.isclose(denom, 0.0):
#         return {format(i, f'0{len(measure_idxs)}b'): 0.0 for i in range(2**len(measure_idxs))}
#     out = {k: v / denom for k, v in num.items()}
#     return {k: out[k] for k in sorted(out)}  # stable key order
#
# def plot_distribution(dist: dict, title: str):
#     labels = list(dist.keys())
#     values = [dist[k] for k in labels]
#     plt.figure()
#     plt.bar(labels, values)
#     plt.xlabel("Measured bitstring (order = measure_qargs; first = MSB)")
#     plt.ylabel("Probability")
#     plt.title(title)
#     plt.tight_layout()
#     plt.show()

import numpy as np
import matplotlib.pyplot as plt
from qiskit.quantum_info import Statevector

# ---- tiny helpers ----

def ancilla_logical_index(qc, reg_name="c0", pos=0):
    """Index (in qc.qubits order) of bit `pos` in register `reg_name` (version-proof)."""
    qr = next(r for r in qc.qregs if r.name == reg_name)
    try:
        return qc.find_bit(qr[pos]).index    # modern Terra
    except AttributeError:
        return qc.qubits.index(qr[pos])      # older Terra

def overlap_abs_up_to_global_phase(a, b):
    """|⟨a|b⟩| for two statevectors (arrays). ==1 if equal up to global phase."""
    a = np.asarray(a, dtype=complex).ravel()
    b = np.asarray(b, dtype=complex).ravel()
    if a.size != b.size:
        return 0.0
    return abs(np.vdot(a, b))

def conditional_dist_qargs(sv_logical, measure_qargs, anc_idx, anc_value=0):
    """
    P(measure_qargs | anc_idx = anc_value), with qargs in *logical* order.
    Keys use the given measure order; first is MSB.
    """
    joint = sv_logical.probabilities_dict(qargs=list(measure_qargs) + [anc_idx])
    num, denom = {}, 0.0
    keep = str(int(anc_value))
    for bitstr, p in joint.items():       # "meas_bits ... anc"
        if bitstr[-1] == keep:
            k = bitstr[:-1]               # drop ancilla bit
            num[k] = num.get(k, 0.0) + p
            denom += p
    if np.isclose(denom, 0.0):
        return {format(i, f'0{len(measure_qargs)}b'): 0.0 for i in range(2**len(measure_qargs))}
    out = {k: v/denom for k, v in num.items()}
    return {k: out[k] for k in sorted(out)}  # stable order

def plot_distribution(dist, title):
    xs = list(dist.keys()); ys = [dist[k] for k in xs]
    plt.figure(); plt.bar(xs, ys)
    plt.xlabel("bitstring (order = measure_qargs; first = MSB)")
    plt.ylabel("Probability")
    plt.title(title)
    plt.tight_layout(); plt.show()

# ---- DROP-IN: use your existing helpers to resolve order, then condition+measure ----

def mbqc_q21_given_c0_zero(qc_in, transpiled_qc, psi, anc_reg="c0", anc_pos=0):
    """
    psi: Graphix MBQC statevector (flat array) on declared outputs.
    Uses your extract_logical_to_physical + undo_layout_on_state to obtain a *logical-ordered*
    Statevector, regardless of what order Graphix used, then returns and plots P(q2 q1 | c0=0).
    """
    N = transpiled_qc.num_qubits
    sv_mbqc = Statevector(psi, dims=[2]*N)

    # 1) Get logical->physical mapping from your transpiled circuit
    mapping = extract_logical_to_physical(qc_in, transpiled_qc)

    # 2) Try to undo layout. If undo is identity up to global phase, psi was already logical.
    sv_undo = undo_layout_on_state(sv_mbqc, mapping)
    ov = overlap_abs_up_to_global_phase(sv_undo.data, sv_mbqc.data)

    if np.isclose(ov, 1.0, atol=1e-10):
        sv_logical = sv_mbqc          # Graphix already returned logical order
        order_note = "MBQC outputs already in logical order"
    else:
        sv_logical = sv_undo          # Graphix returned physical; we converted to logical
        order_note = "Converted from physical to logical order via mapping"

    # 3) Indices in *logical* order
    anc_idx = ancilla_logical_index(qc_in, reg_name=anc_reg, pos=anc_pos)
    measure_qargs = [2, 1]            # you asked for q2 (MSB), q1 (LSB)

    # 4) Conditional distribution and plot
    dist = conditional_dist_qargs(sv_logical, measure_qargs, anc_idx, anc_value=0)
    plot_distribution(dist, title=f"P(q2 q1 | {anc_reg}=0) — {order_note}")

    return {"distribution": dist, "order_note": order_note, "overlap_abs": ov, "mapping": list(mapping)}





def main():

    user_vec = [0, 0]
    qc_in, input_vec = circuits.minimal_qrs(user_vec)

    print(qc_in)

    bw_pattern, col_map, transpiled_qc = brickwork_transpiler.transpile(
        qc_in, input_vec, routing_method="sabre", layout_method="sabre", with_ancillas=False
    )
    bw_pattern.standardize()
    bw_pattern.shift_signals()

    import graphix.noise_models.noise_model as noise_model

    tn = bw_pattern.simulate_pattern(backend="tensornetwork", graph_prep="parallel")
    psi = tn.to_statevector()


    mapping = extract_logical_to_physical(qc_in, transpiled_qc)


    # If you simulated with your MBQC engine and got a flat numpy array `psi`:
    sv_logical_from_mbqc = undo_layout_on_state(psi, mapping, total_qubits=transpiled_qc.num_qubits)

    import numpy as np
    from qiskit.quantum_info import Statevector

    import numpy as np
    import matplotlib.pyplot as plt
    from qiskit.quantum_info import Statevector

    # ---- inputs / knobs ----
    amps = np.asarray(sv_logical_from_mbqc, dtype=complex)
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from qiskit.quantum_info import Statevector

    # --- knobs ---
    target_qubit = 0  # c0 at bit-5 (little-endian: rightmost char is qubit 0)
    flip_plot_labels = True  # display-only MSB↔LSB flip
    user_feature = "demo"  # customize per run
    eps = 1e-12  # drop tiny probabilities post-selection
    top_k = None  # e.g., 40 to keep only the top-40 bars (None = keep all)

    # --- statevector -> probabilities (nonzero only) ---
    amps = np.asarray(sv_logical_from_mbqc, dtype=complex)
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

    # import numpy as np
    # from qiskit.quantum_info import Statevector
    # import numpy as np
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

    import numpy as np

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

    # import numpy as np
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
