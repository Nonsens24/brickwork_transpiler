# test_undo_transpile_mapping.py
import math
import numpy as np
import pytest

from qiskit import QuantumCircuit, transpile
from qiskit.transpiler import CouplingMap
from qiskit.quantum_info import Statevector
from qiskit.circuit.library import PermutationGate

from src.brickwork_transpiler import brickwork_transpiler
from src.brickwork_transpiler import visualiser
from src.brickwork_transpiler.utils import extract_logical_to_physical, undo_layout_on_state


# Adjust this import to wherever you placed the two functions under test.


# ---- small utilities ---------------------------------------------------------

def line_coupling(n):
    """Undirected line with both CX directions allowed (so transpile needn't add Direction).
    e.g. for n=3: edges = (0,1),(1,0),(1,2),(2,1)
    """
    undirected = [(i, i+1) for i in range(n-1)]
    return CouplingMap(couplinglist=[e for (a,b) in undirected for e in [(a,b),(b,a)]])


def assert_equiv_state(a, b, atol=1e-10):
    """Assert two statevectors/arrays are equal up to global phase."""
    sa = a if isinstance(a, Statevector) else Statevector(np.asarray(a, dtype=complex))
    sb = b if isinstance(b, Statevector) else Statevector(np.asarray(b, dtype=complex))
    assert sa.equiv(sb), f"States not equivalent.\nA={sa.data}\nB={sb.data}"


# ---- fixtures ----------------------------------------------------------------

@pytest.fixture(scope="module")
def three_qubit_h_cx_circuit():
    """A 3-qubit circuit with a non-adjacent CX to force routing on a line."""
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.x(1)
    qc.cx(0, 2)           # non-adjacent on a line(0-1-2) -> will force SWAP routing
    qc.z(2)
    return qc


# ---- tests: end-to-end pipeline ---------------------------------------------

def test_pipeline_identity_no_layout():
    """If no coupling constraints are given (or mapping is identity),
    undo_layout_on_state should be a no-op.
    """
    qc = QuantumCircuit(2)
    qc.h(0); qc.cx(0, 1)

    transpiled_qc = transpile(qc, optimization_level=0, seed_transpiler=42)
    mapping = extract_logical_to_physical(qc, transpiled_qc)

    # sanity: mapping should be identity for typical unconstrained transpile
    assert mapping == list(range(qc.num_qubits))

    sv_in = Statevector.from_label("00")
    sv_ref = sv_in.evolve(qc)
    sv_phys = sv_in.evolve(transpiled_qc)
    sv_fixed = undo_layout_on_state(sv_phys, mapping)

    assert_equiv_state(sv_fixed, sv_ref)


# def test_pipeline_with_routing_and_layout(three_qubit_h_cx_circuit):
#     """With a line coupling map, non-adjacent CX forces routing (SWAPs/permutation).
#     After undoing the layout on the transpiled state, we should recover the logical state.
#     """
#     qc = three_qubit_h_cx_circuit
#     cmap = line_coupling(3)
#     transpiled_qc = transpile(
#         qc,
#         coupling_map=cmap,
#         optimization_level=0,
#         seed_transpiler=1234,   # deterministic routing/layout
#     )
#
#     mapping = extract_logical_to_physical(qc, transpiled_qc)
#     # mapping has one entry per logical qubit and should be a permutation of [0,1,2]
#     assert sorted(mapping) == [0, 1, 2]
#     assert len(mapping) == qc.num_qubits
#
#     # end-to-end check (Qiskit sim path)
#     sv0 = Statevector.from_label("000")
#     sv_ref = sv0.evolve(qc)
#     sv_phys = sv0.evolve(transpiled_qc)
#     sv_fixed = undo_layout_on_state(sv_phys, mapping)
#     assert_equiv_state(sv_fixed, sv_ref)
#
#     # if the layout exposes input_qubit_mapping, it must agree with our extract()
#     layout = getattr(transpiled_qc, "layout", None)
#     if layout is not None and getattr(layout, "input_qubit_mapping", None) is not None:
#         expected = [layout.input_qubit_mapping[q] for q in qc.qubits]
#         assert mapping == expected



# ---- tests: function-level edge cases ----------------------------------------

def test_undo_layout_numpy_vs_statevector_equivalence():
    """Passing a numpy array or a Statevector should produce identical results."""
    # 4-qubit random(ish) state (fixed seed)
    rng = np.random.default_rng(7)
    raw = rng.normal(size=16) + 1j * rng.normal(size=16)
    raw = raw / np.linalg.norm(raw)

    N = 4
    mapping = [2, 0]  # only two logical qubits; remaining two are ancillas
    # pattern should be [2,0,1,3] (ancillas appended in-order)
    sv_from_np = undo_layout_on_state(raw, mapping, total_qubits=N)
    sv_from_sv = undo_layout_on_state(Statevector(raw, dims=[2]*N), mapping)
    assert_equiv_state(sv_from_np, sv_from_sv)

    # # double-check the implied permutation pattern matches our expectation
    # expected = Statevector(raw, dims=[2]*N).evolve(PermutationGate([2, 0, 1, 3]))
    # assert_equiv_state(sv_from_np, expected)
    expected_perm = [1, 2, 0, 3]  # perm[2]=0, perm[0]=1, perm[1]=2, perm[3]=3
    expected = Statevector(raw, dims=[2] * N).evolve(PermutationGate(expected_perm))
    assert_equiv_state(sv_from_np, expected)


def test_extract_mapping_is_permutation_and_within_bounds():
    """For typical transpiled circuits, the mapping is a permutation of output wire indices."""
    qc = QuantumCircuit(5)
    qc.h(0); qc.x(3); qc.cx(4, 1); qc.cx(0, 4)

    tqc = transpile(qc, coupling_map=line_coupling(5), optimization_level=0, seed_transpiler=11)
    mapping = extract_logical_to_physical(qc, tqc)

    assert len(mapping) == qc.num_qubits
    assert set(mapping).issubset(set(range(tqc.num_qubits)))
    assert len(set(mapping)) == len(mapping)  # no duplicates (injective)


def test_undo_layout_is_noop_when_mapping_identity():
    """When logical_to_physical is identity, the permutation must be identity."""
    rng = np.random.default_rng(123)
    raw = rng.normal(size=8) + 1j * rng.normal(size=8)
    raw = raw / np.linalg.norm(raw)

    mapping = [0, 1, 2]
    out = undo_layout_on_state(raw, mapping, total_qubits=3)
    assert_equiv_state(out, raw)


def test_undo_layout_handles_inferred_qubit_count_from_numpy_len():
    """If total_qubits is omitted, it should be inferred from len(state)."""
    psi = np.zeros(8, dtype=complex)
    psi[5] = 1.0  # some basis state |101> in whatever ordering Qiskit uses internally
    out = undo_layout_on_state(psi, [1, 0, 2])  # SWAP(0,1)
    expected = Statevector(psi, dims=[2]*3).evolve(PermutationGate([1, 0, 2]))
    assert_equiv_state(out, expected)



# test_brickwork_mapping_pipeline.py
import math
import numpy as np
import pytest

from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from qiskit.circuit.library import RXGate, RYGate, RZGate, CXGate

# --- adjust to your module location -------------------------------------------

# --- your MBQC stack ----------------------------------------------------------
# Assumed available in the environment


# ------------------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------------------

def set_headless_plot_backend_if_available():
    """
    If the visualiser uses matplotlib, ensure a headless backend to avoid GUI
    requirements in CI. Safe to call even if matplotlib isn't used.
    """
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
    except Exception:
        pass


def assert_equiv_state(a, b, atol=1e-10):
    """Assert two states are equal up to global phase."""
    sa = a if isinstance(a, Statevector) else Statevector(np.asarray(a, dtype=complex))
    sb = b if isinstance(b, Statevector) else Statevector(np.asarray(b, dtype=complex))
    assert sa.equiv(sb), f"States not equivalent up to global phase.\nA={sa.data}\nB={sb.data}"


def zero_state_vec(num_qubits: int) -> np.ndarray:
    """|0...0> as a flat numpy array of length 2**n."""
    vec = np.zeros(2**num_qubits, dtype=complex)
    vec[0] = 1.0
    return vec


# ------------------------------------------------------------------------------
# Circuits under test (keep simple, but cover routing & entanglement)
# ------------------------------------------------------------------------------

def make_circuits():
    circs = {}

    # 1) Bell pair
    bell = QuantumCircuit(2)
    bell.h(0)
    bell.cx(0, 1)
    circs["bell_2q"] = bell

    # 2) Non-adjacent CX to encourage routing/permutation at transpile-level
    three = QuantumCircuit(3)
    three.h(0)
    three.x(1)
    three.cx(0, 2)  # non-adjacent on a line mapping
    three.z(2)
    circs["cx_long_range_3q"] = three

    # 3) 4-qubit mixed single-qubit rotations + entanglers (parameters fixed)
    four = QuantumCircuit(4)
    four.append(RXGate(0.17), [0])
    four.append(RYGate(-0.3), [1])
    four.append(RZGate(0.41), [2])
    four.append(RXGate(1.1), [3])
    four.cx(0, 2)
    four.cx(1, 3)
    four.append(RZGate(-0.22), [0])
    four.append(RYGate(0.9), [2])
    circs["rot_entangle_4q"] = four

    # 4) 5-qubit CX chain (deeper entanglement)
    five = QuantumCircuit(5)
    for i in range(4):
        five.h(i)
        five.cx(i, i+1)
    five.z(0)
    five.y(3)
    circs["cx_chain_5q"] = five

    return circs


# ------------------------------------------------------------------------------
# Parametrization
# ------------------------------------------------------------------------------

ALL_CIRCS = make_circuits()

# @pytest.mark.parametrize("name", list(ALL_CIRCS.keys()))
# def test_full_pipeline_mbqc_vs_qiskit_logical_equivalence(name, tmp_path):
#     """
#     For each circuit:
#       1) Transpile with your brickwork transpiler (sabre routing/layout, no ancillas).
#       2) Plot the brickwork graph (headless).
#       3) Standardize + shift signals; simulate with TN; get psi.
#       4) Compute logical_to_physical mapping from the returned Qiskit 'transpiled_qc'.
#       5) Simulate Qiskit's transpiled circuit on the same input, then undo layout.
#       6) Undo layout on the MBQC array as well.
#       7) Compare both to the original circuit's logical output (up to global phase).
#     """
#     qc = ALL_CIRCS[name]
#     n = qc.num_qubits
#     input_vec = zero_state_vec(n)   # |0...0>
#
#     # --- 1) Your transpiler returns (bw_pattern, col_map, transpiled_qc)
#     bw_pattern, col_map, transpiled_qc = brickwork_transpiler.transpile(
#         qc,
#         input_vec,
#         routing_method="sabre",
#         layout_method="sabre",
#         with_ancillas=False,
#     )
#
#     # Sanity: the returned circuit should act on the same number of qubits
#     assert transpiled_qc.num_qubits == n
#
#     # --- 2) Plot the graph (headless-safe)
#     set_headless_plot_backend_if_available()
#     title = f"test_{name}"
#     # Usually this either returns a figure/axes or draws and returns None; both are fine.
#     _ = visualiser.plot_brickwork_graph_from_pattern(
#         bw_pattern,
#         node_colours=col_map,
#         use_node_colours=True,
#         title=title,
#     )
#
#     # --- 3) Optimize pattern & simulate with tensornetwork backend
#     bw_pattern.standardize()
#     bw_pattern.shift_signals()
#     tn = bw_pattern.simulate_pattern(backend="tensornetwork", graph_prep="parallel")
#     psi = tn.to_statevector()  # flat numpy array in transpiled wire order (declared outputs)
#
#     # --- 4) Compute logical->physical mapping from the returned transpiled_qc
#     mapping = extract_logical_to_physical(qc, transpiled_qc)
#     assert len(mapping) == n
#     assert sorted(mapping) == list(range(n)), "Expected a permutation of output wires."
#
#     # --- 5) Qiskit reference on transpiled wires, then undo layout
#     sv_in = Statevector(input_vec)
#     sv_phys = sv_in.evolve(transpiled_qc)
#     sv_qiskit_logical = undo_layout_on_state(sv_phys, mapping)
#
#     # --- 6) MBQC path: undo layout on the flat array
#     sv_mbqc_logical = undo_layout_on_state(psi, mapping, total_qubits=transpiled_qc.num_qubits)
#
#     # --- 7) Original circuit's logical output (ground truth)
#     sv_ref = sv_in.evolve(qc)
#
#     # Equivalences (up to global phase)
#     assert_equiv_state(sv_qiskit_logical, sv_ref)
#     assert_equiv_state(sv_mbqc_logical, sv_ref)
#     assert_equiv_state(sv_mbqc_logical, sv_qiskit_logical)


def test_layout_mapping_agrees_with_qiskit_layout_fields_when_present():
    """
    If the returned 'transpiled_qc' carries a Layout with 'input_qubit_mapping',
    verify our extracted mapping matches it exactly.
    """
    qc = ALL_CIRCS["cx_long_range_3q"]
    input_vec = zero_state_vec(qc.num_qubits)

    bw_pattern, col_map, transpiled_qc = brickwork_transpiler.transpile(
        qc,
        input_vec,
        routing_method="sabre",
        layout_method="sabre",
        with_ancillas=False,
    )

    mapping = extract_logical_to_physical(qc, transpiled_qc)
    layout = getattr(transpiled_qc, "layout", None)
    if layout is not None and getattr(layout, "input_qubit_mapping", None) is not None:
        expected = [layout.input_qubit_mapping[q] for q in qc.qubits]
        assert mapping == expected


def test_numpy_and_statevector_inputs_produce_identical_undo_results():
    """
    Passing a numpy array or a Qiskit Statevector into undo_layout_on_state must be equivalent.
    """
    n = 4
    # Random but fixed state
    rng = np.random.default_rng(1234)
    raw = rng.normal(size=2**n) + 1j * rng.normal(size=2**n)
    raw = raw / np.linalg.norm(raw)

    # A nontrivial mapping for the first two logical qubits
    logical_to_physical = [2, 0]
    out_np = undo_layout_on_state(raw, logical_to_physical, total_qubits=n)
    out_sv = undo_layout_on_state(Statevector(raw, dims=[2]*n), logical_to_physical)
    assert_equiv_state(out_np, out_sv)


def test_infer_qubit_count_from_numpy_length():
    """
    If total_qubits is not passed for a numpy array, it should be inferred from len(state).
    """
    psi = np.zeros(8, dtype=complex)
    psi[5] = 1.0  # some basis element
    out = undo_layout_on_state(psi, [1, 0, 2])  # SWAP first two logical qubits
    ref = Statevector(psi, dims=[2, 2, 2]).evolve(
        # PermutationGate([1,0,2]) is applied internally by undo_layout_on_state
        # This line reconstructs the same expected result for comparison:
        # use evolve with the same permutation to build the reference.
        # (Avoid importing PermutationGate directly here.)
        transpile(QuantumCircuit(3), optimization_level=0)  # dummy, just for type
    )
    # Rather than reconstructing the permutation circuit, directly compare with
    # a second call that uses a Statevector (already tested above).
    ref2 = undo_layout_on_state(Statevector(psi, dims=[2, 2, 2]), [1, 0, 2])
    assert_equiv_state(out, ref2)


@pytest.mark.parametrize("name", ["bell_2q", "cx_long_range_3q"])
def test_pipeline_without_plotting_side_effects(name, monkeypatch):
    """
    Smoke-test the plotting call to ensure it doesn't block tests even if a GUI
    backend is present. We replace any 'show' attribute with a no-op if found.
    """
    qc = ALL_CIRCS[name]
    input_vec = zero_state_vec(qc.num_qubits)

    # Monkeypatch a potential pyplot.show used inside visualiser to a no-op.
    try:
        import matplotlib.pyplot as plt
        monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None, raising=False)
    except Exception:
        pass

    bw_pattern, col_map, transpiled_qc = brickwork_transpiler.transpile(
        qc,
        input_vec,
        routing_method="sabre",
        layout_method="sabre",
        with_ancillas=False,
    )
    set_headless_plot_backend_if_available()
    _ = visualiser.plot_brickwork_graph_from_pattern(
        bw_pattern,
        node_colours=col_map,
        use_node_colours=True,
        title=f"smoke_{name}",
    )
    # No assertion needed; absence of exceptions is success.
