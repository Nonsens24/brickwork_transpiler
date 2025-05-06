from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from src.brickwork_transpiler import decomposer, visualiser, graph_builder, pattern_converter, utils, \
    brickwork_transpiler


def test_bell_proj_from_plus():

    # 1) Create the |++> state directly
    input_vec = Statevector.from_label('++')  # two-qubit plus state

    # 2) Define your 2-qubit circuit (no H gates needed)
    qc = QuantumCircuit(2)
    qc.cx(0, 1)

    # Transpile!
    bw_pattern, output_ref = brickwork_transpiler.transpile(qc, input_vec)

    # Simulate the generated pattern
    outstate = bw_pattern.simulate_pattern(backend='statevector').flatten()

    # Compare output up to global phase
    assert utils.assert_equal_up_to_global_phase(outstate, output_ref.data)


def test_cx_from_0():
    # 1) Create the |++> state directly
    input_vec = Statevector.from_label('++')  # two-qubit plus state

    # 2) Define your 2-qubit circuit (no H gates needed)
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.h(1)
    qc.cx(0, 1)

    # Transpile!
    bw_pattern, output_ref = brickwork_transpiler.transpile(qc, input_vec)

    # Simulate the generated pattern
    outstate = bw_pattern.simulate_pattern(backend='statevector').flatten()

    # Compare output up to global phase
    assert utils.assert_equal_up_to_global_phase(outstate, output_ref.data)