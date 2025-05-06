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
    visualiser.plot_brickwork_graph_from_pattern(bw_pattern, title="bw: test_bell_proj_from_plus")

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
    visualiser.plot_brickwork_graph_from_pattern(bw_pattern, title="bw: test_cx_from_0")

    # Simulate the generated pattern
    outstate = bw_pattern.simulate_pattern(backend='statevector').flatten()

    # Compare output up to global phase
    assert utils.assert_equal_up_to_global_phase(outstate, output_ref.data)

def test_system_shift():
    # 1) Create the |++> state directly
    input_vec = Statevector.from_label('+++')  # three-qubit plus state

    # 2) Define your 2-qubit circuit (no H gates needed)
    qc = QuantumCircuit(3)
    qc.t(0)
    qc.t(1)
    qc.cx(0, 1)
    qc.s(2)
    qc.s(0)

    # Transpile!
    bw_pattern, output_ref = brickwork_transpiler.transpile(qc, input_vec)
    visualiser.plot_brickwork_graph_from_pattern(bw_pattern, title="bw: test_system_shift")

    # Simulate the generated pattern
    outstate = bw_pattern.simulate_pattern(backend='statevector').flatten()

    # Compare output up to global phase
    assert utils.assert_equal_up_to_global_phase(outstate, output_ref.data)

def test_four_in_cx_cancel():
    # 1) Create the |++> state directly
    input_vec = Statevector.from_label('+++')  # three-qubit plus state

    # 2) Define your 2-qubit circuit (no H gates needed)
    qc = QuantumCircuit(4)
    qc.cx(0, 1)
    qc.cx(0, 1)
    qc.h(3)


    # Transpile!
    bw_pattern, output_ref = brickwork_transpiler.transpile(qc, input_vec)
    visualiser.plot_brickwork_graph_from_pattern(bw_pattern, title="bw: test_four_in_cx_cancel")

    # Simulate the generated pattern
    outstate = bw_pattern.simulate_pattern(backend='statevector').flatten()

    # Compare output up to global phase
    assert utils.assert_equal_up_to_global_phase(outstate, output_ref.data)

def test_five_in_optimise_gates():
    # 1) Create the |++> state directly
    input_vec = Statevector.from_label('+++')  # three-qubit plus state

    # 2) Define your 2-qubit circuit (no H gates needed)
    qc = QuantumCircuit(5)
    qc.cx(0, 1)
    qc.s(2)
    qc.rx(np.pi/2, 2)
    qc.t(2)
    qc.x(2)
    qc.cx(3, 4)

    # Transpile!
    bw_pattern, output_ref = brickwork_transpiler.transpile(qc, input_vec)
    visualiser.plot_brickwork_graph_from_pattern(bw_pattern, title="bw: test_five_in_optimise_gates")

    # Simulate the generated pattern
    outstate = bw_pattern.simulate_pattern(backend='statevector').flatten()

    # Compare output up to global phase
    assert utils.assert_equal_up_to_global_phase(outstate, output_ref.data)