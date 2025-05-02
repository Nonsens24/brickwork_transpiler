import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from src.brickwork_transpiler import bricks, utils


def test_random_rotation_constant_angles():
    a = -1/4
    b = 1/4
    c = 0

    test_state = bricks.arbitrary_brick(a, b, c).simulate_pattern(backend='statevector').flatten()

    # Reference circuit
    qc = QuantumCircuit(1)
    qc.h(0)  # init to |+>
    qc.rz(np.pi * a, 0)
    qc.rx(np.pi * b, 0)
    qc.rz(np.pi * c, 0)

    ref_state = Statevector.from_instruction(qc).data

    assert utils.assert_equal_up_to_global_phase(test_state, ref_state)

def test_arbitrary_rotation_hadamard_from_plus():
    a = 1/4
    b = 1/4
    c = 1/4

    test_state = bricks.arbitrary_brick(a, b, c).simulate_pattern(backend='statevector').flatten()

    # Reference circuit
    qc = QuantumCircuit(1)
    qc.h(0)  # init to |+>
    qc.rz(np.pi * a, 0)
    qc.rx(np.pi * b, 0)
    qc.rz(np.pi * c, 0)

    ref_state = Statevector.from_instruction(qc).data

    assert utils.assert_equal_up_to_global_phase(test_state, ref_state) # Compensating for global phase tol def = 1e-6

def test_arbitrary_rotation_s_from_plus():
    a = 1/4
    b = 0
    c = 0

    test_state = bricks.arbitrary_brick(a, b, c)

    test_result_state = test_state.simulate_pattern(backend='statevector').flatten()

    # Reference circuit
    qc = QuantumCircuit(1)
    qc.h(0) #init to |+>
    qc.rz(np.pi * a, 0)
    qc.rx(np.pi * b, 0)
    qc.rz(np.pi * c, 0)

    ref_state = Statevector.from_instruction(qc).data

    assert utils.assert_equal_up_to_global_phase(test_result_state, ref_state)

def test_arbitrary_rotations_t_from_plus_non_clifford():
    a = 1/8
    b = 0
    c = 0

    test_state = bricks.arbitrary_brick(a, b, c)

    test_result_state = test_state.simulate_pattern(backend='statevector').flatten()

    # Reference circuit
    qc = QuantumCircuit(1)
    qc.h(0) #init to |+>
    qc.rz(np.pi * a, 0)
    qc.rx(np.pi * b, 0)
    qc.rz(np.pi * c, 0)

    ref_state = Statevector.from_instruction(qc).data

    assert utils.assert_equal_up_to_global_phase(test_result_state, ref_state)

def test_arbitrary_rotations_zeroes_from_plus():
    a = 0
    b = 0
    c = 0

    test_state = bricks.arbitrary_brick(a, b, c)
    test_result_state = test_state.simulate_pattern(backend='statevector').flatten()

    # Reference circuit
    qc = QuantumCircuit(1)
    qc.h(0)  # init to |+>
    qc.rz(np.pi * a, 0)
    qc.rx(np.pi * b, 0)
    qc.rz(np.pi * c, 0)

    ref_state = Statevector.from_instruction(qc).data

    assert utils.assert_equal_up_to_global_phase(test_result_state, ref_state)


def test_arbitrary_rotations_random_angles_from_plus():
    a = np.random.uniform(0, 2)
    b = np.random.uniform(0, 2)
    c = np.random.uniform(0, 2)

    test_state = bricks.arbitrary_brick(a, b, c)
    test_result_state = test_state.simulate_pattern(backend='statevector').flatten()

    # Reference circuit
    qc = QuantumCircuit(1)
    qc.h(0) #init to |+>
    qc.rz(np.pi * a, 0)
    qc.rx(np.pi * b, 0)
    qc.rz(np.pi * c, 0)

    ref_state = Statevector.from_instruction(qc).data

    assert utils.assert_equal_up_to_global_phase(test_result_state, ref_state)

# def test_arbitrary_rotation_hadamard_from_input_state():
#     a = 1 / 4
#     b = 1 / 4
#     c = 1 / 4
#
#     assert np.allclose(bricks.arbitrary_brick(a, b, c).simulate_pattern(backend='statevector').flatten(),
#                        [0.70710678 + 0.j, 0.70710678 + 0.j])