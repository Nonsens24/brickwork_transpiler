import numpy as np
from matplotlib import pyplot as plt
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from graphix.transpiler import Circuit
from qiskit.visualization import dag_drawer, circuit_drawer

from libs.gospel.gospel.brickwork_state_transpiler.brickwork_state_transpiler import transpile
from src.brickwork_transpiler import utils, visualiser, brickwork_transpiler
from src.brickwork_transpiler.utils import permute_qubits


def test_different_arbitrary_rotations_full_brick():

    # 1) Define a simple 1-qubit circuit: Hadamard followed by T gate
    circuit = Circuit(4)

    # circuit.rx(0, 0.01)
    # circuit.rx(1, 0.01)

    circuit.rz(0, np.pi / 4)
    circuit.rx(0, np.pi / 4)
    circuit.rz(0, np.pi / 5)

    circuit.rz(1, np.pi / 3)
    circuit.rx(1, np.pi / 4)
    circuit.rz(1, np.pi / 4)

    circuit.rz(2, np.pi / 4)
    circuit.rx(2, np.pi / 4)
    circuit.rz(2, np.pi / 2)

    circuit.rz(3, np.pi / 4)
    circuit.rx(3, np.pi / 4)
    circuit.rz(3, np.pi / 4)


    # Ref circ
    input_vec = Statevector.from_label('++++')  # two-qubit plus state

    # 2) Define your 2-qubit circuit (no H gates needed)
    qc = QuantumCircuit(4)

    qc.rz(np.pi / 4, 0)
    qc.rx(np.pi / 4, 0)
    qc.rz(np.pi / 5, 0)

    qc.rz(np.pi / 3, 1)
    qc.rx(np.pi / 4, 1)
    qc.rz(np.pi / 4, 1)

    qc.rz(np.pi / 4, 2)
    qc.rx(np.pi / 4, 2)
    qc.rz(np.pi / 2, 2)

    qc.rz(np.pi / 4, 3)
    qc.rx(np.pi / 4, 3)
    qc.rz(np.pi / 4, 3)





    qc2 = qc.reverse_bits()


    # ref_state = Statevector.from_instruction(qc).data
    ref_state = input_vec.evolve(qc2)

    # Transpile!
    bw_pattern = transpile(circuit)

    visualiser.plot_graphix_noise_graph(bw_pattern)

    # Simulate the generated pattern
    outstate = bw_pattern.simulate_pattern(backend='statevector').flatten()

    # Compare output up to global phase
    assert utils.assert_equal_up_to_global_phase(outstate, ref_state.data)





def test_four_arbitrary_rotations_full_same_bricks():

    circuit = Circuit(4)
    circuit.rz(0, np.pi / 4)
    circuit.rx(0, np.pi / 4)
    circuit.rz(0, np.pi / 5)

    circuit.rz(1, np.pi / 3)
    circuit.rx(1, np.pi / 4)
    circuit.rz(1, np.pi / 4)

    circuit.rz(2, np.pi / 4)
    circuit.rx(2, np.pi / 4)
    circuit.rz(2, np.pi / 2)

    circuit.rz(3, np.pi / 4)
    circuit.rx(3, np.pi / 4)
    circuit.rz(3, np.pi / 4)

    # Ref circ
    input_vec = Statevector.from_label('++++')  # two-qubit plus state

    # 2) Define your 2-qubit circuit (no H gates needed)
    qc = QuantumCircuit(4)
    qc.rz(np.pi / 4, 0)
    qc.rx(np.pi / 4, 0)
    qc.rz(np.pi / 5, 0)

    qc.rz(np.pi / 3, 1)
    qc.rx(np.pi / 4, 1)
    qc.rz(np.pi / 4, 1)

    qc.rz(np.pi / 4, 2)
    qc.rx(np.pi / 4, 2)
    qc.rz(np.pi / 2, 2)

    qc.rz(np.pi / 4, 3)
    qc.rx(np.pi / 4, 3)
    qc.rz(np.pi / 4, 3)


    my_bw_pattern, output_refQQQQ, col_map = brickwork_transpiler.transpile(qc, input_vec)
    my_outstate = my_bw_pattern.simulate_pattern(backend='statevector').flatten()

    qc_perm = permute_qubits(qc, perm=[2, 0, 3, 1])
    ref_state = input_vec.evolve(qc_perm)
    visualiser.plot_brickwork_graph_from_pattern(my_bw_pattern,
                                                 node_colours=col_map,
                                                 use_node_colours=True,
                                                 title="Brickwork Graph: The Big Test")

    # Transpile!
    gospel_bw_pattern = transpile(circuit)

    visualiser.plot_graphix_noise_graph(gospel_bw_pattern)

    # Simulate the generated pattern
    gospel_outstate = gospel_bw_pattern.simulate_pattern(backend='statevector').flatten()

    # GEt ref state
    # print(f"My outstate {my_outstate}")
    # print(f"Gospel outstate {gospel_outstate}")
    # print(f"ref_state {ref_state.data}")

    # Compare output up to global phase
    assert utils.assert_equal_up_to_global_phase(ref_state.data, my_outstate)

