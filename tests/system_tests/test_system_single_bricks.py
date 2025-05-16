import itertools
import unittest

import numpy as np
from graphix import Circuit
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from libs.gospel.gospel.brickwork_state_transpiler.brickwork_state_transpiler import transpile
from src.brickwork_transpiler import decomposer, visualiser, graph_builder, pattern_converter, utils, \
    brickwork_transpiler
from src.brickwork_transpiler.utils import reorder_via_transpose, reorder_via_transpose_n, permute_qubits


class TestSingleGates(unittest.TestCase):

    def test_single_CX_top_bottom(self):

        # 1) Create the |++> state directly
        input_vec = Statevector.from_label('++')  # two-qubit plus state

        # 2) Define your 2-qubit circuit (no H gates needed)
        qc = QuantumCircuit(2)
        qc.cx(0, 1)

        # Transpile!
        bw_pattern, output_ref, col_map = brickwork_transpiler.transpile(qc, input_vec)
        visualiser.plot_brickwork_graph_from_pattern(bw_pattern,
                                                     node_colours=col_map,
                                                     use_node_colours=True,
                                                     title="Brickwork Graph: test_single_CX_top_bottom")

        # Simulate the generated pattern
        outstate = bw_pattern.simulate_pattern(backend='statevector').flatten()

        # Compare output up to global phase
        assert utils.assert_equal_up_to_global_phase(outstate, output_ref.data)


    def test_single_CX_bottom_top(self):

        # 1) Create the |++> state directly
        input_vec = Statevector.from_label('++')  # two-qubit plus state

        # 2) Define your 2-qubit circuit (no H gates needed)
        qc = QuantumCircuit(2)
        qc.cx(1, 0)

        # Transpile!
        bw_pattern, output_ref, col_map = brickwork_transpiler.transpile(qc, input_vec)
        visualiser.plot_brickwork_graph_from_pattern(bw_pattern,
                                                     node_colours=col_map,
                                                     use_node_colours=True,
                                                     title="Brickwork Graph: test_single_CX_bottom_top")
        # Simulate the generated pattern
        outstate = bw_pattern.simulate_pattern(backend='statevector').flatten()

        # Compare output up to global phase
        assert utils.assert_equal_up_to_global_phase(outstate, output_ref.data)

    def test_hadamard_half_brick(self):
        # 1) Create the |++> state directly
        input_vec = Statevector.from_label('+')  # two-qubit plus state

        # 2) Define your 2-qubit circuit (no H gates needed)
        qc = QuantumCircuit(1)
        qc.h(0)
        # qc.x(0)

        # Transpile!
        bw_pattern, output_ref, col_map = brickwork_transpiler.transpile(qc, input_vec)
        visualiser.plot_brickwork_graph_from_pattern(bw_pattern,
                                                     node_colours=col_map,
                                                     use_node_colours=True,
                                                     title="Brickwork Graph: test_euler_rotation_half_brick")
        # Simulate the generated pattern
        outstate = bw_pattern.simulate_pattern(backend='statevector').flatten()

        # Compare output up to global phase
        assert utils.assert_equal_up_to_global_phase(outstate, output_ref.data)

    def test_pauli_x_half_brick(self):
        # 1) Create the |++> state directly
        input_vec = Statevector.from_label('+')  # two-qubit plus state

        # 2) Define your 2-qubit circuit (no H gates needed)
        qc = QuantumCircuit(1)
        qc.x(0)

        # Transpile!
        bw_pattern, output_ref, col_map = brickwork_transpiler.transpile(qc, input_vec)
        visualiser.plot_brickwork_graph_from_pattern(bw_pattern,
                                                     node_colours=col_map,
                                                     use_node_colours=True,
                                                     title="Brickwork Graph: test_euler_rotation_half_brick")
        # Simulate the generated pattern
        outstate = bw_pattern.simulate_pattern(backend='statevector').flatten()

        # Compare output up to global phase
        assert utils.assert_equal_up_to_global_phase(outstate, output_ref.data)

    def test_euler_rotation_half_brick(self):
        # 1) Create the |++> state directly
        input_vec = Statevector.from_label('+')  # two-qubit plus state

        # 2) Define your 2-qubit circuit (no H gates needed)
        qc = QuantumCircuit(1)
        qc.rx(np.pi/3, 0)
        qc.rx(np.pi / 4, 0)
        qc.rz(np.pi / 5, 0)
        qc.rx(np.pi / 2, 0)
        qc.rz(np.pi / 5, 0)

        # Transpile!
        bw_pattern, output_ref, col_map = brickwork_transpiler.transpile(qc, input_vec)
        visualiser.plot_brickwork_graph_from_pattern(bw_pattern,
                                                     node_colours=col_map,
                                                     use_node_colours=True,
                                                     title="Brickwork Graph: test_euler_rotation_half_brick")
        # Simulate the generated pattern
        outstate = bw_pattern.simulate_pattern(backend='statevector').flatten()

        # Compare output up to global phase
        assert utils.assert_equal_up_to_global_phase(outstate, output_ref.data)

    def test_euler_rotation_id_full_brick(self):
        # 1) Create the |++> state directly
        input_vec = Statevector.from_label('++')  # two-qubit plus state

        # 2) Define your 2-qubit circuit (no H gates needed)
        qc = QuantumCircuit(2)
        qc.h(0)


        # Transpile!
        bw_pattern, output_ref, col_map = brickwork_transpiler.transpile(qc, input_vec)
        visualiser.plot_brickwork_graph_from_pattern(bw_pattern,
                                                     node_colours=col_map,
                                                     use_node_colours=True,
                                                     title="Brickwork Graph: test_euler_rotation_id_full_brick")
        # Simulate the generated pattern
        outstate = bw_pattern.simulate_pattern(backend='statevector').flatten()

        # change order qiskit ref state to match Graphix'
        output_ref_graphix_order = reorder_via_transpose(output_ref.data)

        # Compare output up to global phase
        assert utils.assert_equal_up_to_global_phase(outstate, output_ref_graphix_order)

class TestmultipleGates(unittest.TestCase):

    def test_both_H_full_brick(self):
        # 1) Create the |++> state directly
        input_vec = Statevector.from_label('++')  # two-qubit plus state

        # 2) Define your 2-qubit circuit (no H gates needed)
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.h(1)

        # Transpile!
        bw_pattern, output_ref, col_map = brickwork_transpiler.transpile(qc, input_vec)
        visualiser.plot_brickwork_graph_from_pattern(bw_pattern,
                                                     node_colours=col_map,
                                                     use_node_colours=True,
                                                     title="Brickwork Graph: test_both_H_full_brick")
        # Simulate the generated pattern
        outstate = bw_pattern.simulate_pattern(backend='statevector').flatten()

        # Compare output up to global phase
        assert utils.assert_equal_up_to_global_phase(outstate, output_ref.data)

    def test_different_arbitrary_rotations_full_brick(self):
        # 1) Create the |++> state directly
        input_vec = Statevector.from_label('++')  # two-qubit plus state

        # 2) Define your 2-qubit circuit (no H gates needed)
        qc = QuantumCircuit(2)
        qc.rz(np.pi / 2, 0)
        qc.rx(np.pi / 4, 0)
        qc.rz(np.pi / 2, 0)

        qc.rz(np.pi / 2, 1)
        qc.rx(np.pi / 3, 1)
        qc.rz(np.pi / 2, 1)

        # Transpile!
        bw_pattern, output_ref, col_map = brickwork_transpiler.transpile(qc, input_vec)

        visualiser.plot_brickwork_graph_from_pattern(bw_pattern,
                                                     node_colours=col_map,
                                                     use_node_colours=True,
                                                     title="Brickwork Graph: test_different_arbitrary_rotations_full_brick")
        # Simulate the generated pattern
        outstate = bw_pattern.simulate_pattern(backend='statevector').flatten()

        # change order qiskit ref state to match Graphix'
        output_ref_graphix_order = reorder_via_transpose(output_ref.data)

        # Compare output up to global phase
        assert utils.assert_equal_up_to_global_phase(outstate, output_ref_graphix_order)

    def test_three_same_arbitrary_rotations_full_brick(self):
        # 1) Create the |++> state directly
        input_vec = Statevector.from_label('+++')  # two-qubit plus state

        # 2) Define your 2-qubit circuit (no H gates needed)
        qc = QuantumCircuit(3)
        qc.rz(np.pi / 2, 0)
        qc.rx(np.pi / 4, 0)
        qc.rz(np.pi / 2, 0)

        qc.rz(np.pi / 2, 1)
        qc.rx(np.pi / 4, 1)
        qc.rz(np.pi / 2, 1)

        qc.rz(np.pi / 2, 2)
        qc.rx(np.pi / 4, 2)
        qc.rz(np.pi / 2, 2)

        # Transpile!
        bw_pattern, output_ref, col_map = brickwork_transpiler.transpile(qc, input_vec)
        visualiser.plot_brickwork_graph_from_pattern(bw_pattern,
                                                     node_colours=col_map,
                                                     use_node_colours=True,
                                                     title="Brickwork Graph: test_three_same_arbitrary_rotations_full_brick")
        # Simulate the generated pattern
        outstate = bw_pattern.simulate_pattern(backend='statevector').flatten()

        # change order qiskit ref state to match Graphix'
        output_ref_graphix_order = reorder_via_transpose(output_ref.data)

        # Compare output up to global phase
        assert utils.assert_equal_up_to_global_phase(outstate, output_ref_graphix_order)


    def test_three_different_arbitrary_rotations_full_brick(self):
        # 1) Create the |++> state directly
        input_vec = Statevector.from_label('+++')  # two-qubit plus state

        # 2) Define your 2-qubit circuit (no H gates needed)
        qc = QuantumCircuit(3)
        qc.rz(np.pi / 2, 0)
        qc.rx(np.pi / 4, 0)
        qc.rz(np.pi / 3, 0)

        qc.rz(np.pi / 3, 1)
        qc.rx(np.pi / 4, 1)
        qc.rz(np.pi / 2, 1)

        qc.rz(np.pi / 2, 2)
        qc.rx(np.pi / 4, 2)
        qc.rz(np.pi / 8, 2)

        qc_ref = QuantumCircuit(3)

        qc_ref.rz(np.pi / 3, 0)
        qc_ref.rx(np.pi / 4, 0)
        qc_ref.rz(np.pi / 2, 0)

        qc_ref.rz(np.pi / 2, 1)
        qc_ref.rx(np.pi / 4, 1)
        qc_ref.rz(np.pi / 3, 1)

        qc_ref.rz(np.pi / 2, 2)
        qc_ref.rx(np.pi / 4, 2)
        qc_ref.rz(np.pi / 8, 2)

        # Transpile!
        bw_pattern, output_ref, col_map = brickwork_transpiler.transpile(qc, input_vec)
        visualiser.plot_brickwork_graph_from_pattern(bw_pattern,
                                                     node_colours=col_map,
                                                     use_node_colours=True,
                                                     title="Brickwork Graph: test_three_different_arbitrary_rotations_full_brick")
        # Simulate the generated pattern
        outstate = bw_pattern.simulate_pattern(backend='statevector').flatten()

        reference_output = input_vec.evolve(qc_ref)
        # output_ref_graphix_order = reorder_via_transpose(output_ref.data)

        # Compare output up to global phase
        assert utils.assert_equal_up_to_global_phase(outstate, reference_output.data)

    def test_three_symmetric_rows_arbitrary_rotations(self):
        # 1) Create the |++> state directly
        input_vec = Statevector.from_label('+++')  # two-qubit plus state

        # 2) Define your 2-qubit circuit (no H gates needed)
        qc = QuantumCircuit(3)
        qc.rz(np.pi / 2, 0)
        qc.rx(np.pi / 4, 0)
        qc.rz(np.pi / 4, 0)

        qc.rz(np.pi / 2, 1)
        qc.rx(np.pi / 4, 1)
        qc.rz(np.pi / 2, 1)

        qc.rz(np.pi / 2, 2)
        qc.rx(np.pi / 4, 2)
        qc.rz(np.pi / 4, 2)

        qc_ref = QuantumCircuit(3)
        qc_ref.rz(np.pi / 2, 1)
        qc_ref.rx(np.pi / 4, 1)
        qc_ref.rz(np.pi / 4, 1)

        qc_ref.rz(np.pi / 2, 0)
        qc_ref.rx(np.pi / 4, 0)
        qc_ref.rz(np.pi / 2, 0)

        qc_ref.rz(np.pi / 2, 2)
        qc_ref.rx(np.pi / 4, 2)
        qc_ref.rz(np.pi / 4, 2)

        reference_output = input_vec.evolve(qc_ref)

        # Transpile!
        bw_pattern, output_ref, col_map = brickwork_transpiler.transpile(qc, input_vec)
        visualiser.plot_brickwork_graph_from_pattern(bw_pattern,
                                                     node_colours=col_map,
                                                     use_node_colours=True,
                                                     title="Brickwork Graph: test_three_symmetric_rows_arbitrary_rotations")
        # Simulate the generated pattern
        outstate = bw_pattern.simulate_pattern(backend='statevector').flatten()

        # change order qiskit ref state to match Graphix'
        # output_ref_graphix_order = reorder_via_transpose_n(output_ref.data)

        # Compare output up to global phase
        assert utils.assert_equal_up_to_global_phase(outstate, reference_output.data)

    def test_three_updif_arbitrary_rotations_full_brick(self):
        # 1) Create the |++> state directly
        input_vec = Statevector.from_label('+++')  # two-qubit plus state

        # 2) Define your 2-qubit circuit (no H gates needed)
        qc = QuantumCircuit(3)
        qc.rz(np.pi / 2, 0)
        qc.rx(np.pi / 4, 0)
        qc.rz(np.pi / 4, 0)

        qc.rz(np.pi / 2, 1)
        qc.rx(np.pi / 4, 1)
        qc.rz(np.pi / 2, 1)

        qc.rz(np.pi / 2, 2)
        qc.rx(np.pi / 4, 2)
        qc.rz(np.pi / 2, 2)

        qc_ref = QuantumCircuit(3)

        qc_ref.rz(np.pi / 2, 1)
        qc_ref.rx(np.pi / 4, 1)
        qc_ref.rz(np.pi / 4, 1)

        qc_ref.rz(np.pi / 2, 0)
        qc_ref.rx(np.pi / 4, 0)
        qc_ref.rz(np.pi / 2, 0)

        qc_ref.rz(np.pi / 2, 2)
        qc_ref.rx(np.pi / 4, 2)
        qc_ref.rz(np.pi / 2, 2)

        reference_output = input_vec.evolve(qc_ref)

        # Transpile!
        bw_pattern, output_ref, col_map = brickwork_transpiler.transpile(qc, input_vec)
        visualiser.plot_brickwork_graph_from_pattern(bw_pattern,
                                                     node_colours=col_map,
                                                     use_node_colours=True,
                                                     title="Brickwork Graph: test_three_updif_arbitrary_rotations_full_brick")
        # Simulate the generated pattern
        outstate = bw_pattern.simulate_pattern(backend='statevector').flatten()

        # change order qiskit ref state to match Graphix'
        # output_ref_graphix_order = reorder_via_transpose(output_ref.data)

        # Compare output up to global phase
        assert utils.assert_equal_up_to_global_phase(outstate, reference_output.data)

    def test_four_arbitrary_rotations_full_same_bricks(self):

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

        qc_perm = permute_qubits(qc, permutation=[2, 0, 3, 1])
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



