import itertools
import unittest
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from src.brickwork_transpiler import visualiser, utils, brickwork_transpiler
from src.brickwork_transpiler.utils import get_qubit_entries, calculate_ref_state_from_qiskit_circuit


class TestSingleGates(unittest.TestCase):

    def test_single_CX_top_bottom(self):

        # Initialise to |++>
        input_vec = Statevector.from_label('++')  # two-qubit plus state

        # Define quantum circuit
        qc = QuantumCircuit(2)
        qc.cx(0, 1)

        # Transpile!
        bw_pattern, output_ref, col_map = brickwork_transpiler.transpile(qc, input_vec)
        ref_state = calculate_ref_state_from_qiskit_circuit(bw_pattern, qc, input_vec)
        visualiser.plot_brickwork_graph_from_pattern(bw_pattern,
                                                     node_colours=col_map,
                                                     use_node_colours=True,
                                                     title="Brickwork Graph: test_single_CX_top_bottom")

        # Simulate the generated pattern
        outstate = bw_pattern.simulate_pattern(backend='statevector').flatten()

        # Compare output up to global phase
        assert utils.assert_equal_up_to_global_phase(outstate, ref_state.data)


    def test_single_CX_bottom_top(self):

        # Initialise to |++>
        input_vec = Statevector.from_label('++')  # two-qubit plus state

        # Define quantum circuit
        qc = QuantumCircuit(2)
        qc.cx(1, 0)

        # Transpile!
        bw_pattern, output_ref, col_map = brickwork_transpiler.transpile(qc, input_vec)
        ref_state = calculate_ref_state_from_qiskit_circuit(bw_pattern, qc, input_vec)
        visualiser.plot_brickwork_graph_from_pattern(bw_pattern,
                                                     node_colours=col_map,
                                                     use_node_colours=True,
                                                     title="Brickwork Graph: test_single_CX_bottom_top")
        # Simulate the generated pattern
        outstate = bw_pattern.simulate_pattern(backend='statevector').flatten()

        # Compare output up to global phase
        assert utils.assert_equal_up_to_global_phase(outstate, ref_state.data)

    def test_single_CX_target_bot_input_diff(self):
        # Initialise to |++>
        input_vec = Statevector.from_label('++')  # two-qubit plus state

        # Define quantum circuit
        qc = QuantumCircuit(2)
        qc.rz(np.pi, 0)     # |-+>
        qc.cx(0, 1)

        # Transpile!
        bw_pattern, output_ref, col_map = brickwork_transpiler.transpile(qc, input_vec)
        ref_state = calculate_ref_state_from_qiskit_circuit(bw_pattern, qc, input_vec)
        visualiser.plot_brickwork_graph_from_pattern(bw_pattern,
                                                     node_colours=col_map,
                                                     use_node_colours=True,
                                                     title="Brickwork Graph: test_single_CX_target_bot_input_diff")

        # Simulate the generated pattern
        outstate = bw_pattern.simulate_pattern(backend='statevector').flatten()

        # Compare output up to global phase
        assert utils.assert_equal_up_to_global_phase(outstate, ref_state.data)

    def test_merge_rotations(self):
        # Initialise to |++>
        input_vec = Statevector.from_label('++')  # two-qubit plus state

        # Define quantum circuit
        qc = QuantumCircuit(2)
        qc.rz(np.pi/2, 0)
        qc.rx(np.pi/3, 0)
        qc.rz(np.pi/3, 0)

        qc.rz(np.pi/6, 1)
        qc.rx(np.pi/7, 1)
        qc.rz(np.pi/3, 1)

        qc.rz(np.pi/9, 0)
        qc.rx(np.pi/2, 0)
        qc.rz(np.pi, 0)

        qc.rz(np.pi/4, 1)
        qc.rx(np.pi/2, 1)
        qc.rz(np.pi/4, 1)

        qc.h(0)
        qc.h(1)

        # Transpile!
        bw_pattern, output_ref, col_map = brickwork_transpiler.transpile(qc, input_vec)
        ref_state = calculate_ref_state_from_qiskit_circuit(bw_pattern, qc, input_vec)
        visualiser.plot_brickwork_graph_from_pattern(bw_pattern,
                                                     node_colours=col_map,
                                                     use_node_colours=True,
                                                     title="Brickwork Graph: test_merge_rotations")
        # Simulate the generated pattern
        outstate = bw_pattern.simulate_pattern(backend='statevector').flatten()

        # Compare output up to global phase
        assert utils.assert_equal_up_to_global_phase(outstate, ref_state.data)

    def test_merge_rotations_before_cx(self):
        # Initialise to |++>
        input_vec = Statevector.from_label('++')  # two-qubit plus state

        # Define quantum circuit
        qc = QuantumCircuit(2)
        qc.rz(np.pi / 2, 0)
        qc.rx(np.pi / 3, 0)
        qc.rz(np.pi / 3, 0)

        qc.rz(np.pi / 6, 1)
        qc.rx(np.pi / 7, 1)
        qc.rz(np.pi / 3, 1)

        qc.rz(np.pi / 9, 0)
        qc.rx(np.pi / 2, 0)
        qc.rz(np.pi, 0)

        qc.rz(np.pi / 4, 1)
        qc.rx(np.pi / 2, 1)
        qc.rz(np.pi / 4, 1)

        qc.h(0)
        qc.h(1)

        qc.cx(1, 0)

        # Transpile!
        bw_pattern, output_ref, col_map = brickwork_transpiler.transpile(qc, input_vec)
        ref_state = calculate_ref_state_from_qiskit_circuit(bw_pattern, qc, input_vec)
        visualiser.plot_brickwork_graph_from_pattern(bw_pattern,
                                                     node_colours=col_map,
                                                     use_node_colours=True,
                                                     title="Brickwork Graph: test_merge_rotations_before_cx")
        # Simulate the generated pattern
        outstate = bw_pattern.simulate_pattern(backend='statevector').flatten()

        # Compare output up to global phase
        assert utils.assert_equal_up_to_global_phase(outstate, ref_state.data)

    def test_merge_rotations_after_cx(self):
        # Initialise to |++>
        input_vec = Statevector.from_label('++')  # two-qubit plus state

        # Define quantum circuit
        qc = QuantumCircuit(2)
        qc.cx(1, 0)

        qc.rz(np.pi / 2, 0)
        qc.rx(np.pi / 3, 0)
        qc.rz(np.pi / 3, 0)

        qc.rz(np.pi / 6, 1)
        qc.rx(np.pi / 7, 1)
        qc.rz(np.pi / 3, 1)

        qc.rz(np.pi / 9, 0)
        qc.rx(np.pi / 2, 0)
        qc.rz(np.pi, 0)

        qc.rz(np.pi / 4, 1)
        qc.rx(np.pi / 2, 1)
        qc.rz(np.pi / 4, 1)

        qc.h(0)
        qc.h(1)

        # Transpile!
        bw_pattern, output_ref, col_map = brickwork_transpiler.transpile(qc, input_vec)
        ref_state = calculate_ref_state_from_qiskit_circuit(bw_pattern, qc, input_vec)
        visualiser.plot_brickwork_graph_from_pattern(bw_pattern,
                                                     node_colours=col_map,
                                                     use_node_colours=True,
                                                     title="Brickwork Graph: test_merge_rotations_after_cx")
        # Simulate the generated pattern
        outstate = bw_pattern.simulate_pattern(backend='statevector').flatten()

        # Compare output up to global phase
        assert utils.assert_equal_up_to_global_phase(outstate, ref_state.data)

    def test_single_CX_target_top_input_diff(self):
        # Initialise to |++>
        input_vec = Statevector.from_label('++')  # two-qubit plus state

        # Define quantum circuit
        qc = QuantumCircuit(2)
        qc.rz(np.pi, 0)     # |-+>
        qc.cx(1, 0)

        # Transpile!
        bw_pattern, output_ref, col_map = brickwork_transpiler.transpile(qc, input_vec)
        ref_state = calculate_ref_state_from_qiskit_circuit(bw_pattern, qc, input_vec)
        visualiser.plot_brickwork_graph_from_pattern(bw_pattern,
                                                     node_colours=col_map,
                                                     use_node_colours=True,
                                                     title="Brickwork Graph: test_single_CX_target_top_input_diff")
        # Simulate the generated pattern
        outstate = bw_pattern.simulate_pattern(backend='statevector').flatten()

        # Compare output up to global phase
        assert utils.assert_equal_up_to_global_phase(outstate, ref_state.data)

    def test_hadamard_half_brick(self):
        # Initialise to |++>
        input_vec = Statevector.from_label('+')  # one-qubit plus state

        # Define quantum circuit
        qc = QuantumCircuit(1)
        qc.h(0)

        # Transpile!
        bw_pattern, output_ref, col_map = brickwork_transpiler.transpile(qc, input_vec)
        ref_state = calculate_ref_state_from_qiskit_circuit(bw_pattern, qc, input_vec)
        visualiser.plot_brickwork_graph_from_pattern(bw_pattern,
                                                     node_colours=col_map,
                                                     use_node_colours=True,
                                                     title="Brickwork Graph: test_euler_rotation_half_brick")
        # Simulate the generated pattern
        outstate = bw_pattern.simulate_pattern(backend='statevector').flatten()

        # Compare output up to global phase
        assert utils.assert_equal_up_to_global_phase(outstate, ref_state.data)

    def test_pauli_x_half_brick(self):
        # Initialise to |++>
        input_vec = Statevector.from_label('+')  # one-qubit plus state

        # Define quantum circuit
        qc = QuantumCircuit(1)
        qc.x(0)

        # Transpile!
        bw_pattern, output_ref, col_map = brickwork_transpiler.transpile(qc, input_vec)
        ref_state = calculate_ref_state_from_qiskit_circuit(bw_pattern, qc, input_vec)
        visualiser.plot_brickwork_graph_from_pattern(bw_pattern,
                                                     node_colours=col_map,
                                                     use_node_colours=True,
                                                     title="Brickwork Graph: test_euler_rotation_half_brick")
        # Simulate the generated pattern
        outstate = bw_pattern.simulate_pattern(backend='statevector').flatten()

        # Compare output up to global phase
        assert utils.assert_equal_up_to_global_phase(outstate, ref_state.data)

    def test_euler_rotation_half_brick(self):
        # Initialise to |++>
        input_vec = Statevector.from_label('+')  # one-qubit plus state

        # Define quantum circuit
        qc = QuantumCircuit(1)
        qc.rx(np.pi/3, 0)
        qc.rx(np.pi / 4, 0)
        qc.rz(np.pi / 5, 0)
        qc.rx(np.pi / 2, 0)
        qc.rz(np.pi / 5, 0)

        # Transpile!
        bw_pattern, output_ref, col_map = brickwork_transpiler.transpile(qc, input_vec)
        ref_state = calculate_ref_state_from_qiskit_circuit(bw_pattern, qc, input_vec)
        visualiser.plot_brickwork_graph_from_pattern(bw_pattern,
                                                     node_colours=col_map,
                                                     use_node_colours=True,
                                                     title="Brickwork Graph: test_euler_rotation_half_brick")
        # Simulate the generated pattern
        outstate = bw_pattern.simulate_pattern(backend='statevector').flatten()

        # Compare output up to global phase
        assert utils.assert_equal_up_to_global_phase(outstate, ref_state.data)

    def test_euler_rotation_id_full_brick(self):
        # Initialise to |++>
        input_vec = Statevector.from_label('++')  # two-qubit plus state

        # Define quantum circuit
        qc = QuantumCircuit(2)
        qc.h(0)

        # Transpile!
        bw_pattern, output_ref, col_map = brickwork_transpiler.transpile(qc, input_vec)
        ref_state = calculate_ref_state_from_qiskit_circuit(bw_pattern, qc, input_vec)
        visualiser.plot_brickwork_graph_from_pattern(bw_pattern,
                                                     node_colours=col_map,
                                                     use_node_colours=True,
                                                     title="Brickwork Graph: test_euler_rotation_id_full_brick")
        # Simulate the generated pattern
        outstate = bw_pattern.simulate_pattern(backend='statevector').flatten()


        # Compare output up to global phase
        assert utils.assert_equal_up_to_global_phase(outstate, ref_state.data)

class TestmultipleGates(unittest.TestCase):

    def test_both_H_full_brick(self):
        # Initialise to |++>
        input_vec = Statevector.from_label('++')  # two-qubit plus state

        # Define quantum circuit
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.h(1)

        # Transpile!
        bw_pattern, output_ref, col_map = brickwork_transpiler.transpile(qc, input_vec)
        ref_state = calculate_ref_state_from_qiskit_circuit(bw_pattern, qc, input_vec)
        visualiser.plot_brickwork_graph_from_pattern(bw_pattern,
                                                     node_colours=col_map,
                                                     use_node_colours=True,
                                                     title="Brickwork Graph: test_both_H_full_brick")
        # Simulate the generated pattern
        outstate = bw_pattern.simulate_pattern(backend='statevector').flatten()

        # Compare output up to global phase
        assert utils.assert_equal_up_to_global_phase(outstate, ref_state.data)

    def test_different_arbitrary_rotations_full_brick(self):
        # Initialise to |++>
        input_vec = Statevector.from_label('++')  # two-qubit plus state

        # Define quantum circuit
        qc = QuantumCircuit(2)
        qc.rz(np.pi / 2, 0)
        qc.rx(np.pi / 4, 0)
        qc.rz(np.pi / 2, 0)

        qc.rz(np.pi / 2, 1)
        qc.rx(np.pi / 3, 1)
        qc.rz(np.pi / 2, 1)

        # Transpile!
        bw_pattern, output_ref, col_map = brickwork_transpiler.transpile(qc, input_vec)
        ref_state = calculate_ref_state_from_qiskit_circuit(bw_pattern, qc, input_vec)
        visualiser.plot_brickwork_graph_from_pattern(bw_pattern,
                                                     node_colours=col_map,
                                                     use_node_colours=True,
                                                     title="Brickwork Graph: test_different_arbitrary_rotations_full_brick")
        # Simulate the generated pattern
        outstate = bw_pattern.simulate_pattern(backend='statevector').flatten()

        # Compare output up to global phase
        assert utils.assert_equal_up_to_global_phase(outstate, ref_state.data)

    def test_three_same_arbitrary_rotations_full_brick(self):
        # Initialise to |+++>
        input_vec = Statevector.from_label('+++')  # two-qubit plus state

        # Define Quantum circuit
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
        ref_state = calculate_ref_state_from_qiskit_circuit(bw_pattern, qc, input_vec)
        visualiser.plot_brickwork_graph_from_pattern(bw_pattern,
                                                     node_colours=col_map,
                                                     use_node_colours=True,
                                                     title="Brickwork Graph: test_three_same_arbitrary_rotations_full_brick")
        # Simulate the generated pattern
        outstate = bw_pattern.simulate_pattern(backend='statevector').flatten()


        # Compare output up to global phase
        assert utils.assert_equal_up_to_global_phase(outstate, ref_state.data)


    def test_three_different_arbitrary_rotations_full_brick(self):
        # Initialise to |+++>
        input_vec = Statevector.from_label('+++')  # two-qubit plus state

        # Define quantum circuit
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

        # Transpile!
        bw_pattern, output_ref, col_map = brickwork_transpiler.transpile(qc, input_vec)
        ref_state = calculate_ref_state_from_qiskit_circuit(bw_pattern, qc, input_vec)
        visualiser.plot_brickwork_graph_from_pattern(bw_pattern,
                                                     node_colours=col_map,
                                                     use_node_colours=True,
                                                     title="Brickwork Graph: test_three_different_arbitrary_rotations_full_brick")

        outstate = bw_pattern.simulate_pattern(backend='statevector').flatten()

        print(f"outstate: {outstate}")
        print(f"ref state: {ref_state.data}")

        # Compare output up to global phase
        assert utils.assert_equal_up_to_global_phase(outstate, ref_state.data)

    def test_three_symmetric_rows_arbitrary_rotations(self):
        # Initialise to |+++>
        input_vec = Statevector.from_label('+++')  # two-qubit plus state

        # Define circuit
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

        bw_pattern, output_ref, col_map = brickwork_transpiler.transpile(qc, input_vec)
        ref_state = calculate_ref_state_from_qiskit_circuit(bw_pattern, qc, input_vec)
        visualiser.plot_brickwork_graph_from_pattern(bw_pattern,
                                                     node_colours=col_map,
                                                     use_node_colours=True,
                                                     title="Brickwork Graph: test_three_symmetric_rows_arbitrary_rotations")
        # Simulate the generated pattern
        outstate = bw_pattern.simulate_pattern(backend='statevector').flatten()

        # Compare output up to global phase
        assert utils.assert_equal_up_to_global_phase(outstate, ref_state.data)

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

        bw_pattern, output_ref, col_map = brickwork_transpiler.transpile(qc, input_vec)
        ref_state = calculate_ref_state_from_qiskit_circuit(bw_pattern, qc, input_vec)
        visualiser.plot_brickwork_graph_from_pattern(bw_pattern,
                                                     node_colours=col_map,
                                                     use_node_colours=True,
                                                     title="Brickwork Graph: test_three_updif_arbitrary_rotations_full_brick")

        outstate = bw_pattern.simulate_pattern(backend='statevector').flatten()

        assert utils.assert_equal_up_to_global_phase(outstate, ref_state.data)

    def test_four_arbitrary_rotations_full_arb_bricks(self):

        input_vec = Statevector.from_label('++++')  # two-qubit plus state

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


        bw_pattern, output_refQQQQ, col_map = brickwork_transpiler.transpile(qc, input_vec)
        ref_state = calculate_ref_state_from_qiskit_circuit(bw_pattern, qc, input_vec)
        visualiser.plot_brickwork_graph_from_pattern(bw_pattern,
                                                     node_colours=col_map,
                                                     use_node_colours=True,
                                                     title="Brickwork Graph: test_four_arbitrary_rotations_full_arb_bricks")

        my_outstate = bw_pattern.simulate_pattern(backend='statevector').flatten()

        # Compare output up to global phase
        assert utils.assert_equal_up_to_global_phase(ref_state.data, my_outstate)

    def test_five_arbitrary_rotations_full_arb_bricks(self):
        input_vec = Statevector.from_label('+++++')  # two-qubit plus state

        qc = QuantumCircuit(5)
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

        qc.rz(np.pi / 7, 4)
        qc.rx(np.pi / 7, 4)
        qc.rz(np.pi / 7, 4)

        bw_pattern, output_refQQQQ, col_map = brickwork_transpiler.transpile(qc, input_vec)
        ref_state = calculate_ref_state_from_qiskit_circuit(bw_pattern, qc, input_vec)
        visualiser.plot_brickwork_graph_from_pattern(bw_pattern,
                                                     node_colours=col_map,
                                                     use_node_colours=True,
                                                     title="Brickwork Graph: test_five_arbitrary_rotations_full_arb_bricks")

        my_outstate = bw_pattern.simulate_pattern(backend='statevector').flatten()

        # print(f"my_state: {my_outstate}")
        # print(f"ref_state: {ref_state.data}")

        # Compare output up to global phase
        assert utils.assert_equal_up_to_global_phase(ref_state.data, my_outstate)

    # Takes long to run
    # def test_six_arbitrary_rotations_full_arb_bricks(self):
    #     input_vec = Statevector.from_label('++++++')  # two-qubit plus state
    #
    #     qc = QuantumCircuit(6)
    #     qc.rz(np.pi / 4, 0)
    #     qc.rx(np.pi / 4, 0)
    #     qc.rz(np.pi / 5, 0)
    #
    #     qc.rz(np.pi / 3, 1)
    #     qc.rx(np.pi / 4, 1)
    #     qc.rz(np.pi / 4, 1)
    #
    #     qc.rz(np.pi / 4, 2)
    #     qc.rx(np.pi / 4, 2)
    #     qc.rz(np.pi / 2, 2)
    #
    #     qc.rz(np.pi / 4, 3)
    #     qc.rx(np.pi / 4, 3)
    #     qc.rz(np.pi / 4, 3)
    #
    #     qc.rz(np.pi / 7, 4)
    #     qc.rx(np.pi / 7, 4)
    #     qc.rz(np.pi / 7, 4)
    #
    #     qc.rz(np.pi / 3, 5)
    #     qc.rx(np.pi / 4, 5)
    #     qc.rz(np.pi / 5, 5)
    #
    #     bw_pattern, output_refQQQQ, col_map = brickwork_transpiler.transpile(qc, input_vec)
    #     ref_state = calculate_ref_state_from_qiskit_circuit(bw_pattern, qc, input_vec)
    #     visualiser.plot_brickwork_graph_from_pattern(bw_pattern,
    #                                                  node_colours=col_map,
    #                                                  use_node_colours=True,
    #                                                  title="Brickwork Graph: test_six_arbitrary_rotations_full_arb_bricks")
    #
    #     my_outstate = bw_pattern.simulate_pattern(backend='statevector').flatten()
    #
    #     # Compare output up to global phase
    #     assert utils.assert_equal_up_to_global_phase(ref_state.data, my_outstate)
