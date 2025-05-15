import unittest

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from src.brickwork_transpiler import decomposer, visualiser, graph_builder, pattern_converter, utils, \
    brickwork_transpiler


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

    # def test_euler_rotation_id_full_brick(self):
    #     # 1) Create the |++> state directly
    #     input_vec = Statevector.from_label('++')  # two-qubit plus state
    #
    #     # 2) Define your 2-qubit circuit (no H gates needed)
    #     qc = QuantumCircuit(2)
    #     qc.h(0)
    #
    #     # Transpile!
    #     bw_pattern, output_ref, col_map = brickwork_transpiler.transpile(qc, input_vec)
    #     visualiser.plot_brickwork_graph_from_pattern(bw_pattern,
    #                                                  node_colours=col_map,
    #                                                  use_node_colours=True,
    #                                                  title="Brickwork Graph: test_euler_rotation_id_full_brick")
    #     # Simulate the generated pattern
    #     outstate = bw_pattern.simulate_pattern(backend='statevector').flatten()
    #
    #     # Compare output up to global phase
    #     assert utils.assert_equal_up_to_global_phase(outstate, output_ref.data)

    def test_euler_rotation_full_brick(self):
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
                                                     title="Brickwork Graph: test_euler_rotation_full_brick")
        # Simulate the generated pattern
        outstate = bw_pattern.simulate_pattern(backend='statevector').flatten()

        # Compare output up to global phase
        assert utils.assert_equal_up_to_global_phase(outstate, output_ref.data)


