import itertools
import unittest
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from src.brickwork_transpiler import visualiser, utils, brickwork_transpiler
from src.brickwork_transpiler.utils import get_qubit_entries, calculate_ref_state_from_qiskit_circuit, \
    extract_logical_to_physical, undo_layout_on_state


class TestSingleGates(unittest.TestCase):

    def test_single_CX_top_bottom(self):

        # Initialise to |++>
        input_vec = Statevector.from_label('++')  # two-qubit plus state

        # Define quantum circuit
        qc = QuantumCircuit(2)
        qc.cx(0, 1)

        bw_pattern, col_map, transpiled_qc = brickwork_transpiler.transpile(qc, input_vec,
                                                                            routing_method="sabre",
                                                                            layout_method="sabre",
                                                                            with_ancillas=False)
        # ref_state = calculate_ref_state_from_qiskit_circuit(bw_pattern, qc, input_vec)
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

        # ref_state = calculate_ref_state_from_qiskit_circuit(bw_pattern, qc, input_vec)

        mapping = extract_logical_to_physical(qc, transpiled_qc)

        sv_phys = Statevector(input_vec).evolve(transpiled_qc)
        undo_layout_on_state(sv_phys, mapping)

        # If you simulated with your MBQC engine and got a flat numpy array `psi`:
        sv_logical_from_mbqc = undo_layout_on_state(psi, mapping, total_qubits=transpiled_qc.num_qubits)

        # Compare output up to global phase
        assert utils.assert_equal_up_to_global_phase(sv_logical_from_mbqc.data, psi)


    def test_single_CX_bottom_top(self):

        # Initialise to |++>
        input_vec = Statevector.from_label('++')  # two-qubit plus state

        # Define quantum circuit
        qc = QuantumCircuit(2)
        qc.cx(1, 0)

        bw_pattern, col_map, transpiled_qc = brickwork_transpiler.transpile(qc, input_vec,
                                                                            routing_method="sabre",
                                                                            layout_method="sabre",
                                                                            with_ancillas=False)
        # ref_state = calculate_ref_state_from_qiskit_circuit(bw_pattern, qc, input_vec)
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

        # ref_state = calculate_ref_state_from_qiskit_circuit(bw_pattern, qc, input_vec)

        mapping = extract_logical_to_physical(qc, transpiled_qc)

        sv_phys = Statevector(input_vec).evolve(transpiled_qc)
        undo_layout_on_state(sv_phys, mapping)

        # If you simulated with your MBQC engine and got a flat numpy array `psi`:
        sv_logical_from_mbqc = undo_layout_on_state(psi, mapping, total_qubits=transpiled_qc.num_qubits)

        # Compare output up to global phase
        assert utils.assert_equal_up_to_global_phase(sv_logical_from_mbqc.data, psi)

    def test_single_CX_target_bot_input_diff(self):
        # Initialise to |++>
        input_vec = Statevector.from_label('++')  # two-qubit plus state

        # Define quantum circuit
        qc = QuantumCircuit(2)
        qc.rz(np.pi, 0)     # |-+>
        qc.cx(0, 1)

        bw_pattern, col_map, transpiled_qc = brickwork_transpiler.transpile(qc, input_vec,
                                                                            routing_method="sabre",
                                                                            layout_method="sabre",
                                                                            with_ancillas=False)
        # ref_state = calculate_ref_state_from_qiskit_circuit(bw_pattern, qc, input_vec)
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

        # ref_state = calculate_ref_state_from_qiskit_circuit(bw_pattern, qc, input_vec)

        mapping = extract_logical_to_physical(qc, transpiled_qc)

        sv_phys = Statevector(input_vec).evolve(transpiled_qc)
        undo_layout_on_state(sv_phys, mapping)

        # If you simulated with your MBQC engine and got a flat numpy array `psi`:
        sv_logical_from_mbqc = undo_layout_on_state(psi, mapping, total_qubits=transpiled_qc.num_qubits)

        # Compare output up to global phase
        assert utils.assert_equal_up_to_global_phase(sv_logical_from_mbqc.data, psi)
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

        bw_pattern, col_map, transpiled_qc = brickwork_transpiler.transpile(qc, input_vec,
                                                                            routing_method="sabre",
                                                                            layout_method="sabre",
                                                                            with_ancillas=False)
        # ref_state = calculate_ref_state_from_qiskit_circuit(bw_pattern, qc, input_vec)
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

        # ref_state = calculate_ref_state_from_qiskit_circuit(bw_pattern, qc, input_vec)

        mapping = extract_logical_to_physical(qc, transpiled_qc)

        sv_phys = Statevector(input_vec).evolve(transpiled_qc)
        undo_layout_on_state(sv_phys, mapping)

        # If you simulated with your MBQC engine and got a flat numpy array `psi`:
        sv_logical_from_mbqc = undo_layout_on_state(psi, mapping, total_qubits=transpiled_qc.num_qubits)

        # Compare output up to global phase
        assert utils.assert_equal_up_to_global_phase(sv_logical_from_mbqc.data, psi)

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

        bw_pattern, col_map, transpiled_qc = brickwork_transpiler.transpile(qc, input_vec,
                                                                            routing_method="sabre",
                                                                            layout_method="sabre",
                                                                            with_ancillas=False)
        # ref_state = calculate_ref_state_from_qiskit_circuit(bw_pattern, qc, input_vec)
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

        # ref_state = calculate_ref_state_from_qiskit_circuit(bw_pattern, qc, input_vec)

        mapping = extract_logical_to_physical(qc, transpiled_qc)

        sv_phys = Statevector(input_vec).evolve(transpiled_qc)
        undo_layout_on_state(sv_phys, mapping)

        # If you simulated with your MBQC engine and got a flat numpy array `psi`:
        sv_logical_from_mbqc = undo_layout_on_state(psi, mapping, total_qubits=transpiled_qc.num_qubits)

        # Compare output up to global phase
        assert utils.assert_equal_up_to_global_phase(sv_logical_from_mbqc.data, psi)

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

        bw_pattern, col_map, transpiled_qc = brickwork_transpiler.transpile(qc, input_vec,
                                                                            routing_method="sabre",
                                                                            layout_method="sabre",
                                                                            with_ancillas=False)
        # ref_state = calculate_ref_state_from_qiskit_circuit(bw_pattern, qc, input_vec)
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

        # ref_state = calculate_ref_state_from_qiskit_circuit(bw_pattern, qc, input_vec)

        mapping = extract_logical_to_physical(qc, transpiled_qc)

        sv_phys = Statevector(input_vec).evolve(transpiled_qc)
        undo_layout_on_state(sv_phys, mapping)

        # If you simulated with your MBQC engine and got a flat numpy array `psi`:
        sv_logical_from_mbqc = undo_layout_on_state(psi, mapping, total_qubits=transpiled_qc.num_qubits)

        # Compare output up to global phase
        assert utils.assert_equal_up_to_global_phase(sv_logical_from_mbqc.data, psi)

    def test_single_CX_target_top_input_diff(self):
        # Initialise to |++>
        input_vec = Statevector.from_label('++')  # two-qubit plus state

        # Define quantum circuit
        qc = QuantumCircuit(2)
        qc.rz(np.pi, 0)     # |-+>
        qc.cx(1, 0)

        bw_pattern, col_map, transpiled_qc = brickwork_transpiler.transpile(qc, input_vec,
                                                                            routing_method="sabre",
                                                                            layout_method="sabre",
                                                                            with_ancillas=False)
        # ref_state = calculate_ref_state_from_qiskit_circuit(bw_pattern, qc, input_vec)
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

        # ref_state = calculate_ref_state_from_qiskit_circuit(bw_pattern, qc, input_vec)

        mapping = extract_logical_to_physical(qc, transpiled_qc)

        sv_phys = Statevector(input_vec).evolve(transpiled_qc)
        undo_layout_on_state(sv_phys, mapping)

        # If you simulated with your MBQC engine and got a flat numpy array `psi`:
        sv_logical_from_mbqc = undo_layout_on_state(psi, mapping, total_qubits=transpiled_qc.num_qubits)

        # Compare output up to global phase
        assert utils.assert_equal_up_to_global_phase(sv_logical_from_mbqc.data, psi)


    def test_hadamard_half_brick(self):
        # Initialise to |++>
        input_vec = Statevector.from_label('+')  # one-qubit plus state

        # Define quantum circuit
        qc = QuantumCircuit(1)
        qc.h(0)

        bw_pattern, col_map, transpiled_qc = brickwork_transpiler.transpile(qc, input_vec,
                                                                            routing_method="sabre",
                                                                            layout_method="sabre",
                                                                            with_ancillas=False)
        # ref_state = calculate_ref_state_from_qiskit_circuit(bw_pattern, qc, input_vec)
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

        # ref_state = calculate_ref_state_from_qiskit_circuit(bw_pattern, qc, input_vec)

        mapping = extract_logical_to_physical(qc, transpiled_qc)

        sv_phys = Statevector(input_vec).evolve(transpiled_qc)
        undo_layout_on_state(sv_phys, mapping)

        # If you simulated with your MBQC engine and got a flat numpy array `psi`:
        sv_logical_from_mbqc = undo_layout_on_state(psi, mapping, total_qubits=transpiled_qc.num_qubits)

        # Compare output up to global phase
        assert utils.assert_equal_up_to_global_phase(sv_logical_from_mbqc.data, psi)

    def test_pauli_x_half_brick(self):
        # Initialise to |++>
        input_vec = Statevector.from_label('+')  # one-qubit plus state

        # Define quantum circuit
        qc = QuantumCircuit(1)
        qc.x(0)

        bw_pattern, col_map, transpiled_qc = brickwork_transpiler.transpile(qc, input_vec,
                                                                            routing_method="sabre",
                                                                            layout_method="sabre",
                                                                            with_ancillas=False)
        # ref_state = calculate_ref_state_from_qiskit_circuit(bw_pattern, qc, input_vec)
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

        # ref_state = calculate_ref_state_from_qiskit_circuit(bw_pattern, qc, input_vec)

        mapping = extract_logical_to_physical(qc, transpiled_qc)

        sv_phys = Statevector(input_vec).evolve(transpiled_qc)
        undo_layout_on_state(sv_phys, mapping)

        # If you simulated with your MBQC engine and got a flat numpy array `psi`:
        sv_logical_from_mbqc = undo_layout_on_state(psi, mapping, total_qubits=transpiled_qc.num_qubits)

        # Compare output up to global phase
        assert utils.assert_equal_up_to_global_phase(sv_logical_from_mbqc.data, psi)

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

        bw_pattern, col_map, transpiled_qc = brickwork_transpiler.transpile(qc, input_vec,
                                                                            routing_method="sabre",
                                                                            layout_method="sabre",
                                                                            with_ancillas=False)
        # ref_state = calculate_ref_state_from_qiskit_circuit(bw_pattern, qc, input_vec)
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

        # ref_state = calculate_ref_state_from_qiskit_circuit(bw_pattern, qc, input_vec)

        mapping = extract_logical_to_physical(qc, transpiled_qc)

        sv_phys = Statevector(input_vec).evolve(transpiled_qc)
        undo_layout_on_state(sv_phys, mapping)

        # If you simulated with your MBQC engine and got a flat numpy array `psi`:
        sv_logical_from_mbqc = undo_layout_on_state(psi, mapping, total_qubits=transpiled_qc.num_qubits)

        # Compare output up to global phase
        assert utils.assert_equal_up_to_global_phase(sv_logical_from_mbqc.data, psi)

    def test_euler_rotation_id_full_brick(self):
        # Initialise to |++>
        input_vec = Statevector.from_label('++')  # two-qubit plus state

        # Define quantum circuit
        qc = QuantumCircuit(2)
        qc.h(0)

        bw_pattern, col_map, transpiled_qc = brickwork_transpiler.transpile(qc, input_vec,
                                                                            routing_method="sabre",
                                                                            layout_method="sabre",
                                                                            with_ancillas=False)
        # ref_state = calculate_ref_state_from_qiskit_circuit(bw_pattern, qc, input_vec)
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

        # ref_state = calculate_ref_state_from_qiskit_circuit(bw_pattern, qc, input_vec)

        mapping = extract_logical_to_physical(qc, transpiled_qc)

        sv_phys = Statevector(input_vec).evolve(transpiled_qc)
        undo_layout_on_state(sv_phys, mapping)

        # If you simulated with your MBQC engine and got a flat numpy array `psi`:
        sv_logical_from_mbqc = undo_layout_on_state(psi, mapping, total_qubits=transpiled_qc.num_qubits)

        # Compare output up to global phase
        assert utils.assert_equal_up_to_global_phase(sv_logical_from_mbqc.data, psi)

class TestmultipleGates(unittest.TestCase):

    def test_both_H_full_brick(self):
        # Initialise to |++>
        input_vec = Statevector.from_label('++')  # two-qubit plus state

        # Define quantum circuit
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.h(1)

        bw_pattern, col_map, transpiled_qc = brickwork_transpiler.transpile(qc, input_vec,
                                                                            routing_method="sabre",
                                                                            layout_method="sabre",
                                                                            with_ancillas=False)
        # ref_state = calculate_ref_state_from_qiskit_circuit(bw_pattern, qc, input_vec)
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

        # ref_state = calculate_ref_state_from_qiskit_circuit(bw_pattern, qc, input_vec)

        mapping = extract_logical_to_physical(qc, transpiled_qc)

        sv_phys = Statevector(input_vec).evolve(transpiled_qc)
        undo_layout_on_state(sv_phys, mapping)

        # If you simulated with your MBQC engine and got a flat numpy array `psi`:
        sv_logical_from_mbqc = undo_layout_on_state(psi, mapping, total_qubits=transpiled_qc.num_qubits)

        # Compare output up to global phase
        assert utils.assert_equal_up_to_global_phase(sv_logical_from_mbqc.data, psi)

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

        bw_pattern, col_map, transpiled_qc = brickwork_transpiler.transpile(qc, input_vec,
                                                                            routing_method="sabre",
                                                                            layout_method="sabre",
                                                                            with_ancillas=False)
        # ref_state = calculate_ref_state_from_qiskit_circuit(bw_pattern, qc, input_vec)
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

        # ref_state = calculate_ref_state_from_qiskit_circuit(bw_pattern, qc, input_vec)

        mapping = extract_logical_to_physical(qc, transpiled_qc)

        sv_phys = Statevector(input_vec).evolve(transpiled_qc)
        undo_layout_on_state(sv_phys, mapping)

        # If you simulated with your MBQC engine and got a flat numpy array `psi`:
        sv_logical_from_mbqc = undo_layout_on_state(psi, mapping, total_qubits=transpiled_qc.num_qubits)

        # Compare output up to global phase
        assert utils.assert_equal_up_to_global_phase(sv_logical_from_mbqc.data, psi)

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

        bw_pattern, col_map, transpiled_qc = brickwork_transpiler.transpile(qc, input_vec,
                                                                            routing_method="sabre",
                                                                            layout_method="sabre",
                                                                            with_ancillas=False)
        # ref_state = calculate_ref_state_from_qiskit_circuit(bw_pattern, qc, input_vec)
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

        # ref_state = calculate_ref_state_from_qiskit_circuit(bw_pattern, qc, input_vec)

        mapping = extract_logical_to_physical(qc, transpiled_qc)

        sv_phys = Statevector(input_vec).evolve(transpiled_qc)
        undo_layout_on_state(sv_phys, mapping)

        # If you simulated with your MBQC engine and got a flat numpy array `psi`:
        sv_logical_from_mbqc = undo_layout_on_state(psi, mapping, total_qubits=transpiled_qc.num_qubits)

        # Compare output up to global phase
        assert utils.assert_equal_up_to_global_phase(sv_logical_from_mbqc.data, psi)


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

        bw_pattern, col_map, transpiled_qc = brickwork_transpiler.transpile(qc, input_vec,
                                                                            routing_method="sabre",
                                                                            layout_method="sabre",
                                                                            with_ancillas=False)
        # ref_state = calculate_ref_state_from_qiskit_circuit(bw_pattern, qc, input_vec)
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

        # ref_state = calculate_ref_state_from_qiskit_circuit(bw_pattern, qc, input_vec)

        mapping = extract_logical_to_physical(qc, transpiled_qc)

        sv_phys = Statevector(input_vec).evolve(transpiled_qc)
        undo_layout_on_state(sv_phys, mapping)

        # If you simulated with your MBQC engine and got a flat numpy array `psi`:
        sv_logical_from_mbqc = undo_layout_on_state(psi, mapping, total_qubits=transpiled_qc.num_qubits)

        # Compare output up to global phase
        assert utils.assert_equal_up_to_global_phase(sv_logical_from_mbqc.data, psi)

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

        bw_pattern, col_map, transpiled_qc = brickwork_transpiler.transpile(qc, input_vec,
                                                                            routing_method="sabre",
                                                                            layout_method="sabre",
                                                                            with_ancillas=False)
        # ref_state = calculate_ref_state_from_qiskit_circuit(bw_pattern, qc, input_vec)
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

        # ref_state = calculate_ref_state_from_qiskit_circuit(bw_pattern, qc, input_vec)

        mapping = extract_logical_to_physical(qc, transpiled_qc)

        sv_phys = Statevector(input_vec).evolve(transpiled_qc)
        undo_layout_on_state(sv_phys, mapping)

        # If you simulated with your MBQC engine and got a flat numpy array `psi`:
        sv_logical_from_mbqc = undo_layout_on_state(psi, mapping, total_qubits=transpiled_qc.num_qubits)

        # Compare output up to global phase
        assert utils.assert_equal_up_to_global_phase(sv_logical_from_mbqc.data, psi)

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

        bw_pattern, col_map, transpiled_qc = brickwork_transpiler.transpile(qc, input_vec,
                                                                            routing_method="sabre",
                                                                            layout_method="sabre",
                                                                            with_ancillas=False)
        # ref_state = calculate_ref_state_from_qiskit_circuit(bw_pattern, qc, input_vec)
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

        # ref_state = calculate_ref_state_from_qiskit_circuit(bw_pattern, qc, input_vec)

        mapping = extract_logical_to_physical(qc, transpiled_qc)

        sv_phys = Statevector(input_vec).evolve(transpiled_qc)
        undo_layout_on_state(sv_phys, mapping)

        # If you simulated with your MBQC engine and got a flat numpy array `psi`:
        sv_logical_from_mbqc = undo_layout_on_state(psi, mapping, total_qubits=transpiled_qc.num_qubits)

        # Compare output up to global phase
        assert utils.assert_equal_up_to_global_phase(sv_logical_from_mbqc.data, psi)

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

        bw_pattern, col_map, transpiled_qc = brickwork_transpiler.transpile(qc, input_vec,
                                                                            routing_method="sabre",
                                                                            layout_method="sabre",
                                                                            with_ancillas=False)
        # ref_state = calculate_ref_state_from_qiskit_circuit(bw_pattern, qc, input_vec)
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

        # ref_state = calculate_ref_state_from_qiskit_circuit(bw_pattern, qc, input_vec)

        mapping = extract_logical_to_physical(qc, transpiled_qc)

        sv_phys = Statevector(input_vec).evolve(transpiled_qc)
        undo_layout_on_state(sv_phys, mapping)

        # If you simulated with your MBQC engine and got a flat numpy array `psi`:
        sv_logical_from_mbqc = undo_layout_on_state(psi, mapping, total_qubits=transpiled_qc.num_qubits)

        # Compare output up to global phase
        assert utils.assert_equal_up_to_global_phase(sv_logical_from_mbqc.data, psi)


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

        bw_pattern, col_map, transpiled_qc = brickwork_transpiler.transpile(qc, input_vec,
                                                                            routing_method="sabre",
                                                                            layout_method="sabre",
                                                                            with_ancillas=False)
        # ref_state = calculate_ref_state_from_qiskit_circuit(bw_pattern, qc, input_vec)
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

        # ref_state = calculate_ref_state_from_qiskit_circuit(bw_pattern, qc, input_vec)

        mapping = extract_logical_to_physical(qc, transpiled_qc)

        sv_phys = Statevector(input_vec).evolve(transpiled_qc)
        undo_layout_on_state(sv_phys, mapping)

        # If you simulated with your MBQC engine and got a flat numpy array `psi`:
        sv_logical_from_mbqc = undo_layout_on_state(psi, mapping, total_qubits=transpiled_qc.num_qubits)

        # Compare output up to global phase
        assert utils.assert_equal_up_to_global_phase(sv_logical_from_mbqc.data, psi)


    def test_six_arbitrary_rotations_full_arb_bricks(self):
        input_vec = Statevector.from_label('++++++')  # two-qubit plus state

        qc = QuantumCircuit(6)
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

        qc.rz(np.pi / 3, 5)
        qc.rx(np.pi / 4, 5)
        qc.rz(np.pi / 5, 5)

        bw_pattern, col_map, transpiled_qc = brickwork_transpiler.transpile(qc, input_vec,
                                                             routing_method="sabre",
                                                             layout_method="sabre",
                                                             with_ancillas=False)
        # ref_state = calculate_ref_state_from_qiskit_circuit(bw_pattern, qc, input_vec)
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

        # ref_state = calculate_ref_state_from_qiskit_circuit(bw_pattern, qc, input_vec)

        mapping = extract_logical_to_physical(qc, transpiled_qc)

        sv_phys = Statevector(input_vec).evolve(transpiled_qc)
        undo_layout_on_state(sv_phys, mapping)

        # If you simulated with your MBQC engine and got a flat numpy array `psi`:
        sv_logical_from_mbqc = undo_layout_on_state(psi, mapping, total_qubits=transpiled_qc.num_qubits)

        # Compare output up to global phase
        assert utils.assert_equal_up_to_global_phase(sv_logical_from_mbqc.data, psi)
