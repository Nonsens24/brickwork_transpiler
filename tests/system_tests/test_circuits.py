# import numpy as np
# from qiskit import QuantumCircuit
# from qiskit.quantum_info import Statevector
# from src.brickwork_transpiler import brickwork_transpiler, visualiser, utils
# from src.brickwork_transpiler.utils import calculate_ref_state_from_qiskit_circuit
#
#
# def test_cx_from_zero():
#     # 1) Create the |++> state directly
#     input_vec = Statevector.from_label('++')  # two-qubit plus state
#
#     # 2) Define your 2-qubit circuit (no H gates needed)
#     qc = QuantumCircuit(2)
#     qc.h(0)
#     qc.h(1)
#     qc.cx(0, 1)
#
#     # Transpile!
#     bw_pattern, output_ref, col_map = brickwork_transpiler.transpile(qc, input_vec)
#     ref_state = calculate_ref_state_from_qiskit_circuit(bw_pattern, qc, input_vec)
#     visualiser.plot_brickwork_graph_from_pattern(bw_pattern,
#                                                  node_colours=col_map,
#                                                  use_node_colours=True,
#                                                  title="Brickwork Graph: test_cx_from_zero")
#
#     # Simulate the generated pattern
#     outstate = bw_pattern.simulate_pattern(backend='statevector').flatten()
#
#     # Compare output up to global phase
#     assert utils.assert_equal_up_to_global_phase(outstate, ref_state.data)
#
# def test_cx_from_zero_upper():
#     # 1) Create the |++> state directly
#     input_vec = Statevector.from_label('++')  # two-qubit plus state
#
#     # 2) Define your 2-qubit circuit (no H gates needed)
#     qc = QuantumCircuit(2)
#     qc.h(0)
#     qc.cx(0, 1)
#
#     # Transpile!
#     bw_pattern, output_ref, col_map = brickwork_transpiler.transpile(qc, input_vec)
#     ref_state = calculate_ref_state_from_qiskit_circuit(bw_pattern, qc, input_vec)
#     visualiser.plot_brickwork_graph_from_pattern(bw_pattern,
#                                                  node_colours=col_map,
#                                                  use_node_colours=True,
#                                                  title="Brickwork Graph: test_cx_from_zero")
#
#     # Simulate the generated pattern
#     outstate = bw_pattern.simulate_pattern(backend='statevector').flatten()
#
#     # Compare output up to global phase
#     assert utils.assert_equal_up_to_global_phase(outstate, ref_state.data)
#
#
#
# # # 49qubits >12min
# # def test_system_shift():
# #     # 1) Create the |++> state directly
# #     input_vec = Statevector.from_label('+++')  # three-qubit plus state
# #
# #     # 2) Define your 2-qubit circuit (no H gates needed)
# #     qc = QuantumCircuit(3)
# #     qc.t(0)
# #     qc.t(1)
# #     qc.cx(0, 1)
# #     qc.s(2)
# #     qc.s(0)
# #
# #     # Transpile!
# #     bw_pattern, output_ref, col_map = brickwork_transpiler.transpile(qc, input_vec)
# #     ref_state = calculate_ref_state_from_qiskit_circuit(bw_pattern, qc, input_vec)
# #     visualiser.plot_brickwork_graph_from_pattern(bw_pattern,
# #                                                  node_colours=col_map,
# #                                                  use_node_colours=True,
# #                                                  title="Brickwork Graph: test_system_shift")
# #
# #     # Simulate the generated pattern
# #     outstate = bw_pattern.simulate_pattern(backend='statevector').flatten()
# #
# #     # Compare output up to global phase
# #     assert utils.assert_equal_up_to_global_phase(outstate, ref_state.data)
#
#
# def test_four_in_cx_cancel():
#     # 1) Create the |++> state directly
#     input_vec = Statevector.from_label('++++')  # three-qubit plus state
#
#     # 2) Define your 2-qubit circuit (no H gates needed)
#     qc = QuantumCircuit(4)
#     qc.cx(0, 1)
#     qc.cx(0, 1)
#     qc.h(3)
#
#     # Transpile!
#     bw_pattern, output_ref, col_map = brickwork_transpiler.transpile(qc, input_vec)
#     ref_state = calculate_ref_state_from_qiskit_circuit(bw_pattern, qc, input_vec)
#     visualiser.plot_brickwork_graph_from_pattern(bw_pattern,
#                                                  node_colours=col_map,
#                                                  use_node_colours=True,
#                                                  title="Brickwork Graph: test_four_in_cx_cancel")
#
#     # Simulate the generated pattern
#     outstate = bw_pattern.simulate_pattern(backend='statevector').flatten()
#
#     # Compare output up to global phase
#     assert utils.assert_equal_up_to_global_phase(outstate, ref_state.data)
#
# def test_five_in_optimise_gates():  # TODO: Fix first CX shift!
#     # 1) Create the |++> state directly
#     input_vec = Statevector.from_label('+++++')  # three-qubit plus state
#
#     # 2) Define your 2-qubit circuit (no H gates needed)
#     qc = QuantumCircuit(5)
#     qc.cx(0, 1)
#     qc.s(2)
#     qc.rx(np.pi/3, 2)
#     qc.t(2)
#     qc.x(2)
#     qc.cx(3, 4)
#
#     # Transpile!
#     bw_pattern, output_ref, col_map = brickwork_transpiler.transpile(qc, input_vec)
#     ref_state = calculate_ref_state_from_qiskit_circuit(bw_pattern, qc, input_vec)
#     visualiser.plot_brickwork_graph_from_pattern(bw_pattern,
#                                                  node_colours=col_map,
#                                                  use_node_colours=True,
#                                                  title="Brickwork Graph: test_five_in_optimise_gates")
#
#     # Simulate the generated pattern
#     outstate = bw_pattern.simulate_pattern(backend='statevector').flatten()
#
#     # Compare output up to global phase
#     assert utils.assert_equal_up_to_global_phase(outstate, ref_state.data)