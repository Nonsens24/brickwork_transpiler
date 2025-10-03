import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
# from src.brickwork_transpiler import brickwork_transpiler, visualiser, utils, circuits
from brickwork_transpiler import brickwork_transpiler, visualiser, utils, circuits

# from brickwork_transpiler import utils, brickwork_transpiler, circuits  # etc.

from src.brickwork_transpiler.utils import calculate_ref_state_from_qiskit_circuit, undo_layout_on_state, \
    extract_logical_to_physical, reference_state_auto


#

def test_system_shift():
    # minimaLqrs.build_graph()
    # minimaLqrs.run_and_plot_minimal_qrs_only_db()

    input_vec = Statevector.from_label('+++')  # three-qubit plus state

    # 2) Define your 2-qubit circuit (no H gates needed)
    qc = QuantumCircuit(3)
    qc.t(0)
    qc.t(1)
    qc.cx(0, 1)
    qc.s(2)
    qc.s(0)

    bw_pattern, col_map, transpiled_qc = brickwork_transpiler.transpile(qc, input_vec,
                                                                        routing_method="sabre",
                                                                        layout_method="trivial",
                                                                        with_ancillas=False)

    # Optimise for tensornetwork backand and simulation efficiency
    bw_pattern.standardize()  # puts commands into N-E-M-(X/Z/C) order
    bw_pattern.shift_signals()  # optional but recommended; reduces feedforward

    tn = bw_pattern.simulate_pattern(backend="tensornetwork", graph_prep="parallel")
    psi = tn.to_statevector()  # state on your declared outputs

    # Independently calculate a reference state to check output
    reference_output = utils.calculate_ref_state_from_qiskit_circuit(bw_pattern, qc, transpiled_qc, input_vec)

    # Compare output up to global phase
    assert utils.assert_equal_up_to_global_phase(psi, reference_output)

def test_cx_from_zero():
    # 1) Create the |++> state directly
    input_vec = Statevector.from_label('++')  # two-qubit plus state

    # 2) Define your 2-qubit circuit (no H gates needed)
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.h(1)
    qc.cx(0, 1)

    bw_pattern, col_map, transpiled_qc = brickwork_transpiler.transpile(qc, input_vec,
                                                                        routing_method="sabre",
                                                                        layout_method="trivial",
                                                                        with_ancillas=False)

    # Optimise for tensornetwork backand and simulation efficiency
    bw_pattern.standardize()  # puts commands into N-E-M-(X/Z/C) order
    bw_pattern.shift_signals()  # optional but recommended; reduces feedforward

    tn = bw_pattern.simulate_pattern(backend="tensornetwork", graph_prep="parallel")
    psi = tn.to_statevector()  # state on your declared outputs

    # Independently calculate a reference state to check output
    reference_output = utils.calculate_ref_state_from_qiskit_circuit(bw_pattern, qc, transpiled_qc, input_vec)

    # Compare output up to global phase
    assert utils.assert_equal_up_to_global_phase(psi, reference_output)


def test_cx_from_zero_upper():
    # 1) Create the |++> state directly
    input_vec = Statevector.from_label('++')  # two-qubit plus state

    # 2) Define your 2-qubit circuit (no H gates needed)
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)

    bw_pattern, col_map, transpiled_qc = brickwork_transpiler.transpile(qc, input_vec,
                                                         routing_method="sabre",
                                                         layout_method="trivial",
                                                         with_ancillas=False)

    # Optimise for tensornetwork backand and simulation efficiency
    bw_pattern.standardize()  # puts commands into N-E-M-(X/Z/C) order
    bw_pattern.shift_signals()  # optional but recommended; reduces feedforward

    tn = bw_pattern.simulate_pattern(backend="tensornetwork", graph_prep="parallel")
    psi = tn.to_statevector()  # state on your declared outputs

    # Independently calculate a reference state to check output
    reference_output = utils.calculate_ref_state_from_qiskit_circuit(bw_pattern, qc, transpiled_qc, input_vec)

    # Compare output up to global phase
    assert utils.assert_equal_up_to_global_phase(psi, reference_output)



def test_four_in_cx_cancel():
    # 1) Create the |++> state directly
    input_vec = Statevector.from_label('++++')  # three-qubit plus state

    # 2) Define your 2-qubit circuit (no H gates needed)
    qc = QuantumCircuit(4)
    qc.cx(0, 1)
    qc.cx(0, 1)
    qc.h(3)

    bw_pattern, col_map, transpiled_qc = brickwork_transpiler.transpile(qc, input_vec,
                                                         routing_method="sabre",
                                                         layout_method="trivial",
                                                         with_ancillas=False)

    # Optimise for tensornetwork backand and simulation efficiency
    bw_pattern.standardize()  # puts commands into N-E-M-(X/Z/C) order
    bw_pattern.shift_signals()  # optional but recommended; reduces feedforward

    tn = bw_pattern.simulate_pattern(backend="tensornetwork", graph_prep="parallel")
    psi = tn.to_statevector()  # state on your declared outputs

    # Independently calculate a reference state to check output
    reference_output = utils.calculate_ref_state_from_qiskit_circuit(bw_pattern, qc, transpiled_qc, input_vec)

    # Compare output up to global phase
    assert utils.assert_equal_up_to_global_phase(psi, reference_output)

def test_five_in_optimise_gates():
    # 1) Create the |++> state directly
    input_vec = Statevector.from_label('+++++')  # three-qubit plus state

    # 2) Define your 2-qubit circuit (no H gates needed)
    qc = QuantumCircuit(5)
    qc.cx(0, 1)
    qc.s(2)
    qc.rx(np.pi/3, 2)
    qc.t(2)
    qc.x(2)
    qc.cx(3, 4)

    bw_pattern, col_map, transpiled_qc = brickwork_transpiler.transpile(qc, input_vec,
                                                         routing_method="sabre",
                                                         layout_method="trivial",
                                                         with_ancillas=False)
    # ref_state = calculate_ref_state_from_qiskit_circuit(bw_pattern, qc, input_vec)
    # visualiser.plot_brickwork_graph_from_pattern(bw_pattern,
    #                                              node_colours=col_map,
    #                                              use_node_colours=True,
    #                                              title="test_cx_from_zero_upper",
    #                                              )

    # Optimise for tensornetwork backand and simulation efficiency
    bw_pattern.standardize()  # puts commands into N-E-M-(X/Z/C) order
    bw_pattern.shift_signals()  # optional but recommended; reduces feedforward

    tn = bw_pattern.simulate_pattern(backend="tensornetwork", graph_prep="parallel")
    psi = tn.to_statevector()  # state on your declared outputs

    # Independently calculate a reference state to check output
    reference_output = utils.calculate_ref_state_from_qiskit_circuit(bw_pattern, qc, transpiled_qc, input_vec)

    # Compare output up to global phase
    assert utils.assert_equal_up_to_global_phase(psi, reference_output)

def test_minimal_qrs():

    qc, input_vec = circuits.minimal_qrs([0, 0])

    qc.draw(output='mpl',
                        fold=40,
                        style="iqp"
                        )


    bw_pattern, col_map, transpiled_qc = brickwork_transpiler.transpile(qc, input_vec,
                                                                        routing_method="sabre",
                                                                        layout_method="trivial",
                                                                        with_ancillas=True)

    # visualiser.plot_brickwork_graph_from_pattern(bw_pattern,
    #                                              node_colours=col_map,
    #                                              use_node_colours=True,
    #                                              title="test_cx_from_zero_upper",
    #                                              )

    # Optimise for tensornetwork backand and simulation efficiency
    bw_pattern.standardize()  # puts commands into N-E-M-(X/Z/C) order
    bw_pattern.shift_signals()  # optional but recommended; reduces feedforward

    tn = bw_pattern.simulate_pattern(backend="tensornetwork", graph_prep="parallel")
    psi = tn.to_statevector()  # state on your declared outputs

    # Independently calculate a reference state to check output
    reference_output = utils.calculate_ref_state_from_qiskit_circuit(bw_pattern, qc, transpiled_qc, input_vec)

    # Compare output up to global phase
    assert utils.assert_equal_up_to_global_phase(psi, reference_output)

def test_minimal_qrs_no_ancillae():

    qc, input_vec = circuits.minimal_qrs([0, 0])

    qc.draw(output='mpl',
                        fold=40,
                        style="iqp"
                        )


    bw_pattern, col_map, transpiled_qc = brickwork_transpiler.transpile(qc, input_vec,
                                                                        routing_method="sabre",
                                                                        layout_method="trivial",
                                                                        with_ancillas=False)
    # visualiser.plot_brickwork_graph_from_pattern(bw_pattern,
    #                                              node_colours=col_map,
    #                                              use_node_colours=True,
    #                                              title="test_cx_from_zero_upper",
    #                                              )

    # Optimise for tensornetwork backand and simulation efficiency
    bw_pattern.standardize()  # puts commands into N-E-M-(X/Z/C) order
    bw_pattern.shift_signals()  # optional but recommended; reduces feedforward

    tn = bw_pattern.simulate_pattern(backend="tensornetwork", graph_prep="parallel")
    psi = tn.to_statevector()  # state on your declared outputs

    # Independently calculate a reference state to check output
    reference_output = utils.calculate_ref_state_from_qiskit_circuit(bw_pattern, qc, transpiled_qc, input_vec)

    # Compare output up to global phase
    assert utils.assert_equal_up_to_global_phase(psi, reference_output)


def test_small_with_ancillae_not_required():

    qc, input_vec = circuits.h_and_cx_circ()

    qc.draw(output='mpl',
                        fold=40,
                        style="iqp"
                        )


    bw_pattern, col_map, transpiled_qc = brickwork_transpiler.transpile(qc, input_vec,
                                                                        routing_method="sabre",
                                                                        layout_method="trivial",
                                                                        with_ancillas=True)

    # Optimise for tensornetwork backand and simulation efficiency
    bw_pattern.standardize()  # puts commands into N-E-M-(X/Z/C) order
    bw_pattern.shift_signals()  # optional but recommended; reduces feedforward

    tn = bw_pattern.simulate_pattern(backend="tensornetwork", graph_prep="parallel")
    psi = tn.to_statevector()  # state on your declared outputs

    # Independently calculate a reference state to check output
    reference_output = utils.calculate_ref_state_from_qiskit_circuit(bw_pattern, qc, transpiled_qc, input_vec)

    # Compare output up to global phase
    assert utils.assert_equal_up_to_global_phase(psi, reference_output)


def test_medium_circ_with_ancillae_not_required():

    qc, input_vector = circuits.big_shifter_both_up_low_rotation_brick_shifted()

    qc.draw(output='mpl',
                        fold=40,
                        style="iqp"
                        )


    bw_pattern, col_map, transpiled_qc = brickwork_transpiler.transpile(qc, input_vector,
                                                                        routing_method="sabre",
                                                                        layout_method="trivial",
                                                                        with_ancillas=True)

    # Optimise for tensornetwork backand and simulation efficiency
    bw_pattern.standardize()  # puts commands into N-E-M-(X/Z/C) order
    bw_pattern.shift_signals()  # optional but recommended; reduces feedforward

    tn = bw_pattern.simulate_pattern(backend="tensornetwork", graph_prep="parallel")
    psi = tn.to_statevector()  # state on your declared outputs

    # Independently calculate a reference state to check output
    reference_output = utils.calculate_ref_state_from_qiskit_circuit(bw_pattern, qc, transpiled_qc, input_vector)

    # Compare output up to global phase
    assert utils.assert_equal_up_to_global_phase(psi, reference_output)


def test_qft_3_with_ancillae():

    qc, input_vector = circuits.qft(3)

    qc.draw(output='mpl',
                        fold=40,
                        style="iqp"
                        )


    bw_pattern, col_map, transpiled_qc = brickwork_transpiler.transpile(qc, input_vector,
                                                                        routing_method="sabre",
                                                                        layout_method="trivial",
                                                                        with_ancillas=True)
    # visualiser.plot_brickwork_graph_from_pattern(bw_pattern,
    #                                              node_colours=col_map,
    #                                              use_node_colours=True,
    #                                              title="test_cx_from_zero_upper",
    #                                              )

    # Optimise for tensornetwork backand and simulation efficiency
    bw_pattern.standardize()  # puts commands into N-E-M-(X/Z/C) order
    bw_pattern.shift_signals()  # optional but recommended; reduces feedforward

    tn = bw_pattern.simulate_pattern(backend="tensornetwork", graph_prep="parallel")
    psi = tn.to_statevector()  # state on your declared outputs

    # Independently calculate a reference state to check output
    reference_output = utils.calculate_ref_state_from_qiskit_circuit(bw_pattern, qc, transpiled_qc, input_vector)

    # Compare output up to global phase
    assert utils.assert_equal_up_to_global_phase(psi, reference_output)


def test_qft_3_without_ancillae():

    qc, input_vector = circuits.qft(3)

    qc.draw(output='mpl',
                        fold=40,
                        style="iqp"
                        )

    bw_pattern, col_map, transpiled_qc = brickwork_transpiler.transpile(qc, input_vector,
                                                                        routing_method="sabre",
                                                                        layout_method="trivial",
                                                                        with_ancillas=False)

    # Optimise for tensornetwork backand and simulation efficiency
    bw_pattern.standardize()  # puts commands into N-E-M-(X/Z/C) order
    bw_pattern.shift_signals()  # optional but recommended; reduces feedforward

    tn = bw_pattern.simulate_pattern(backend="tensornetwork", graph_prep="parallel")
    psi = tn.to_statevector()  # state on your declared outputs

    # Independently calculate a reference state to check output
    reference_output = utils.calculate_ref_state_from_qiskit_circuit(bw_pattern, qc, transpiled_qc, input_vector)

    # Compare output up to global phase
    assert utils.assert_equal_up_to_global_phase(psi, reference_output)