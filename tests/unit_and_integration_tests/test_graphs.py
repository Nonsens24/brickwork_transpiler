import numpy as np
from qiskit import QuantumCircuit
from src.brickwork_transpiler import decomposer, graph_builder, visualiser


def test_bw_graph_size_group_size_two_even():

    num_qubits = np.random.random_integers(2, 10)

    if num_qubits % 2 != 0: #enforce even
        num_qubits += 1

    qc = QuantumCircuit(num_qubits)

    sample = np.random.choice(num_qubits, size=2, replace=False)
    random_gates = np.random.choice(num_qubits, replace=False, size = np.random.random_integers(1, num_qubits))

    for i in random_gates:
        qc.h(i)

    qc.cx(sample[0], sample[1])

    decomposed_qc = decomposer.decompose_qc_to_bricks_qiskit(qc)
    qc_mat = decomposer.instructions_to_matrix_dag(decomposed_qc)
    bw_graph_data = graph_builder.generate_brickwork_graph_from_instruction_matrix(qc_mat)
    bw_nx_graph = graph_builder.to_networkx_graph(bw_graph_data)

    # Debug info
    # visualiser.print_matrix(qc_mat)
    # visualiser.plot_graph(bw_nx_graph)
    # print(qc.draw())

    assert bw_nx_graph.number_of_nodes() == (num_qubits * (2 * 4)) + num_qubits

def test_bw_graph_size_group_size_two_odd():

    num_qubits = np.random.random_integers(3, 10)

    if num_qubits % 2 == 0:     #enforce odd
        num_qubits += 1

    qc = QuantumCircuit(num_qubits)

    sample = np.random.choice(num_qubits, size=2, replace=False)
    # enforce unuqueness
    random_gates = np.random.choice(num_qubits, replace=False, size = np.random.random_integers(1, num_qubits))

    for i in random_gates:
        qc.h(i)

    qc.cx(sample[0], sample[1])

    decomposed_qc = decomposer.decompose_qc_to_bricks_qiskit(qc)
    qc_mat = decomposer.instructions_to_matrix_dag(decomposed_qc)
    bw_graph_data = graph_builder.generate_brickwork_graph_from_instruction_matrix(qc_mat)
    bw_nx_graph = graph_builder.to_networkx_graph(bw_graph_data)

    # Debug info
    # visualiser.print_matrix(qc_mat)
    # visualiser.plot_graph(bw_nx_graph)
    # print(qc.draw())

    assert bw_nx_graph.number_of_nodes() == (num_qubits * (2 * 4)) + num_qubits
