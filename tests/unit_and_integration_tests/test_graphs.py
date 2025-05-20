import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from src.brickwork_transpiler import decomposer, graph_builder, visualiser, pattern_converter, brickwork_transpiler


def gen_test_graph(qc):
    # Decompose to CX, rzrxrz, id
    decomposed_qc = decomposer.decompose_qc_to_bricks_qiskit(qc, opt=3)

    # Optiise instruction matrix with dependency graph
    qc_mat, cx_mat = decomposer.instructions_to_matrix_dag(decomposed_qc)
    qc_mat_aligned = decomposer.align_bricks(cx_mat, qc_mat)

    # Build the graph from the optimised and formatted instruction matrix
    bw_graph_data = graph_builder.generate_brickwork_graph_from_instruction_matrix(qc_mat_aligned)

    # Get the networkx graph structure
    bw_nx_graph = graph_builder.to_networkx_graph(bw_graph_data)

    return bw_nx_graph

def test_bw_graph_size_group_size_two_even():

    qc= QuantumCircuit(5)
    qc.h(0)
    qc.h(3)
    qc.cx(2, 3)

    bw_nx_graph = gen_test_graph(qc)

    # Debug info
    # visualiser.print_matrix(qc_mat)
    # visualiser.plot_graph(bw_nx_graph)
    # print(qc.draw())

    assert bw_nx_graph.number_of_nodes() == (5 * (3 * 4)) + 5


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

    qc.cx(2, 1)

    bw_nx_graph = gen_test_graph(qc)

    assert bw_nx_graph.number_of_nodes() == (num_qubits * (2 * 4)) + num_qubits


def test_shift_bug():
    input_vector = Statevector.from_label('+++++')
    qc_bugged = QuantumCircuit(5)

    qc_bugged.cx(1, 0)

    qc_bugged.h(1)
    qc_bugged.cx(1, 2)
    qc_bugged.rz(np.pi / 2, 2)

    qc_bugged.rx(np.pi / 3, 2)
    qc_bugged.rz(np.pi / 4, 3)

    qc_bugged.cx(3, 4)
    qc_bugged.rz(np.pi / 4, 3)
    qc_bugged.h(4)

    bw_nx_graph = gen_test_graph(qc_bugged)

    # input qubits * (brick depth * number of bricks) + output qubits
    assert bw_nx_graph.number_of_nodes() == 5 * (4 * 5) + 5

def test_cx_alignment_left():
    input_vector = Statevector.from_label('+++++')

    qc = QuantumCircuit(5)
    qc.cx(2, 3)
    qc.cx(1, 2)

    bw_nx_graph = gen_test_graph(qc)

    # input qubits * (brick depth * number of bricks) + output qubits
    assert bw_nx_graph.number_of_nodes() == 5 * (4 * 2) + 5

def test_cx_alignment_right():
    input_vector = Statevector.from_label('+++++')

    qc = QuantumCircuit(5)
    qc.cx(1, 2)
    qc.cx(2, 3)

    bw_nx_graph = gen_test_graph(qc)

    # input qubits * (brick depth * number of bricks) + output qubits
    assert bw_nx_graph.number_of_nodes() == 5 * (4 * 3) + 5

def test_cx_alignment_depth_right():
    input_vector = Statevector.from_label('++++++')

    qc = QuantumCircuit(6)
    qc.cx(1, 2)
    qc.cx(4, 5)

    bw_nx_graph = gen_test_graph(qc)

    # input qubits * (brick depth * number of bricks) + output qubits
    assert bw_nx_graph.number_of_nodes() == 6 * (4 * 2) + 6

def test_cx_alignment_depth_left():
    input_vector = Statevector.from_label('++++++')

    qc = QuantumCircuit(6)
    qc.cx(1, 2)
    qc.cx(4, 5)

    bw_nx_graph = gen_test_graph(qc)

    # input qubits * (brick depth * number of bricks) + output qubits
    assert bw_nx_graph.number_of_nodes() == 6 * (4 * 2) + 6


def test_cx_alignment_depth_left_double():
    input_vector = Statevector.from_label('++++++++')

    qc = QuantumCircuit(8)
    qc.cx(2, 1)
    qc.cx(4, 5)
    qc.cx(6, 7)

    bw_nx_graph = gen_test_graph(qc)

    # input qubits * (brick depth * number of bricks) + output qubits
    assert bw_nx_graph.number_of_nodes() == 8 * (4 * 2) + 8

def test_cx_alignment_depth_right_double():
    input_vector = Statevector.from_label('++++++++')

    qc = QuantumCircuit(8)
    qc.cx(1, 0)
    qc.cx(3, 4)
    qc.cx(5, 6)

    bw_nx_graph = gen_test_graph(qc)

    # input qubits * (brick depth * number of bricks) + output qubits
    assert bw_nx_graph.number_of_nodes() == 8 * (4 * 2) + 8

def test_cx_alignment_depth_middle_shift():
    input_vector = Statevector.from_label('++++++++')

    qc = QuantumCircuit(8)
    qc.cx(0, 1)
    qc.cx(3, 4)
    qc.cx(6, 7)

    bw_nx_graph = gen_test_graph(qc)
    # input qubits * (brick depth * number of bricks) + output qubits
    assert bw_nx_graph.number_of_nodes() == 8 * (4 * 2) + 8

def test_cx_alignment_depth_outer_shift():
    input_vector = Statevector.from_label('+++++++++')

    qc = QuantumCircuit(9)
    qc.cx(2, 1)
    qc.cx(5, 4)
    qc.cx(8, 7)

    bw_nx_graph = gen_test_graph(qc)

    # input qubits * (brick depth * number of bricks) + output qubits
    assert bw_nx_graph.number_of_nodes() == 9 * (4 * 2) + 9


def test_cx_alignment_depth_outer_shift_four_cx():
    input_vector = Statevector.from_label('+++++++++')

    qc = QuantumCircuit(10)
    qc.cx(0, 1)
    qc.cx(3, 4)
    qc.cx(6, 5)
    qc.cx(8, 9)

    bw_nx_graph = gen_test_graph(qc)

    # input qubits * (brick depth * number of bricks) + output qubits
    assert bw_nx_graph.number_of_nodes() == 10 * (4 * 2) + 10


def test_cx_alignment_zig_zag():
    input_vector = Statevector.from_label('+++++++++++')

    qc = QuantumCircuit(11)
    qc.cx(0, 1)
    qc.cx(3, 4)
    qc.cx(6, 7)
    qc.cx(9, 10)

    bw_nx_graph = gen_test_graph(qc)

    # input qubits * (brick depth * number of bricks) + output qubits
    assert bw_nx_graph.number_of_nodes() == 11 * (4 * 2) + 11


def test_cx_alignment_zag_zig():
    input_vector = Statevector.from_label('++++++++++++')

    qc = QuantumCircuit(12)
    qc.cx(2, 1)
    qc.cx(5, 4)
    qc.cx(8, 7)
    qc.cx(11, 10)

    bw_nx_graph = gen_test_graph(qc)

    # input qubits * (brick depth * number of bricks) + output qubits
    assert bw_nx_graph.number_of_nodes() == 12 * (4 * 2) + 12


def test_cx_shifting_rotations():
    qc = QuantumCircuit(3)
    qc.cx(2, 1)
    qc.h(0)

    bw_nx_graph = gen_test_graph(qc)

    visualiser.plot_graph(bw_nx_graph)

    # input qubits * (brick depth * number of bricks) + output qubits
    assert bw_nx_graph.number_of_nodes() == 3 * (4 * 2) + 3

def test_cx_shifting_rotations_duplication():
    qc = QuantumCircuit(5)
    qc.cx(0, 1)
    qc.h(2)
    qc.cx(1, 2)
    qc.cx(4, 3)

    bw_nx_graph = gen_test_graph(qc)

    visualiser.plot_graph(bw_nx_graph)

    # input qubits * (brick depth * number of bricks) + output qubits
    assert bw_nx_graph.number_of_nodes() == 5 * (4 * 2) + 5





