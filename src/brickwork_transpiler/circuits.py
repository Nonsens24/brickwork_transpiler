import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector


#
# input_vector = Statevector.from_label('+++++')
#
# qc = QuantumCircuit(5)
#
# qc.cx(0, 1)
#
# qc.rz(np.pi / 4, 1)
# qc.rx(np.pi / 4, 1)
# qc.rz(np.pi / 4, 1)
# qc.cx(0, 1)
#
# qc.rz(np.pi / 4, 2)
# qc.rx(np.pi / 4, 2)
# qc.rz(np.pi / 4, 2)
# qc.cx(2, 1)
#
# qc.rz(np.pi / 4, 3)
# qc.rx(np.pi / 4, 3)
# qc.rz(np.pi / 4, 3)
# qc.cx(2, 3)
#
# qc.rz(np.pi / 4, 4)
# qc.rx(np.pi / 4, 4)
# qc.rz(np.pi / 4, 4)
# qc.cx(4, 3)

def h_and_cx_circ():
    input_vector = Statevector.from_label('++')

    qc = QuantumCircuit(2)

    qc.h(0)
    qc.cx(0, 1)

    return qc, input_vector

def cx_and_h_circ():
    input_vector = Statevector.from_label('++')

    qc = QuantumCircuit(2)

    qc.cx(0, 1)
    qc.h(1)

    return qc, input_vector


def thesis_single():
    input_vector = Statevector.from_label('+')

    qc = QuantumCircuit(1)

    qc.h(0)
    qc.rx(np.pi/3, 0)
    qc.t(0)

    return qc, input_vector

def thesis_cx_ctrl_top():

    input_vector = Statevector.from_label('++')

    qc = QuantumCircuit(2)

    qc.cx(0, 1)

    return qc, input_vector


def thesis_cx_ctrl_bot():
    input_vector = Statevector.from_label('++')

    qc = QuantumCircuit(2)

    qc.cx(1, 0)

    return qc, input_vector

def thesis_cx_swap():

    input_vector = Statevector.from_label('+++')

    qc = QuantumCircuit(3)

    qc.cx(0, 2)

    return qc, input_vector


def shift_bug2():
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



    return qc_bugged, input_vector

def rz_index_bug():
    input_vector = Statevector.from_label('+++++')

    qc_bugged = QuantumCircuit(5)

    qc_bugged.h(0)
    qc_bugged.cx(1, 0)

    qc_bugged.h(1)
    qc_bugged.cx(2, 1)

    return qc_bugged, input_vector

def big_shifter():
    input_vector = Statevector.from_label('+++++')

    qc_bugged = QuantumCircuit(5)

    qc_bugged.h(0)
    qc_bugged.cx(1, 0)

    qc_bugged.h(1)
    qc_bugged.cx(2, 1)
    qc_bugged.rz(np.pi / 2, 2)

    qc_bugged.rx(np.pi / 3, 2)
    qc_bugged.rz(np.pi / 4, 3)

    qc_bugged.cx(3, 4)
    qc_bugged.rz(np.pi / 4, 3)
    qc_bugged.h(4)

    qc_bugged.cx(1, 0)

    qc_bugged.h(1)
    qc_bugged.cx(1, 2)
    qc_bugged.rz(np.pi / 2, 2)

    qc_bugged.rx(np.pi / 3, 2)
    qc_bugged.rz(np.pi / 4, 3)

    qc_bugged.cx(3, 4)
    qc_bugged.rz(np.pi / 4, 3)
    qc_bugged.h(4)

    qc_bugged.cx(1, 0)

    qc_bugged.h(1)
    qc_bugged.cx(1, 2)
    qc_bugged.rz(np.pi / 2, 2)

    qc_bugged.rx(np.pi / 3, 2)
    qc_bugged.rz(np.pi / 4, 3)

    qc_bugged.cx(3, 4)
    qc_bugged.rz(np.pi / 4, 3)
    qc_bugged.h(4)

def big_shifter_upper_rotation_brick_shifted():
    input_vector = Statevector.from_label('+++++')

    qc_bugged = QuantumCircuit(3)

    qc_bugged.cx(1, 2)

    qc_bugged.h(1)
    qc_bugged.cx(0, 1)



    return qc_bugged, input_vector

def shift_mixer():
    input_vector = Statevector.from_label('+++')

    qc_bugged = QuantumCircuit(3)

    qc_bugged.cx(1, 2)
    qc_bugged.h(1)
    qc_bugged.cx(0, 1)

    qc_bugged.cx(1, 2)
    qc_bugged.h(1)
    qc_bugged.cx(0, 1)

    qc_bugged.h(0)
    qc_bugged.cx(1, 0)
    qc_bugged.h(1)
    qc_bugged.cx(2, 1)

    qc_bugged.h(0)
    qc_bugged.cx(1, 0)
    qc_bugged.h(1)
    qc_bugged.cx(2, 1)

    return qc_bugged, input_vector


def big_shifter_both_up_low_rotation_brick_shifted():
    input_vector = Statevector.from_label('+++++')

    qc_bugged = QuantumCircuit(5)

    qc_bugged.h(0)
    qc_bugged.cx(1, 0)

    qc_bugged.h(1)
    qc_bugged.cx(2, 1)
    qc_bugged.rz(np.pi / 2, 2)

    qc_bugged.rx(np.pi / 3, 2)
    qc_bugged.rz(np.pi / 4, 3)

    qc_bugged.cx(3, 4)
    qc_bugged.rz(np.pi / 4, 3)
    qc_bugged.h(4)

    qc_bugged.cx(1, 0)

    qc_bugged.h(1)
    qc_bugged.cx(1, 2)
    qc_bugged.rz(np.pi / 2, 2)

    qc_bugged.rx(np.pi / 3, 2)
    qc_bugged.rz(np.pi / 4, 3)

    qc_bugged.cx(3, 4)
    qc_bugged.rz(np.pi / 4, 3)
    qc_bugged.h(4)

    qc_bugged.cx(1, 0)

    qc_bugged.h(1)
    qc_bugged.cx(1, 2)
    qc_bugged.rz(np.pi / 2, 2)

    qc_bugged.rx(np.pi / 3, 2)
    qc_bugged.rz(np.pi / 4, 3)

    qc_bugged.cx(3, 4)
    qc_bugged.rz(np.pi / 4, 3)
    qc_bugged.h(4)


    return qc_bugged, input_vector

def cnot_alignment_bug():
    input_vector = Statevector.from_label('++++++')

    qc = QuantumCircuit(6)
    qc.cx(0, 1)
    qc.cx(3, 4)


    return qc, input_vector

def cnot_alignment_bug_double_trouble():
    input_vector = Statevector.from_label('+++++')

    qc = QuantumCircuit(5)
    qc.cx(0, 1)
    qc.h(2)
    # qc.cx(1, 2)
    qc.cx(4, 3)


    return qc, input_vector

def test_circ():
    input_vector = Statevector.from_label('+++++')

    qc= QuantumCircuit(5)
    qc.h(0)
    qc.h(3)
    qc.cx(2, 3)

    return qc, input_vector

def cover_circ():
    input_vector = Statevector.from_label('+++++++++++++++++++++')

    qc = QuantumCircuit(21)
    qc.h(2)
    qc.cx(0,2)
    qc.h(0)


    qc.h(4)
    qc.cx(3,5)
    qc.h(6)

    qc.h(8)
    qc.cx(6,7)
    qc.h(6)

    qc.h(7)
    qc.rx(np.pi / 3, 7)
    qc.cx(7, 8)
    qc.h(8)
    qc.rz(np.pi / 2, 9)

    qc.rx(-np.pi / 3, 10)
    qc.rz(np.pi / 3, 11)
    qc.h(11)
    qc.rz(np.pi / 2, 10)

    qc.h(14)
    qc.cx(11,14)
    qc.h(11)

    qc.h(15)
    qc.cx(16,15)
    qc.h(16)

    qc.h(20)
    qc.cx(17,20)
    qc.h(17)

    return qc, input_vector


def test_small_cx():
    input_vector = Statevector.from_label('++++')

    qc= QuantumCircuit(4)
    qc.cx(0, 3)

    return qc, input_vector

def test_large_cx():
    input_vector = Statevector.from_label('++++++')

    qc= QuantumCircuit(6)
    qc.cx(0, 5)

    return qc, input_vector

def test_large_cx_8():
    input_vector = Statevector.from_label('++++++++')

    qc= QuantumCircuit(8)
    qc.cx(7, 0)

    return qc, input_vector

def test_large_cx_uneven():
    input_vector = Statevector.from_label('+++++++')

    qc= QuantumCircuit(7)
    qc.cx(6, 0)

    return qc, input_vector

def test_large_cx_two():
    input_vector = Statevector.from_label('+++++++++++')

    qc= QuantumCircuit(10)
    qc.cx(9, 0)
    qc.cx(6, 3)

    return qc, input_vector

def test_large_cx_two_rev():
    input_vector = Statevector.from_label('+++++++++++')

    qc= QuantumCircuit(10)
    qc.cx(6, 3)
    qc.cx(9, 0)


    return qc, input_vector

def test_large_cx_multiple():
    input_vector = Statevector.from_label('+++++++++++++')

    qc= QuantumCircuit(12)
    qc.cx(11, 0)
    qc.cx(9, 2)
    qc.cx(4, 7)

    return qc, input_vector

def presentation_circ():
    input_vector = Statevector.from_label('+++++')

    qc = QuantumCircuit(5)
    qc.h(0)
    qc.rx(np.pi / 3, 0)
    qc.cx(0, 1)
    qc.h(1)
    qc.cx(1, 0)
    qc.cx(1, 2)
    qc.rz(np.pi / 2, 2)
    qc.cx(2, 1)

    qc.rx(-np.pi / 3, 3)
    qc.rz(np.pi / 4, 4)
    qc.cx(3, 4)
    qc.h(4)
    qc.rz(np.pi / 2, 3)

    return qc, input_vector

def noisy_circuit_rotation():
    input_vector = Statevector.from_label('+')

    qc = QuantumCircuit(1)
    qc.h(0)
    # qc.h(1)

    return qc, input_vector

def dag_example_circ():
    input_vector = Statevector.from_label('+++++')

    qc = QuantumCircuit(5)
    qc.h(0)
    qc.rx(np.pi / 3, 0)
    qc.cx(0, 1)
    qc.h(1)
    qc.cx(1, 0)
    qc.cx(3, 2)

    qc.rx(-np.pi / 3, 3)
    qc.rz(np.pi / 4, 4)
    qc.cx(3, 4)

    qc.rz(np.pi / 2, 3)

    return qc, input_vector

def qft(n):
    """
    Returns a QuantumCircuit implementing the n-qubit Quantum Fourier Transform.
    """
    input_vector = Statevector.from_label('+' * n)

    qc = QuantumCircuit(n, name='QFT')
    # Apply QFT
    for j in range(n):
        qc.h(j)
        for k in range(j+1, n):
            lam = np.pi / float(2 ** (k - j))
            qc.cp(lam, k, j)
    # Swap qubits to reverse order
    for i in range(n // 2):
        qc.swap(i, n - i - 1)

    return qc, input_vector