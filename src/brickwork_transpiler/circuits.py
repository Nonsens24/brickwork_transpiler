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