# --- Quantum subroutines ---
import numpy as np
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit


def load_database(qc, idq, fq, feature_mat):
    for qb in idq: qc.h(qb)
    for idx, feats in enumerate(feature_mat):
        bits = format(idx, f'0{len(idq)}b')[::-1]
        for b, qb in enumerate(idq):
            if bits[b] == '0': qc.x(qb)
        for fbit, bit in enumerate(feats):
            if bit: qc.mcx(idq, fq[fbit])
        for b, qb in enumerate(idq):
            if bits[b] == '0': qc.x(qb)

def load_user(qc, uq, user_vec):
    for i, bit in enumerate(user_vec):
        if bit: qc.x(uq[i])

def qknn_block(qc, fq, uq, c0):
    l = len(fq)
    for i in range(l):
        qc.cx(uq[l-1-i], fq[i])
    qc.h(c0)
    for f in fq:
        qc.cp(np.pi/l, f, c0)
        qc.p(-np.pi/(2*l), f)
    qc.h(c0)
    for i in range(l):
        qc.cx(uq[l-1-i], fq[i])

def reflect_zero(qc, qubits):
    qc.x(qubits)
    qc.h(qubits[-1])
    qc.mcx(qubits[:-1], qubits[-1])
    qc.h(qubits[-1])
    qc.x(qubits)

def qrs_full_grover(feature_mat, user_feature, grover_rounds):
    N = len(feature_mat)
    q = int(np.log2(N))
    l = len(feature_mat[0])
    idq = QuantumRegister(q, 'id')
    fq  = QuantumRegister(l, 'f')
    uq  = QuantumRegister(l, 'u')
    c0  = QuantumRegister(1, 'c0')
    qA  = QuantumRegister(1, 'qA')   # Grover ancilla
    # Classical registers for measurement
    idc = ClassicalRegister(q, 'cid')
    c0c = ClassicalRegister(1, 'cc0')
    qc = QuantumCircuit(idq, fq, uq, c0, qA, idc, c0c)

    # State prep
    load_database(qc, idq, fq, feature_mat)
    load_user(qc, uq, user_feature)
    qknn_block(qc, fq, uq, c0[0])

    # Prepare Grover ancilla in |–⟩
    qc.x(qA[0]); qc.h(qA[0])

    for _ in range(grover_rounds):
        # (1) Oracle: phase flip if feature == user_feature
        target_bits = user_feature[::-1]
        for i, bit in enumerate(target_bits):
            if bit == 0: qc.x(fq[i])
        qc.mcx(fq, qA[0])
        for i, bit in enumerate(target_bits):
            if bit == 0: qc.x(fq[i])
        # (2) Uncompute kNN (U†)
        qknn_block(qc, fq, uq, c0[0]); qc = qc.inverse()
        # (3) Reflect about all zeroes
        tot = list(idq) + list(fq) + list(uq) + [c0[0]]
        reflect_zero(qc, tot)
        # (4) Redo kNN (U)
        qknn_block(qc, fq, uq, c0[0])
    qc.h(qA[0]); qc.x(qA[0])

    # --- Measurement: IDs and c0
    for i in range(q): qc.measure(idq[i], idc[i])
    qc.measure(c0[0], c0c[0])

    return qc