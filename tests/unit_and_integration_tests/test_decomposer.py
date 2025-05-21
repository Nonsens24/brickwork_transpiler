import numpy as np
import pytest
import unittest
from qiskit import QuantumCircuit

from src.brickwork_transpiler import visualiser, decomposer
from src.brickwork_transpiler.decomposer import align_bricks, instructions_to_matrix_dag


# Dummy instruction class for testing
class DummyInstr:
    def __init__(self, name):
        self.name = name
    def __repr__(self):
        return f"DummyInstr('{self.name}')"


def mk_matrix(n_qubits, columns):
    """
    Build an n_qubits × n_columns instruction matrix from a list of columns,
    where each column is a list of length n_qubits of instruction-name lists.
    """
    n_cols = len(columns)
    matrix = [[[] for _ in range(n_cols)] for _ in range(n_qubits)]
    for c, col in enumerate(columns):
        if len(col) != n_qubits:
            raise ValueError(f"Column {c} must have {n_qubits} entries, got {len(col)}")
        for q, names in enumerate(col):
            for name in names:
                if name:  # skip empty strings
                    matrix[q][c].append(DummyInstr(name))
    return matrix


def names(matrix):
    """Extract instruction names from the (n_qubits×n_cols) matrix for assertions."""
    # matrix is a list of lists: matrix[q][c] is a list of instrs
    n_q = len(matrix)
    n_c = len(matrix[0]) if n_q>0 else 0
    names_mat = [[None]*n_c for _ in range(n_q)]
    for q in range(n_q):
        for c in range(n_c):
            names_mat[q][c] = [instr.name for instr in matrix[q][c]]
    return names_mat

def get_matrices(qc):
    # Decompose to CX, rzrxrz, id
    decomposed_qc = decomposer.decompose_qc_to_bricks_qiskit(qc, opt=3)

    # Optiise instruction matrix with dependency graph
    qc_mat, cx_mat = decomposer.instructions_to_matrix_dag(decomposed_qc)
    qc_mat_aligned = decomposer.align_bricks(cx_mat, qc_mat)

    return qc_mat, cx_mat, qc_mat_aligned

class TestBrickAlignment(unittest.TestCase):

    def test_no_cx_only_rotations(self):
        # cols = [
        #     [['rz','rx'], ['rz'], ['rx']],
        #     [['rz'], ['rx','rz'], ['rz','rx','rz']],
        # ]

        qc = QuantumCircuit(3)
        qc.rz(np.pi/2, 0)
        qc.rx(np.pi / 3, 0)
        qc.rz(np.pi / 4, 1)
        qc.rz(np.pi / 5, 2)

        qc.rz(np.pi / 6, 0)
        qc.rx(np.pi / 7, 1)
        qc.rz(np.pi / 8, 1)
        qc.rz(np.pi / 9, 2)
        qc.rx(np.pi / 7, 2)
        qc.rz(np.pi / 8, 2)

        qc_mat, cx_mat, qc_mat_aligned = get_matrices(qc)

        # No padding: same number of columns
        assert len(qc_mat_aligned[0]) == 1
        # Check content matches input
        for col_id, col in enumerate(zip(*qc_mat_aligned)):
            for row in col:
                for instr in row:
                    assert instr.name.startswith('r')


    def test_pure_even_cx_alignment_target_bot(self):
        # Single column with cx on qubits 0-1 (even parity)
        qc = QuantumCircuit(3)
        qc.cx(0, 1)

        qc_mat, cx_mat, qc_mat_aligned = get_matrices(qc)

        assert len(qc_mat[0]) == 1
        # First brick should be identity
        # Second brick has CXs on qubits 1 and 2
        assert names(qc_mat)[0][0] == ['cx0c']
        assert names(qc_mat)[1][0] == ['cx0t']

        assert len(cx_mat[0]) == 1
        # First brick should be identity
        assert all(cell == [] for cell in cx_mat[2][0])
        # Second brick has CXs on qubits 1 and 2
        assert names(cx_mat)[0][0] == ['cx0c']
        assert names(cx_mat)[1][0] == ['cx0t']

        # initial brick_idx=0 (even), needs odd -> one pad, then CX brick
        assert len(qc_mat_aligned[0]) == 1
        # First brick should be identity
        assert all(cell == [] for cell in qc_mat_aligned[2][0])
        # Second brick has CXs on qubits 1 and 2
        assert names(qc_mat_aligned)[0][0] == ['cx0c']
        assert names(qc_mat_aligned)[1][0] == ['cx0t']


    def test_pure_even_cx_alignment2(self):
        # Single column with cx on qubits 0-1 (even parity)
        qc = QuantumCircuit(3)
        qc.cx(1, 0)

        qc_mat, cx_mat, qc_mat_aligned = get_matrices(qc)

        assert len(qc_mat[0]) == 1
        # First brick should be identity
        # Second brick has CXs on qubits 1 and 2
        assert names(qc_mat)[0][0] == ['cx0t']
        assert names(qc_mat)[1][0] == ['cx0c']

        assert len(cx_mat[0]) == 1
        # First brick should be identity
        assert all(cell == [] for cell in cx_mat[2][0])
        # Second brick has CXs on qubits 1 and 2
        assert names(cx_mat)[0][0] == ['cx0t']
        assert names(cx_mat)[1][0] == ['cx0c']

        # initial brick_idx=0 (even), needs odd -> one pad, then CX brick
        assert len(qc_mat_aligned[0]) == 1
        # First brick should be identity
        assert all(cell == [] for cell in qc_mat_aligned[2][0])
        # Second brick has CXs on qubits 1 and 2
        assert names(qc_mat_aligned)[0][0] == ['cx0t']
        assert names(qc_mat_aligned)[1][0] == ['cx0c']


    def test_pure_odd_cx_alignment(self):
        # Single column with cx on qubits 1-2 (odd parity)
        qc = QuantumCircuit(3)
        qc.cx(2, 1)

        qc_mat, cx_mat, qc_mat_aligned = get_matrices(qc)

        assert len(qc_mat[0]) == 1
        # First brick should be identity
        # Second brick has CXs on qubits 1 and 2
        assert names(qc_mat)[1][0] == ['cx0t']
        assert names(qc_mat)[2][0] == ['cx0c']

        assert len(cx_mat[0]) == 2
        # First brick should be identity
        assert all(cell == [] for cell in cx_mat[0][0:1])
        # Second brick has CXs on qubits 1 and 2
        assert names(cx_mat)[1][1] == ['cx0t']
        assert names(cx_mat)[2][1] == ['cx0c']

        # initial brick_idx=0 (even), needs odd -> one pad, then CX brick
        assert len(qc_mat_aligned[0]) == 2
        # First brick should be identity
        assert all(cell == [] for cell in qc_mat_aligned[0][0:1])
        # Second brick has CXs on qubits 1 and 2
        assert names(qc_mat_aligned)[1][1] == ['cx0t']
        assert names(qc_mat_aligned)[2][1] == ['cx0c']


    def test_mixed_same_parity_cx(self):
        # Two independent CNOTs on (0,1) and (2,3), both even parity and non-adjacent
        qc = QuantumCircuit(4)
        qc.cx(1, 0)
        qc.cx(3, 2)

        qc_mat, cx_mat, qc_mat_aligned = get_matrices(qc)

        # Both CNOTs should be in a single brick without padding

        assert len(qc_mat[0]) == 1

        # Check cx(0,1)
        assert 'cx0t' in qc_mat[0][0][0].name and 'cx0c' in qc_mat[1][0][0].name
        # Check cx(3,4)
        assert 'cx1t' in qc_mat[2][0][0].name and 'cx1c' in qc_mat[3][0][0].name


        assert len(cx_mat[0]) == 1

        # Check cx(0,1)
        assert 'cx0t' in cx_mat[0][0][0].name and 'cx0c' in cx_mat[1][0][0].name
        # Check cx(3,4)
        assert 'cx1t' in cx_mat[2][0][0].name and 'cx1c' in cx_mat[3][0][0].name


        assert len(qc_mat_aligned[0]) == 1

        # Check cx(0,1)
        assert 'cx0t' in qc_mat_aligned[0][0][0].name and 'cx0c' in qc_mat_aligned[1][0][0].name
        # Check cx(3,4)
        assert 'cx1t' in qc_mat_aligned[2][0][0].name and 'cx1c' in qc_mat_aligned[3][0][0].name



    def test_mixed_diff_parity_cx(self):

        qc = QuantumCircuit(5)
        qc.cx(1, 0)
        qc.cx(3, 4)

        qc_mat, cx_mat, qc_mat_aligned = get_matrices(qc)

        # visualiser.print_matrix(qc_mat)
        # visualiser.print_matrix(cx_mat)
        # visualiser.print_matrix(qc_mat_aligned)

        # Test before shift
        assert len(qc_mat[0]) == 1
        # First brick handles cx(0,1)
        assert 'cx0t' in names(qc_mat)[0][0]
        assert 'cx0c' in names(qc_mat)[1][0]
        assert names(qc_mat)[2][0] == []
        # Second brick handles cx(3,4)
        assert 'cx1c' in names(qc_mat)[3][0]
        assert 'cx1t' in names(qc_mat)[4][0]

        # Test after shift
        assert len(qc_mat_aligned[0]) == 2
        # First brick handles cx(0,1)
        assert 'cx0t' in names(qc_mat_aligned)[0][0]
        assert 'cx0c' in names(qc_mat_aligned)[1][0]
        assert names(qc_mat_aligned)[3][0] == []
        # Second brick handles cx(3,4)
        assert 'cx1c' in names(qc_mat_aligned)[3][1]
        assert 'cx1t' in names(qc_mat_aligned)[4][1]


    def test_rotations_with_cx(self):
        # Two columns: rotations only, then mixed CX and rotations
        cols = [
            [['rz'], ['rx'], ['rz'], []],
            [['cx'], ['cx'], [], ['rz']],
        ]
        qc = QuantumCircuit(4)
        qc.rz(np.pi / 3, 0)
        qc.rx(np.pi / 4, 1)
        qc.rz(np.pi / 5, 2)
        qc.cx(1, 0)
        qc.rz(np.pi / 3, 3)

        qc_mat, cx_mat, qc_mat_aligned = get_matrices(qc)

        visualiser.print_matrix(qc_mat)
        visualiser.print_matrix(cx_mat)
        visualiser.print_matrix(qc_mat_aligned)

        #Before shift
        cols_qc = list(zip(*qc_mat))

        assert len(cols_qc) == 2

        instr00 = cols_qc[0][0][0]
        instr01 = cols_qc[0][1][0]
        instr02 = cols_qc[0][2][0]
        instr03 = cols_qc[0][3][0]

        instr10 = cols_qc[1][0][0]
        instr11 = cols_qc[1][1][0]

        assert instr00.name == 'rz'
        assert instr01.name == 'rx'
        assert instr02.name == 'rz'
        assert instr03.name == 'rz'

        assert instr10.name == 'cx0t'
        assert instr11.name == 'cx0c'

        # After shift
        cols = list(zip(*qc_mat_aligned))

        assert len(cols) == 3

        instr00 = cols[0][0]
        instr01 = cols[0][1]
        instr02 = cols[0][2][0]
        instr03 = cols[0][3][0]

        instr20 = cols[2][0][0]
        instr21 = cols[2][1][0]

        assert instr00 == []
        assert instr01 == []
        assert instr02.name == 'rz'
        assert instr03.name == 'rz'

        assert instr20.name == 'cx0t'
        assert instr21.name == 'cx0c'



class TestInstructionsToMatrixDAG(unittest.TestCase):

    def test_single_qubit_rotations(self):
        qc = QuantumCircuit(2)
        qc.rz(0.5, 0)
        qc.rx(1.0, 1)
        qc_mat, cx_mat = instructions_to_matrix_dag(qc)

        self.assertEqual(len(qc_mat), 2)
        # Expect one column
        self.assertEqual(len(qc_mat[0]), 1)
        self.assertEqual(len(qc_mat[1]), 1)

        instr0 = qc_mat[0][0][0]
        instr1 = qc_mat[1][0][0]
        self.assertIn(instr0.name, ['rz'])
        self.assertIn(instr1.name, ['rx'])


    def test_single_cx_gate(self):
        qc = QuantumCircuit(2)
        qc.cx(0, 1)
        qc_mat, cx_mat = instructions_to_matrix_dag(qc)

        # Should produce one column
        self.assertEqual(len(qc_mat[0]), 1)
        self.assertEqual(len(qc_mat[1]), 1)

        names = {instr.name for instr in qc_mat[0][0]} | {instr.name for instr in qc_mat[1][0]}
        self.assertIn('cx0c', names)
        self.assertIn('cx0t', names)


    def test_multiple_cx_gates(self):
        qc = QuantumCircuit(3)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc_mat, cx_mat = instructions_to_matrix_dag(qc)

        # Should have at least two columns depending on atomic grouping
        # We verify the names exist with unique indices
        names = set()
        for q in qc_mat:
            for col in q:
                names |= {instr.name for instr in col}

        # Two CX pairs → should have cx0c/t and cx1c/t
        self.assertIn('cx0c', names)
        self.assertIn('cx0t', names)
        self.assertIn('cx1c', names)
        self.assertIn('cx1t', names)


    def test_single_cx_gate_rev_order(self):
        qc = QuantumCircuit(2)
        qc.cx(1, 0)
        qc_mat, cx_mat = instructions_to_matrix_dag(qc)

        # Should produce one column
        self.assertEqual(len(qc_mat[0]), 1)
        self.assertEqual(len(qc_mat[1]), 1)

        names = {instr.name for instr in qc_mat[0][0]} | {instr.name for instr in qc_mat[1][0]}
        self.assertIn('cx0t', names)
        self.assertIn('cx0c', names)


if __name__ == '__main__':
    pytest.main()
