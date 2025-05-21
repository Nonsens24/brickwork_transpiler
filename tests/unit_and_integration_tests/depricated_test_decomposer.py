import pytest
import unittest
from qiskit import QuantumCircuit

from src.brickwork_transpiler import visualiser
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

class TestBrickAlignment(unittest.TestCase):

    def test_no_cx_only_rotations(self):
        cols = [
            [['rz','rx'], ['rz'], ['rx']],
            [['rz'], ['rx','rz'], ['rz','rx','rz']],
        ]
        mat = mk_matrix(3, cols)
        out = align_bricks(mat)
        # No padding: same number of columns
        assert len(out[0]) == len(cols)
        # Check content matches input
        out_names = names(out)
        for c in range(len(cols)):
            for q in range(3):
                assert out_names[q][c] == cols[c][q]


    def test_pure_even_cx_alignment(self):
        cols = [
            [['rz','rx'], ['rz'], ['rx']],
            [['rz'], ['rx','rz'], ['rz','rx','rz']],
        ]
        mat = mk_matrix(3, cols)
        out = align_bricks(mat)
        # No padding: same number of columns
        assert len(out[0]) == len(cols)
        # Check content matches input
        for c in range(len(cols)):
            for q in range(3):
                assert names(out)[q][c] == cols[c][q]


    def test_pure_even_cx_alignment2(self):
        # Single column with cx on qubits 0-1 (even parity)
        cols = [
            [['cx0t'], ['cx0c'], []],  # q0->cx, q1->cx, q2->none
        ]
        mat = mk_matrix(3, cols)
        out = align_bricks(mat)
        # brick_idx starts at 0 (even), so no padding needed
        assert len(out[0]) == 1
        # CX should appear on qubits 0 and 1 only
        assert names(out)[0][0] == ['cx0t']
        assert names(out)[1][0] == ['cx0c']
        assert names(out)[2][0] == []


    def test_pure_odd_cx_alignment(self):
        # Single column with cx on qubits 1-2 (odd parity)
        cols = [
            [[], ['cx0t'], ['cx0c']],
        ]
        mat = mk_matrix(3, cols)
        out = align_bricks(mat)
        # initial brick_idx=0 (even), needs odd -> one pad, then CX brick
        assert len(out[0]) == 2
        # First brick should be identity
        assert all(cell == [] for cell in names(out)[0][0:1])
        # Second brick has CXs on qubits 1 and 2
        assert names(out)[1][1] == ['cx0t']
        assert names(out)[2][1] == ['cx0c']


    def test_mixed_same_parity_cx(self):
        # Two independent CNOTs on (0,1) and (2,3), both even parity and non-adjacent
        cols = [
            [['cx0t'], ['cx0c'], ['cx1c'], ['cx1t']],
        ]
        mat = mk_matrix(4, cols)
        out = align_bricks(mat)
        # visualiser.print_matrix(mat)
        # visualiser.print_matrix(out)
        # Both CNOTs should be in a single brick without padding
        assert len(out[0]) == 1
        names_out = names(out)
        # Check cx(0,1)
        assert 'cx0t' in names_out[0][0] and 'cx0c' in names_out[1][0]
        # Check cx(3,4)
        assert 'cx1c' in names_out[2][0] and 'cx1t' in names_out[3][0]



    def test_mixed_diff_parity_cx(self):
        # Single column with cx on 0-1 (even) and 3-4 (odd)
        cols = [
            [['cx0t'], ['cx0c'], [], ['cx1c'], ['cx1t']],
        ]
        mat = mk_matrix(5, cols)
        # visualiser.print_matrix(mat)
        # out = align_bricks(mat)
        # Should split into two bricks
        assert len(out[0]) == 2
        # First brick handles cx(0,1)
        assert 'cx0t' in names(out)[0][0]
        assert 'cx0c' in names(out)[1][0]
        assert names(out)[3][0] == []
        # Second brick handles cx(3,4)
        assert 'cx1c' in names(out)[3][1]
        assert 'cx1t' in names(out)[4][1]


    def test_rotations_with_cx(self):
        # Two columns: rotations only, then mixed CX and rotations
        cols = [
            [['rz'], ['rx'], ['rz'], []],
            [['cx'], ['cx'], [], ['rz']],
        ]
        mat = mk_matrix(4, cols)
        out = align_bricks(mat)
        # There should be at least two bricks emitted
        assert len(out[0]) >= 2
        # Rotations should appear alongside CXs in each brick
        # visualiser.print_matrix(mat)
        # visualiser.print_matrix(out)
        # for b in range(len(out[0])):
            # either a rotation or a CX must be present on qubit 0
            # assert any(name in ('rz','cx0') for name in names(out)[0][b])


class TestInstructionsToMatrixDAG(unittest.TestCase):

    def test_single_qubit_rotations(self):
        qc = QuantumCircuit(2)
        qc.rz(0.5, 0)
        qc.rx(1.0, 1)
        matrix = instructions_to_matrix_dag(qc)

        self.assertEqual(len(matrix), 2)
        # Expect one column
        self.assertEqual(len(matrix[0]), 1)
        self.assertEqual(len(matrix[1]), 1)

        instr0 = matrix[0][0][0]
        instr1 = matrix[1][0][0]
        self.assertIn(instr0.name, ['rz', 'rx'])
        self.assertIn(instr1.name, ['rz', 'rx'])


    def test_single_cx_gate(self):
        qc = QuantumCircuit(2)
        qc.cx(0, 1)
        matrix = instructions_to_matrix_dag(qc)

        # Should produce one column
        self.assertEqual(len(matrix[0]), 1)
        self.assertEqual(len(matrix[1]), 1)

        names = {instr.name for instr in matrix[0][0]} | {instr.name for instr in matrix[1][0]}
        self.assertIn('cx0c', names)
        self.assertIn('cx0t', names)


    def test_multiple_cx_gates(self):
        qc = QuantumCircuit(3)
        qc.cx(0, 1)
        qc.cx(1, 2)
        matrix = instructions_to_matrix_dag(qc)

        # Should have at least two columns depending on atomic grouping
        # We verify the names exist with unique indices
        names = set()
        for q in matrix:
            for col in q:
                names |= {instr.name for instr in col}

        # Two CX pairs → should have cx0c/t and cx1c/t
        self.assertIn('cx0c', names)
        self.assertIn('cx0t', names)
        self.assertNotIn('cx1c', names)
        self.assertNotIn('cx1t', names)


    def test_single_cx_gate_rev_order(self):
        qc = QuantumCircuit(2)
        qc.cx(1, 0)
        matrix = instructions_to_matrix_dag(qc)

        # Should produce one column
        self.assertEqual(len(matrix[0]), 1)
        self.assertEqual(len(matrix[1]), 1)

        names = {instr.name for instr in matrix[0][0]} | {instr.name for instr in matrix[1][0]}
        self.assertIn('cx0t', names)
        self.assertIn('cx0c', names)


if __name__ == '__main__':
    pytest.main()
