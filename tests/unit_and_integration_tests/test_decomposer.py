import pytest

from src.brickwork_transpiler import visualiser
from src.brickwork_transpiler.decomposer import incorporate_bricks
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


def test_no_cx_only_rotations():
    cols = [
        [['rz','rx'], ['rz'], ['rx']],
        [['rz'], ['rx','rz'], ['rz','rx','rz']],
    ]
    mat = mk_matrix(3, cols)
    out = incorporate_bricks(mat)
    # No padding: same number of columns
    assert len(out[0]) == len(cols)
    # Check content matches input
    out_names = names(out)
    for c in range(len(cols)):
        for q in range(3):
            assert out_names[q][c] == cols[c][q]


def test_pure_even_cx_alignment():
    cols = [
        [['rz','rx'], ['rz'], ['rx']],
        [['rz'], ['rx','rz'], ['rz','rx','rz']],
    ]
    mat = mk_matrix(3, cols)
    out = incorporate_bricks(mat)
    # No padding: same number of columns
    assert len(out[0]) == len(cols)
    # Check content matches input
    for c in range(len(cols)):
        for q in range(3):
            assert names(out)[q][c] == cols[c][q]


def test_pure_even_cx_alignment2():
    # Single column with cx on qubits 0-1 (even parity)
    cols = [
        [['cx'], ['cx'], []],  # q0->cx, q1->cx, q2->none
    ]
    mat = mk_matrix(3, cols)
    out = incorporate_bricks(mat)
    # brick_idx starts at 0 (even), so no padding needed
    assert len(out[0]) == 1
    # CX should appear on qubits 0 and 1 only
    assert names(out)[0][0] == ['cx0']
    assert names(out)[1][0] == ['cx0']
    assert names(out)[2][0] == []


def test_pure_odd_cx_alignment():
    # Single column with cx on qubits 1-2 (odd parity)
    cols = [
        [[], ['cx'], ['cx']],
    ]
    mat = mk_matrix(3, cols)
    out = incorporate_bricks(mat)
    # initial brick_idx=0 (even), needs odd -> one pad, then CX brick
    assert len(out[0]) == 2
    # First brick should be identity
    assert all(cell == [] for cell in names(out)[0][0:1])
    # Second brick has CXs on qubits 1 and 2
    assert names(out)[1][1] == ['cx0']
    assert names(out)[2][1] == ['cx0']

def test_mixed_same_parity_cx():
    # Two independent CNOTs on (0,1) and (2,3), both even parity and non-adjacent
    cols = [
        [['cx'], ['cx'], ['cx'], ['cx']],
    ]
    mat = mk_matrix(4, cols)
    out = incorporate_bricks(mat)
    # visualiser.print_matrix(mat)
    # visualiser.print_matrix(out)
    # Both CNOTs should be in a single brick without padding
    assert len(out[0]) == 1
    names_out = names(out)
    # Check cx(0,1)
    assert 'cx0' in names_out[0][0] and 'cx0' in names_out[1][0]
    # Check cx(3,4)
    assert 'cx1' in names_out[2][0] and 'cx1' in names_out[3][0]



def test_mixed_diff_parity_cx():
    # Single column with cx on 0-1 (even) and 3-4 (odd)
    cols = [
        [['cx'], ['cx'], [], ['cx'], ['cx']],
    ]
    mat = mk_matrix(5, cols)
    out = incorporate_bricks(mat)
    # Should split into two bricks
    assert len(out[0]) == 2
    # First brick handles cx(0,1)
    assert 'cx0' in names(out)[0][0]
    assert 'cx0' in names(out)[1][0]
    assert names(out)[3][0] == []
    # Second brick handles cx(3,4)
    assert 'cx1' in names(out)[3][1]
    assert 'cx1' in names(out)[4][1]


# def test_rotations_with_cx():
#     # Two columns: rotations only, then mixed CX and rotations
#     cols = [
#         [['rz'], ['rx'], ['rz'], []],
#         [['cx'], ['cx'], [], ['rz']],
#     ]
#     mat = mk_matrix(4, cols)
#     out = incorporate_bricks(mat)
#     # There should be at least two bricks emitted
#     assert len(out[0]) >= 2
#     # Rotations should appear alongside CXs in each brick
#     visualiser.print_matrix(out)
#     for b in range(len(out[0])):
#         # either a rotation or a CX must be present on qubit 0
#         assert any(name in ('rz','cx0') for name in names(out)[0][b])


if __name__ == '__main__':
    pytest.main()
