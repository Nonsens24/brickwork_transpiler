import unittest

import pytest
from src.brickwork_transpiler import utils
import numpy as np


def test_identical_states():
    a = [1, 0]
    b = [1, 0]

    assert utils.assert_equal_up_to_global_phase(a, b) is True


def test_global_phase_equivalence():
    a = [1/np.sqrt(2), 1j/np.sqrt(2)]
    global_phase = np.exp(1j * np.pi / 4)
    b = [x * global_phase for x in a]

    assert utils.assert_equal_up_to_global_phase(a, b) is True


def test_not_equal_states():
    a = [1, 0]
    b = [0, 1]

    with pytest.raises(
        AssertionError,
        match="States are not equal up to global phase"
    ):
        utils.assert_equal_up_to_global_phase(a, b)


def test_zero_norm_vector_raises():
    a = [0, 0]
    b = [1, 0]

    with pytest.raises(
        AssertionError,
        match=r"One of the states' norms is zero \(cannot compare\)"
    ):
        utils.assert_equal_up_to_global_phase(a, b)


def test_numerical_tolerance():
    a = [1/np.sqrt(2), 1/np.sqrt(2)]
    noise = 1e-7
    b = [1/np.sqrt(2) + noise, 1/np.sqrt(2) - noise]

    assert utils.assert_equal_up_to_global_phase(a, b, tol=1e-5) is True


def test_at_border_tolerance():
    a = [1/np.sqrt(2), 1/np.sqrt(2)]
    noise = 1e-6
    b = [1/np.sqrt(2) + noise, 1/np.sqrt(2) - noise]

    assert utils.assert_equal_up_to_global_phase(a, b) is True


def test_next_to_border_tolerance():
    a = [1/np.sqrt(2), 1/np.sqrt(2)]

    # make noise large enough that epsilon^2 > 1e-6
    noise = 1e-3 + 1e-4
    b = [1/np.sqrt(2) + noise, 1/np.sqrt(2) - noise]

    with pytest.raises(AssertionError):
        utils.assert_equal_up_to_global_phase(a, b)


def test_global_phase_4d():
    # a 4-dimensional state
    raw = np.array([1, 1j, -1, 0.5j], dtype=complex)
    state = raw / np.linalg.norm(raw)

    # apply a π/3 global phase
    phase = np.exp(1j * np.pi / 3)
    phased = phase * state

    # should return True
    assert utils.assert_equal_up_to_global_phase(state, phased) is True


from src.brickwork_transpiler.utils import feature_to_generator

class TestFeatureMatrixToGenerator(unittest.TestCase):

    def test_empty_matrix(self):
        with pytest.raises(ValueError) as e:
            feature_to_generator([])
        assert "must have at least one row" in str(e.value)

    def test_row_length_mismatch(self):
        # one row has length 3, the other length 2
        fm = [
            [0, 1, 0],
            [1, 0]
        ]
        with pytest.raises(ValueError) as e:
            feature_to_generator(fm)
        assert "same length l" in str(e.value)

    @pytest.mark.parametrize("fm", [
        [[0]],  # 1 row → N=1=2^0 OK
        [[0, 1], [1, 1]],  # 2 rows → N=2=2^1 OK
        [[0, 0, 0, 0]],  # 1 row, l=4
    ])
    def test_power_of_two_ok(self):
        # Only tests that no ValueError is raised on N=power-of-two check
        # May still error later if non-linear
        fm = [
            [[0]],  # 1 row → N=1=2^0 OK
            [[0, 1], [1, 1]],  # 2 rows → N=2=2^1 OK
            [[0, 0, 0, 0]],  # 1 row, l=4
        ]
        try:
            feature_to_generator(fm)
        except ValueError as e:
            # If non-linear, must be mismatch at i=0 or later, not N check
            assert "not a power of 2" not in str(e)

    def test_n_not_power_of_two(self):
        # 3 rows → N=3 is not a power of 2
        fm = [
            [0, 0],
            [1, 0],
            [0, 1]
        ]
        with pytest.raises(ValueError) as e:
            feature_to_generator(fm)
        assert "not a power of 2" in str(e.value)

    def test_non_linear_zero_vector(self):
        # First row must be zero for linearity
        fm = [
            [1, 0],  # f(0) != 0
            [0, 1]
        ]
        with pytest.raises(ValueError) as e:
            feature_to_generator(fm)
        assert "mismatch at i=0" in str(e.value)

    def test_non_linear_mismatch_at_other_i(self):
        # A 4-entry table that fails f(1)⊕f(2)=f(3)
        fm = [
            [0, 0, 0],  # f(0)
            [1, 0, 0],  # f(1)
            [0, 1, 0],  # f(2)
            [1, 1, 1],  # f(3) should be [1,1,0]
        ]
        with pytest.raises(ValueError) as e:
            feature_to_generator(fm)
        assert "mismatch at i=3" in str(e.value)

    def test_linear_2x1_identity(self):
        # N=2, q=1, l=1: f(0)=0, f(1)=1
        fm = [
            [0],
            [1]
        ]
        G = feature_to_generator(fm)
        # G should be [[1]] since f(1)=1
        assert G == [[1]]

    def test_linear_4x5_example(self):
        # the toy 4×5 example that is linear:
        fm = [
            [0, 0, 0, 0, 0],  # 0
            [1, 0, 1, 0, 1],  # g0
            [0, 1, 0, 1, 0],  # g1
            [1, 1, 1, 1, 1],  # g0⊕g1
        ]
        G = feature_to_generator(fm)
        # Expect l=5 rows, q=2 cols:
        expected = [
            [0, 1],  # G[:,0]=f(1), G[:,1]=f(2)
            [0, 0],
            [0, 1],
            [0, 0],
            [0, 1],
        ]
        # But f(1)=[1,0,1,0,1], f(2)=[0,1,0,1,0], so G[j][0]=fm[1][j], G[j][1]=fm[2][j]
        expected = [[fm[i][j] for i in (1, 2)] for j in range(5)]
        assert G == expected

    def test_random_linear_code(self):
        # generate a random 2-dimensional subspace in GF(2)^5
        q, l = 3, 5
        # pick G randomly
        G0 = np.random.randint(0, 2, (l, q)).tolist()
        # build fm by computing f(i)=G0*i mod 2
        fm = []
        for i in range(2 ** q):
            bits = [(i >> k) & 1 for k in range(q)]
            fm.append([sum(G0[j][k] * bits[k] for k in range(q)) % 2 for j in range(l)])
        # Now feature_to_generator should recover the same G0 (maybe with column order swapped)
        G1 = feature_to_generator(fm)
        # Verify it reproduces fm
        for i, row in enumerate(fm):
            bits = [(i >> k) & 1 for k in range(q)]
            reconstructed = [sum(G1[j][k] * bits[k] for k in range(q)) % 2 for j in range(l)]
            assert reconstructed == row
