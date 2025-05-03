import numpy as np

def assert_equal_up_to_global_phase(state1, state2, tol=1e-6):
    """
    Assert that two quantum state vectors are equal up to global phase.

    Parameters:
    - state1, state2: iterable of complex numbers (e.g., output of a simulator)
    - tol: numerical tolerance for isclose()

    Raises:
    - AssertionError with diagnostic info if the assertion fails.
    """
    state1 = np.array(state1, dtype=complex)
    state2 = np.array(state2, dtype=complex)

    # Normalize (just in case)
    norm1 = np.linalg.norm(state1)
    norm2 = np.linalg.norm(state2)

    if np.isclose(norm1, 0.0, atol=tol) or np.isclose(norm2, 0.0, atol=tol):
        raise AssertionError("One of the states is zero-norm (cannot compare).")

    state1 /= norm1
    state2 /= norm2

    inner_product = np.vdot(state1, state2)
    magnitude = np.abs(inner_product)

    if not np.isclose(magnitude, 1.0, atol=tol, rtol=0.0): # added rtol for border checking
        raise AssertionError(
            f"States are not equal up to global phase.\n"
            f"Inner product: {inner_product}\n"
            f"Absolute value: {magnitude:.6f} (should be close to 1)"
        )
        # return False

    else: return True
