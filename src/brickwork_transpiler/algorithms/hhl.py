import linear_solvers
import numpy as np
from qiskit import Aer, QuantumCircuit
from qiskit.utils import QuantumInstance
import linear_solvers.hhl
from qiskit.circuit.library import QFT  # used internally by HHL

# git clone https://github.com/anedumla/quantum_linear_solvers.git
# cd quantum_linear_solvers
# pip install .


def generate_example_hhl_QC() -> QuantumCircuit:
    # 1. Define A and b
    #    A = [[1, 0.5], [0.5, 1]], which is Hermitian and well-conditioned.
    A = np.array([[1.0, 0.5],
                  [0.5, 1.0]])
    #    b = [1, 0] normalized to unit length for state prep.
    b = np.array([1.0, 0.0])
    b = b / np.linalg.norm(b)

    # 2. Create a QuantumInstance using Aer (statevector simulator)
    quantum_instance = QuantumInstance(
        Aer.get_backend("aer_simulator_statevector")
    )

    # 3. Initialize the HHL solver
    #    Internally, HHL will set up registers for:
    #      - |b> state preparation
    #      - QPE on unitary exp(i A t)
    #      - Controlled rotation for λ_j^{-1}
    #      - Inverse QPE
    hhl_solver = linear_solvers.HHL(quantum_instance=quantum_instance)

    # 4. Build the QuantumCircuit
    #    'construct_circuit' accepts the matrix and the vector as NumPy arrays (or circuits).
    hhl_circuit = hhl_solver.construct_circuit(A, b)

    # 5. Draw the circuit with matplotlib
    #    Note: In Jupyter, you can use hhl_circuit.draw("mpl"). Here we print ASCII.
    print(hhl_circuit.draw(output="text"))

    return hhl_circuit
