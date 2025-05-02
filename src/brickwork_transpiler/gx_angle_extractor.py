from graphix.transpiler import Circuit
import numpy as np

# Used for sanity check with measurement angles

def print_pattern():

    print("Graphix angle transpilation (no optimisation):")

    circuit = Circuit(2)
    circuit.rz(0, np.pi/4)
    circuit.rx(0, np.pi/8)
    circuit.rz(0, np.pi/16)
    pattern = circuit.transpile().pattern

    pattern.print_pattern()

def print_standard_pattern():
    print("Graphix angle transpilation (with optimisation):")

    circuit = Circuit(2)
    circuit.rz(0, np.pi / 4)
    circuit.rx(0, np.pi / 8)
    circuit.rz(0, np.pi / 16)
    pattern = circuit.transpile().pattern
    pattern.standardize()
    pattern.shift_signals()

    pattern.print_pattern()