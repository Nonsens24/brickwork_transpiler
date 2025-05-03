import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

import bricks
import utils
import visualiser

def main():

    # print("Laying a brick:")
    # pattern = bricks.arbitrary_brick(1/4, 1/4, 1/4)
    # pattern.print_pattern()

    # TODO: get graph structure from pattern
    # visualiser.plot_graph(pattern)

    # ARbitrary Rotation gate:
    # outstate = pattern.simulate_pattern(backend='statevector').flatten()
    # print("brick MBQC output:", outstate)
    #
    # qc = QuantumCircuit(1)
    #
    # qc.h(0)
    # qc.rz(np.pi * 1/4, 0)
    # qc.rx(np.pi * 1/4, 0)
    # qc.rz(np.pi * 1/4, 0)

    # CX gate:
    print("Laying a brick:")
    pattern = bricks.CX_brick()

    outstate = pattern.simulate_pattern(backend='statevector').flatten()
    print("brick MBQC output:", outstate)

    qc = QuantumCircuit(2)

    # Initialise to |+>
    qc.h(0)
    qc.h(1)

    # cnot them
    qc.cx(0, 1)

    sv2 = Statevector.from_instruction(qc).data
    print("reference output: ", sv2)

    if utils.assert_equal_up_to_global_phase(outstate, sv2):
        print("Same up to global phase!")


if __name__ == "__main__":
    main()
