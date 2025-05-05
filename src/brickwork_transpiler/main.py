import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

import bricks
import utils
import visualiser
from src.brickwork_transpiler import decomposer


def main():

    # Decomposer
    qc = QuantumCircuit(4)

    qc.h(2)
    qc.h(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.s(0)
    qc.s(0)
    qc.t(1)
    qc.cx(0, 1)


    decomposed_qc = decomposer.decompose_qc_to_bricks_qiskit(qc)

    qc_mat = decomposer.instuctions_to_matrix(decomposed_qc)

    visualiser.print_matrix(qc_mat)

    # print("Laying a brick:")
    # pattern = bricks.arbitrary_brick(1/4, 1/4, 1/4)
    # pattern.print_pattern()
    #
    # # TODO: get graph structure from pattern
    # # visualiser.plot_graph(pattern)
    #
    # # ARbitrary Rotation gate:
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
    # print("Laying a brick:")
    # pattern = bricks.CX_bottom_target_brick()
    #
    # outstate = pattern.simulate_pattern(backend='statevector').flatten()
    # print("brick MBQC output:", outstate)
    #
    # qc = QuantumCircuit(2)
    #
    # # Initialise to |+>
    # qc.h(0)
    # qc.h(1)
    #
    # # cnot them
    # qc.cx(0, 1)
    #
    # sv2 = Statevector.from_instruction(qc).data
    # print("reference output: ", sv2)
    #
    # if utils.assert_equal_up_to_global_phase(outstate, sv2):
    #     print("Same up to global phase!")


if __name__ == "__main__":
    main()
