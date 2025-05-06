import numpy as np
from graphix.states import BasicStates
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

import bricks
import utils
import visualiser
from src.brickwork_transpiler import decomposer, graph_builder, pattern_converter
from src.brickwork_transpiler.visualiser import plot_graph


def main():

    # 1) Create the |++> state directly
    psi = Statevector.from_label('++')  # two-qubit plus state

    # 2) Define your 2-qubit circuit (no H gates needed)
    qc = QuantumCircuit(2)
    qc.cx(0, 1)


    # Decomposer
    # qc = QuantumCircuit(2)
    # qc.h(1)
    # qc.h(0)
    # qc.cx(0, 1)

    # qc.h(2)
    # qc.t(3)
    # qc.s(3)
    # qc.cx(1, 2)
    #
    # qc.cx(3,4)
    #
    # qc.t(3)
    # qc.h(2)
    # qc.cx(2, 3)
    # qc.t(3)
    #
    # qc.cx(5, 6)


    decomposed_qc = decomposer.decompose_qc_to_bricks_qiskit(qc, 3)

    qc_mat = decomposer.instructions_to_matrix_dag(decomposed_qc)
    visualiser.print_matrix(qc_mat)

    qc_mat = decomposer.incorporate_bricks(qc_mat)
    visualiser.print_matrix(qc_mat)

    bw_graph_data = graph_builder.generate_brickwork_graph_from_instruction_matrix(qc_mat)

    bw_nx_graph = graph_builder.to_networkx_graph(bw_graph_data)

    visualiser.plot_graph(bw_nx_graph)

    bw_pattern = pattern_converter.to_pattern(qc_mat, bw_nx_graph)
    # bw_pattern.print_pattern(lim = 10000)

    visualiser.plot_brickwork_graph_from_pattern(bw_pattern)
    # visualiser.visualize_brickwork_graph(bw_pattern)

    print("Starting simulation of bw pattern. This might take a while...")
    outstate = bw_pattern.simulate_pattern(backend='statevector').flatten()
    print("Graphix simulator output:", outstate)

    # Calculate reference statevector
    psi_out = psi.evolve(qc)
    print("Qiskit reference state vector: ", psi_out.data)

    # sv2 = Statevector.from_instruction(qc).data
    # print("Qiskit reference output: ", sv2)

    utils.assert_equal_up_to_global_phase(outstate, psi_out.data)

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
