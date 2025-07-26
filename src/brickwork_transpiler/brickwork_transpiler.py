from qiskit import QuantumCircuit
from src.brickwork_transpiler import decomposer, graph_builder, pattern_converter, utils, visualiser


def transpile(qc: QuantumCircuit, input_vector=None, routing_method=None, layout_method=None):

    # Decompose to CX, rzrxrz, id   -   Need opt = 3 for SU(2) rotation merging
    decomposed_qc = decomposer.decompose_qc_to_bricks_qiskit(qc, opt=3, routing_method=routing_method, layout_method=layout_method)

    print(decomposed_qc.draw())

    # Optiise instruction matrix with dependency graph
    qc_mat, cx_mat = decomposer.instructions_to_matrix_dag(decomposed_qc)
    visualiser.print_matrix(qc_mat)
    qc_mat_aligned = decomposer.align_bricks(cx_mat, qc_mat)
    visualiser.print_matrix(qc_mat)

    # print("qc_mat:")
    # visualiser.print_matrix(qc_mat)
    # print("cx_mat:")
    # visualiser.print_matrix(cx_mat)
    # print("qc_mat_aligned:")
    # visualiser.print_matrix(qc_mat_aligned)

    # Build the graph from the optimised and formatted instruction matrix
    bw_graph_data = graph_builder.generate_brickwork_graph_from_instruction_matrix(qc_mat_aligned)

    # Get the networkx graph structure
    bw_nx_graph = graph_builder.to_networkx_graph(bw_graph_data)

    # Add angles and convert to Graphix pattern
    print("Calculating measurement dependencies...")
    bw_pattern, col_map = pattern_converter.to_pattern(qc_mat_aligned, bw_nx_graph)

    # Calculate reference statevector
    # reference_output = utils.calculate_ref_state_from_qiskit_circuit(bw_pattern, qc, input_vector)

    return bw_pattern, col_map #reference_output, col_map