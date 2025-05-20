from qiskit import QuantumCircuit
from src.brickwork_transpiler import decomposer, graph_builder, pattern_converter, utils, visualiser


def transpile(qc: QuantumCircuit, input_vector):

    # Decompose to CX, rzrxrz, id
    decomposed_qc = decomposer.decompose_qc_to_bricks_qiskit(qc, opt=3)

    # Optiise instruction matrix with dependency graph
    qc_mat, cx_mat = decomposer.instructions_to_matrix_dag(decomposed_qc)
    qc_mat_aligned = decomposer.align_bricks(cx_mat, qc_mat)

    # Build the graph from the optimised and formatted instruction matrix
    bw_graph_data = graph_builder.generate_brickwork_graph_from_instruction_matrix(qc_mat_aligned)

    # Get the networkx graph structure
    bw_nx_graph = graph_builder.to_networkx_graph(bw_graph_data)

    # Add angles and convert to Graphix pattern
    bw_pattern, col_map = pattern_converter.to_pattern(qc_mat_aligned, bw_nx_graph)

    # Calculate reference statevector
    reference_output = utils.calculate_ref_state_from_qiskit_circuit(bw_pattern, qc, input_vector)

    return bw_pattern, reference_output, col_map