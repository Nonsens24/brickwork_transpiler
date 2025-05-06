from qiskit import QuantumCircuit
from src.brickwork_transpiler import decomposer, graph_builder, pattern_converter


def transpile(qc: QuantumCircuit, input_vector):

    # Decompose to CX, rzrxrz, id
    decomposed_qc = decomposer.decompose_qc_to_bricks_qiskit(qc, opt=3)

    # Optiise instruction matrix with dependency graph
    qc_mat = decomposer.instructions_to_matrix_dag(decomposed_qc)

    # Align CX with brick long brick short brick layout
    qc_mat_formatted = decomposer.incorporate_bricks(qc_mat)

    # Build the graph from the optimised and formatted instruction matrix
    bw_graph_data = graph_builder.generate_brickwork_graph_from_instruction_matrix(qc_mat_formatted)

    # Get the networkx graph structure
    bw_nx_graph = graph_builder.to_networkx_graph(bw_graph_data)

    # Add angles and convert to Graphix pattern
    bw_pattern = pattern_converter.to_pattern(qc_mat_formatted, bw_nx_graph)

    # Calculate reference statevector
    reference_output = input_vector.evolve(qc)

    return bw_pattern, reference_output