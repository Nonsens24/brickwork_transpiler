from qiskit import QuantumCircuit
from . import decomposer, graph_builder, pattern_converter, utils, visualiser



def transpile(qc: QuantumCircuit, input_vector=None, routing_method=None, layout_method=None, return_mat: bool = False,
              file_writer=None, with_ancillas=True, plot_decomposed: bool = False, opt = 3):

    # Decompose to CX, rzrxrz, id   -   Need opt = 3 for SU(2) rotation merging
    print("Decomposing circuit...")
    decomposed_qc = decomposer.decompose_qc_to_bricks_qiskit(qc, opt=opt, routing_method=routing_method,
                                                              layout_method=layout_method, file_writer=file_writer,
                                                             with_ancillas=with_ancillas, draw=plot_decomposed)

    print("Creating instruction matrix from DAG...")
    # Optimise Instruction matrix with dependency graph
    qc_mat, cx_mat = decomposer.instructions_to_matrix_dag(decomposed_qc)

    print("Aligning instruction matrix...")
    # Align gates according to brickwork topology
    qc_mat_aligned = decomposer.align_bricks(cx_mat, qc_mat)

    # Depth analysis only requires the number of columns of the aligned matrix, saving resources
    if return_mat:
        return qc_mat_aligned

    print("Building graph structure...")
    # Build the graph from the optimised and formatted instruction matrix
    bw_graph_data = graph_builder.generate_brickwork_graph_from_instruction_matrix(qc_mat_aligned)

    # Get the networkx graph structure
    bw_nx_graph = graph_builder.to_networkx_graph(bw_graph_data)

    # Add angles and convert to Graphix pattern
    print("Calculating measurement dependencies...")
    bw_pattern, col_map = pattern_converter.to_pattern(qc_mat_aligned, bw_nx_graph)

    return bw_pattern, col_map, decomposed_qc