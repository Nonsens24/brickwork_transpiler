from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from src.brickwork_transpiler import decomposer, visualiser, graph_builder, pattern_converter, utils


def test_bell_proj_from_plus():

    # 1) Create the |++> state directly
    psi = Statevector.from_label('++')  # two-qubit plus state

    # 2) Define your 2-qubit circuit (no H gates needed)
    qc = QuantumCircuit(2)
    qc.cx(0, 1)

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

    # Simulate the generated pattern
    outstate = bw_pattern.simulate_pattern(backend='statevector').flatten()

    # Calculate reference statevector
    psi_out = psi.evolve(qc)

    # Compare output up to global phase
    assert utils.assert_equal_up_to_global_phase(outstate, psi_out.data)