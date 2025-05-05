from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


def decompose_qc_to_bricks_qiskit(qc):

    basis = ['rz', 'rx', 'cx', 'id']  # include 'id' for explicit barriers/timing
    qc_basis = transpile(qc, basis_gates=basis, optimization_level=0)

    print(qc_basis.draw())

    return qc_basis


def group_into_columns(qc: QuantumCircuit):
    """
    Takes a circuit already transpiled to ['rz','rx','cx']
    and returns a list of "columns", each column containing
    either all single-qubit rotations (RZ/RX) or all CX gates,
    alternating between the two.

    Args:
        qc (QuantumCircuit): Transpiled circuit with only 'rz', 'rx', and 'cx' gates.

    Returns:
        List[List[Tuple[instruction, qargs, cargs]]]:
            A list of columns, where each column is a list of
            (instruction, qargs, cargs) tuples.
    """
    data = qc.data  # Flattened list of (instruction, qargs, cargs)
    columns = []    # To store completed columns
    current = []    # Accumulates instructions for the current column
    collecting_singles = True  # Start by collecting single-qubit rotations

    for instr, qargs, cargs in data:
        is_cx = (instr.name == 'cx')  # Check if this is a CX gate

        # Transition from collecting singles to CXs
        if collecting_singles and is_cx:
            if current:
                columns.append(current)  # Save completed single-qubit column
            current = [(instr, qargs, cargs)]
            collecting_singles = False  # Now collect CX gates

        # Transition from collecting CXs to singles
        elif not collecting_singles and not is_cx:
            if current:
                columns.append(current)  # Save completed CX column
            current = [(instr, qargs, cargs)]
            collecting_singles = True  # Now collect single-qubit rotations

        else:
            # Continue adding to the current column
            current.append((instr, qargs, cargs))

    # Append the final column if non-empty
    if current:
        columns.append(current)

    return columns


def instuctions_to_matrix(qc: QuantumCircuit):
    """
    Converts the grouped columns into a 2D matrix of shape
    (num_qubits x num_columns), where each entry is a list of
    Gate objects acting on that specific qubit at that stage.

    This is the transpose of columns_as_matrix: rows now correspond
    to qubits, and columns correspond to alternating RZ–RX–RZ or CX layers.

    Args:
        qc (QuantumCircuit): Transpiled circuit used for grouping.

    Returns:
        List[List[List[instruction]]]: A matrix where matrix[q][c]
            is a list of instructions on qubit q in column c.
    """
    # First, get the raw columns grouping
    cols = group_into_columns(qc)
    n_qubits = qc.num_qubits
    n_cols = len(cols)

    # Initialize an empty matrix: one row per qubit, one column per stage
    matrix = [[[] for _ in range(n_cols)] for _ in range(n_qubits)]

    # Populate matrix[q][c] by inspecting each column's operations
    for c_idx, col in enumerate(cols):
        for instr, qargs, _ in col:
            for qubit in qargs:
                matrix[qubit._index][c_idx].append(instr)

    return matrix
