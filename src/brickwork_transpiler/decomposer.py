from qiskit import QuantumCircuit, transpile
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGOpNode

def decompose_qc_to_bricks_qiskit(qc, opt=1, draw=False):

    basis = ['rz', 'rx', 'cx', 'id']  # include 'id' for explicit barriers/timing
    qc_basis = transpile(qc, basis_gates=basis, optimization_level=opt)

    if draw:
        print(qc_basis.draw())

    return qc_basis


def group_with_dag_atomic_rotations(qc: QuantumCircuit):
    """
    DAG + saturated two-phase scheduling, but:
      • In the rotation phase we IGNORE qubit conflicts,
        so that rz–rx–rz (or any chain) on the *same* qubit
        all go into the *same* rotation column.
      • In the CX phase we still pack non-overlapping CXs per column.
    """
    dag = circuit_to_dag(qc)
    op_nodes = list(dag.topological_op_nodes())
    total = len(op_nodes)
    scheduled = set()
    columns = []

    while len(scheduled) < total:
        # --- rotation phase: drain *all* ready rotations (rz/rx) ---
        rot_layer = []
        while True:
            added = False
            for node in op_nodes:
                if node in scheduled or node in rot_layer:
                    continue
                if node.op.name not in ('rz', 'rx'):
                    continue
                # only consider op-node predecessors
                preds = [p for p in dag.predecessors(node)
                         if isinstance(p, DAGOpNode)]
                # all preds must be already in scheduled OR in this layer
                if any((p not in scheduled and p not in rot_layer) for p in preds):
                    continue

                # *no* busy/conflict check here — we allow multiple on same qubit
                rot_layer.append(node)
                added = True

            if not added:
                break

        if rot_layer:
            columns.append([
                (node.op, node.qargs, node.cargs) for node in rot_layer
            ])
            scheduled.update(rot_layer)

        # --- CX phase: drain all ready non-conflicting CXs ---
        cx_layer = []
        busy_qubits = set()
        while True:
            added = False
            for node in op_nodes:
                if node in scheduled or node in cx_layer:
                    continue
                if node.op.name != 'cx':
                    continue
                preds = [p for p in dag.predecessors(node)
                         if isinstance(p, DAGOpNode)]
                if any((p not in scheduled and p not in cx_layer) for p in preds):
                    continue
                qs = [q._index for q in node.qargs]
                if any(q in busy_qubits for q in qs):
                    continue

                cx_layer.append(node)
                busy_qubits.update(qs)
                added = True

            if not added:
                break

        if cx_layer:
            columns.append([
                (node.op, node.qargs, node.cargs) for node in cx_layer
            ])
            scheduled.update(cx_layer)

        # sanity check: if we got *nothing* in both phases, there's a real stall
        if not rot_layer and not cx_layer:
            raise RuntimeError(
                "Stalled scheduling: no ready rotations or CXs; "
                "circuit may have a cycle"
            )

    return columns


def instructions_to_matrix_dag(qc: QuantumCircuit):
    """
    Builds the per-qubit × per-column instruction matrix
    from the atomic-rotation grouping above.
    """
    cols = group_with_dag_atomic_rotations(qc)
    n_q = qc.num_qubits
    n_c = len(cols)
    matrix = [[[] for _ in range(n_c)] for _ in range(n_q)]

    for c_idx, col in enumerate(cols):
        for instr, qargs, _ in col:
            for q in qargs:
                matrix[q._index][c_idx].append(instr)

    return matrix

##### Iterative method below not used #####

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
