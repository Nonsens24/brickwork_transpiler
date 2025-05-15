import copy

from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Instruction
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGOpNode

from qiskit.visualization import dag_drawer

from src.brickwork_transpiler import visualiser


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

    dag_drawer(dag) #Vis
    # Save the DAGDependency graph to a file
    dag_drawer(dag,  filename='dag_dependency.png')

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
    print("Building dependency graph...")
    cols = group_with_dag_atomic_rotations(qc)
    n_q = qc.num_qubits
    n_c = len(cols)
    matrix = [[[] for _ in range(n_c)] for _ in range(n_q)]

    print("Building instruction matrix")

    for c_idx, col in enumerate(cols):  # c_idx is column index
        cx_idx = 0
        for instr, qargs, _ in col:     # qargs[0] is the control qubit
            control_qubit = True
            for q in qargs:
                # print(f"ISNTR: {instr}")
                # print(f"qargs: {qargs[0]}")
                # Enhance Instruction data for CX
                instr_mut = instr.to_mutable()
                if instr_mut.name == 'cx' and control_qubit:
                    print(f"{cx_idx}c -- CONTROL inserted into name -- {instr_mut.name}")
                    instr_mut.name = f"cx{cx_idx}c"
                    control_qubit = False
                elif instr_mut.name == 'cx' and not control_qubit:
                    instr_mut.name = f"cx{cx_idx}t"
                    print(f"{cx_idx}t -- TARGET inserted into name -- {instr_mut.name}")

                matrix[q._index][c_idx].append(instr_mut)
            cx_idx += 1 # increment CX id after both parts of CX have been identified and logged

    visualiser.print_matrix(matrix)
    return matrix

# top control bot target
# ISNTR: Instruction(name='cx', num_qubits=2, num_clbits=0, params=[])
# qargs: (Qubit(QuantumRegister(2, 'q'), 0), Qubit(QuantumRegister(2, 'q'), 1))
# ISNTR: Instruction(name='cx', num_qubits=2, num_clbits=0, params=[])
# qargs: (Qubit(QuantumRegister(2, 'q'), 0), Qubit(QuantumRegister(2, 'q'), 1))

# bot target
# ISNTR: Instruction(name='cx', num_qubits=2, num_clbits=0, params=[])
# qargs: (Qubit(QuantumRegister(2, 'q'), 1), Qubit(QuantumRegister(2, 'q'), 0))
# ISNTR: Instruction(name='cx', num_qubits=2, num_clbits=0, params=[])
# qargs: (Qubit(QuantumRegister(2, 'q'), 1), Qubit(QuantumRegister(2, 'q'), 0))


def enumerate_cx_in_cols(matrix):
    num_qubits = len(matrix)
    num_cols   = len(matrix[0])

    for c in range(num_cols):
        cx_counter = 0
        r = 0
        while r < num_qubits - 1:
            cell = matrix[r][c]
            if cell and 'cx' in cell[0].name:
                print(f"Cell name: {cell[0].name}")

                # Qiskit instruction
                if isinstance(cell[0], Instruction):
                    m0 = cell[0].to_mutable()
                    m1 = matrix[r + 1][c][0].to_mutable()
                # Dummy test instruction
                else:
                    m0 = copy.deepcopy(cell[0])
                    m1 = copy.deepcopy(matrix[r+1][c][0])

                # Set values
                m0.name = f"cx{cx_counter}"
                m1.name = f"cx{cx_counter}"

                # Update
                matrix[r][c][0]   = m0
                matrix[r+1][c][0] = m1

                cx_counter += 1
                r += 2  # CX comes in pairs
                continue

            r += 1

    return matrix


def incorporate_bricks(matrix):
    n_q = len(matrix)
    n_c = len(matrix[0])
    new_cols = []
    brick_idx = 0

    matrix = enumerate_cx_in_cols(matrix)

    visualiser.print_matrix(matrix)

    for c in range(n_c):
        # 1) collect rotations
        rotations = [
            [instr for instr in matrix[q][c] if instr.name in ('rz', 'rx')]
            for q in range(n_q)
        ]
        # 2) map each numeric suffix to its starting qubit i
        num_to_i = {}
        for i in range(n_q - 1):
            names_i   = {instr.name for instr in matrix[i][c]   if instr.name.startswith('cx')}
            names_ip1 = {instr.name for instr in matrix[i+1][c] if instr.name.startswith('cx')}
            for full in names_i & names_ip1:
                num_to_i[full[2:]] = i

        # 3) no CNOTs → pure‐rotation brick
        if not num_to_i:
            new_cols.append(rotations)
            brick_idx += 1
            continue

        # 4) schedule in as few bricks as possible, grouping same‐parity, non‐conflicting CNOTs
        unscheduled = list(num_to_i.keys())
        while unscheduled:
            parity   = brick_idx % 2
            # pick all groups whose i has correct parity
            to_place = [n for n in unscheduled if (num_to_i[n] % 2) == parity]
            if not to_place:
                # nothing fits → blank brick
                new_cols.append([[] for _ in range(n_q)])
                brick_idx += 1
                continue

            # build a brick with all rotations + all selected cx<n>
            layer = []
            for q in range(n_q):
                ops = list(rotations[q])
                for n in to_place:
                    i = num_to_i[n]
                    if q == i or q == i+1:
                        # grab the cx<n> on this wire
                        for instr in matrix[q][c]:
                            if instr.name == f'cx{n}':
                                ops.append(instr)
                                break
                layer.append(ops)

            new_cols.append(layer)
            brick_idx += 1
            # mark them done
            for n in to_place:
                unscheduled.remove(n)

    # transpose back to [qubit][brick]
    return [
        [new_cols[b][q] for b in range(len(new_cols))]
        for q in range(n_q)
    ]



# def incorporate_bricks(matrix):
#     """
#     Inserts the minimal number of identity‐only bricks so that, for each original column c:
#       • Each CNOT(i,i+1) goes into a brick whose index % 2 == i % 2.
#       • If multiple CNOTs in c have mixed parity, they’re split into separate bricks.
#       • All rotations (rz/rx) appear in every emitted brick.
#
#     Args:
#       matrix: List[List[List[Instruction]]], shape (n_qubits, n_cols),
#               where matrix[q][c] is the list of Instruction objects
#               (rz, rx, cx) acting on qubit q in original column c.
#
#     Returns:
#       new_matrix: same format, with identity‐only columns inserted
#                   to enforce the even/odd‐brick requirement.
#     """
#     n_q = len(matrix)
#     n_c = len(matrix[0])
#     new_cols = []
#     brick_idx = 0  # counts output bricks (0-based)
#
#     for c in range(n_c):
#         # 1) Extract all rz/rx rotations from this column
#         rotations = [
#             [instr for instr in matrix[q][c] if instr.name in ('rz', 'rx')]
#             for q in range(n_q)
#         ]
#
#         # 2) Reconstruct the list of adjacent‐qubit CNOTs in this column
#         #    by looking for pairs (i, i+1) both carrying a cx.
#         cx_pairs = []
#         for i in range(n_q - 1):
#             has_i = any(instr.name == 'cx' for instr in matrix[i][c])
#             has_ip1 = any(instr.name == 'cx' for instr in matrix[i + 1][c])
#             if has_i and has_ip1:
#                 cx_pairs.append(i)  # store the “i” of (i, i+1)
#
#         if not cx_pairs:
#             # --- No CNOTs: one brick carrying just the rotations ---
#             new_cols.append(rotations)
#             brick_idx += 1
#             continue
#
#         # 3) Schedule each CNOT(i,i+1) in its own or shared brick, in pass order:
#         unscheduled = list(cx_pairs)
#         while unscheduled:
#             curr_parity = brick_idx % 2
#             # find all pairs that can go here
#             to_place = [i for i in unscheduled if (i % 2) == curr_parity]
#
#             if not to_place:
#                 # no matching CNOTs ⇒ pad an identity brick
#                 new_cols.append([[] for _ in range(n_q)])
#                 brick_idx += 1
#                 continue
#
#             # build a brick: add all rotations + only the cx for these pairs
#             layer = []
#             for q in range(n_q):
#                 ops = list(rotations[q])
#                 for i in to_place:
#                     if q == i or q == i + 1:
#                         # grab exactly one of the original cx instrs for this wire
#                         for instr in matrix[q][c]:
#                             if instr.name == 'cx':
#                                 ops.append(instr)
#                                 break
#                 layer.append(ops)
#
#             new_cols.append(layer)
#             brick_idx += 1
#
#             # mark those CNOTs done
#             for i in to_place:
#                 unscheduled.remove(i)
#
#     # transpose back into [qubit][brick] format
#     n_new = len(new_cols)
#     return [
#         [new_cols[b][q] for b in range(n_new)]
#         for q in range(n_q)
#     ]


##### Iterative method below -- not used #####

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
