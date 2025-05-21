from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Instruction
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGOpNode
from qiskit.visualization import dag_drawer

from src.brickwork_transpiler import visualiser


def decompose_qc_to_bricks_qiskit(qc, opt=1, draw=False):

    # print("Decomposing Quantum circuit to generator set...", end=" ")

    basis = ['rz', 'rx', 'cx', 'id']  # include 'id' for explicit barriers/timing
    qc_basis = transpile(qc, basis_gates=basis, optimization_level=opt)

    if draw:
        print(qc_basis.draw())

    # print("Done")

    return qc_basis


def group_with_dag_atomic_rotations_layers(qc: QuantumCircuit):
    """
    DAG + saturated two-phase scheduling:
      • In the rotation phase we IGNORE qubit conflicts,
        so that rz–rx–rz (or any chain) on the *same* qubit
        all go into the *same* rotation column.
      • In the CX phase does pack non-overlapping CXs per column.
    """
    # print("Building dependency graph...", end=" ")

    dag = circuit_to_dag(qc)
    op_nodes = list(dag.topological_op_nodes())
    total = len(op_nodes)
    scheduled = set()
    columns = []

    dag_drawer(dag) #Vis
    # Save the DAGDependency graph to a file
    if True:   # add save option
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

                # *no* busy/conflict check here — allow multiple on same qubit
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

    # print("Done")
    return columns


from qiskit.converters import circuit_to_dag
from qiskit import QuantumCircuit
from qiskit.dagcircuit import DAGOpNode

def group_with_dag_atomic_mixed(qc: QuantumCircuit):
    """
    DAG + saturated single‐phase scheduling:
      1) In each column, drain *all* ready rotations (rz, rx)—
         merging entire chains of them (e.g. rz→rx→rz) into one layer,
         with no busy‐qubit checks among these rotations.
      2) Then greedily pack ready CXs that don’t touch any qubit
         already used by those rotations or by each other.
      Repeat until every operation is scheduled.
    """
    dag = circuit_to_dag(qc)
    op_nodes = list(dag.topological_op_nodes())
    total = len(op_nodes)
    scheduled = set()
    columns = []

    while len(scheduled) < total:
        col_nodes = []
        busy_qubits = set()

        # --- 1) Drain *all* ready rotations into rot_layer ---
        rot_layer = []
        while True:
            added = False
            for node in op_nodes:
                if node in scheduled or node in rot_layer:
                    continue
                if node.op.name not in ('rz', 'rx'):
                    continue
                # only consider op‐node predecessors
                preds = [p for p in dag.predecessors(node)
                         if isinstance(p, DAGOpNode)]
                # allow preds to be either already scheduled or in rot_layer
                if any((p not in scheduled and p not in rot_layer) for p in preds):
                    continue
                # this rotation is ready — absorb into this rotation‐chain
                rot_layer.append(node)
                added = True
            if not added: # All nodes were added
                break

        # add all those rotations at once (no busy_qubits update here)
        if rot_layer:
            col_nodes.extend(rot_layer)

        # --- 2) Greedily pack CXs respecting busy_qubits ---
        # mark any qubits used by these rotations as busy for CXs
        busy_qubits.update(q._index for node in rot_layer for q in node.qargs)

        added = True
        while added:
            added = False
            for node in op_nodes:
                if node in scheduled or node in col_nodes:
                    continue
                if node.op.name != 'cx':
                    continue
                # dep check
                preds = [p for p in dag.predecessors(node)
                         if isinstance(p, DAGOpNode)]
                if any(p not in scheduled for p in preds):
                    continue
                # conflict check
                qs = [q._index for q in node.qargs]
                if any(q in busy_qubits for q in qs):
                    continue
                # can schedule this CX
                col_nodes.append(node)
                busy_qubits.update(qs)
                added = True

        if not col_nodes:
            raise RuntimeError("Stalled scheduling: circular dependency detected.")

        # record this mixed column
        columns.append([
            (node.op, node.qargs, node.cargs)
            for node in col_nodes
        ])
        scheduled.update(col_nodes)

    return columns


def instructions_to_matrix_dag(qc: QuantumCircuit):
    """
    Builds the per-qubit × per-column instruction matrix
    from the atomic-rotation grouping above.
    Also encodes all necessary gate info into the matrix
    """

    cols = group_with_dag_atomic_mixed(qc)

    n_q = qc.num_qubits
    n_c = len(cols)
    matrix = [[[] for _ in range(n_c)] for _ in range(n_q)]
    cx_matrix = [[[] for _ in range(n_c)] for _ in range(n_q)]
    # rx_matrix = [[[] for _ in range(n_c)] for _ in range(n_q)]

    # print("Building instruction matrix...", end=" ")
    cx_idx = 0  #Unique cx identifier
    for c_idx, col in enumerate(cols):  # c_idx is the column index

        for instr, qargs, _ in col:     # qargs[0] is the control qubit
            control_qubit = True        # Hence the first iteration is always control
            for q in qargs:
                # Enhance Instruction data for CX
                instr_mut = instr.to_mutable()
                if instr_mut.name == 'cx' and control_qubit:
                    instr_mut.name = f"cx{cx_idx}c"
                    cx_matrix[q._index][c_idx].append(instr_mut)  # Create cx matrix for alignment
                    control_qubit = False
                elif instr_mut.name == 'cx' and not control_qubit:
                    instr_mut.name = f"cx{cx_idx}t"
                    cx_matrix[q._index][c_idx].append(instr_mut)

                matrix[q._index][c_idx].append(instr_mut)
            if instr.name.startswith('cx'):
                cx_idx += 1 # increment CX id after both parts of CX have been identified and logged

    # print("Done")
    return matrix, cx_matrix

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

# def align_cx_matrix(cx_matrix):
#
#     for col_id, col in enumerate(cx_matrix):
#         for row_id, row in enumerate(col):
#             for instr in row:
#                 if instr.name.startswith('cx'):

# def align_cx_matrix(cx_matrix):
#     """
#     Shift each CX‐brick (two rows with same cx#) rightwards
#     so that top_row_index %2 == column_index %2.
#     Modifies cx_matrix in place and returns it.
#     """
#     n_rows = len(cx_matrix)
#     if n_rows == 0:
#         return cx_matrix
#
#     cx_matrix_no_shift = cx_matrix  # Safe this for alignment
#
#     print("cx_matrix unaligned:")
#     visualiser.print_matrix(cx_matrix)
#
#     # make sure every row has the same number of columns
#     n_cols = max(len(r) for r in cx_matrix)
#     for row in cx_matrix:
#         if len(row) < n_cols:
#             row.extend([[]] * (n_cols - len(row)))
#
#     # scan columns from rightmost to leftmost
#     for col_id in range(n_cols - 1, -1, -1):
#         # group → list of row‐indices in this column
#         group_rows: dict[str, List[int]] = {}
#         for row_id in range(n_rows):
#             for instr in cx_matrix[row_id][col_id]:
#                 if instr.name.startswith('cx'):
#                     # extract the numeric group ID between "cx" and final suffix
#                     gid = instr.name[2:-1]
#                     group_rows.setdefault(gid, []).append(row_id)
#
#         # now for each 2-row brick in this column...
#         for gid, rows in group_rows.items():
#             top = min(rows)
#             # if parity mismatches, shift both rows into col_id+1
#             if top % 2 != col_id % 2:
#                 target = col_id + 1
#
#                 # If current instruction to be shifted + 2 = the same as non shifted matrix + 2 it means that the next
#                 # cx is not shifted but this one is, shifting the rotation into the next cx
#                 if cx_matrix[gid][target + 2] == cx_matrix_no_shift[gid][col_id+2]:
#                     # TODO: shift the gate by two and make the whole matrix work again...
#                 # if we run off the right edge, extend all rows by one empty column
#                 if target >= n_cols:
#                     for r in cx_matrix:
#                         r.append([])
#                     n_cols += 1
#
#                 for r in rows:
#                     # pull out just the cx# instructions for this gid
#                     moving = [
#                         ins for ins in cx_matrix[r][col_id]
#                         if ins.name.startswith('cx') and ins.name[2:-1] == gid
#                     ]
#                     # remove them from the old spot
#                     cx_matrix[r][col_id] = [
#                         ins for ins in cx_matrix[r][col_id]
#                         if not(ins in moving)
#                     ]
#                     # place them (as a list) into the new column
#                     cx_matrix[r][target] = moving
#
#     print("cx_matrix aligned:")
#     visualiser.print_matrix(cx_matrix)
#
#     return cx_matrix


import copy
from typing import List, Dict

import copy
from typing import List, Dict


import copy
from typing import List, Dict


# def align_cx_matrix(cx_matrix: List[List[List[Instruction]]]) -> List[List[List[Instruction]]]:
#     """
#     Shift each CX‐brick (two rows with same cx#) rightwards so that:
#       1. top_row_index % 2 == column_index % 2 (parity alignment)
#       2. if originally separated by at least one empty column, preserve that spacing
#
#     Returns a new aligned cx_matrix without modifying the original.
#     """
#     # Deep copy to preserve original
#     original = copy.deepcopy(cx_matrix)
#     n_rows = len(original)
#     if n_rows == 0:
#         return original
#
#     # Determine maximum columns in the original matrix
#     n_cols = max(len(row) for row in original)
#
#     # Collect information about each CX‐brick (group)
#     group_info: Dict[str, Dict] = {}
#     for r in range(n_rows):
#         for c in range(n_cols):
#             cell = original[r][c] if c < len(original[r]) else []
#             for instr in cell:
#                 if instr.name.startswith("cx"):
#                     gid = instr.name[2:-1]
#                     info = group_info.setdefault(gid, {"rows": set(), "orig_col": c})
#                     info["rows"].add(r)
#                     info["orig_col"] = min(info["orig_col"], c)
#
#     # Build a list of bricks sorted by their original column
#     bricks = []
#     for gid, info in group_info.items():
#         top_row = min(info["rows"])
#         bricks.append({
#             "gid": gid,
#             "rows": sorted(info["rows"]),
#             "orig_col": info["orig_col"],
#             "top": top_row
#         })
#     bricks.sort(key=lambda b: b["orig_col"])
#
#     # Compute original spacings between consecutive bricks
#     for i in range(len(bricks)):
#         if i == 0:
#             bricks[i]["orig_spacing"] = 0
#         else:
#             prev_col = bricks[i-1]["orig_col"]
#             curr_col = bricks[i]["orig_col"]
#             bricks[i]["orig_spacing"] = curr_col - prev_col
#
#     # Prepare an empty aligned matrix
#     aligned: List[List[List[Instruction]]] = [[] for _ in range(n_rows)]
#
#     last_placed_col = -999
#     for i, brick in enumerate(bricks):
#         desired_col = brick["orig_col"]
#         # Parity constraint
#         if brick["top"] % 2 != desired_col % 2:
#             desired_col += 1
#         # Spacing constraint
#         spacing = brick.get("orig_spacing", 0)
#         if spacing > 1 and last_placed_col >= 0:
#             spacing_req = spacing
#         else:
#             spacing_req = 0
#         if spacing_req > 0:
#             place_col = max(desired_col, last_placed_col + spacing_req)
#         else:
#             place_col = desired_col
#
#         # Extend rows
#         for row in aligned:
#             while len(row) <= place_col:
#                 row.append([])
#
#         # Move CX instructions
#         for r in brick["rows"]:
#             instrs = [instr for instr in original[r][brick["orig_col"]]
#                       if instr.name.startswith(f"cx{brick['gid']}")]
#             aligned[r][place_col] = instrs
#
#         last_placed_col = place_col
#
#     return aligned

from typing import List, Dict
import copy

def align_cx_matrix(cx_matrix: List[List[List[Instruction]]]) -> List[List[List[Instruction]]]:
    """
    Shift each CX “brick” so that
      1. top_row % 2 == column % 2  (parity)
      2. for any prior brick sharing a qubit row, there's at least one empty column
         (or preserve original gap, if it was larger).

    Returns a new matrix; does not mutate the input.
    """
    original = copy.deepcopy(cx_matrix)
    n_rows = len(original)
    n_cols = max((len(r) for r in original), default=0)

    # Gather each brick’s rows and original column
    group_info: Dict[str, Dict] = {}
    for r in range(n_rows):
        for c in range(n_cols):
            cell = original[r][c] if c < len(original[r]) else []
            for instr in cell:
                if instr.name.startswith("cx"):
                    gid = instr.name[2:-1]
                    info = group_info.setdefault(gid, {"rows": set(), "orig_col": c})
                    info["rows"].add(r)
                    info["orig_col"] = min(info["orig_col"], c)

    # Build and sort bricks by orig_col
    bricks = []
    for gid, info in group_info.items():
        bricks.append({
            "gid": gid,
            "rows": sorted(info["rows"]),
            "orig_col": info["orig_col"],
            "top": min(info["rows"])
        })
    bricks.sort(key=lambda b: b["orig_col"])

    # Prepare the output grid
    aligned: List[List[List[Instruction]]] = [[] for _ in range(n_rows)]
    placed_cols: Dict[str, int] = {}

    for brick in bricks:
        gid      = brick["gid"]
        rows     = brick["rows"]
        ocol     = brick["orig_col"]
        top_row  = brick["top"]

        # 1) Parity‐aligned base column
        if (ocol % 2) == (top_row % 2):
            base_col = ocol
        else:
            base_col = ocol + 1

        # 2) Compute the floor from spacing constraints against every overlapping brick
        floor = 0
        for prev in bricks:
            pg = prev["gid"]
            if pg not in placed_cols:
                continue
            # only consider if they share at least one qubit
            if set(prev["rows"]) & set(rows):
                orig_gap = ocol - prev["orig_col"]
                # at least one empty column, or preserve if originally larger
                needed = placed_cols[pg] + max(1, orig_gap)
                floor = max(floor, needed)

        # 3) Final placement is the minimal ≥ both base_col and floor that satisfies parity
        place = max(base_col, floor)
        if place % 2 != top_row % 2:
            place += 1

        # grow output rows as needed
        for r in aligned:
            while len(r) <= place:
                r.append([])

        # copy exactly those CX‐instructions from the original column
        for r in rows:
            instrs = [
                instr for instr in original[r][ocol]
                if instr.name.startswith(f"cx{gid}")
            ]
            aligned[r][place] = instrs

        placed_cols[gid] = place

    return aligned


from typing import List, Optional

from typing import List
import copy

from typing import List
import copy


from typing import List

from typing import List

def insert_rotations_adjecant_to_cx(
    aligned_cx: List[List[List[Instruction]]],
    original:   List[List[List[Instruction]]]
) -> List[List[List[Instruction]]]:
    """
    aligned_cx: CX-only matrix after align_cx_matrix
    original:   full matrix (CX + rotations)
    Returns new matrix where each rotation-list from original[i][j]:
      - if original[i][j-1] has a CX → insert after that CX
      - elif original[i][j+1] has a CX → insert in the group *before* that CX
      - else → insert into column 0
    """
    # deep-copy so we can append
    new_mat = [
        [list(cell) for cell in row]
        for row in aligned_cx
    ]
    n_rows = len(new_mat)
    if n_rows == 0:
        return new_mat

    # normalize width
    width = max(len(r) for r in new_mat)
    for r in new_mat:
        if len(r) < width:
            r.extend([[]] * (width - len(r)))

    for i in range(n_rows):
        row_orig = original[i]
        for j, cell in enumerate(row_orig):
            rots = [ins for ins in cell if not ins.name.startswith('cx')]
            if not rots:
                continue

            dest = 0
            group_instr = None

            # 1) LEFT neighbor → after-CX
            if j > 0:
                left_cx = [ins for ins in row_orig[j-1] if ins.name.startswith('cx')]
                if left_cx:
                    group_instr = left_cx[0]
                    # find its column in aligned_cx
                    for c, aligned_cell in enumerate(new_mat[i]):
                        if any(ins.name == group_instr.name for ins in aligned_cell):
                            dest = c + 1
                            break

            # 2) RIGHT neighbor → before-CX (now one left of CX)
            if group_instr is None and j+1 < len(row_orig):
                right_cx = [ins for ins in row_orig[j+1] if ins.name.startswith('cx')]
                if right_cx:
                    group_instr = right_cx[0]
                    for c, aligned_cell in enumerate(new_mat[i]):
                        if any(ins.name == group_instr.name for ins in aligned_cell):
                            dest = max(c - 1, 0)
                            break

            # 3) ensure room
            if dest >= len(new_mat[i]):
                extra = dest - len(new_mat[i]) + 1
                for r in new_mat:
                    r.extend([[]] * extra)

            # append in order
            new_mat[i][dest].extend(rots)

    return new_mat



from typing import List, Optional

def insert_rotations(
    cx_matrix: List[List[List[Instruction]]],
    orig_matrix: List[List[List[Instruction]]]
) -> List[List[List[Instruction]]]:
    """
    Given:
      - cx_matrix   : the output of align_cx_matrix (only cx* in each cell)
      - orig_matrix : the original grid (cx* + rotations in each cell)
    Returns a new grid where every rotation list from orig_matrix[i][j]
    has been re-inserted into row i either just after its “prev” CX or
    just before its “next” CX, in the aligned cx_matrix.
    """
    n_rows = len(cx_matrix)
    if n_rows == 0:
        return []

    # ensure all rows of cx_matrix are the same width
    aligned_ncols = max(len(r) for r in cx_matrix)
    for row in cx_matrix:
        row.extend([[]] * (aligned_ncols - len(row)))

    # deep-copy the cx_matrix so we can append rotations
    new_mat: List[List[List[Instruction]]] = [
        [list(cell) for cell in row]
        for row in cx_matrix
    ]

    for i in range(n_rows):
        # collect the original CX positions & names in this row
        orig_row = orig_matrix[i]
        cx_positions: List[tuple[int,str]] = []
        for j, cell in enumerate(orig_row):
            for instr in cell:
                if instr.name.startswith('cx'):
                    cx_positions.append((j, instr.name))

        # for each rotation-only cell in the original:
        for j, cell in enumerate(orig_row):
            rots = [ins for ins in cell if not ins.name.startswith('cx')]
            if not rots:
                continue

            # find prev CX (max pos < j) and next CX (min pos > j)
            prev: Optional[tuple[int,str]] = None
            next_: Optional[tuple[int,str]] = None
            for pos, name in cx_positions:
                if pos < j and (prev is None or pos > prev[0]):
                    prev = (pos, name)
                if pos > j and (next_ is None or pos < next_[0]):
                    next_ = (pos, name)

            # decide insertion column in the aligned matrix
            if prev:
                # insert just after the aligned prev CX
                _, prev_name = prev
                dest = next(
                    idx for idx, c in enumerate(new_mat[i])
                    if any(ins.name == prev_name for ins in c)
                ) + 1
            elif next_:
                # no prev, but there's a next → insert just before it
                _, next_name = next_
                dest = next(
                    idx for idx, c in enumerate(new_mat[i])
                    if any(ins.name == next_name for ins in c)
                )
            else:
                # no CX in this row at all → stick at column 0
                dest = 0

            # grow matrix if we fell off the right edge
            if dest >= len(new_mat[i]):
                extra = dest - len(new_mat[i]) + 1
                for row in new_mat:
                    row.extend([[]] * extra)

            # if that slot is already a CX (collision), try the other side
            cell_at_dest = new_mat[i][dest]
            if any(ins.name.startswith('cx') for ins in cell_at_dest):
                if prev and next_:
                    # try before the next CX
                    _, next_name = next_
                    dest = next(
                        idx for idx, c in enumerate(new_mat[i])
                        if any(ins.name == next_name for ins in c)
                    )
                else:
                    # lone prev or lone next: flip direction
                    if prev:
                        dest -= 1
                    else:
                        dest += 1
                if dest < 0:
                    dest = 0
                if dest >= len(new_mat[i]):
                    extra = dest - len(new_mat[i]) + 1
                    for row in new_mat:
                        row.extend([[]] * extra)

            # finally, append the rotations in order
            new_mat[i][dest].extend(rots)

    return new_mat



from typing import List

import logging
logger = logging.getLogger(__name__)

def align_bricks_insert_blank(matrix):
    """
    Align CX and rotation operations into parity-based bricks.

    Args:
        matrix (List[List[Instruction]]): A list of qubit rows, each containing
            time-ordered columns of gate instructions. Instruction names include
            'rz', 'rx', 'cx{n}c' (control), or 'cx{n}t' (target). Suffix and role
            are parsed directly from instr.name.

    Returns:
        List[List[List[Instruction]]]: Reorganized matrix with shape [n_qubits][n_bricks].
    """

    # print("Aligning bricks...", end=" ")

    n_q = len(matrix)
    if not matrix:
        return []

    # Transpose to iterate columns
    cols = list(zip(*matrix))
    bricks = []
    brick_idx = 0

    for col in cols:
        # Collect single-qubit rotations
        rotations = [[instr for instr in row if instr.name in ('rz', 'rx')]
                     for row in col]

        # Map CX suffix to top-qubit index regardless of control/target role
        suffix_top = {}
        for i, row in enumerate(col):
            for instr in row:
                if instr.name.startswith('cx'):
                    suffix = instr.name[2:-1]
                    suffix_top[suffix] = min(suffix_top.get(suffix, i), i)

        # If no CX, schedule pure-rotation brick
        if not suffix_top:
            bricks.append(rotations)
            brick_idx += 1
            continue

        # Schedule CX bricks by parity of top-qubit row
        unscheduled = set(suffix_top)
        while unscheduled:
            parity = brick_idx % 2
            # select suffixes whose top row matches current parity
            to_place = {s for s, top in suffix_top.items() if top % 2 == parity and s in unscheduled}

            if not to_place:
                # insert empty brick to flip parity
                bricks.append([[] for _ in range(n_q)])
                brick_idx += 1
                continue

            # build a brick: combine rotations + all cx instructions with these suffixes
            layer = []
            for row in col:
                ops = [instr for instr in row if instr.name in ('rz', 'rx')]
                for suffix in to_place:
                    ops.extend(instr for instr in row if instr.name.startswith(f'cx{suffix}'))
                layer.append(ops)

            bricks.append(layer)
            brick_idx += 1
            unscheduled -= to_place

    # print("Done")
    # transpose back to [qubit][brick]
    return [[brick[i] for brick in bricks] for i in range(n_q)]


def align_bricks_two_phase(matrix):
    n_q = len(matrix)
    if n_q == 0:
        return []

    # Transpose to get columns
    cols = list(zip(*matrix))

    # Phase 1: schedule CX bricks
    # --------------------------------
    # bricks_cx: list of layers; each layer is a list-of-lists for each qubit
    bricks_cx = []
    # for each suffix, remember its top-qubit index
    for col in cols:
        # collect all CX suffixes in this column
        suffix_top = {}
        for i, row in enumerate(col):
            for instr in row:
                if instr.name.startswith('cx'):
                    s = instr.name[2:-1]
                    suffix_top[s] = min(suffix_top.get(s, i), i)

        # assign each suffix to a brick of matching parity
        # we track: next brick index for parity 0 and parity 1
        next_brick = {0: 0, 1: 0}
        for suffix, top in sorted(suffix_top.items(), key=lambda x: x[1]):
            p = top % 2
            bidx = next_brick[p]
            # if we need a new brick, append an empty layer
            while bidx >= len(bricks_cx):
                bricks_cx.append([[] for _ in range(n_q)])
            # place BOTH control and target into that layer
            for i, row in enumerate(col):
                for instr in row:
                    if instr.name.startswith(f'cx{suffix}'):
                        bricks_cx[bidx][i].append(instr)
            # increment for next same‐parity suffix
            next_brick[p] += 1

    # Phase 2: pack rotations
    # --------------------------------
    # we'll build bricks_full starting from a deep copy of bricks_cx
    from copy import deepcopy
    bricks_full = deepcopy(bricks_cx)

    for col in cols:
        # extract single‐qubit rotations in this column
        rotations = [(i, instr)
                     for i, row in enumerate(col)
                     for instr in row
                     if instr.name in ('rz', 'rx')]
        for q_idx, rot in rotations:
            # try to insert into earliest brick with no CX at q_idx
            placed = False
            for layer in bricks_full:
                # check if this qubit is free of CX here
                if not any(op.name.startswith('cx') for op in layer[q_idx]):
                    layer[q_idx].append(rot)
                    placed = True
                    break
            if not placed:
                # need a brand‐new brick
                new_layer = [[] for _ in range(n_q)]
                new_layer[q_idx].append(rot)
                bricks_full.append(new_layer)

    # transpose back to [qubit][brick]
    return [[layer[q] for layer in bricks_full] for q in range(n_q)]




from typing import List, Set, Dict, Optional


def align_bricks(cx_mat, orig):
    cx_mat_aligned = align_cx_matrix(cx_mat)
    qc_mat_aligned = insert_rotations_adjecant_to_cx(cx_mat_aligned, orig)

    return qc_mat_aligned








def align_bricks_with_insertion(matrix):
    """
    Align CX and rotation operations into parity-based bricks,
    pushing CX operations forward rather than inserting empty bricks.

    If parity doesn't match at a given column, remaining CXs are moved to the next
    column; they overwrite identity slots or push further if conflicts occur.

    Args:
        matrix (List[List[Instruction]]): Each sublist is a qubit's time-ordered
            list of gate instructions per column; Instruction.name in
            {'rz', 'rx', 'cx{n}c', 'cx{n}t'}.

    Returns:
        List[List[List[Instruction]]]: Scheduled bricks [n_qubits][n_bricks].
    """
    n_q = len(matrix)
    if n_q == 0:
        return []

    # Transpose and copy matrix into mutable column structure
    cols = [[list(instrs) for instrs in col] for col in zip(*matrix)]
    bricks = []
    brick_idx = 0
    col_idx = 0

    while col_idx < len(cols):
        col = cols[col_idx]
        # Extract single-qubit rotations for this column
        rotations = [[instr for instr in row if instr.name in ('rz', 'rx')] for row in col]

        # Map CX suffix to its top-qubit index
        suffix_top = {}
        for i, row in enumerate(col):
            for instr in row:
                if instr.name.startswith('cx'):
                    suffix = instr.name[2:-1]
                    suffix_top[suffix] = min(suffix_top.get(suffix, i), i)

        # If no CX in this column, schedule pure-rotation brick
        if not suffix_top:
            bricks.append(rotations)
            brick_idx += 1
            col_idx += 1
            continue

        # Try to schedule all CX suffix-groups
        unscheduled = set(suffix_top)
        while unscheduled:
            parity = brick_idx % 2
            # Find suffixes matching current parity
            to_place = {s for s, top in suffix_top.items() if (top % 2) == parity and s in unscheduled}

            if not to_place:
                # Push remaining CXs to next column instead of empty brick
                next_idx = col_idx + 1
                # Extend columns if needed
                if next_idx >= len(cols):
                    cols.append([[] for _ in range(n_q)])
                # Move unscheduled CX instructions
                for suffix in unscheduled:
                    for qi, row in enumerate(col):
                        for instr in row:
                            if instr.name.startswith(f'cx{suffix}'):
                                cols[next_idx][qi].append(instr)
                # Stop scheduling this column
                break

            # Build and append this brick
            layer = []
            for row in col:
                ops = [instr for instr in row if instr.name in ('rz', 'rx')]
                for suffix in to_place:
                    ops.extend(instr for instr in row if instr.name.startswith(f'cx{suffix}'))
                layer.append(ops)
            bricks.append(layer)
            brick_idx += 1
            unscheduled -= to_place

        # Move to next column
        col_idx += 1

    # Transpose back to [qubit][brick]
    return [[brick[i] for brick in bricks] for i in range(n_q)]


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
