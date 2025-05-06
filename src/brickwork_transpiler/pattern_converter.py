import numpy as np
from graphix import generate_from_graph

def parse_euler_angles(cell):

    rotations = [0.0, 0.0, 0.0, 0.0]
    index = 0
    for instr in cell:
        if instr.name == 'rz':
            if index != 0 and index != 2:
                raise AssertionError(f"Index wrong: {index}, instuction: {instr}")
            rotations[index] = -float(instr.params[0]) / np.pi # - for convetion graphix -- add source
        elif instr.name == 'rx':
            rotations[1] = -float(instr.params[0]) / np.pi #rx is always the second entry
        index = index + 1

    return rotations

def lay_brick(angles, brick_type, r, c, cell, node_colours):

    if brick_type == "id":
        angles.update({
            (r, (c * 4) + 0): 0.0,
            (r, (c * 4) + 1): 0.0,
            (r, (c * 4) + 2): 0.0,
            (r, (c * 4) + 3): 0.0,
        })
        node_colours.update({
            (r, (c * 4) + 0): 'lightblue',
            (r, (c * 4) + 1): 'lightblue',
            (r, (c * 4) + 2): 'lightblue',
            (r, (c * 4) + 3): 'lightblue',
        })
    elif brick_type == "euler_rot":
        local_angles = parse_euler_angles(cell)
        angles.update({
            (r, (c * 4) + 0): local_angles[0],
            (r, (c * 4) + 1): local_angles[1],
            (r, (c * 4) + 2): local_angles[2],
            (r, (c * 4) + 3): local_angles[3],
        })
        node_colours.update({
            (r, (c * 4) + 0): 'lightgreen',
            (r, (c * 4) + 1): 'lightgreen',
            (r, (c * 4) + 2): 'lightgreen',
            (r, (c * 4) + 3): 'lightgreen',
        })
    # target bottom only for now
    elif brick_type == "CX":
        angles.update({
            (r, (c * 4) + 0): 0.0,
            (r, (c * 4) + 1): 0.0,
            (r, (c * 4) + 2): -1/4,
            (r, (c * 4) + 3): 0.0,

            (r+1, (c * 4) + 0): 0.0,
            (r+1, (c * 4) + 1): -1/4,
            (r+1, (c * 4) + 2): 0.0,
            (r+1, (c * 4) + 3): 1/4,
        })
        node_colours.update({
            (r, (c * 4) + 0): 'lightcoral',
            (r, (c * 4) + 1): 'lightcoral',
            (r, (c * 4) + 2): 'lightcoral',
            (r, (c * 4) + 3): 'lightcoral',

            (r + 1, (c * 4) + 0): 'lightcoral',
            (r + 1, (c * 4) + 1): 'lightcoral',
            (r + 1, (c * 4) + 2): 'lightcoral',
            (r + 1, (c * 4) + 3): 'lightcoral',
        })

    # TODO: Discern between target top and bottom - also add colours (different shade??)
    # CX target top
    elif brick_type == "CXtt":
        angles.update({
            (r, (c * 4) + 0): 0.0,
            (r, (c * 4) + 1): -1/4,
            (r, (c * 4) + 2): 0.0,
            (r, (c * 4) + 3): 1/4,

            (r+1, (c * 4) + 0): 0.0,
            (r+1, (c * 4) + 1): 0/0,
            (r+1, (c * 4) + 2): -1/4,
            (r+1, (c * 4) + 3): 0.0,
        })
    # CX target bottom
    elif brick_type == "CXtb":
        angles.update({
            (r, (c * 4) + 0): 0.0,
            (r, (c * 4) + 1): 0.0,
            (r, (c * 4) + 2): -1/4,
            (r, (c * 4) + 3): 0.0,

            (r+1, (c * 4) + 0): 0.0,
            (r+1, (c * 4) + 1): -1/4,
            (r+1, (c * 4) + 2): 0.0,
            (r+1, (c * 4) + 3): 1/4,
        })
    else:
        raise AssertionError(f"Unknown brick type: {brick_type} instuction type: {cell}")


def to_pattern(insturction_matrix, structure_graph):
    num_qubits = len(insturction_matrix)
    num_cols = len(insturction_matrix[0])

    angles = {}
    node_colours = {} # keep track of bricks to mae the graph readable
    cx_placed = False


    for c in range(num_cols):
        for r in range(num_qubits):
            # skip the next row brick since cx already made it
            if cx_placed:
                cx_placed = False
                continue

            cell = insturction_matrix[r][c]

            if cell == []:
                # print("lay identity brick")
                lay_brick(angles=angles, brick_type="id", r=r, c=c, cell=None, node_colours=node_colours)
            else:
                for instr in cell:
                    # print("instr:", instr)
                    if instr.name.startswith('cx'):
                        # print("lay cx brick")
                        lay_brick(angles=angles, brick_type="CX", r=r, c=c, cell=cell, node_colours=node_colours)
                        cx_placed = True
                    elif instr.name == 'rz' or instr.name == 'rx':
                        # print("lay euler rotation brick")
                        lay_brick(angles=angles, brick_type="euler_rot", r=r, c=c, cell=cell, node_colours=node_colours)
                    else:
                        raise AssertionError(f"Unrecognized instruction: {instr.name}")

    # print("Angles: ", angles)

    inputs = []  # Initialise to the |+> state as with ubqc
    outputs = [(i, (num_cols * 4)) for i in range(num_qubits)] # 4 times the amount of bricks

    brickwork_pattern = generate_from_graph(structure_graph, angles, inputs, outputs)

    return brickwork_pattern, node_colours