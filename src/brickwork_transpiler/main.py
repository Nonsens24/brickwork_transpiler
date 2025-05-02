from graphix.fundamentals import Plane
from graphix.opengraph import Measurement
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

import bricks
import utils
import visualiser

def main():

    inside_graph = visualiser.create_tuple_node_graph(1, 3)
    measurements = {(0, 0): Measurement(0, Plane.XY),
                    (0, 1): Measurement(0, Plane.XY)}  # angle(pi)=0 ⇒ X-basis INITIALISE TO 0
    inputs = []
    outputs = [(0, 2)]

    # gx_angle_extractor.print_pattern()
    # gx_angle_extractor.print_standard_pattern()

    print("Laying an arbitrary brick:")
    pattern = bricks.arbitrary_brick(1 / 4, 1 / 4, 1 / 4)
    # pattern.print_pattern()


    print("Laying an arbitrary brick2:")
    pattern = bricks.arbitrary_brick(1 / 4, -1 / 4, 1 / 2)

    outstate = pattern.simulate_pattern(backend='statevector').flatten()
    print("  raw MBQC output:", outstate)

    # print("laying a brick")
    # pattern = bricks.make_H_brick()
    #
    # print("simulating H graph:")
    # outstate = pattern.simulate_pattern(backend='statevector').flatten()
    # print("  raw MBQC output:", outstate)
    #
    #
    qc = QuantumCircuit(1)
    qc.h(0)
    qc.s(0)
    # qc.rz(1/2 * np.pi, 0)
    sv = Statevector.from_instruction(qc).data
    print("  ideal S|+>       :", sv)

    qc2 = QuantumCircuit(1)
    qc2.h(0)
    qc2.rz(0, 0)
    qc.rx(0, 0)
    qc.rz(0, 0)
    sv2 = Statevector.from_instruction(qc2).data
    print("  ideal T|+>       :", sv2)

    utils.assert_equal_up_to_global_phase(sv, sv2)
    #
    # overlap = np.abs(np.vdot(sv, outstate))
    # print("overlap with ideal T:", overlap)
    #
    # pattern2 = bricks.make_T_brick()
    # outstate = pattern2.simulate_pattern(backend='statevector').flatten()
    # print(outstate)
    #
    # qc = QuantumCircuit(1)
    # qc.h(0)
    # qc.t(0)
    # sv = Statevector.from_instruction(qc).data
    # print("  ideal T|+>       :", sv)
    #
    # overlap = np.abs(np.vdot(sv, outstate))
    # print("overlap with ideal T:", overlap)

    # pattern = bricks.make_T_brick()

    # **this is essential**: apply the feed‑forward Pauli corrections
    #
    # print("simulating T graph:")
    # outstate = pattern.simulate_pattern(backend='statevector').flatten()
    # print("  raw MBQC output:", outstate)


    # qc = QuantumCircuit(1)
    # qc.h(0)
    # qc.t(0)
    # sv = Statevector.from_instruction(qc).data
    # print("  ideal T|+>       :", sv)
    #
    # overlap = np.abs(np.vdot(sv, outstate))
    # print("overlap with ideal T:", overlap)



if __name__ == "__main__":
    main()
