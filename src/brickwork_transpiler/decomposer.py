from matplotlib import pyplot as plt
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Instruction
from qiskit.transpiler import CouplingMap
from qiskit.transpiler.passes.synthesis import HLSConfig
from qiskit.visualization import dag_drawer


from src.brickwork_transpiler import visualiser
#
#
# def decompose_qc_to_bricks_qiskit(qc: QuantumCircuit, opt=3, draw=False,
#                                   routing_method: str = 'basic',
#                                   layout_method: str ='trivial',
#                                   file_writer=None):
#     #
#     # # print("Decomposing Quantum circuit to generator set...", end=" ")
#     #
#     # basis = ['rz', 'rx', 'cx', 'id']  # include 'id' for explicit barriers/timing
#     # qc_basis = transpile(qc, basis_gates=basis, optimization_level=opt)
#     #
#     # if draw:
#     #     print(qc_basis.draw())
#     #
#     # # print("Done")
#
#     # 1. Define your basis
#     basis = ['rz', 'rx', 'cx', 'id']
#
#     # print(qc)
#
#     # 2. Define the coupling map for your device/simulator:
#     edges = []
#     coupling = None
#     if qc.num_qubits != 1:
#         for i in range(qc.num_qubits - 1):
#             edges.append([i, i + 1])  # allow i → i+1
#             edges.append([i + 1, i])  # allow i+1 → i
#         coupling = CouplingMap(edges)
#     # if qc.num_qubits > 1:
#     #     coupling = CouplingMap([[i, i+1] for i in range(qc.num_qubits -1)]) # -1 so the amount of qubits remains right
#     # else:
#     #     coupling = CouplingMap([[0, 0]])
#
#     # pm = PassManager()
#     # # 1) Unroll everything down to rz, rx, cx
#     # pm.append(UnrollCustomDefinitions(basis))
#     # # 2) Collect every 1-qubit run and decompose to RZ–RX–RZ
#     # pm.append(OneQubitEulerDecomposition(basis=basis))
#
# # | Layout Method    | Strategy                                               |
# # | ---------------- | ------------------------------------------------------ |
# # | `trivial`        | 1-to-1 mapping                                         |
# # | `dense`          | Densest‐subgraph heuristic                             |
# # | `noise_adaptive` | Minimize error rates (readout & 2-qubit)               |
# # | `sabre`          | Sabre seed + forwards/backwards swap refinement        |
# # | `default`        | VF2 perfect + Sabre fallback (or `trivial` at level 0) |
#
#
#     # 3. Transpile with routing
#     # Lookahead routing: evaluates cost-to-go windows to insert SWAPs globally
#     # Sabre: runs routing forward and backward to refine qubit movements in both directions
#     # Stochastic routing: randomizes swap choices to explore bidirectional traffic patterns
#     qc_mapped = transpile(
#         qc,
#         basis_gates=basis,
#         coupling_map=coupling,
#         layout_method=layout_method,  # keep initial virtual→physical mapping
#         routing_method= routing_method, #'stochastic', #'sabre', #'lookahead', #'basic',  # use the simple SWAP-insertion pass
#         optimization_level=opt
#     )
#
#     # qc_mapped.draw(output='mpl',
#     #                    fold=40,
#     #                    )
#     # # plt.savefig(f"images/Circuits/circuit_dag_ex_decomposed.png", dpi=300, bbox_inches="tight")
#     # plt.show()
#
#     if draw:
#         print(qc_mapped.draw())
#
#
#
#     if file_writer:
#         print("decomp written")
#         file_writer.set("decomposed_depth", qc_mapped.depth())
#         file_writer.set("num_gates_transpiled", sum(qc_mapped.count_ops().values()))
#         file_writer.set("num_gates_original", sum(qc.count_ops().values()))
#
#
#     return qc_mapped

from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit.transpiler import CouplingMap

# ── Linear-CX MCX decomposition for Qiskit-Terra 0.45.0 ──────────────
from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit.transpiler import CouplingMap                      # still here in 0.45
from qiskit.transpiler.passes.synthesis import HLSConfig       # path in 0.45.x
# ^— MCXRecursion is the one-ancilla Barenco–Iten algorithm (≤ 16 k−24 CX)
#     The helper lives in mcxtools sub-module in 0.45.0.
# ────────────────────────────────────────────────────────────────
#  decompose_qc_to_bricks_qiskit  –  linear-CX MCX for any Terra
#  * works on your current Qiskit-Terra 0.45.0
#  * no HLS plugin names → no 'method not found' errors
#  * adds exactly ONE clean ancilla and uses MCXRecursive
# ────────────────────────────────────────────────────────────────
# -------------------------------------------------------------
#  Linear-CX MCX transpilation helper (Qiskit-Terra 0.45.x)
# -------------------------------------------------------------
from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit.circuit.library import MCXRecursive             # built-in linear gate
from qiskit.circuit.library.standard_gates import MCXGate
from qiskit.transpiler import CouplingMap

from qiskit import transpile
from qiskit.transpiler import CouplingMap
# from pytket.extensions.qiskit import qiskit_to_tk, tk_to_qiskit
# from pytket.passes import DecomposeBoxes, ZXOptimizer

# #
# def _linearise_mcx(qc: QuantumCircuit) -> QuantumCircuit:
#     """Rewrite every >2-controlled X using MCXRecursive (linear CX)."""
#     out = QuantumCircuit(*qc.qregs, *qc.cregs, name=qc.name)
#
#     # ensure a clean ancilla register exists and remember its first qubit
#     if not any(r.name == "anc" for r in qc.qregs):
#         anc_reg = QuantumRegister(1, "anc")
#         out.add_register(anc_reg)
#     else:
#         anc_reg = next(r for r in qc.qregs if r.name == "anc")
#
#     anc_qubit = anc_reg[0]
#
#     for inst, qargs, cargs in qc.data:
#         if isinstance(inst, MCXGate) and inst.num_ctrl_qubits > 2:
#             k = inst.num_ctrl_qubits
#             mcx_lin = MCXRecursive(k)
#
#             if k >= 5:            # docs: recursion needs 1 ancilla iff k > 4
#                 out.append(mcx_lin, qargs + [anc_qubit], cargs)  # k+2 qubits
#             else:
#                 out.append(mcx_lin, qargs, cargs)                # k+1 qubits
#         else:
#             out.append(inst, qargs, cargs)
#     return out

from qiskit.circuit.library import MCU1Gate, MCPhaseGate


# from qiskit.circuit import QuantumCircuit, QuantumRegister
# from qiskit.circuit.library import MCXGate
#
# def linearise_mcx(qc: QuantumCircuit) -> QuantumCircuit:
#     out = QuantumCircuit(*qc.qregs, *qc.cregs, name=qc.name)
#     if not any(r.name == "anc" for r in qc.qregs):
#         out.add_register(QuantumRegister(1, "anc"))
#     anc = next(r for r in out.qregs if r.name == "anc")[0]
#
#     for inst, qargs, cargs in qc.data:
#         if isinstance(inst, MCXGate) and inst.num_ctrl_qubits > 2:
#             k = inst.num_ctrl_qubits
#             ctrls, tgt = qargs[:-1], qargs[-1]
#             ancilla_qubits = [anc] if k > 4 else None
#             out.mcx(ctrls, tgt, ancilla_qubits=ancilla_qubits, mode="recursion")
#         else:
#             out.append(inst, qargs, cargs)
#     return out


# def linearise_mcu1_explicit(qc: QuantumCircuit) -> QuantumCircuit:
#     out = QuantumCircuit(*qc.qregs, *qc.cregs, name=qc.name)
#     if not any(r.name == "anc" for r in qc.qregs):
#         out.add_register(QuantumRegister(1, "anc"))
#     anc = next(r for r in out.qregs if r.name == "anc")[0]
#
#     for inst, qargs, cargs in qc.data:
#         is_mcu1 = isinstance(inst, MCU1Gate) and inst.num_ctrl_qubits > 0
#         is_mcphase = isinstance(inst, MCPhaseGate) and inst.num_ctrl_qubits > 0
#         if is_mcu1 or is_mcphase:
#             theta = float(inst.params[0])
#             controls, target = qargs[:-1], qargs[-1]
#             all_ctrls = list(controls) + [target]
#             # compute conjunction into ancilla
#             if len(all_ctrls) == 1:
#                 out.rz(theta, target)
#             else:
#                 # build AND chain: V-chain style
#                 out.ccx(all_ctrls[0], all_ctrls[1], anc)
#                 for q in all_ctrls[2:]:
#                     out.ccx(q, anc, anc)
#                 # apply phase
#                 out.rz(theta, anc)
#                 # uncompute
#                 for q in reversed(all_ctrls[2:]):
#                     out.ccx(q, anc, anc)
#                 out.ccx(all_ctrls[0], all_ctrls[1], anc)
#         else:
#             out.append(inst, qargs, cargs)
#     return out
#
#
# from qiskit.transpiler import PassManager, CouplingMap
# from qiskit.transpiler.passes import (
#     Unroll3qOrMore, BasisTranslator,
#     TrivialLayout, ApplyLayout,
#     SabreSwap, GateDirection,
#     CXCancellation, Optimize1qGatesDecomposition,
# )
# from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary
#
# def decompose_cp_to_rz_cx(qc: QuantumCircuit) -> QuantumCircuit:
#     out = QuantumCircuit(*qc.qregs, *qc.cregs, name=qc.name)
#     for inst, qargs, cargs in qc.data:
#         if inst.name == "cp":
#             lam = float(inst.params[0])
#             ctrl, tgt = qargs[0], qargs[1]
#             out.rz(lam / 2, tgt)
#             out.cx(ctrl, tgt)
#             out.rz(-lam / 2, tgt)
#             out.cx(ctrl, tgt)
#         else:
#             out.append(inst, qargs, cargs)
#     return out
#
#
# def preserving_passmanager(n_qubits: int, seed: int = 11) -> PassManager:
#     cmap = CouplingMap.from_line(n_qubits)
#     return PassManager([
#         Unroll3qOrMore(),
#         BasisTranslator(SessionEquivalenceLibrary, ["rz", "rx", "cx", "id"]),
#         TrivialLayout(cmap), ApplyLayout(),
#         SabreSwap(cmap, heuristic="decay", seed=seed),
#         GateDirection(coupling_map=cmap),
#         CXCancellation(),
#         Optimize1qGatesDecomposition(["rz", "rx", "cx", "id"]),
#     ])
#
#
# def decompose_qc_to_bricks_qiskit(
#         qc: QuantumCircuit,
#         draw: bool = False,
#         file_writer=None,
#         seed: int = 11,
# ):
#     # 1. Linear MCX
#     qc_lin = _linearise_mcx(qc)
#     # 2. Linear MCU1 / phase
#     qc_lin = linearise_mcu1(qc_lin)
#
#
#     # Diagnostic before mapping
#     print("Logical ops:", qc_lin.decompose(reps=2).count_ops())
#
#     # 4. Map without undoing linearisation
#     pm = preserving_passmanager(qc_lin.num_qubits, seed=seed)
#     qc_mapped = pm.run(qc_lin)
#
#     if draw:
#         print(qc_mapped.draw())
#
#     if file_writer:
#         file_writer.set("decomposed_depth",     qc_mapped.depth())
#         file_writer.set("num_gates_transpiled", sum(qc_mapped.count_ops().values()))
#         file_writer.set("num_gates_original",   sum(qc.count_ops().values()))
#
#     print("Final ops:", qc_mapped.count_ops())
#     return qc_mapped
#
#
# def print_logical_stats(qc, label="logical"):
#     ops = qc.count_ops()
#     print(f"[{label}] depth={qc.depth():5d}  CX={ops.get('cx',0):6d}  RZ={ops.get('rz',0):6d}  RX={ops.get('rx',0):6d}  total1q={sum(v for k,v in ops.items() if k in ['rz','rx']):6d}")

from qiskit.circuit.library import MCXRecursive, MCPhaseGate
from math import pi
from qiskit import QuantumCircuit, AncillaRegister, transpile
from qiskit.circuit.library import (
    MCXRecursive,
    MCXGate,
    MCPhaseGate,
    PhaseGate,          # single-qubit phase
)
from qiskit.transpiler import CouplingMap
from qiskit.transpiler.passes.synthesis import HLSConfig


# -------------------------------------------------------------------
# 1) strip “Q” composite wrappers but keep MCX / MCU1 in place
# -------------------------------------------------------------------
def selective_decompose(qc, keep=('mcx', 'mcu1', 'mcphase')):
    out = QuantumCircuit(*qc.qregs, *qc.cregs, name=qc.name)
    for inst, q, c in qc.data:
        if inst.name.lower() in keep:
            out.append(inst, q, c)
        elif inst.definition:
            out.compose(
                selective_decompose(inst.definition, keep), qubits=q, inplace=True
            )
        else:
            out.append(inst, q, c)
    return out

from math import pi
from qiskit import QuantumCircuit, AncillaRegister
from qiskit.circuit.library import MCXRecursive, MCXGate, PhaseGate

from math import pi
from qiskit import QuantumCircuit, AncillaRegister
from qiskit.circuit.library import MCXRecursive, MCXGate, PhaseGate

def linearise_multi_ctrl(qc: QuantumCircuit) -> QuantumCircuit:
    """Linear-depth replacements for MCX (>2-ctrl) and MCU1/MCP (>1-ctrl)."""

    # ▸ make sure we have at least TWO ancillas available
    if not any(r.name == 'anc' for r in qc.qregs):
        qc.add_register(AncillaRegister(2, 'anc'))
    elif len(next(r for r in qc.qregs if r.name == 'anc')) < 2:
        qc.add_register(AncillaRegister(2 - len(next(r for r in qc.qregs if r.name == 'anc')), 'anc_ext'))

    anc_reg = next(r for r in qc.qregs if r.name.startswith('anc'))
    phase_anc, scratch_anc = anc_reg[:2]

    out = QuantumCircuit(*qc.qregs, *qc.cregs, name=qc.name)

    for inst, q, c in qc.data:

        # ---------- MCX (>2 controls) ----------
        if isinstance(inst, MCXGate) and inst.num_ctrl_qubits > 2:
            k = inst.num_ctrl_qubits
            if k >= 5:
                # controls + phase_anc (single ancilla) + target
                out.append(MCXRecursive(k), q + [phase_anc], c)
            else:          # k = 3 or 4 → no ancilla
                out.append(MCXRecursive(k), q, c)

        # ---------- MCU1 / MCPhase (>1 control) ----------
        elif inst.name.lower() in ('mcu1', 'mcphase') and len(q) > 2:
            theta = float(inst.params[0]) % (2 * pi)
            ctrls, target = q[:-1], q[-1]
            k = len(ctrls)

            # Step 1: compute AND of controls into phase_anc
            if k >= 5:
                out.append(MCXRecursive(k), ctrls + [scratch_anc, phase_anc])
            else:
                out.append(MCXRecursive(k), ctrls + [phase_anc])

            # Step 2: controlled-phase from phase_anc to target
            out.append(PhaseGate(theta).control(1), [phase_anc, target])

            # Step 3: un-compute phase_anc
            if k >= 5:
                out.append(MCXRecursive(k), ctrls + [scratch_anc, phase_anc])
            else:
                out.append(MCXRecursive(k), ctrls + [phase_anc])

        # ---------- everything else ----------
        else:
            out.append(inst, q, c)

    return out


# def linearise_multi_ctrl(qc: QuantumCircuit) -> QuantumCircuit:
#     """Replace MCX (>2-ctrl) and MCU1/MCPhase (>1-ctrl) with linear (1-anc) circuits."""
#     # ▸ ensure ≥2 ancillas exist in qc
#     if not any(r.name == 'anc' for r in qc.qregs):
#         qc.add_register(AncillaRegister(2, 'anc'))
#     anc_reg = next(r for r in qc.qregs if r.name.startswith('anc'))
#     phase_anc, scratch_anc = anc_reg[:2]
#
#     out = QuantumCircuit(*qc.qregs, *qc.cregs, name=qc.name)
#
#     for inst, q, c in qc.data:
#         # ---------- MCX (>2 controls) ----------
#         if isinstance(inst, MCXGate) and inst.num_ctrl_qubits > 2:
#             k = inst.num_ctrl_qubits
#             if k >= 5:  # needs ancillas
#                 out.append(MCXRecursive(k),
#                            q + [scratch_anc, phase_anc], c)
#             else:  # k = 3 or 4, no ancilla
#                 out.append(MCXRecursive(k), q, c)
#         # if isinstance(inst, MCXGate) and inst.num_ctrl_qubits > 2:
#         #     k = inst.num_ctrl_qubits
#         #     extra = [scratch_anc] if k >= 5 else []
#         #     out.append(MCXRecursive(k), q + extra + [phase_anc], c)
#
#         # ---------- MCU1  /  MCPhase (>1 control) ----------
#         elif inst.name.lower() in ('mcu1', 'mcphase') and len(q) > 2:
#             theta  = float(inst.params[0]) % (2 * pi)
#             ctrls, target = q[:-1], q[-1]
#             k = len(ctrls)
#
#             extra = [scratch_anc] if k >= 5 else []
#             out.append(MCXRecursive(k), ctrls + extra + [phase_anc])
#             out.append(PhaseGate(theta).control(1), [phase_anc, target])
#             out.append(MCXRecursive(k), ctrls + extra + [phase_anc])
#
#         # ---------- everything else ----------
#         else:
#             out.append(inst, q, c)
#
#     return out




# -------------------------------------------------------------------
# 3) wrapper you call from your code
# -------------------------------------------------------------------
def decompose_qc_to_bricks_qiskit(
        qc: QuantumCircuit,
        opt: int = 3,
        draw: bool = False,
        routing_method: str = "basic",
        layout_method: str = "trivial",
        file_writer=None,
):
    print("before:", qc.count_ops())  # {'mcphase': 1}

    qc_sel = selective_decompose(qc)
    print("Selectively decomposed: ", qc_sel.count_ops())
    qc_lin = linearise_multi_ctrl(qc_sel)
    print("after :", qc_lin.count_ops())  # ≈ 188 cx, 189 rz

    # only MCX needs a plug-in now (Qiskit 0.46.3 already has it)
    hls_cfg = HLSConfig()#mcx={'synthesis_method': 'recursion'})

    # simple line-coupling (swap for backend.target if you have one)
    basis = ["rz", "rx", "cx", "id"]
    coupling = (CouplingMap(
        [[i, i + 1] for i in range(qc_lin.num_qubits - 1)]
        + [[i + 1, i] for i in range(qc_lin.num_qubits - 1)]
    ) if qc_lin.num_qubits > 1 else None)

    print("Transpiling...")
    qc_final = transpile(
        qc_lin,
        basis_gates=basis,
        coupling_map=coupling,
        layout_method=layout_method,
        routing_method=routing_method,
        hls_config=hls_cfg,
        optimization_level=opt,
    )

    if draw:
        print(qc_final.draw())

    print("post-HLS :", qc_final.count_ops())

    if file_writer:
        file_writer.set("decomposed_depth",     qc_final.depth())
        file_writer.set("num_gates_transpiled", sum(qc_final.count_ops().values()))
        file_writer.set("num_gates_original",   sum(qc_sel.count_ops().values()))


    return qc_final




# def _linearise_multi_control_gates(qc: QuantumCircuit) -> QuantumCircuit:
#     """Rewrite every multi-controlled X and MCU1 gates using linear decompositions."""
#     out = QuantumCircuit(*qc.qregs, *qc.cregs, name=qc.name)
#
#     # Ensure ancilla register exists
#     if not any(r.name == "anc" for r in qc.qregs):
#         anc_reg = QuantumRegister(1, "anc")
#         out.add_register(anc_reg)
#     else:
#         anc_reg = next(r for r in qc.qregs if r.name == "anc")
#     anc_qubit = anc_reg[0]
#
#     for inst, qargs, cargs in qc.data:
#         gate_name = inst.name.lower()
#
#         if gate_name == 'mcx' and inst.num_ctrl_qubits > 2:
#             k = inst.num_ctrl_qubits
#             mcx_lin = MCXRecursive(k)
#             if k >= 5:
#                 out.append(mcx_lin, qargs + [anc_qubit], cargs)
#             else:
#                 out.append(mcx_lin, qargs, cargs)
#
#         elif gate_name in ('mcu1', 'mcphase') and inst.num_ctrl_qubits >= 2:
#             linear_mcu1 = MCPhaseGate(inst.params[0], inst.num_ctrl_qubits)
#
#             # Exactly (num_ctrl_qubits + 1) qubits required
#             # Thus, ancilla is NOT always needed explicitly here (MCPhaseGate manages ancilla internally).
#             # Just provide original qargs, the transpiler will handle ancilla if configured.
#             out.append(linear_mcu1, qargs, cargs)
#
#         else:
#             out.append(inst, qargs, cargs)
#
#     return out
#
#
#
# from qiskit.circuit.library import MCPhaseGate, MCXGate
#
# def selective_decompose(qc, gate_names_to_preserve=('mcphase', 'mcu1', 'mcx')):
#     """
#     Decomposes composite gates selectively, preserving MCPhaseGate and MCXGate.
#     """
#     decomposed_qc = QuantumCircuit(*qc.qregs, *qc.cregs, name=qc.name)
#
#     for inst, qargs, cargs in qc.data:
#         gate_name = inst.name.lower()
#         if gate_name in gate_names_to_preserve:
#             # preserve these gates
#             decomposed_qc.append(inst, qargs, cargs)
#         elif inst.definition is not None:
#             # Decompose recursively if composite and not in preserve list
#             inner_decomposed = selective_decompose(inst.definition, gate_names_to_preserve)
#             decomposed_qc.compose(inner_decomposed, qubits=qargs, inplace=True)
#
#         else:
#             decomposed_qc.append(inst, qargs, cargs)
#
#     return decomposed_qc
#
#
#
# def decompose_qc_to_bricks_qiskit(
#         qc: QuantumCircuit,
#         opt: int = 3,
#         draw: bool = False,
#         routing_method: str = "basic",
#         layout_method: str = "trivial",
#         file_writer=None,
# ):
#
#     qc_sel = selective_decompose(qc)
#
#     from qiskit import QuantumRegister, AncillaRegister, QuantumCircuit
#
#     # Create your working registers explicitly
#     qreg_main = QuantumRegister(qc_sel.num_qubits, 'q')
#     ancilla_reg = AncillaRegister(1, 'anc')
#
#     # Create a new QuantumCircuit explicitly with the ancilla last
#     qc_with_anc = QuantumCircuit(qreg_main, ancilla_reg, qc_sel.cregs[0])
#
#     # Append your original (selectively decomposed) circuit explicitly onto this circuit
#     qc_with_anc.compose(qc_sel, qubits=qreg_main[:], inplace=True)
#
#     qc_lin = _linearise_multi_control_gates(qc_with_anc)
#
#     from qiskit.transpiler import CouplingMap
#
#     basis = ["rz", "rx", "cx"]
#
#     # Explicit fully connected coupling (guarantees ancilla availability)
#     num_qubits = qc_lin.num_qubits
#     coupling = CouplingMap.from_full(num_qubits)
#
#     hls_cfg = HLSConfig(
#         mcp={'synthesis_method': 'recursion'},
#         mcx={'synthesis_method': 'recursion'}
#     )
#
#     qc_mapped = transpile(
#         qc_lin,
#         basis_gates=basis,
#         coupling_map=coupling,
#         hls_config=hls_cfg,
#         optimization_level=3,
#         initial_layout=list(range(num_qubits)),  # explicit initial layout includes ancilla last
#     )
#
#     print("post-HLS : ", qc_mapped.count_ops())

    # from qiskit.circuit.library import MCPhaseGate  # replacement for MCU1Gate
    # from qiskit.transpiler.passes.synthesis import HLSConfig  # correct import
    #
    # hls_cfg = HLSConfig(
    #     mcp={'synthesis_method': 'recursion'},  # Efficient linear method for MCPhaseGate
    #     mcx={'synthesis_method': 'recursion'}  # Keep MCX also linear
    # )
    #
    # qc_lin = _linearise_multi_control_gates(qc_sel)
    # print("After lin decomposition:", qc_sel.count_ops())
    #
    # # simple line-coupling (swap for backend.target if you have one)
    # basis = ["rz", "rx", "cx", "id"]
    # coupling = (CouplingMap(
    #     [[i, i + 1] for i in range(qc_lin.num_qubits - 1)]
    #     + [[i + 1, i] for i in range(qc_lin.num_qubits - 1)]
    # ) if qc_lin.num_qubits > 1 else None)
    #
    # print("Transpiling...")
    # qc_mapped = transpile(
    #     qc_lin,
    #     basis_gates=basis,
    #     coupling_map=coupling,
    #     layout_method=layout_method,
    #     routing_method=routing_method,
    #     hls_config=hls_cfg,
    #     optimization_level=opt,
    # )
    #
    # print("post-HLS : ", qc_mapped.count_ops())

    if draw:
        print(qc_mapped.draw())

    if file_writer:
        file_writer.set("decomposed_depth",     qc_mapped.depth())
        file_writer.set("num_gates_transpiled", sum(qc_mapped.count_ops().values()))
        file_writer.set("num_gates_original",   sum(qc.count_ops().values()))

    return qc_mapped



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

from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGOpNode

# def group_with_dag_atomic_mixed_ready(qc: QuantumCircuit):
#     """
#     DAG + saturated single‐phase scheduling, with a dynamic ready set for efficiency.
#     """
#     dag = circuit_to_dag(qc)
#     op_nodes = list(dag.topological_op_nodes())
#     total = len(op_nodes)
#     scheduled = set()
#     columns = []
#
#     # Map from each node to its unscheduled predecessor count
#     unscheduled_predecessors = {node: 0 for node in op_nodes}
#     successors = {node: [] for node in op_nodes}
#     for node in op_nodes:
#         preds = [p for p in dag.predecessors(node) if isinstance(p, DAGOpNode)]
#         unscheduled_predecessors[node] = len(preds)
#         for p in preds:
#             successors[p].append(node)
#
#     # Initially ready nodes (no unscheduled predecessors)
#     ready_rot = {node for node in op_nodes if unscheduled_predecessors[node] == 0 and node.op.name in ('rz', 'rx')}
#     ready_cx = {node for node in op_nodes if unscheduled_predecessors[node] == 0 and node.op.name == 'cx'}
#
#     while len(scheduled) < total:
#         col_nodes = []
#         busy_qubits = set()
#         rot_layer = []
#
#         # --- 1) Drain all ready rotations (rz, rx) ---
#         # All ready rz/rx can be scheduled in this layer, regardless of qubit conflicts
#         rot_layer = list(ready_rot)
#         col_nodes.extend(rot_layer)
#
#         # No busy_qubits update needed here for rotations
#         busy_qubits.update(q._index for node in rot_layer for q in node.qargs)
#
#         # --- 2) Greedily pack ready CXs that do not touch busy qubits ---
#         cx_to_add = set()
#         for node in ready_cx:
#             qs = [q._index for q in node.qargs]
#             if not any(q in busy_qubits for q in qs):
#                 col_nodes.append(node)
#                 cx_to_add.add(node)
#                 busy_qubits.update(qs)
#
#         # If nothing could be scheduled, something is wrong
#         if not col_nodes:
#             raise RuntimeError("Stalled scheduling: circular dependency detected.")
#
#         # Mark as scheduled
#         scheduled.update(col_nodes)
#         columns.append([
#             (node.op, node.qargs, node.cargs)
#             for node in col_nodes
#         ])
#
#         # Update the ready sets for next iteration
#         # Remove scheduled nodes
#         ready_rot.difference_update(rot_layer)
#         ready_cx.difference_update(cx_to_add)
#
#         # For each scheduled node, update its successors' unscheduled count
#         for node in col_nodes:
#             for succ in successors[node]:
#                 unscheduled_predecessors[succ] -= 1
#                 if unscheduled_predecessors[succ] == 0:
#                     if succ.op.name in ('rz', 'rx'):
#                         ready_rot.add(succ)
#                     elif succ.op.name == 'cx':
#                         ready_cx.add(succ)
#                     # Extend here for other gate types if needed
#
#     return columns

from collections import defaultdict
import heapq
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit.dagnode import DAGOpNode
from qiskit.circuit import QuantumCircuit


def group_with_dag_atomic_mixed(qc: QuantumCircuit):
    """
    Faster but behaviour‑identical version of the ‘DAG + saturated single‑phase
    scheduling’ algorithm.  Complexity: Θ(V + E) instead of Θ(V·d).

    See original docstring for semantics (unchanged).
    """
    dag = circuit_to_dag(qc)
    op_nodes = list(dag.topological_op_nodes())
    idx = {node: i for i, node in enumerate(op_nodes)}          # stable order key

    # --- build static dependency graph ---------------------------------------
    indeg = {}                          # remaining unscheduled predecessors
    succs = defaultdict(list)           # forward adjacency
    for n in op_nodes:
        preds = [p for p in dag.predecessors(n) if isinstance(p, DAGOpNode)]
        indeg[n] = len(preds)
        for p in preds:
            succs[p].append(n)

    # --- two priority queues keep the ready set partitioned by gate type -----
    ready_rot, ready_cx = [], []        # (topo_index, node)

    def push_ready(node):
        """Put node into the correct ready‑queue, preserving topo order."""
        h = ready_rot if node.op.name in ('rz', 'rx') else ready_cx
        heapq.heappush(h, (idx[node], node))

    for n in op_nodes:                  # initial ready set
        if indeg[n] == 0:
            push_ready(n)

    scheduled = set()
    columns = []

    while ready_rot or ready_cx:        # until every op scheduled
        col_nodes, busy = [], set()

        # -------- 1) drain all ready rotations *recursively* ---------------
        while ready_rot:
            _, node = heapq.heappop(ready_rot)
            col_nodes.append(node)
            scheduled.add(node)

            for suc in succs[node]:
                indeg[suc] -= 1
                if indeg[suc] == 0:
                    push_ready(suc)

        # mark their qubits busy for the CX‑packing phase
        busy.update(q._index for n in col_nodes for q in n.qargs)

        # -------- 2) greedily pack compatible ready CXs ---------------------
        tmp = []
        while ready_cx:
            _, node = heapq.heappop(ready_cx)
            qidx = [q._index for q in node.qargs]
            if any(q in busy for q in qidx):        # conflict → next column
                tmp.append((idx[node], node))
                continue

            # schedule this CX
            col_nodes.append(node)
            busy.update(qidx)
            scheduled.add(node)

            for suc in succs[node]:
                indeg[suc] -= 1
                if indeg[suc] == 0:
                    push_ready(suc)

        # left‑over CXs still ready but blocked by busy qubits
        for item in tmp:
            heapq.heappush(ready_cx, item)

        if not col_nodes:                              # should never happen
            raise RuntimeError("Stalled scheduling: circular dependency detected.")

        columns.append([(n.op, n.qargs, n.cargs) for n in col_nodes])

    return columns


# def group_with_dag_atomic_mixed(qc: QuantumCircuit):
#     """
#     DAG + saturated single‐phase scheduling:
#       1) In each column, drain *all* ready rotations (rz, rx)—
#          merging entire chains of them (e.g. rz→rx→rz) into one layer,
#          with no busy‐qubit checks among these rotations.
#       2) Then greedily pack ready CXs that don’t touch any qubit
#          already used by those rotations or by each other.
#       Repeat until every operation is scheduled.
#     """
#     dag = circuit_to_dag(qc)
#     # dag.draw(filename='images/DAGs/thesis_dag.png')  # 'mpl' uses matplotlib to save as PNG
#     op_nodes = list(dag.topological_op_nodes())
#     total = len(op_nodes)
#     scheduled = set()
#     columns = []
#
#     while len(scheduled) < total:
#         col_nodes = []
#         busy_qubits = set()
#
#         # --- 1) Drain *all* ready rotations into rot_layer ---
#         rot_layer = []
#         while True:
#             added = False
#             for node in op_nodes:
#                 if node in scheduled or node in rot_layer:
#                     continue
#                 if node.op.name not in ('rz', 'rx'):
#                     continue
#                 # only consider op‐node predecessors
#                 preds = [p for p in dag.predecessors(node)
#                          if isinstance(p, DAGOpNode)]
#                 # allow preds to be either already scheduled or in rot_layer
#                 if any((p not in scheduled and p not in rot_layer) for p in preds):
#                     continue
#                 # this rotation is ready — absorb into this rotation‐chain
#                 rot_layer.append(node)
#                 added = True
#             if not added: # All nodes were added
#                 break
#
#         # add all those rotations at once (no busy_qubits update here)
#         if rot_layer:
#             col_nodes.extend(rot_layer)
#
#         # --- 2) Greedily pack CXs respecting busy_qubits ---
#         # mark any qubits used by these rotations as busy for CXs
#         busy_qubits.update(q._index for node in rot_layer for q in node.qargs)
#
#         added = True
#         while added:
#             added = False
#             for node in op_nodes:
#                 if node in scheduled or node in col_nodes:
#                     continue
#                 if node.op.name != 'cx':
#                     continue
#                 # dep check
#                 preds = [p for p in dag.predecessors(node)
#                          if isinstance(p, DAGOpNode)]
#                 if any(p not in scheduled for p in preds):
#                     continue
#                 # conflict check
#                 qs = [q._index for q in node.qargs]
#                 if any(q in busy_qubits for q in qs):
#                     continue
#                 # can schedule this CX
#                 col_nodes.append(node)
#                 busy_qubits.update(qs)
#                 added = True
#
#         if not col_nodes:
#             raise RuntimeError("Stalled scheduling: circular dependency detected.")
#
#         # record this mixed column
#         columns.append([
#             (node.op, node.qargs, node.cargs)
#             for node in col_nodes
#         ])
#         scheduled.update(col_nodes)
#
#     return columns


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
from qiskit.circuit.instruction import Instruction

#
# def align_cx_matrix(cx_matrix: List[List[List[Instruction]]]) -> List[List[List[Instruction]]]:
#     """
#     Shift each CX “brick” so that
#       1. top_row % 2 == column % 2  (parity)
#       2. for any prior brick sharing a qubit row, there's at least one empty column
#          (or preserve original gap, if it was larger).
#
#     Returns a new matrix; does not mutate the input.
#     """
#     original = copy.deepcopy(cx_matrix)
#     n_rows = len(original)
#     n_cols = max((len(r) for r in original), default=0)
#
#     # Gather each brick’s rows and original column
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
#     # Build and sort bricks by orig_col
#     bricks = []
#     for gid, info in group_info.items():
#         bricks.append({
#             "gid": gid,
#             "rows": sorted(info["rows"]),
#             "orig_col": info["orig_col"],
#             "top": min(info["rows"])
#         })
#     bricks.sort(key=lambda b: b["orig_col"])
#
#     # Prepare the output grid
#     aligned: List[List[List[Instruction]]] = [[] for _ in range(n_rows)]
#     placed_cols: Dict[str, int] = {}
#
#     for brick in bricks:
#         gid      = brick["gid"]
#         rows     = brick["rows"]
#         ocol     = brick["orig_col"]
#         top_row  = brick["top"]
#
#         # 1) Parity‐aligned base column
#         if (ocol % 2) == (top_row % 2):
#             base_col = ocol
#         else:
#             base_col = ocol + 1
#
#         # 2) Compute the floor from spacing constraints against every overlapping brick
#         floor = 0
#         for prev in bricks:
#             pg = prev["gid"]
#             if pg not in placed_cols:
#                 continue
#             # only consider if they share at least one qubit
#             if set(prev["rows"]) & set(rows):
#                 orig_gap = ocol - prev["orig_col"]
#                 # at least one empty column, or preserve if originally larger
#                 needed = placed_cols[pg] + max(1, orig_gap)
#                 floor = max(floor, needed)
#
#         # 3) Final placement is the minimal ≥ both base_col and floor that satisfies parity
#         place = max(base_col, floor)
#         if place % 2 != top_row % 2:
#             place += 1
#
#         # grow output rows as needed
#         for r in aligned:
#             while len(r) <= place:
#                 r.append([])
#
#         # copy exactly those CX‐instructions from the original column
#         for r in rows:
#             instrs = [
#                 instr for instr in original[r][ocol]
#                 if instr.name.startswith(f"cx{gid}")
#             ]
#             aligned[r][place] = instrs
#
#         placed_cols[gid] = place
#
#     # visualiser.print_matrix(aligned)
#     return aligned

from typing import List, Dict
from qiskit.circuit.instruction import Instruction

# helper type
Matrix = List[List[List[Instruction]]]


def align_cx_matrix(cx_matrix: Matrix) -> Matrix:
    """
    Behaviour‑preserving rewrite of the original algorithm.

    Complexity improvement
    ----------------------
    * old: Θ(B²) because every brick scanned every earlier brick
    * new: Θ(B · r)  (r ≤ 2) using per‑row aggregates
    """

    # -------------------------------------------------------------------------
    # 1.  Collect bricks (same as before, but without the expensive deepcopy)
    # -------------------------------------------------------------------------
    n_rows = len(cx_matrix)
    n_cols = max((len(r) for r in cx_matrix), default=0)

    group_info: Dict[str, Dict] = {}
    for r in range(n_rows):
        for c in range(n_cols):
            for instr in (cx_matrix[r][c] if c < len(cx_matrix[r]) else []):
                if instr.name.startswith("cx"):
                    gid = instr.name[2:-1]
                    info = group_info.setdefault(gid, {"rows": set(), "orig_col": c})
                    info["rows"].add(r)
                    info["orig_col"] = min(info["orig_col"], c)

    bricks = [
        {
            "gid": gid,
            "rows": sorted(info["rows"]),
            "orig_col": info["orig_col"],
            "top": min(info["rows"]),
        }
        for gid, info in group_info.items()
    ]
    bricks.sort(key=lambda b: b["orig_col"])        # stable original order

    # -------------------------------------------------------------------------
    # 2.  Fast placement using per‑row aggregates
    # -------------------------------------------------------------------------
    aligned: Matrix = [[] for _ in range(n_rows)]

    row_max_delta: Dict[int, int] = {}              # row → max(placed - orig)
    row_same_orig: Dict[int, Dict[int, int]] = {}   # row → {orig_col: placed}

    placed_cols: Dict[str, int] = {}                # gid → placed column

    for brick in bricks:
        gid, rows, ocol, top_row = (
            brick["gid"],
            brick["rows"],
            brick["orig_col"],
            brick["top"],
        )

        # --- 1) parity‑aligned base column -----------------------------------
        base_col = ocol if (ocol % 2) == (top_row % 2) else ocol + 1

        # --- 2) spacing floor from *aggregates* ------------------------------
        floor = 0
        for r in rows:
            # prior bricks on this row             needed = (placed - orig) + ocol
            if r in row_max_delta:
                floor = max(floor, row_max_delta[r] + ocol)

            # special case: previous brick at *same* original column
            if (same := row_same_orig.get(r, {}).get(ocol)) is not None:
                floor = max(floor, same + 1)

        # --- 3) minimal ≥ both base_col and floor, keeping parity ------------
        place = max(base_col, floor)
        if place % 2 != top_row % 2:
            place += 1

        # --- 4) ensure output grid wide enough --------------------------------
        for r in aligned:
            while len(r) <= place:
                r.append([])

        # --- 5) copy exactly the CXs that belong to this brick ----------------
        for r in rows:
            instrs = [
                instr
                for instr in cx_matrix[r][ocol]
                if instr.name.startswith(f"cx{gid}")
            ]
            aligned[r][place] = instrs

            # update per‑row aggregates
            delta = place - ocol
            row_max_delta[r] = max(row_max_delta.get(r, -1), delta)
            row_same_orig.setdefault(r, {})[ocol] = place

        placed_cols[gid] = place

    return aligned

from typing import List, Dict
from qiskit.circuit.instruction import Instruction


Matrix = List[List[List[Instruction]]]           # alias for readability


def insert_rotations_adjecant_to_cx(
    aligned_cx: Matrix,
    original:   Matrix,
) -> Matrix:
    """
    Behaviour‑preserving, asymptotically faster version.

    For every rotation list in `original[i][j]`
      – if `original[i][j‑1]` contains a CX, insert *after* that CX
      – elif `original[i][j+1]` contains a CX, insert *before* that CX
      – else insert into column 0.

    Complexity: Θ(R · C) instead of Θ(R · C · W)
    (R = rows, C = max columns in `original`, W = final width of output)
    """
    # ------------------------------------------------------------------
    # 0) Deep‑copy aligned CX matrix so we can mutate it
    # ------------------------------------------------------------------
    new_mat: Matrix = [[list(cell) for cell in row] for row in aligned_cx]

    n_rows = len(new_mat)
    if n_rows == 0:
        return new_mat

    width = max(len(r) for r in new_mat)           # current uniform width

    # ------------------------------------------------------------------
    # 1) Per‑row dict:  CX‑instruction.name  →  column index in new_mat
    # ------------------------------------------------------------------
    cx_col: List[Dict[str, int]] = []
    for row in new_mat:
        lookup: Dict[str, int] = {}
        for c, cell in enumerate(row):
            for ins in cell:                       # only CXs matter
                if ins.name.startswith("cx") and ins.name not in lookup:
                    lookup[ins.name] = c           # first (leftmost) column
        cx_col.append(lookup)

    # ------------------------------------------------------------------
    # 2) Walk `original`, deposit rotations using O(1) look‑ups
    # ------------------------------------------------------------------
    for i in range(n_rows):
        row_orig = original[i]
        lookup   = cx_col[i]

        for j, cell in enumerate(row_orig):
            # extract rotations once
            rots = [ins for ins in cell if not ins.name.startswith("cx")]
            if not rots:
                continue

            dest = 0
            # ---- rule 1: look at left neighbour ----------------------
            if j > 0:
                for ins in row_orig[j - 1]:
                    if ins.name.startswith("cx"):
                        dest = lookup[ins.name] + 1
                        break

            # ---- rule 2: otherwise look at right neighbour ----------
            if dest == 0 and j + 1 < len(row_orig):
                for ins in row_orig[j + 1]:
                    if ins.name.startswith("cx"):
                        dest = max(lookup[ins.name] - 1, 0)
                        break

            # ---- rule 3: enlarge grid if necessary ------------------
            if dest >= width:
                extra = dest - width + 1
                for r in new_mat:
                    r.extend([[]] * extra)
                width += extra            # keep invariant

            # ---- final: append rotations in order -------------------
            new_mat[i][dest].extend(rots)

    return new_mat



from typing import List
#
# def insert_rotations_adjecant_to_cx(
#     aligned_cx: List[List[List[Instruction]]],
#     original:   List[List[List[Instruction]]]
# ) -> List[List[List[Instruction]]]:
#     """
#     aligned_cx: CX-only matrix after align_cx_matrix
#     original:   full matrix (CX + rotations)
#     Returns new matrix where each rotation-list from original[i][j]:
#       - if original[i][j-1] has a CX → insert after that CX
#       - elif original[i][j+1] has a CX → insert in the group *before* that CX
#       - else → insert into column 0
#     """
#     # deep-copy so we can append
#     new_mat = [
#         [list(cell) for cell in row]
#         for row in aligned_cx
#     ]
#     n_rows = len(new_mat)
#     if n_rows == 0:
#         return new_mat
#
#     # normalize width
#     width = max(len(r) for r in new_mat)
#     for r in new_mat:
#         if len(r) < width:
#             r.extend([[]] * (width - len(r)))
#
#     for i in range(n_rows):
#         row_orig = original[i]
#         for j, cell in enumerate(row_orig):
#             rots = [ins for ins in cell if not ins.name.startswith('cx')]
#             if not rots:
#                 continue
#
#             dest = 0
#             group_instr = None
#
#             # 1) LEFT neighbor → after-CX
#             if j > 0:
#                 left_cx = [ins for ins in row_orig[j-1] if ins.name.startswith('cx')]
#                 if left_cx:
#                     group_instr = left_cx[0]
#                     # find its column in aligned_cx
#                     for c, aligned_cell in enumerate(new_mat[i]):
#                         if any(ins.name == group_instr.name for ins in aligned_cell):
#                             dest = c + 1
#                             break
#
#             # 2) RIGHT neighbor → before-CX (now one left of CX)
#             if group_instr is None and j+1 < len(row_orig):
#                 right_cx = [ins for ins in row_orig[j+1] if ins.name.startswith('cx')]
#                 if right_cx:
#                     group_instr = right_cx[0]
#                     for c, aligned_cell in enumerate(new_mat[i]):
#                         if any(ins.name == group_instr.name for ins in aligned_cell):
#                             dest = max(c - 1, 0)
#                             break
#
#             # 3) ensure room
#             if dest >= len(new_mat[i]):
#                 extra = dest - len(new_mat[i]) + 1
#                 for r in new_mat:
#                     r.extend([[]] * extra)
#
#             # append in order
#             new_mat[i][dest].extend(rots)
#
#     return new_mat



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
