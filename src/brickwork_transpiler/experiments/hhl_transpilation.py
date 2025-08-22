import os, csv, math, numpy as np

from numpy.f2py.capi_maps import c2capi_map
from qiskit import Aer, QuantumCircuit
from qiskit.utils import QuantumInstance
import linear_solvers                # from quantum_linear_solvers
from linear_solvers import HHL
import src.brickwork_transpiler.utils
from src.brickwork_transpiler import brickwork_transpiler


# ---------------------------------------------------------------------------
# CSV helper – identical header to your QFT experiment
# ---------------------------------------------------------------------------
def get_writer(file_name: str,
               file_path: str = "src/brickwork_transpiler/experiments/data/output_data/"):

    header = ["decomposed_depth", "transpiled_depth", "original_depth",
              "num_gates_original", "num_gates_transpiled", "N", "kappa", "s_sparsity"]
    full_path = os.path.join(file_path, file_name)

    os.makedirs(file_path, exist_ok=True)
    if not os.path.isfile(full_path):
        with open(full_path, "w", newline="") as fh:
            csv.writer(fh).writerow(header)

    return src.brickwork_transpiler.utils.BufferedCSVWriter(full_path, header)

# # ---------------------------------------------------------------------------
# # Utility: build a small-κ, Hermitian, 2^n × 2^n matrix
# # ---------------------------------------------------------------------------
# def make_well_conditioned_matrix(num_qubits: int, kappa_target: float = 1.5):
#     """
#     Returns A_n = U D U^† with spectrum in [1, kappa_target]
#     so that κ(A_n) ≈ kappa_target and dimension = 2**num_qubits.
#     """
#     dim = 1 << num_qubits                         # 2**num_qubits
#     # Diagonal spectrum between 1 and kappa_target
#     evals = np.linspace(1.0, kappa_target, dim)
#     D = np.diag(evals)
#
#     # Random unitary via QR of a complex Ginibre matrix
#     X = (np.random.randn(dim, dim) +
#          1j * np.random.randn(dim, dim)) / math.sqrt(2.0)
#     Q, _ = np.linalg.qr(X)                        # unitary Q
#     A = Q @ D @ Q.conj().T
#     # Ensure exactly Hermitian (numerical precision)
#     A = (A + A.conj().T) / 2.0
#     return A, evals

import numpy as np
import math
from typing import Tuple

# ---------------------------------------------------------------------------
# Simplest HHL-friendly matrix:
#   - A0 = tridiagonal Laplacian on a path (open chain): s = 3
#   - Positive definite (no zero mode), Hermitian
#   - Rescaled so spec(A) ⊂ [1, kappa_target]
#   - Returns (A_dense, evals) to match your current interface
# ---------------------------------------------------------------------------
def make_well_conditioned_matrix(num_qubits: int,
                                 kappa_target: float = 1.5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build an s=3 sparse Hermitian A of size N=2**num_qubits, with spectrum in [1, kappa_target].

    We use the path-graph Laplacian with open boundaries (tridiagonal Toeplitz):
        A0 = 2 I - T - T^T
    whose eigenvalues are known in closed form:
        λ_j(A0) = 2 - 2 cos(π j / (N+1)),   j = 1..N  (all > 0)

    We then map [λ_min, λ_max] -> [1, kappa_target] via A = a*A0 + b*I.
    """
    N = 1 << num_qubits  # matrix dimension = 2**num_qubits

    # --- build tridiagonal A0 (dense for simplicity; keep s=3 structure) ---
    A0 = np.zeros((N, N), dtype=float)
    np.fill_diagonal(A0, 2.0)
    i = np.arange(N - 1)
    A0[i, i + 1] = -1.0
    A0[i + 1, i] = -1.0
    # (Open chain: no wrap-around; ensures positive definiteness.)

    # --- closed-form eigen-extrema for A0 on a path graph ---
    # λ_min = 2 - 2 cos(π/(N+1))
    # λ_max = 2 - 2 cos(π N/(N+1)) = 2 + 2 cos(π/(N+1))
    theta = math.pi / (N + 1.0)
    lam_min = 2.0 - 2.0 * math.cos(theta)
    lam_max = 2.0 + 2.0 * math.cos(theta)

    # --- affine rescaling A = a*A0 + b*I so that spec(A) ⊂ [1, kappa_target] ---
    if abs(lam_max - lam_min) < 1e-15:
        # Degenerate (won't happen here), just return identity
        A = np.eye(N, dtype=float)
        evals = np.ones(N, dtype=float)
        return A, evals

    a = (kappa_target - 1.0) / (lam_max - lam_min)
    b = 1.0 - a * lam_min

    A = a * A0 + b * np.eye(N, dtype=float)

    # (Optional) eigenvalues if you want to log them quickly without eig():
    j = np.arange(1, N + 1, dtype=float)
    evals0 = 2.0 - 2.0 * np.cos(math.pi * j / (N + 1.0))
    evals = a * evals0 + b

    return A, evals


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------
def experiment_hhl_transpilation(max_qubits: int = 5):
    """
    Builds and transpiles HHL circuits for 1 … max_qubits system qubits.
    Results saved to experiments_hhl.csv in the same format as QFT data.
    """
    filename = "experiments_hhl.csv"
    writer   = get_writer(filename)

    backend   = Aer.get_backend("aer_simulator_statevector")
    qinstance = QuantumInstance(backend)

    for n in range(1, max_qubits + 1):

        from linear_solvers.matrices import TridiagonalToeplitz
        # ---- 1) Well-conditioned tridiagonal Toeplitz (κ constant in n) ----
        # Choose a > 2|b|, e.g., a=1, b=-1/6 -> κ = (a+2|b|)/(a-2|b|) = 2
        # A_circuit = TridiagonalToeplitz(
        #     num_state_qubits=n,
        #     main_diag=1.0,
        #     off_diag=-1.0 / 6.0,
        #     trotter_steps=2,  # increase if you need tighter simulation accuracy
        #     # evolution_time=1.0,   # default okay; lower for a smaller Trotter error
        #     name=f"tridi_{n}"
        # )

        # ---- 2) Efficient |b> state-prep: O(n) gates, not O(N) data loading ----
        # Option A: uniform superposition |b> = H^{⊗ n} |0...0>
        b_prep = QuantumCircuit(n, name=f"b_uniform_{n}")
        b_prep.h(range(n))

        from scipy import sparse
        A = sparse.random(n, n, density=0.05, format='csr')
        A = (A + A.T) / 2  # Hermitian, non-1D

        # b_prep    = np.random.rand(1 << n)
        # b_prep   = b_prep / np.linalg.norm(b_prep)


        # ---- 3) Build HHL circuit using circuit-based rhs preparation ----
        hhl = HHL(quantum_instance=qinstance)
        circ = hhl.construct_circuit(A, b_prep)

        # # 1) Generate well-conditioned matrix A_n and normalised |b>
        # # A, evals = make_well_conditioned_matrix(n)
        # b    = np.random.rand(1 << n)
        # b    = b / np.linalg.norm(b)
        #
        # # 2) Build HHL circuit
        # from linear_solvers.matrices import TridiagonalToeplitz
        #
        # # n = number of system qubits
        # # A_circuit = TridiagonalToeplitz(n, main_diag=2.0, off_diag=1.0, trotter_steps=2)  # s=3, structured
        # A_circuit = TridiagonalToeplitz(
        #     num_state_qubits=10,
        #     main_diag=1.0,
        #     off_diag=-1.0 / 6.0,
        #     trotter_steps=2
        # )
        #
        # hhl   = HHL(quantum_instance=qinstance)
        # circ  = hhl.construct_circuit(A_circuit, b)
        # # 3) Brickwork transpilation

        instr_mat = brickwork_transpiler.transpile(
            circ,
            routing_method="sabre",
            layout_method="sabre",
            return_mat=True,
            file_writer=writer,
            with_ancillas=True,
            plot_decomposed=False
        )

        # 4) Collect metrics
        writer.set("original_depth",      circ.depth())
        writer.set("transpiled_depth",    len(instr_mat[0]))
        writer.set("N",    2**n)
        # kappa = float(np.max(evals) / np.min(evals))
        # writer.set("kappa", kappa)
        writer.set("s_sparsity", 3)
        writer.flush()

    plot_single_hhl_dataset("hhl_transpilation_cost_experiment_plot_with_ancillae")


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D


def plot_single_hhl_dataset(name_of_plot: str = "hhl_default.png"):
    """
    Visualise brickwork–transpilation cost for the HHL circuits stored in
    experiments_hhl.csv.

    Required CSV columns:
        decomposed_depth, transpiled_depth, N

    Three curves are shown:
        • decomposed circuit depth
        • brickwork-transpiled depth
        • c·log²(n)                      ← NEW
    plus the original reference c·log³(n).

    The x-axis is n = 2^N (powers of two displayed as clean 10^k ticks).
    """


    # --------------------------------------------------------------------
    # Load data
    # --------------------------------------------------------------------
    base_dir = "src/brickwork_transpiler/experiments/data/output_data/"
    filename = "experiments_hhl.csv"


    df = pd.read_csv(os.path.join(base_dir, filename))
    df = df[df["N"] > 0]

    # --- inputs already loaded above this block ---
    # df with columns: decomposed_depth, transpiled_depth, N
    # -----------------------------------------------------

    # Interpret CSV's N as exponent n; avoid forming huge integers
    n = df["N"].to_numpy(dtype=float)  # e.g. [2,4,8,16,...]
    N = np.power(2, n)
    logN = np.log(N)  # log(2**n)
    Nsize = np.exp(logN)  # equals 2**n, in float

    decomposed_depth = df["decomposed_depth"].to_numpy(dtype=float)
    transpiled_depth = df["transpiled_depth"].to_numpy(dtype=float)

    # Reference curves: choose constants, match names to powers
    c_log2, c_log3 = 14000.0, 1000.0 #10k to
    ref_log2 = c_log2 * (logN)
    ref_log3 = c_log3 * (logN**2)

    # --------------------------------------------------------------------
    # Styling
    # --------------------------------------------------------------------
    colours = {"decomposed_depth": "orange",
               "transpiled_depth": "green",
               "ref_log2": "red",
               "ref_log3": "blue"}
    markers = {"decomposed_depth": "s",
               "transpiled_depth": "^"}
    labels = {"decomposed_depth": "Decomposed depth",
              "transpiled_depth": "Brickwork depth",
              "ref_log2": r"$c_{1}\,\cdot\log N$",
              "ref_log3": r"$c_{2}\,\cdot\log^{2} N$"}

    mark_every = max(1, len(Nsize) // 8)

    # Robust bounds for log-scale: finite & positive only
    y_all = np.r_[decomposed_depth, transpiled_depth, ref_log2, ref_log3].astype(float)
    y_all = y_all[np.isfinite(y_all) & (y_all > 0)]
    if y_all.size == 0:
        raise ValueError("No positive finite y-values for log-scale.")
    y_min = max(1.0, y_all.min())
    y_max = y_all.max() * 1.3
    if y_min == y_max:  # degenerate case padding
        pad = 0.05 * y_min
        y_min, y_max = y_min - pad, y_max + pad

    # --------------------------------------------------------------------
    # Figure
    # --------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))

    for key in ("decomposed_depth", "transpiled_depth"):
        ax.plot(Nsize, locals()[key],
                label=labels[key],
                marker=markers[key],
                markevery=mark_every,
                linestyle="-",
                color=colours[key],
                linewidth=2.7,
                markersize=8,
                alpha=0.97,
                markeredgecolor="white",
                markeredgewidth=1.7)

    # Reference curves (no markers)
    ax.plot(Nsize, ref_log3,
            label=labels["ref_log3"],
            color=colours["ref_log3"],
            linestyle="--",
            linewidth=2.4,
            alpha=0.9)
    ax.plot(Nsize, ref_log2,
            label=labels["ref_log2"],
            color=colours["ref_log2"],
            linestyle="-.",
            linewidth=2.4,
            alpha=0.9)

    print("y_min=", y_min, "y_max=", y_max)
    print("finite?", np.isfinite([y_min, y_max]))
    print("positive?", (y_min > 0, y_max > 0))

    # --------------------------------------------------------------------
    # Axes and legend
    # --------------------------------------------------------------------
    ax.set_title("HHL Transpilation Cost", fontsize=15, fontweight="bold")
    ax.set_xlabel(r"Problem size $N = 2^{n}$", fontsize=13)
    ax.set_ylabel("Depth (log scale)", fontsize=13)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(y_min, y_max)

    # Clean powers-of-10 ticks on N=2^n (x-axis)
    ax.xaxis.set_major_locator(mticker.LogLocator(base=10.0))
    ax.xaxis.set_major_formatter(mticker.LogFormatterMathtext(base=10.0))
    ax.xaxis.set_minor_locator(mticker.NullLocator())

    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.grid(axis="y", which="both", linestyle=":", alpha=0.35)

    legend_handles = [
        Line2D([0], [0], color=colours["decomposed_depth"],
               marker=markers["decomposed_depth"], linestyle="-",
               markersize=8, label=labels["decomposed_depth"],
               markeredgecolor="white", markeredgewidth=1.7),
        Line2D([0], [0], color=colours["transpiled_depth"],
               marker=markers["transpiled_depth"], linestyle="-",
               markersize=8, label=labels["transpiled_depth"],
               markeredgecolor="white", markeredgewidth=1.7),
        Line2D([0], [0], color=colours["ref_log2"], linestyle="-.",
               linewidth=3, label=labels["ref_log2"]),
        Line2D([0], [0], color=colours["ref_log3"], linestyle="--",
               linewidth=3, label=labels["ref_log3"]),
    ]
    ax.legend(handles=legend_handles, fontsize=13, frameon=False, loc="best")

    plt.tight_layout()
    plt.savefig(f"images/qrs/qrs_{name_of_plot}", dpi=300, bbox_inches="tight")
    plt.show()

    #
    # # Interpret the CSV's N as the exponent n
    # n = df["N"].to_numpy(dtype=float)  # e.g., [2, 4, 8, 16, ...]
    # logN = n * np.log(2.0)  # log(2**n) = n*log(2)
    #
    # decomposed_depth = df["decomposed_depth"].to_numpy(dtype=float)
    # transpiled_depth = df["transpiled_depth"].to_numpy(dtype=float)
    #
    # # Reference curves: choose constants for the intended powers of log(N)
    # c_log2 = 300.0
    # # c_log3 = 1000.0
    # ref_log2 = c_log2 * (logN ** 2)  # c · (log N)^2
    # ref_log3 = c_log3 * (logN ** 3)  # c · (log N)^3
    #
    # # c1 = 1000
    # c2 = 100
    #
    # N = np.power(df["N"].to_numpy(dtype=float), 2.0)
    # # N = np.power(2.0, n)  # float64, no overflow here
    # # or better, avoid huge N entirely:
    # logN = np.log(N)  # since log(2**n) = n*log(2)
    # ref_log2 = c1 * (logN ** 2)
    # ref_log3 = c2 * (logN ** 3)

    # # --------------------------------------------------------------------
    # # Styling
    # # --------------------------------------------------------------------
    # colours = {"decomposed_depth": "orange",
    #            "transpiled_depth": "green",
    #            "ref_log2":         "red",
    #            "ref_log3":         "blue"}
    # markers = {"decomposed_depth": "s",
    #            "transpiled_depth": "^"}
    # labels  = {"decomposed_depth": "Decomposed depth",
    #            "transpiled_depth": "Brickwork depth",
    #            "ref_log2":         r"$c_1 \cdot \log N$",
    #            "ref_log3":         r"$c_2 \cdot \log^{2} N$"}
    #
    # mark_every = max(1, len(N) // 8)
    #
    # y_min = max(1, np.min([*decomposed_depth, *transpiled_depth,
    #                        *ref_log2, *ref_log3]))
    # y_max = np.max([*decomposed_depth, *transpiled_depth,
    #                 *ref_log2, *ref_log3]) * 1.3
    #
    # # --------------------------------------------------------------------
    # # Figure
    # # --------------------------------------------------------------------
    # fig, ax = plt.subplots(figsize=(8, 6))
    #
    # for key in ("decomposed_depth", "transpiled_depth"):
    #     ax.plot(N, locals()[key],
    #             label=labels[key],
    #             marker=markers[key],
    #             markevery=mark_every,
    #             linestyle="-",
    #             color=colours[key],
    #             linewidth=2.7,
    #             markersize=8,
    #             alpha=0.97,
    #             markeredgecolor="white",
    #             markeredgewidth=1.7)
    #
    # # Reference curves (no markers)
    # ax.plot(N, ref_log3,
    #         label=labels["ref_log3"],
    #         color=colours["ref_log3"],
    #         linestyle="--",
    #         linewidth=2.4,
    #         alpha=0.9)
    # ax.plot(N, ref_log2,
    #         label=labels["ref_log2"],
    #         color=colours["ref_log2"],
    #         linestyle="-.",
    #         linewidth=2.4,
    #         alpha=0.9)
    #
    # print("y_min=", y_min, "y_max=", y_max)
    # print("finite?", np.isfinite([y_min, y_max]))
    # print("positive?", (y_min > 0, y_max > 0))
    #
    # # --------------------------------------------------------------------
    # # Axes and legend
    # # --------------------------------------------------------------------
    # ax.set_title("HHL Transpilation Cost", fontsize=15, fontweight="bold")
    # ax.set_xlabel(r"Problem size $N = 2^{n}$", fontsize=13)
    # ax.set_ylabel("Depth (log scale)", fontsize=13)
    # ax.set_xscale("log")
    # ax.set_yscale("log")
    # ax.set_ylim(y_min, y_max)
    #
    # # — clean powers-of-10 ticks only —
    # ax.xaxis.set_major_locator(mticker.LogLocator(base=10.0))
    # ax.xaxis.set_major_formatter(mticker.LogFormatterMathtext(base=10.0))
    # ax.xaxis.set_minor_locator(mticker.NullLocator())
    #
    # ax.tick_params(axis="both", which="major", labelsize=12)
    # ax.grid(axis="y", which="both", linestyle=":", alpha=0.35)
    #
    # # Legend (matching markers/linestyles)
    # legend_handles = [
    #     Line2D([0], [0], color=colours["decomposed_depth"],
    #            marker=markers["decomposed_depth"], linestyle="-",
    #            markersize=8, label=labels["decomposed_depth"],
    #            markeredgecolor="white", markeredgewidth=1.7),
    #     Line2D([0], [0], color=colours["transpiled_depth"],
    #            marker=markers["transpiled_depth"], linestyle="-",
    #            markersize=8, label=labels["transpiled_depth"],
    #            markeredgecolor="white", markeredgewidth=1.7),
    #     Line2D([0], [0], color=colours["ref_log2"], linestyle="-.",
    #            linewidth=3, label=labels["ref_log2"]),
    #     Line2D([0], [0], color=colours["ref_log3"], linestyle="--",
    #            linewidth=3, label=labels["ref_log3"]),
    # ]
    # ax.legend(handles=legend_handles, fontsize=13, frameon=False, loc="best")
    #
    # plt.tight_layout()
    # plt.savefig(f"images/qrs/qrs_{name_of_plot}", dpi=300, bbox_inches="tight")
    # plt.show()
