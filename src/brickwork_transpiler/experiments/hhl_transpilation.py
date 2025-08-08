import os, csv, math, numpy as np

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
              "num_gates_original", "num_gates_transpiled"]
    full_path = os.path.join(file_path, file_name)

    os.makedirs(file_path, exist_ok=True)
    if not os.path.isfile(full_path):
        with open(full_path, "w", newline="") as fh:
            csv.writer(fh).writerow(header)

    return src.brickwork_transpiler.utils.BufferedCSVWriter(full_path, header)

# ---------------------------------------------------------------------------
# Utility: build a small-κ, Hermitian, 2^n × 2^n matrix
# ---------------------------------------------------------------------------
def make_well_conditioned_matrix(num_qubits: int, kappa_target: float = 1.5):
    """
    Returns A_n = U D U^† with spectrum in [1, kappa_target]
    so that κ(A_n) ≈ kappa_target and dimension = 2**num_qubits.
    """
    dim = 1 << num_qubits                         # 2**num_qubits
    # Diagonal spectrum between 1 and kappa_target
    evals = np.linspace(1.0, kappa_target, dim)
    D = np.diag(evals)

    # Random unitary via QR of a complex Ginibre matrix
    X = (np.random.randn(dim, dim) +
         1j * np.random.randn(dim, dim)) / math.sqrt(2.0)
    Q, _ = np.linalg.qr(X)                        # unitary Q
    A = Q @ D @ Q.conj().T
    # Ensure exactly Hermitian (numerical precision)
    A = (A + A.conj().T) / 2.0
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

    # for n in range(1, max_qubits + 1):
    #     # 1) Generate well-conditioned matrix A_n and normalised |b>
    #     A, _ = make_well_conditioned_matrix(n)
    #     b    = np.random.rand(1 << n)
    #     b    = b / np.linalg.norm(b)
    #
    #     # 2) Build HHL circuit
    #     hhl   = HHL(quantum_instance=qinstance)
    #     circ  = hhl.construct_circuit(A, b)
    #
    #     # 3) Brickwork transpilation
    #     instr_mat = brickwork_transpiler.transpile(
    #         circ,
    #         routing_method="sabre",
    #         layout_method="sabre",
    #         return_mat=True,
    #         file_writer=writer
    #     )
    #
    #     # 4) Collect metrics
    #     writer.set("original_depth",      circ.depth())
    #     writer.set("transpiled_depth",    len(instr_mat[0]))
    #     writer.flush()

    plot_single_hhl_dataset("hhl_transpilation_cost_experiment_plot")


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D


def plot_single_hhl_dataset(name_of_plot: str = "hhl_default.png"):
    """
    Visualise brickwork–transpilation cost for the HHL circuits stored in
    experiments_hhl.csv.  The file must contain (at least) the columns
        decomposed_depth, transpiled_depth, num_gates_original
    produced by experiment_hhl_transpilation().
    """
    # --------------------------------------------------------------------
    # Load data   – drop n = 0 to avoid log(0) singularities
    # --------------------------------------------------------------------
    base_dir = "src/brickwork_transpiler/experiments/data/output_data/"
    filename = "experiments_hhl.csv"

    df = pd.read_csv(os.path.join(base_dir, filename))
    df = df[df["num_gates_original"] > 0]

    n                = df["num_gates_original"].to_numpy()       # x-axis
    decomposed_depth = df["decomposed_depth"].to_numpy()
    transpiled_depth = df["transpiled_depth"].to_numpy()

    # Reference curve  c·log³ n  (poly-log depth expected for HHL)
    c      = 4.0
    ref    = c * np.log(n) ** 3

    # --------------------------------------------------------------------
    # Styling
    # --------------------------------------------------------------------
    colours = {"decomposed_depth": "orange",
               "transpiled_depth": "green",
               "ref":              "blue"}
    markers = {"decomposed_depth": "s",
               "transpiled_depth": "^"}
    labels  = {"decomposed_depth": "Decomposed circuit depth",
               "transpiled_depth": "Brickwork graph depth",
               "ref":              r"$c \cdot \log^{3} n$"}

    mark_every = max(1, len(n) // 8)                      # sparsify markers

    y_min = max(1, np.min([decomposed_depth, transpiled_depth, ref]))
    y_max = np.max([decomposed_depth, transpiled_depth, ref]) * 1.3

    # --------------------------------------------------------------------
    # Figure
    # --------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))

    for key in ("decomposed_depth", "transpiled_depth"):
        ax.plot(n, locals()[key],
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

    ax.plot(n, ref,
            label=labels["ref"],
            color=colours["ref"],
            linestyle="--",
            linewidth=2.4,
            alpha=0.9)

    # --------------------------------------------------------------------
    # Axes and legend
    # --------------------------------------------------------------------
    ax.set_title("HHL Transpilation Cost", fontsize=15, fontweight="bold")
    ax.set_xlabel(r"Input circuit size $n$ (log scale)", fontsize=13)
    ax.set_ylabel(r"Depth (log scale)", fontsize=13)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(y_min, y_max)

    ax.xaxis.set_major_locator(mticker.LogLocator(base=10, numticks=10))
    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"{int(v):d}" if v >= 1 else "")
    )
    ax.xaxis.set_minor_locator(mticker.LogLocator(base=10,
                                                  subs=np.arange(1, 10) * 0.1,
                                                  numticks=100))
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())

    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.grid(axis="y", which="both", linestyle=":", alpha=0.35)

    # Custom legend with matching markers
    legend_handles = [
        Line2D([0], [0], color=colours["decomposed_depth"],
               marker=markers["decomposed_depth"], linestyle="-",
               markersize=8, label=labels["decomposed_depth"],
               markeredgecolor="white", markeredgewidth=1.7),
        Line2D([0], [0], color=colours["transpiled_depth"],
               marker=markers["transpiled_depth"], linestyle="-",
               markersize=8, label=labels["transpiled_depth"],
               markeredgecolor="white", markeredgewidth=1.7),
        Line2D([0], [0], color=colours["ref"], linestyle="--", linewidth=3,
               label=labels["ref"]),
    ]
    ax.legend(handles=legend_handles, fontsize=13, frameon=False, loc="best")

    plt.tight_layout()
    plt.savefig(f"images/qrs/qrs_{name_of_plot}", dpi=300, bbox_inches="tight")
    plt.show()
