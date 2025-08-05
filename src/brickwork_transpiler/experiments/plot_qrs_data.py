import matplotlib.pyplot as plt
import numpy as np

def plot_grover_circuit_depths_bars():
    # Obtained data
    data = {
        'Full -- one match': {
            'iterations': [1, 2, 3, 4, 6, 8],
            'orig_depth': [21, 38, 73, 144, 283, 576],
            'decomposed_depth': [298, 1094, 4885, 20916, 83629, 358550],
            'transpiled_depth': [362, 1467, 5486, 23761, 93316, 382663],
        },
        'Full -- no match': {
            'iterations': [0, 0, 0, 0, 0, 0],
            'orig_depth': [21, 38, 73, 144, 283, 576],
            'decomposed_depth': [294, 1079, 4869, 20431, 84129, 353961],
            'transpiled_depth': [372, 1450, 5458, 23740, 93501, 381813],
        },
        'Full -- subset match': {
            'iterations': [1, 1, 2, 3, 4, 6],
            'orig_depth': [21, 38, 73, 144, 283, 576],
            'decomposed_depth': [283, 1124, 4848, 21098, 83845, 357780],
            'transpiled_depth': [351, 1487, 5454, 23955, 93214, 382436],
        },
    }

    # Set up plot
    plt.figure(figsize=(15, 7))
    width = 0.22
    x = np.arange(len(data['Full -- one match']['orig_depth']))

    colors = {
        'Original circuit depth': 'tab:blue',
        'Only decomposed depth': 'tab:orange',
        'Circuit depth after transpilation': 'tab:green'
    }

    # Offsets for grouped bars
    offsets = [-width, 0, width]
    experiment_labels = list(data.keys())

    for i, (exp_name, exp_data) in enumerate(data.items()):
        offset = (i - 1) * (width * 3 + 0.05)  # Center the groups
        # Bar positions for this experiment
        pos = x + offset

        plt.bar(pos + offsets[0], exp_data['orig_depth'],
                width=width, color=colors['Original circuit depth'], label='Original circuit depth' if i == 0 else "")
        plt.bar(pos + offsets[1], exp_data['decomposed_depth'],
                width=width, color=colors['Only decomposed depth'], label='Only decomposed depth' if i == 0 else "")
        plt.bar(pos + offsets[2], exp_data['transpiled_depth'],
                width=width, color=colors['Circuit depth after transpilation'], label='Circuit depth after transpilation' if i == 0 else "")

        # Annotate grover iterations
        for j, p in enumerate(pos):
            max_height = max(exp_data['orig_depth'][j], exp_data['decomposed_depth'][j], exp_data['transpiled_depth'][j])
            plt.text(p, max_height * 1.15, f"{exp_data['iterations'][j]}", ha='center', va='bottom', fontsize=10, fontweight='bold', color='black')

        # X tick labels
        if i == 1:  # Centered group: show the orig_depth as x-tick labels
            plt.xticks(pos, exp_data['orig_depth'])

    plt.yscale('log')
    plt.xlabel("Original circuit depth (problem size)")
    plt.ylabel("Circuit depth (log scale)")
    plt.title("Grover Circuit Depths across Experiments\n(Iterations annotated above each group)")
    plt.legend(loc='upper left')
    plt.grid(axis='y', which='both', linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.show()

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

def get_total_gates(gate_list):
    """Returns the sum of all gates for each entry in the experiment."""
    return [sum(entry.values()) for entry in gate_list]

def plot_grover_database_scaling():
    data = {
        'Full -- one match': {
            'iterations': [1, 2, 3, 4, 6, 8],
            'orig_depth': [21, 38, 73, 144, 283, 576],
            'decomposed_depth': [284, 1126, 4874, 20937, 83952, 356365],
            'transpiled_depth': [354, 1496, 5496, 23882, 93400, 382146],
            'gates': [
                {'cx': 289, 'rz': 155, 'rx': 68},
                {'cx': 1003, 'rz': 531, 'rx': 225},
                {'cx': 3560, 'rz': 2902, 'rx': 1533},
                {'cx': 16917, 'rz': 11814, 'rx': 4872},
                {'cx': 69713, 'rz': 48645, 'rx': 21115},
                {'cx': 291657, 'rz': 213647, 'rx': 99040},
            ]
        },
        'Full -- no match': {
            'iterations': [0, 0, 0, 0, 0, 0],
            'orig_depth': [21, 38, 73, 144, 283, 576],
            'decomposed_depth': [298, 1119, 4864, 20761, 83588, 358118],
            'transpiled_depth': [359, 1497, 5439, 23701, 93198, 382490],
            'gates': [
                {'cx': 276, 'rz': 164, 'rx': 79},
                {'cx': 1001, 'rz': 520, 'rx': 214},
                {'cx': 3564, 'rz': 2948, 'rx': 1578},
                {'cx': 16944, 'rz': 11757, 'rx': 4829},
                {'cx': 69798, 'rz': 48451, 'rx': 20890},
                {'cx': 291528, 'rz': 214746, 'rx': 100287},
            ]
        },
        'Full -- subset match': {
            'iterations': [1, 1, 2, 3, 4, 6],
            'orig_depth': [21, 38, 73, 144, 283, 576],
            'decomposed_depth': [296, 1112, 4874, 20725, 84358, 360447],
            'transpiled_depth': [359, 1501, 5442, 23806, 93284, 382696],
            'gates': [
                {'cx': 276, 'rz': 163, 'rx': 79},
                {'cx': 988, 'rz': 502, 'rx': 195},
                {'cx': 3555, 'rz': 2941, 'rx': 1583},
                {'cx': 16982, 'rz': 11656, 'rx': 4721},
                {'cx': 69681, 'rz': 49044, 'rx': 21562},
                {'cx': 290926, 'rz': 216861, 'rx': 102493},
            ]
        },
    }

    # Database size: 2^2, ..., 2^7
    x_exponents = np.arange(2, 8)
    database_sizes = 2 ** x_exponents
    x_labels = [f"$2^{exp}$" for exp in x_exponents]

    # Colours as requested
    colours = {
        "orig_depth": "red",
        "decomposed_depth": "orange",
        "transpiled_depth": "green",
        "nlogn": "blue",
    }
    markers = {
        "orig_depth": "o",
        "decomposed_depth": "s",
        "transpiled_depth": "^",
    }
    depth_labels = {
        "orig_depth": "Original circuit depth",
        "decomposed_depth": "Only decomposed depth",
        "transpiled_depth": "Circuit depth after transpilation",
        "nlogn": r"$n \log n$ (sum of gates)",
    }

    # Precompute all gate sums and n*log n for y-limits
    all_yvals = []
    nlogn_dict = {}
    for exp, exp_data in data.items():
        gate_sums = get_total_gates(exp_data['gates'])
        nlogn = [n * np.log(n) if n > 0 else 0 for n in gate_sums]
        nlogn_dict[exp] = nlogn
        all_yvals.extend(exp_data['orig_depth'])
        all_yvals.extend(exp_data['decomposed_depth'])
        all_yvals.extend(exp_data['transpiled_depth'])
        all_yvals.extend(nlogn)

    # Global y-limits (log scale)
    ymin = max(1, min(all_yvals))
    ymax = max(all_yvals) * 1.3  # a little headroom

    fig, axs = plt.subplots(1, 3, figsize=(18, 6.5), sharey=True)

    for ax, (exp_name, exp_data) in zip(axs, data.items()):
        x = database_sizes
        # Main lines
        for key in ['orig_depth', 'decomposed_depth', 'transpiled_depth']:
            ax.plot(
                x, exp_data[key],
                label=depth_labels[key],
                marker=markers[key],
                linestyle='-',
                color=colours[key],
                linewidth=2.7,
                markersize=11,
                alpha=0.97,
                markeredgecolor='white',
                markeredgewidth=1.7,
            )
        # n*log n line
        nlogn = nlogn_dict[exp_name]
        ax.plot(
            x, nlogn,
            label=depth_labels['nlogn'],
            color=colours['nlogn'],
            linestyle='--',
            linewidth=2.5,
            marker=None,
            alpha=0.99,
        )
        # Annotate Grover iterations on transpiled line, numbers in black
        for xi, yi, iters in zip(x, exp_data['transpiled_depth'], exp_data['iterations']):
            ax.annotate(
                str(iters), (xi, yi),
                textcoords="offset points", xytext=(0, 13), ha='center',
                fontsize=11, fontweight='bold', color='black',
                bbox=dict(boxstyle='round,pad=0.14', fc='white', ec='none', alpha=0.60)
            )
        ax.set_title(exp_name, fontsize=15, fontweight='bold')
        ax.set_xlabel("Database size $N = 2^x$", fontsize=13)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylim(ymin, ymax)
        ax.set_xticks(database_sizes)
        ax.set_xticklabels(x_labels)
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.grid(axis='y', which='both', linestyle=':', alpha=0.35)
    axs[0].set_ylabel("Circuit depth / $n\\log n$ (log scale)", fontsize=13)

    # Legend
    legend_elements = [
        Line2D([0], [0], color=colours['orig_depth'], marker=markers['orig_depth'], linestyle='-',
               markersize=11, label=depth_labels['orig_depth'], markeredgecolor='white', markeredgewidth=1.7),
        Line2D([0], [0], color=colours['decomposed_depth'], marker=markers['decomposed_depth'], linestyle='-',
               markersize=11, label=depth_labels['decomposed_depth'], markeredgecolor='white', markeredgewidth=1.7),
        Line2D([0], [0], color=colours['transpiled_depth'], marker=markers['transpiled_depth'], linestyle='-',
               markersize=11, label=depth_labels['transpiled_depth'], markeredgecolor='white', markeredgewidth=1.7),
        Line2D([0], [0], color=colours['nlogn'], linestyle='--', linewidth=3,
               label=depth_labels['nlogn']),
    ]
    fig.legend(
        handles=legend_elements,
        loc='lower center', ncol=4, fontsize=13, frameon=False,
        bbox_to_anchor=(0.52, -0.03)
    )

    plt.tight_layout(rect=[0, 0.04, 1, 0.97])
    plt.show()

    return nlogn_dict



import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
import os

def plot_qrs_with_db_scaling_from_files(name_of_plot="default.png"):
    base_dir = "src/brickwork_transpiler/experiments/data/output_data/"
    file_map = {
        "No db -- one match": "experiments_qrs_no_db_one_match.csv",
        "No db -- no match": "experiment_qrs_no_db_no_matching_element.csv",
        "No db -- subset match": "experiment_qrs_no_db_subset_grover.csv",
        "No db -- one match duplicates": "experiment_qrs_no_db_one_match_duplicates.csv",
    }

    # file_map = {
    #     "Full -- one match": "experiments_qrs_full_one_match.csv",
    #     "Full -- no match": "experiment_qrs_full_no_matching_element.csv",
    #     "Full -- subset match": "experiment_qrs_full_subset_grover.csv",
    #     "Full -- one match duplicates": "experiment_qrs_full_one_match_duplicates.csv",
    # }

    plot_exps = ["No db -- one match", "No db -- no match", "No db -- subset match", "No db -- one match duplicates"]
    data = {}

    # Read all data
    for exp in plot_exps:
        fname = os.path.join(base_dir, file_map[exp])
        df = pd.read_csv(fname)
        data[exp] = {
            "iterations": df['num_iterations'].to_numpy(),
            "orig_depth": df['original_depth'].to_numpy(),
            "decomposed_depth": df['decomposed_depth'].to_numpy(),
            "transpiled_depth": df['transpiled_depth'].to_numpy(),
            "num_gates_original": df['num_gates_original'].to_numpy(),
            "num_gates_transpiled": df['num_gates_transpiled'].to_numpy(),
        }

    max_len = max(len(d['orig_depth']) for d in data.values())
    x_exponents = np.arange(2, 2 + max_len)
    database_sizes = 2 ** x_exponents
    x_labels = [f"$2^{exp}$" for exp in x_exponents]

    colours = {
        "orig_depth": "red",
        "decomposed_depth": "orange",
        "transpiled_depth": "green",
        "nlogn_orig": "blue",  # now dark blue
    }
    markers = {
        "orig_depth": "o",
        "decomposed_depth": "s",
        "transpiled_depth": "^",
    }
    depth_labels = {
        "orig_depth": "Original circuit depth",
        "decomposed_depth": "Only decomposed depth",
        "transpiled_depth": "Circuit depth after transpilation",
        "nlogn_orig": r"$c \cdot n \log n$ (original gates)",  # c is explained below
    }

    # Set this constant to scale the nlogn line
    scaling_const = 1  # <-- CHANGE THIS VALUE as needed

    all_yvals = []
    nlogn_orig_dict = {}
    for exp, exp_data in data.items():
        n_gates_orig = exp_data["num_gates_original"]
        # Multiply by scaling constant here
        nlogn_orig = scaling_const * 396 * n_gates_orig * np.log(n_gates_orig) + 137 * n_gates_orig
        nlogn_orig_dict[exp] = nlogn_orig
        all_yvals.extend(exp_data["orig_depth"])
        all_yvals.extend(exp_data["decomposed_depth"])
        all_yvals.extend(exp_data["transpiled_depth"])
        all_yvals.extend(nlogn_orig)
    ymin = max(1, min(all_yvals))
    ymax = max(all_yvals) * 1.3

    # Use constrained_layout for robust legend placement
    fig, axs = plt.subplots(2, 2, figsize=(16, 13), sharey=True, constrained_layout=True)
    axs = axs.flatten()

    for idx, exp in enumerate(plot_exps):
        ax = axs[idx]
        exp_data = data[exp]
        x = database_sizes[:len(exp_data["orig_depth"])]
        for key in ['orig_depth', 'decomposed_depth', 'transpiled_depth']:
            ax.plot(
                x, exp_data[key],
                label=depth_labels[key],
                marker=markers[key],
                linestyle='-',
                color=colours[key],
                linewidth=2.7,
                markersize=11,
                alpha=0.97,
                markeredgecolor='white',
                markeredgewidth=1.7,
            )
        nlogn_orig = nlogn_orig_dict[exp]
        ax.plot(
            x, nlogn_orig,
            label=depth_labels['nlogn_orig'],
            color=colours['nlogn_orig'],
            linestyle='--',
            linewidth=2.4,
            alpha=0.9,
        )
        for xi, yi, iters in zip(x, exp_data['transpiled_depth'], exp_data['iterations']):
            ax.annotate(
                str(iters), (xi, yi),
                textcoords="offset points", xytext=(0, 13), ha='center',
                fontsize=11, fontweight='bold', color='black',
                bbox=dict(boxstyle='round,pad=0.14', fc='white', ec='none', alpha=0.60)
            )
        ax.set_title(exp, fontsize=15, fontweight='bold')
        ax.set_xlabel("Database size $N = 2^x$", fontsize=13)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylim(ymin, ymax)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels[:len(x)])
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.grid(axis='y', which='both', linestyle=':', alpha=0.35)
    axs[0].set_ylabel("Circuit depth / $n\\log n$ (log scale)", fontsize=13)
    axs[2].set_ylabel("Circuit depth / $n\\log n$ (log scale)", fontsize=13)

    # Place the legend and keep a handle to it
    legend_elements = [
        Line2D([0], [0], color=colours['orig_depth'], marker=markers['orig_depth'], linestyle='-',
               markersize=11, label=depth_labels['orig_depth'], markeredgecolor='white', markeredgewidth=1.7),
        Line2D([0], [0], color=colours['decomposed_depth'], marker=markers['decomposed_depth'], linestyle='-',
               markersize=11, label=depth_labels['decomposed_depth'], markeredgecolor='white', markeredgewidth=1.7),
        Line2D([0], [0], color=colours['transpiled_depth'], marker=markers['transpiled_depth'], linestyle='-',
               markersize=11, label=depth_labels['transpiled_depth'], markeredgecolor='white', markeredgewidth=1.7),
        # Only one nlogn line, now dark blue
        Line2D([0], [0], color=colours['nlogn_orig'], linestyle='--', linewidth=3,
               label=depth_labels['nlogn_orig']),
    ]
    fig.legend(
        handles=legend_elements,
        loc='lower center', ncol=4, fontsize=13, frameon=False,
        bbox_to_anchor=(0.52, -0.03)
    )

    plt.savefig(f"images/qrs/qrs_{name_of_plot}", dpi=300, bbox_inches='tight')
    plt.show()

# Example usage:
# plot_qrs_with_db_scaling_from_files(name_of_plot="your_custom_plotname.png")

# gates_sums now contains the sum of all gates per experiment

# def plot_grover_database_scaling():
#     data = {
#         'Full -- one match': {
#             'iterations': [1, 2, 3, 4, 6, 8],
#             'orig_depth': [21, 38, 73, 144, 283, 576],
#             'decomposed_depth': [298, 1094, 4885, 20916, 83629, 358550],
#             'transpiled_depth': [362, 1467, 5486, 23761, 93316, 382663],
#         },
#         'Full -- no match': {
#             'iterations': [0, 0, 0, 0, 0, 0],
#             'orig_depth': [21, 38, 73, 144, 283, 576],
#             'decomposed_depth': [294, 1079, 4869, 20431, 84129, 353961],
#             'transpiled_depth': [372, 1450, 5458, 23740, 93501, 381813],
#         },
#         'Full -- subset match': {
#             'iterations': [1, 1, 2, 3, 4, 6],
#             'orig_depth': [21, 38, 73, 144, 283, 576],
#             'decomposed_depth': [283, 1124, 4848, 21098, 83845, 357780],
#             'transpiled_depth': [351, 1487, 5454, 23955, 93214, 382436],
#         },
#     }
#
#     # Database size: 2^2, ..., 2^7
#     x_exponents = np.arange(2, 8)
#     database_sizes = 2 ** x_exponents
#     x_labels = [f"$2^{exp}$" for exp in x_exponents]
#
#     # Requested colors
#     colours = {
#         "orig_depth": "lightcoral",
#         "decomposed_depth": "lightsalmon",
#         "transpiled_depth": "lightgreen",
#         "reserved": "lightblue"  # Not used, as requested
#     }
#     markers = {
#         "orig_depth": "o",
#         "decomposed_depth": "s",
#         "transpiled_depth": "^",
#     }
#     depth_labels = {
#         "orig_depth": "Original circuit depth",
#         "decomposed_depth": "Only decomposed depth",
#         "transpiled_depth": "Circuit depth after transpilation",
#     }
#
#     fig, axs = plt.subplots(1, 3, figsize=(18, 6.5), sharey=True)
#
#     for ax, (exp_name, exp_data) in zip(axs, data.items()):
#         x = database_sizes
#         for key in ['orig_depth', 'decomposed_depth', 'transpiled_depth']:
#             ax.plot(
#                 x, exp_data[key],
#                 label=depth_labels[key],
#                 marker=markers[key],
#                 linestyle='-',
#                 color=colours[key],
#                 linewidth=2.7,
#                 markersize=11,
#                 alpha=0.97,
#                 markeredgecolor='white',
#                 markeredgewidth=1.7,
#             )
#         # Annotate Grover iterations on transpiled line, numbers in black
#         for xi, yi, iters in zip(x, exp_data['transpiled_depth'], exp_data['iterations']):
#             ax.annotate(
#                 str(iters), (xi, yi),
#                 textcoords="offset points", xytext=(0, 13), ha='center',
#                 fontsize=11, fontweight='bold', color='black',
#                 bbox=dict(boxstyle='round,pad=0.14', fc='white', ec='none', alpha=0.60)
#             )
#         ax.set_title(exp_name, fontsize=15, fontweight='bold')
#         ax.set_xlabel("Database size $N = 2^x$", fontsize=13)
#         ax.set_xscale('log')
#         ax.set_yscale('log')
#         ax.set_xticks(database_sizes)
#         ax.set_xticklabels(x_labels)
#         ax.tick_params(axis='both', which='major', labelsize=12)
#         ax.grid(axis='y', which='both', linestyle=':', alpha=0.35)
#     axs[0].set_ylabel("Circuit depth (log scale)", fontsize=13)
#
#     # Clean legend for the three line types
#     legend_elements = [
#         Line2D([0], [0], color=colours['orig_depth'], marker=markers['orig_depth'], linestyle='-',
#                markersize=11, label=depth_labels['orig_depth'], markeredgecolor='white', markeredgewidth=1.7),
#         Line2D([0], [0], color=colours['decomposed_depth'], marker=markers['decomposed_depth'], linestyle='-',
#                markersize=11, label=depth_labels['decomposed_depth'], markeredgecolor='white', markeredgewidth=1.7),
#         Line2D([0], [0], color=colours['transpiled_depth'], marker=markers['transpiled_depth'], linestyle='-',
#                markersize=11, label=depth_labels['transpiled_depth'], markeredgecolor='white', markeredgewidth=1.7),
#     ]
#     fig.legend(
#         handles=legend_elements,
#         loc='lower center', ncol=3, fontsize=13, frameon=False,
#         bbox_to_anchor=(0.52, -0.02)
#     )
#
#     plt.tight_layout(rect=[0, 0.03, 1, 0.97])
#     plt.show()
