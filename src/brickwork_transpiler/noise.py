from graphix.channels import depolarising_channel, dephasing_channel, KrausChannel

from src.brickwork_transpiler import pattern_converter
import sys
sys.path.append('/Users/rexfleur/Documents/TUDelft/Master_CESE/Thesis/Code/gospel')  # Full path to the cloned repo
from gospel.brickwork_state_transpiler import generate_random_pauli_pattern


def to_noisy_pattern(bw_pattern, p_depol, p_dephase, seed=None):

    # Build channels
    kc1 = depolarising_channel(p_depol)
    kc2 = dephasing_channel(p_dephase)
    # Inject after each input node
    for q in bw_pattern.input_nodes:
        bw_pattern.add(KrausChannel(kc1.kraus_ops, [q]))
        bw_pattern.add(KrausChannel(kc2.kraus_ops, [q]))

    return bw_pattern

