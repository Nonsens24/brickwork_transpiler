import numpy as np
from copy import deepcopy
from graphix.channels import depolarising_channel, two_qubit_depolarising_channel
from graphix.pattern import Pattern
from graphix.command import M, E

from graphix.channels import depolarising_channel, two_qubit_depolarising_channel
from graphix.noise_models.noise_model import NoiseModel


class SimulationNoiseModel(NoiseModel):
    def __init__(self, p1, p2):
        super().__init__()
        self.p1 = p1  # single-qubit error prob
        self.p2 = p2  # two-qubit error prob

    def prepare_qubit(self):
        return depolarising_channel(self.p1)

    def measure(self):
        return depolarising_channel(self.p1)

    def byproduct_x(self):
        return depolarising_channel(self.p1)

    def byproduct_z(self):
        return depolarising_channel(self.p1)

    def clifford(self):
        return depolarising_channel(self.p1)

    def entangle(self):
        # CZ or other two-qubit gate
        return two_qubit_depolarising_channel(self.p2)

    def tick_clock(self):
        # Optional: apply idle decoherence every timestep
        return depolarising_channel(self.p1)

    def confuse_result(self, cmd):
        return cmd

class DepolarisingInjector:
    def __init__(self, single_prob: float, two_prob: float):
        self.single_prob = single_prob
        self.two_prob = two_prob
        self.single_chan = depolarising_channel(prob=single_prob)
        self.two_chan = two_qubit_depolarising_channel(prob=two_prob)

    def inject(self, pattern: Pattern) -> Pattern:
        noisy_pattern = deepcopy(pattern)
        new_seq = []

        for cmd in noisy_pattern._Pattern__seq:
            if isinstance(cmd, M):
                # Depolarising noise on measurement angles (simple model)
                if np.random.rand() < self.single_prob:
                    # With depolarizing noise, measurement angle randomly flips (X,Y-plane).
                    noisy_angle = cmd.angle + np.random.choice([0, 1])
                    # print("CMD ANGLE", cmd.angle)
                else:
                    noisy_angle = cmd.angle
                new_seq.append(M(cmd.node, cmd.plane, noisy_angle,
                                 cmd.s_domain, cmd.t_domain))

            elif isinstance(cmd, E):
                # Depolarizing noise for entanglement probabilistically removes edges
                if np.random.rand() >= self.two_prob:
                    new_seq.append(cmd)
                # else edge is dropped due to depolarizing noise

            else:
                new_seq.append(cmd)

        noisy_pattern._Pattern__seq = new_seq
        return noisy_pattern


