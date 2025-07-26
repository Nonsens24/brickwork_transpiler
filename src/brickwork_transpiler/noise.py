import numpy as np
from copy import deepcopy
from graphix.channels import depolarising_channel, two_qubit_depolarising_channel
from graphix.pattern import Pattern
from graphix.command import M, E

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


