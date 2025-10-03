from graphix import Pattern
from graphix.command import M, X, Z
import numpy as np
from copy import deepcopy
from typing import Dict, Tuple, List, Any, Optional


def encode_pattern(pattern: Pattern,
                theta_lookup: Optional[Dict[Tuple[int,int], float]] = None,
                rng: Optional[np.random.Generator] = None,
                remove_dependencies: bool = False):
    """
    Blind each measurement angle δ = φ + θ + r   (mod 2), all in units of π.
    """
    if rng is None:
        rng = np.random.default_rng()
    if theta_lookup is None:
        theta_lookup = {}

    new_pat = deepcopy(pattern)
    new_seq: List[Any] = []
    info_list: List[Dict[str, Any]] = []

    for cmd in new_pat._Pattern__seq:
        if isinstance(cmd, M):
            phi = cmd.angle  # treat as given (φ or already-adapted φ′)

            # choose θ if not supplied
            if cmd.node not in theta_lookup:
                theta_lookup[cmd.node] = rng.integers(0, 8) / 4.0  # {0, 0.25, ..., 1.75}
            theta = theta_lookup[cmd.node]

            r = int(rng.integers(0, 2))  # {0,1}

            # compute blinded angle δ in units of π (mod 2)
            delta = ( (phi % 2.0) + theta + r ) % 2.0

            # write blinded command with δ
            new_seq.append(M(cmd.node, cmd.plane, delta, cmd.s_domain, cmd.t_domain))

            # record metadata (include δ for convenience)
            info_list.append(dict(
                node=cmd.node,
                phi=phi,
                theta=theta,
                r=r,
                delta=delta,
                x_byproducts=set(),
                z_byproducts=set(),
                s_domain=set(cmd.s_domain),
                t_domain=set(cmd.t_domain),
            ))

        elif isinstance(cmd, X):
            if info_list:
                info_list[-1]["x_byproducts"].add(cmd.node)
            if not remove_dependencies:
                new_seq.append(cmd)

        elif isinstance(cmd, Z):
            if info_list:
                info_list[-1]["z_byproducts"].add(cmd.node)
            if not remove_dependencies:
                new_seq.append(cmd)

        else:
            new_seq.append(cmd)

    new_pat._Pattern__seq = new_seq
    return new_pat, info_list

