from graphix import Pattern
from graphix.command import M
import numpy as np
from copy import deepcopy

from graphix import Pattern
from graphix.command import M, X, Z
import numpy as np
from copy import deepcopy
from typing import Dict, Tuple, List, Any, Optional


def bfk_encoder(pattern: Pattern,
                theta_lookup: Optional[Dict[Tuple[int,int], float]] = None,
                rng: Optional[np.random.Generator] = None,
                remove_dependencies: bool = False):
    """
    Blind each measurement angle δ = φ + θ + r   (mod 2), all in units of π.

    We DO NOT evaluate feed-forward parities s,t. Instead, we copy the
    dependency sets (s_domain, t_domain) into the info list so you can
    reconstruct/adapt later if needed.

    Parameters
    ----------
    pattern : Pattern
        Original pattern containing Ms with (angle, s_domain, t_domain).
        `angle` is treated as the angle to be blinded (φ or φ', your choice).
    theta_lookup : dict[node, float] or None
        Optional pre-chosen θ values (units of π). If None, sample uniformly from {0, 1/4, ..., 7/4}.
    rng : np.random.Generator or None
        Random source. If None, uses default Generator.
    remove_dependencies: bool
        if true the return Pattern type does not contain X- and Z byproducts

    Returns
    -------
    new_pattern : Pattern
        Copy of the pattern with each M angle replaced by δ.
    info_list : list[dict]
        For each measured node:
            {
              'node': node,
              'phi': original_angle,
              'theta': θ,
              'r': r,
              'delta': δ,
              's_domain': set(...),
              't_domain': set(...)
            }
        (No s_par, t_par, phi_prime are computed here.)
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
            phi = cmd.angle  # interpret as "already adapted" or logical, as you choose

            # choose θ if not supplied
            if cmd.node not in theta_lookup:
                theta_lookup[cmd.node] = rng.integers(0, 8) / 4.0  # {0, 0.25, ..., 1.75}
            theta = theta_lookup[cmd.node]

            r = int(rng.integers(0, 2))           # {0,1}

            # write blinded command
            new_seq.append(M(cmd.node, cmd.plane, theta, cmd.s_domain, cmd.t_domain))

            # store everything needed for reconstruction/adaptation
            info_list.append(dict(node=cmd.node,
                                  phi=phi,
                                  theta=theta,
                                  r=r,
                                  x_byproducts=set(),
                                  z_byproducts=set(),
                                  s_domain=set(cmd.s_domain),
                                  t_domain=set(cmd.t_domain)))

        elif isinstance(cmd, X):
        # update last measurement’s x_byproducts
            if info_list:
                # print(cmd.node)
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
