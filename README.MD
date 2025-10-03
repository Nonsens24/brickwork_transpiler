## Brickwork graph transpiler

Takes a unitary circuit and transpiles it into a brickwork graph state. The program builds the brickwork
graph by combining bicks (i.e., arbitrary single qubit rotations and CX gates). Graphix (https://github.com/TeamGraphix/graphix)
graph generator is mainly used to calculate the Z/X correction dependencies.

The generated brickwork graph will be used for simulating UBQC (https://arxiv.org/abs/0807.4154).



# Brickwork Transpiler

[![Build](https://img.shields.io/github/actions/workflow/status/<org>/<repo>/ci.yml?branch=main)](https://github.com/<org>/<repo>/actions)
[![Coverage](https://img.shields.io/codecov/c/github/<org>/<repo>)](https://app.codecov.io/gh/<org>/<repo>)
[![PyPI](https://img.shields.io/pypi/v/brickwork-transpiler.svg)](https://pypi.org/project/brickwork-transpiler/)
[![Python](https://img.shields.io/pypi/pyversions/brickwork-transpiler.svg)](https://pypi.org/project/brickwork-transpiler/)
[![License](https://img.shields.io/github/license/<org>/<repo>.svg)](./LICENSE)
[![Docs](https://img.shields.io/badge/docs-latest-blue.svg)](https://<org>.github.io/<repo>/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://black.readthedocs.io/)
[![mypy](https://img.shields.io/badge/mypy-checked-informational)](http://mypy-lang.org/)
[![Tests](https://img.shields.io/badge/tests-pytest-brightgreen)](https://docs.pytest.org/)

A layout-aware transpiler that converts **Qiskit circuits** into **brickwork (MBQC)** patterns and back, with provable statevector equivalence even under **SABRE** layouts, routing swaps, and ancilla management. Includes utilities for layout extraction, ancilla padding, reference-state generation, and robust comparisons up to global phase.

---

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [Transpile](#transpile)
  - [Padding inputs when the register widens](#padding-inputs-when-the-register-widens)
  - [Extract and undo layout](#extract-and-undo-layout)
  - [Reference state (layout-aware)](#reference-state-layout-aware)
- [Testing](#testing)
- [Design Notes](#design-notes)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## Features

- **Qiskit → Brickwork** translation that emits an MBQC pattern plus an optional color map for visualization.
- **Layout awareness:** uses `final_index_layout()` when available, or reconstructs from `initial_layout` and `routing_permutation()` (e.g., SABRE).
- **Ancilla handling:** automatic padding with `|+⟩` on higher indices to match widened registers after transpilation.
- **Reference evaluation:** computes layout-aware reference states and matches MBQC simulator output after reordering and partial trace.
- **Permutation utilities:** safe construction of `PermutationGate` patterns using the `perm[src]=dest` convention.
- **Robust tests:** helpers for global-phase equivalence, mixed-state detection, and layout synchronization.

---

## Installation

### From PyPI
```bash
pip install brickwork-transpiler
```
