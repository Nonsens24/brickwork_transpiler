###Brickwork graph transpiler

Takes a unitary circuit and transpiles it into a brickwork graph state. The program builds the brickwork
graph by combining bicks (i.e., arbitrary single qubit rotations and CX gates). Graphix (https://github.com/TeamGraphix/graphix)
graph generator is mainly used to calculate the Z/X correction dependencies.

The generated brickwork graph will be used for simulating UBQC (https://arxiv.org/abs/0807.4154).