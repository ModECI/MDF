# Interactions between MDF and Quantum computing technologies

Starting summer 2021, we will develop tools for interfacing between MDF and quantum computers. This interface is motivated by expectations that quantum hardware will provide speedups for solving Ising-type MDF problems. We will address both gate- and annealing- based quantum computers:
* for gate-based quantum computers, we will bridge from MDF to [OpenQASM](https://github.com/Qiskit/openqasm), the leading quantum Intermediate Representation.
* for annealing-based quantum computers, we will target platforms such as [D-Wave Ocean](https://docs.ocean.dwavesys.com/en/stable/).

Our work will be agnostic to the exact quantum algorithm/solver used, though we will provide sample implementations using Variational Quantum Eigensolver ([VQE](https://www.nature.com/articles/ncomms5213)) and [Quantum Approximate Optimization Algorithm](https://arxiv.org/abs/1411.4028).
