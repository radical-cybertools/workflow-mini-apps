# RADICAL Workflow Mini-Apps

Workflow Mini-Apps are small, self-contained representations of scientific workflows.
Each mini-app captures the key tasks, data flow and performance characteristics of a
complex workflow without the deployment constraints of the full application.

This repository provides:

- **`wfMiniAPI`**: emulation kernels (compute, I/O, communication, host/device data
  movement) that stand in for real application work inside workflow tasks;
- **example workflows** built on [AsyncFlow](https://github.com/radical-cybertools/radical.asyncflow)
  and [RHAPSODY](https://github.com/radical-cybertools/rhapsody): `motif_2`, `motif_3`,
  `motif_4`, `motif_6` and `dragon-mpi`, plus standalone task scripts in `examples/tutorial/`.

The AI-steered simulation mini-app lives in
[DeepDriveSim](https://github.com/radical-collaboration/DeepDriveSim).

See [Getting started](getting-started.md) to install and run an example.
