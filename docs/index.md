# RADICAL Workflow Mini-Apps

## What a mini-app is

Workflow Mini-Apps are small, self-contained representations of scientific workflows.
Each mini-app captures the key tasks, data flow and performance characteristics of a
complex workflow without the deployment constraints of the full application.

## What this repository provides

- **`wfMiniAPI`**: emulation kernels (compute, I/O, communication, host/device data
  movement) that stand in for real application work inside workflow tasks, and
  `MiniApp`, which records each run and verifies its telemetry;
- **example workflows** built on [AsyncFlow](https://github.com/radical-cybertools/radical.asyncflow)
  and [RHAPSODY](https://github.com/radical-cybertools/rhapsody): `motif_2`, `motif_3`,
  `motif_4`, `motif_6` and `dragon-mpi`, plus standalone task scripts in `examples/tutorial/`.

The AI-steered simulation mini-app lives in
[DeepDriveSim](https://github.com/radical-collaboration/DeepDriveSim).

## Where to go next

| To | Read |
|---|---|
| install and run an example | [Getting started](getting-started.md) |
| use the emulation kernels in your tasks | [Kernels](kernels.md) |
| understand what a run records, and read it | [Telemetry](telemetry.md) |
| attach `MiniApp` to a workflow | [MiniApp](api/miniapp.md) |
| check that the workflows behave as before | [Equivalence harness](testing/equivalence.md), [Validation on an HPC node](testing/hpc-validation.md) |
