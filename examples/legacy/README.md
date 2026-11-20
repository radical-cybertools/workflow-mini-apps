# Legacy examples

These mini-apps target the old RADICAL-Pilot / EnTK API and are **not maintained**.
They are kept for reference only.

- `exalearn/`: Neutron Diffraction (InverseProblem). Its `Executables/` import the
  kernels through a symlink that no longer resolves (`kernel.py`, and a `kernal` typo).
  To run them, replace those imports with `import wfMiniAPI.kernel`.
