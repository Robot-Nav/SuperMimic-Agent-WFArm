# Sources and acknowledgements

No robot model or third-party asset is vendored in this directory.

| Source | Relationship to this project |
|---|---|
| [SuperMimic](https://arxiv.org/abs/2610.28480) | Learned differentiable dynamics and short-horizon BPTT learning mechanism |
| [MimicAgent](https://arxiv.org/abs/2609.24145) | Programmatic reference generation and diagnostic refinement |
| [SuperMimic-Reproduced](https://github.com/CuboiLeo/SuperMimic-Reproduced) | Inherited independent implementation; MIT attribution retained |
| [PyTorch](https://pytorch.org/) | Neural networks and automatic differentiation |
| [MuJoCo](https://github.com/google-deepmind/mujoco) | Physics and native model inspection |
| [NumPy](https://numpy.org/), [SciPy](https://scipy.org/) | Numerical and data-processing tools |
| [mjlab](https://github.com/mujocolab/mjlab), [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp) | Optional inherited G1 simulation path |
| [OSQP](https://osqp.org/), [quadprog](https://github.com/quadprog/quadprog) | Optional offline constrained teachers |

See [NOTICE](../NOTICE.md) for author attribution and [LICENSE](../LICENSE) for
the software license. Dependencies and assets retain their own licenses.
