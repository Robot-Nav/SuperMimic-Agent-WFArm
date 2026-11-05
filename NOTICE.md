# Attribution

The project extends [SuperMimic-Reproduced](https://github.com/CuboiLeo/SuperMimic-Reproduced),
an independent implementation by Yuxi Liu / CuboiLeo. Its MIT copyright notice
is retained in [LICENSE](LICENSE). WF-Arm modules add wheel-legged manipulation,
fixed-base arm control, physical reference screening and a MimicAgent-style frontend.

The learning mechanism is based on **SuperMimic: Scaling Humanoid Whole-Body
Control with One Differentiable World Model**, Qiayuan Liao, Yiyang Shao,
Junfeng Long, Kevin Zakka, Roberto Horowitz, Pieter Abbeel and Koushil Sreenath,
[arXiv:2610.28480](https://arxiv.org/abs/2610.28480).

Reference synthesis is inspired by **MimicAgent: Quadruped Skills via
Prompt-to-Trajectory Generation**, Lucky Kant Nayak, Narayanan Palghat
Parameswaran, Neehar Peri and Deva Ramanan,
[arXiv:2609.24145](https://arxiv.org/abs/2609.24145).

The implementation uses [PyTorch](https://pytorch.org/),
[MuJoCo](https://github.com/google-deepmind/mujoco),
[NumPy](https://numpy.org/) and [SciPy](https://scipy.org/).
The inherited G1 path additionally uses [mjlab](https://github.com/mujocolab/mjlab)
and [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp).
Optional offline teachers use [OSQP](https://osqp.org/) and
[quadprog](https://github.com/quadprog/quadprog).

This repository is an independent adaptation. The original research teams are
not the maintainers of the WF-Arm extensions. Dependency licenses remain with
their respective projects. Private robot descriptions, meshes, calibration
and experimental data are not distributed under this repository's MIT license.
