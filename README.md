# SuperMimic-Agent-WFArm

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![MuJoCo](https://img.shields.io/badge/Simulation-MuJoCo-009688)](https://mujoco.org/)
[![Status](https://img.shields.io/badge/Status-Exploratory-orange)](docs/PROJECT_ANALYSIS_CN.md)

**English · [简体中文](README_CN.md)**

An exploratory open-source project on **wheel-legged mobile manipulation**. It studies how a four-wheel, four-leg platform and a six-joint arm can share a learned dynamics model, how a fixed-base arm can learn pose reaching and trajectory tracking, and how generated references can support that learning. Discussion, experiments and contributions are welcome.

The main learning mechanism comes from [SuperMimic](https://arxiv.org/abs/2610.28480): fit a differentiable world model to real simulator transitions, then optimize the policy through short imagined rollouts. A [MimicAgent](https://arxiv.org/abs/2609.24145)-style frontend proposes reference motions and checks them before they enter training.

This repository distributes training and evaluation code. Robot descriptions, meshes, calibration, datasets, checkpoints, videos and internal experiment reports are supplied separately or kept private. Full WF-Arm simulation requires authorized robot assets. The asset-free tests run without them.

---

**png**：<img width="1440" height="540" alt="fixed_arm_showcase_contact_sheet" src="https://github.com/user-attachments/assets/fec88e0f-fa53-402b-a322-af3309023700" />


**video**：


**Process**:  
In this project, the teacher is the expert demonstration data generator, the student is the control policy being trained, and the world model is responsible for learning the dynamics. The three are connected through the following process:  

1. **Collect Data: Expert demonstrations and policy tries**  
The MimicAgent trajectory program proposes the desired motion, and the demonstration controller executes it in MuJoCo via motor commands. The actual "state, action, next state" that passes the physical filter becomes the expert demonstration. The current policy also runs in MuJoCo to collect data from its own executions.  

2. **Train the World Model: Learn the results of actions**  
The world model uses real state transitions to learn "current state + action → next state," for example, how a joint position command changes the angles and velocities of a robotic arm. It’s responsible for predicting motion outcomes.  

3. **Train the Control Policy: Learn what actions to output**  
The student policy outputs actions based on the current state and targets. The world model predicts the resulting motion, and task rewards like end-point errors are used to backpropagate gradients through the world model to update the policy parameters. The world model parameters are temporarily fixed while updating the policy. Expert demonstration actions can also provide imitation guidance to help the policy learn.  

4. **Test in MuJoCo: Check if the policy actually works**  
The trained policy is used to control the robotic arm in practice, checking if reaching and tracking are successful, errors, collisions, or boundary violations. Actions that the expert can perform must be independently completed by the student to be considered learned. Accurate world model predictions still need actual control validation. Final independent test data is kept separate and not used for training.



---

## What the name means

- **SuperMimic**: the learned-world-model and backpropagation-through-time learning mechanism.
- **Agent**: a reference-generation interface that keeps the task, phase plan, program and diagnostic feedback together. It currently accepts programs from a coding agent through files; it does not run an unattended external LLM service.
- **WFArm**: the wheel-legged platform and its arm. Both whole-body coordination and a fixed-base arm task are included.

## Project scope

| Component | Control interface | Main source |
|---|---|---|
| Whole-body manipulation | One Actor produces 12 leg position commands, 4 wheel velocity commands and 6 arm position commands | `src/supermimic/wf/` |
| Fixed-base arm baseline | Six rate-limited joint position targets; TCP pose/twist goals in the base frame | `frameworks/fixed_base_arm/arm_smc/` |
| Structured arm controller | Collision-checked reaching references, learned task feedback gains and a retained posture policy | `frameworks/fixed_base_arm/variants/supermimic_robust_control_v7/arm_smc/` |
| Reference frontend | Task/plan/program, forward-kinematics checks, diagnostic feedback, optional physical screening | `src/supermimic/mimicagent/` |
| Evaluation | Held terminal poses, continuous tracking, failure checks and reference split validation | The corresponding task and evaluation modules |

The whole-body policy and the fixed-base arm are separate control settings. The structured arm implementation uses planning and analytic Jacobian feedback in addition to learning. Its behavior should not be described as a pure neural controller. The repository does not establish full-workspace or hardware qualification.

The inherited G1 motion-tracking code is kept as a compact research foundation for the common dynamics, FK and BPTT machinery. G1 tracking is not the focus of the WF-Arm work.

## MimicAgent × SuperMimic

The two methods meet at the reference data interface:

1. **MimicAgent-style generation** turns a task and phase plan into a trajectory program, checks the proposal and passes diagnostic feedback to revisions.
2. **Physical execution and screening** run the motor commands in MuJoCo. Only independently accepted real state/action sequences become demonstration data.
3. **SuperMimic-style learning** uses those references for model fitting, optional BC and imagined state-tracking guidance, together with transitions collected by the current student policy.
4. **Real-physics evaluation** checks the learned student separately from the reference generator and its offline teacher.

The frontend supplies useful proposals; the learned world model and BPTT train the controller. This is the integration implemented here. It does not establish the original papers' benchmark results or show that reference generation has already improved the final student.

## Learning mechanism

Let $s_t$ be the simulator state, $a_t$ the executed command and $\xi$ the environment context. The learned transition is

$$
\hat{s}_{t+1}=F_\theta(\hat{s}_t,a_t,\xi).
$$

The model predicts bounded residuals around explicit state integration. It is trained autoregressively over $K$ steps:

$$
\mathcal{L}_{\mathrm{WM}}
=\frac{1}{K}\sum_{k=1}^{K}
\left(\frac1m\left\|E(\hat{s}_{t+k},s_{t+k})\right\|_2^2
+\beta\,\mathrm{BCE}(\hat d_{t+k},d_{t+k})\right).
$$

$E$ is a scaled state error and $\hat d$ is a failure estimate. The WF whole-body implementation uses $\beta=0.1$. Analytic forward kinematics supplies differentiable TCP observations and task rewards.

During an Actor update, world-model and Critic parameters are frozen while gradients through their inputs remain available:

$$
J(\phi)=\mathbb{E}\!\left[
\sum_{t=0}^{H-1}\gamma^t C_t r_{t+1}
+\gamma^H C_H V_\psi(\hat s_H,g_H,\xi)
\right],\qquad
C_t=\prod_{j=1}^{t}(1-\hat d_j).
$$

The policy maximizes $J$ through BPTT; optional BC and reference-tracking losses provide guidance. The Critic uses a detached $\lambda$-return target. Failure probabilities limit imagined continuation, but they do not replace actual collision and limit checks.

The six-axis structured branch additionally learns bounded positive task feedback gains. A damped Jacobian maps the task-space correction to joint motion, while a posture or planned-reference term fills the remaining directions. Its complete equations and the differences between branches are explained in the [Chinese project analysis](docs/PROJECT_ANALYSIS_CN.md).

## Repository layout

```text
src/supermimic/
  wf/                     Whole-body simulation, model, policy and training
  mimicagent/             Reference proposal and physical-screening frontend
  world_model/            Shared learned-dynamics utilities
  imagination/            Imagined rollouts and Actor/Critic objectives
  kinematics/             Differentiable kinematics
frameworks/fixed_base_arm/
  arm_smc/                Six-axis baseline training and inference
  variants/supermimic_robust_control_v7/arm_smc/
                          Structured arm training and planning
configs/examples/         Portable examples without private robot parameters
examples/mimicagent/       Small trajectory programs and a phase plan
scripts/                  Selected training/data/evaluation entry points
tests/                    Release tests
third_party/              Attribution, not vendored robot assets
tools/                    Release preparation and content audit
docs/                     Project analysis, asset setup and repository metadata
```

The research workspace may contain additional local history. It is not part of the GitHub release.

## Installation

The WF path uses Python 3.11 or 3.12. The local CPU reference environment was Python 3.11, PyTorch 2.7, MuJoCo 3.10 and NumPy 1.26. A CUDA-enabled PyTorch installation accelerates learning; the WF reference simulator itself runs on the CPU.

```bash
git clone https://github.com/Robot-Nav/SuperMimic-Agent-WFArm.git
cd SuperMimic-Agent-WFArm
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

| Dependency | Role | Installation |
|---|---|---|
| PyTorch | Dynamics, policies, autograd and optimization | Base |
| MuJoCo | Motor-driven physics, contacts and model inspection | Base |
| NumPy, SciPy | Data processing, numerical routines and rotations | Base |
| pytest | Tests | `.[dev]` |
| OSQP, quadprog | Optional offline QP teachers | `.[teacher]` |
| imageio, imageio-ffmpeg, matplotlib | Optional recording and plots | `.[media]` |
| mjlab, MuJoCo Warp, Warp and G1 data tools | Inherited G1 path | `.[g1]`, Python 3.12 in a separate environment |

Install a PyTorch wheel matching your GPU and driver before using CUDA. The G1 extra has its own pinned mjlab revision and newer simulation stack; it is not required for WF training.

## Running

### 1. Run the asset-free tests

```bash
python -m pytest -q tests/test_release_core.py
```

These check state integration, Actor/Critic objectives, gradients, asset lookup and release boundaries. They do not measure robot skill.

### 2. Connect authorized private assets

```bash
export WFARM_ASSET_DIR=/path/to/authorized/wf-arm-tbot
python tools/check_assets.py
```

The directory contains the private URDF, its referenced collision meshes, joint metadata and TCP calibration. The fixed-base task also reads private operating limits. See [asset setup](docs/PRIVATE_ASSETS.md). None of these files should be added to Git.

### 3. Check the whole-body training loop

```bash
python -m supermimic.wf.train --smoke --device cpu --output outputs/wf_smoke
```

For a longer experiment, copy `configs/examples/wf_train.json`, select a fresh output directory and set the training budget:

```bash
python -m supermimic.wf.train --config configs/examples/wf_train.json
```

The example is a starting configuration, not a published trained policy. Use `--resume` for a full checkpoint continuation. `--warm-start` transfers a different subset of state and has different training semantics.

### 4. Check the six-axis baseline

Run from the repository root:

```bash
PYTHONPATH=src:frameworks/fixed_base_arm python -m arm_smc.train \
  --config configs/examples/fixed_arm_smoke.json
```

This generates small disjoint reference sets and exercises learning and evaluation. Increase data, model and optimization budgets before studying performance. The structured branch retains earlier policy components and may require locally trained source checkpoints; its code is not a replacement for those training stages.

### 5. Generate and screen references

```bash
python scripts/wf_mimicagent_pilot.py \
  --output outputs/mimicagent_reference --terminal-feedback

python scripts/wf_mimicagent_smoke.py \
  --library outputs/mimicagent_reference/physical_library \
  --output outputs/mimicagent_training_check
```

The optional terminal feedback uses an existing offline teacher. Passing kinematic checks alone does not authorize a proposal as a real dynamics transition. The exporter admits only successful independently executed rollouts, verifies their hashes and keeps request-level training/validation separation.

## Evaluation and limitations

Report task success and physical failure separately. Reaching requires the terminal pose and velocity tolerances to hold over consecutive final frames. Tracking requires a sufficient fraction of the requested trajectory to meet the tolerances, including the denominator for frames lost after a failure. Keep test requests outside training, BC, world-model fitting and checkpoint selection.

Low world-model MSE, a completed training budget, successful reference trajectories and attractive animations answer different questions. None alone demonstrates a capable deployed policy. Current code includes local action-response audits, real-physics policy checks and replay-boundary handling to inspect these gaps. Experiments under 0.5–1.0 m/s whole-body motion need their own evaluation.

## License, references and discussion

### Acknowledgements

- We thank the **SuperMimic authors** for the differentiable-world-model and BPTT learning framework.
- We thank the **MimicAgent authors** for the prompt-to-trajectory and diagnostic-refinement approach.
- We thank **Yuxi Liu / CuboiLeo** for the inherited SuperMimic-Reproduced implementation, and the maintainers of the simulation and learning libraries used here.

Full paper titles, authors and project links are recorded in [NOTICE](NOTICE.md).

Code is licensed under [MIT](LICENSE). Original copyright and attribution are retained; see [NOTICE](NOTICE.md) and [third-party references](third_party/README.md). Robot assets and local experimental products are not included in the software license.

- [SuperMimic paper](https://arxiv.org/abs/2610.28480)
- [MimicAgent paper](https://arxiv.org/abs/2609.24145)
- [SuperMimic-Reproduced](https://github.com/CuboiLeo/SuperMimic-Reproduced)

Use [Issues](https://github.com/Robot-Nav/SuperMimic-Agent-WFArm/issues) to discuss training behavior, reproducibility and alternative control designs. When reporting a problem, include the code revision, software versions, configuration and a concise failure description. Keep private robot files and credentials out of reports.

[简体中文](README_CN.md) · [中文项目分析](docs/PROJECT_ANALYSIS_CN.md)
