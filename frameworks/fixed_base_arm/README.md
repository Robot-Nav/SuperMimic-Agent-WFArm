# Fixed-base six-axis arm

[English overview](../../README.md) · [中文说明](../../README_CN.md) · [算法分析](../../docs/PROJECT_ANALYSIS_CN.md)

The baseline fixes the chassis, legs, wheels and gripper in an in-memory copy
of the privately supplied whole-robot model. Only six arm joints remain active.
The task covers TCP pose reaching and continuous pose/twist tracking.

## Baseline

`arm_smc/` contains the simulator adapter, differentiable FK, learned residual
dynamics, deterministic Actor, lambda-return Critic, reference generation,
online collection, evaluation and checkpoint handling.

Inputs use TCP goals in the chassis base frame. Quaternion order is wxyz.
The controller returns six absolute joint position targets in radians; internally
it limits their change per control interval. Gripper and base commands are outside
this six-axis interface.

```bash
# Run from the repository root with authorized assets configured.
export WFARM_ASSET_DIR=/path/to/authorized/wf-arm-tbot
python tools/check_assets.py --fixed-arm
PYTHONPATH=src:frameworks/fixed_base_arm python -m arm_smc.train \
  --config configs/examples/fixed_arm_smoke.json
```

The example first creates small independent reference splits, then exercises
model fitting, BC, online BPTT and evaluation. Outputs stay under this framework's
`runs/`; generated data and checkpoints are excluded from the source release.

## Structured branch

`variants/supermimic_robust_control_v7/arm_smc/` contains the structured policy,
collision-checked reaching reference planner and matching evaluation path.
It combines a retained posture policy, bounded learned task gains and damped
Jacobian feedback. It does not replace the physical model with an analytic
controller or treat its geometric checks as student results.

This branch preserves earlier policy components and can require locally trained
source checkpoints and goal encoders. Those weights and historical run recipes
are not included. The baseline example does not reproduce the complete structured
training sequence. See the analysis for the equations and the distinction between
planning, learning and execution.

## Assets and evaluation

The adapter reads geometry, TCP calibration and operating limits from separately
supplied resources. See [private asset setup](../../docs/PRIVATE_ASSETS.md).

Reaching and tracking have separate criteria. Final tests remain outside model
fitting, BC and checkpoint selection. Fixed-base simulation does not establish
free-base stability, full-workspace coverage or hardware transfer.
