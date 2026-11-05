"""Export a selected six-joint arm controller without simulator dependencies.

The serialized TorchScript contains the network, FK and kinematic constants.
Only PyTorch is needed to load it. Export checks use TRAIN samples exclusively.
"""
import argparse
import importlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from .data import load, Replay
from .io import atomic_json, sha256


class StatelessPolicy(nn.Module):
    """B=1 float32: state(q6,qd6), last executed target6, BASE goal13 -> target6."""
    def __init__(self, controller):
        super().__init__()
        self.task=controller.task
        self.actor=controller.learner.actor

    def forward(self,state,previous_target,goal_in_base):
        quat=goal_in_base[:,3:7]
        quat=quat/torch.linalg.vector_norm(quat,dim=-1,keepdim=True).clamp_min(1e-8)
        goal=torch.cat((goal_in_base[:,:3]+self.task.base_position,quat,goal_in_base[:,7:]),-1)
        return self.task.command(self.actor,state,previous_target,goal)


def controller_module(config):
    architecture=config.get("actor_architecture","mlp")
    if architecture=="structured_taskspace_supermimic":
        return "variants.supermimic_robust_control_v7.arm_smc.inference"
    if architecture=="frozen_v4_with_learned_goal_representation":
        return "variants.supermimic_goal_representation_v6.arm_smc.inference"
    if architecture=="frozen_v4_with_global_reach_residual":
        return "variants.supermimic_global_reach_v5.arm_smc.inference"
    if architecture=="frozen_base_with_local_residual":
        variant=("supermimic_kinematic_features_v4" if config.get("residual_use_kinematic_features",False)
                 else "supermimic_residual_v3")
        return f"variants.{variant}.arm_smc.inference"
    if architecture=="mlp":return "arm_smc.inference"
    raise ValueError(f"unknown actor architecture: {architecture}")


@torch.no_grad()
def export(checkpoint,output,references,samples=64):
    if samples<8:raise ValueError("at least eight independent TRAIN states required")
    torch.set_num_threads(2)
    checkpoint=Path(checkpoint).resolve();output=Path(output).resolve()
    manifest=json.loads((Path(references)/"manifest.json").read_text())
    if manifest["split"] not in ("train","additional_train"):
        raise ValueError("export verification accepts only TRAIN fixtures")
    for row in manifest["clips"]:
        if sha256(Path(references)/row["file"])!=row["sha256"]:
            raise RuntimeError("TRAIN fixture digest changed")
    ck=torch.load(checkpoint,map_location="cpu",weights_only=False)
    module=controller_module(ck["config"])
    controller=importlib.import_module(module).ArmController(checkpoint,device="cpu")
    policy=StatelessPolicy(controller).eval().requires_grad_(False)
    replay=Replay(load(references),"cpu",7317)
    batch=replay.starts(samples)
    goals=batch["goal"].clone();goals[:,:3]-=policy.task.base_position
    # Include both equivalent quaternion signs and non-unit input normalization.
    goals[::2,3:7]*=-2.
    arguments=(batch["state"][:1],batch["previous"][:1],goals[:1])
    traced=torch.jit.freeze(torch.jit.trace(policy,arguments,strict=True).eval())
    output.parent.mkdir(parents=True,exist_ok=True)
    temporary=output.with_suffix(output.suffix+".tmp")
    torch.jit.save(traced,str(temporary))
    reloaded=torch.jit.load(str(temporary)).eval()
    maximum_error=0.;maximum_delta=0.
    for i in range(samples):
        args=(batch["state"][i:i+1],batch["previous"][i:i+1],goals[i:i+1])
        reference=policy(*args);actual=reloaded(*args)
        torch.testing.assert_close(actual,reference,atol=1e-6,rtol=1e-6)
        assert actual.shape==(1,6) and bool(torch.isfinite(actual).all())
        assert bool((actual>=policy.task.low-1e-6).all() and (actual<=policy.task.high+1e-6).all())
        delta=(actual-args[1]).abs().max().item()
        assert delta<=policy.task.dt*policy.task.speed+1e-6
        maximum_error=max(maximum_error,(actual-reference).abs().max().item())
        maximum_delta=max(maximum_delta,delta)
    for _ in range(20):reloaded(*arguments)
    timings=[]
    for _ in range(100):
        start=time.perf_counter();reloaded(*arguments);timings.append(time.perf_counter()-start)
    temporary.replace(output)
    result=dict(checkpoint=str(checkpoint),checkpoint_sha256=sha256(checkpoint),
        artifact=str(output),artifact_sha256=sha256(output),actor_architecture=ck["config"].get("actor_architecture","mlp"),
        python_controller=module+".ArmController",runtime="PyTorch TorchScript CPU; no MuJoCo/assets required to load",
        dtype="float32",batch=1,frequency_hz=1/policy.task.dt,
        inputs=dict(state="[1,12]: measured q6(rad), qd6(rad/s)",
                    previous_target="[1,6]: last actually executed joint target(rad)",
                    goal_in_base="[1,13]: xyz(m), quaternion_wxyz, linear(m/s), angular(rad/s); BASE frame"),
        output="[1,6]: absolute joint position targets(rad)",joint_names=list(controller.joint_names),
        target_speed_limit_rad_s=policy.task.speed,maximum_checked_target_delta_rad=maximum_delta,
        train_verification_samples=samples,maximum_export_error_rad=maximum_error,
        cpu_median_inference_ms=float(np.median(timings)*1000),cpu_p95_inference_ms=float(np.quantile(timings,.95)*1000),
        valid_quaternion_required=True,previous_target_required=True,
        operation="Chassis stationary. Visual detection, camera->base transform, goal selection and gripper commands are separate.",
        hardware_qualified=False,quality="See independent native-physics evaluation; export equivalence alone is not task qualification")
    if ck["config"].get("actor_architecture")=="structured_taskspace_supermimic":
        result.update(policy_role="Six-joint continuous TCP tracker; large reaches use the Python ArmController planner",
            reaching_interface="ArmController.command(...,mode='reach');planner generates internal reference",
            tracking_interface="This TorchScript forward or ArmController.command(...,mode='tracking')",
            feedback_damping=ck["config"].get("feedback_damping",.05),
            independent_velocity_gains=ck["config"].get("independent_velocity_gains",False),
            joint2_command_upper_rad=ck["config"]["joint2_command_upper"],
            joint2_braking_lookahead_s=ck["config"]["joint2_braking_lookahead"])
    atomic_json(output.with_suffix(".json"),result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint",required=True);p.add_argument("--output",required=True)
    p.add_argument("--references",required=True);p.add_argument("--samples",type=int,default=64)
    a=p.parse_args();print(json.dumps(export(a.checkpoint,a.output,a.references,a.samples),indent=2))


if __name__=="__main__":main()
