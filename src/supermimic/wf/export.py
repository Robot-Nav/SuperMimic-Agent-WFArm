"""Export the unified Actor and its analytic observation transform to TorchScript.

The exported interface takes canonical state, last executed action and world TCP
goal. It contains no world model, reference generator, IK or simulator stepping.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch import nn

from .evaluate import config_from_dict
from .networks import Actor
from .simulator import BatchedSimulator
from .task import WholeBodyTask
from .train import model_manifest


class InferencePolicy(nn.Module):
    def __init__(self, actor, task):
        super().__init__()
        self.actor, self.task = actor, task

    def forward(self, state, previous_action, world_goal):
        quaternion = world_goal[..., 3:7]
        goal = torch.cat((world_goal[..., :3], quaternion/quaternion.norm(dim=-1, keepdim=True).clamp_min(1e-8),world_goal[...,7:]), -1)
        return self.task.process_action(self.actor(self.task.actor_input(state, previous_action, goal)))


def export_checkpoint(checkpoint, output):
    checkpoint, output = Path(checkpoint), Path(output)
    metadata_path = output.with_suffix(output.suffix+'.json')
    if output.exists() or metadata_path.exists():
        raise FileExistsError('choose a new export path')
    torch.set_num_threads(2)
    ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
    cfg = config_from_dict(ck['config'])
    sim = BatchedSimulator(num_envs=1, asset_dir=cfg.asset_dir, dt=cfg.sim.dt,
        timestep=cfg.sim.timestep, friction=1., leg_action_scale=cfg.sim.leg_action_scale,
        arm_action_scale=cfg.sim.arm_action_scale, wheel_action_scale=cfg.sim.wheel_action_scale,
        arm_initial_position=cfg.sim.arm_initial_position)
    try:
        if model_manifest(sim.spec)['physics_hash'] != ck['manifest']['physics_hash']:
            raise ValueError('checkpoint physics differs from the current robot')
        task = WholeBodyTask(sim.model, sim.spec, reward_cfg=cfg.reward)
        actor = Actor(cfg.policy); actor.load_state_dict(ck['actor'])
        policy = InferencePolicy(actor, task).eval()
        sim.settle(steps=cfg.sim.reset_settle_steps)
        state = torch.tensor(sim.read_state(), dtype=torch.float32)
        goal = task.initialize_goal(state,task.tcp_pose(state).detach()); goal[:, 0] += .1
        previous = torch.tensor(sim.applied_action,dtype=torch.float32)
        # Fixed robot topology is deliberately traced; the batch axis is dynamic.
        with torch.no_grad():
            traced = torch.jit.trace(policy, (state, previous, goal), check_trace=False)
            max_error = 0.
            rng = torch.Generator().manual_seed(91357)
            for count in (1, 8):
                batch_state = state.expand(count, -1).clone()
                batch_state[:, 7:29] += .01*torch.randn(count, 22, generator=rng)
                batch_state[:, 29:] += .01*torch.randn(count, 28, generator=rng)
                batch_previous = (previous.expand(count,-1)+.02*torch.randn(count, 22, generator=rng)).clamp(-1.,1.)
                batch_goal = goal.expand(count, -1).clone()
                batch_goal[:, :3] += .03*torch.randn(count, 3, generator=rng)
                expected = policy(batch_state, batch_previous, batch_goal)
                actual = traced(batch_state, batch_previous, batch_goal)
                torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
                max_error = max(max_error, float((actual-expected).abs().max()))
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.jit.save(traced, str(output))
        metadata = dict(format='wf_unified_torchscript_v1', checkpoint=str(checkpoint.resolve()),
            checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            export_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
            physics_hash=ck['manifest']['physics_hash'], control_period_s=cfg.sim.dt,
            inputs={'state': '[N,57]: base xyz world(3), quaternion wxyz(4), joints(22), linear velocity world(3), angular velocity body(3), joint velocity(22)',
                    'previous_action': '[N,22]: last executed normalized command; initialize from reset_action after settling',
                    'world_goal': '[N,7]: TCP xyz world, quaternion wxyz'},
            output='[N,22] executable normalized commands; feed the actual executed commands back at the next step',
            joint_order=list(sim.spec.joint_names), q0=sim.spec.q0.tolist(),
            reset_joint_position=sim.initial_q.tolist(),reset_action=sim.initial_action[0].tolist(),
            reset_settle_steps=cfg.sim.reset_settle_steps,goal_size=task.command_size,
            goal_follow_frame=cfg.reward.goal_follow_frame,
            action_scale=sim.spec.action_scale.tolist(), kp=sim.spec.kp.tolist(), kd=sim.spec.kd.tolist(),
            effort_limits=sim.spec.effort.tolist(),
            operating_action_lower=list(cfg.policy.action_lower),operating_action_upper=list(cfg.policy.action_upper),
            operating_bound_source=cfg.policy.action_bound_source,operating_bound_sha256=cfg.policy.action_bound_sha256,
            actuator_semantics='indices 0:12 and 16:22 position targets q0+scale*action; indices 12:16 wheel velocity targets scale*action',
            required_inputs='finite float32 tensors, nonzero base/goal quaternions, canonical joint order',
            parity_batch_sizes=[1, 8], parity_max_abs_error=max_error,
            scope='simulation policy; source TCP and motor parameters are not calibrated hardware parameters')
        if cfg.policy.base_command_conditioning:
            metadata['format']='wf_command_unified_torchscript_v2'
            metadata['inputs']['world_goal']='[N,11]: TCP xyz world, quaternion wxyz, base vx/vy/wz body, command-mode flag; mode=1 transports TCP goal with the base'
        if cfg.reward.goal_program:
            metadata['format']='wf_goal_program_torchscript_v3'
            metadata['inputs']['world_goal']='[N,12]: world TCP xyz+wxyz, body vx/vy/wz, command mode, normalized elapsed goal time'
            metadata['phase_protocol']=dict(reset=0.,increment_per_control_step=1./cfg.reward.program_steps,clamp_max=1.,
                description='caller advances phase after executing each action; reset on a new reaching program')
        if cfg.reward.program_context:
            metadata['format']='wf_goal_program_context_torchscript_v4'
            metadata['inputs']['world_goal']='[N,37]: task xyz+wxyz/vx/vy/wz/mode (11), elapsed phase (1), initial base xyz+wxyz (7), initial normalized nonwheel joints (18)'
            metadata['program_context_protocol']='cache initial base pose and normalized leg+arm joints on each new reaching program; keep fixed while advancing phase'
        metadata_path.write_text(json.dumps(metadata, indent=2)+'\n')
        return metadata
    finally:
        sim.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True); p.add_argument('--output', required=True)
    args = p.parse_args()
    result = export_checkpoint(args.checkpoint, args.output)
    print(json.dumps({k: result[k] for k in ('export_sha256', 'physics_hash', 'parity_max_abs_error')}))


if __name__ == '__main__':
    main()
