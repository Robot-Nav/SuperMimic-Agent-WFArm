"""Held-out real-physics reaching evaluation and paired actuator interventions."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .config import TrainConfig
from .networks import Actor
from .robot import build_model
from .simulator import BatchedSimulator
from .task import WholeBodyRewardCfg, WholeBodyTask


def physical_failure(task, state, metrics):
    """No timeout/success labels: only task limits and actual forbidden contacts."""
    return (task.terminated(state)
            | torch.as_tensor(metrics["forbidden_contact"], device=state.device)
            | torch.as_tensor(metrics["joint_limit_violation"] > 0.08, device=state.device))


@torch.no_grad()
def evaluate_policy(actor, task, model, spec, *, episodes=16, level=0, seed=10000,
                    steps=600, hold_steps=15, dt=0.02, friction=1.0, ablation="none", goals=None,
                    settle_steps=0, goal_seed_device="policy",initial_states=None,initial_actions=None,
                    arm_initial_position=None):
    if episodes < 1 or steps < 1 or hold_steps < 1 or hold_steps > steps:
        raise ValueError("require episodes > 0 and 1 <= hold_steps <= steps")
    if not 0 <= level <= 3:
        raise ValueError("level must be in [0, 3]")
    device = task.device
    if ablation == "base_locked":
        # The intervention's anchor is model data. Keep it private so even an
        # exception cannot alter the caller's physics hash or future rollouts.
        model = copy.copy(model)
        spec = replace(spec, model=model)
    sim = BatchedSimulator(model, spec, num_envs=episodes, seed=seed, dt=dt, friction=friction,
                           arm_initial_position=arm_initial_position)
    sim.reset()
    sim.settle(steps=settle_steps)
    if initial_states is not None:
        initial=torch.as_tensor(initial_states).detach().cpu().numpy()
        if initial.shape!=(episodes,57) or not np.isfinite(initial).all() or (np.linalg.norm(initial[:,3:7],axis=-1)<1e-8).any():
            raise ValueError('initial_states must be finite canonical (N,57) states with nonzero quaternion')
        sim.write_state(initial)
    s0 = torch.as_tensor(sim.read_state(), dtype=torch.float32, device=device)
    if goal_seed_device not in ("policy", "cpu"):
        raise ValueError("goal_seed_device must be policy or cpu")
    if goals is None:
        if goal_seed_device == "cpu":
            fixture_task = task if task.device.type == "cpu" else WholeBodyTask(model, spec, "cpu", task.cfg)
            goals = fixture_task.sample_goals(s0.cpu(), level=level,
                generator=torch.Generator().manual_seed(seed)).to(device)
        else:
            goals = task.sample_goals(s0, level=level,
                generator=torch.Generator(device=device).manual_seed(seed))
    else:
        goals = torch.as_tensor(goals, device=device, dtype=torch.float32).clone()
        allowed=(7,11,12,37) if task.cfg.program_context else ((7,11,12) if task.cfg.goal_program else ((7,11) if task.cfg.base_command_conditioning else (7,)))
        if goals.ndim!=2 or goals.shape[0]!=episodes or goals.shape[1] not in allowed or not torch.isfinite(goals).all():
            raise ValueError("goals must be finite with shape (episodes, 7), [xyz, quaternion wxyz]")
        lengths = goals[:, 3:7].norm(dim=-1, keepdim=True)
        if (lengths < 1e-8).any():
            raise ValueError("goal quaternions must have nonzero norm")
        goals[:, 3:7] /= lengths
    needs_context=task.cfg.program_context and goals.shape[-1]!=37
    goals=task.command_goal(goals)
    if needs_context:goals=task.initialize_goal(s0,goals)
    initial_goals=goals.clone()
    # All interventions start from the same freely settled state and world goals.
    # A physical base weld anchors that state instead of pulling it back upward
    # into the transient reset pose. Fixed-friction fixtures are identical here.
    if ablation == "base_locked":
        initial = sim.read_state()[:, :7]
        if not np.allclose(initial, initial[:1], atol=1e-8):
            raise ValueError("shared-model base weld requires identical fixture initial poses")
        model.eq_data[spec.base_weld_eq, 3:6] = initial[0, :3]
        model.eq_data[spec.base_weld_eq, 6:10] = initial[0, 3:7]
    sim.set_ablation(ablation)
    if initial_actions is not None:
        hold=np.asarray(torch.as_tensor(initial_actions).detach().cpu(),dtype=float)
        if hold.shape!=(episodes,22) or not np.isfinite(hold).all() or np.abs(hold).max()>1.00001:
            raise ValueError('invalid initial command history')
        sim.ablation_action[:]=hold
    # Sampling from the exact same initial state/seed makes goals identical across
    # interventions. No IK or successful-goal filtering is applied.
    target_hash = hashlib.sha256(goals.cpu().numpy().tobytes()).hexdigest()
    previous = torch.as_tensor(sim.applied_action,dtype=torch.float32,device=device).clone()
    if initial_actions is not None:
        previous=torch.as_tensor(initial_actions,dtype=torch.float32,device=device).clone()
        if previous.shape!=(episodes,22) or not torch.isfinite(previous).all() or (previous.abs()>1.00001).any():
            raise ValueError('invalid initial command history')
    fixture_hash=(hashlib.sha256(s0.cpu().numpy().tobytes()+previous.cpu().numpy().tobytes()).hexdigest()
        if initial_states is not None or initial_actions is not None else None)
    alive = torch.ones(episodes, dtype=torch.bool, device=device)
    failed = ~alive
    hold = torch.zeros(episodes, dtype=torch.long, device=device)
    best_hold = hold.clone()
    first_hold = torch.full_like(hold, -1)
    counts = torch.zeros(episodes, device=device)
    sum_pos2, sum_rot2 = counts.clone(), counts.clone()
    reward_sum = counts.clone()
    final_pos, final_rot = counts.clone(), counts.clone()
    final_linear_speed, final_angular_speed = counts.clone(), counts.clone()
    max_tilt = counts.clone()
    def roll_pitch(q):
        w,x,y,z=q.unbind(-1)
        return torch.stack((torch.atan2(2*(w*x+y*z),1-2*(x*x+y*y)),
                            torch.asin((2*(w*y-z*x)).clamp(-1.,1.))),-1)
    initial_rp=roll_pitch(s0[:,3:7])
    max_rp_change=torch.zeros(episodes,2,device=device)
    max_leg_motion,max_arm_motion=counts.clone(),counts.clone()
    min_height, max_height = s0[:, 2].clone(), s0[:, 2].clone()
    max_movement = counts.clone()
    slip_sum = counts.clone()
    contact_count = counts.clone()
    contact_sum = counts.clone()
    torque_fraction_max = counts.clone()
    limit_max = counts.clone()
    base_linear_error,base_angular_error=counts.clone(),counts.clone()
    final_base_velocity=torch.zeros(episodes,3,device=device)
    tail_base_velocity=final_base_velocity.clone();tail_base_error2=counts.clone();tail_base_counts=counts.clone()
    input_mode = actor.training
    actor.eval()
    for step in range(steps):
        state = torch.as_tensor(sim.read_state(), dtype=torch.float32, device=device)
        action = actor(task.actor_input(state, previous, goals))
        action[~alive] = 0
        sn = torch.as_tensor(sim.step(action.cpu().numpy()), dtype=torch.float32, device=device)
        executed = torch.as_tensor(sim.applied_action, dtype=torch.float32, device=device)
        physical = sim.metrics()
        goals=task.step_goal(state,sn,goals)
        rewards = task.reward(sn, executed, previous, goals)
        if task.cfg.base_command_conditioning:
            command_error=task.base_command_error(sn,goals)
            base_linear_error=torch.where(alive,command_error[...,:2].norm(dim=-1),base_linear_error)
            base_angular_error=torch.where(alive,command_error[...,2].abs(),base_angular_error)
            actual=command_error+goals[...,7:10]
            final_base_velocity=torch.where(alive[:,None],actual,final_base_velocity)
            if step>=max(0,steps-50):
                tail_base_velocity+=actual*alive[:,None]
                tail_base_error2+=command_error[...,:2].square().sum(-1)*alive
                tail_base_counts+=alive
        now_failed = physical_failure(task, sn, physical) & alive
        safe = torch.as_tensor(physical["safe"], device=device)
        success = task.success(sn, goals) & safe & alive & ~now_failed
        hold = torch.where(success, hold + 1, torch.zeros_like(hold))
        best_hold = torch.maximum(best_hold, hold)
        first_hold = torch.where((hold >= hold_steps) & (first_hold < 0), step + 1, first_hold)
        pe, re = rewards["position_error"], rewards["orientation_error"]
        final_pos = torch.where(alive, pe, final_pos)
        final_rot = torch.where(alive, re, final_rot)
        final_linear_speed = torch.where(alive, rewards["tcp_speed"], final_linear_speed)
        final_angular_speed = torch.where(alive, rewards["tcp_angular_speed"], final_angular_speed)
        counts += alive
        sum_pos2 += torch.where(alive, pe.square(), 0.)
        sum_rot2 += torch.where(alive, re.square(), 0.)
        reward_sum += torch.where(alive, rewards["total"], 0.)
        max_movement = torch.maximum(max_movement, torch.where(alive, (sn[:, :2] - s0[:, :2]).norm(dim=-1), 0.))
        max_rp_change=torch.maximum(max_rp_change,torch.where(alive[:,None],(roll_pitch(sn[:,3:7])-initial_rp).abs(),0.))
        max_leg_motion=torch.maximum(max_leg_motion,torch.where(alive,(sn[:,7:19]-s0[:,7:19]).norm(dim=-1),0.))
        max_arm_motion=torch.maximum(max_arm_motion,torch.where(alive,(sn[:,23:29]-s0[:,23:29]).norm(dim=-1),0.))
        height = sn[:, 2]
        min_height = torch.where(alive, torch.minimum(min_height, height), min_height)
        max_height = torch.where(alive, torch.maximum(max_height, height), max_height)
        tensor = lambda value: torch.as_tensor(value, dtype=torch.float32, device=device)
        max_tilt = torch.maximum(max_tilt, torch.where(alive, tensor(physical["tilt"]), 0.))
        slip_sum += tensor(physical["slip"]).sum(-1) * alive
        contact_count += tensor(physical["wheel_contact"]).sum(-1) * alive
        contact_sum += tensor(physical["wheel_contact"]).mean(-1) * alive
        torque_fraction_max = torch.maximum(torque_fraction_max,
            torch.where(alive, (tensor(physical["torque_peak"])/task.effort).amax(-1), 0.))
        limit_max = torch.maximum(limit_max, torch.where(alive, tensor(physical["joint_limit_violation"]), 0.))
        failed |= now_failed
        alive &= ~now_failed
        previous = executed.clone()
        if now_failed.any():
            # Preserve the terminal metrics above, then keep inactive worlds
            # numerically healthy. Their subsequent steps never enter scores.
            sim.reset(np.flatnonzero(now_failed.cpu().numpy()))
            previous[now_failed] = torch.as_tensor(sim.initial_action[now_failed.cpu().numpy()],dtype=torch.float32,device=device)
        if not alive.any():
            break
    actor.train(input_mode)
    n = counts.clamp_min(1)
    records = []
    for i in range(episodes):
        records.append({
            "goal": initial_goals[i].cpu().tolist(), "failed": bool(failed[i]),
            "held_success": bool((best_hold[i] >= hold_steps) & ~failed[i]),
            "final_held_success": bool((hold[i] >= hold_steps) & ~failed[i]),
            "time_limit_reached": bool(~failed[i]),
            "unsuccessful_timeout": bool(~failed[i] & (hold[i] < hold_steps)),
            "time_to_first_hold_s": float(first_hold[i]*dt) if first_hold[i] >= 0 else None,
            "steps": int(counts[i]), "final_position_error_m": float(final_pos[i]),
            "reward_sum": float(reward_sum[i]),
            "budgeted_reward_mean": float((reward_sum[i] - 5.0*(steps-counts[i]))/steps),
            "final_orientation_error_rad": float(final_rot[i]),
            "final_tcp_linear_speed_m_s": float(final_linear_speed[i]),
            "final_tcp_angular_speed_rad_s": float(final_angular_speed[i]),
            "rms_position_error_m": float((sum_pos2[i]/n[i]).sqrt()),
            "rms_orientation_error_rad": float((sum_rot2[i]/n[i]).sqrt()),
            "max_base_displacement_m": float(max_movement[i]),
            "max_roll_change_rad": float(max_rp_change[i,0]),
            "max_pitch_change_rad": float(max_rp_change[i,1]),
            "max_leg_joint_motion_rad": float(max_leg_motion[i]),
            "max_arm_joint_motion_rad": float(max_arm_motion[i]),
            "height_range_m": float(max_height[i]-min_height[i]),
            "max_tilt_rad": float(max_tilt[i]),
            "mean_contact_slip_m_s": float(slip_sum[i]/contact_count[i].clamp_min(1)),
            "wheel_contact_fraction": float(contact_sum[i]/n[i]),
            "max_torque_limit_fraction": float(torque_fraction_max[i]),
            "max_joint_limit_violation_rad": float(limit_max[i]),
        })
        if task.cfg.base_command_conditioning:
            records[-1].update(final_world_goal=goals[i,:7].cpu().tolist(),
                final_base_linear_command_error_m_s=float(base_linear_error[i]),
                final_base_yaw_command_error_rad_s=float(base_angular_error[i]),
                final_base_velocity_body_m_s_and_rad_s=final_base_velocity[i].cpu().tolist(),
                tail_base_velocity_body_m_s_and_rad_s=(tail_base_velocity[i]/tail_base_counts[i]).cpu().tolist() if tail_base_counts[i]>0 else None,
                tail_base_linear_error_rms_m_s=float((tail_base_error2[i]/tail_base_counts[i]).sqrt()) if tail_base_counts[i]>0 else None)
    mean = lambda name: float(np.mean([r[name] for r in records]))
    result={
        "status": "measured_in_mujoco; does_not_imply_convergence_or_hardware_validation",
        "seed": seed, "level": level, "ablation": ablation, "episodes": episodes,
        "horizon_steps": steps, "hold_steps": hold_steps, "friction": friction,
        "reset_settle_steps": settle_steps, "reset_settle_seconds": settle_steps*dt,
        "reset_arm_joint_target_rad": None if arm_initial_position is None else list(arm_initial_position),
        "goal_seed_device": goal_seed_device,
        "target_hash": target_hash, "reward_config": asdict(task.cfg),
        "held_success_rate": mean("held_success"),
        "budgeted_reward_mean": mean("budgeted_reward_mean"),
        "post_failure_reward_per_unexecuted_step": -5.0,
        "final_held_success_rate": mean("final_held_success"), "failure_rate": mean("failed"),
        "unsuccessful_timeout_rate": mean("unsuccessful_timeout"),
        "position_pass_rate": float((final_pos <= task.cfg.success_position).float().mean()),
        "orientation_pass_rate": float((final_rot <= task.cfg.success_orientation).float().mean()),
        "linear_speed_pass_rate": float((final_linear_speed <= task.cfg.success_linear_speed).float().mean()),
        "angular_speed_pass_rate": float((final_angular_speed <= task.cfg.success_angular_speed).float().mean()),
        "position_error_p50_m": float(np.median([r["final_position_error_m"] for r in records])),
        "position_error_p95_m": float(np.quantile([r["final_position_error_m"] for r in records], 0.95)),
        "orientation_error_p95_rad": float(np.quantile([r["final_orientation_error_rad"] for r in records], 0.95)),
        "mean_base_displacement_m": mean("max_base_displacement_m"),
        "mean_height_range_m": mean("height_range_m"),
        "mean_contact_slip_m_s": mean("mean_contact_slip_m_s"),
        "records": records,
    }
    if fixture_hash:result['initial_condition_hash']=fixture_hash
    return result


def config_from_dict(raw):
    from .config import CurriculumConfig, DemonstrationConfig, GuardConfig, ModelConfig, PolicyConfig, SimConfig
    raw = dict(raw)
    for name, cls in (("sim", SimConfig), ("model", ModelConfig),
                      ("policy", PolicyConfig), ("curriculum", CurriculumConfig),
                      ("reward", WholeBodyRewardCfg), ("guard", GuardConfig),
                      ("demonstrations", DemonstrationConfig)):
        if isinstance(raw.get(name), dict):
            raw[name] = cls(**raw[name])
    return TrainConfig(**raw)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--episodes", type=int, default=32)
    p.add_argument("--level", type=int, default=3, choices=range(4))
    p.add_argument("--steps", type=int)
    p.add_argument("--seed", type=int, default=20000)
    p.add_argument("--friction", type=float, default=1.0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--asset-dir")
    p.add_argument("--goal", type=float, nargs=7, metavar=("X", "Y", "Z", "QW", "QX", "QY", "QZ"))
    p.add_argument("--goals-json", help="JSON array of world goals [[x,y,z,qw,qx,qy,qz], ...]")
    p.add_argument('--reference-library',help='Evaluate actual reference starting states, previous commands and fixed goals')
    p.add_argument('--reference-split',choices=('train','validation','all'),default='validation')
    p.add_argument("--ablation", choices=("none", "base_locked", "legs_locked", "arm_locked", "all"), default="none")
    p.add_argument("--output", default="logs/wf_goal/eval.json")
    args = p.parse_args()
    torch.set_num_threads(4)
    ck = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    cfg = config_from_dict(ck["config"])
    model, spec = build_model(args.asset_dir or cfg.asset_dir, cfg.sim.timestep,
        leg_action_scale=cfg.sim.leg_action_scale, arm_action_scale=cfg.sim.arm_action_scale,
        wheel_action_scale=cfg.sim.wheel_action_scale)
    from .train import model_manifest
    if model_manifest(spec)["physics_hash"] != ck["manifest"]["physics_hash"]:
        raise ValueError("checkpoint physics differs from the current model; evaluate with the original assets/configuration")
    task = WholeBodyTask(model, spec, args.device, WholeBodyRewardCfg(**ck["reward_config"]))
    actor = Actor(cfg.policy).to(args.device)
    actor.load_state_dict(ck["actor"])
    if sum(bool(x) for x in (args.goal,args.goals_json,args.reference_library))>1:
        p.error('select one goal input: --goal, --goals-json, or --reference-library')
    explicit_goals = ([args.goal] if args.goal else json.loads(Path(args.goals_json).read_text()) if args.goals_json else None)
    initial={}
    if args.reference_library:
        from .demonstrations import Demonstrations
        demos=Demonstrations(args.reference_library,task,ck['manifest']['physics_hash'])
        rows=(torch.arange(demos.count,device=task.device) if args.reference_split=='all'
              else demos.training if args.reference_split=='train' else demos.validation)
        explicit_goals=demos.goal[rows]
        initial=dict(initial_states=demos.states[rows,0],initial_actions=demos.initial_action[rows])
    if explicit_goals is not None:
        args.episodes = len(explicit_goals)
    modes = ("none", "base_locked", "legs_locked", "arm_locked") if args.ablation == "all" else (args.ablation,)
    results = [evaluate_policy(actor, task, model, spec, episodes=args.episodes, level=args.level,
        seed=args.seed, steps=args.steps or cfg.sim.episode_steps,
        hold_steps=cfg.curriculum.hold_steps, dt=cfg.sim.dt, friction=args.friction,
        ablation=mode, goals=explicit_goals, settle_steps=cfg.sim.reset_settle_steps,
        goal_seed_device=cfg.curriculum.goal_seed_device,
        arm_initial_position=cfg.sim.arm_initial_position,**initial) for mode in modes]
    for result in results:
        result.update(checkpoint=str(Path(args.checkpoint).resolve()), checkpoint_iteration=ck["iteration"],
                      physics_hash=ck["manifest"]["physics_hash"])
    output = results if args.ablation == "all" else results[0]
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    for r in results:
        print(json.dumps({k: v for k, v in r.items() if k not in ("records", "reward_config")}, allow_nan=False))


if __name__ == "__main__":
    main()
