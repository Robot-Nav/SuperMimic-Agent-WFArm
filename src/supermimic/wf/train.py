"""Unified whole-body control using a learned model and BPTT.

Optional synthetic-reference guidance is an extension of the original
demonstration-free goal task; deployment still uses only proprioception and goal.

Run with PYTHONPATH=src python -m supermimic.wf.train --smoke. The CPU MuJoCo
collector is a correctness backend; --device cuda moves learning, not physics.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import copy
import hashlib
import json
from pathlib import Path
import time
import signal
import os
from types import SimpleNamespace

import mujoco
import numpy as np
import torch

from supermimic.imagination.objective import gae_targets, policy_objective
from supermimic.training.schedule import cosine_lr
from .config import TrainConfig, smoke_config
from .actor_trust import ActorTrustStep
from .evaluate import config_from_dict, evaluate_policy, physical_failure
from .guard import PolicyGuard
from .model import WholeBodyWorldModel
from .networks import Actor, Critic
from .replay import GoalReplayBuffer
from .simulator import BatchedSimulator
from .task import WholeBodyRewardCfg, WholeBodyTask
from .replay import ModelBatch, PolicyBatch


def model_manifest(spec):
    """Fingerprint effective physics, canonical ordering and source assets."""
    digest = hashlib.sha256()
    source_hashes = {}
    for relative in ("urdf/wf-arm-tbot.urdf", "config/robot.json", "config/robot_calibration.json"):
        source_hashes[relative] = hashlib.sha256((spec.asset_dir / relative).read_bytes()).hexdigest()
        digest.update(source_hashes[relative].encode())
    for field in ("body_mass", "body_inertia", "body_ipos", "body_iquat", "body_pos", "body_quat",
                  "jnt_type", "jnt_axis", "jnt_pos", "jnt_range", "dof_damping", "dof_armature",
                  "geom_type", "geom_size", "geom_pos", "geom_quat", "geom_contype", "geom_conaffinity",
                  "geom_dataid", "mesh_vert", "mesh_face", "geom_solref", "geom_solimp",
                  "actuator_gear", "actuator_ctrlrange", "eq_data"):
        digest.update(np.asarray(getattr(spec.model, field)).tobytes())
    for field in ("q0", "kp", "kd", "effort", "velocity", "action_scale"):
        digest.update(np.asarray(getattr(spec, field)).tobytes())
    digest.update(str(spec.model.opt.timestep).encode())
    digest.update(np.asarray(spec.model.opt.gravity).tobytes())
    # Sliding friction is a per-episode context, not fixed model provenance.
    digest.update(np.asarray(spec.model.geom_friction[:, 1:]).tobytes())
    digest.update(str((spec.model.opt.integrator, spec.model.opt.solver, spec.model.opt.iterations)).encode())
    digest.update("|".join(spec.joint_names).encode())
    return {"physics_hash": digest.hexdigest(), "asset_hashes": source_hashes,
        "mujoco_version": mujoco.__version__, "torch_version": torch.__version__,
        "joint_order": list(spec.joint_names), "nq": spec.model.nq, "nv": spec.model.nv,
        "nu": spec.model.nu, "mass_kg": float(spec.model.body_mass.sum()),
        "kp": spec.kp.tolist(), "kd": spec.kd.tolist(), "action_scale": spec.action_scale.tolist(),
        "parameter_status": "simulation assumptions; not identified hardware parameters",
        "tcp_status": "uncalibrated source-model TCP; no hardware accuracy claim",
        "collector": "serial native MuJoCo CPU, learning can use CPU or CUDA"}


class Trainer:
    def __init__(self, cfg: TrainConfig):
        cfg.validate()
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable; use --device cpu for validation")
        torch.set_num_threads(cfg.cpu_threads)
        torch.manual_seed(cfg.seed)
        self.rng = np.random.default_rng(cfg.seed)
        self.output = Path(cfg.output)
        if cfg.warm_start and Path(cfg.warm_start).resolve().parent == self.output.resolve():
            raise ValueError("warm-start output must differ from the source checkpoint directory")
        self.output.mkdir(parents=True, exist_ok=True)
        self.sim = BatchedSimulator(num_envs=cfg.sim.num_envs, seed=cfg.seed, dt=cfg.sim.dt,
            timestep=cfg.sim.timestep, asset_dir=cfg.asset_dir,
            friction_range=(cfg.sim.friction_min, cfg.sim.friction_max),
            leg_action_scale=cfg.sim.leg_action_scale, arm_action_scale=cfg.sim.arm_action_scale,
            wheel_action_scale=cfg.sim.wheel_action_scale,arm_initial_position=cfg.sim.arm_initial_position)
        self.task = WholeBodyTask(self.sim.model, self.sim.spec, self.device, cfg.reward)
        self.sim.settle(steps=cfg.sim.reset_settle_steps)
        self.manifest = model_manifest(self.sim.spec)
        self.excitation=None
        if cfg.model.excitation_file:
            from .excitation import DynamicsExcitation
            self.excitation=DynamicsExcitation(cfg.model.excitation_file,self.device,self.manifest['physics_hash'],cfg.sim.dt)
        self.demonstrations = None
        self.reference_pretrain_updates = 0
        self.reference_model_pretrain_updates = 0
        self.reference_selection = None
        self.response_check=None
        self.model_response_ready=False
        self.response_last_check=-1
        if cfg.demonstrations.directory:
            from .demonstrations import Demonstrations
            self.demonstrations = Demonstrations(cfg.demonstrations.directory,self.task,self.manifest["physics_hash"],
                cfg.demonstrations.query_file,cfg.demonstrations.query_fraction,cfg.demonstrations.command_sampling_fraction,
                cfg.demonstrations.zero_start_sampling_fraction,cfg.demonstrations.broad_anchor_sampling_fraction)
            if not np.isclose(self.demonstrations.manifest["dt"],cfg.sim.dt):
                raise ValueError("reference and training control periods differ")
            if cfg.policy.action_lower:
                lower=torch.tensor(cfg.policy.action_lower,device=self.device)
                upper=torch.tensor(cfg.policy.action_upper,device=self.device)
                if ((self.demonstrations.actions<lower-1e-5)|(self.demonstrations.actions>upper+1e-5)).any():
                    raise ValueError('reference commands exceed the configured operating bounds; regenerate references')
                if ((self.demonstrations.initial_action<lower-1e-5)|(self.demonstrations.initial_action>upper+1e-5)).any():
                    raise ValueError('reference initial commands exceed the configured operating bounds')
        self.wm = WholeBodyWorldModel(self.sim.spec.q0, cfg.model, cfg.sim.dt).to(self.device)
        self.actor = Actor(cfg.policy).to(self.device)
        self.critic = Critic(cfg.policy).to(self.device)
        self.wm_opt = torch.optim.Adam(self.wm.parameters(), lr=cfg.model.lr)
        if cfg.policy.architecture=='goal_program':
            actor_groups=[dict(params=self.actor.prior.parameters(),lr_scale=cfg.policy.program_prior_lr_scale,bc_lr_scale=1.,name='prior'),
                          dict(params=self.actor.feedback.parameters(),lr_scale=cfg.policy.feedback_lr_scale,
                               bc_lr_scale=1.,name='feedback')]
            if cfg.policy.increment_feedback:
                actor_groups.append(dict(params=self.actor.increment_feedback.parameters(),
                    lr_scale=cfg.policy.increment_feedback_lr_scale,bc_lr_scale=1.,name='increment_feedback'))
        elif cfg.policy.architecture=='matrix' and cfg.policy.feedback_lr_scale!=1.:
            feedback_names={'log_damping','log_wheel_stop'}
            ordinary=[p for name,p in self.actor.named_parameters() if name not in feedback_names]
            feedback=[p for name,p in self.actor.named_parameters() if name in feedback_names]
            actor_groups=[{'params':ordinary,'lr_scale':1.},
                {'params':feedback,'lr_scale':cfg.policy.feedback_lr_scale,
                 'bc_lr_scale':cfg.policy.feedback_lr_scale,'name':'feedback',
                 'lr':cfg.policy.lr*cfg.policy.feedback_lr_scale}]
        elif cfg.policy.goal_conditioning == "direct" and hasattr(self.actor,'goal_skip'):
            ordinary = [p for name,p in self.actor.named_parameters() if not name.startswith("goal_skip.")]
            actor_groups = [{"params": ordinary, "lr_scale": 1.0},
                {"params": self.actor.goal_skip.parameters(), "lr_scale": cfg.policy.goal_skip_lr_scale,
                 "lr": cfg.policy.lr*cfg.policy.goal_skip_lr_scale}]
        else:
            actor_groups = self.actor.parameters()
        self.actor_opt = torch.optim.Adam(actor_groups, lr=cfg.policy.lr, betas=(0.9, 0.95))
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg.policy.critic_lr)
        self.buffer = GoalReplayBuffer(cfg.sim.num_envs, cfg.replay_steps, self.device, cfg.model.validation_fraction,
                                      goal_size=self.task.command_size)
        self.previous = torch.as_tensor(self.sim.initial_action,dtype=torch.float32,device=self.device).clone()
        self.age = torch.zeros(cfg.sim.num_envs, dtype=torch.long, device=self.device)
        self.goal_hold=torch.zeros_like(self.age)
        self.level = cfg.curriculum.initial_level
        self.goals = self.sample_goals(self.state()).detach()
        self.iteration = self.model_updates = self.actor_updates = 0
        self.accepted_actor_updates = self.pending_actor_updates = 0
        self.warm_start_info = None
        self.curriculum_checks = 0
        self.validation_error = float("inf")
        self.latest = {}
        self.runtime_optimization=None
        self.clearance_optimization=None
        self.stop_requested = False
        self.completed = {"episodes": 0, "failed": 0}
        if cfg.sim.switch_goals_after_hold:self.completed['goal_switches']=0
        self.guard = (PolicyGuard(return_tolerance=cfg.guard.score_tolerance,
            failure_slack=cfg.guard.allowed_failure_increase, min_lr_scale=cfg.guard.min_lr_scale,
            prefer_success=cfg.guard.prefer_success,prefer_safety=cfg.guard.prefer_safety)
            if cfg.guard.enabled else None)
        self.guard_level = self.level
        self.guard_verified_iteration = -1
        if cfg.resume:
            self.load(cfg.resume)
        elif cfg.warm_start:
            self.load_dynamics(cfg.warm_start)
        if cfg.program_prior and not cfg.resume:
            from .networks import load_goal_program_prior
            source=torch.load(cfg.program_prior,map_location=self.device,weights_only=False)
            if source['manifest']['physics_hash']!=self.manifest['physics_hash'] or source['phase_duration_steps']!=cfg.reward.program_steps:
                raise ValueError('program-prior physics or duration differs')
            load_goal_program_prior(self.actor,source)
            self.warm_start_info=self.warm_start_info or {}
            self.warm_start_info['program_prior']=dict(source=str(Path(cfg.program_prior).resolve()),
                sha256=hashlib.sha256(Path(cfg.program_prior).read_bytes()).hexdigest(),reference_selection=source['reference_selection'])
            (self.output/'warm_start.json').write_text(json.dumps(self.warm_start_info,indent=2)+'\n')
        if not cfg.resume and self.guard is not None:
            self.latest.update(self.check_guard(rebase=True))
        (self.output / "config.json").write_text(json.dumps(asdict(cfg), indent=2) + "\n")
        (self.output / "manifest.json").write_text(json.dumps(self.manifest, indent=2) + "\n")

    def state(self):
        return torch.as_tensor(self.sim.read_state(), dtype=torch.float32, device=self.device)

    def sample_goals(self,state):
        goals=self.task.sample_goals(state,level=self.level)
        fraction=self.cfg.demonstrations.goal_bank_fraction
        if self.demonstrations is not None and fraction:
            mask=torch.rand(len(state),device=self.device)<fraction
            pool=self.demonstrations.training
            pool=pool[self.demonstrations.levels[pool]<=self.level]
            if not len(pool):return goals
            selected=pool[torch.randint(len(pool),(int(mask.sum()),),device=self.device)]
            broad_fraction=self.cfg.demonstrations.broad_anchor_sampling_fraction
            broad_pool=pool[self.demonstrations.broad_anchor[pool]]
            if broad_fraction and len(broad_pool):
                broad_mask=torch.rand(len(selected),device=self.device)<broad_fraction
                selected[broad_mask]=broad_pool[torch.randint(len(broad_pool),(int(broad_mask.sum()),),device=self.device)]
            command_fraction=self.cfg.demonstrations.command_sampling_fraction
            if command_fraction and self.demonstrations.goal.shape[-1]>=11:
                command_pool=pool[self.demonstrations.goal[pool,10]>.5]
                if len(command_pool):
                    command_mask=torch.rand(len(selected),device=self.device)<command_fraction
                    selected[command_mask]=command_pool[torch.randint(len(command_pool),(int(command_mask.sum()),),device=self.device)]
            if self.cfg.reward.base_command_conditioning:
                reference=self.demonstrations.goal[selected].clone()
                reference[...,10]=1.
                moved=self.task.transport_goal(self.demonstrations.states[selected,0],state[mask],reference)
                goals[mask,:7]=moved[...,:7]
            else:
                goals[mask]=self.demonstrations.goal[selected]
            if self.cfg.demonstrations.goal_bank_preserve_command and self.cfg.reward.base_command_conditioning:
                goals[mask,7:11]=self.demonstrations.goal[selected,7:11]
            goals[mask]=self.perturb_goals(goals[mask])
        goals=self.apply_command_curriculum(goals)
        return self.task.initialize_goal(state,goals) if self.cfg.reward.program_context else goals

    def apply_command_curriculum(self,goals):
        schedule=self.cfg.base_speed_curriculum
        if not schedule or not self.cfg.reward.base_command_conditioning:return goals
        step=getattr(self,'iteration',0);maximum=schedule[0][1]
        for boundary,speed in schedule:
            if step>=boundary:maximum=speed
        result=goals.clone();result[...,7]*=maximum/self.cfg.reward.base_command_ranges[0]
        return result

    def perturb_goals(self,goals):
        d=self.cfg.demonstrations
        if not d.goal_bank_position_noise and not d.goal_bank_rotation_noise:return goals
        from .math import quat_mul,quat_from_rotvec
        result=goals.clone()
        if d.goal_bank_position_noise:
            result[...,:3]+=(2*torch.rand_like(result[...,:3])-1)*d.goal_bank_position_noise
        if d.goal_bank_rotation_noise:
            rotation=(2*torch.rand_like(result[...,:3])-1)*d.goal_bank_rotation_noise
            result[...,3:7]=quat_mul(result[...,3:7],quat_from_rotvec(rotation))
        return result

    def request_stop(self):
        """Finish the current transition/update and save instead of interrupting it."""
        self.stop_requested = True

    @torch.no_grad()
    def collect(self):
        s = self.state()
        xi = torch.as_tensor(self.sim.xi.copy(), dtype=torch.float32, device=self.device)
        identifying=(self.cfg.model_only or self.iteration < self.cfg.warmup_steps or
                (self.cfg.model.verify_action_response and not self.model_response_ready) or
                (self.demonstrations is not None and self.iteration<self.cfg.demonstrations.policy_start_steps))
        if identifying:
            command = torch.as_tensor(self.sim.initial_action,dtype=torch.float32,device=self.device)
            if self.cfg.reward.goal_program:self.age.zero_()
        else:
            command = self.actor(self.task.actor_input(s, self.previous, self.goals))
        # Independent small excitation, around nominal posture during burn-in.
        # This is not an expert action or a pretrained stabilizing policy.
        std=(self.cfg.sim.delta_exploration_std if self.cfg.policy.action_mode in ('delta','target') and not identifying
             else self.cfg.sim.exploration_std)
        action = (command + std * torch.randn_like(command)).clamp(-1, 1)
        if self.cfg.sim.exploration_delta_limit is not None:
            limit=self.cfg.sim.exploration_delta_limit
            action=self.previous+(action-self.previous).clamp(-limit,limit)
        action = self.actor.operating_action(action)
        sn = torch.as_tensor(self.sim.step(action.cpu().numpy()), dtype=torch.float32, device=self.device)
        if not torch.isfinite(sn).all():
            raise FloatingPointError("non-finite simulator transition; inspect physics before learning")
        executed = torch.as_tensor(self.sim.applied_action.copy(), dtype=torch.float32, device=self.device)
        metrics = self.sim.metrics()
        failed = physical_failure(self.task, sn, metrics)
        self.age += 1
        timeout = self.age >= self.cfg.sim.episode_steps
        next_goals=self.task.step_goal(s,sn,self.goals,advance=not identifying)
        terms = self.task.reward(sn, executed, self.previous, next_goals)
        self.buffer.add(s, sn, executed, self.previous, self.goals, xi, failed, timeout)
        self.goals=next_goals.detach()
        self.previous = executed
        ended = failed | timeout
        if self.cfg.reward.base_command_conditioning and not self.cfg.demonstrations.goal_bank_preserve_command:
            change=((self.age%self.cfg.reward.base_command_resample_steps)==0)&~ended
            if change.any():
                sampled=self.apply_command_curriculum(self.task.sample_goals(sn[change],self.level))
                self.goals[change,7:11]=sampled[...,7:11]
        if self.cfg.sim.switch_goals_after_hold:
            success=terms['success'] & torch.as_tensor(metrics['safe'],device=self.device) & ~ended
            self.goal_hold=torch.where(success,self.goal_hold+1,torch.zeros_like(self.goal_hold))
            switch=(self.goal_hold>=self.cfg.sim.switch_goals_after_hold).nonzero().flatten()
            if len(switch):
                # Physical state and the executed command remain continuous.
                # Replay already copied the goal that generated this transition.
                self.goals[switch]=self.sample_goals(sn[switch]).detach()
                self.goal_hold[switch]=0
                self.completed['goal_switches']=self.completed.get('goal_switches',0)+len(switch)
        if ended.any():
            ids = ended.nonzero().squeeze(-1)
            self.completed["episodes"] += len(ids)
            self.completed["failed"] += int(failed[ids].sum())
            self.sim.reset(ids.cpu().numpy())
            if self.cfg.sim.reset_joint_noise:
                reset_state = self.sim.read_state()[ids.cpu().numpy()].copy()
                nonwheel = np.r_[0:12, 16:22]
                noise = self.rng.normal(0, self.cfg.sim.reset_joint_noise, (len(ids), 18))
                reset_state[:, 7 + nonwheel] = np.clip(reset_state[:, 7 + nonwheel] + noise,
                    self.sim.spec.low[nonwheel], self.sim.spec.high[nonwheel])
                self.sim.write_state(reset_state, ids=ids.cpu().numpy())
            self.sim.settle(ids.cpu().numpy(), steps=self.cfg.sim.reset_settle_steps)
            self.goals[ids] = self.sample_goals(self.state()[ids])
            self.age[ids] = 0
            self.goal_hold[ids]=0
            self.previous[ids] = torch.as_tensor(self.sim.initial_action[ids.cpu().numpy()],dtype=torch.float32,device=self.device)
        return {**{"reward_"+name: float(value.float().mean()) for name, value in terms.items()
                   if name not in ("total", "position_error", "orientation_error", "tcp_speed", "tcp_angular_speed", "success")},
                "sim_reward": float(terms["total"].mean()),
                "sim_position_error_m": float(terms["position_error"].mean()),
                "sim_orientation_error_rad": float(terms["orientation_error"].mean()),
                "sim_tcp_linear_speed": float(terms["tcp_speed"].mean()),
                "sim_tcp_angular_speed": float(terms["tcp_angular_speed"].mean()),
                "sim_contact_fraction": float(np.mean(metrics["wheel_contact"])),
                "sim_slip_m_s": float(np.sum(metrics["slip"]) / max(1, np.sum(metrics["wheel_contact"]))),
                "sim_failed_this_step": int(failed.sum())}

    def update_model(self):
        c = self.cfg.model
        if not self.buffer.has_windows(c.window) or not self.buffer.has_windows(c.window, validation=True):
            return {"model_waiting_for_windows": True}
        total = {}
        self.wm.train()
        for _ in range(c.batches_per_update):
            batch = self.sample_model_training_batch(c.batch_size, c.window)
            loss, terms = self.wm.window_loss(batch)
            if not torch.isfinite(loss):
                raise FloatingPointError("world-model loss became non-finite")
            self.wm_opt.zero_grad(set_to_none=True)
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(self.wm.parameters(), c.grad_clip, error_if_nonfinite=True)
            self.wm_opt.step()
            for name, value in terms.items():
                total[name] = total.get(name, 0.) + float(value) / c.batches_per_update
            total["grad_norm"] = float(grad)
            self.model_updates += 1
        total.update({'source_'+name+'_windows':count for name,count in self.model_source_counts.items()})
        validation = self.wm.validation(self.buffer.sample_windows(c.batch_size, c.window, validation=True))
        self.validation_error = float(validation[-1])
        total["validation_error_1"] = float(validation[0])
        total["validation_error_k"] = self.validation_error
        if self.demonstrations is not None:
            reference_validation=self.wm.validation(self.demonstrations.sample_model(c.batch_size,c.window,True))
            total["reference_validation_error_k"]=float(reference_validation[-1])
            self.validation_error=max(self.validation_error,float(reference_validation[-1]))
        if self.excitation is not None:
            val=self.wm.validation(self.excitation.sample(c.batch_size,c.window,True))
            total['excitation_validation_error_k']=float(val[-1])
            self.validation_error=max(self.validation_error,float(val[-1]))
        return {"wm_" + name: value for name, value in total.items()}

    def sample_model_training_batch(self, batch_size, horizon):
        """Allocate disjoint source slots before sampling any physical windows.

        Both configured fractions refer to the complete minibatch. Replacing
        the suffix of an already mixed batch can silently remove references.
        """
        reference = (round(batch_size*self.cfg.demonstrations.model_fraction)
                     if self.demonstrations is not None else 0)
        excitation = (round(batch_size*self.cfg.model.excitation_fraction)
                      if self.excitation is not None else 0)
        if min(reference,excitation)<0 or reference+excitation>batch_size:
            raise ValueError('world-model reference and excitation fractions exceed the minibatch')
        online = batch_size-reference-excitation
        self.model_source_counts = dict(online=online,reference=reference,excitation=excitation)
        parts=[]
        if online:parts.append(self.buffer.sample_windows(online,horizon))
        if reference:parts.append(self.demonstrations.sample_model(reference,horizon))
        if excitation:parts.append(self.excitation.sample(excitation,horizon))
        return ModelBatch(torch.cat([p.states for p in parts],1),
            torch.cat([p.actions for p in parts],1),torch.cat([p.xi for p in parts]),
            torch.cat([p.failed for p in parts],1))

    def mix_excitation(self,batch,horizon):
        if self.excitation is None:return batch
        n=round(len(batch.xi)*self.cfg.model.excitation_fraction)
        if not n:return batch
        part=self.excitation.sample(n,horizon)
        keep=len(batch.xi)-n
        return ModelBatch(torch.cat((batch.states[:,:keep],part.states),1),
            torch.cat((batch.actions[:,:keep],part.actions),1),torch.cat((batch.xi[:keep],part.xi)),
            torch.cat((batch.failed[:,:keep],part.failed),1))

    def imagined_rollout(self, batch, horizon, references=None, reference_weight=0.):
        s, prev, goal, xi = batch.state, batch.prev_action, batch.goal, batch.xi
        rewards, task_rewards, dones, inputs = [], [], [], []
        survival = [torch.ones(len(s), device=self.device)]
        bodies=self.task.fk(s)
        for step in range(horizon):
            policy_input = self.task.actor_input(s, prev, goal,bodies=bodies)
            inputs.append(policy_input)
            # Noise does not corrupt target features or previous command.
            noise = torch.zeros_like(policy_input)
            noise[..., :50] = self.cfg.policy.obs_noise_std * torch.randn_like(policy_input[..., :50])
            a = self.task.process_action(self.actor(policy_input + noise))
            sn = self.wm.step(s, a, xi)
            goal=self.task.step_goal(s,sn,goal)
            next_bodies=self.task.fk(sn)
            reward=self.task.reward(sn, a, prev, goal,bodies=next_bodies)["total"]
            task_rewards.append(reward)
            if references is not None and reference_weight:
                n=references.shape[1]
                error=self.wm.layout.error(sn[-n:],references[step+1]).square().mean(-1)
                reward=reward-torch.cat((torch.zeros(len(sn)-n,device=self.device),reference_weight*error))
            rewards.append(reward)
            learned = torch.sigmoid(self.wm.termination_logits(sn, xi))
            analytic = self.task.failure_probability(sn)
            d = 1 - (1 - learned) * (1 - analytic)
            dones.append(d)
            survival.append(survival[-1] * (1 - d))
            s, prev, bodies = sn, a, next_bodies
        inputs.append(self.task.actor_input(s, prev, goal,bodies=bodies))
        return SimpleNamespace(rewards=torch.stack(rewards), survival=torch.stack(survival),
                               task_rewards=torch.stack(task_rewards),
                               dones=torch.stack(dones), inputs=torch.stack(inputs))

    def update_policy(self):
        c = self.cfg.policy
        if self.cfg.model_only:
            return {"actor_gated":"model_only"}
        if (self.demonstrations is not None and self.cfg.demonstrations.policy_start_steps>0
                and self.iteration<=self.cfg.demonstrations.policy_start_steps):
            return {"actor_gated":"real_action_excitation_warmup"}
        if self.model_updates < self.cfg.model.actor_start_updates:
            return {"actor_gated": "model_warmup"}
        if not np.isfinite(self.validation_error) or self.validation_error > self.cfg.model.max_validation_error:
            return {"actor_gated": "heldout_model_error"}
        m=self.cfg.model
        if m.verify_action_response:
            if self.response_last_check<0 or self.model_updates-self.response_last_check>=m.response_check_interval:
                if self.response_check is None:
                    from .response import ActionResponseCheck
                    self.response_check=ActionResponseCheck(self.sim.model,self.sim.spec,dt=self.cfg.sim.dt,
                        settle_steps=self.cfg.sim.reset_settle_steps,device=self.device,
                        arm_initial_position=self.cfg.sim.arm_initial_position)
                measured=self.response_check.evaluate(self.wm,min_cosine=m.response_min_cosine,max_relative_error=m.response_max_relative_error)
                self.model_response_ready=measured['passed'];self.response_last_check=self.model_updates
                self.latest.update(wm_response_ready=self.model_response_ready,wm_response_min_cosine=measured['min_cosine'],
                                   wm_response_max_relative_error=measured['max_relative_error'])
                (self.output/f'model_response_{self.model_updates:07d}.json').write_text(json.dumps(measured,indent=2)+'\n')
            if not self.model_response_ready:return {'actor_gated':'heldout_action_response'}
        progress = self.accepted_actor_updates if c.horizon_progress == "accepted" else self.actor_updates
        ramp = min(1., progress / max(1, c.horizon_ramp_updates))
        horizon = round(c.initial_horizon + ramp * (c.horizon - c.initial_horizon))
        batch = self.buffer.sample_starts(c.batch_size)
        if c.goal_relabel_fraction:
            mask = torch.rand(len(batch.state), device=self.device) < c.goal_relabel_fraction
            batch.goal = batch.goal.clone()
            batch.goal[mask] = self.sample_goals(batch.state[mask]).detach()
        references=None
        d=self.cfg.demonstrations
        guidance=max(0.,1.-self.actor_updates/d.decay_updates) if self.demonstrations is not None else 0.
        if self.demonstrations is not None:
            n=round(c.batch_size*d.start_fraction)
            if n:
                demo,references=self.demonstrations.sample_policy(n,horizon)
                if self.cfg.reward.goal_program:
                    demo.goal=self.perturb_goals(demo.goal)
                    if d.goal_bank_position_noise or d.goal_bank_rotation_noise:
                        # A trajectory reaching the original endpoint is not a
                        # matching state target for a newly perturbed 6D goal.
                        # Keep the physical start, learn its new task reward,
                        # and retain BC on separate correctly paired examples.
                        references=None
                batch=PolicyBatch(*(torch.cat((getattr(batch,key)[:c.batch_size-n],getattr(demo,key)))
                    for key in ("state","prev_action","goal","xi")))
        # Freeze model and critic weights, retaining their state/action Jacobians.
        for module in (self.wm, self.critic):
            module.requires_grad_(False)
        try:
            ro = self.imagined_rollout(batch, horizon,references,d.tracking_weight*guidance)
            terminal_values = self.critic(ro.inputs[-1], batch.xi)
            discounted_reward = policy_objective(ro, None, c.gamma)
            objective = policy_objective(ro, terminal_values, c.gamma)
            if not torch.isfinite(objective):
                raise FloatingPointError("imagined return became non-finite")
            self.actor_opt.zero_grad(set_to_none=True)
            bc_loss=(self.demonstrations.loss(self.actor,self.task,d.batch_size,noise_std=d.observation_noise_std)
                     if self.demonstrations is not None and d.bc_weight*guidance else objective.new_zeros(()))
            if self.demonstrations is not None and d.hold_weight:
                bc_loss=bc_loss+d.hold_weight*self.demonstrations.hold_loss(self.actor,self.task,d.batch_size)
            (-objective / horizon+d.bc_weight*guidance*bc_loss).backward()
            gn = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), c.grad_clip, error_if_nonfinite=True)
            torch.nn.utils.clip_grad_value_(self.actor.parameters(), c.grad_value_clip)
            self.actor_opt.step()
        finally:
            for module in (self.wm, self.critic):
                module.requires_grad_(True)
        with torch.no_grad():
            inputs = ro.inputs.detach()
            contexts = batch.xi[None].expand(horizon + 1, -1, -1)
            values = self.critic(inputs, contexts)
            # In a goal-conditioned policy, a teacher corridor is an auxiliary
            # Actor training signal. It is absent at deployment and may only
            # apply to part of an imagined batch. Optionally keep the Critic's
            # target on the observable goal task instead of that hidden label.
            critic_rewards=ro.rewards if d.tracking_in_critic else ro.task_rewards
            targets = gae_targets(critic_rewards.detach(), ro.dones.detach(), values, c.gamma, c.gae_lambda)
        for _ in range(c.critic_epochs):
            predicted = self.critic(inputs[:-1], contexts[:-1])
            loss = (predicted - targets).square().mean()
            self.critic_opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), c.grad_clip, error_if_nonfinite=True)
            self.critic_opt.step()
        self.actor_updates += 1
        self.pending_actor_updates += 1
        if self.guard is None:
            self.accepted_actor_updates += 1
            self.pending_actor_updates = 0
        return {"actor_gated": False, "actor_objective": float(objective.detach()), "actor_grad_norm": float(gn),
                "reference_state_tracking_applied": references is not None and bool(d.tracking_weight*guidance),
                "imag_reward_mean": float(ro.rewards.detach().mean()),
                "imag_discounted_reward": float(discounted_reward.detach()),
                "critic_bootstrap": float((objective-discounted_reward).detach()),
                "imag_survival_H": float(ro.survival[-1].detach().mean()),
                "critic_loss": float(loss.detach()), "horizon": horizon,
                "reference_bc_loss":float(bc_loss.detach()),"reference_guidance_scale":guidance,
                "critic_reference_tracking_in_target":bool(d.tracking_in_critic)}

    def evaluate(self, *, episodes, seed):
        actor,task=self.evaluation_modules()
        fixture={}
        if self.demonstrations is not None and self.cfg.demonstrations.evaluate_goal_bank:
            from .math import quat_mul,quat_from_rotvec
            # Private CPU fixture generator: evaluation never consumes the
            # training RNG or consults the tested Actor for target selection.
            generator=torch.Generator().manual_seed(seed)
            physical=BatchedSimulator(self.sim.model,self.sim.spec,num_envs=episodes,seed=seed,dt=self.cfg.sim.dt,friction=1.,
                arm_initial_position=self.cfg.sim.arm_initial_position)
            physical.settle(steps=self.cfg.sim.reset_settle_steps)
            initial=torch.tensor(physical.read_state(),dtype=torch.float32);physical.close()
            target_task=WholeBodyTask(self.sim.model,self.sim.spec,'cpu',self.task.cfg)
            goals=target_task.sample_goals(initial,self.level,generator)
            pool=self.demonstrations.training.cpu()
            selected=pool[torch.randint(len(pool),(episodes,),generator=generator)]
            broad_fraction=self.cfg.demonstrations.broad_anchor_sampling_fraction
            broad_pool=pool[self.demonstrations.broad_anchor[pool.to(self.device)].cpu()]
            if broad_fraction and len(broad_pool):
                broad_mask=torch.rand(episodes,generator=generator)<broad_fraction
                selected[broad_mask]=broad_pool[torch.randint(len(broad_pool),(int(broad_mask.sum()),),generator=generator)]
            if self.cfg.demonstrations.command_sampling_fraction:
                command_pool=pool[self.demonstrations.goal[pool.to(self.device),10].cpu()>.5]
                if len(command_pool):
                    command_mask=torch.rand(episodes,generator=generator)<self.cfg.demonstrations.command_sampling_fraction
                    selected[command_mask]=command_pool[torch.randint(len(command_pool),(int(command_mask.sum()),),generator=generator)]
            source=self.demonstrations.goal[selected.to(self.device)].cpu().clone();source[:,10]=1.
            relocated=target_task.transport_goal(self.demonstrations.states[selected.to(self.device),0].cpu(),initial,source)
            goals[:,:7]=relocated[:,:7]
            d=self.cfg.demonstrations
            if d.goal_bank_preserve_command:goals[:,7:11]=self.demonstrations.goal[selected.to(self.device),7:11].cpu()
            goals[:,:3]+=(2*torch.rand((episodes,3),generator=generator)-1)*d.goal_bank_position_noise
            rotation=(2*torch.rand((episodes,3),generator=generator)-1)*d.goal_bank_rotation_noise
            goals[:,3:7]=quat_mul(goals[:,3:7],quat_from_rotvec(rotation))
            if self.cfg.reward.program_context:goals=target_task.initialize_goal(initial,goals)
            fixture=dict(goals=goals.to(task.device),initial_states=initial.to(task.device))
        result=evaluate_policy(actor, task, self.sim.model, self.sim.spec,
            episodes=episodes, level=self.level, seed=seed, steps=self.cfg.sim.episode_steps,
            hold_steps=self.cfg.curriculum.hold_steps, dt=self.cfg.sim.dt,
            settle_steps=self.cfg.sim.reset_settle_steps,
            goal_seed_device=self.cfg.curriculum.goal_seed_device,
            arm_initial_position=self.cfg.sim.arm_initial_position,**fixture)
        if fixture:result['goal_sampling']='fixed new-pose offsets around training anchors; original held-out anchors evaluated separately'
        return result

    def evaluation_modules(self):
        if self.cfg.curriculum.evaluation_device=='cpu' and self.device.type!='cpu':
            actor=copy.deepcopy(self.actor).cpu()
            task=WholeBodyTask(self.sim.model,self.sim.spec,'cpu',self.task.cfg)
            return actor,task
        return self.actor,self.task

    def evaluate_guard(self):
        main=self.evaluate(episodes=self.cfg.guard.episodes,seed=self.cfg.seed+15000)
        n=self.cfg.guard.reference_episodes;anchors=self.cfg.guard.reference_anchor_episodes
        if not (n or anchors) or self.demonstrations is None:return main
        actor,task=self.evaluation_modules()
        indices=self.demonstrations.validation[:n]
        validation_count=len(indices)
        if anchors:indices=torch.cat((indices,self.demonstrations.training[:anchors]))
        goals=self.demonstrations.goal[indices].to(task.device)
        initial=(dict(initial_states=self.demonstrations.states[indices,0].to(task.device),
            initial_actions=self.demonstrations.initial_action[indices].to(task.device))
            if self.cfg.demonstrations.validate_initial_states else {})
        ref=evaluate_policy(actor,task,self.sim.model,self.sim.spec,episodes=len(goals),goals=goals,
            level=self.level,seed=self.cfg.seed+16000,steps=self.cfg.sim.episode_steps,
            hold_steps=self.cfg.curriculum.hold_steps,dt=self.cfg.sim.dt,
            settle_steps=self.cfg.sim.reset_settle_steps,goal_seed_device='cpu',
            arm_initial_position=self.cfg.sim.arm_initial_position,**initial)
        for i,record in enumerate(ref['records']):record['reference_role']='held_out_goal' if i<validation_count else 'known_training_anchor'
        original=copy.deepcopy(main)
        records=[dict(x,validation_component='task') for x in main['records']]+[dict(x,validation_component='reference') for x in ref['records']]
        mean=lambda key:float(np.mean([x[key] for x in records]))
        main.update(episodes=len(records),records=records,
            target_hash=hashlib.sha256(np.asarray([x['goal'] for x in records],dtype=np.float32).tobytes()).hexdigest(),
            held_success_rate=mean('held_success'),final_held_success_rate=mean('final_held_success'),
            failure_rate=mean('failed'),budgeted_reward_mean=mean('budgeted_reward_mean'),
            unsuccessful_timeout_rate=mean('unsuccessful_timeout'),
            position_error_p50_m=float(np.median([x['final_position_error_m'] for x in records])),
            position_error_p95_m=float(np.quantile([x['final_position_error_m'] for x in records],.95)),
            orientation_error_p95_rad=float(np.quantile([x['final_orientation_error_rad'] for x in records],.95)),
            mean_base_displacement_m=mean('max_base_displacement_m'),mean_height_range_m=mean('height_range_m'),
            mean_contact_slip_m_s=mean('mean_contact_slip_m_s'),component_evaluations={'task':original,'reference':ref})
        for name,field,threshold in (('position','final_position_error_m',task.cfg.success_position),
                ('orientation','final_orientation_error_rad',task.cfg.success_orientation),
                ('linear_speed','final_tcp_linear_speed_m_s',task.cfg.success_linear_speed),
                ('angular_speed','final_tcp_angular_speed_rad_s',task.cfg.success_angular_speed)):
            main[name+'_pass_rate']=float(np.mean([x[field]<=threshold for x in records]))
        if initial:
            main['initial_condition_hash']=hashlib.sha256(('nominal_task:'+ref['initial_condition_hash']).encode()).hexdigest()
        return main

    def check_curriculum(self):
        c = self.cfg.curriculum
        evaluated_level = self.level
        evaluation = self.evaluate(episodes=c.eval_episodes, seed=self.cfg.seed + 10000)
        (self.output / f"validation_{self.iteration:07d}.json").write_text(json.dumps(evaluation, indent=2, allow_nan=False)+"\n")
        passed = evaluation["final_held_success_rate"] >= c.promote_success_rate
        self.curriculum_checks = self.curriculum_checks + 1 if passed else 0
        if self.curriculum_checks >= c.promote_checks and self.level < c.max_level:
            self.level += 1
            self.curriculum_checks = 0
            # Existing episodes keep their world goals; new goals use new level.
            if self.guard is not None:
                self.latest.update(self.check_guard(rebase=True))
        return {"eval_final_held_success_rate": evaluation["final_held_success_rate"],
                "eval_failure_rate": evaluation["failure_rate"], "curriculum_level": self.level,
                "eval_iteration": self.iteration, "eval_level": evaluated_level}

    def check_guard(self, rebase=False):
        """Additional physical validation, separate from final test goals.

        Candidate failures remain useful world-model data; only the Actor,
        Critic and their optimizer states are restored upon rejection.
        """
        if self.guard is None:
            return {}
        evaluation = self.evaluate_guard()
        critic_args = (self.critic, self.critic_opt) if self.cfg.guard.rollback_critic else ()
        trials = []
        alpha = 1.0
        if not rebase and self.cfg.guard.backtrack_alphas:
            full_ok, _ = self.guard.assess(evaluation)
            if not full_ok:
                accepted = self.guard.state_dict()["weights"]
                search = ActorTrustStep(accepted["actor"], accepted["actor_opt"],
                    self.actor.state_dict(), self.actor_opt.state_dict(),
                    alphas=self.cfg.guard.backtrack_alphas)
                trials.append({"alpha": 1.0, "score": evaluation["budgeted_reward_mean"],
                               "failure_rate": evaluation["failure_rate"]})
                for fraction in self.cfg.guard.backtrack_alphas:
                    search.apply(self.actor, self.actor_opt, fraction)
                    evaluation = self.evaluate_guard()
                    alpha = fraction
                    trials.append({"alpha": fraction, "score": evaluation["budgeted_reward_mean"],
                                   "failure_rate": evaluation["failure_rate"]})
                    if self.guard.assess(evaluation)[0]:
                        break
        if rebase:
            outcome = self.guard.capture_baseline(evaluation, self.actor, self.actor_opt, *critic_args)
            self.guard_level = self.level
        else:
            outcome = self.guard.observe(evaluation, self.actor, self.actor_opt, *critic_args)
            if outcome["guard_accepted"]:
                self.accepted_actor_updates += round(self.pending_actor_updates*alpha)
            self.pending_actor_updates = 0
        outcome["guard_step_fraction"] = alpha if outcome["guard_accepted"] else 0.0
        if 'component_evaluations' in evaluation:
            outcome.update(guard_task_success=evaluation['component_evaluations']['task']['final_held_success_rate'],
                           guard_reference_success=evaluation['component_evaluations']['reference']['final_held_success_rate'])
        outcome["accepted_actor_updates"] = self.accepted_actor_updates
        outcome["guard_iteration"] = self.iteration
        outcome["guard_level"] = self.level
        self.guard_verified_iteration = self.iteration
        suffix = "baseline" if rebase else "candidate"
        (self.output / f"guard_{self.iteration:07d}_{suffix}.json").write_text(
            json.dumps({"evaluation": evaluation, "decision": outcome, "backtracking": trials}, indent=2, allow_nan=False)+"\n")
        return outcome

    def save(self, name="ckpt_latest.pt"):
        replay = {key: value.cpu() if isinstance(value, torch.Tensor) else value
                  for key, value in vars(self.buffer).items() if key != "device"}
        checkpoint = {"format": "wf_goal_bptt_v1", "config": asdict(self.cfg),
            "runtime_optimization":self.runtime_optimization,
            "clearance_optimization":self.clearance_optimization,
            "reward_config": asdict(self.task.cfg), "manifest": self.manifest,
            "iteration": self.iteration, "model_updates": self.model_updates,
            "actor_updates": self.actor_updates, "actor": self.actor.state_dict(),
            "accepted_actor_updates": self.accepted_actor_updates,
            "pending_actor_updates": self.pending_actor_updates, "warm_start_info": self.warm_start_info,
            "reference_pretrain_updates":self.reference_pretrain_updates,
            "reference_model_pretrain_updates":self.reference_model_pretrain_updates,
            "excitation_fingerprint":self.excitation.fingerprint if self.excitation is not None else None,
            "reference_selection":self.reference_selection,
            "model_response_ready":self.model_response_ready,"response_last_check":self.response_last_check,
            "reference_fingerprint":self.demonstrations.fingerprint if self.demonstrations is not None else None,
            "world_model": self.wm.state_dict(), "critic": self.critic.state_dict(),
            "wm_opt": self.wm_opt.state_dict(), "actor_opt": self.actor_opt.state_dict(),
            "critic_opt": self.critic_opt.state_dict(), "replay": replay,
            "simulation": self.sim.snapshot(), "previous": self.previous, "goals": self.goals,
            "age": self.age, "level": self.level, "curriculum_checks": self.curriculum_checks,
            "goal_hold":self.goal_hold,
            "validation_error": self.validation_error, "completed": self.completed,
            "guard": self.guard.state_dict() if self.guard is not None else None,
            "guard_level": self.guard_level, "guard_verified_iteration": self.guard_verified_iteration,
            "torch_rng": torch.get_rng_state(), "numpy_rng": self.rng.bit_generator.state,
            "cuda_rng": torch.cuda.get_rng_state_all() if self.device.type == "cuda" else None,
            "validation_status": "pipeline implementation; assess competence with held-out physical evaluations"}
        path = self.output / name
        temp = path.with_suffix(".tmp")
        torch.save(checkpoint, temp)
        temp.replace(path)

    def save_actor_checkpoint(self):
        """Retain a small Actor snapshot; the complete latest checkpoint resumes training."""
        path=self.output/f'actor_{self.iteration:07d}.pt'
        if path.exists():raise FileExistsError(path)
        checkpoint=dict(format='wf_goal_bptt_v1',config=asdict(self.cfg),
            runtime_optimization=self.runtime_optimization,
            clearance_optimization=self.clearance_optimization,
            reward_config=asdict(self.task.cfg),manifest=self.manifest,
            iteration=self.iteration,model_updates=self.model_updates,actor_updates=self.actor_updates,
            accepted_actor_updates=self.accepted_actor_updates,actor=self.actor.state_dict(),
            validation_status='periodic candidate; requires physical evaluation; not a competence claim')
        temp=path.with_suffix('.tmp');torch.save(checkpoint,temp);temp.replace(path)

    def load_dynamics(self, path):
        """Transfer only self-collected physics data/model; Actor and value stay new.

        No optimizer/guard state from the faulty run is reused, and the source
        checkpoint is never overwritten. This is not expert-data pretraining.
        """
        ck = torch.load(path, map_location=self.device, weights_only=False)
        if ck.get("format") != "wf_goal_bptt_v1" or ck["manifest"]["physics_hash"] != self.manifest["physics_hash"]:
            raise ValueError("warm-start requires a WF checkpoint with identical robot physics")
        self.wm.load_state_dict(ck["world_model"])
        if self.cfg.warm_start_actor:
            actor_ck=(torch.load(self.cfg.warm_start_actor_path,map_location=self.device,weights_only=False)
                if self.cfg.warm_start_actor_path else ck)
            if actor_ck['manifest']['physics_hash']!=self.manifest['physics_hash']:
                raise ValueError('separate Actor source uses different robot physics')
            source_policy = config_from_dict(actor_ck["config"]).policy
            if (source_policy.goal_conditioning != self.cfg.policy.goal_conditioning
                    or source_policy.goal_position_scale != self.cfg.policy.goal_position_scale
                    or source_policy.ignore_previous_action != self.cfg.policy.ignore_previous_action
                    or source_policy.action_mode != self.cfg.policy.action_mode
                    or source_policy.action_delta_limit != self.cfg.policy.action_delta_limit
                    or source_policy.architecture != self.cfg.policy.architecture
                    or source_policy.constraint_observation != self.cfg.policy.constraint_observation
                    or source_policy.full_goal_orientation != self.cfg.policy.full_goal_orientation
                    or WholeBodyRewardCfg(**actor_ck["reward_config"]).goal_orientation_frame != self.cfg.reward.goal_orientation_frame):
                raise ValueError("retaining Actor requires identical goal representation")
            if self.cfg.policy.base_command_conditioning and not source_policy.base_command_conditioning:
                from .networks import load_command_upgrade
                load_command_upgrade(self.actor,actor_ck['actor'])
            elif self.cfg.policy.program_context and not source_policy.program_context:
                from .networks import load_program_context_upgrade
                load_program_context_upgrade(self.actor,actor_ck['actor'])
            elif self.cfg.policy.increment_feedback and not source_policy.increment_feedback:
                from .networks import load_increment_feedback_upgrade
                load_increment_feedback_upgrade(self.actor,actor_ck['actor'])
            else:
                self.actor.load_state_dict(actor_ck["actor"])
        copied = ck["replay"]["E"] == self.buffer.E and ck["replay"]["T"] == self.buffer.T
        if copied:
            for name, value in ck["replay"].items():
                if name=='goal' and value.shape[-1]!=self.buffer.goal.shape[-1]:
                    value=self.task.command_goal(value)
                    if self.cfg.reward.program_context:
                        value=self.task.initialize_goal(ck['replay']['state'].to(self.device),value.to(self.device));value[...,11]=0.
                setattr(self.buffer, name, value.to(self.device).clone() if isinstance(value, torch.Tensor) else value)
            # Fresh physics/reset/context is not the successor of the old run's
            # final transition. Preserve data, but break every model window here.
            if self.buffer.total:
                self.buffer.ended[(self.buffer.total-1) % self.buffer.T] = True
        self.model_updates = int(ck["model_updates"])
        if copied and self.buffer.has_windows(self.cfg.model.window, validation=True):
            batch = self.buffer.sample_windows(self.cfg.model.batch_size, self.cfg.model.window, validation=True)
            self.validation_error = float(self.wm.validation(batch)[-1])
        self.warm_start_info = {"source": str(Path(path).resolve()), "source_iteration": ck["iteration"],
            "replay_copied": copied, "actor": "self_trained_source" if self.cfg.warm_start_actor else "random_init", "critic": "random_init",
            "optimizers": "new", "guard": "new", "simulation": "fresh_settled_reset"}
        if self.cfg.warm_start_actor_path:
            p=Path(self.cfg.warm_start_actor_path)
            self.warm_start_info.update(actor_source=str(p.resolve()),actor_source_sha256=hashlib.sha256(p.read_bytes()).hexdigest())
        (self.output / "warm_start.json").write_text(json.dumps(self.warm_start_info, indent=2)+"\n")

    def load(self, path):
        ck = torch.load(path, map_location=self.device, weights_only=False)
        if ck.get("format") != "wf_goal_bptt_v1":
            raise ValueError("expected a WF goal-conditioned checkpoint, not a G1 checkpoint")
        if ck["manifest"]["physics_hash"] != self.manifest["physics_hash"]:
            raise ValueError("robot physics changed since checkpoint")
        expected = asdict(self.cfg)
        previous = asdict(config_from_dict(ck["config"]))
        # Paths, device and total budget can change; learning/task semantics cannot.
        for name in ("sim", "model", "policy", "curriculum", "reward", "guard", "demonstrations", "replay_steps", "seed", "warmup_steps", "update_every"):
            if json.dumps(expected[name], sort_keys=True) != json.dumps(previous.get(name), sort_keys=True):
                raise ValueError(f"resume configuration mismatch in {name}")
        source_reward = asdict(WholeBodyRewardCfg(**ck["reward_config"]))
        if json.dumps(source_reward, sort_keys=True) != json.dumps(asdict(self.task.cfg), sort_keys=True):
            raise ValueError("reward/success configuration changed since checkpoint")
        self.actor.load_state_dict(ck["actor"])
        self.wm.load_state_dict(ck["world_model"])
        self.critic.load_state_dict(ck["critic"])
        self.wm_opt.load_state_dict(ck["wm_opt"])
        self.actor_opt.load_state_dict(ck["actor_opt"])
        self.critic_opt.load_state_dict(ck["critic_opt"])
        for name, value in ck["replay"].items():
            setattr(self.buffer, name, value.to(self.device) if isinstance(value, torch.Tensor) else value)
        self.sim.restore(ck["simulation"])
        for name in ("previous", "goals", "age", "iteration", "model_updates", "actor_updates", "level",
                     "curriculum_checks", "validation_error", "completed"):
            setattr(self, name, ck[name])
        if self.guard is not None:
            if ck.get("guard") is None:
                raise ValueError("resume requires a physical policy guard snapshot")
            self.guard.load_state_dict(ck["guard"])
        self.guard_level = ck.get("guard_level", self.level)
        self.guard_verified_iteration = ck.get("guard_verified_iteration", -1)
        self.accepted_actor_updates = ck.get("accepted_actor_updates", 0)
        self.pending_actor_updates = ck.get("pending_actor_updates", 0)
        self.goal_hold=ck.get('goal_hold',torch.zeros_like(self.age)).to(self.device)
        self.warm_start_info = ck.get("warm_start_info")
        expected_fingerprint=self.demonstrations.fingerprint if self.demonstrations is not None else None
        if ck.get("reference_fingerprint")!=expected_fingerprint:
            raise ValueError("reference library changed since checkpoint")
        if ck.get('excitation_fingerprint')!=(self.excitation.fingerprint if self.excitation is not None else None):
            raise ValueError('real-excitation data changed since checkpoint')
        self.reference_pretrain_updates=ck.get("reference_pretrain_updates",0)
        self.reference_model_pretrain_updates=ck.get("reference_model_pretrain_updates",0)
        self.reference_selection=ck.get("reference_selection")
        self.model_response_ready=ck.get("model_response_ready",False)
        self.response_last_check=ck.get("response_last_check",-1)
        torch.set_rng_state(ck["torch_rng"].cpu())
        self.rng.bit_generator.state = ck["numpy_rng"]
        if ck["cuda_rng"] is not None and self.device.type == "cuda":
            torch.cuda.set_rng_state_all([x.cpu() for x in ck["cuda_rng"]])

    def evaluate_reference_validation(self):
        """One fixed physical fixture for BC selection, including its incumbent."""
        d=self.cfg.demonstrations
        rows=self.demonstrations.validation
        heldout_count=len(rows)
        if self.cfg.reward.goal_program and self.cfg.guard.reference_anchor_episodes:
            rows=torch.cat((rows,self.demonstrations.training[:self.cfg.guard.reference_anchor_episodes]))
        goals=self.demonstrations.goal[rows]
        actor,task=self.evaluation_modules()
        common=dict(level=self.level,seed=self.cfg.seed+17000,steps=self.cfg.sim.episode_steps,
            hold_steps=self.cfg.curriculum.hold_steps,dt=self.cfg.sim.dt,
            settle_steps=self.cfg.sim.reset_settle_steps,goal_seed_device='cpu',
            arm_initial_position=self.cfg.sim.arm_initial_position)
        initial=(dict(initial_states=self.demonstrations.states[rows,0].to(task.device),
            initial_actions=self.demonstrations.initial_action[rows].to(task.device))
            if d.validate_initial_states else {})
        result=evaluate_policy(actor,task,self.sim.model,self.sim.spec,
            episodes=len(goals),goals=goals.to(task.device),**common,**initial)
        for i,record in enumerate(result['records']):
            index=int(rows[i]);source=self.demonstrations.manifest['clips'][index]
            record.update(reference_role='held_out_goal' if i<heldout_count else 'known_training_anchor',
                reference_file=source['file'],reference_family=source['family'],reference_subtype=source.get('subtype'))
        score=lambda e:100*e['final_held_success_rate']-100*e['failure_rate']+e['budgeted_reward_mean']
        scores=[score(result)];failures=[result['failure_rate']];successes=[result['final_held_success_rate']]
        if d.task_selection_episodes:
            other=evaluate_policy(actor,task,self.sim.model,self.sim.spec,episodes=d.task_selection_episodes,**common)
            result['task_validation']=other
            scores.append(score(other));failures.append(other['failure_rate']);successes.append(other['final_held_success_rate'])
        value=float(np.mean(scores))
        key=(-float(np.mean(failures)),float(np.mean(successes)),value) if self.cfg.guard.prefer_safety else (value,)
        result['reference_pretrain_updates']=self.reference_pretrain_updates
        (self.output/f'reference_validation_{self.reference_pretrain_updates:07d}.json').write_text(json.dumps(result,indent=2)+'\n')
        return result,value,key

    def pretrain_references(self):
        """BC uses actual actuator labels; held-out clips never supply gradients."""
        d=self.cfg.demonstrations
        if self.demonstrations is None or self.reference_pretrain_updates>=d.pretrain_updates:
            return
        for group in self.actor_opt.param_groups:
            # Keep the legacy goal-skip branch unamplified; Matrix velocity
            # feedback has a separate, explicitly configured learning time scale.
            group["lr"]=d.pretrain_lr*group.get('bc_lr_scale',1.)
        best=float("inf")
        best_weights=None
        best_optimizer=None
        best_physical=-float("inf")
        best_physical_key=None
        if d.physical_selection_interval:
            # A new correction round must compete with the accepted incumbent,
            # rather than silently choosing the least bad new BC candidate.
            initial,best_physical,best_physical_key=self.evaluate_reference_validation()
            best_weights=copy.deepcopy(self.actor.state_dict())
            best_optimizer=copy.deepcopy(self.actor_opt.state_dict())
            self.reference_selection=dict(update=self.reference_pretrain_updates,
                criterion='heldout_clip_physical_rollout',score=best_physical,
                success_rate=initial['final_held_success_rate'],failure_rate=initial['failure_rate'])
        with (self.output/"reference_pretrain.jsonl").open("a") as log:
            while self.reference_pretrain_updates<d.pretrain_updates and not self.stop_requested:
                loss=self.demonstrations.loss(self.actor,self.task,d.batch_size,noise_std=d.observation_noise_std)
                if d.hold_weight:
                    loss=loss+d.hold_weight*self.demonstrations.hold_loss(self.actor,self.task,d.batch_size)
                self.actor_opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(),self.cfg.policy.grad_clip,error_if_nonfinite=True)
                self.actor_opt.step()
                self.reference_pretrain_updates+=1
                if self.reference_pretrain_updates%d.validation_interval==0:
                    with torch.no_grad():
                        # Fixed full validation clips: selection has no minibatch noise.
                        validation=self.demonstrations.validation_loss(self.actor,self.task)
                        if not d.physical_selection_interval and float(validation)<best:
                            best=float(validation)
                            best_weights=copy.deepcopy(self.actor.state_dict())
                            best_optimizer=copy.deepcopy(self.actor_opt.state_dict())
                            self.reference_selection={"update":self.reference_pretrain_updates,"criterion":"heldout_clip_action_loss","score":best}
                    record={"reference_pretrain_updates":self.reference_pretrain_updates,
                            "bc_train_loss":float(loss.detach()),"bc_validation_loss":float(validation)}
                    line=json.dumps(record,allow_nan=False)
                    print(line,flush=True);log.write(line+"\n");log.flush()
                if d.physical_selection_interval and self.reference_pretrain_updates%d.physical_selection_interval==0:
                    evaluation,score,key=self.evaluate_reference_validation()
                    print(json.dumps({"reference_pretrain_updates":self.reference_pretrain_updates,
                        "reference_validation_held_success":evaluation["final_held_success_rate"],
                        "reference_validation_failure":evaluation["failure_rate"],
                        "task_validation_success":evaluation.get('task_validation',{}).get('final_held_success_rate')}),flush=True)
                    if best_physical_key is None or key>best_physical_key:
                        best_physical_key=key
                        best_physical=score
                        best_weights=copy.deepcopy(self.actor.state_dict())
                        best_optimizer=copy.deepcopy(self.actor_opt.state_dict())
                        self.reference_selection={"update":self.reference_pretrain_updates,"criterion":"heldout_clip_physical_rollout","score":score,
                            "success_rate":evaluation['final_held_success_rate'],"failure_rate":evaluation['failure_rate']}
        if best_weights is not None:
            self.actor.load_state_dict(best_weights)
            self.actor_opt.load_state_dict(best_optimizer)
        # The supervised policy becomes the baseline for subsequent physical checks.
        if self.guard is not None:
            self.latest.update(self.check_guard(rebase=True))
        self.save("ckpt_reference.pt")

    def pretrain_reference_model(self):
        """Optional simulator-transition bootstrap; online fitting continues later."""
        d=self.cfg.demonstrations;c=self.cfg.model
        if self.demonstrations is None:return
        with (self.output/'reference_world_model.jsonl').open('a') as log:
            while self.reference_model_pretrain_updates<d.model_pretrain_updates and not self.stop_requested:
                batch=self.demonstrations.sample_model(c.batch_size,c.window)
                if self.buffer.has_windows(c.window):
                    # Preserve identified action effects while extending the
                    # state distribution to a new reference library.
                    n=c.batch_size//2
                    old=self.buffer.sample_windows(n,c.window)
                    batch=ModelBatch(torch.cat((batch.states[:,:c.batch_size-n],old.states),1),
                        torch.cat((batch.actions[:,:c.batch_size-n],old.actions),1),
                        torch.cat((batch.xi[:c.batch_size-n],old.xi)),
                        torch.cat((batch.failed[:,:c.batch_size-n],old.failed),1))
                batch=self.mix_excitation(batch,c.window)
                loss,terms=self.wm.window_loss(batch)
                self.wm_opt.zero_grad(set_to_none=True);loss.backward()
                torch.nn.utils.clip_grad_norm_(self.wm.parameters(),c.grad_clip,error_if_nonfinite=True)
                self.wm_opt.step();self.reference_model_pretrain_updates+=1;self.model_updates+=1
                if self.reference_model_pretrain_updates%100==0:
                    val=self.wm.validation(self.demonstrations.sample_model(c.batch_size,c.window,True))
                    self.validation_error=float(val[-1])
                    record={'reference_model_updates':self.reference_model_pretrain_updates,
                        'state_loss':float(terms['state_loss']),'validation_error_k':self.validation_error}
                    line=json.dumps(record,allow_nan=False);print(line,flush=True);log.write(line+'\n');log.flush()

    def run(self):
        self.pretrain_reference_model()
        if not self.cfg.model_only:
            self.pretrain_references()
        if self.cfg.keep_actor_checkpoints and not (self.output/f'actor_{self.iteration:07d}.pt').exists():
            self.save_actor_checkpoint()
        if self.cfg.compile_fk and self.runtime_optimization is None:
            from .runtime import compile_fk
            path=self.output/'fk_compilation.json'
            path.write_text(json.dumps(dict(stage='compiling',iteration=self.iteration))+'\n')
            print(json.dumps(dict(stage='compiling_pure_fk',iteration=self.iteration)),flush=True)
            self.runtime_optimization=compile_fk(self.task,self.state(),
                (self.cfg.sim.num_envs,self.cfg.policy.batch_size))
            path.write_text(json.dumps(self.runtime_optimization,indent=2)+'\n')
            print(json.dumps(self.runtime_optimization),flush=True)
        if self.cfg.compile_clearance and self.clearance_optimization is None:
            from .runtime import compile_clearance
            path=self.output/'clearance_compilation.json'
            path.write_text(json.dumps(dict(stage='compiling',iteration=self.iteration))+'\n')
            compiled=compile_clearance(self.task,self.state(),
                (self.cfg.sim.num_envs,self.cfg.policy.batch_size))
            self.clearance_optimization=compiled
            path.write_text(json.dumps(compiled,indent=2)+'\n')
            print(json.dumps(dict(clearance_compilation=compiled)),flush=True)
        started = time.monotonic()
        run_end = min(self.cfg.iterations, self.iteration+self.cfg.run_steps) if self.cfg.run_steps else self.cfg.iterations
        with (self.output / "metrics.jsonl").open("a") as log:
            while (self.iteration < run_end and not self.stop_requested
                   and (self.cfg.actor_update_limit is None or self.actor_updates<self.cfg.actor_update_limit)):
                metrics = self.collect()
                self.iteration += 1
                if self.iteration >= self.cfg.warmup_steps and self.iteration % self.cfg.update_every == 0:
                    for optimizer, base_lr in ((self.wm_opt, self.cfg.model.lr),
                                               (self.actor_opt, self.cfg.policy.lr),
                                               (self.critic_opt, self.cfg.policy.critic_lr)):
                        lr = cosine_lr(self.iteration, self.cfg.iterations, base_lr, final_ratio=0.1)
                        if optimizer is self.actor_opt and self.guard is not None:
                            lr *= self.guard.learning_rate_scale
                        for group in optimizer.param_groups:
                            group["lr"] = lr*group.get("lr_scale",1.0)
                    self.latest.update(self.update_model())
                    self.latest.update(self.update_policy())
                if self.guard is not None and self.iteration % self.cfg.guard.interval == 0:
                    self.latest.update(self.check_guard())
                if self.iteration % self.cfg.curriculum.eval_interval == 0:
                    self.latest.update(self.check_curriculum())
                if (self.iteration % self.cfg.log_interval == 0 or self.iteration == run_end
                    or (self.cfg.actor_update_limit is not None and self.actor_updates>=self.cfg.actor_update_limit)):
                    record = {"iteration": self.iteration, "elapsed_s": time.monotonic()-started,
                              "model_updates": self.model_updates, "actor_updates": self.actor_updates,
                              "accepted_actor_updates": self.accepted_actor_updates,
                              "curriculum_level": self.level, "completed_episodes": self.completed["episodes"],
                              "completed_goal_switches":self.completed.get('goal_switches',0),
                              "failed_episodes": self.completed["failed"], **metrics, **self.latest}
                    record.update(actor_lr=self.actor_opt.param_groups[0]["lr"],
                                  goal_skip_lr=self.actor_opt.param_groups[-1 if hasattr(self.actor,'goal_skip') else 0]["lr"],
                                  critic_lr=self.critic_opt.param_groups[0]["lr"])
                    if self.cfg.policy.architecture=='matrix' and self.cfg.policy.feedback_lr_scale!=1.:
                        record['feedback_lr']=self.actor_opt.param_groups[-1]['lr']
                    if self.cfg.policy.architecture=='goal_program':
                        record.update(program_prior_lr=self.actor_opt.param_groups[0]['lr'],
                                      program_feedback_lr=self.actor_opt.param_groups[1]['lr'])
                        if self.cfg.policy.increment_feedback:
                            record['program_increment_feedback_lr']=self.actor_opt.param_groups[2]['lr']
                    line = json.dumps(record, allow_nan=False)
                    print(line, flush=True)
                    log.write(line + "\n")
                    log.flush()
                if self.iteration % self.cfg.checkpoint_interval == 0:
                    if self.guard is not None and self.guard_verified_iteration != self.iteration:
                        self.latest.update(self.check_guard())
                    self.save()
                    if self.cfg.keep_actor_checkpoints:
                        self.save_actor_checkpoint()
            if self.guard is not None and self.guard_verified_iteration != self.iteration:
                if self.stop_requested:
                    # Restore the known accepted policy without starting a long
                    # new physical evaluation after the user presses Ctrl+C.
                    saved = self.guard.state_dict()["weights"]
                    self.actor.load_state_dict(saved["actor"])
                    self.actor_opt.load_state_dict(saved["actor_opt"])
                    if self.cfg.guard.rollback_critic:
                        self.critic.load_state_dict(saved["critic"])
                        self.critic_opt.load_state_dict(saved["critic_opt"])
                    self.pending_actor_updates = 0
                else:
                    self.latest.update(self.check_guard())
            self.save()
            if self.cfg.keep_actor_checkpoints and not (self.output/f'actor_{self.iteration:07d}.pt').exists():
                self.save_actor_checkpoint()
        return {"iteration": self.iteration, "model_updates": self.model_updates,
                "actor_updates": self.actor_updates, "checkpoint": str(self.output / "ckpt_latest.pt"),
                "physical_guard_enabled": self.guard is not None,
                "guard_verified_iteration": self.guard_verified_iteration,
                "stopped_by_request": self.stop_requested,
                "validation_status": "pipeline run; physical task success must be evaluated separately"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="JSON config with nested sim/model/policy/curriculum groups")
    parser.add_argument("--smoke", action="store_true", help="small pipeline check, not a trained policy")
    parser.add_argument("--output")
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--run-steps", type=int, help="steps in this invocation; keep full learning-rate schedule")
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--device")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--resume")
    parser.add_argument("--warm-start", help="reuse learned dynamics/online replay, reset Actor/Critic/optimizers")
    parser.add_argument("--asset-dir")
    parser.add_argument("--model-only",action="store_true",default=None,help="collect independent excitation and fit dynamics without Actor updates")
    args = parser.parse_args()
    if args.config and args.smoke:
        parser.error("select --config or --smoke")
    if args.resume and args.warm_start:
        parser.error("select --resume or --warm-start")
    default_config = Path(__file__).resolve().parents[3] / "configs/wf_goal.json"
    if args.config:
        cfg = config_from_dict(json.loads(Path(args.config).read_text()))
    elif args.smoke:
        cfg = smoke_config()
    elif default_config.exists():
        cfg = config_from_dict(json.loads(default_config.read_text()))
    else:
        cfg = TrainConfig()
    for key in ("output", "iterations", "run_steps", "device", "seed", "resume", "warm_start", "asset_dir", "model_only"):
        if getattr(args, key) is not None:
            setattr(cfg, key, getattr(args, key))
    if args.resume:
        cfg.warm_start = None
    if args.warm_start:
        cfg.resume = None
    if args.num_envs is not None:
        cfg.sim.num_envs = args.num_envs
    trainer = Trainer(cfg)
    pidfile = trainer.output / "training.pid"
    pidfile.write_text(str(os.getpid())+"\n")
    previous_sigint = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, lambda signum, frame: trainer.request_stop())
    try:
        print(json.dumps(trainer.run()))
    finally:
        signal.signal(signal.SIGINT, previous_sigint)
        trainer.sim.close()
        if pidfile.exists() and pidfile.read_text().strip() == str(os.getpid()):
            pidfile.unlink()


if __name__ == "__main__":
    main()
