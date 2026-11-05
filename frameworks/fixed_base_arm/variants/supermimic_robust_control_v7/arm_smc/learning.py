"""SuperMimic mechanism: k-step learned dynamics, frozen-model actor BPTT,
analytic FK/reward heads, learned termination and lambda-return critic.

This is a robot-specific adaptation, not a claim to reproduce paper results.
"""
import torch
import copy
from torch import nn
from torch.nn import functional as F

from .task import advance_goal
from .rotations import rot6d

RESIDUAL_ARCH="frozen_base_with_local_residual"
GLOBAL_ARCH="frozen_v4_with_global_reach_residual"
GOAL_ARCH="frozen_v4_with_learned_goal_representation"
STRUCTURED_ARCH="structured_taskspace_supermimic"


def mlp(inputs, hidden, outputs, zero=False):
    layers = []
    for width in hidden:
        layers.extend((nn.Linear(inputs, width), nn.SiLU())); inputs = width
    last = nn.Linear(inputs, outputs)
    if zero:
        nn.init.zeros_(last.weight); nn.init.zeros_(last.bias)
    layers.append(last)
    return nn.Sequential(*layers)


class ResidualFeedbackActor(nn.Module):
    """Frozen six-rate base plus a learned correction with compact support.

    The correction is exactly zero outside the configured TCP error region.
    No Jacobian or reference joint lookup is used by this deployed network.
    """
    outputs_rates=True

    def __init__(self,inputs,config):
        super().__init__()
        if inputs!=42:raise ValueError("residual gate expects the fixed-arm42 input convention")
        self.base=mlp(inputs,config["actor_hidden"],6).eval().requires_grad_(False)
        self.needs_kinematic_features=config.get("residual_use_kinematic_features",False)
        self.include_source_rate=config.get("residual_include_source_rate",False)
        correction_inputs=inputs+(24 if self.needs_kinematic_features else 0)+(6 if self.include_source_rate else 0)
        self.correction=mlp(correction_inputs,config.get("residual_hidden",[128,128]),6,zero=True)
        self.position_radius=config.get("residual_position_radius",.15)
        self.rotation_radius=config.get("residual_rotation_radius",.8)
        self.rate_scale=config.get("residual_rate_scale",.5)
        if min(self.position_radius,self.rotation_radius,self.rate_scale)<=0:
            raise ValueError("positive residual support and rate scale required")
        self.register_buffer("identity_rotation",rot6d(torch.tensor([[1.,0.,0.,0.]]))[0])

    def gate(self,observation):
        position=(observation[...,18:21]*(.15/self.position_radius)).square().sum(-1)
        rotation=((observation[...,21:27]-self.identity_rotation)/self.rotation_radius).square().sum(-1)
        return ((1-position).clamp_min(0).square()*(1-rotation).clamp_min(0).square())[...,None]

    def forward(self,observation,kinematic_features=None):
        base=torch.tanh(self.base(observation))
        if self.needs_kinematic_features:
            if kinematic_features is None:raise ValueError("kinematic correction requires24 derived error features")
            inputs=torch.cat((observation,kinematic_features),-1)
        else:inputs=observation
        if self.include_source_rate:inputs=torch.cat((inputs,base),-1)
        correction=self.rate_scale*self.gate(observation)*torch.tanh(self.correction(inputs))
        return (base+correction).clamp(-1,1)


class GlobalReachActor(nn.Module):
    """Preserve the complete v4 arm policy; learn only a global correction.

    The new branch has an exact dead zone around precise TCP tracking, and
    smoothly reaches full strength for large position or orientation errors.
    Source weights are frozen, while source state derivatives remain available
    to learned-world-model BPTT. No reference joints are available at runtime.
    """
    outputs_rates=True
    needs_kinematic_features=True

    def __init__(self,inputs,config):
        super().__init__()
        self.source=ResidualFeedbackActor(inputs,config).eval().requires_grad_(False)
        self.global_correction=mlp(inputs+24+6,config.get("global_hidden",[256,256,128]),6,zero=True)
        self.position_inner=config.get("global_position_inner",.02)
        self.position_outer=config.get("global_position_outer",.15)
        self.rotation_inner=config.get("global_rotation_inner",.1)
        self.rotation_outer=config.get("global_rotation_outer",.8)
        self.rate_scale=config.get("global_rate_scale",2.)
        if not (0<=self.position_inner<self.position_outer and
                0<=self.rotation_inner<self.rotation_outer and 0<self.rate_scale<=2):
            raise ValueError("invalid global correction gate/rate bounds")

    def gate(self,observation):
        position=(observation[...,18:21]*.15).square().sum(-1)
        rotation=(observation[...,21:27]-self.source.identity_rotation).square().sum(-1)
        def transition(squared,inner,outer):
            x=((squared-inner**2)/(outer**2-inner**2)).clamp(0,1)
            return x.square()*(3-2*x)
        p=transition(position,self.position_inner,self.position_outer)
        r=transition(rotation,self.rotation_inner,self.rotation_outer)
        return (1-(1-p)*(1-r))[...,None]

    def forward(self,observation,kinematic_features):
        source=self.source(observation,kinematic_features)
        inputs=torch.cat((observation,kinematic_features,source),-1)
        correction=self.rate_scale*self.gate(observation)*torch.tanh(self.global_correction(inputs))
        return (source+correction).clamp(-1,1)


class GoalRepresentationActor(GlobalReachActor):
    """Learned pose representation for coarse reaching; retained fine tracking.

    Multiple FK-supervised neural heads represent possible joint solutions.
    One forward FK comparison selects a representation; there is no iterative
    inverse solve. SuperMimic trains the six-rate correction around a bounded
    joint-error initialization. The complete v4 policy is used in the inner
    tracking region, with smooth blending outside it.
    """
    needs_goal_representation=True

    def __init__(self,inputs,config):
        super().__init__(inputs,config)
        self.goal_heads=config.get("goal_heads",4)
        self.goal_encoder=mlp(9,config.get("goal_encoder_hidden",[512,512,256]),self.goal_heads*6)
        self.goal_encoder.add_module(str(len(self.goal_encoder)),nn.Tanh())
        self.goal_encoder.eval().requires_grad_(False)
        self.global_correction=mlp(inputs+24+6+12,config.get("global_hidden",[256,256,128]),6,zero=True)
        low=torch.tensor(config["goal_joint_low"],dtype=torch.float32)
        high=torch.tensor(config["goal_joint_high"],dtype=torch.float32)
        self.register_buffer("joint_scale",(high-low)/2)
        self.coarse_rate_scale=config.get("goal_coarse_rate_scale",.65)
        self.coarse_position_scale=config.get("goal_coarse_position_scale",.12)
        if not (0<self.coarse_rate_scale<=1 and self.coarse_position_scale>0):
            raise ValueError("invalid bounded joint-error initialization")

    def forward(self,observation,kinematic_features,goal_features):
        source=self.source(observation,kinematic_features)
        inputs=torch.cat((observation,kinematic_features,source,goal_features),-1)
        coarse=self.coarse_rate_scale*torch.tanh(
            (goal_features[...,:6]-observation[...,12:18])*self.joint_scale/self.coarse_position_scale)
        correction=self.rate_scale*torch.tanh(self.global_correction(inputs))
        global_rate=(coarse+correction).clamp(-1,1)
        gate=self.gate(observation)
        return ((1-gate)*source+gate*global_rate).clamp(-1,1)


class StructuredFeedbackActor(nn.Module):
    """Six learned-rate outputs with bounded, positive task feedback gains.

    SuperMimic trains the gain network through learned-dynamics BPTT. Native
    kinematics supply a damped task-space direction; the retained SuperMimic
    policy supplies posture motion. A planner may supply a generated joint
    reference for large reaches. It is never a recorded reference lookup.
    """
    outputs_rates=True
    needs_feedback_context=True

    def __init__(self,inputs,config):
        super().__init__()
        self.source=GlobalReachActor(inputs,config).eval().requires_grad_(False)
        self.goal_heads=config.get("goal_heads",4)
        self.goal_encoder=mlp(9,config.get("goal_encoder_hidden",[512,512,256]),self.goal_heads*6)
        self.goal_encoder.add_module(str(len(self.goal_encoder)),nn.Tanh())
        self.goal_encoder.eval().requires_grad_(False)
        self.independent_velocity_gains=config.get("independent_velocity_gains",False)
        self.gains=mlp(inputs+24+6,config.get("feedback_hidden",[128,128]),12 if self.independent_velocity_gains else 6,zero=True)
        self.gain_scale=config.get("feedback_gain_scale",.3)
        self.damping=config.get("feedback_damping",.05)
        self.speed=config["joint_target_speed"]
        self.register_buffer("weights",torch.tensor([5.,5.,5.,2.,2.,2.]))
        self.register_buffer("identity",torch.eye(6))
        if not 0<self.gain_scale<1:raise ValueError("feedback gains must remain strictly positive")
        if not .01<=self.damping<=.15:raise ValueError("damped feedback regularization outside diagnostic range")

    def forward(self,observation,kinematic_features,jacobian,pose_error,velocity_error,goal_twist,planned_nominal,planned_mask):
        source=self.source(observation,kinematic_features)
        gain=1+self.gain_scale*torch.tanh(self.gains(torch.cat((observation,kinematic_features,source),-1)))
        if self.independent_velocity_gains:
            position_gain,velocity_gain=gain[...,:6],gain[...,6:]
            correction=2*position_gain*pose_error+2*velocity_gain*velocity_error
        else:correction=2*gain*(pose_error+velocity_error)
        linear,angular=correction[...,:3],correction[...,3:]
        correction=torch.cat((linear*torch.clamp(.2/linear.norm(dim=-1,keepdim=True).clamp_min(1e-8),max=1),
                              angular*torch.clamp(.8/angular.norm(dim=-1,keepdim=True).clamp_min(1e-8),max=1)),-1)
        desired=goal_twist+correction
        matrix=jacobian*self.weights[...,None]
        inverse=torch.linalg.solve(matrix.transpose(-1,-2)@matrix+self.damping**2*self.identity,
                                   matrix.transpose(-1,-2)*self.weights)
        nominal=planned_mask*planned_nominal+(1-planned_mask)*source*self.speed
        rate=(inverse@desired[...,None]).squeeze(-1)
        rate=rate+((self.identity-inverse@jacobian)@nominal[...,None]).squeeze(-1)
        def normalize(value):
            return value*torch.clamp(self.speed/value.abs().amax(-1,keepdim=True).clamp_min(1e-8),max=1)/self.speed
        fine=self.source.source.gate(observation)
        return (fine*normalize(rate)+(1-fine)*normalize(nominal)).clamp(-1,1)


class WorldModel(nn.Module):
    def __init__(self, task, hidden=(256, 256, 256), joint_limit_margin=None):
        super().__init__()
        self.dt = task.dt
        self.register_buffer("center", (task.low+task.high)/2)
        self.register_buffer("scale", (task.high-task.low)/2)
        self.joint_limit_margin=joint_limit_margin
        if joint_limit_margin is not None:
            if not 0<joint_limit_margin<=.04:
                raise ValueError("model joint-limit allowance must be in (0,.04] rad")
            # Physical URDF limits differ from the narrower wrist command
            # envelope. These constants are rebuilt from assets, not learned.
            self.register_buffer("physical_low",task.physical_low.clone(),persistent=False)
            self.register_buffer("physical_high",task.physical_high.clone(),persistent=False)
        self.dynamics = mlp(18, hidden, 12, zero=True)
        self.termination = mlp(12, (64, 64), 1)
        nn.init.constant_(self.termination[-1].bias, -4.)

    def features(self, state):
        return torch.cat(((state[..., :6]-self.center)/self.scale, state[..., 6:]/5), -1)

    def step(self, state, action):
        raw = self.dynamics(torch.cat((self.features(state), (action-self.center)/self.scale), -1))
        velocity = state[..., 6:] + 3*torch.tanh(raw[..., :6])
        position = state[..., :6] + self.dt*velocity + .02*torch.tanh(raw[..., 6:])
        if self.joint_limit_margin is not None:
            low=self.physical_low-self.joint_limit_margin
            high=self.physical_high+self.joint_limit_margin
            outward=((position>high)&(velocity>0))|((position<low)&(velocity<0))
            position=position.clamp(min=low,max=high)
            velocity=torch.where(outward,torch.zeros_like(velocity),velocity)
        return torch.cat((position, velocity), -1)

    def failure_probability(self, state):
        return torch.sigmoid(self.termination(self.features(state)).squeeze(-1))

    def loss(self, states, actions, failed):
        current = states[0]
        losses, bces = [], []
        for j in range(len(actions)):
            current = self.step(current, actions[j])
            error = current-states[j+1]
            losses.append((error[..., :6]/.15).square().mean() + error[..., 6:].square().mean())
            logits = self.termination(self.features(current)).squeeze(-1)
            bces.append(F.binary_cross_entropy_with_logits(logits, failed[j]))
        loss = torch.stack(losses).mean() + .1*torch.stack(bces).mean()
        return loss, losses[-1].detach()


class Learner:
    def __init__(self, task, config):
        self.task, self.cfg = task, config
        self.device = task.low.device
        task.success_margin_weight=config.get("success_margin_weight",0.)
        task.success_margin_fraction=config.get("success_margin_fraction",.8)
        if config.get("actor_architecture")==STRUCTURED_ARCH:
            self.actor=StructuredFeedbackActor(task.input_dim,config).to(self.device)
            self.task.joint2_command_upper=config.get("joint2_command_upper",2.98)
            self.task.joint2_braking_lookahead=config.get("joint2_braking_lookahead",.25)
        elif config.get("actor_architecture")==GOAL_ARCH:
            self.actor=GoalRepresentationActor(task.input_dim,config).to(self.device)
        elif config.get("actor_architecture")==GLOBAL_ARCH:
            self.actor=GlobalReachActor(task.input_dim,config).to(self.device)
        elif config.get("actor_architecture")==RESIDUAL_ARCH:
            self.actor=ResidualFeedbackActor(task.input_dim,config).to(self.device)
        else:
            self.actor = mlp(task.input_dim, config["actor_hidden"], 6).to(self.device)
            nn.init.normal_(self.actor[-1].weight, std=.001); nn.init.zeros_(self.actor[-1].bias)
        if config.get("train_local_correction_only",False):
            if not isinstance(self.actor,GlobalReachActor):
                raise ValueError("local-only refinement requires a retained global/source hierarchy")
            self.actor.requires_grad_(False)
            self.actor.source.correction.requires_grad_(True)
        self.wm = WorldModel(task, config["model_hidden"],config.get("model_joint_limit_margin")).to(self.device)
        self.critic = mlp(task.input_dim, (256, 128), 1).to(self.device)
        self.actor_opt = torch.optim.Adam([p for p in self.actor.parameters() if p.requires_grad], lr=config["actor_lr"], betas=(.9, .95))
        self.wm_opt = torch.optim.Adam(self.wm.parameters(), lr=config["model_lr"])
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=config["critic_lr"])
        self.actor_updates = self.model_updates = 0
        self.anchor_actor = None
        self.imagined_step = self._imagined_step
        self.policy_step = self._policy_step
        if config.get("compile_imagined_step",False):
            self.imagined_step=torch.compile(self._imagined_step,fullgraph=True,dynamic=False)
            self.policy_step=torch.compile(self._policy_step,fullgraph=True,dynamic=False)

    def set_policy_anchor(self,state_dict):
        self.anchor_actor=copy.deepcopy(self.actor).eval().requires_grad_(False)
        self.anchor_actor.load_state_dict(state_dict)

    def _policy_step(self,state,previous,goal):
        return self.task.command(self.actor,state,previous,goal)

    def inference_action(self,state,previous,goal):
        # PyTorch 2.7/Triton emits an invalid scalar broadcast for this FK at
        # batch=1. Reuse the verified batch=8 kernel. Every operation is per-row.
        if self.cfg.get("compile_imagined_step",False) and state.shape[0]==1:
            return self.policy_step(state.expand(8,-1).contiguous(),previous.expand(8,-1).contiguous(),
                                    goal.expand(8,-1).contiguous())[:1]
        return self.policy_step(state,previous,goal)

    def _imagined_step(self,state,previous,goal,next_goal):
        task=self.task
        bodies=task.fk(state)
        observation=task.observation(state,previous,goal,bodies)
        action=task.command(self.actor,state,previous,goal,bodies)
        next_state=self.wm.step(state,action)
        reward_goal=goal if self.cfg.get("aligned_reward",False) else next_goal
        reward=task.reward(next_state,action,previous,reward_goal)
        failure=self.wm.failure_probability(next_state)
        return observation,action,next_state,reward,failure

    def update_model(self, batch):
        self.wm_opt.zero_grad(set_to_none=True)
        loss, error = self.wm.loss(batch["states"], batch["actions"], batch["failed"])
        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite world-model loss")
        loss.backward(); nn.utils.clip_grad_norm_(self.wm.parameters(), 1.)
        self.wm_opt.step(); self.model_updates += 1
        return {"model_loss": float(loss.detach()), "model_error_k": float(error)}

    def bc_loss(self, batch):
        pred = self.task.command(self.actor, batch["state"], batch["previous"], batch["goal"])
        # Normalize increments, not full joint ranges, so small rate-limited
        # commands have a useful supervision gradient.
        return ((pred-batch["action"])/(self.task.dt*self.task.speed)).square().mean()

    def pretrain_actor(self, batch):
        self.actor_opt.zero_grad(set_to_none=True)
        loss = self.bc_loss(batch)
        loss.backward(); nn.utils.clip_grad_norm_(self.actor.parameters(), 1.)
        self.actor_opt.step()
        return float(loss.detach())

    def update_actor(self, start, demonstration, horizon, anchor_batch=None):
        cfg, task = self.cfg, self.task
        gamma, lam = cfg["gamma"], cfg["lambda"]
        state, previous, goal = start["state"], start["previous"], start["goal"]
        future = start.get("future_goals")
        self.wm.requires_grad_(False); self.critic.requires_grad_(False)
        rewards, failures, observations = [], [], []
        survival = torch.ones(len(state), device=self.device)
        objective = torch.zeros_like(survival)
        try:
            for j in range(horizon):
                next_goal = future[j+1] if future is not None else advance_goal(goal, task.dt)
                obs,action,next_state,reward,failure=self.imagined_step(state,previous,goal,next_goal)
                observations.append(obs)
                survival = survival*(1-failure)
                objective = objective + gamma**j*survival*reward
                rewards.append(reward); failures.append(failure)
                state, previous, goal = next_state, action, next_goal
            observations.append(task.observation(state, previous, goal))
            bootstrap = self.critic(observations[-1]).squeeze(-1)
            objective = objective + cfg.get("bootstrap_actor_weight",1.)*gamma**horizon*survival*bootstrap
            bc = self.bc_loss(demonstration)
            stage_start=(0 if cfg.get("preserve_guidance_schedule",False) else
                         cfg.get("guidance_stage_start_updates",cfg.get("stage_start_updates",0)))
            fraction=(self.actor_updates-stage_start)/max(1,cfg["actor_updates"]-stage_start)
            weight = max(cfg["bc_weight_min"], cfg["bc_weight"]*(1-fraction))
            loss = -objective.mean()/horizon + weight*bc
            deviation=loss.new_zeros(())
            deviation_weight=cfg.get("policy_deviation_weight",0.)
            if deviation_weight:
                if self.anchor_actor is None or anchor_batch is None:
                    raise RuntimeError("policy deviation penalty requires frozen source actions and training states")
                args=(anchor_batch["state"],anchor_batch["previous"],anchor_batch["goal"])
                with torch.no_grad():reference_action=task.command(self.anchor_actor,*args)
                current_action=task.command(self.actor,*args)
                deviation=((current_action-reference_action)/(task.dt*task.speed)).square().mean()
                loss=loss+deviation_weight*deviation
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite actor objective")
            self.actor_opt.zero_grad(set_to_none=True)
            loss.backward(); grad = nn.utils.clip_grad_norm_(self.actor.parameters(), 1.)
            nn.utils.clip_grad_value_(self.actor.parameters(), .1)
            self.actor_opt.step(); self.actor_updates += 1
        finally:
            self.wm.requires_grad_(True); self.critic.requires_grad_(True)
        with torch.no_grad():
            observations = torch.stack(observations).detach()
            values = self.critic(observations).squeeze(-1)
            target = values[-1]
            targets = []
            for j in reversed(range(horizon)):
                target = rewards[j].detach() + gamma*(1-failures[j].detach())*((1-lam)*values[j+1]+lam*target)
                targets.append(target)
            targets = torch.stack(list(reversed(targets)))
            inputs = observations[:-1].reshape(-1, task.input_dim)
            targets = targets.reshape(-1)
        for _ in range(cfg["critic_epochs"]):
            self.critic_opt.zero_grad(set_to_none=True)
            closs = F.mse_loss(self.critic(inputs).squeeze(-1), targets)
            closs.backward(); nn.utils.clip_grad_norm_(self.critic.parameters(), 1.)
            self.critic_opt.step()
        return dict(actor_loss=float(loss.detach()), actor_grad_norm=float(grad),
            bc_loss=float(bc.detach()), bc_weight=weight, critic_loss=float(closs.detach()),
            policy_deviation_loss=float(deviation.detach()),policy_deviation_weight=deviation_weight,
            imagined_survival=float(survival.mean().detach()), horizon=horizon)
