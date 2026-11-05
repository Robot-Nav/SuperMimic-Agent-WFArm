"""Randomly initialized, unified deterministic Actor and privileged Critic."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from supermimic.world_model.dynamics import MLP


class Actor(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # The default path retains the original architecture and floating-point
        # operations, so checkpoints predating direct conditioning still load
        # and produce exactly the same outputs.
        self.goal_conditioning = getattr(cfg, "goal_conditioning", "encoded")
        self.goal_position_scale = float(getattr(cfg, "goal_position_scale", .1))
        self.ignore_previous_action = getattr(cfg,"ignore_previous_action",False)
        self.action_mode=getattr(cfg,"action_mode","absolute")
        self.action_delta_limit=float(getattr(cfg,"action_delta_limit",.01))
        self.target_supervision_scale=getattr(cfg,'target_supervision_scale','absolute')
        self.architecture=getattr(cfg,'architecture','mlp')
        self.constraint_observation=getattr(cfg,'constraint_observation',False)
        self.obs_dim=76 if self.constraint_observation else 72
        self.base_command_conditioning=getattr(cfg,'base_command_conditioning',False)
        self.program_context=getattr(cfg,'program_context',False)
        self.has_increment_feedback=getattr(cfg,'increment_feedback',False)
        self.goal_dim=12+6*int(getattr(cfg,'full_goal_orientation',False))+4*int(self.base_command_conditioning)+int(self.architecture=='goal_program')+27*int(self.program_context)
        self.gain_bound=float(getattr(cfg,'gain_bound',0.))
        self.has_operating_bounds=bool(getattr(cfg,'action_lower',()))
        if self.has_operating_bounds:
            self.register_buffer('operating_lower',torch.tensor(cfg.action_lower,dtype=torch.float32),persistent=False)
            self.register_buffer('operating_upper',torch.tensor(cfg.action_upper,dtype=torch.float32),persistent=False)
        if self.goal_conditioning not in ("encoded", "direct"):
            raise ValueError("goal_conditioning must be 'encoded' or 'direct'")
        if not math.isfinite(self.goal_position_scale) or self.goal_position_scale <= 0:
            raise ValueError("goal_position_scale must be finite and positive")
        if self.architecture=='goal_program':
            if not cfg.full_goal_orientation or not cfg.base_command_conditioning or cfg.action_mode!='target':
                raise ValueError('goal program requires full pose, commands and target actions')
            self.register_buffer('goal_identity',torch.tensor([1.,0.,0.,0.,1.,0.]),persistent=False)
            self.prior=MLP(41 if self.program_context else 14,cfg.hidden,22,last_gain=.01)
            self.feedback=MLP(self.obs_dim+self.goal_dim,cfg.feedback_hidden,22,zero_last=True)
            self.feedback_output_scale=cfg.feedback_output_scale
            self.feedback_start_phase=cfg.feedback_start_phase
            self.feedback_full_phase=cfg.feedback_full_phase
            if self.has_increment_feedback:
                self.increment_feedback=MLP(self.obs_dim+self.goal_dim,cfg.increment_feedback_hidden,22,zero_last=True)
                self.increment_feedback_scale=cfg.increment_feedback_scale
            return
        if self.architecture=='matrix':
            self.register_buffer('goal_identity',torch.tensor([1.,0.,0.,0.,1.,0.]),persistent=False)
            self.signal_dim=19 if self.constraint_observation else 9
            if self.base_command_conditioning:self.signal_dim+=3
            self.gain=MLP(self.obs_dim+self.goal_dim,cfg.hidden,22*self.signal_dim,last_gain=.01)
            self.log_damping=nn.Parameter(torch.full((22,),float(getattr(cfg,'log_damping_init',-2.))))
            self.log_wheel_stop=nn.Parameter(torch.zeros(4))
            return
        self.encoder = MLP(self.goal_dim, (128, 64), cfg.latent_dim)
        direct = self.goal_conditioning == "direct"
        self.decoder = MLP(self.obs_dim + cfg.latent_dim + (self.goal_dim if direct else 0), cfg.hidden, 22, last_gain=0.01)
        if direct:
            self.register_buffer("goal_identity", torch.tensor([1., 0., 0., 0., 1., 0.]), persistent=False)
            # A learned dense projection to every motor makes a small target
            # change visible from initialization. This is not an IK Jacobian or
            # a prescribed split between legs, wheels and arm control.
            self.goal_skip = nn.Linear(9, 22, bias=False)
            nn.init.normal_(self.goal_skip.weight, mean=0., std=.001)

    def normalize_goal(self, goal_features):
        if self.goal_conditioning == "encoded":
            return goal_features
        return torch.cat((goal_features[..., :3]/self.goal_position_scale,
                          goal_features[..., 3:9]-self.goal_identity,
                          goal_features[..., 9:]), dim=-1)

    def operating_action(self,action):
        if self.has_operating_bounds:
            return torch.minimum(torch.maximum(action,self.operating_lower),self.operating_upper)
        return action

    def forward(self, observation, goal_features=None, supervise=False):
        if goal_features is None:
            observation, goal_features = observation[..., :self.obs_dim], observation[..., self.obs_dim:]
        previous=observation[...,self.obs_dim-22:self.obs_dim]
        if self.ignore_previous_action:
            # Avoid the trivial previous-command copying shortcut in small-data
            # imitation. The critic still sees it for the action-rate reward.
            observation=torch.cat((observation[...,:self.obs_dim-22],torch.zeros_like(previous)),dim=-1)
        if self.architecture=='goal_program':
            if self.program_context:
                context=torch.cat((goal_features[...,9:18],goal_features[...,19:46],goal_features[...,-4:],goal_features[...,18:19]),-1)
            else:
                context=torch.cat((goal_features[...,9:18],goal_features[...,-4:],goal_features[...,18:19]),-1)
            prior=torch.tanh(self.prior(context))
            if supervise:return prior
            correction=self.feedback_output_scale*torch.tanh(self.feedback(torch.cat((observation,self.normalize_goal(goal_features)),-1)))
            if self.feedback_full_phase>self.feedback_start_phase:
                gate=((goal_features[...,18:19]-self.feedback_start_phase)/(self.feedback_full_phase-self.feedback_start_phase)).clamp(0.,1.)
                gate=gate.square()*(3.-2.*gate)
                correction=gate*correction
            desired=self.operating_action(prior+correction)
            delta=(desired-previous).clamp(-self.action_delta_limit,self.action_delta_limit)
            if self.has_increment_feedback:
                # Learn a joint action increment AFTER the prior's saturated
                # target projection. At zero initialization forward commands
                # exactly match the source policy, while this correction has
                # a local gradient where the prior target gradient is blocked.
                # The final rate and operating projections remain physical.
                residual=self.action_delta_limit*self.increment_feedback_scale*torch.tanh(
                    self.increment_feedback(torch.cat((observation,self.normalize_goal(goal_features)),-1)))
                delta=(delta+residual).clamp(-self.action_delta_limit,self.action_delta_limit)
            return self.operating_action(previous+delta)
        if self.architecture=='matrix':
            goal=self.normalize_goal(goal_features)
            gain=self.gain(torch.cat((observation,goal),dim=-1)).reshape(*observation.shape[:-1],22,self.signal_dim)
            if self.gain_bound:gain=self.gain_bound*torch.tanh(gain/self.gain_bound)
            signal=goal[...,:9]
            if self.constraint_observation:
                signal=torch.cat((signal,observation[...,40:43]/.2,observation[...,43:46]*2,observation[...,50:54]),dim=-1)
            if self.base_command_conditioning:
                actual=torch.stack((observation[...,40]/.2,observation[...,41]/.2,observation[...,45]*5),-1)
                command_error=goal[...,-1:]*(goal[...,-4:-1]-actual)
                signal=torch.cat((signal,command_error),-1)
            raw=(gain*signal.unsqueeze(-2)).sum(-1)
            raw=raw-F.softplus(self.log_damping)*observation[...,18:40]
            stop=torch.cat((torch.zeros_like(previous[...,:12]),F.softplus(self.log_wheel_stop)*previous[...,12:16],
                            torch.zeros_like(previous[...,16:])),dim=-1)
            return self.operating_action((previous+self.action_delta_limit*torch.tanh(raw-stop)).clamp(-1,1))
        if self.goal_conditioning == "direct":
            goal = self.normalize_goal(goal_features)
            z = torch.tanh(self.encoder(goal))
            raw = self.decoder(torch.cat((observation, z, goal), dim=-1))
            out=torch.tanh(raw+self.goal_skip(goal[..., :9]))
        else:
            z = torch.tanh(self.encoder(goal_features))
            out=torch.tanh(self.decoder(torch.cat([observation, z], dim=-1)))
        if self.action_mode=="delta":
            return self.operating_action((previous+self.action_delta_limit*out).clamp(-1,1))
        if self.action_mode=='target':
            if self.architecture=='residual_target':
                # Position feedback is a feature/skip inside the ONE Actor.
                # The network jointly learns every leg, wheel and arm residual;
                # no kinematic solver or reference path exists at deployment.
                current=torch.cat((observation[...,:12],torch.zeros_like(out[...,12:16]),observation[...,12:18]),-1)
                scale=out.new_tensor([.5]*12+[.08602150537634409]*4+[.2]*6)
                desired=current+scale*out
            else:
                desired=out
            if supervise:return desired
            desired=self.operating_action(desired)
            return self.operating_action(previous+(desired-previous).clamp(-self.action_delta_limit,self.action_delta_limit))
        return self.operating_action(out)

    def supervised_action(self, inputs):
        """Fit desired targets before a target-mode slew limiter saturates."""
        return self.forward(inputs,supervise=True)


class Critic(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # Includes prior action, so action-rate reward is Markov; context is privileged.
        goal_dim=12+6*int(getattr(cfg,'full_goal_orientation',False))+4*int(getattr(cfg,'base_command_conditioning',False))+int(cfg.architecture=='goal_program')+27*int(getattr(cfg,'program_context',False))
        self.net = MLP((76 if getattr(cfg,'constraint_observation',False) else 72)+goal_dim+1, cfg.critic_hidden, 1)

    def forward(self, actor_input, xi):
        return self.net(torch.cat([actor_input, (xi - 0.9) / 0.3], dim=-1)).squeeze(-1)


def load_command_upgrade(actor, source_state):
    """Add zero-initialized command lanes while preserving the legacy Actor.

    This upgrades a single full-body network; it does not add a base controller.
    Autonomy-mode outputs must match the source before further training.
    """
    if not actor.base_command_conditioning or actor.architecture!='matrix':
        raise ValueError('command upgrade requires a command-conditioned Matrix Actor')
    target=actor.state_dict()
    for name,value in source_state.items():
        if target[name].shape==value.shape:
            target[name]=value
        elif name.startswith('gain.') and value.ndim==2 and target[name].shape[0]==value.shape[0]:
            target[name].zero_();target[name][:,:value.shape[1]]=value
        elif name.startswith('gain.') and value.shape[0]%22==0:
            old_signals=value.shape[0]//22
            shape=(22,actor.signal_dim)+tuple(value.shape[1:])
            out=target[name].reshape(shape);out.zero_()
            out[:,:old_signals]=value.reshape((22,old_signals)+tuple(value.shape[1:]))
        else:
            raise ValueError(f'unsupported Actor upgrade parameter: {name}')
    actor.load_state_dict(target)


def load_goal_program_prior(actor,checkpoint):
    """Load the learned whole-body goal/phase network; feedback starts at zero."""
    if actor.architecture!='goal_program' or checkpoint.get('format')!='wf_phase_policy_candidate_v1' or not checkpoint.get('goal_only_input') or not checkpoint.get('phase_enabled',True):
        raise ValueError('expected a learned goal-only phase prior candidate')
    state={name.removeprefix('network.'):value for name,value in checkpoint['actor'].items() if name.startswith('network.')}
    if actor.program_context:
        expanded=actor.prior.state_dict();weight=expanded['net.0.weight'];weight.zero_()
        weight[:,:9]=state['net.0.weight'][:,:9];weight[:,36:]=state['net.0.weight'][:,9:]
        state['net.0.weight']=weight
    actor.prior.load_state_dict(state)
    if actor.has_operating_bounds:
        torch.testing.assert_close(checkpoint['actor']['lower'].to(actor.operating_lower),actor.operating_lower)
        torch.testing.assert_close(checkpoint['actor']['upper'].to(actor.operating_upper),actor.operating_upper)


def load_increment_feedback_upgrade(actor,source):
    """Retain the complete source Actor; new increment feedback stays zero."""
    if not actor.has_increment_feedback:raise ValueError('increment feedback must be enabled')
    target=actor.state_dict()
    for name,value in source.items():
        if name not in target or target[name].shape!=value.shape:raise ValueError('source Actor shape mismatch')
        target[name]=value
    missing=set(target)-set(source)
    if any(not name.startswith('increment_feedback.') for name in missing):
        raise ValueError('source Actor is missing a pre-existing parameter')
    actor.load_state_dict(target)


def load_program_context_upgrade(actor,source):
    """Add zero-weight origin/start context while retaining policy behavior."""
    target=actor.state_dict()
    for key,value in source.items():
        if target[key].shape==value.shape:target[key]=value
        elif key=='prior.net.0.weight':
            target[key].zero_();target[key][:,:9]=value[:,:9];target[key][:,36:]=value[:,9:]
        elif key=='feedback.net.0.weight':
            prefix=actor.obs_dim+19
            target[key].zero_();target[key][:,:prefix]=value[:,:prefix];target[key][:,prefix+27:]=value[:,prefix:]
        else:raise ValueError(f'unsupported program context parameter: {key}')
    actor.load_state_dict(target)
