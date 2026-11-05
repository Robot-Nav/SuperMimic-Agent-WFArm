"""SuperMimic mechanism: k-step learned dynamics, frozen-model actor BPTT,
analytic FK/reward heads, learned termination and lambda-return critic.

This is a robot-specific adaptation, not a claim to reproduce paper results.
"""
import torch
from torch import nn
from torch.nn import functional as F

from .task import advance_goal


def mlp(inputs, hidden, outputs, zero=False):
    layers = []
    for width in hidden:
        layers.extend((nn.Linear(inputs, width), nn.SiLU())); inputs = width
    last = nn.Linear(inputs, outputs)
    if zero:
        nn.init.zeros_(last.weight); nn.init.zeros_(last.bias)
    layers.append(last)
    return nn.Sequential(*layers)


class WorldModel(nn.Module):
    def __init__(self, task, hidden=(256, 256, 256)):
        super().__init__()
        self.dt = task.dt
        self.register_buffer("center", (task.low+task.high)/2)
        self.register_buffer("scale", (task.high-task.low)/2)
        self.dynamics = mlp(18, hidden, 12, zero=True)
        self.termination = mlp(12, (64, 64), 1)
        nn.init.constant_(self.termination[-1].bias, -4.)

    def features(self, state):
        return torch.cat(((state[..., :6]-self.center)/self.scale, state[..., 6:]/5), -1)

    def step(self, state, action):
        raw = self.dynamics(torch.cat((self.features(state), (action-self.center)/self.scale), -1))
        velocity = state[..., 6:] + 3*torch.tanh(raw[..., :6])
        position = state[..., :6] + self.dt*velocity + .02*torch.tanh(raw[..., 6:])
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
        self.actor = mlp(task.input_dim, config["actor_hidden"], 6).to(self.device)
        nn.init.normal_(self.actor[-1].weight, std=.001); nn.init.zeros_(self.actor[-1].bias)
        self.wm = WorldModel(task, config["model_hidden"]).to(self.device)
        self.critic = mlp(task.input_dim, (256, 128), 1).to(self.device)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=config["actor_lr"], betas=(.9, .95))
        self.wm_opt = torch.optim.Adam(self.wm.parameters(), lr=config["model_lr"])
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=config["critic_lr"])
        self.actor_updates = self.model_updates = 0

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

    def update_actor(self, start, demonstration, horizon):
        cfg, task = self.cfg, self.task
        gamma, lam = cfg["gamma"], cfg["lambda"]
        state, previous, goal = start["state"], start["previous"], start["goal"]
        self.wm.requires_grad_(False); self.critic.requires_grad_(False)
        rewards, failures, observations = [], [], []
        survival = torch.ones(len(state), device=self.device)
        objective = torch.zeros_like(survival)
        try:
            for j in range(horizon):
                bodies = task.fk(state)
                obs = task.observation(state, previous, goal, bodies)
                observations.append(obs)
                action = task.command(self.actor, state, previous, goal, bodies)
                next_state = self.wm.step(state, action)
                next_goal = advance_goal(goal, task.dt)
                reward = task.reward(next_state, action, previous, next_goal)
                failure = self.wm.failure_probability(next_state)
                survival = survival*(1-failure)
                objective = objective + gamma**j*survival*reward
                rewards.append(reward); failures.append(failure)
                state, previous, goal = next_state, action, next_goal
            observations.append(task.observation(state, previous, goal))
            bootstrap = self.critic(observations[-1]).squeeze(-1)
            objective = objective + gamma**horizon*survival*bootstrap
            bc = self.bc_loss(demonstration)
            weight = max(cfg["bc_weight_min"], cfg["bc_weight"]*(1-self.actor_updates/cfg["actor_updates"]))
            loss = -objective.mean()/horizon + weight*bc
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
            imagined_survival=float(survival.mean().detach()), horizon=horizon)
