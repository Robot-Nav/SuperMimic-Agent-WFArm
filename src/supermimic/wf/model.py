"""One learned differentiable model of the complete robot, conditioned on friction.

The residual integration / k-step fitting mechanism follows SuperMimic. Layout,
wheel-phase invariance and termination features are specific to this experiment.
"""

import torch
from torch import nn
from torch.nn import functional as F

from supermimic.world_model.dynamics import MLP
from supermimic.wf.math import (
    heading_quat, quat_apply, quat_apply_inverse, quat_conjugate, quat_from_rotvec,
    quat_mul, quat_normalize, rot6d_from_quat, rotvec_from_quat,
)


class StateLayout(nn.Module):
    state_dim, action_dim, feature_dim, delta_dim = 57, 22, 53, 56

    def __init__(self, q0, dt=0.02):
        super().__init__()
        self.dt = dt
        self.register_buffer("q0", torch.as_tensor(q0, dtype=torch.float32))
        self.register_buffer("nonwheel", torch.tensor(list(range(12)) + list(range(16, 22))))
        self.register_buffer("qd_scale", torch.tensor([5.] * 12 + [20.] * 4 + [5.] * 6))
        self.register_buffer("velocity_delta_scale", torch.tensor([2.] * 3 + [4.] * 3 + [8.] * 12 + [20.] * 4 + [8.] * 6))
        self.register_buffer("position_delta_scale", torch.tensor([0.02] * 3 + [0.05] * 3 + [0.03] * 22))

    def canonical(self, s):
        h = heading_quat(s[..., 3:7])
        q = (s[..., 7:29] - self.q0)[..., self.nonwheel]
        return torch.cat([
            s[..., 2:3], rot6d_from_quat(quat_mul(quat_conjugate(h), s[..., 3:7])),
            quat_apply_inverse(h, s[..., 29:32]), s[..., 32:35] / 2,
            q, s[..., 35:57] / self.qd_scale,
        ], dim=-1)

    def integrate(self, s, raw_delta):
        h = heading_quat(s[..., 3:7])
        dv = torch.tanh(raw_delta[..., :28]) * self.velocity_delta_scale
        dp = torch.tanh(raw_delta[..., 28:]) * self.position_delta_scale
        v = s[..., 29:32] + quat_apply(h, dv[..., :3])
        w = s[..., 32:35] + dv[..., 3:6]
        qd = s[..., 35:57] + dv[..., 6:28]
        pos = s[..., :3] + self.dt * v + quat_apply(h, dp[..., :3])
        quat = quat_normalize(quat_mul(s[..., 3:7], quat_from_rotvec(self.dt * w + dp[..., 3:6])))
        q = s[..., 7:29] + self.dt * qd + dp[..., 6:28]
        return torch.cat([pos, quat, q, v, w, qd], dim=-1)

    def error(self, predicted, target):
        # Wheel phase is unobservable to the network and task for cylindrical wheels.
        return torch.cat([
            (predicted[..., :3] - target[..., :3]) / 0.1,
            rotvec_from_quat(quat_mul(quat_conjugate(target[..., 3:7]), predicted[..., 3:7])) / 0.25,
            (predicted[..., 7:29] - target[..., 7:29])[..., self.nonwheel] / 0.3,
            predicted[..., 29:32] - target[..., 29:32],
            (predicted[..., 32:35] - target[..., 32:35]) / 2,
            (predicted[..., 35:57] - target[..., 35:57]) / self.qd_scale,
        ], dim=-1)


class WholeBodyWorldModel(nn.Module):
    def __init__(self, q0, cfg, dt=0.02):
        super().__init__()
        self.layout = StateLayout(q0, dt)
        self.dynamics = MLP(53 + 22 + 1, cfg.hidden, 56, zero_last=True)
        self.termination = MLP(53 + 1, (64, 64), 1)
        # Rare failures should not kill all initial imagined returns.
        nn.init.constant_(self.termination.net[-1].bias, -3.0)

    def step(self, state, action, xi):
        features = self.layout.canonical(state)
        delta = self.dynamics(torch.cat([features, action, (xi - 0.9) / 0.3], dim=-1))
        return self.layout.integrate(state, delta)

    def termination_logits(self, state, xi):
        return self.termination(torch.cat([self.layout.canonical(state), (xi - 0.9) / 0.3], dim=-1)).squeeze(-1)

    def window_loss(self, batch):
        state = batch.states[0]
        errors, bces = [], []
        for j in range(len(batch.actions)):
            state = self.step(state, batch.actions[j], batch.xi)
            errors.append(self.layout.error(state, batch.states[j + 1]).square().mean())
            logits = self.termination_logits(state, batch.xi)
            bces.append(F.binary_cross_entropy_with_logits(logits, batch.failed[j]))
        state_loss, term_loss = torch.stack(errors).mean(), torch.stack(bces).mean()
        return state_loss + 0.1 * term_loss, {
            "state_loss": state_loss.detach(), "termination_bce": term_loss.detach(),
            "error_1": errors[0].detach(), "error_k": errors[-1].detach(),
        }

    @torch.no_grad()
    def validation(self, batch):
        state = batch.states[0]
        errors = []
        for j in range(len(batch.actions)):
            state = self.step(state, batch.actions[j], batch.xi)
            errors.append(self.layout.error(state, batch.states[j + 1]).square().mean())
        return torch.stack(errors)
