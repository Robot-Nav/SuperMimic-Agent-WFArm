"""Transition replay with explicit next states and episode-safe model windows.

Validation environments are permanently excluded from fitting, including after
ring wrap. Time limits break windows but never become physical failure labels.
Unlike motion replay, no frame IDs, references or demonstrations are required.
"""

from dataclasses import dataclass

import torch


@dataclass
class ModelBatch:
    states: torch.Tensor  # (K+1, B, 57), including pre-reset terminal next states
    actions: torch.Tensor  # executed normalized action (K,B,22)
    xi: torch.Tensor  # (B,1), constant throughout each sampled window
    failed: torch.Tensor  # (K,B); time limits excluded


@dataclass
class PolicyBatch:
    state: torch.Tensor
    prev_action: torch.Tensor
    goal: torch.Tensor
    xi: torch.Tensor


class GoalReplayBuffer:
    def __init__(self, num_envs, capacity, device="cpu", validation_fraction=0.2, goal_size=7):
        if num_envs < 2 or capacity < 2 or not 0 < validation_fraction < 1:
            raise ValueError("replay requires >=2 environments, capacity >=2 and a validation split")
        self.E, self.T = num_envs, capacity
        self.device = torch.device(device)
        zeros = lambda *tail: torch.zeros(capacity, num_envs, *tail, device=self.device)
        self.state, self.next_state = zeros(57), zeros(57)
        self.action, self.prev_action = zeros(22), zeros(22)
        self.goal, self.xi = zeros(goal_size), zeros(1)
        self.failed = torch.zeros(capacity, num_envs, dtype=torch.bool, device=self.device)
        self.ended = torch.zeros_like(self.failed)
        self.seq = torch.full((capacity,), -1, dtype=torch.long, device=self.device)
        self.total = 0
        nval = max(1, min(num_envs - 1, round(num_envs * validation_fraction)))
        self.validation_envs = torch.arange(num_envs - nval, num_envs, device=self.device)
        self.training_envs = torch.arange(num_envs - nval, device=self.device)

    @torch.no_grad()
    def add(self, state, next_state, action, prev_action, goal, xi, failed, timeout):
        i = self.total % self.T
        for name, value in (("state", state), ("next_state", next_state), ("action", action),
                            ("prev_action", prev_action), ("goal", goal), ("xi", xi)):
            getattr(self, name)[i].copy_(value)
        self.failed[i].copy_(failed)
        self.ended[i].copy_(failed | timeout)
        self.seq[i] = self.total
        self.total += 1

    def _indices(self, n, k=1, validation=False):
        if k < 1 or k > self.T:
            raise ValueError("window must be in [1, capacity]")
        envs = self.validation_envs if validation else self.training_envs
        t = torch.arange(self.T, device=self.device)
        valid = (self.seq >= 0)[:, None].expand(self.T, len(envs)).clone()
        for j in range(1, k):
            idx = (t + j) % self.T
            valid &= (self.seq[idx] == self.seq + j)[:, None]
            valid &= ~self.ended[(t + j - 1) % self.T][:, envs]
        candidates = valid.nonzero()
        if not candidates.numel():
            raise RuntimeError(f"no {'validation' if validation else 'training'} windows of length {k}")
        rows = candidates[torch.randint(len(candidates), (n,), device=self.device)]
        return rows[:, 0], envs[rows[:, 1]]

    def has_windows(self, k, validation=False):
        try:
            self._indices(1, k, validation)
        except RuntimeError:
            return False
        return True

    def sample_windows(self, n, k, validation=False):
        t, e = self._indices(n, k, validation)
        tt = (t[None] + torch.arange(k, device=self.device)[:, None]) % self.T
        return ModelBatch(
            torch.cat([self.state[t, e][None], self.next_state[tt, e[None]]], dim=0),
            self.action[tt, e[None]], self.xi[t, e], self.failed[tt, e[None]].float(),
        )

    def sample_starts(self, n, validation=False):
        t, e = self._indices(n, validation=validation)
        return PolicyBatch(self.state[t, e], self.prev_action[t, e], self.goal[t, e], self.xi[t, e])
