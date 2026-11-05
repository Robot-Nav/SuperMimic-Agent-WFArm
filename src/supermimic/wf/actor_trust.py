"""Backtrack a unified Actor's parameter step before real-physics acceptance.

No controller, IK solution, demonstration or action-space routing is introduced:
every candidate is the same learned Actor with interpolated network weights.
The trainer evaluates these candidates on its fixed training-validation goals;
this helper neither runs physics nor decides that a policy solves the task.
"""

from __future__ import annotations

import copy
import math

import torch


def clone_state(value):
    """Independent detached CPU copy of nested module/optimizer state."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: clone_state(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clone_state(item) for item in value]
    if isinstance(value, tuple):
        return tuple(clone_state(item) for item in value)
    return copy.deepcopy(value)


class ActorTrustStep:
    """Immutable endpoints for weight-space line search of a proposed update.

    ``state_at(alpha) = accepted + alpha * (candidate - accepted)``. Trials
    always use these original endpoints, so trying alpha=.25 after alpha=.5
    does not compound the interpolation. For a partial step, Adam's moments and
    counters are restored from the accepted snapshot rather than interpolated.
    A full step may keep the corresponding candidate optimizer state, if given.
    The Critic and world model are deliberately outside this helper's scope.
    """

    format = "wf_actor_trust_step_v1"

    def __init__(self, accepted_actor_state, accepted_optimizer_state,
                 candidate_actor_state, candidate_optimizer_state=None,
                 alphas=(1., .5, .25, .125)):
        self.alphas = tuple(float(a) for a in alphas)
        if (not self.alphas or any(not math.isfinite(a) or not 0 < a <= 1 for a in self.alphas)
                or any(a <= b for a, b in zip(self.alphas, self.alphas[1:]))):
            raise ValueError("alphas must be a nonempty strictly decreasing sequence in (0, 1]")
        self._accepted = clone_state(accepted_actor_state)
        self._candidate = clone_state(candidate_actor_state)
        self._accepted_optimizer = clone_state(accepted_optimizer_state)
        self._candidate_optimizer = clone_state(candidate_optimizer_state)
        if self._accepted.keys() != self._candidate.keys():
            raise ValueError("accepted and candidate Actor state keys differ")
        for key, accepted in self._accepted.items():
            candidate = self._candidate[key]
            if not isinstance(accepted, torch.Tensor) or not isinstance(candidate, torch.Tensor):
                raise ValueError("Actor state entries must be tensors")
            if accepted.shape != candidate.shape or accepted.dtype != candidate.dtype:
                raise ValueError(f"Actor tensor layout differs in {key}")
            if accepted.is_floating_point() or accepted.is_complex():
                if not torch.isfinite(accepted).all() or not torch.isfinite(candidate).all():
                    raise ValueError(f"nonfinite Actor weights in {key}")
            elif not torch.equal(accepted, candidate):
                raise ValueError(f"discrete Actor buffer {key} cannot be interpolated")

    @staticmethod
    def _validate_alpha(alpha):
        alpha = float(alpha)
        if not math.isfinite(alpha) or not 0 <= alpha <= 1:
            raise ValueError("alpha must be finite and in [0, 1]")
        return alpha

    def state_at(self, alpha):
        alpha = self._validate_alpha(alpha)
        result = {}
        for name, accepted in self._accepted.items():
            candidate = self._candidate[name]
            if alpha == 0 or not (accepted.is_floating_point() or accepted.is_complex()):
                result[name] = accepted.clone()
            elif alpha == 1:
                result[name] = candidate.clone()
            else:
                result[name] = torch.lerp(accepted, candidate, alpha)
        return result

    def candidates(self):
        """Yield independent (alpha, Actor state) pairs in evaluation order."""
        for alpha in self.alphas:
            yield alpha, self.state_at(alpha)

    def apply(self, actor, actor_opt, alpha):
        alpha = self._validate_alpha(alpha)
        actor.load_state_dict(self.state_at(alpha))
        use_candidate_optimizer = alpha == 1 and self._candidate_optimizer is not None
        optimizer = self._candidate_optimizer if use_candidate_optimizer else self._accepted_optimizer
        actor_opt.load_state_dict(clone_state(optimizer))
        return {"actor_step_alpha": alpha,
                "actor_step_optimizer": "candidate" if use_candidate_optimizer else "accepted"}

    def restore(self, actor, actor_opt):
        """Restore accepted weights and optimizer exactly, without aliasing."""
        return self.apply(actor, actor_opt, 0.)

    @torch.no_grad()
    def action_change(self, actor, inputs, process_action=None, *, alpha=None):
        """Compare executable outputs on the same replay inputs, without mutation.

        If alpha is omitted, compare the Actor's current weights to the accepted
        endpoint. Otherwise compare the selected interpolated candidate. This
        is a diagnostic, not a guarantee about closed-loop contact dynamics.
        """
        current = clone_state(actor.state_dict())
        modes = [(module, module.training) for module in actor.modules()]

        def output():
            if isinstance(inputs, dict):
                result = actor(**inputs)
            elif isinstance(inputs, (tuple, list)):
                result = actor(*inputs)
            else:
                result = actor(inputs)
            return process_action(result) if process_action is not None else result

        try:
            actor.eval()
            actor.load_state_dict(self.state_at(0))
            accepted = output().detach().clone()
            actor.load_state_dict(current if alpha is None else self.state_at(alpha))
            candidate = output()
            if accepted.shape != candidate.shape or candidate.ndim < 2 or candidate.shape[-1] != 22:
                raise ValueError("expected batched unified 22-action Actor outputs")
            if not torch.isfinite(accepted).all() or not torch.isfinite(candidate).all():
                raise ValueError("nonfinite Actor outputs on replay inputs")
            delta = (candidate-accepted).reshape(-1, 22)
            return {"action_change_max": float(delta.abs().max()),
                    "action_change_rms": float(delta.square().mean().sqrt()),
                    "action_change_by_joint_max": delta.abs().amax(0).cpu().tolist(),
                    "action_change_by_joint_rms": delta.square().mean(0).sqrt().cpu().tolist(),
                    "action_change_samples": len(delta)}
        finally:
            actor.load_state_dict(current)
            for module, mode in modes:
                module.training = mode

    def state_dict(self):
        return clone_state({"format": self.format, "alphas": self.alphas,
            "accepted_actor_state": self._accepted,
            "accepted_optimizer_state": self._accepted_optimizer,
            "candidate_actor_state": self._candidate,
            "candidate_optimizer_state": self._candidate_optimizer})

    @classmethod
    def from_state_dict(cls, state):
        if state.get("format") != cls.format:
            raise ValueError("unsupported Actor trust checkpoint format")
        return cls(state["accepted_actor_state"], state["accepted_optimizer_state"],
                   state["candidate_actor_state"], state["candidate_optimizer_state"], state["alphas"])

    def load_state_dict(self, state):
        restored = self.from_state_dict(state)
        self.__dict__.update(restored.__dict__)

