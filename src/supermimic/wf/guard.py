"""Real-physics acceptance of BPTT policy candidates.

This is an additional engineering safeguard, not part of the SuperMimic paper.
The evaluator must use the same training-validation goals, initial conditions and
full episode budget for baseline and candidates. Never tune this gate on final
test seeds. An accepted candidate is still one unified learned policy; rejected
weights and their optimizer moments are restored without changing the learned
world model or deleting failure transitions from replay.
"""

from __future__ import annotations

import copy
import math

import torch


def _snapshot(value):
    """Detached CPU copy, including optimizer tensor states and scalar options."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _snapshot(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_snapshot(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_snapshot(item) for item in value)
    return copy.deepcopy(value)


class PolicyGuard:
    """Compare real returns and physical failures, restoring rejected candidates.

    ``budgeted_reward_mean`` must divide each episode's cumulative reward by its
    full configured horizon, including an absorbing failure penalty for every
    omitted step. It must not be an average over alive steps only. By default
    hold-success is recorded while dense progress selects candidates. Optional
    ``prefer_success`` first preserves physical failures, then prefers a higher
    final held-success rate, comparing dense progress within the same success
    tier. This prevents a dense-reward improvement from undoing task success.

    Optional critic snapshots should be supplied in both capture and observe:
    they avoid keeping critic bootstrap values fitted to a rejected candidate.
    LR scheduling remains the trainer's responsibility; multiply its schedule by
    ``learning_rate_scale`` after each call, including following resume.
    """

    format = "wf_policy_guard_v1"
    identity_keys = ("target_hash", "seed", "level", "episodes", "horizon_steps",
                     "hold_steps", "friction", "ablation", "reset_settle_steps", "goal_seed_device","initial_condition_hash")

    def __init__(self, return_tolerance=0.01, failure_slack=0.,
                 rejection_lr_factor=.5, min_lr_scale=.01, prefer_success=False,prefer_safety=False):
        if not math.isfinite(return_tolerance) or return_tolerance < 0:
            raise ValueError("return_tolerance must be finite and nonnegative")
        if not 0 <= failure_slack <= 1:
            raise ValueError("failure_slack must be in [0, 1]")
        if not 0 < rejection_lr_factor <= 1 or not 0 < min_lr_scale <= 1:
            raise ValueError("LR factors must be in (0, 1]")
        self.return_tolerance = float(return_tolerance)
        self.failure_slack = float(failure_slack)
        self.rejection_lr_factor = float(rejection_lr_factor)
        self.min_lr_scale = float(min_lr_scale)
        self.prefer_success = bool(prefer_success)
        self.prefer_safety = bool(prefer_safety)
        self.learning_rate_scale = 1.
        self.accepted = 0
        self.rejected = 0
        self._baseline = None
        self._weights = None
        self.best_return = None
        self.best_failure = None
        self.best_success = None

    @staticmethod
    def _metrics(evaluation):
        try:
            score = float(evaluation["budgeted_reward_mean"])
            failure = float(evaluation["failure_rate"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("guard evaluation requires budgeted_reward_mean and failure_rate") from exc
        if not math.isfinite(score) or not math.isfinite(failure) or not 0 <= failure <= 1:
            return None
        return score, failure

    @staticmethod
    def _success(evaluation):
        try:
            value = float(evaluation["final_held_success_rate"])
        except (KeyError, TypeError, ValueError):
            return None
        return value if math.isfinite(value) and 0 <= value <= 1 else None

    def _validate_identity(self, evaluation):
        if self._baseline is None:
            raise RuntimeError("capture_baseline must evaluate the initial actor before observing candidates")
        for key in self.identity_keys:
            default = {"reset_settle_steps": 0, "goal_seed_device": "policy"}.get(key)
            if self._baseline.get(key, default) != evaluation.get(key, default):
                raise ValueError(f"guard validation changed in {key}; evaluate and rebase the incumbent first")

    def assess(self, evaluation):
        """Return (accepted, reason) without changing weights or guard state.

        The trainer's backtracking search must use this same rule before loading
        a candidate snapshot. Changed validation fixtures remain an error rather
        than becoming a misleading comparison across unrelated goals.
        """
        self._validate_identity(evaluation)
        metrics = self._metrics(evaluation)
        if metrics is None:
            return False, "nonfinite_metrics"
        score, failure = metrics
        if failure > self.best_failure+self.failure_slack+1e-12:
            return False, "physical_failure_regression"
        if self.prefer_success:
            success = self._success(evaluation)
            if success is None:
                return False, "invalid_success_metrics"
        if self.prefer_safety and failure<self.best_failure-1e-12:
            return True,'physical_safety_improved'
        if self.prefer_success:
            if success < self.best_success-1e-12:
                return False, "held_success_regression"
            if success > self.best_success+1e-12:
                return True, "held_success_improved"
        if score < self.best_return-self.return_tolerance:
            return False, "real_return_regression"
        return True, "real_physics_accepted"

    def _capture(self, actor, actor_opt, critic, critic_opt):
        if (critic is None) != (critic_opt is None):
            raise ValueError("provide both critic and critic_opt, or neither")
        result = {"actor": _snapshot(actor.state_dict()), "actor_opt": _snapshot(actor_opt.state_dict())}
        if critic is not None:
            result.update(critic=_snapshot(critic.state_dict()), critic_opt=_snapshot(critic_opt.state_dict()))
        return result

    def capture_baseline(self, evaluation, actor, actor_opt, critic=None, critic_opt=None):
        """Evaluate the initial random actor before the first candidate update.

        Call again only when the validation problem changes (e.g. curriculum
        level), evaluating the incumbent on that new fixed problem first.
        Counters and the LR multiplier persist across such explicit rebasing.
        """
        metrics = self._metrics(evaluation)
        if metrics is None:
            raise ValueError("baseline evaluation must have finite valid metrics")
        success = self._success(evaluation)
        if self.prefer_success and success is None:
            raise ValueError("baseline final_held_success_rate must be finite and in [0, 1]")
        weights = self._capture(actor, actor_opt, critic, critic_opt)
        self._baseline = _snapshot(evaluation)
        self._weights = weights
        self.best_return, self.best_failure = metrics
        self.best_success = success
        return {"guard_accepted": True, "guard_reason": "baseline",
                "guard_return": self.best_return, "guard_failure_rate": self.best_failure,
                "guard_best_return": self.best_return, "guard_best_failure_rate": self.best_failure,
                "guard_best_success_rate": self.best_success,
                "guard_held_success_rate": evaluation.get("final_held_success_rate"),
                "guard_accept_count": self.accepted, "guard_reject_count": self.rejected,
                "guard_learning_rate_scale": self.learning_rate_scale}

    # Accommodate the concise name used in early experiment notes.
    capturebaseline = capture_baseline

    def observe(self, evaluation, actor, actor_opt, critic=None, critic_opt=None):
        self._validate_identity(evaluation)
        with_critic = "critic" in self._weights
        if with_critic != (critic is not None and critic_opt is not None):
            raise ValueError("candidate must use the same critic snapshot configuration as baseline")
        if (critic is None) != (critic_opt is None):
            raise ValueError("provide both critic and critic_opt, or neither")
        accepted, reason = self.assess(evaluation)
        metrics = self._metrics(evaluation)
        success = self._success(evaluation)
        if metrics is None:
            score, failure = None, None
        else:
            score, failure = metrics
        if accepted:
            safer=self.prefer_safety and failure<self.best_failure-1e-12
            self._weights = self._capture(actor, actor_opt, critic, critic_opt)
            self._baseline = _snapshot(evaluation)
            # Tolerance is relative to the historical high water mark. Repeated
            # small accepted decreases cannot silently drift without a bound.
            if safer or (self.prefer_success and success > self.best_success+1e-12):
                # A new success tier gets its own dense-return reference. A high
                # reward from a less successful tier cannot permanently block it.
                self.best_return = score
            else:
                self.best_return = max(self.best_return, score)
            self.best_failure = min(self.best_failure, failure)
            if success is not None:
                self.best_success = success if safer or self.best_success is None else max(self.best_success, success)
            self.accepted += 1
        else:
            actor.load_state_dict(self._weights["actor"])
            # Optimizer loading can retain tensor references (notably Adam's
            # CPU step counter, and CPU moments). The next candidate must not
            # mutate the accepted snapshot through those shared references.
            actor_opt.load_state_dict(_snapshot(self._weights["actor_opt"]))
            if with_critic:
                critic.load_state_dict(self._weights["critic"])
                critic_opt.load_state_dict(_snapshot(self._weights["critic_opt"]))
            self.learning_rate_scale = max(self.min_lr_scale,
                self.learning_rate_scale*self.rejection_lr_factor)
            self.rejected += 1
        return {"guard_accepted": accepted, "guard_reason": reason,
                "guard_return": score, "guard_failure_rate": failure,
                "guard_best_return": self.best_return, "guard_best_failure_rate": self.best_failure,
                "guard_best_success_rate": self.best_success,
                "guard_held_success_rate": success if self.prefer_success else evaluation.get("final_held_success_rate"),
                "guard_learning_rate_scale": self.learning_rate_scale,
                "guard_accept_count": self.accepted, "guard_reject_count": self.rejected}

    def state_dict(self):
        return _snapshot({"format": self.format, "return_tolerance": self.return_tolerance,
            "failure_slack": self.failure_slack, "rejection_lr_factor": self.rejection_lr_factor,
            "prefer_success": self.prefer_success,
            "prefer_safety": self.prefer_safety,
            "min_lr_scale": self.min_lr_scale, "learning_rate_scale": self.learning_rate_scale,
            "accepted": self.accepted, "rejected": self.rejected,
            "baseline": self._baseline, "weights": self._weights,
            "best_return": self.best_return, "best_failure": self.best_failure,
            "best_success": self.best_success})

    def load_state_dict(self, state):
        if state.get("format") != self.format:
            raise ValueError("unsupported policy guard checkpoint format")
        for key in ("return_tolerance", "failure_slack", "rejection_lr_factor", "min_lr_scale"):
            if state[key] != getattr(self, key):
                raise ValueError(f"guard configuration mismatch in {key}")
        if state.get("prefer_success", False) != self.prefer_success:
            raise ValueError("guard configuration mismatch in prefer_success")
        if state.get('prefer_safety',False)!=self.prefer_safety:
            raise ValueError('guard configuration mismatch in prefer_safety')
        state = _snapshot(state)
        best_success = state.get("best_success", self._success(state.get("baseline") or {}))
        if self.prefer_success and state["baseline"] is not None:
            try:
                valid_success = math.isfinite(float(best_success)) and 0 <= float(best_success) <= 1
            except (TypeError, ValueError):
                valid_success = False
            if not valid_success:
                raise ValueError("success-first guard snapshot requires finite best_success")
        self.learning_rate_scale = state["learning_rate_scale"]
        self.accepted, self.rejected = state["accepted"], state["rejected"]
        self._baseline, self._weights = state["baseline"], state["weights"]
        self.best_return, self.best_failure = state["best_return"], state["best_failure"]
        self.best_success = best_success
