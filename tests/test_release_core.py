"""Asset-free regression checks for the published learning and release code."""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from supermimic.assets import resolve_wfarm_assets
from supermimic.imagination.objective import gae_targets, policy_objective
from supermimic.wf.config import ModelConfig
from supermimic.wf.model import StateLayout, WholeBodyWorldModel

ROOT = Path(__file__).resolve().parents[1]


def nominal(count=3):
    state = torch.zeros(count, 57)
    state[:, 2] = .5
    state[:, 3] = 1.
    return state


def test_residual_integration_preserves_constant_velocity_and_unit_rotation():
    layout = StateLayout(torch.zeros(22), dt=.02)
    state = nominal(); state[:, 29] = .4; state[:, 35:] = .2
    result = layout.integrate(state, torch.zeros(len(state), 56))
    torch.testing.assert_close(result[:, :3], state[:, :3] + .02 * state[:, 29:32])
    torch.testing.assert_close(result[:, 7:29], state[:, 7:29] + .02 * state[:, 35:])
    torch.testing.assert_close(result[:, 3:7].norm(dim=-1), torch.ones(len(state)))


def test_wheel_phase_is_invariant_but_wheel_velocity_is_observable():
    layout = StateLayout(torch.zeros(22))
    original = nominal(); changed = original.clone(); changed[:, 19:23] = 11.
    torch.testing.assert_close(layout.canonical(original), layout.canonical(changed))
    torch.testing.assert_close(layout.error(original, changed), torch.zeros(3, 52))
    changed[:, 47:51] = 1.
    assert not torch.equal(layout.canonical(original), layout.canonical(changed))


def test_frozen_dynamics_keep_action_gradients():
    model = WholeBodyWorldModel(torch.zeros(22), ModelConfig(hidden=(16, 16)))
    # Avoid zero-initialized output hiding the gradient path in this check.
    with torch.no_grad():
        model.dynamics.net[-1].weight.normal_(std=.02)
    model.requires_grad_(False)
    action = torch.randn(3, 22, requires_grad=True)
    result = model.step(nominal(), action, torch.ones(3, 1))
    result[:, 29:].square().mean().backward()
    assert action.grad is not None and torch.isfinite(action.grad).all()
    assert action.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in model.parameters())


def test_survival_weights_and_terminal_bootstrap_match_closed_form():
    rewards = torch.tensor([[2.], [3.]], requires_grad=True)
    rollout = SimpleNamespace(rewards=rewards, survival=torch.tensor([[1.], [.5], [.25]]))
    result = policy_objective(rollout, torch.tensor([4.]), .9)
    expected = 2 + .9 * .5 * 3 + .9**2 * .25 * 4
    torch.testing.assert_close(result, torch.tensor(expected))
    result.backward()
    torch.testing.assert_close(rewards.grad[:, 0], torch.tensor([1., .45]))


def test_terminal_lambda_target_has_no_bootstrap_or_target_gradient():
    rewards = torch.tensor([[2.], [3.]], requires_grad=True)
    failed = torch.tensor([[1.], [0.]])
    values = torch.tensor([[10.], [20.], [30.]], requires_grad=True)
    targets = gae_targets(rewards, failed, values, gamma=.9, lam=.95)
    assert not targets.requires_grad
    assert targets[0, 0] == 2
    torch.testing.assert_close(targets[1, 0], torch.tensor(30.))


def test_assets_are_explicit_and_missing_files_fail_early(tmp_path, monkeypatch):
    monkeypatch.setenv("WFARM_ASSET_DIR", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="supplied separately"):
        resolve_wfarm_assets()
    # Filename-only fixtures test discovery, not URDF geometry or model validity.
    for relative in ("urdf/wf-arm-tbot.urdf", "config/robot.json", "config/robot_calibration.json"):
        target = tmp_path / relative; target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("")
    assert resolve_wfarm_assets() == tmp_path


def test_release_allowlist_rejects_models_weights_history_and_symlinks(tmp_path):
    spec = importlib.util.spec_from_file_location("release_audit", ROOT / "tools/release_audit.py")
    audit = importlib.util.module_from_spec(spec); spec.loader.exec_module(audit)
    assert audit.allowed("src/supermimic/wf/train.py")
    assert audit.allowed("src/supermimic/data/replay_buffer.py")
    for path in ("wf-arm-tbot/urdf/model.urdf", "weights/actor.pt", "docs/history/run.md",
                 "frameworks/fixed_base_arm/assets/robot.json", "data/calibration.json"):
        assert not audit.allowed(path)
    (tmp_path / "README.md").symlink_to(ROOT / "README.md")
    result = audit.scan(tmp_path, ["README.md"])
    assert not result["passed"]


def test_release_examples_contain_no_private_resource_path():
    import json

    config = json.loads((ROOT / "configs/examples/fixed_arm_smoke.json").read_text())
    assert config["asset_dir"] == "$WFARM_ASSET_DIR"
    assert not config.get("warm_start")
    assert not config.get("goal_encoder_checkpoint")
    assert config["device"] == "cpu"


@pytest.mark.private_assets
def test_private_model_build_when_explicitly_requested(monkeypatch):
    import os

    if not os.environ.get("WFARM_ASSET_DIR"):
        pytest.skip("private model tests require an explicit WFARM_ASSET_DIR")
    from supermimic.wf.robot import build_model
    _, spec = build_model(arm_action_scale=None)
    assert spec.num_actions == 22
