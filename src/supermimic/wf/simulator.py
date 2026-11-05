"""Small, sequential CPU batches of the real MuJoCo contact simulator.

This is a correctness backend, not a GPU simulator. Each environment owns an
MjData. A shared MjModel's contact friction is switched before stepping each
environment; calls must remain serial. There is no learned or fake dynamics in
this module, and step never resets a failed environment automatically.
"""

from __future__ import annotations

import copy
from typing import Any

import mujoco
import numpy as np

from .robot import RobotSpec, build_model


class BatchedSimulator:
    def __init__(self, model: mujoco.MjModel | None = None,
                 spec: RobotSpec | None = None, num_envs: int = 1, seed: int = 0,
                 dt: float = .02, friction_range: tuple[float, float] = (.6, 1.2),
                 friction: float | None = None, timestep: float = .005,
                 asset_dir: str | None = None, leg_action_scale: float = .35,
                 arm_action_scale: float | None = .5, wheel_action_scale: float = 25.,
                 arm_initial_position=None):
        if spec is None:
            if model is not None:
                raise ValueError("provide spec with an externally constructed model")
            model, spec = build_model(asset_dir, timestep=timestep,
                leg_action_scale=leg_action_scale, arm_action_scale=arm_action_scale,
                wheel_action_scale=wheel_action_scale)
        elif model is None:
            model = spec.model
        if model is not spec.model:
            raise ValueError("model and RobotSpec must refer to the same model")
        if num_envs < 1 or dt <= 0:
            raise ValueError("num_envs and dt must be positive")
        substeps = round(dt / model.opt.timestep)
        if substeps < 1 or not np.isclose(substeps * model.opt.timestep, dt, atol=1e-10):
            raise ValueError("dt must be an integer multiple of model.opt.timestep")
        if friction is not None:
            friction_range = (friction, friction)
        if not (0 < friction_range[0] <= friction_range[1]):
            raise ValueError("friction_range must be positive and ordered")
        self.model, self.spec = model, spec
        self.num_envs, self.dt, self.substeps = int(num_envs), float(dt), substeps
        self.initial_q=spec.q0.copy()
        if arm_initial_position is not None:
            pose=np.asarray(arm_initial_position,dtype=float)
            if pose.shape!=(6,) or not np.isfinite(pose).all() or np.any(pose<spec.low[16:]) or np.any(pose>spec.high[16:]):
                raise ValueError('initial arm pose violates source joint bounds')
            self.initial_q[16:]=pose
        initial=(self.initial_q-spec.q0)/spec.action_scale;initial[spec.wheel_indices]=0.
        if np.abs(initial).max()>1.000001:raise ValueError('initial arm pose cannot be commanded with this action scale; use the full source range')
        self.initial_action=np.broadcast_to(initial,(self.num_envs,22)).copy()
        self.ablation_action=self.initial_action.copy()
        self.friction_range = tuple(float(v) for v in friction_range)
        self.rng = np.random.default_rng(seed)
        self.data = [mujoco.MjData(model) for _ in range(self.num_envs)]
        self.xi = np.ones((self.num_envs, 1), dtype=np.float64)
        self.applied_action = np.zeros((self.num_envs, 22), dtype=np.float64)
        self.torque = np.zeros((self.num_envs, 22), dtype=np.float64)
        self.torque_peak = np.zeros_like(self.torque)
        self._metrics: dict[str, np.ndarray] = {}
        self.ablation = "none"
        self.reset()

    def _ids(self, ids: Any = None) -> np.ndarray:
        result = np.arange(self.num_envs) if ids is None else np.asarray(ids, dtype=int).reshape(-1)
        if np.any(result < 0) or np.any(result >= self.num_envs):
            raise IndexError("environment index outside batch")
        return result

    def _friction(self, i: int) -> None:
        # Same coefficient on both sides avoids MuJoCo's max-friction mixing
        # accidentally hiding the requested per-environment coefficient.
        self.model.geom_friction[:, 0] = self.xi[i, 0]

    def reset(self, ids: Any = None, state: np.ndarray | None = None) -> np.ndarray:
        ids = self._ids(ids)
        self.xi[ids, 0] = self.rng.uniform(*self.friction_range, size=len(ids))
        for i in ids:
            self._friction(int(i))
            mujoco.mj_resetData(self.model, self.data[i])
            self.data[i].qpos[self.spec.qpos_indices] = self.initial_q
            self.data[i].eq_active[self.spec.base_weld_eq] = self.ablation == "base_locked"
            mujoco.mj_forward(self.model, self.data[i])
        self.applied_action[ids] = self.initial_action[ids]
        self.ablation_action[ids] = self.initial_action[ids]
        self.torque[ids] = 0
        self.torque_peak[ids] = 0
        if state is not None:
            self.write_state(state, ids=ids)
        self._update_metrics()
        return self.read_state()

    def read_state(self) -> np.ndarray:
        """[p3, quat_wxyz4, jointq22, v_world3, omega_body3, jointqd22]."""
        out = np.empty((self.num_envs, 57), dtype=np.float64)
        s = self.spec
        bq, bv = s.base_qpos_index, s.base_dof_index
        for i, d in enumerate(self.data):
            out[i, :7] = d.qpos[bq:bq + 7]
            out[i, 7:29] = d.qpos[s.qpos_indices]
            # MuJoCo free-joint translation velocity is in world axes; its
            # angular velocity is in the local body axes used by our state.
            out[i, 29:35] = d.qvel[bv:bv + 6]
            out[i, 35:57] = d.qvel[s.dof_indices]
        return out

    def write_state(self, state: np.ndarray, ids: Any = None) -> None:
        ids = self._ids(ids)
        state = np.asarray(state, dtype=np.float64)
        if state.shape != (len(ids), 57) or not np.isfinite(state).all():
            raise ValueError(f"state must be finite with shape ({len(ids)}, 57)")
        quat_norm = np.linalg.norm(state[:, 3:7], axis=1)
        if np.any(quat_norm < 1e-10):
            raise ValueError("base quaternion must have nonzero norm")
        s = self.spec
        bq, bv = s.base_qpos_index, s.base_dof_index
        for row, i in enumerate(ids):
            d = self.data[i]
            d.qpos[bq:bq + 7] = state[row, :7]
            d.qpos[bq + 3:bq + 7] /= quat_norm[row]
            d.qpos[s.qpos_indices] = state[row, 7:29]
            d.qvel[bv:bv + 6] = state[row, 29:35]
            d.qvel[s.dof_indices] = state[row, 35:57]
            d.qacc_warmstart[:] = 0
            self._friction(int(i))
            mujoco.mj_forward(self.model, d)
        self._update_metrics()

    def step(self, action: np.ndarray, ids: Any = None) -> np.ndarray:
        """Advance selected worlds using full-batch normalized commands.

        Unselected MjData integration states, times and command/torque records
        remain untouched. The returned state always contains the full batch.
        """
        ids = self._ids(ids)
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (self.num_envs, 22) or not np.isfinite(action).all():
            raise ValueError(f"action must be finite with shape ({self.num_envs}, 22)")
        command = np.clip(action[ids], -1., 1.)
        if self.ablation == "legs_locked":
            command[:, :12] = self.ablation_action[ids,:12]
        elif self.ablation == "arm_locked":
            command[:, 16:22] = self.ablation_action[ids,16:22]
        s = self.spec
        self.torque_peak[ids] = 0
        position = np.ones(22, dtype=bool)
        position[s.wheel_indices] = False
        requested_target = s.q0 + s.action_scale * command
        target = np.clip(requested_target, s.low, s.high)
        wheel_target = np.clip(s.action_scale[s.wheel_indices] *
                               command[:, s.wheel_indices],
                               -s.velocity[s.wheel_indices], s.velocity[s.wheel_indices])
        # Replay stores the normalized command that was actually executed after
        # joint-limit/velocity target clipping, not an impossible requested target.
        command[:, position] = np.where(
            target[:, position] != requested_target[:, position],
            (target[:, position] - s.q0[position]) / s.action_scale[position],
            command[:, position])
        wheel_requested = command[:, s.wheel_indices] * s.action_scale[s.wheel_indices]
        command[:, s.wheel_indices] = np.where(
            wheel_target != wheel_requested, wheel_target / s.action_scale[s.wheel_indices],
            command[:, s.wheel_indices])
        self.applied_action[ids] = np.clip(command, -1, 1)
        for row, i in enumerate(ids):
            d = self.data[i]
            self._friction(int(i))
            for _ in range(self.substeps):
                q, qd = d.qpos[s.qpos_indices], d.qvel[s.dof_indices]
                torque = s.kp * (target[row] - q) - s.kd * qd
                torque[s.wheel_indices] = s.kd[s.wheel_indices] * (
                    wheel_target[row] - qd[s.wheel_indices])
                # Do not accelerate further into an exceeded source velocity
                # limit. This complements torque limits, it is not a hard
                # physically impossible clamp of the integrated joint speed.
                torque[(np.abs(qd) > s.velocity) & (torque * qd > 0)] = 0
                torque = np.clip(torque, -s.effort, s.effort)
                d.ctrl[s.actuator_ids] = torque
                mujoco.mj_step(self.model, d)
                self.torque[i] = torque
                self.torque_peak[i] = np.maximum(self.torque_peak[i], np.abs(torque))
            # mj_step can leave position-dependent caches at the previous
            # integration point; make contacts/sites agree with read_state.
            mujoco.mj_forward(self.model, d)
        self._update_metrics()
        return self.read_state()

    def settle(self, ids: Any = None, steps: int = 50) -> np.ndarray:
        """Hold the configured reset joint targets before assigning goals.

        This uses real MuJoCo steps and the same PD/limits as collection. It adds
        neither gravity compensation nor target trajectories, and advances only
        the selected environments. Reset itself remains unchanged.
        """
        if isinstance(steps, bool) or int(steps) != steps or steps < 0:
            raise ValueError("settling steps must be a nonnegative integer")
        ids = self._ids(ids)
        command = self.initial_action.copy()
        for _ in range(int(steps)):
            self.step(command, ids=ids)
        return self.read_state()

    def _update_metrics(self) -> None:
        n, s = self.num_envs, self.spec
        wheel_contact = np.zeros((n, 4), dtype=bool)
        wheel_normal_force = np.zeros((n, 4))
        slip = np.zeros((n, 4))
        forbidden = np.zeros(n, dtype=bool)
        height, tilt = np.empty(n), np.empty(n)
        force = np.zeros(6)
        jacp, jacr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
        wheel_lookup = {int(g): k for k, g in enumerate(s.wheel_geom_ids)}
        for i, d in enumerate(self.data):
            height[i] = d.xpos[s.base_body, 2]
            tilt[i] = np.arccos(np.clip(d.xmat[s.base_body, 8], -1., 1.))
            for ci in range(d.ncon):
                c = d.contact[ci]
                if c.dist > 0:
                    continue
                g1, g2 = int(c.geom1), int(c.geom2)
                ground = g1 == s.ground_geom or g2 == s.ground_geom
                other = g2 if g1 == s.ground_geom else g1
                if ground and other in wheel_lookup:
                    k = wheel_lookup[other]
                    # Geometrically generated contacts can be inactive or carry
                    # no normal load. They are not supported rolling contacts.
                    if c.efc_address < 0:
                        continue
                    mujoco.mj_contactForce(self.model, d, ci, force)
                    normal_force = max(0., force[0])
                    if normal_force <= 1e-6:
                        continue
                    wheel_contact[i, k] = True
                    wheel_normal_force[i, k] += normal_force
                    mujoco.mj_jac(self.model, d, jacp, jacr, c.pos, int(s.wheel_body_ids[k]))
                    v = jacp @ d.qvel
                    # Floor tangent is world XY; includes wheel spin, body
                    # translation and articulated leg motion at the contact.
                    slip[i, k] = max(slip[i, k], float(np.linalg.norm(v[:2])))
                else:
                    forbidden[i] = True
        state = self.read_state()
        finite = np.isfinite(state).all(axis=1)
        excess = np.maximum(s.low - state[:, 7:29], state[:, 7:29] - s.high)
        limit = np.maximum(0., excess).max(axis=1)
        safe = finite & (height > .20) & (tilt < 1.0) & (limit < .05) & ~forbidden
        self._metrics = dict(wheel_contact=wheel_contact, contacts=wheel_contact,
                             wheel_normal_force=wheel_normal_force, slip=slip,
                             torque=self.torque.copy(), torque_peak=self.torque_peak.copy(),
                             base_height=height, tilt=tilt,
                             joint_limit_violation=limit, forbidden_contact=forbidden,
                             finite=finite, safe=safe)

    def metrics(self) -> dict[str, np.ndarray]:
        """Copy of physical contact/safety metrics for the latest state."""
        return {k: v.copy() for k, v in self._metrics.items()}

    def set_ablation(self, mode: str = "none") -> None:
        """Physical base weld, or fixture command hold for the legs/arm.

        The weld holds the default reset base pose. Leg/arm modes suppress
        commands with ordinary PD actuation; they do not weld those joints.
        Call before reset for paired evaluation episodes.
        """
        if mode not in ("none", "base_locked", "legs_locked", "arm_locked"):
            raise ValueError(f"unknown ablation: {mode}")
        self.ablation = mode
        for i, d in enumerate(self.data):
            d.eq_active[self.spec.base_weld_eq] = mode == "base_locked"
            self._friction(i)
            mujoco.mj_forward(self.model, d)
        self._update_metrics()

    @property
    def contacts(self) -> np.ndarray:
        return self._metrics["wheel_contact"].copy()

    @property
    def slip(self) -> np.ndarray:
        return self._metrics["slip"].copy()

    def snapshot(self) -> dict[str, Any]:
        """Full integration state, controls, friction and RNG for exact replay."""
        flag = int(mujoco.mjtState.mjSTATE_INTEGRATION)
        states = np.empty((self.num_envs, mujoco.mj_stateSize(self.model, flag)))
        for i, d in enumerate(self.data):
            mujoco.mj_getState(self.model, d, states[i], flag)
        return dict(integration_state=states, xi=self.xi.copy(),
                    applied_action=self.applied_action.copy(), torque=self.torque.copy(),
                    torque_peak=self.torque_peak.copy(),
                    rng=copy.deepcopy(self.rng.bit_generator.state), state_flag=flag,
                    ablation=self.ablation,
                    ablation_action=self.ablation_action.copy(),
                    num_envs=self.num_envs)

    def restore(self, snapshot: dict[str, Any]) -> None:
        if snapshot["num_envs"] != self.num_envs:
            raise ValueError("snapshot batch size differs")
        self.xi[:] = snapshot["xi"]
        self.applied_action[:] = snapshot["applied_action"]
        self.torque[:] = snapshot["torque"]
        self.torque_peak[:] = snapshot.get("torque_peak", np.abs(self.torque))
        self.rng.bit_generator.state = copy.deepcopy(snapshot["rng"])
        self.ablation = snapshot.get("ablation", "none")
        self.ablation_action[:]=snapshot.get('ablation_action',self.initial_action)
        for i, d in enumerate(self.data):
            self._friction(i)
            mujoco.mj_setState(self.model, d, snapshot["integration_state"][i],
                              snapshot["state_flag"])
            mujoco.mj_forward(self.model, d)
        self._update_metrics()

    def close(self) -> None:
        """Compatibility with environment lifecycles; no renderer is opened."""
