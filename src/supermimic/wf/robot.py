"""Native MuJoCo model of the wheel-legged mobile manipulator.

The original URDF is not changed. The two fingers are welded at 0.04 m total
opening, a free base and floor are added, and only collision geometry is loaded.
Gains, rotor inertias and control scales below are simulation assumptions, not
identified hardware parameters. Source effort/velocity limits are also not a
claim about measured motor capabilities.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from supermimic.assets import resolve_wfarm_assets


def _rpy_matrix(rpy: np.ndarray) -> np.ndarray:
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


@dataclass
class RobotSpec:
    """Name-resolved model addresses in policy order, independent of XML order."""

    model: mujoco.MjModel
    joint_names: tuple[str, ...]
    q0: np.ndarray
    qpos_indices: np.ndarray
    dof_indices: np.ndarray
    low: np.ndarray
    high: np.ndarray
    effort: np.ndarray
    velocity: np.ndarray
    kp: np.ndarray
    kd: np.ndarray
    action_scale: np.ndarray
    actuator_ids: np.ndarray
    wheel_indices: np.ndarray
    base_body: int
    base_qpos_index: int
    base_dof_index: int
    tcp_body: int
    tcp_site: int
    tcp_local_pos: np.ndarray
    tcp_local_quat: np.ndarray
    wheel_body_ids: np.ndarray
    wheel_geom_ids: np.ndarray
    wheel_radius: float
    ground_geom: int
    base_weld_eq: int
    default_height: float
    asset_dir: Path

    @property
    def joint_names22(self) -> tuple[str, ...]:
        return self.joint_names

    @property
    def num_actions(self) -> int:
        return 22

    @property
    def state_dim(self) -> int:
        return 57


def build_model(asset_dir: str | Path | None = None, timestep: float = 0.005, *,
                leg_action_scale: float = .35, arm_action_scale: float | None = .5,
                wheel_action_scale: float = 25., with_mj_spec: bool = False
                ) -> tuple[mujoco.MjModel, RobotSpec]:
    """Build a floating-base, 22-motor model directly from the portable URDF.

    ``asset_dir`` is the wf-arm-tbot directory. No files are written and no
    mjlab, Isaac Lab, ROS, or GPU dependency is required.
    """
    if timestep <= 0:
        raise ValueError("timestep must be positive")
    for name, value in (("leg_action_scale", leg_action_scale),
                        ("arm_action_scale", arm_action_scale),
                        ("wheel_action_scale", wheel_action_scale)):
        if value is not None and (not np.isfinite(value) or value <= 0):
            raise ValueError(f"{name} must be finite and positive")
    root = resolve_wfarm_assets(asset_dir)
    source = root / "urdf/wf-arm-tbot.urdf"
    config = json.loads((root / "config/robot.json").read_text())
    tcp = json.loads((root / "config/robot_calibration.json").read_text())["tcp"]
    tcp_parent = tcp["parent_link"]
    tcp_pos = np.asarray(tcp["xyz_m"], dtype=np.float64)
    tcp_rpy = np.asarray(tcp["rpy_rad"], dtype=np.float64)
    if (tcp_pos.shape != (3,) or tcp_rpy.shape != (3,) or
            not np.isfinite(tcp_pos).all() or not np.isfinite(tcp_rpy).all()):
        raise ValueError("TCP xyz_m and rpy_rad must each contain three finite values")
    tcp_quat = np.empty(4)
    mujoco.mju_mat2Quat(tcp_quat, _rpy_matrix(tcp_rpy).reshape(-1))
    urdf = ET.parse(source).getroot()
    if tcp_parent not in {link.get("name") for link in urdf.findall("link")}:
        raise ValueError(f"TCP parent_link does not exist in robot: {tcp_parent}")
    joint_names = tuple(config["leg_wheel_arm_joint_order"])
    if len(joint_names) != 22 or len(set(joint_names)) != 22:
        raise ValueError("expected 22 distinct leg/wheel/arm joint names")
    joints = {j.get("name"): j for j in urdf.findall("joint")}
    for link in urdf.findall("link"):
        for visual in list(link.findall("visual")):
            link.remove(visual)
        for i, collision in enumerate(link.findall("collision")):
            if not collision.get("name"):
                collision.set("name", f"{link.get('name')}_collision_{i}")
    for mesh in urdf.findall(".//mesh"):
        mesh.set("filename", str((source.parent / mesh.get("filename")).resolve()))
    # URDF origin is parent->joint, then translation axis*q in joint coordinates.
    # Preserve that composition rather than adding q directly in parent axes.
    for name in config["gripper_joint_order"]:
        joint = joints[name]
        origin = joint.find("origin")
        xyz = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
        rpy = np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")
        axis = np.fromstring(joint.find("axis").get("xyz"), sep=" ")
        q = float(config["default_joint_pos"][name])
        xyz += _rpy_matrix(rpy) @ (axis * q)
        origin.set("xyz", " ".join(f"{x:.17g}" for x in xyz))
        joint.set("type", "fixed")
        for child in list(joint):
            if child.tag not in ("origin", "parent", "child"):
                joint.remove(child)
    ET.SubElement(urdf, "link", name="world")
    free = ET.SubElement(urdf, "joint", name="wf_free_base", type="floating")
    ET.SubElement(free, "parent", link="world")
    ET.SubElement(free, "child", link="base")
    extension = ET.SubElement(urdf, "mujoco")
    ET.SubElement(extension, "compiler", fusestatic="false", discardvisual="true")
    mj = mujoco.MjSpec.from_string(ET.tostring(urdf, encoding="unicode"))
    mj.option.timestep = timestep
    mj.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    mj.option.iterations = 50
    mj.option.gravity = [0, 0, -9.81]
    mj.worldbody.add_geom(name="wf_ground", type=mujoco.mjtGeom.mjGEOM_PLANE,
                         size=[0, 0, 0.1], friction=[1.0, 0.005, 0.0001],
                         rgba=[0.25, 0.28, 0.3, 1], condim=3)
    mj.body(tcp_parent).add_site(
        name="wf_tcp", pos=tcp_pos, quat=tcp_quat, size=[0.008, 0.008, 0.008],
        rgba=[0, 0.8, 0.3, 1])
    # Inactive physical intervention used by evaluation, not a per-step teleport.
    mj.add_equality(name="wf_base_lock", type=mujoco.mjtEq.mjEQ_WELD,
                    objtype=mujoco.mjtObj.mjOBJ_BODY,
                    name1="world", name2="base", active=False,
                    data=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
                    solref=[.005, 1])
    # Adjacent bodies have intentionally overlapping construction surfaces.
    # Other pairs (including arm vs legs/base) remain eligible for collision.
    for joint in urdf.findall("joint"):
        parent, child = joint.find("parent").get("link"), joint.find("child").get("link")
        if parent != "world":
            mj.add_exclude(bodyname1=parent, bodyname2=child)
    for name in joint_names:
        limit = joints[name].find("limit")
        effort = float(limit.get("effort"))
        mj.add_actuator(name=f"motor_{name}", target=name,
                        trntype=mujoco.mjtTrn.mjTRN_JOINT,
                        gear=[1, 0, 0, 0, 0, 0],
                        ctrllimited=True, ctrlrange=[-effort, effort],
                        forcelimited=True, forcerange=[-effort, effort])
    model = mj.compile()
    getid = lambda kind, name: mujoco.mj_name2id(model, kind, name)
    joint_ids = np.array([getid(mujoco.mjtObj.mjOBJ_JOINT, n) for n in joint_names])
    qadr, dadr = model.jnt_qposadr[joint_ids].copy(), model.jnt_dofadr[joint_ids].copy()
    q0 = np.array([config["default_joint_pos"][n] for n in joint_names], dtype=np.float64)
    low, high, effort, velocity = [], [], [], []
    for name in joint_names:
        limit = joints[name].find("limit")
        continuous = joints[name].get("type") == "continuous"
        low.append(-np.inf if continuous else float(limit.get("lower")))
        high.append(np.inf if continuous else float(limit.get("upper")))
        effort.append(float(limit.get("effort")))
        velocity.append(float(limit.get("velocity")))
    wheel_ids = np.arange(12, 16)
    wheel_bodies = np.array([getid(mujoco.mjtObj.mjOBJ_BODY, n.removesuffix("_joint"))
                             for n in joint_names[12:16]])
    wheel_geoms = np.array([getid(mujoco.mjtObj.mjOBJ_GEOM,
                                 f"{n.removesuffix('_joint')}_collision")
                            for n in joint_names[12:16]])
    base_joint = getid(mujoco.mjtObj.mjOBJ_JOINT, "wf_free_base")
    base_qadr, base_dadr = int(model.jnt_qposadr[base_joint]), int(model.jnt_dofadr[base_joint])
    # Modest reflected rotor inertia/damping are explicit simulation assumptions.
    model.dof_armature[dadr] = np.r_[np.full(12, 0.05), np.full(4, 0.01), np.full(6, 0.02)]
    model.dof_damping[dadr] = np.r_[np.full(12, 0.2), np.full(4, 0.03), np.full(6, 0.1)]
    kp = np.r_[np.tile([200., 250., 300.], 4), np.zeros(4), [50., 70., 60., 30., 20., 15.]]
    kd = np.r_[np.tile([8., 8., 10.], 4), np.full(4, 2.), [2., 2., 1.5, 1., .8, .5]]
    arm_scale = (np.maximum(q0[16:22] - np.asarray(low)[16:22],
                            np.asarray(high)[16:22] - q0[16:22])
                 if arm_action_scale is None else np.full(6, arm_action_scale))
    scale = np.r_[np.full(12, leg_action_scale), np.full(4, wheel_action_scale), arm_scale]
    data = mujoco.MjData(model)
    data.qpos[qadr] = q0
    mujoco.mj_forward(model, data)
    # Cylinders are horizontal in the nominal pose; compute support extent for
    # every wheel so the initial placement is derived from geometry, not guessed.
    wheel_bottoms = []
    for geom in wheel_geoms:
        axis = data.geom_xmat[geom].reshape(3, 3)[:, 2]
        radius, half_width = model.geom_size[geom, :2]
        extent_z = radius * np.sqrt(max(0., 1 - axis[2] ** 2)) + half_width * abs(axis[2])
        wheel_bottoms.append(data.geom_xpos[geom, 2] - extent_z)
    height = float(-min(wheel_bottoms) + 0.002)
    # Hinge qpos0 also defines the kinematic reference. Do not assign q0 there:
    # reset explicitly writes the nominal joint configuration instead.
    model.qpos0[base_qadr:base_qadr + 3] = [0, 0, height]
    model.qpos0[base_qadr + 3:base_qadr + 7] = [1, 0, 0, 0]
    weld_id = getid(mujoco.mjtObj.mjOBJ_EQUALITY, "wf_base_lock")
    model.eq_data[weld_id, 3:6] = [0, 0, height]
    model.eq_data[weld_id, 6:10] = [1, 0, 0, 0]
    tcp_site = getid(mujoco.mjtObj.mjOBJ_SITE, "wf_tcp")
    spec = RobotSpec(
        model=model, joint_names=joint_names, q0=q0, qpos_indices=qadr, dof_indices=dadr,
        low=np.asarray(low), high=np.asarray(high), effort=np.asarray(effort),
        velocity=np.asarray(velocity), kp=kp, kd=kd, action_scale=scale,
        actuator_ids=np.array([getid(mujoco.mjtObj.mjOBJ_ACTUATOR, f"motor_{n}")
                               for n in joint_names]), wheel_indices=wheel_ids,
        base_body=getid(mujoco.mjtObj.mjOBJ_BODY, "base"),
        base_qpos_index=base_qadr, base_dof_index=base_dadr,
        tcp_body=int(model.site_bodyid[tcp_site]), tcp_site=tcp_site,
        tcp_local_pos=model.site_pos[tcp_site].copy(),
        tcp_local_quat=model.site_quat[tcp_site].copy(),
        wheel_body_ids=wheel_bodies, wheel_geom_ids=wheel_geoms, wheel_radius=.093,
        ground_geom=getid(mujoco.mjtObj.mjOBJ_GEOM, "wf_ground"),
        base_weld_eq=weld_id,
        default_height=height, asset_dir=root)
    if (model.nq, model.nv, model.nu) != (29, 28, 22):
        raise ValueError(f"unexpected model dimensions: {model.nq}, {model.nv}, {model.nu}")
    if not np.isclose(model.body_mass.sum(), config["source_mass_sum_kg"], atol=1e-8):
        raise ValueError("robot mass changed during conversion")
    return (model, spec, mj) if with_mj_spec else (model, spec)
