"""A fixed-base robot: nq=nv=nu=6, with privately supplied whole-robot geometry.

Only an in-memory URDF copy is edited. All chassis joints are fixed at their
nominal standing pose, the fingers at their nominal opening, and the base is
welded to the world by the absence of a free joint. No base action is possible.
Gains/inertias remain simulation assumptions, not hardware identification.
"""
from dataclasses import dataclass
from pathlib import Path
import json
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from supermimic.assets import resolve_wfarm_assets


@dataclass
class ArmSpec:
    names: tuple
    qadr: np.ndarray
    dadr: np.ndarray
    actuators: np.ndarray
    low: np.ndarray
    high: np.ndarray
    effort: np.ndarray
    velocity: np.ndarray
    kp: np.ndarray
    kd: np.ndarray
    tcp: int
    base: int
    wheel_geoms: np.ndarray


def build_model(asset_dir, timestep=.005):
    root = resolve_wfarm_assets(asset_dir)
    config = json.loads((root / "config/robot.json").read_text())
    tcp = json.loads((root / "config/robot_calibration.json").read_text())["tcp"]
    source = root / "urdf/wf-arm-tbot.urdf"
    urdf = ET.parse(source).getroot()
    names = tuple(config["leg_wheel_arm_joint_order"][-6:])
    joints = {j.get("name"): j for j in urdf.findall("joint")}
    for link in urdf.findall("link"):
        for visual in list(link.findall("visual")):
            link.remove(visual)
        for i, collision in enumerate(link.findall("collision")):
            if not collision.get("name"):
                collision.set("name", f"{link.get('name')}_collision_{i}")
    for mesh in urdf.findall(".//mesh"):
        mesh.set("filename", str((source.parent / mesh.get("filename")).resolve()))
    for name, joint in joints.items():
        if joint.get("type") == "fixed" or name in names:
            continue
        value = float(config["default_joint_pos"][name])
        origin = joint.find("origin")
        if origin is None:
            origin = ET.SubElement(joint, "origin")
        xyz = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
        rotation = Rotation.from_euler("xyz", np.fromstring(origin.get("rpy", "0 0 0"), sep=" "))
        axis = np.fromstring(joint.find("axis").get("xyz", "1 0 0"), sep=" ")
        if joint.get("type") == "prismatic":
            xyz += rotation.apply(axis * value)
        else:
            rotation = rotation * Rotation.from_rotvec(axis * value)
        origin.set("xyz", " ".join(map(str, xyz)))
        origin.set("rpy", " ".join(map(str, rotation.as_euler("xyz"))))
        joint.set("type", "fixed")
        for child in list(joint):
            if child.tag not in ("origin", "parent", "child"):
                joint.remove(child)
    extension = ET.SubElement(urdf, "mujoco")
    ET.SubElement(extension, "compiler", fusestatic="false", discardvisual="true")
    mj = mujoco.MjSpec.from_string(ET.tostring(urdf, encoding="unicode"))
    mj.option.timestep = timestep
    mj.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    mj.option.iterations = 50
    mj.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
                          size=[0, 0, .1], friction=[1., .005, .0001])
    rotation = Rotation.from_euler("xyz", tcp["rpy_rad"]).as_quat()
    mj.body(tcp["parent_link"]).add_site(name="tcp", pos=tcp["xyz_m"],
                                        quat=rotation[[3, 0, 1, 2]], size=[.008]*3)
    for joint in urdf.findall("joint"):
        mj.add_exclude(bodyname1=joint.find("parent").get("link"),
                       bodyname2=joint.find("child").get("link"))
    limits = [joints[name].find("limit") for name in names]
    effort = np.array([float(v.get("effort")) for v in limits])
    for name, force in zip(names, effort):
        mj.add_actuator(name=f"motor_{name}", target=name,
                        trntype=mujoco.mjtTrn.mjTRN_JOINT, gear=[1, 0, 0, 0, 0, 0],
                        ctrllimited=True, ctrlrange=[-force, force],
                        forcelimited=True, forcerange=[-force, force])
    model = mj.compile()
    if (model.nq, model.nv, model.nu) != (6, 6, 6):
        raise ValueError("fixed arm must have exactly six physical DOFs and actuators")
    joint_ids = np.array([model.joint(name).id for name in names])
    qadr = model.jnt_qposadr[joint_ids].copy()
    dadr = model.jnt_dofadr[joint_ids].copy()
    wheels = np.array([model.geom(f"{leg}_foot_collision").id for leg in ("fl", "fr", "rl", "rr")])
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    bottoms = []
    for geom in wheels:
        axis = data.geom_xmat[geom].reshape(3, 3)[:, 2]
        radius, half_width = model.geom_size[geom, :2]
        extent = radius * np.sqrt(max(0, 1-axis[2]**2)) + half_width*abs(axis[2])
        bottoms.append(data.geom_xpos[geom, 2] - extent)
    base = model.body("base").id
    model.body_pos[base, 2] = -min(bottoms) + .002
    model.dof_armature[dadr] = .02
    model.dof_damping[dadr] = .1
    low = np.array([float(v.get("lower")) for v in limits])
    high = np.array([float(v.get("upper")) for v in limits])
    operating = json.loads((root / "config/operating_limits.json").read_text())
    low[4] = max(low[4], float(operating["arm_joint5_lower_rad"]))
    high[4] = min(high[4], float(operating["arm_joint5_upper_rad"]))
    if low[4] >= high[4]:
        raise ValueError("private wrist operating limits have an empty interval")
    spec = ArmSpec(names, qadr, dadr,
        np.array([model.actuator(f"motor_{name}").id for name in names]), low, high,
        effort, np.array([float(v.get("velocity")) for v in limits]),
        np.array([50., 70., 60., 30., 20., 15.]), np.array([2., 2., 1.5, 1., .8, .5]),
        model.site("tcp").id, base, wheels)
    return model, spec


class Simulator:
    def __init__(self, asset_dir, num_envs=8, dt=.02, timestep=.005, seed=0):
        self.model, self.spec = build_model(asset_dir, timestep)
        self.num_envs, self.dt = num_envs, dt
        self.substeps = round(dt/timestep)
        if self.substeps < 1 or not np.isclose(self.substeps*timestep, dt):
            raise ValueError("dt must be a multiple of timestep")
        self.rng = np.random.default_rng(seed)
        self.data = [mujoco.MjData(self.model) for _ in range(num_envs)]
        self.previous = np.zeros((num_envs, 6))
        self.torque = np.zeros_like(self.previous)
        self.failed = np.zeros(num_envs, dtype=bool)
        self.dynamic_bodies = set()
        for body in range(1, self.model.nbody):
            if self.model.body_dofnum[body] or int(self.model.body_parentid[body]) in self.dynamic_bodies:
                self.dynamic_bodies.add(body)
        self.reset()

    def reset(self, ids=None, q=None):
        ids = np.arange(self.num_envs) if ids is None else np.asarray(ids)
        positions = np.zeros((len(ids), 6)) if q is None else np.asarray(q).reshape(len(ids), 6)
        if not np.isfinite(positions).all() or np.any(positions < self.spec.low) or np.any(positions > self.spec.high):
            raise ValueError("reset pose outside operating limits")
        for row, i in enumerate(ids):
            d = self.data[i]
            mujoco.mj_resetData(self.model, d)
            d.qpos[self.spec.qadr] = positions[row]
            mujoco.mj_forward(self.model, d)
        self.previous[ids] = positions
        self.torque[ids] = 0
        self.failed[ids] = False
        return self.state()

    def state(self):
        s = self.spec
        return np.stack([np.r_[d.qpos[s.qadr], d.qvel[s.dadr]] for d in self.data]).astype(np.float32)

    def pose(self):
        out = []
        for d in self.data:
            q = np.empty(4)
            mujoco.mju_mat2Quat(q, d.site_xmat[self.spec.tcp])
            out.append(np.r_[d.site_xpos[self.spec.tcp], q])
        return np.array(out, dtype=np.float32)

    def collision(self, d):
        # Static wheel/floor contacts are irrelevant to fixed-base training.
        # Every non-adjacent contact involving an arm descendant is a failure.
        for c in d.contact:
            if c.dist > 0:
                continue
            for geom in (c.geom1, c.geom2):
                body = int(self.model.geom_bodyid[geom])
                if body in self.dynamic_bodies:
                    return True
        return False

    def step(self, target):
        target = np.asarray(target, dtype=float)
        if target.shape != (self.num_envs, 6) or not np.isfinite(target).all():
            raise ValueError("expected exactly six finite joint position targets")
        s = self.spec
        target = np.clip(target, s.low, s.high)
        self.failed[:] = False
        for i, d in enumerate(self.data):
            for _ in range(self.substeps):
                torque = s.kp*(target[i]-d.qpos[s.qadr]) - s.kd*d.qvel[s.dadr]
                torque[(np.abs(d.qvel[s.dadr]) > s.velocity) & (torque*d.qvel[s.dadr] > 0)] = 0
                torque = np.clip(torque, -s.effort, s.effort)
                d.ctrl[s.actuators] = torque
                mujoco.mj_step(self.model, d)
                self.failed[i] |= self.collision(d)
                self.torque[i] = torque
            mujoco.mj_forward(self.model, d)
            self.failed[i] |= self.collision(d)
        self.previous[:] = target
        state = self.state()
        self.failed |= (~np.isfinite(state).all(-1) |
            (state[:, :6] < s.low-.04).any(-1) | (state[:, :6] > s.high+.04).any(-1))
        return state
