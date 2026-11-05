"""Analytic, differentiable fixed-base FK, observations and task rewards."""
import numpy as np
import torch
from torch import nn
import mujoco

from .rotations import (quat_apply, quat_apply_inverse, quat_mul, quat_conjugate,
                        quat_from_rotvec, quat_normalize, rot6d, rotvec_from_quat)
from .clearance import SurfaceClearance


class Task(nn.Module):
    input_dim = 42

    def __init__(self, model, spec, dt=.02, speed=.8, clearance_points=32):
        super().__init__()
        self.dt, self.speed = dt, speed
        self.links = []
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        # Collapse fixed body transforms; only six joints need runtime recursion.
        frame = [0]*model.nbody
        local_p = np.zeros((model.nbody, 3))
        local_q = np.zeros((model.nbody, 4)); local_q[:, 0] = 1
        origin_p, origin_q, axes, anchors, refs = [], [], [], [], []
        def rotate(q, v):
            out = np.empty(3); mujoco.mju_rotVecQuat(out, v, q); return out
        def multiply(a, b):
            out = np.empty(4); mujoco.mju_mulQuat(out, a, b); return out
        for b in range(1, model.nbody):
            parent = int(model.body_parentid[b])
            p = local_p[parent] + rotate(local_q[parent], model.body_pos[b])
            q = multiply(local_q[parent], model.body_quat[b])
            if model.body_jntnum[b]:
                j = int(model.body_jntadr[b])
                if model.body_jntnum[b] != 1 or model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE:
                    raise ValueError("expected only the six scalar arm hinge joints")
                index = int(np.flatnonzero(spec.qadr == model.jnt_qposadr[j])[0])
                self.links.append((frame[parent], index))
                origin_p.append(p); origin_q.append(q)
                axes.append(model.jnt_axis[j]); anchors.append(model.jnt_pos[j])
                refs.append(model.qpos0[model.jnt_qposadr[j]])
                frame[b] = len(self.links)
            else:
                frame[b] = frame[parent]; local_p[b] = p; local_q[b] = q
        def buf(name, values, dtype=torch.float32):
            self.register_buffer(name, torch.as_tensor(np.asarray(values).copy(), dtype=dtype))
        for name, values in (("origin_p", origin_p), ("origin_q", origin_q), ("axes", axes),
                ("anchors", anchors), ("refs", refs), ("body_local_p", local_p[1:]),
                ("body_local_q", local_q[1:]), ("low", spec.low), ("high", spec.high),
                ("velocity", spec.velocity), ("effort", spec.effort), ("kp", spec.kp), ("kd", spec.kd),
                ("tcp_pos", model.site_pos[spec.tcp]), ("tcp_quat", model.site_quat[spec.tcp]),
                ("base_position", model.body_pos[spec.base])):
            buf(name, values)
        buf("frames", frame[1:], torch.long)
        self.tcp_body = int(model.site_bodyid[spec.tcp])-1
        self.clearance = (SurfaceClearance(model, wheel_geom_ids=spec.wheel_geoms,
                         maximum=clearance_points) if clearance_points else None)

    def fk(self, state):
        position = torch.zeros_like(state[..., :3])
        identity = torch.cat((torch.ones_like(state[..., :1]), position), -1)
        positions, rotations, velocities, angular = [position], [identity], [position], [position]
        for k, (parent, index) in enumerate(self.links):
            p0 = positions[parent] + quat_apply(rotations[parent], self.origin_p[k].expand_as(position))
            r0 = quat_mul(rotations[parent], self.origin_q[k].expand_as(identity))
            axis = quat_apply(r0, self.axes[k].expand_as(position))
            anchor = p0 + quat_apply(r0, self.anchors[k].expand_as(position))
            rotation = quat_mul(r0, quat_from_rotvec((state[..., index:index+1]-self.refs[k])*self.axes[k]))
            p = anchor - quat_apply(rotation, self.anchors[k].expand_as(position))
            joint_w = axis*state[..., index+6:index+7]
            v = velocities[parent] + torch.cross(angular[parent], p-positions[parent], dim=-1)
            v = v + torch.cross(joint_w, p-anchor, dim=-1)
            positions.append(p); rotations.append(rotation); velocities.append(v)
            angular.append(angular[parent]+joint_w)
        p = torch.stack(positions, -2)[..., self.frames, :]
        r = torch.stack(rotations, -2)[..., self.frames, :]
        offset = quat_apply(r, self.body_local_p.expand_as(p))
        pos, quat = p+offset, quat_mul(r, self.body_local_q.expand_as(r))
        body_v = torch.stack(velocities, -2)[..., self.frames, :]
        body_w = torch.stack(angular, -2)[..., self.frames, :]
        body_v = body_v + torch.cross(body_w, offset, dim=-1)
        tcp_offset = quat_apply(quat[..., self.tcp_body, :], self.tcp_pos.expand_as(position))
        tcp_p = pos[..., self.tcp_body, :] + tcp_offset
        tcp_q = quat_mul(quat[..., self.tcp_body, :], self.tcp_quat.expand_as(identity))
        tcp_w = body_w[..., self.tcp_body, :]
        tcp_v = body_v[..., self.tcp_body, :] + torch.cross(tcp_w, tcp_offset, dim=-1)
        return dict(pos=pos, quat=quat, tcp_pose=torch.cat((tcp_p, tcp_q), -1), tcp_v=tcp_v, tcp_w=tcp_w)

    def errors(self, state, goal, bodies=None):
        b = self.fk(state) if bodies is None else bodies
        current = b["tcp_pose"]
        relative = quat_mul(quat_conjugate(current[..., 3:7]), quat_normalize(goal[..., 3:7]))
        ep = goal[..., :3]-current[..., :3]
        er = rotvec_from_quat(relative)
        ev, ew = goal[..., 7:10]-b["tcp_v"], goal[..., 10:13]-b["tcp_w"]
        return ep, er, ev, ew

    def observation(self, state, previous, goal, bodies=None):
        b = self.fk(state) if bodies is None else bodies
        ep, er, _, _ = self.errors(state, goal, b)
        relative = quat_mul(quat_conjugate(b["tcp_pose"][..., 3:7]), goal[..., 3:7])
        center, scale = (self.low+self.high)/2, (self.high-self.low)/2
        return torch.cat(((state[..., :6]-center)/scale, state[..., 6:]/self.velocity,
            (previous-center)/scale, ep/.15, rot6d(relative),
            (goal[..., :3]-self.base_position)/.8, rot6d(goal[..., 3:7]),
            goal[..., 7:10]/.3, goal[..., 10:13]), -1)

    def command(self, actor, state, previous, goal, bodies=None):
        # Six network outputs are bounded target velocities; the six exported
        # motor commands are rate-limited joint position targets in radians.
        rate = torch.tanh(actor(self.observation(state, previous, goal, bodies)))
        return torch.minimum(torch.maximum(previous + self.dt*self.speed*rate, self.low), self.high)

    def reward(self, state, action, previous, goal, bodies=None):
        b = self.fk(state) if bodies is None else bodies
        ep, er, ev, ew = self.errors(state, goal, b)
        pe, re = ep.square().sum(-1), er.square().sum(-1)
        reward = (2*torch.exp(-pe/.03**2) + torch.exp(-pe/.15**2) - 2*pe
            + 2*torch.exp(-re/.15**2) + torch.exp(-re/.6**2) - .15*re
            + .3*torch.exp(-ev.square().sum(-1)/.08**2)
            + .2*torch.exp(-ew.square().sum(-1)/.2**2))
        margin = torch.minimum(state[..., :6]-self.low, self.high-state[..., :6])
        reward = reward - .3*((.05-margin).clamp_min(0)/.05).square().mean(-1)
        torque = self.kp*(action-state[..., :6])-self.kd*state[..., 6:]
        reward = reward - .01*(torque/self.effort).square().mean(-1)
        reward = reward - .01*((action-previous)/(self.dt*self.speed)).square().mean(-1)
        if self.clearance is not None:
            clearance = self.clearance(b)
            reward = reward - ((.02-clearance["robot"].amin(-1)).clamp_min(0)/.02).square().clamp_max(100)
            reward = reward - ((.03-clearance["ground"].amin(-1)).clamp_min(0)/.03).square().clamp_max(100)
        return reward

    def success(self, state, goal):
        ep, er, ev, ew = self.errors(state, goal)
        return ((ep.norm(dim=-1) < .03) & (er.norm(dim=-1) < .15)
                & (ev.norm(dim=-1) < .08) & (ew.norm(dim=-1) < .2))


def advance_goal(goal, dt):
    """Short-horizon constant WORLD-frame twist; quaternion scalar first."""
    return torch.cat((goal[..., :3]+dt*goal[..., 7:10],
        quat_normalize(quat_mul(quat_from_rotvec(dt*goal[..., 10:13]), goal[..., 3:7])),
        goal[..., 7:13]), -1)
