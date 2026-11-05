"""Goal-conditioned whole-body task shared by simulation and imagined rollouts.

The state is [base xyz(3), quaternion wxyz(4), canonical joints(22), world linear
velocity(3), body angular velocity(3), joint velocity(22)]. Wheel positions remain
in simulator state but are deliberately absent from observations and rewards.
All reward terms below are analytic Torch functions, including contact proxies;
they do not claim to reconstruct the simulator's nonsmooth contact forces.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .math import (following_frame, following_velocity, normalize, projected_gravity, quat_angle, quat_apply,
                   quat_apply_inverse, quat_error, quat_from_angle_axis,
                   quat_from_rotvec, quat_mul, quat_normalize, rot6d)


@dataclass
class WholeBodyRewardCfg:
    goal_orientation_frame: str = "tcp"  # Legacy error, or absolute base-frame target.
    constraint_observation: bool = False
    base_command_conditioning: bool = False
    goal_follow_frame: str = "body"  # Legacy checkpoints retain full-body following.
    full_goal_orientation: bool = False
    goal_program: bool = False
    program_context: bool = False
    program_steps: int = 600
    base_command_fraction: float = .5
    base_command_ranges: tuple[float, ...] = (.10, 0., .18)
    base_command_resample_steps: int = 150
    base_command_weight: float = 2.
    base_linear_scale: float = .08
    base_linear_scales: tuple[float,...] = ()
    base_linear_weights: tuple[float,...] = ()
    base_angular_scale: float = .18
    base_straight_fraction: float = 0.
    base_endpoint_fraction: float = 0.
    success_base_linear: float = .025
    success_base_angular: float = .06
    # Reward is per control step; all penalties use dimensionless normalizations.
    position_scales: tuple[float, ...] = (0.08, 0.25, 0.8)
    position_weights: tuple[float, ...] = (2.0, 1.0, 0.5)
    orientation_scale: float = 0.5
    orientation_weight: float = 0.7
    # Empty tuples retain the exact legacy single-scale expression/checkpoints.
    orientation_scales: tuple[float, ...] = ()
    orientation_weights: tuple[float, ...] = ()
    arrival_speed_scale: float = 0.15
    arrival_weight: float = 0.3
    tilt_free_angle: float = 0.35
    tilt_weight: float = 0.35
    tilt_barrier_weight: float = 0.
    tilt_barrier_scale: float = .1  # Radians beyond tilt_free_angle for one normalized unit.
    support_margin: float = 0.015
    support_scale: float = 0.1
    support_weight: float = 0.25
    contact_band: float = 0.025
    contact_temperature: float = 0.01
    penetration_scale: float = 0.025
    penetration_weight: float = 0.4
    contact_weight: float = 0.15
    slip_speed_scale: float = 0.5
    lateral_slip_weight: float = 0.15
    rolling_slip_weight: float = 0.1
    joint_limit_weight: float = 0.3
    joint_soft_margin: float = 0.
    joint_soft_limit_weight: float = 0.
    # Experimental COM-distance proxy can penalize adjacent CAD parts; keep
    # disabled unless calibrated. Actual simulator forbidden contacts still fail.
    self_collision_weight: float = 0.
    settling_weight: float = 0.
    wrist_safe_lower: float | None = None
    wrist_safe_upper: float | None = None
    wrist_limit_weight: float = 0.
    action_rate_weight: float = 0.01
    torque_weight: float = 0.005
    power_weight: float = 0.005
    collision_weight: float = 0.3
    collision_scale: float = 0.04
    geometry_collision_weight: float = 0.
    geometry_ground_weight: float = 0.
    geometry_clearance_margin: float = .025
    geometry_clearance_scale: float = .025
    geometry_surface_points: int = 128
    arm_proxy_radius: float = 0.04
    leg_proxy_radius: float = 0.04
    base_proxy_radius: float = 0.15
    ground_weight: float = 0.3
    min_base_height: float = 0.18
    max_base_height: float = 1.1
    min_upright_cosine: float = 0.35
    termination_temperature: float = 0.025
    success_position: float = 0.03
    success_orientation: float = 0.15
    success_linear_speed: float = 0.08
    success_angular_speed: float = 0.2
    minimum_goal_height: float = 0.10
    # Translation radius (base frame); pose noise grows separately with curriculum.
    goal_translation_radii: tuple[float, ...] = (0.06, 0.18, 0.40, 0.75)
    goal_rotation_radii: tuple[float, ...] = (0.08, 0.25, 0.55, 1.0)


RewardCfg = WholeBodyRewardCfg


class WholeBodyTask(nn.Module):
    """Robot-specific analytic task, with model-derived differentiable FK.

    ``spec.qpos_indices`` and ``spec.dof_indices`` map the canonical policy order
    into MuJoCo's tree order. This keeps the action/state ordering independent of
    XML layout. Buffers follow normal ``nn.Module.to(device/dtype)`` semantics.
    """

    state_dim, action_dim, obs_dim, goal_dim, input_dim = 57, 22, 72, 12, 84

    def __init__(self, model, spec, device="cpu", reward_cfg=None):
        super().__init__()
        self.cfg = reward_cfg or WholeBodyRewardCfg()
        if self.cfg.goal_follow_frame not in ("body","heading"):
            raise ValueError("goal following frame must be body or heading")
        self.obs_dim=76 if self.cfg.constraint_observation else 72
        if self.cfg.goal_program and (not self.cfg.full_goal_orientation or not self.cfg.base_command_conditioning):
            raise ValueError('goal program requires full orientation and base command features')
        if self.cfg.program_context and not self.cfg.goal_program:raise ValueError('program context requires goal-program policy')
        self.goal_dim=12+6*int(self.cfg.full_goal_orientation)+4*int(self.cfg.base_command_conditioning)+int(self.cfg.goal_program)+27*int(self.cfg.program_context)
        self.command_size=37 if self.cfg.program_context else (12 if self.cfg.goal_program else (11 if self.cfg.base_command_conditioning else 7))
        self.input_dim=self.obs_dim+self.goal_dim
        self.spec = spec
        self.nbody = int(model.nbody) - 1
        self.base_body = int(spec.base_body) - 1
        self.tcp_body = int(spec.tcp_body) - 1
        self.body_names = [model.body(i).name for i in range(1, model.nbody)]
        qaddresses = np.asarray(spec.qpos_indices)
        daddresses = np.asarray(spec.dof_indices)
        self.links = []
        axes, anchors, refs = [], [], []
        for b in range(1, model.nbody):
            parent = int(model.body_parentid[b]) - 1
            nj = int(model.body_jntnum[b])
            if nj > 1:
                raise ValueError("WholeBodyTask supports one scalar joint per body")
            typ, idx = -1, -1
            axis, anchor, ref = np.zeros(3), np.zeros(3), 0.
            if nj:
                j = int(model.body_jntadr[b])
                typ = int(model.jnt_type[j])
                if typ == 0:
                    if parent != -1:
                        raise ValueError("free joint must belong to root")
                elif typ in (2, 3):
                    found = np.flatnonzero(qaddresses == int(model.jnt_qposadr[j]))
                    if len(found) != 1:
                        raise ValueError(f"joint {model.joint(j).name} absent from canonical spec")
                    idx = int(found[0])
                    if daddresses[idx] != int(model.jnt_dofadr[j]):
                        raise ValueError("qpos/dof mapping is inconsistent")
                    axis, anchor = model.jnt_axis[j], model.jnt_pos[j]
                    ref = float(model.qpos0[int(model.jnt_qposadr[j])])
                else:
                    raise ValueError(f"unsupported joint type {typ}")
            if parent >= b - 1:
                raise ValueError("body tree must be topologically ordered")
            self.links.append((parent, typ, idx))
            axes.append(axis)
            anchors.append(anchor)
            refs.append(ref)

        def buf(name, value, dtype=torch.float32):
            self.register_buffer(name, torch.as_tensor(np.asarray(value).copy(), dtype=dtype, device=device))

        buf("body_pos", model.body_pos[1:])
        buf("body_quat", model.body_quat[1:])
        buf("body_com_local", model.body_ipos[1:])
        buf("body_mass", model.body_mass[1:])
        buf("joint_axes", axes)
        buf("joint_anchors", anchors)
        buf("joint_refs", refs)
        buf("q0", spec.q0)
        buf("low", spec.low)
        buf("high", spec.high)
        buf("effort", spec.effort)
        buf("velocity", spec.velocity)
        buf("kp", spec.kp)
        buf("kd", spec.kd)
        buf("action_scale", spec.action_scale)
        # A module buffer follows TorchScript map_location; constructing this
        # tensor inside goal_features fixes the traced allocation to the CPU.
        buf("command_feature_scale", [.2,.2,.4])
        buf("wheel_indices", spec.wheel_indices, torch.long)
        nonwheel = [i for i in range(22) if i not in set(np.asarray(spec.wheel_indices).tolist())]
        buf("nonwheel_indices", nonwheel, torch.long)
        buf("wheel_body_indices", np.asarray(spec.wheel_body_ids) - 1, torch.long)
        buf("wheel_geom_pos", model.geom_pos[np.asarray(spec.wheel_geom_ids)])
        buf("tcp_local_pos", spec.tcp_local_pos)
        buf("tcp_local_quat", spec.tcp_local_quat)
        buf("wheel_radius", spec.wheel_radius)
        self.default_height = float(spec.default_height)
        wheel_joint_bodies = [next(i for i, (_, _, j) in enumerate(self.links) if j == k)
                              for k in np.asarray(spec.wheel_indices)]
        buf("wheel_joint_body_indices", wheel_joint_bodies, torch.long)
        # Conservative sphere proxies, explicitly a smooth learning aid rather than
        # an exact substitute for the simulator's full collision geometry.
        arm = [i for i, n in enumerate(self.body_names)
               if n in ("piper_link3", "piper_link4", "piper_link5", "piper_link6", "piper_gripper_base")]
        legs = [i for i, n in enumerate(self.body_names)
                if n.endswith(("_hip", "_thigh", "_calf"))]
        obstacles = [self.base_body] + legs
        buf("collision_arm", arm, torch.long)
        buf("collision_obstacles", obstacles, torch.long)
        pairs=[(int(i),int(j)) for k,i in enumerate(arm) for j in arm[k+1:]
               if self.links[int(j)][0]!=int(i) and self.links[int(i)][0]!=int(j)]
        buf('self_collision_pairs',np.asarray(pairs,dtype=int).reshape(-1,2),torch.long)
        buf("obstacle_radii", [self.cfg.base_proxy_radius] + [self.cfg.leg_proxy_radius]*len(legs))
        self.surface_clearance=None
        if self.cfg.geometry_collision_weight or self.cfg.geometry_ground_weight:
            from .geometry_clearance import SurfaceClearance
            self.surface_clearance=SurfaceClearance(model,wheel_geom_ids=spec.wheel_geom_ids,
                maximum=self.cfg.geometry_surface_points).to(self.device)

    @property
    def device(self):
        return self.q0.device

    @property
    def dtype(self):
        return self.q0.dtype

    @staticmethod
    def split(s):
        return s[..., :3], s[..., 3:7], s[..., 7:29], s[..., 29:32], s[..., 32:35], s[..., 35:57]

    def fk(self, s):
        """All body-origin poses/velocities, COMs, joint world axes, and TCP pose.

        ``ang_vel`` is in world coordinates; the state's base angular velocity is
        in the base frame. Output body indices exclude MuJoCo's world body.
        """
        rp, rq, q, rv, rw, qd = self.split(s)
        rq = quat_normalize(rq)
        p, r, v, w, axes = [], [], [], [], []
        for i, (parent, typ, idx) in enumerate(self.links):
            if typ == 0:
                bp, br, bv, bw = rp, rq, rv, quat_apply(rq, rw)
                axis_w = torch.zeros_like(rp)
            else:
                if parent < 0:
                    raise ValueError("only the floating robot tree is supported")
                bp = p[parent] + quat_apply(r[parent], self.body_pos[i].expand_as(rp))
                br = quat_mul(r[parent], self.body_quat[i].expand_as(rq))
                bw = w[parent]
                bv = v[parent] + torch.cross(bw, bp-p[parent], dim=-1)
                axis_w = quat_apply(br, self.joint_axes[i].expand_as(rp))
                if typ == 3:
                    anchor = bp + quat_apply(br, self.joint_anchors[i].expand_as(rp))
                    br = quat_mul(br, quat_from_angle_axis(q[..., idx]-self.joint_refs[i],
                                                          self.joint_axes[i].expand_as(rp)))
                    bp = anchor - quat_apply(br, self.joint_anchors[i].expand_as(rp))
                    joint_w = axis_w * qd[..., idx, None]
                    bv = v[parent] + torch.cross(w[parent], bp-p[parent], dim=-1) + torch.cross(joint_w, bp-anchor, dim=-1)
                    bw = w[parent] + joint_w
                elif typ == 2:
                    displacement = axis_w * (q[..., idx]-self.joint_refs[i])[..., None]
                    bp = bp + displacement
                    bv = bv + torch.cross(bw, displacement, dim=-1) + axis_w*qd[..., idx, None]
            p.append(bp); r.append(br); v.append(bv); w.append(bw); axes.append(axis_w)
        pos, quat = torch.stack(p, -2), torch.stack(r, -2)
        vel, ang = torch.stack(v, -2), torch.stack(w, -2)
        com = pos + quat_apply(quat, self.body_com_local.expand_as(pos))
        tcp_offset = quat_apply(quat[..., self.tcp_body, :], self.tcp_local_pos.expand_as(rp))
        tcp_pos = pos[..., self.tcp_body, :] + tcp_offset
        tcp_quat = quat_mul(quat[..., self.tcp_body, :], self.tcp_local_quat.expand_as(rq))
        tcp_vel = vel[..., self.tcp_body, :] + torch.cross(ang[..., self.tcp_body, :], tcp_offset, dim=-1)
        return {"pos": pos, "quat": quat, "lin_vel": vel, "ang_vel": ang,
                "com": com, "total_com": (com*self.body_mass[..., None]).sum(-2)/self.body_mass.sum(),
                "axes": torch.stack(axes, -2), "tcp_pos": tcp_pos, "tcp_quat": tcp_quat,
                "tcp_lin_vel": tcp_vel, "tcp_ang_vel": ang[..., self.tcp_body, :]}

    def tcp_pose(self, s):
        b = self.fk(s)
        return torch.cat((b["tcp_pos"], b["tcp_quat"]), -1)

    def observation(self, s, prev_action, bodies=None):
        p, q, joints, v, w, qd = self.split(s)
        q = quat_normalize(q)
        # Per-joint normalization keeps radian joints and high-speed wheel drives
        # commensurate. The actor consumes this vector without another scaler.
        ji = self.nonwheel_indices
        core=torch.cat(((joints[..., ji]-self.q0[ji])/self.action_scale[ji].clamp_min(.1),
                          qd/self.velocity.clamp_min(1.), quat_apply_inverse(q, v), w/2,
                          projected_gravity(q), p[..., 2:3]/self.default_height), -1)
        if self.cfg.constraint_observation:
            b=self.fk(s) if bodies is None else bodies
            core=torch.cat((core,b['axes'][...,self.wheel_joint_body_indices,2]/.1),dim=-1)
        return torch.cat((core,prev_action),dim=-1)

    def goal_features(self, s, goal, bodies=None):
        b = self.fk(s) if bodies is None else bodies
        base_q = quat_normalize(s[..., 3:7])
        features=torch.cat((quat_apply_inverse(base_q, goal[..., :3]-b["tcp_pos"]),
                          rot6d(quat_error(base_q if self.cfg.goal_orientation_frame=="base" else b["tcp_quat"], goal[..., 3:7])),
                          quat_apply_inverse(base_q, goal[..., :3]-s[..., :3])), -1)
        if self.cfg.full_goal_orientation:
            features=torch.cat((features,rot6d(quat_error(base_q,goal[...,3:7]))),-1)
        if self.cfg.goal_program:
            goal=self.command_goal(goal)
            features=torch.cat((features,goal[...,11:12]),-1)
        if self.cfg.program_context:
            origin=goal[...,12:19];origin_q=quat_normalize(origin[...,3:7])
            original_target=torch.cat((quat_apply_inverse(origin_q,goal[...,:3]-origin[...,:3]),rot6d(quat_error(origin_q,goal[...,3:7]))),-1)
            # Following-body commands retain a body-relative target as the
            # chassis translates/rotates. World-fixed tasks retain their origin.
            following_target=features[...,9:18]
            if self.cfg.goal_follow_frame=="heading":
                frame_pos,frame_q=following_frame(s,"heading")
                # Initial base height, rather than current height, anchors the
                # program input while the legs help solve the reaching task.
                frame_pos=torch.cat((frame_pos[...,:2],origin[...,2:3]),-1)
                following_target=torch.cat((quat_apply_inverse(frame_q,goal[...,:3]-frame_pos),
                    rot6d(quat_error(frame_q,goal[...,3:7]))),-1)
            original_target=torch.where(goal[...,10:11]>.5,following_target,original_target)
            features=torch.cat((features,original_target,goal[...,19:37]),-1)
        if self.cfg.base_command_conditioning:
            goal=self.command_goal(goal)
            features=torch.cat((features,goal[...,7:10]/self.command_feature_scale,goal[...,10:11]),-1)
        return features

    def command_goal(self, goal):
        if self.cfg.base_command_conditioning and goal.shape[-1]==7:
            goal=torch.cat((goal,torch.zeros_like(goal[...,:4])),-1)
        if self.cfg.goal_program and goal.shape[-1]==11:
            goal=torch.cat((goal,torch.zeros_like(goal[...,:1])),-1)
        if self.cfg.program_context and goal.shape[-1]==12:
            goal=torch.cat((goal,torch.zeros((*goal.shape[:-1],25),device=goal.device,dtype=goal.dtype)),-1)
        return goal

    def initialize_goal(self,state,goal):
        goal=self.command_goal(goal)
        if not self.cfg.program_context:return goal
        joints=(state[...,7:29][...,self.nonwheel_indices]-self.q0[self.nonwheel_indices])/self.action_scale[self.nonwheel_indices].clamp_min(.1)
        return torch.cat((goal[...,:12],state[...,:7],joints),-1)

    def step_goal(self,before,after,goal,*,advance=True):
        moved=self.transport_goal(before,after,goal)
        if self.cfg.goal_program and advance:
            moved=torch.cat((moved[...,:11],(moved[...,11:12]+1./self.cfg.program_steps).clamp_max(1.),moved[...,12:]),-1)
        return moved

    def transport_goal(self, before, after, goal):
        if not self.cfg.base_command_conditioning:return goal
        goal=self.command_goal(goal)
        p0,q0=following_frame(before,self.cfg.goal_follow_frame)
        p1,q1=following_frame(after,self.cfg.goal_follow_frame)
        local=quat_apply_inverse(q0,goal[...,:3]-p0)
        position=p1+quat_apply(q1,local)
        inverse=q0*goal.new_tensor([1.,-1.,-1.,-1.])
        orientation=quat_mul(quat_mul(q1,inverse),goal[...,3:7])
        moved=torch.cat((position,orientation,goal[...,7:]),-1)
        return torch.where(goal[...,10:11]>.5,moved,goal)

    def base_command_error(self, state, goal):
        goal=self.command_goal(goal)
        velocity=quat_apply_inverse(quat_normalize(state[...,3:7]),state[...,29:32])
        actual=torch.stack((velocity[...,0],velocity[...,1],state[...,34]),-1)
        return actual-goal[...,7:10]

    def tracking_velocities(self, state, bodies, goal):
        linear,angular=bodies['tcp_lin_vel'],bodies['tcp_ang_vel']
        if not self.cfg.base_command_conditioning:return linear,angular
        goal=self.command_goal(goal);mode=goal[...,10:11]
        frame_pos,_=following_frame(state,self.cfg.goal_follow_frame)
        frame_velocity,world_omega=following_velocity(state,self.cfg.goal_follow_frame)
        follow_velocity=frame_velocity+torch.cross(world_omega,bodies['tcp_pos']-frame_pos,dim=-1)
        return linear-mode*follow_velocity,angular-mode*world_omega

    def actor_input(self, s, prev_action, goal, *, bodies=None):
        if self.cfg.program_context and goal.shape[-1]!=37:
            raise ValueError('initialize the goal origin once per reaching program, before Actor inference')
        if self.cfg.constraint_observation:
            b=self.fk(s) if bodies is None else bodies
            return torch.cat((self.observation(s,prev_action,b),self.goal_features(s,goal,b)),-1)
        return torch.cat((self.observation(s, prev_action,bodies), self.goal_features(s, goal,bodies)), -1)

    def process_action(self, action):
        """Same executable normalized set-points as the MuJoCo controller.

        With asymmetric source limits, q0 +/- scale may extend beyond a joint's
        range. Imagination must fit/execute the clipped target rather than learn
        an impossible command. Wheel coordinates instead bound angular speed.
        """
        lower = (self.low-self.q0)/self.action_scale
        upper = (self.high-self.q0)/self.action_scale
        wheel_limit = self.velocity[self.wheel_indices]/self.action_scale[self.wheel_indices]
        lower = lower.scatter(0, self.wheel_indices, -wheel_limit)
        upper = upper.scatter(0, self.wheel_indices, wheel_limit)
        return torch.minimum(torch.maximum(action.clamp(-1, 1), lower), upper)

    executed_action = process_action

    def wheel_geometry(self, bodies):
        i = self.wheel_body_indices
        offset = quat_apply(bodies["quat"][..., i, :], self.wheel_geom_pos.expand_as(bodies["pos"][..., i, :]))
        centers = bodies["pos"][..., i, :] + offset
        vel = bodies["lin_vel"][..., i, :] + torch.cross(bodies["ang_vel"][..., i, :], offset, dim=-1)
        axle = bodies["axes"][..., self.wheel_joint_body_indices, :]
        vertical = torch.zeros_like(axle); vertical[..., 2] = 1
        lateral = normalize(axle - (axle*vertical).sum(-1, keepdim=True)*vertical)
        rolling = normalize(torch.cross(lateral, vertical, dim=-1))
        ground_offset = -vertical*self.wheel_radius
        contact_v = vel + torch.cross(bodies["ang_vel"][..., i, :], ground_offset, dim=-1)
        return {"centers": centers, "velocities": vel, "axes": axle,
                "gaps": centers[..., 2]-self.wheel_radius,
                "lateral_slip": (contact_v*lateral).sum(-1),
                "rolling_slip": (contact_v*rolling).sum(-1)}

    def reward(self, s_next, action, prev_action, goal, *, bodies=None):
        c = self.cfg
        action = self.process_action(action)
        b = self.fk(s_next) if bodies is None else bodies
        wg = self.wheel_geometry(b)
        p, q, joints, _, _, qd = self.split(s_next)
        pe2 = (goal[..., :3]-b["tcp_pos"]).square().sum(-1)
        ae = quat_angle(quat_error(b["tcp_quat"], goal[..., 3:7]))
        task_linear,task_angular=self.tracking_velocities(s_next,b,goal)
        speed2 = task_linear.square().sum(-1)
        near = torch.exp(-pe2/c.position_scales[0]**2)
        orientation = (sum(w*torch.exp(-ae.square()/sc**2)
                           for w, sc in zip(c.orientation_weights, c.orientation_scales))
                       if c.orientation_scales else
                       c.orientation_weight*torch.exp(-ae.square()/c.orientation_scale**2))
        terms = {"position": sum(w*torch.exp(-pe2/sc**2) for w, sc in zip(c.position_weights, c.position_scales)),
                 "orientation": orientation,
                 "arrival": c.arrival_weight*near*torch.exp(-speed2/c.arrival_speed_scale**2)}
        upright = -projected_gravity(quat_normalize(q))[..., 2]
        terms["tilt"] = -c.tilt_weight*torch.relu(torch.cos(torch.as_tensor(c.tilt_free_angle, device=q.device, dtype=q.dtype))-upright).square()
        if c.tilt_barrier_weight:
            edge=torch.cos(torch.as_tensor(c.tilt_free_angle,device=q.device,dtype=q.dtype))
            scale=edge-torch.cos(torch.as_tensor(c.tilt_free_angle+c.tilt_barrier_scale,device=q.device,dtype=q.dtype))
            excess=torch.relu(edge-upright)/scale.clamp_min(1e-5)
            terms['tilt_barrier']=-c.tilt_barrier_weight*excess.square().clamp_max(25.)
        # A geometric quadrilateral support proxy in ground xy; FL,FR,RR,RL.
        poly = wg["centers"][..., [0, 1, 3, 2], :2]
        edge = torch.roll(poly, -1, -2)-poly
        # Treat wheel mass as balanced about its axle: tiny CAD COM eccentricity
        # must not make the learning reward depend on an arbitrary wheel phase.
        wi = self.wheel_body_indices
        balanced_com = b["total_com"]+((wg["centers"]-b["com"][..., wi, :])*self.body_mass[wi, None]).sum(-2)/self.body_mass.sum()
        delta = balanced_com[..., None, :2]-poly
        area = (poly[..., 0]*torch.roll(poly[..., 1], -1, -1)-poly[..., 1]*torch.roll(poly[..., 0], -1, -1)).sum(-1)
        orientation = torch.where(area >= 0, torch.ones_like(area), -torch.ones_like(area))
        signed = orientation[..., None]*(edge[..., 0]*delta[..., 1]-edge[..., 1]*delta[..., 0])/torch.linalg.vector_norm(edge, dim=-1).clamp_min(1e-5)
        terms["support"] = -c.support_weight*(torch.relu(c.support_margin-signed)/c.support_scale).square().mean(-1).clamp_max(10)
        gate = torch.sigmoid((c.contact_band-wg["gaps"])/c.contact_temperature)
        terms["contact"] = -c.contact_weight*(1-gate).mean(-1)
        terms["penetration"] = -c.penetration_weight*(torch.relu(-wg["gaps"])/c.penetration_scale).square().mean(-1)
        terms["lateral_slip"] = -c.lateral_slip_weight*(gate*(wg["lateral_slip"]/c.slip_speed_scale).square()).mean(-1)
        terms["rolling_slip"] = -c.rolling_slip_weight*(gate*(wg["rolling_slip"]/c.slip_speed_scale).square()).mean(-1)
        ji = self.nonwheel_indices
        width = (self.high[ji]-self.low[ji]).clamp_min(0.1)
        violation = torch.relu(self.low[ji]-joints[..., ji])+torch.relu(joints[..., ji]-self.high[ji])
        terms["joint_limit"] = -c.joint_limit_weight*(violation/width).square().sum(-1)
        if c.joint_soft_margin and c.joint_soft_limit_weight:
            margin=c.joint_soft_margin*width
            soft=torch.relu(self.low[ji]+margin-joints[...,ji])+torch.relu(joints[...,ji]-self.high[ji]+margin)
            terms['joint_soft_limit']=-c.joint_soft_limit_weight*(soft/margin).square().sum(-1)
        if c.wrist_limit_weight and c.wrist_safe_lower is not None and c.wrist_safe_upper is not None:
            violation=torch.relu(c.wrist_safe_lower-joints[...,20])+torch.relu(joints[...,20]-c.wrist_safe_upper)
            terms['wrist_operating_limit']=-c.wrist_limit_weight*(violation/.1).square()
        if c.self_collision_weight and self.self_collision_pairs.numel():
            pairs=self.self_collision_pairs
            distance=(b['com'][...,pairs[:,0],:]-b['com'][...,pairs[:,1],:]).square().sum(-1).add(1e-10).sqrt()
            overlap=torch.relu(2*c.arm_proxy_radius+.025-distance)
            terms['self_collision']=-c.self_collision_weight*(overlap/c.collision_scale).square().mean(-1)
        if c.settling_weight:
            angular2=task_angular.square().sum(-1)
            terms['settling']=c.settling_weight*torch.exp(-pe2/c.success_position**2-ae.square()/c.success_orientation**2
                -speed2/c.success_linear_speed**2-angular2/c.success_angular_speed**2)
        if self.surface_clearance is not None:
            distances=self.surface_clearance(b)
            for key,weight in [('robot',c.geometry_collision_weight),('ground',c.geometry_ground_weight)]:
                if weight:
                    violation=torch.relu(c.geometry_clearance_margin-distances[key])/c.geometry_clearance_scale
                    # The nearest part controls collision risk; averaging every
                    # surface point would hide a finger touching the chassis.
                    penalty=violation.square().topk(min(2,violation.shape[-1]),dim=-1).values.mean(-1)
                    terms['geometry_collision' if key=='robot' else 'geometry_ground']=-weight*penalty
        terms["action_rate"] = -c.action_rate_weight*(action-prev_action).square().mean(-1)
        # Post-state PD and smooth torque saturation are an analytic effort
        # proxy, not the simulator's measured substep torque or motor energy.
        target = torch.minimum(torch.maximum(self.q0+self.action_scale*action, self.low), self.high)
        tau = self.kp*(target-joints)-self.kd*qd
        wheel_tau = self.kd[self.wheel_indices]*(self.action_scale[self.wheel_indices]*action[..., self.wheel_indices]-qd[..., self.wheel_indices])
        wheel_mask = torch.zeros_like(self.kp).scatter(0, self.wheel_indices, 1.)
        expanded_wheel = torch.zeros_like(tau).scatter(-1, self.wheel_indices.expand_as(wheel_tau), wheel_tau)
        tau = tau*(1-wheel_mask)+expanded_wheel
        tau = self.effort*torch.tanh(tau/self.effort.clamp_min(1e-3))
        terms["torque"] = -c.torque_weight*(tau/self.effort.clamp_min(1e-3)).square().mean(-1)
        terms["power"] = -c.power_weight*(torch.sqrt((tau*qd).square()+1e-8)/(self.effort*self.velocity).clamp_min(1e-3)).mean(-1)
        if self.collision_arm.numel():
            ac = b["com"][..., self.collision_arm, :]
            oc = b["com"][..., self.collision_obstacles, :]
            distance = torch.sqrt((ac[..., :, None, :]-oc[..., None, :, :]).square().sum(-1)+1e-10)
            overlap = torch.relu(c.arm_proxy_radius+self.obstacle_radii-distance)
            terms["collision"] = -c.collision_weight*(overlap/c.collision_scale).square().mean((-1, -2))
            terms["ground"] = -c.ground_weight*(torch.relu(c.arm_proxy_radius-ac[..., 2])/c.collision_scale).square().mean(-1)
        else:
            terms["collision"] = torch.zeros_like(pe2)
            terms["ground"] = torch.zeros_like(pe2)
        if c.base_command_conditioning:
            goal=self.command_goal(goal);error=self.base_command_error(s_next,goal)
            linear=(sum(w*torch.exp(-error[...,:2].square().sum(-1)/sc**2)
                for sc,w in zip(c.base_linear_scales,c.base_linear_weights))/sum(c.base_linear_weights)
                if c.base_linear_scales else torch.exp(-error[...,:2].square().sum(-1)/c.base_linear_scale**2))
            terms['base_velocity']=c.base_command_weight*goal[...,10]*(
                linear+torch.exp(-error[...,2].square()/c.base_angular_scale**2))/2
        terms["total"] = sum(terms.values())
        terms["position_error"] = torch.sqrt(pe2+1e-12)
        terms["orientation_error"] = ae
        terms["tcp_speed"] = torch.sqrt(speed2+1e-12)
        terms["tcp_angular_speed"] = torch.sqrt(task_angular.square().sum(-1)+1e-12)
        terms["success"] = self._success_metrics(pe2, ae, speed2, task_angular.square().sum(-1))
        if c.base_command_conditioning:
            passed=(error[...,:2].norm(dim=-1)<=c.success_base_linear)&(error[...,2].abs()<=c.success_base_angular)
            terms['success']=terms['success']&((goal[...,10]<.5)|passed)
        return terms

    def _success_metrics(self, pe2, ae, speed2, angspeed2):
        c = self.cfg
        return ((pe2 <= c.success_position**2) & (ae <= c.success_orientation)
                & (speed2 <= c.success_linear_speed**2) & (angspeed2 <= c.success_angular_speed**2))

    def success(self, s, goal):
        b = self.fk(s)
        linear,angular=self.tracking_velocities(s,b,goal)
        passed=self._success_metrics((goal[..., :3]-b["tcp_pos"]).square().sum(-1),
                                     quat_angle(quat_error(b["tcp_quat"], goal[..., 3:7])),
                                     linear.square().sum(-1), angular.square().sum(-1))
        if self.cfg.base_command_conditioning:
            goal=self.command_goal(goal);error=self.base_command_error(s,goal)
            passed=passed&((goal[...,10]<.5)|((error[...,:2].norm(dim=-1)<=self.cfg.success_base_linear)&(error[...,2].abs()<=self.cfg.success_base_angular)))
        return passed

    def terminated(self, s):
        c = self.cfg
        upright = -projected_gravity(quat_normalize(s[..., 3:7]))[..., 2]
        return ((s[..., 2] < c.min_base_height) | (s[..., 2] > c.max_base_height)
                | (upright < c.min_upright_cosine) | ~torch.isfinite(s).all(-1))

    def termination_probability(self, s):
        c = self.cfg
        upright = -projected_gravity(quat_normalize(s[..., 3:7]))[..., 2]
        temp = c.termination_temperature
        safe = (torch.sigmoid((s[..., 2]-c.min_base_height)/temp)
                *torch.sigmoid((c.max_base_height-s[..., 2])/temp)
                *torch.sigmoid((upright-c.min_upright_cosine)/temp))
        return 1-safe

    failure_probability = termination_probability

    def sample_goals(self, s, level=0, generator=None):
        """Sample a local-to-wide pose curriculum without IK or expert trajectories.

        Samples above the ground only; global reachability and obstacle clearance
        are deliberately not asserted. Training must report failures on wide goals.
        """
        level = max(0, min(int(level), 3))
        pose = self.tcp_pose(s)
        noise = torch.randn(*s.shape[:-1], 3, device=s.device, dtype=s.dtype, generator=generator)
        direction = normalize(noise)
        radius = torch.rand(*s.shape[:-1], 1, device=s.device, dtype=s.dtype, generator=generator).pow(1/3)
        delta = direction*radius*self.cfg.goal_translation_radii[level]
        pos = pose[..., :3]+quat_apply(quat_normalize(s[..., 3:7]), delta)
        pos = torch.cat((pos[..., :2], pos[..., 2:3].clamp_min(self.cfg.minimum_goal_height)), -1)
        rotation = torch.randn(*s.shape[:-1], 3, device=s.device, dtype=s.dtype, generator=generator)
        angle = torch.rand(*s.shape[:-1], 1, device=s.device, dtype=s.dtype, generator=generator)*self.cfg.goal_rotation_radii[level]
        quat = quat_normalize(quat_mul(pose[..., 3:7], quat_from_rotvec(normalize(rotation)*angle)))
        goal=torch.cat((pos, quat), -1)
        if self.cfg.base_command_conditioning:
            shape=(*s.shape[:-1],1)
            mode=(torch.rand(shape,device=s.device,generator=generator)<self.cfg.base_command_fraction).to(s.dtype)
            command=(2*torch.rand((*s.shape[:-1],3),device=s.device,generator=generator)-1)*s.new_tensor(self.cfg.base_command_ranges)
            if self.cfg.base_endpoint_fraction:
                endpoint=torch.rand(shape,device=s.device,generator=generator)<self.cfg.base_endpoint_fraction
                command=torch.where(endpoint,torch.where(command>=0,s.new_tensor(1.),s.new_tensor(-1.))*s.new_tensor(self.cfg.base_command_ranges),command)
            if self.cfg.base_straight_fraction:
                straight=torch.rand(shape,device=s.device,generator=generator)<self.cfg.base_straight_fraction
                command=torch.cat((command[...,:1],torch.where(straight,torch.zeros_like(command[...,1:]),command[...,1:])),dim=-1)
            command=torch.where(torch.rand(shape,device=s.device,generator=generator)<.25,torch.zeros_like(command),command)
            goal=torch.cat((goal,command*mode,mode),-1)
        return self.initialize_goal(s,goal)
