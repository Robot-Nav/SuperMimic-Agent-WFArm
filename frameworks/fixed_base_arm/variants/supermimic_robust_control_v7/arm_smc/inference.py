"""Six-command interface for the future base/vision controller integration."""
import numpy as np
import torch

from . import FORMAT
from .learning import Learner
from .robot import Simulator
from .task import Task
from .planning import CollisionGoalPlanner


class ArmController:
    """Input: measured arm q/qd and a TCP pose/twist in the chassis BASE frame.

    Output: six absolute joint position targets in radians, named piper_joint1
    through piper_joint6. Camera/gripper commands are outside this policy.
    Call only after the chassis is stationary; keep the base controller active
    with zero velocity. The caller performs world/camera -> base transforms.
    """
    def __init__(self, checkpoint, device="cpu"):
        ck=torch.load(checkpoint,map_location=device,weights_only=False)
        if ck.get("format")!=FORMAT:raise ValueError("not an independent fixed-arm checkpoint")
        config=dict(ck["config"],compile_imagined_step=False)
        sim=Simulator(config["asset_dir"],1,dt=config["dt"],timestep=config["timestep"])
        self.joint_names=sim.spec.names
        self.task=Task(sim.model,sim.spec,config["dt"],config["joint_target_speed"],0).to(device)
        self.learner=Learner(self.task,config);self.learner.actor.load_state_dict(ck["actor"])
        self.learner.actor.eval();self.device=device;self.previous=None
        self.planner=CollisionGoalPlanner(config,self.learner.actor.goal_encoder,self.task.base_position)
        self.reach_plan=None;self.last_mode=None

    def reset(self, measured_q):
        q=np.asarray(measured_q,dtype=np.float32)
        if q.shape!=(6,) or not np.isfinite(q).all():raise ValueError("six finite joint measurements required")
        self.previous=torch.as_tensor(q[None].copy(),device=self.device)
        self.reach_plan=None;self.last_mode=None

    @torch.no_grad()
    def command(self, measured_q, measured_qd, tcp_pose_in_base, tcp_twist_in_base=None,mode=None):
        q=np.asarray(measured_q,dtype=np.float32);qd=np.asarray(measured_qd,dtype=np.float32)
        pose=np.asarray(tcp_pose_in_base,dtype=np.float32)
        twist=np.zeros(6,dtype=np.float32) if tcp_twist_in_base is None else np.asarray(tcp_twist_in_base,dtype=np.float32)
        if q.shape!=(6,) or qd.shape!=(6,) or pose.shape!=(7,) or twist.shape!=(6,):
            raise ValueError("expected q6, qd6, TCP xyz3+quaternion_wxyz4 and linear3+angular3 twist")
        if not all(np.isfinite(a).all() for a in (q,qd,pose,twist)) or np.linalg.norm(pose[3:])<1e-8:
            raise ValueError("nonfinite measurement or invalid TCP quaternion")
        if self.previous is None:self.reset(q)
        goal=torch.as_tensor(np.r_[pose,twist][None],device=self.device)
        goal[:,:3]+=self.task.base_position
        goal[:,3:7]/=torch.linalg.vector_norm(goal[:,3:7],dim=-1,keepdim=True)
        state=torch.as_tensor(np.r_[q,qd][None],device=self.device)
        mode=("reach" if tcp_twist_in_base is None else "tracking") if mode is None else mode
        if mode not in ("reach","tracking"):raise ValueError("mode must be reach or tracking")
        if mode=="reach":
            changed=self.reach_plan is None or self.last_mode!="reach"
            if not changed:
                remembered=self.reach_plan["goal"]
                shift=float((remembered[:,:3]-goal[:,:3]).norm())
                dot=(remembered[:,3:7]*goal[:,3:7]).sum(-1).abs().clamp(0,1)
                angle=float(2*torch.acos(dot))
                changed=shift>.005 or angle>.02
            if changed:
                target=self.planner.target(state[0].cpu().numpy(),goal[0].cpu().numpy())
                initial=state[:,:6].clone();difference=torch.as_tensor(target[None],device=self.device,dtype=torch.float32)-initial
                self.reach_plan=dict(goal=goal.clone(),initial=initial,difference=difference,
                    duration=max(2.,1.875*float(difference.abs().max())/.5),frame=0)
            plan=self.reach_plan;u=min(1.,plan["frame"]*self.task.dt/plan["duration"])
            blend=10*u**3-15*u**4+6*u**5;derivative=30*u*u*(1-u)*(1-u)/plan["duration"]
            reference=plan["initial"]+blend*plan["difference"]
            velocity=derivative*plan["difference"]
            bodies=self.task.fk(torch.cat((reference,velocity),-1))
            requested=(torch.cat((bodies["tcp_pose"],bodies["tcp_v"],bodies["tcp_w"]),-1) if u<1 else goal)
            goal=torch.cat((requested,reference,velocity,torch.ones_like(goal[:,:1])),-1)
            plan["frame"]+=1
        else:self.reach_plan=None
        self.last_mode=mode
        self.previous=self.task.command(self.learner.actor,state,self.previous,goal)
        return self.previous[0].cpu().numpy().copy()

    def observe_executed_target(self, target):
        """Use when a downstream limiter changes the commanded target."""
        q=np.asarray(target,dtype=np.float32)
        if q.shape!=(6,) or not np.isfinite(q).all():raise ValueError("six finite executed targets required")
        self.previous=torch.as_tensor(q[None].copy(),device=self.device)
