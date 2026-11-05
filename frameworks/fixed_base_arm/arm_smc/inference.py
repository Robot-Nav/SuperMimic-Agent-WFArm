"""Six-command interface for the future base/vision controller integration."""
import numpy as np
import torch

from . import FORMAT
from .learning import Learner
from .robot import Simulator
from .task import Task


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
        config=ck["config"]
        sim=Simulator(config["asset_dir"],1,dt=config["dt"],timestep=config["timestep"])
        self.joint_names=sim.spec.names
        self.task=Task(sim.model,sim.spec,config["dt"],config["joint_target_speed"],0).to(device)
        self.learner=Learner(self.task,config);self.learner.actor.load_state_dict(ck["actor"])
        self.learner.actor.eval();self.device=device;self.previous=None

    def reset(self, measured_q):
        q=np.asarray(measured_q,dtype=np.float32)
        if q.shape!=(6,) or not np.isfinite(q).all():raise ValueError("six finite joint measurements required")
        self.previous=torch.as_tensor(q[None],device=self.device)

    @torch.no_grad()
    def command(self, measured_q, measured_qd, tcp_pose_in_base, tcp_twist_in_base=None):
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
        self.previous=self.task.command(self.learner.actor,state,self.previous,goal)
        return self.previous[0].cpu().numpy().copy()

    def observe_executed_target(self, target):
        """Use when a downstream limiter changes the commanded target."""
        self.reset(target)
