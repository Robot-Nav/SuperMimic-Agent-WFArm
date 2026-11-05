"""Collision-checked joint reference generation for large TCP reaches.

The planner supplies references; the SuperMimic Actor supplies motor commands.
Native collision margins below are used on an isolated planning model only.
"""
import copy
import numpy as np
import torch
from .robot import Simulator
from .rotations import rot6d
from .rotations import world_rotation_error
import mujoco

class CollisionGoalPlanner:
    def __init__(self,config,encoder,base_position):
        self.planner=Simulator(config["asset_dir"],1,dt=config["dt"],timestep=config["timestep"])
        self.planner.model.geom_margin[:]=np.maximum(self.planner.model.geom_margin,.006)
        self.encoder=copy.deepcopy(encoder).cpu().eval().requires_grad_(False)
        self.base_position=base_position.detach().cpu()
        self.joint2_upper=config.get("joint2_command_upper",2.98)
        self.stats=[]

    @torch.no_grad()
    def target(self,state,goal):
        value=torch.as_tensor(goal,dtype=torch.float32)
        features=torch.cat(((value[:3]-self.base_position)/.8,rot6d(value[3:7])),0)
        normalized=self.encoder(features[None]).reshape(4,6)
        low=torch.from_numpy(self.planner.spec.low);high=torch.from_numpy(self.planner.spec.high)
        seeds=(normalized*(high-low)/2+(low+high)/2).numpy()
        return self.plan(np.asarray(state),np.asarray(goal),seeds)

    def pose_jacobian(self,q):
        sim=self.planner;d=sim.data[0]
        d.qpos[sim.spec.qadr]=q;d.qvel[:]=0;mujoco.mj_forward(sim.model,d)
        quat=np.empty(4);mujoco.mju_mat2Quat(quat,d.site_xmat[sim.spec.tcp])
        jp=np.empty((3,6));jr=np.empty_like(jp)
        mujoco.mj_jacSite(sim.model,d,jp,jr,sim.spec.tcp)
        return np.r_[d.site_xpos[sim.spec.tcp],quat],np.r_[jp,jr]

    def path_safe(self,start,end):
        sim=self.planner
        for f in np.linspace(0,1,33):
            sim.data[0].qpos[sim.spec.qadr]=start+f*(end-start)
            mujoco.mj_forward(sim.model,sim.data[0])
            for c in sim.data[0].contact:
                bodies=[int(sim.model.geom_bodyid[g]) for g in (c.geom1,c.geom2)]
                if not any(b in sim.dynamic_bodies for b in bodies):continue
                names={sim.model.body(b).name for b in bodies}
                margin=.0015 if names=={"piper_base_link","piper_link2"} else .002
                if c.dist<margin:return False
        return True

    def plan(self,state,goal,seeds):
        sim=self.planner;weights=np.array([5.,5.,5.,2.,2.,2.])
        low=sim.spec.low+.001;high=sim.spec.high-.001
        high=high.copy();high[1]=min(high[1],self.joint2_upper)
        options=[]
        for seed in [*seeds,state[:6],np.array([0.,1.35,-1.75,0.,.4,0.])]:
            q=np.clip(seed,low,high).copy()
            for _ in range(50):
                pose,jac=self.pose_jacobian(q)
                error=np.r_[goal[:3]-pose[:3],world_rotation_error(goal[3:7],pose[3:7])]
                if np.linalg.norm(error[:3])<.001 and np.linalg.norm(error[3:])<.005:break
                a=weights[:,None]*jac
                delta=np.linalg.solve(a.T@a+.02**2*np.eye(6),a.T@(weights*error))
                delta*=min(1.,.15/max(np.max(np.abs(delta)),1e-12))
                q=np.clip(q+delta,low,high)
            pose,_=self.pose_jacobian(q)
            pe=np.linalg.norm(goal[:3]-pose[:3]);re=np.linalg.norm(world_rotation_error(goal[3:7],pose[3:7]))
            safe=self.path_safe(state[:6],q)
            score=(pe/.03)**2+(re/.15)**2+.01*np.sum((q-state[:6])**2)
            options.append((not safe,score,q.copy(),pe,re))
        options.sort(key=lambda x:(x[0],x[1]));bad,score,q,pe,re=options[0]
        self.stats.append(dict(route_safe=not bad,goal_position_residual_m=float(pe),goal_orientation_residual_rad=float(re)))
        return state[:6].copy() if bad else q
