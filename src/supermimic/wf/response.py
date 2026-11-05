"""Local real-simulator action-response checks for a learned dynamics model.

Low rollout MSE on smooth demonstrations does not identify action derivatives.
These finite perturbations are independent physical validation, not derivatives
through MuJoCo. A pass is local evidence and is supplemented by policy rollouts.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F

from .simulator import BatchedSimulator


class ActionResponseCheck:
    def __init__(self,model,spec,*,dt=.02,settle_steps=100,epsilon=.005,seed=62301,device='cpu',arm_initial_position=None):
        rng=np.random.default_rng(seed);sim=BatchedSimulator(model,spec,num_envs=1,seed=seed,dt=dt,friction=1.,
            arm_initial_position=arm_initial_position)
        sim.settle(steps=settle_steps);snapshot=sim.snapshot()
        baseline=sim.applied_action[0].copy()
        self.epsilon=epsilon;states=[];positive=[];negative=[];real_p=[];real_m=[];groups=[]
        for name,indices in (('legs',np.arange(12)),('wheels',np.arange(12,16)),('arm',np.arange(16,22)),('mixed',np.arange(22))):
            for _ in range(2):
                direction=np.zeros(22);direction[indices]=rng.normal(size=len(indices));direction/=np.linalg.norm(direction)
                sim.restore(snapshot);states.append(sim.read_state()[0].copy())
                real_p.append(sim.step((baseline+epsilon*direction)[None])[0].copy());positive.append(sim.applied_action[0].copy())
                sim.restore(snapshot);real_m.append(sim.step((baseline-epsilon*direction)[None])[0].copy());negative.append(sim.applied_action[0].copy())
                groups.append(name)
        sim.close();self.groups=groups
        self.arm_initial_position=arm_initial_position
        for name,value in (('states',states),('positive',positive),('negative',negative),('real_p',real_p),('real_m',real_m)):
            setattr(self,name,torch.tensor(np.asarray(value),dtype=torch.float32,device=device))
        self.xi=torch.ones(len(states),1,device=device)

    @torch.no_grad()
    def evaluate(self,wm,*,min_cosine=.9,max_relative_error=.5):
        plus=wm.step(self.states,self.positive,self.xi);minus=wm.step(self.states,self.negative,self.xi)
        predicted=wm.layout.error(plus,minus)/(2*self.epsilon)
        actual=wm.layout.error(self.real_p,self.real_m)/(2*self.epsilon)
        actual_norm=actual.norm(dim=-1).clamp_min(1e-8)
        cosine=F.cosine_similarity(predicted,actual,dim=-1)
        relative=(predicted-actual).norm(dim=-1)/actual_norm
        ratio=predicted.norm(dim=-1)/actual_norm
        passed=torch.isfinite(relative).all() and bool((cosine>=min_cosine).all() and (relative<=max_relative_error).all())
        return {'passed':bool(passed),'epsilon':self.epsilon,'min_cosine':float(cosine.min()),
                'max_relative_error':float(relative.max()),'min_magnitude_ratio':float(ratio.min()),
                'records':[{'group':g,'cosine':float(c),'relative_error':float(e),'magnitude_ratio':float(r)}
                           for g,c,e,r in zip(self.groups,cosine,relative,ratio)],
                'reset_arm_joint_target_rad':None if self.arm_initial_position is None else list(self.arm_initial_position),
                'scope':'local configured-reset finite response; not a global gradient guarantee'}
