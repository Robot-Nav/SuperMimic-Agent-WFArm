"""High-level reaching references and six-action SuperMimic tracking."""
import numpy as np
import torch

from .fast_evaluate import evaluate as native_evaluate
from .learning import Learner
from .planning import CollisionGoalPlanner
from .robot import Simulator
from .task import Task


class HierarchicalController:
    def __init__(self,learner,config):
        self.learner=learner;self.task=learner.task;self.config=config
        self.actor_updates=learner.actor_updates
        self.planner=CollisionGoalPlanner(config,learner.actor.goal_encoder,self.task.base_position)

    def start_episodes(self,episodes,sim):
        state=sim.state();self.initial=torch.as_tensor(state[:,:6].copy(),device=self.learner.device)
        targets=state[:,:6].copy();self.reaching=np.array([str(e["mode"])=="reach" for e in episodes])
        for i,e in enumerate(episodes):
            if self.reaching[i]:targets[i]=self.planner.target(state[i],e["goals"][0])
        self.difference=torch.as_tensor(targets,device=self.learner.device)-self.initial
        self.duration=(1.875*self.difference.abs().amax(-1)/.5).clamp_min(2.)
        self.mask=torch.as_tensor(self.reaching[:,None],device=self.learner.device,dtype=torch.float32)
        self.frame=0

    @torch.no_grad()
    def inference_action(self,state,previous,goal):
        u=(self.frame*self.task.dt/self.duration).clamp(0,1)
        blend=10*u**3-15*u**4+6*u**5
        derivative=30*u.square()*(1-u).square()/self.duration
        qref=self.initial+blend[:,None]*self.difference
        qrefdot=derivative[:,None]*self.difference
        body=self.task.fk(torch.cat((qref,qrefdot),-1))
        planned=torch.cat((body["tcp_pose"],body["tcp_v"],body["tcp_w"]),-1)
        target=torch.where(self.mask.bool(),planned,goal)
        augmented=torch.cat((target,qref,qrefdot,self.mask),-1)
        self.last_reference=target
        action=self.learner.inference_action(state,previous,augmented)
        self.frame+=1
        return action


@torch.no_grad()
def evaluate(learner,episodes,asset_dir,config,output=None):
    # A fresh CPU policy copy avoids CUDA batch=1 specialization and preserves
    # the live training weights, optimizer state and RNG during validation.
    cfg=dict(config,device="cpu",compile_imagined_step=False)
    with torch.random.fork_rng(devices=[]):
        sim=Simulator(asset_dir,1,dt=cfg["dt"],timestep=cfg["timestep"])
        task=Task(sim.model,sim.spec,cfg["dt"],cfg["joint_target_speed"],0)
        shadow=Learner(task,cfg)
        shadow.actor.load_state_dict(learner.actor.state_dict())
        shadow.actor_updates=learner.actor_updates
        controller=HierarchicalController(shadow,cfg)
        report=native_evaluate(controller,episodes,asset_dir,cfg,batch_size=cfg.get("evaluation_batch_size",16))
    report.update(controller="collision-checked goal planner + structured SuperMimic six-joint Actor",
        planning=controller.planner.stats,joint2_command_upper=cfg.get("joint2_command_upper",2.98),
        joint2_braking_lookahead=cfg.get("joint2_braking_lookahead",.25),
        criteria_compare_to="external final goal for reach; external continuous goal for tracking",
        planning_geometry="isolated model;1.5mm proximal and2mm other arm clearance; evaluated plant unchanged")
    if output:
        from .io import atomic_json
        atomic_json(output,report)
    return report
