"""Markov corrective whole-body teacher, used only for offline demonstration data.

Unlike a clock-indexed motion, every command depends only on current simulator
state, the previous executed command and the final TCP goal. This gives a goal
Actor corrective rather than phase-dependent labels. No teacher runs at inference.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
import torch

from .evaluate import config_from_dict,physical_failure
from .goal_reference import rotation_error
from .reference import screen_reference
from .simulator import BatchedSimulator
from .task import WholeBodyTask


def corrective_command(sim,goal,previous,action_bounds=None):
    s=sim.spec;d=sim.data[0];m=sim.model;state=sim.read_state()[0]
    quat=np.zeros(4);mujoco.mju_mat2Quat(quat,d.site_xmat[s.tcp_site])
    error=np.asarray(goal[:3])-d.site_xpos[s.tcp_site]
    angle=rotation_error(np.asarray(goal[3:]),quat)
    rotation=d.xmat[s.base_body].reshape(3,3);local=rotation.T@error;local_angle=rotation.T@angle
    jp=np.zeros((3,m.nv));jr=np.zeros_like(jp);mujoco.mj_jacSite(m,d,jp,jr,s.tcp_site)
    jac=np.vstack((jp[:,s.dof_indices[16:]],.3*jr[:,s.dof_indices[16:]]))
    low_all=s.low.copy();high_all=s.high.copy()
    if action_bounds is not None:
        low_all=np.maximum(low_all,s.q0+s.action_scale*action_bounds[0])
        high_all=np.minimum(high_all,s.q0+s.action_scale*action_bounds[1])
    desired=np.r_[error,.3*angle]
    dq=jac.T@np.linalg.solve(jac@jac.T+.025**2*np.eye(6),desired)
    if action_bounds is not None:
        # Reallocate the offline IK step when a joint is at an operating bound.
        # Clipping an unconstrained IK step alone keeps requesting unavailable DOFs.
        qprev=(s.q0+s.action_scale*previous)[16:]
        active=np.zeros(6,dtype=bool)
        for _ in range(3):
            blocked=((qprev<=low_all[16:]+.02)&(dq<0))|((qprev>=high_all[16:]-.02)&(dq>0))
            if not (blocked&~active).any():break
            active|=blocked;j=jac.copy();j[:,active]=0.
            dq=j.T@np.linalg.solve(j@j.T+.025**2*np.eye(6),desired)
    target=s.q0+s.action_scale*previous
    target[16:]+=np.clip(.04*dq-.04*d.qvel[s.dof_indices[16:]],-.006,.006)
    sides=np.array([1.,-1.,1.,-1.]);front=np.array([1.,1.,-1.,-1.])
    height=.015*local[2]-.005*state[31]
    shape=height-.003*local_angle[1]*front+.003*local_angle[0]*sides
    target[np.arange(4)*3+1]+=sides*shape
    target[np.arange(4)*3+2]-=2*sides*shape
    # Keep the wheel axes near the floor plane using the actual hip geometry.
    for leg in range(4):
        hip=m.joint(s.joint_names[3*leg]).id;wheel=m.joint(s.joint_names[12+leg]).id
        axle=d.xaxis[wheel];derivative=np.cross(d.xaxis[hip],axle)[2]
        target[3*leg]+=np.clip(-.04*axle[2]*derivative/(derivative**2+.01),-.002,.002)
    nonwheel=np.r_[0:12,16:22]
    # A preferred margin must not move a safely reached, stationary TCP merely
    # to improve posture. Hard source limits remain active at every goal error.
    margin=.04*np.tanh(np.linalg.norm(error)/.15+np.linalg.norm(angle)/.5)
    low=np.maximum(low_all[nonwheel]+margin*(high_all[nonwheel]-low_all[nonwheel]),s.q0[nonwheel]-s.action_scale[nonwheel])
    high=np.minimum(high_all[nonwheel]-margin*(high_all[nonwheel]-low_all[nonwheel]),s.q0[nonwheel]+s.action_scale[nonwheel])
    target[nonwheel]=np.clip(target[nonwheel],low,high)
    action=(target-s.q0)/s.action_scale
    velocity=np.clip(.4*local[0]+.5*max(0.,error[2]-.03)-.2*(rotation.T@state[29:32])[0],-.20,.20)
    angular=np.clip(.5*local[1]+.15*local_angle[2]-.2*state[34],-.20,.20)
    signs=np.array([-1.,1.,-1.,1.])
    wheel=signs*(velocity+signs*.20*angular)/s.wheel_radius/s.action_scale[12:16]
    action[12:16]=previous[12:16]+np.clip(wheel-previous[12:16],-.003,.003)
    result=np.clip(previous+np.clip(action-previous,-.008,.008),-1,1)
    return np.minimum(np.maximum(result,action_bounds[0]),action_bounds[1]) if action_bounds is not None else result


@torch.no_grad()
def execute(sim,task,goal,*,steps=600,settle_steps=100,joint_noise=0.,seed=0,
            initial_state=None,initial_action=None,action_bounds=None):
    sim.reset()
    if initial_state is not None:sim.write_state(np.asarray(initial_state)[None])
    if joint_noise:
        state=sim.read_state();ids=np.r_[0:12,16:22];state[:,7+ids]+=np.random.default_rng(seed).normal(0,joint_noise,(1,18));sim.write_state(state)
    sim.settle(steps=settle_steps);initial=sim.snapshot()
    previous=np.zeros(22) if initial_action is None else np.asarray(initial_action).copy()
    if previous.shape!=(22,) or not np.isfinite(previous).all() or (np.abs(previous)>1.).any():
        raise ValueError('invalid initial executed action')
    starting_action=previous.copy()
    states=[sim.read_state()[0].copy()];actions=[];safe=[];contacts=[];slip=[];torque=[];failed=[]
    for _ in range(steps):
        command=corrective_command(sim,goal,previous,action_bounds=action_bounds)
        sn=sim.step(command[None])[0].copy();previous=sim.applied_action[0].copy();p=sim.metrics()
        states.append(sn);actions.append(previous.copy());safe.append(bool(p['safe'][0]));contacts.append(p['wheel_contact'][0]);slip.append(p['slip'][0]);torque.append(p['torque_peak'][0])
        failed.append(bool(physical_failure(task,torch.tensor(sn[None],dtype=torch.float32),p)[0]))
        if failed[-1]:break
    return dict(states=np.asarray(states),actions=np.asarray(actions),safe=np.asarray(safe),contacts=np.asarray(contacts),
                slip=np.asarray(slip),torque_peak=np.asarray(torque),failed=np.asarray(failed),xi=sim.xi[0].copy(),goal=np.asarray(goal),
                initial_mj_state=initial['integration_state'][0],initial_state_flag=initial['state_flag'],initial_action=starting_action)


def generate(cfg,output,count=12,seed=90000,level=1):
    from .train import model_manifest
    torch.set_num_threads(2);output=Path(output)
    if (output/'manifest.json').exists():raise FileExistsError('select a new output directory')
    output.mkdir(parents=True,exist_ok=True);c=cfg.sim
    sim=BatchedSimulator(num_envs=1,seed=seed,dt=c.dt,timestep=c.timestep,asset_dir=cfg.asset_dir,friction=1.,
        leg_action_scale=c.leg_action_scale,arm_action_scale=c.arm_action_scale,wheel_action_scale=c.wheel_action_scale)
    task=WholeBodyTask(sim.model,sim.spec,reward_cfg=cfg.reward)
    manifest=dict(format='wf_physical_references_v1',complete=False,provenance=model_manifest(sim.spec),dt=c.dt,
        seed=seed,level=level,requested_clips=count,config=asdict(cfg),clips=[],rejected=[],
        generator_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        goal_origin='sampled requested goal before execution, without hindsight relabelling',
        reference_teacher='offline Markov whole-body corrective controller; not used by deployed Actor')
    bounds=(np.asarray(cfg.policy.action_lower),np.asarray(cfg.policy.action_upper)) if cfg.policy.action_lower else None
    for attempt in range(count*5):
        sim.friction_range=(1.,1.);sim.reset();sim.settle(steps=c.reset_settle_steps)
        goal=task.sample_goals(torch.tensor(sim.read_state(),dtype=torch.float32),level,torch.Generator().manual_seed(seed+attempt))[0].numpy()
        nominal=execute(sim,task,goal,settle_steps=c.reset_settle_steps,action_bounds=bounds)
        result=screen_reference(nominal,task,goal)
        if result['accepted']:
            sim.friction_range=(.85,1.15)
            clip=execute(sim,task,goal,settle_steps=c.reset_settle_steps,joint_noise=.005,seed=seed+attempt,action_bounds=bounds)
            result=screen_reference(clip,task,goal)
        if not result['accepted']:
            manifest['rejected'].append(dict(seed=seed+attempt,goal=goal.tolist(),screen=result));continue
        i=len(manifest['clips']);name=f'clip_{i:04d}'
        np.savez_compressed(output/(name+'.npz'),**clip)
        (output/(name+'_keyframes.json')).write_text(json.dumps({'world_goal':goal.tolist(),'type':'state-feedback corrective reference; no clock input'},indent=2)+'\n')
        manifest['clips'].append(dict(file=name+'.npz',keyframes=name+'_keyframes.json',family='corrective_goal',seed=seed+attempt,
            split='validation' if (seed+attempt)%5==4 else 'train',screen=result,sha256=hashlib.sha256((output/(name+'.npz')).read_bytes()).hexdigest()))
        print(json.dumps({'accepted_clips':i+1,'attempt':attempt+1,**result}),flush=True)
        if i+1>=count:break
    sim.close();manifest['complete']=len(manifest['clips'])==count
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    if not manifest['complete']:raise RuntimeError('insufficient physically validated corrective trajectories')
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default='configs/wf_goal_reference.json');p.add_argument('--output',required=True)
    p.add_argument('--count',type=int,default=12);p.add_argument('--seed',type=int,default=90000);p.add_argument('--level',type=int,default=1,choices=range(4));a=p.parse_args()
    if a.count<5:p.error('at least five clips needed')
    r=generate(config_from_dict(json.loads(Path(a.config).read_text())),a.output,a.count,a.seed,a.level)
    print(json.dumps({'clips':len(r['clips']),'rejected':len(r['rejected'])}))


if __name__=='__main__':main()
