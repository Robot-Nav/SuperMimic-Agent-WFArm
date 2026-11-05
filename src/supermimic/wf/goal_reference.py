"""Offline generation of corrective Cartesian whole-body demonstrations.

Hand-designed base/leg keyframes and a smooth world-TCP reference are executed
using real actuators. Damped arm Jacobians are used ONLY by this offline data
generator. The trained/deployed Actor has no IK, reference sequence or base
controller. Requested goals are fixed before either physical rollout.
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

from .evaluate import config_from_dict, physical_failure
from .reference import minimum_jerk, screen_reference
from .simulator import BatchedSimulator
from .task import WholeBodyTask


def qmul(a,b):
    w,x,y,z=a;v,i,j,k=b
    return np.array([w*v-x*i-y*j-z*k,w*i+x*v+y*k-z*j,w*j-x*k+y*v+z*i,w*k+x*j-y*i+z*v])


def rotation_error(target,current):
    q=qmul(target,current*np.array([1.,-1.,-1.,-1.]))
    q=q/np.linalg.norm(q)
    if q[0]<0:q=-q
    n=np.linalg.norm(q[1:])
    return 2*q[1:] if n<1e-9 else 2*np.arctan2(n,q[0])*q[1:]/n


def slerp(start,goal,u):
    rv=u*rotation_error(goal,start);angle=np.linalg.norm(rv)
    dq=np.r_[np.cos(angle/2),rv*(.5 if angle<1e-9 else np.sin(angle/2)/angle)]
    out=qmul(dq,start)
    return out/np.linalg.norm(out)


def yaw(q):
    w,x,y,z=q
    return np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z))


@torch.no_grad()
def execute_goal_reference(sim,task,goal,*,seed=0,joint_noise=0.,settle_steps=100,move_seconds=4.,hold_seconds=4.,
                           correction_gain=.04,velocity_damping=.04):
    sim.reset()
    if joint_noise:
        state=sim.read_state();ids=np.r_[0:12,16:22]
        state[:,7+ids]+=np.random.default_rng(seed).normal(0,joint_noise,(1,18))
        sim.write_state(state)
    sim.settle(steps=settle_steps)
    initial=sim.snapshot();start=sim.read_state()[0].copy();spec=sim.spec;d=sim.data[0]
    start_tcp=task.tcp_pose(torch.tensor(start[None],dtype=torch.float32))[0].numpy().astype(float)
    goal=np.asarray(goal,float)
    delta=goal[:3]-start_tcp[:3]
    rot=rotation_error(goal[3:],start_tcp[3:])
    # High targets need the shoulder closer underneath the TCP, even when the
    # requested world x barely changes. Otherwise an arm-only extension can be
    # unreachable although whole-body reaching is feasible.
    high_assist=max(0.,delta[2]-.07)
    base_dx=.35*delta[0]+np.sign(delta[0])*.65*max(0.,abs(delta[0])-.22)+1.3*high_assist
    base_dyaw=np.clip(.6*delta[1],-.25,.25)
    final_legs=spec.q0[:12].copy()
    sides=np.array([1.,-1.,1.,-1.]);front=np.array([1.,1.,-1.,-1.])
    shape=np.clip(.4*delta[2]+.5*max(0.,delta[2]-.05),-.12,.12)+front*np.clip(.10*rot[1],-.045,.045)+sides*np.clip(.10*rot[0],-.045,.045)
    final_legs[np.arange(4)*3+1]+=sides*shape
    final_legs[np.arange(4)*3+2]-=2*sides*shape
    final_legs=np.clip(final_legs,np.maximum(spec.low[:12]+.015,spec.q0[:12]-spec.action_scale[:12]),
                       np.minimum(spec.high[:12]-.015,spec.q0[:12]+spec.action_scale[:12]))
    arm_command=spec.q0[16:].copy()
    jacp=np.zeros((3,sim.model.nv));jacr=np.zeros_like(jacp)
    states=[start];actions=[];safe=[];contacts=[];slip=[];torque=[];failed=[];planned=[]
    steps=round((move_seconds+hold_seconds)/sim.dt)
    for step in range(steps):
        t=(step+1)*sim.dt;v=np.clip(t/move_seconds,0,1);u=float(minimum_jerk(v))
        du=(30*v*v*(1-v)*(1-v)/move_seconds) if t<move_seconds else 0.
        reference=np.r_[start_tcp[:3]+u*delta,slerp(start_tcp[3:],goal[3:],u)]
        current_q=np.zeros(4);mujoco.mju_mat2Quat(current_q,d.site_xmat[spec.tcp_site])
        error=np.r_[reference[:3]-d.site_xpos[spec.tcp_site],.3*rotation_error(reference[3:],current_q)]
        mujoco.mj_jacSite(sim.model,d,jacp,jacr,spec.tcp_site)
        jac=np.vstack((jacp[:,spec.dof_indices[16:]],.3*jacr[:,spec.dof_indices[16:]]))
        dq=jac.T@np.linalg.solve(jac@jac.T+.025**2*np.eye(6),error)
        arm_command+=np.clip(correction_gain*dq-velocity_damping*d.qvel[spec.dof_indices[16:]],-.006,.006)
        arm_command=np.clip(arm_command,spec.low[16:]+.01,spec.high[16:]-.01)
        state=sim.read_state()[0];heading=yaw(state[3:7])
        desired_x=start[0]+u*base_dx;desired_y=start[1]
        dx=np.cos(heading)*(desired_x-state[0])+np.sin(heading)*(desired_y-state[1])
        dy=-np.sin(heading)*(desired_x-state[0])+np.cos(heading)*(desired_y-state[1])
        target_yaw=yaw(start[3:7])+u*base_dyaw
        angle=np.arctan2(np.sin(target_yaw-heading),np.cos(target_yaw-heading))
        velocity=np.clip(du*base_dx+.8*dx,-.25,.25)
        angular=np.clip(du*base_dyaw+.8*angle+.5*dy*np.sign(velocity),-.25,.25)
        signs=np.array([-1.,1.,-1.,1.])
        command=np.zeros(22)
        command[:12]=(spec.q0[:12]+u*(final_legs-spec.q0[:12])-spec.q0[:12])/spec.action_scale[:12]
        command[12:16]=signs*(velocity+signs*.20*angular)/spec.wheel_radius/spec.action_scale[12:16]
        command[16:]=(arm_command-spec.q0[16:])/spec.action_scale[16:]
        next_state=sim.step(command[None])[0].copy();m=sim.metrics()
        states.append(next_state);actions.append(sim.applied_action[0].copy());planned.append(reference)
        safe.append(bool(m['safe'][0]));contacts.append(m['wheel_contact'][0]);slip.append(m['slip'][0]);torque.append(m['torque_peak'][0])
        failed.append(bool(physical_failure(task,torch.tensor(next_state[None],dtype=torch.float32),m)[0]))
        if failed[-1]:break
    keyframes={"move_seconds":move_seconds,"hold_seconds":hold_seconds,"world_tcp_start":start_tcp.tolist(),
               "world_tcp_goal":goal.tolist(),"base_dx_m":float(base_dx),"base_dyaw_rad":float(base_dyaw),
               "leg_start_rad":spec.q0[:12].tolist(),"leg_goal_rad":final_legs.tolist(),
               "arm_command_generation":"offline damped Jacobian, physical PD execution; absent from policy"}
    return {"states":np.asarray(states),"actions":np.asarray(actions),"safe":np.asarray(safe),
            "contacts":np.asarray(contacts),"slip":np.asarray(slip),"torque_peak":np.asarray(torque),
            "failed":np.asarray(failed),"xi":sim.xi[0].copy(),"goal":goal,
            "initial_mj_state":initial['integration_state'][0],"initial_state_flag":initial['state_flag'],
            "planned_tcp":np.asarray(planned)},keyframes


def generate(config,output,count=24,seed=51000,level=1):
    from .train import model_manifest
    torch.set_num_threads(2);output=Path(output)
    if (output/'manifest.json').exists():raise FileExistsError("select a new reference output directory")
    output.mkdir(parents=True,exist_ok=True);c=config.sim
    sim=BatchedSimulator(num_envs=1,seed=seed,dt=c.dt,timestep=c.timestep,asset_dir=config.asset_dir,friction=1.,
        leg_action_scale=c.leg_action_scale,arm_action_scale=c.arm_action_scale,wheel_action_scale=c.wheel_action_scale)
    task=WholeBodyTask(sim.model,sim.spec,reward_cfg=config.reward)
    manifest={"format":"wf_physical_references_v1","complete":False,"provenance":model_manifest(sim.spec),
        "dt":c.dt,"seed":seed,"level":level,"requested_clips":count,"config":asdict(config),
        "goal_origin":"sampled Cartesian world pose BEFORE execution; never relabelled after rollout",
        "reference_teacher":"offline synchronized Cartesian/base/leg keyframes with damped arm-Jacobian correction",
        "deployment":"single learned goal Actor only; no offline teacher, IK or reference sequence",
        "clips":[],"rejected":[]}
    for attempt in range(count*4):
        sim.friction_range=(1.,1.);sim.reset();sim.settle(steps=c.reset_settle_steps)
        initial=torch.tensor(sim.read_state(),dtype=torch.float32)
        goal=task.sample_goals(initial,level,torch.Generator().manual_seed(seed+attempt))[0].numpy()
        nominal,_=execute_goal_reference(sim,task,goal,settle_steps=c.reset_settle_steps)
        check_nominal=screen_reference(nominal,task,goal)
        if not check_nominal['accepted']:
            manifest['rejected'].append({"seed":seed+attempt,"goal":goal.tolist(),"reason":"nominal requested goal not held","screen":check_nominal})
            continue
        sim.friction_range=(.85,1.15)
        clip,key=execute_goal_reference(sim,task,goal,settle_steps=c.reset_settle_steps,joint_noise=.005,seed=seed+attempt)
        check=screen_reference(clip,task,goal)
        if not check['accepted']:
            manifest['rejected'].append({"seed":seed+attempt,"goal":goal.tolist(),"reason":"perturbed replay not held","screen":check})
            continue
        i=len(manifest['clips']);name=f'clip_{i:04d}'
        np.savez_compressed(output/(name+'.npz'),**clip)
        (output/(name+'_keyframes.json')).write_text(json.dumps(key,indent=2)+'\n')
        manifest['clips'].append({"file":name+'.npz',"keyframes":name+'_keyframes.json',"family":"goal_reaching",
            "seed":seed+attempt,"split":"validation" if (seed+attempt)%5==4 else "train","screen":check,
            "sha256":hashlib.sha256((output/(name+'.npz')).read_bytes()).hexdigest()})
        print(json.dumps({"accepted_clips":i+1,"attempt":attempt+1,**check}),flush=True)
        if i+1>=count:break
    sim.close();manifest['complete']=len(manifest['clips'])==count
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
    if not manifest['complete']:raise RuntimeError("not enough requested goals passed physical screening")
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default='configs/wf_pose_stage1_repaired.json')
    p.add_argument('--output',required=True);p.add_argument('--count',type=int,default=24)
    p.add_argument('--seed',type=int,default=51000);p.add_argument('--level',type=int,default=1,choices=range(4))
    a=p.parse_args()
    if a.count<5:p.error('at least five complete clips are required')
    r=generate(config_from_dict(json.loads(Path(a.config).read_text())),a.output,a.count,a.seed,a.level)
    print(json.dumps({"clips":len(r['clips']),"rejected":len(r['rejected']),"output":a.output}))


if __name__=='__main__':main()
