"""MimicAgent-inspired skill programs, diagnosis/revision, then real execution.

This is a local adaptation, not the unreleased MimicAgent implementation. The
generated joint-space programs are kinematic proposals; ONLY independently
executed, physically screened state/action transitions enter demonstration RL.
No generated base pose is teleported into a training trajectory.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
import torch

from .evaluate import config_from_dict, physical_failure
from .feedback_reference import corrective_command
from .reference import screen_reference
from .simulator import BatchedSimulator
from .task import WholeBodyTask
from .workspace_audit import sector_masks


class ArmPathPlanner:
    def __init__(self, sim, initial, lower, upper, clearance=0.):
        self.sim, self.initial = sim, initial.copy()
        self.lower, self.upper = lower, upper
        self.wheels = set(sim.spec.wheel_geom_ids.tolist())
        self.clearance=float(clearance)
        if not 0<=self.clearance<=.05:raise ValueError('planning clearance must be in [0, .05] m')
        # Extra geometric margin belongs ONLY to a planning copy. It must not
        # change the contact forces or provenance of the executed simulator.
        self.model=copy.copy(sim.model);self.data=mujoco.MjData(self.model)
        self.model.geom_margin[:]=np.maximum(self.model.geom_margin,self.clearance)
        spec=sim.spec
        self.data.qpos[spec.base_qpos_index:spec.base_qpos_index+7]=initial[:7]
        self.data.qpos[spec.qpos_indices]=initial[7:29]
        self.data.qvel[spec.base_dof_index:spec.base_dof_index+6]=initial[29:35]
        self.data.qvel[spec.dof_indices]=initial[35:57]
        mujoco.mj_forward(self.model,self.data)
        self.pair_clearance={}
        # Existing source fixtures have several sub-centimetre clearances.
        # Preserve at least half of their initial positive separation instead
        # of making the starting robot invalid with a global 15 mm threshold.
        for contact in self.data.contact[:self.data.ncon]:
            if 0<contact.dist<self.clearance:
                pair=tuple(sorted((int(contact.geom1),int(contact.geom2))))
                self.pair_clearance[pair]=min(self.pair_clearance.get(pair,self.clearance),contact.dist*.5)

    def valid(self, q):
        if np.any(q < self.lower) or np.any(q > self.upper):
            return False
        d, spec = self.data, self.sim.spec
        d.qpos[spec.qpos_indices[16:]] = q
        mujoco.mj_forward(self.model, d)
        for c in d.contact[:d.ncon]:
            key=tuple(sorted((int(c.geom1),int(c.geom2))))
            if c.dist > self.pair_clearance.get(key,self.clearance):
                continue
            pair = {int(c.geom1), int(c.geom2)}
            if spec.ground_geom in pair and (pair-{spec.ground_geom}).issubset(self.wheels):
                continue
            return False
        return True

    def edge(self, a, b):
        count = max(1, int(np.ceil(np.linalg.norm(a-b)/.08)))
        return all(self.valid(a+(b-a)*t) for t in np.linspace(0, 1, count+1))

    def plan(self, goal, rng, budget=700):
        # Native soft limits can leave a held joint a fraction of a degree
        # outside its range. Project only the PLANNING copy, never the actual
        # initial state or any world-model transition label.
        start = np.clip(self.initial[23:29],self.lower,self.upper)
        if not self.valid(goal):
            return None
        if self.edge(start, goal):
            return [start.copy(), goal.copy()]
        nodes, parents = [start.copy()], [-1]
        for iteration in range(budget):
            sample = goal if iteration % 5 == 0 else rng.uniform(self.lower, self.upper)
            nearest = int(np.argmin([np.linalg.norm(n-sample) for n in nodes]))
            delta = sample-nodes[nearest]
            new = nodes[nearest]+delta*min(1., .4/max(1e-9, np.linalg.norm(delta)))
            if not self.edge(nodes[nearest], new):
                continue
            nodes.append(new); parents.append(nearest)
            if not self.edge(new, goal):
                continue
            path = [goal.copy(), new]
            parent = nearest
            while parent >= 0:
                path.append(nodes[parent]); parent = parents[parent]
            path.reverse()
            # Shortcuts preserve collision checks, rather than just smoothing
            # through an obstacle between valid keyframes.
            for _ in range(40):
                if len(path) <= 2:
                    break
                i, j = sorted(rng.choice(len(path), 2, replace=False))
                if j > i+1 and self.edge(path[i], path[j]):
                    path = path[:i+1]+path[j:]
            return path
        return None


def execute_program(sim, task, goal, path, *, movement_steps=330, steps=600, bounds=None):
    sim.reset();sim.settle(steps=100)
    initial=sim.snapshot();initial_action=sim.applied_action[0].copy();previous=initial_action.copy()
    states=[sim.read_state()[0].copy()]
    data={k:[] for k in ('actions','safe','contacts','slip','torque_peak','failed')}
    path=np.asarray(path)
    lengths=np.linalg.norm(np.diff(path,axis=0),axis=-1)
    knots=np.r_[0,np.cumsum(lengths)];knots/=max(knots[-1],1e-8)
    for step in range(steps):
        if step < movement_steps:
            phase=(step+1)/movement_steps
            phase=phase**3*(10-15*phase+6*phase**2)
            segment=min(len(path)-2,max(0,int(np.searchsorted(knots,phase)-1)))
            local=(phase-knots[segment])/max(1e-8,knots[segment+1]-knots[segment])
            target=path[segment]+np.clip(local,0,1)*(path[segment+1]-path[segment])
            command=previous.copy()
            desired=(target-sim.spec.q0[16:])/sim.spec.action_scale[16:]
            command[16:]+=np.clip(desired-previous[16:],-.008,.008)
            # Native position servos hold the original standing leg targets.
            command[:16]+=np.clip(-previous[:16],-.003,.003)
        else:
            command=corrective_command(sim,goal,previous,action_bounds=bounds)
        if bounds is not None:command=np.clip(command,bounds[0],bounds[1])
        sn=sim.step(command[None])[0].copy();previous=sim.applied_action[0].copy();metrics=sim.metrics()
        states.append(sn);data['actions'].append(previous.copy());data['safe'].append(bool(metrics['safe'][0]))
        data['contacts'].append(metrics['wheel_contact'][0]);data['slip'].append(metrics['slip'][0]);data['torque_peak'].append(metrics['torque_peak'][0])
        failed=bool(physical_failure(task,torch.tensor(sn[None],dtype=torch.float32),metrics)[0]);data['failed'].append(failed)
        if failed:break
    return dict(states=np.asarray(states),**{k:np.asarray(v) for k,v in data.items()},
        goal=goal,xi=sim.xi[0].copy(),initial_action=initial_action,
        initial_mj_state=initial['integration_state'][0],initial_state_flag=initial['state_flag'])


def generate(cfg, output, count=24, seed=710000, sectors=None, max_attempts=None,clearance=0.,balanced=False):
    from .train import model_manifest
    torch.set_num_threads(1)
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    sim=BatchedSimulator(num_envs=1,seed=seed,friction=1.,dt=cfg.sim.dt,timestep=cfg.sim.timestep,
        leg_action_scale=cfg.sim.leg_action_scale,arm_action_scale=cfg.sim.arm_action_scale,wheel_action_scale=cfg.sim.wheel_action_scale,
        arm_initial_position=cfg.sim.arm_initial_position)
    task=WholeBodyTask(sim.model,sim.spec,reward_cfg=cfg.reward)
    bounds=(np.asarray(cfg.policy.action_lower),np.asarray(cfg.policy.action_upper))
    lower=np.maximum(sim.spec.low,sim.spec.q0+sim.spec.action_scale*bounds[0])[16:]
    upper=np.minimum(sim.spec.high,sim.spec.q0+sim.spec.action_scale*bounds[1])[16:]
    rng=np.random.default_rng(seed);sectors=sectors or ['top','left','right','low_side','underneath','rear']
    quotas={name:count//len(sectors)+(i<count%len(sectors)) for i,name in enumerate(sectors)}
    counts={name:0 for name in sectors}
    if balanced and min(quotas.values())<2:raise ValueError('balanced train/validation needs at least two clips per sector')
    manifest=dict(format='wf_physical_references_v1',complete=False,provenance=model_manifest(sim.spec),dt=cfg.sim.dt,
        seed=seed,level=3,requested_clips=count,config=asdict(cfg),clips=[],rejected=[],
        method='MimicAgent-inspired authored skill programs + constraint diagnosis + actual dynamics filtering; not official code',
        proposal_clearance_m=clearance,
        proposal_clearance_rule='requested margin; pre-existing smaller positive gaps retain at least half their initial separation',
        sector_quotas=quotas if balanced else None,
        split_rule='within each sector, every fifth; for fewer than five, last is validation' if balanced else 'global every fifth clip',
        source_paper='https://arxiv.org/abs/2609.24145',
        goal_origin='requested geometric 6D pose fixed before physical execution; no hindsight target relabeling',
        generator_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    try:
        for attempt in range(max_attempts or count*15):
            eligible=[name for name in sectors if counts[name]<quotas[name]] if balanced else sectors
            sector=eligible[attempt%len(eligible)]
            sim.friction_range=(1.,1.);sim.reset();sim.settle(steps=cfg.sim.reset_settle_steps)
            initial=sim.read_state()[0].copy();planner=ArmPathPlanner(sim,initial,lower,upper,clearance=clearance)
            q=None;goal=None
            for _ in range(1500):
                candidate=rng.uniform(lower,upper)
                if not planner.valid(candidate):continue
                d=planner.data;point=(d.site_xpos[sim.spec.tcp_site]-initial[:3])@d.xmat[sim.spec.base_body].reshape(3,3)
                if d.site_xpos[sim.spec.tcp_site,2]<.1 or not sector_masks(point[None])[sector][0]:continue
                quaternion=np.zeros(4);mujoco.mju_mat2Quat(quaternion,d.site_xmat[sim.spec.tcp_site])
                goal=np.r_[d.site_xpos[sim.spec.tcp_site].copy(),quaternion];q=candidate;break
            if q is None:
                manifest['rejected'].append(dict(attempt=attempt,sector=sector,reason='no sampled collision-free geometric target'))
                continue
            diagnostics=[];accepted=False
            for revision in range(3):
                path=planner.plan(q,rng)
                if path is None:
                    diagnostics.append(dict(revision=revision,reason='no collision-free joint corridor'));continue
                movement_steps=[330,400,260][revision]
                plan=dict(skill=f'reach_{sector}',description=f'Reach a world 6D pose in {sector}; keep a supported wheeled stance',
                    phases=['stable stance','collision-checked joint corridor','Cartesian feedback correction','safe final hold'],
                    joint_waypoints=[x.tolist() for x in path],movement_steps=movement_steps,world_goal=goal.tolist(),
                    proposal_scope='kinematic only; not a simulator transition until execution')
                sim.friction_range=(1.,1.)
                nominal=execute_program(sim,task,goal,path,movement_steps=movement_steps,bounds=bounds,steps=cfg.sim.episode_steps)
                result=screen_reference(nominal,task,goal)
                if result['accepted']:
                    sim.friction_range=(.85,1.15)
                    clip=execute_program(sim,task,goal,path,movement_steps=movement_steps,bounds=bounds,steps=cfg.sim.episode_steps)
                    result=screen_reference(clip,task,goal)
                diagnostics.append(dict(revision=revision,screen=result))
                if result['accepted']:
                    accepted=True;break
            if not accepted:
                manifest['rejected'].append(dict(attempt=attempt,sector=sector,goal=goal.tolist(),diagnostics=diagnostics))
                print(json.dumps(dict(attempt=attempt,sector=sector,accepted=False)),flush=True);continue
            i=len(manifest['clips']);name=f'clip_{i:04d}'
            np.savez_compressed(output/(name+'.npz'),**clip)
            (output/(name+'_keyframes.json')).write_text(json.dumps(plan,indent=2)+'\n')
            sector_index=counts[sector]
            is_validation=((sector_index%5==4 or (quotas[sector]<5 and sector_index==quotas[sector]-1)) if balanced else i%5==4)
            counts[sector]+=1
            manifest['clips'].append(dict(file=name+'.npz',keyframes=name+'_keyframes.json',family='agent_reach_'+sector,
                seed=seed+attempt,level=3,split='validation' if is_validation else 'train',screen=result,
                initial_pose_family='folded_zero' if cfg.sim.arm_initial_position is not None and np.allclose(cfg.sim.arm_initial_position,0.) else 'raised_work_pose',
                diagnostics=diagnostics,sha256=hashlib.sha256((output/(name+'.npz')).read_bytes()).hexdigest()))
            (output/'manifest.partial.json').write_text(json.dumps(manifest,indent=2)+'\n')
            print(json.dumps(dict(accepted_clips=i+1,attempt=attempt,sector=sector,revision=revision)),flush=True)
            if i+1==count:break
    finally:sim.close()
    manifest['complete']=len(manifest['clips'])==count
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    if not manifest['complete']:raise RuntimeError('reference pilot incomplete; rejected programs are not training transitions')
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default='configs/wf_wbc_safe_fresh.json')
    p.add_argument('--output',required=True);p.add_argument('--count',type=int,default=24);p.add_argument('--seed',type=int,default=710000)
    p.add_argument('--sectors',nargs='+');p.add_argument('--max-attempts',type=int);p.add_argument('--clearance',type=float,default=0.);p.add_argument('--balanced',action='store_true')
    a=p.parse_args();generate(config_from_dict(json.loads(Path(a.config).read_text())),a.output,a.count,a.seed,a.sectors,a.max_attempts,a.clearance,a.balanced)


if __name__=='__main__':main()
