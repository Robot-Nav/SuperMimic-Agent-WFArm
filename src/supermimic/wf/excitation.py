"""Real action-perturbation windows around executed whole-body reference states."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .demonstrations import Demonstrations
from .evaluate import config_from_dict, physical_failure
from .replay import ModelBatch
from .simulator import BatchedSimulator
from .task import WholeBodyTask


class DynamicsExcitation:
    def __init__(self, file, device, physics_hash, dt):
        file=Path(file);self.fingerprint=hashlib.sha256(file.read_bytes()).hexdigest()
        with np.load(file,allow_pickle=False) as data:
            meta=json.loads(str(data['metadata'].item()))
            if meta['format']!='wf_real_excitation_v1' or meta['physics_hash']!=physics_hash or meta['dt']!=dt:
                raise ValueError('excitation dynamics provenance mismatch')
            for key in ('states','actions','xi','failed'):
                setattr(self,key,torch.tensor(data[key],dtype=torch.float32,device=device))
            mask=torch.tensor(data['validation'],dtype=torch.bool,device=device)
        self.training=(~mask).nonzero().flatten();self.validation=mask.nonzero().flatten()
        self.device=device
        if not len(self.training) or not len(self.validation):raise ValueError('excitation needs disjoint clips')

    def sample(self,n,horizon,validation=False):
        if horizon>self.actions.shape[1]:raise ValueError('excitation window too short')
        pool=self.validation if validation else self.training
        rows=pool[torch.randint(len(pool),(n,),device=self.device)]
        return ModelBatch(self.states[rows,:horizon+1].transpose(0,1),
            self.actions[rows,:horizon].transpose(0,1),self.xi[rows],self.failed[rows,:horizon].transpose(0,1))


def generate(config,output,windows=1024,seed=740000):
    from .train import model_manifest
    torch.set_num_threads(1);cfg=config_from_dict(json.loads(Path(config).read_text()))
    sim=BatchedSimulator(num_envs=16,seed=seed,dt=cfg.sim.dt,timestep=cfg.sim.timestep,friction=1.,
        leg_action_scale=cfg.sim.leg_action_scale,arm_action_scale=cfg.sim.arm_action_scale,wheel_action_scale=cfg.sim.wheel_action_scale,
        arm_initial_position=cfg.sim.arm_initial_position)
    task=WholeBodyTask(sim.model,sim.spec,reward_cfg=cfg.reward)
    provenance=model_manifest(sim.spec)
    demos=Demonstrations(cfg.demonstrations.directory,task,provenance['physics_hash'])
    rng=np.random.default_rng(seed);states=[];actions=[];xis=[];failed=[];validation=[];origins=[]
    noise_scale=np.array([.006]*12+[.035]*4+[.008]*6)
    lower,upper=np.asarray(cfg.policy.action_lower),np.asarray(cfg.policy.action_upper)
    for start in range(0,windows,16):
        rows=[];times=[];val=[]
        for i in range(16):
            pool=demos.validation if i>=12 else demos.training
            rows.append(int(pool[rng.integers(len(pool))]));times.append(int(rng.integers(demos.length-cfg.model.window)))
            val.append(i>=12)
        rows=np.asarray(rows);times=np.asarray(times)
        sim.reset(state=demos.states[rows,times].numpy());sim.xi[:]=demos.xi[rows].numpy()
        history=[sim.read_state().copy()];aa=[];ff=[]
        perturbation=rng.normal(size=(16,22))*noise_scale
        for step in range(cfg.model.window):
            desired=demos.actions[rows,times+step].numpy()+perturbation
            desired=np.clip(desired,lower,upper)
            sn=sim.step(desired).copy();history.append(sn);aa.append(sim.applied_action.copy())
            ff.append(physical_failure(task,torch.tensor(sn,dtype=torch.float32),sim.metrics()).numpy())
        states.append(np.stack(history).transpose(1,0,2));actions.append(np.stack(aa).transpose(1,0,2))
        failed.append(np.stack(ff).T);xis.append(sim.xi.copy());validation.extend(val)
        origins.extend([dict(clip=int(e),time=int(t)) for e,t in zip(rows,times)])
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    metadata=dict(format='wf_real_excitation_v1',physics_hash=provenance['physics_hash'],dt=cfg.sim.dt,seed=seed,
        source_manifest_sha256=demos.fingerprint,source_directory=cfg.demonstrations.directory,
        collection='actual MuJoCo windows; reset only at recorded initial states; no kinematic next-state labels',origins=origins[:windows])
    np.savez_compressed(output,states=np.concatenate(states)[:windows].astype('float32'),
        actions=np.concatenate(actions)[:windows].astype('float32'),xi=np.concatenate(xis)[:windows].astype('float32'),
        failed=np.concatenate(failed)[:windows],validation=np.asarray(validation)[:windows],metadata=json.dumps(metadata))
    sim.close();print(json.dumps(dict(windows=windows,real_transitions=windows*cfg.model.window,file=str(output))))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    p.add_argument('--windows',type=int,default=1024);p.add_argument('--seed',type=int,default=740000)
    a=p.parse_args();generate(a.config,a.output,a.windows,a.seed)


if __name__=='__main__':main()
