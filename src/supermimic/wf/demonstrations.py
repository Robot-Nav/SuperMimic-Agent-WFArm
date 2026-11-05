"""Immutable, clip-disjoint reference replay for supervised and BPTT guidance."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .replay import ModelBatch, PolicyBatch


class Demonstrations:
    def __init__(self, directory, task, physics_hash,query_file=None,query_fraction=.75,command_fraction=0.,zero_start_fraction=0.,broad_anchor_fraction=0.):
        self.directory=Path(directory)
        raw=(self.directory/"manifest.json").read_bytes()
        self.manifest=json.loads(raw)
        self.fingerprint=hashlib.sha256(raw).hexdigest()
        self.query_inputs=None;self.query_fraction=query_fraction
        self.query_target_mode=None
        self.command_fraction=command_fraction
        self.zero_start_fraction=zero_start_fraction
        self.broad_anchor_fraction=broad_anchor_fraction
        m=self.manifest
        if m.get("format")!="wf_physical_references_v1" or not m.get("complete"):
            raise ValueError("expected a complete physically screened WF reference library")
        if m["provenance"]["physics_hash"]!=physics_hash:
            raise ValueError("reference physics differs from the training robot")
        if m["provenance"]["joint_order"]!=list(task.spec.joint_names):
            raise ValueError("reference joint order differs from the policy")
        arrays={k:[] for k in ("states","actions","goal","xi","initial_action")}
        splits=[]
        for clip in m["clips"]:
            file=self.directory/clip["file"]
            if hashlib.sha256(file.read_bytes()).hexdigest()!=clip["sha256"]:
                raise ValueError("reference clip bytes changed")
            if not clip["screen"]["accepted"]:
                raise ValueError("rejected clip cannot enter training")
            with np.load(file,allow_pickle=False) as d:
                states,actions=d["states"],d["actions"]
                if states.shape!=(len(actions)+1,57) or actions.shape!=(len(actions),22):
                    raise ValueError("invalid reference state/action dimensions")
                if d["failed"].any() or not d["safe"][-50:].all():
                    raise ValueError("reference contains a failure or unsafe final hold")
                for key in arrays:
                    a=(d[key] if key!='initial_action' or key in d.files else np.zeros(22))
                    if key=='goal' and a.shape[-1]<task.command_size:
                        a=np.r_[a,np.zeros(task.command_size-a.shape[-1])]
                    if not np.isfinite(a).all(): raise ValueError("non-finite reference data")
                    if key=='initial_action' and (a.shape!=(22,) or (np.abs(a)>1.00001).any()):
                        raise ValueError('invalid initial reference command history')
                    arrays[key].append(a.copy())
            splits.append(clip["split"])
        device=task.device
        for key,value in arrays.items():
            setattr(self,key,torch.as_tensor(np.stack(value),dtype=torch.float32,device=device))
        self.goal=task.command_goal(self.goal)
        if task.cfg.program_context:self.goal=task.initialize_goal(self.states[:,0],self.goal)
        self.device=device
        self.count,self.length=self.actions.shape[:2]
        self.training=torch.tensor([i for i,s in enumerate(splits) if s=="train"],device=device)
        self.validation=torch.tensor([i for i,s in enumerate(splits) if s=="validation"],device=device)
        default_level=m.get('level',0);default_level=default_level if isinstance(default_level,int) else 0
        self.levels=torch.tensor([x.get('level',default_level) for x in m['clips']],device=device)
        self.zero_start=torch.tensor([x.get('initial_pose_family')=='folded_zero' for x in m['clips']],device=device)
        self.broad_anchor=torch.tensor([x.get('subtype')=='broader_anchor_zero_reach' for x in m['clips']],device=device)
        if not len(self.training) or not len(self.validation):
            raise ValueError("whole clips must be split into training and validation sets")
        if (self.actions.abs()>1.00001).any(): raise ValueError("unexecutable reference action")
        if not np.isclose(m["dt"],task.spec.model.opt.timestep*round(m["dt"]/task.spec.model.opt.timestep)):
            raise ValueError("reference control period is inconsistent with physics")
        self.previous=torch.cat((self.initial_action[:,None],self.actions[:,:-1]),dim=1)
        # Analytic observations are deterministic and need no state gradients in BC.
        # Cache once to avoid recomputing the body tree for every supervised batch.
        with torch.no_grad():
            self.goals_at_states=self.goal[:,None,:].expand(-1,self.length+1,-1)
            if task.cfg.base_command_conditioning:
                self.goals_at_states=task.transport_goal(self.states[:,:1],self.states,self.goals_at_states)
            if task.cfg.goal_program:
                phase=(torch.arange(self.length+1,device=device,dtype=torch.float32)/task.cfg.program_steps).clamp_max(1.)
                self.goals_at_states=torch.cat((self.goals_at_states[...,:11],phase[None,:,None].expand(self.count,-1,-1),self.goals_at_states[...,12:]),-1)
            self.inputs=task.actor_input(self.states[:,:-1],self.previous,
                self.goals_at_states[:,:-1]).detach()
            minimum=torch.tensor([.05]*12+[.015]*4+[.05]*6,device=device)
            self.action_scale=self.actions[self.training].flatten(0,1).std(0).clamp_min(minimum)
            self.delta_scale=(self.actions[self.training]-self.previous[self.training]).flatten(0,1).std(0).clamp_min(.0002)
        if query_file:
            with np.load(query_file,allow_pickle=False) as q:
                metadata=json.loads(str(q['metadata'].item()))
                self.query_target_mode=metadata.get('intended_action_mode')
                if (metadata.get('format')!='wf_corrective_queries_v1' or metadata['source_manifest_sha256']!=self.fingerprint
                        or metadata['physics_hash']!=physics_hash):raise ValueError('query provenance differs from reference library')
                state=torch.tensor(q['state'],dtype=torch.float32,device=device)
                previous=torch.tensor(q['prev_action'],dtype=torch.float32,device=device)
                goal=torch.tensor(q['goal'],dtype=torch.float32,device=device)
                goal=task.command_goal(goal)
                self.query_actions=torch.tensor(q['teacher_action'],dtype=torch.float32,device=device)
                validation=torch.tensor(q['validation'],dtype=torch.bool,device=device)
            if state.shape!=(len(goal),57) or previous.shape!=(len(goal),22) or self.query_actions.shape!=previous.shape:
                raise ValueError('invalid query dimensions')
            if not all(torch.isfinite(v).all() for v in (state,previous,goal,self.query_actions)) or (self.query_actions.abs()>1.00001).any():
                raise ValueError('non-finite or unexecutable queried command')
            with torch.no_grad():self.query_inputs=task.actor_input(state,previous,goal).detach()
            self.query_training=(~validation).nonzero().flatten();self.query_validation=validation.nonzero().flatten()
            if not len(self.query_training) or not len(self.query_validation):raise ValueError('queries require clip-disjoint splits')
            self.query_scale=(self.query_actions[self.query_training]-previous[self.query_training]).std(0).clamp_min(.0002)
            self.fingerprint=hashlib.sha256((self.fingerprint+hashlib.sha256(Path(query_file).read_bytes()).hexdigest()).encode()).hexdigest()

    def indices(self,n,horizon=1,validation=False):
        if not 1<=horizon<=self.length: raise ValueError("invalid reference window")
        pool=self.validation if validation else self.training
        e=pool[torch.randint(len(pool),(n,),device=self.device)]
        if self.broad_anchor_fraction:
            broad_pool=pool[self.broad_anchor[pool]]
            if len(broad_pool):
                mask=torch.rand(n,device=self.device)<self.broad_anchor_fraction
                e[mask]=broad_pool[torch.randint(len(broad_pool),(int(mask.sum()),),device=self.device)]
        if self.zero_start_fraction:
            zero_pool=pool[self.zero_start[pool]]
            if len(zero_pool):
                mask=torch.rand(n,device=self.device)<self.zero_start_fraction
                e[mask]=zero_pool[torch.randint(len(zero_pool),(int(mask.sum()),),device=self.device)]
        if self.command_fraction and self.goal.shape[-1]>=11:
            commands=pool[self.goal[pool,10]>.5]
            if len(commands):
                mask=torch.rand(n,device=self.device)<self.command_fraction
                e[mask]=commands[torch.randint(len(commands),(int(mask.sum()),),device=self.device)]
        t=torch.randint(self.length-horizon+1,(n,),device=self.device)
        return e,t

    def sample_bc(self,n,validation=False):
        e,t=self.indices(n,validation=validation)
        return self.inputs[e,t],self.actions[e,t]

    def loss(self,actor,task,n,validation=False,noise_std=0.):
        if self.query_target_mode and getattr(actor,'action_mode','absolute')!=self.query_target_mode:
            raise ValueError('Actor action semantics differ from the queried teacher targets')
        scale=self.delta_scale if (getattr(actor,'action_mode','absolute')=='delta' or getattr(actor,'target_supervision_scale','absolute')=='delta') else self.action_scale
        if self.query_inputs is None:
            inputs,target=self.sample_bc(n,validation)
        else:
            qn=round(n*self.query_fraction);rn=n-qn
            inputs,target=self.sample_bc(rn,validation)
            pool=self.query_validation if validation else self.query_training
            rows=pool[torch.randint(len(pool),(qn,),device=self.device)]
            inputs=torch.cat((inputs,self.query_inputs[rows]));target=torch.cat((target,self.query_actions[rows]))
            scale=torch.cat((scale.expand(rn,-1),self.query_scale.expand(qn,-1)))
        if noise_std and not validation:
            # Sensor augmentation only. This does NOT create world-model labels
            # or claim perturbed states were executed by the simulator.
            inputs=inputs.clone()
            # Command history is known exactly, rather than a noisy sensor.
            inputs[:,:50]+=noise_std*torch.randn_like(inputs[:,:50])
            o=task.obs_dim
            inputs[:,o:o+3]+=.15*noise_std*torch.randn_like(inputs[:,o:o+3])
            inputs[:,o+3:o+9]+=.25*noise_std*torch.randn_like(inputs[:,o+3:o+9])
            inputs[:,o+9:o+12]+=.15*noise_std*torch.randn_like(inputs[:,o+9:o+12])
        predicted=(actor.supervised_action(inputs) if getattr(actor,'action_mode','absolute')=='target' else task.process_action(actor(inputs)))
        error=(predicted-target)/scale
        # Give the wheel and arm groups equal influence to the twelve leg joints.
        return (error[:,:12].square().mean()+error[:,12:16].square().mean()
                +error[:,16:].square().mean())/3

    def sample_model(self,n,horizon,validation=False):
        e,t=self.indices(n,horizon,validation)
        tt=t[None]+torch.arange(horizon+1,device=self.device)[:,None]
        return ModelBatch(self.states[e[None],tt],self.actions[e[None],tt[:-1]],self.xi[e],
                          torch.zeros(horizon,n,device=self.device))

    @torch.no_grad()
    def validation_loss(self,actor,task,chunk_size=4096):
        """The same desired-action semantics as BC, over every held-out frame.

        Scoring executable slew-limited commands against teacher-forced command
        histories instead can hide a badly predicted desired target. Chunking
        bounds memory independently of the size of the reference library.
        """
        inputs=self.inputs[self.validation].flatten(0,1)
        targets=self.actions[self.validation].flatten(0,1)
        scale=(self.delta_scale if (getattr(actor,'action_mode','absolute')=='delta'
            or getattr(actor,'target_supervision_scale','absolute')=='delta') else self.action_scale)
        total=inputs.new_zeros(())
        for start in range(0,len(inputs),chunk_size):
            x=inputs[start:start+chunk_size];target=targets[start:start+chunk_size]
            predicted=(actor.supervised_action(x) if getattr(actor,'action_mode','absolute')=='target'
                else task.process_action(actor(x)))
            error=(predicted-target)/scale
            loss=(error[:,:12].square().mean()+error[:,12:16].square().mean()+error[:,16:].square().mean())/3
            total+=len(x)*loss
        return total/len(inputs)

    def hold_loss(self,actor,task,n):
        """Goal-consistency regularizer at physically settled reference endpoints.

        Zero-velocity/achieved-goal examples regularize the Actor only. They are
        never inserted into dynamics replay as if they were real transitions.
        """
        e,_=self.indices(n)
        state=self.states[e,-1].clone()
        state[:,29:]=0.
        previous=self.actions[e,-1]
        goal=task.tcp_pose(state).detach()
        if task.cfg.goal_program:
            goal=task.initialize_goal(state,goal);goal[...,11]=1.
        predicted=task.process_action(actor(task.actor_input(state,previous,goal)))
        scale=self.delta_scale if getattr(actor,'action_mode','absolute')=='delta' else self.action_scale
        return ((predicted-previous)/scale).square().mean()

    def sample_policy(self,n,horizon):
        e,t=self.indices(n,horizon)
        tt=t[None]+torch.arange(horizon+1,device=self.device)[:,None]
        return PolicyBatch(self.states[e,t],self.previous[e,t],self.goals_at_states[e,t],self.xi[e]),self.states[e[None],tt]
