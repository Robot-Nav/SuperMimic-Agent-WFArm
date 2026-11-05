"""Standalone six-action training. Never writes to the legacy run or sources."""
import argparse
import json
import os
from pathlib import Path
import signal
import time
import random

import mujoco
import numpy as np
import torch

from . import FORMAT
from .data import generate, Replay
from .evaluate import evaluate
from .io import atomic_json, utcnow
from .learning import Learner
from .robot import Simulator
from .task import Task


def collect(learner, sim, references, config, rng):
    """New student data, including collision failures; no reference actions."""
    sim.reset()
    chosen = [references[rng.integers(len(references))] for _ in range(sim.num_envs)]
    active = np.ones(sim.num_envs,dtype=bool)
    samples = [dict(states=[sim.state()[i]],actions=[],previous=[],goals=[],failed=[],mode=np.array("online"))
               for i in range(sim.num_envs)]
    previous = torch.zeros((sim.num_envs,6),device=learner.device)
    for t in range(config["collection_steps"]):
        goals = np.stack([e["goals"][min(t,len(e["goals"])-1)] for e in chosen])
        with torch.no_grad():
            state=torch.as_tensor(sim.state(),device=learner.device)
            goal=torch.as_tensor(goals,device=learner.device)
            action=learner.task.command(learner.actor,state,previous,goal).cpu().numpy()
        action += rng.normal(0,config["exploration_std"],action.shape)
        delta=np.clip(action-sim.previous,-sim.dt*config["joint_target_speed"],sim.dt*config["joint_target_speed"])
        action=np.clip(sim.previous+delta,sim.spec.low,sim.spec.high)
        before=sim.previous.copy(); next_state=sim.step(action)
        for i in np.flatnonzero(active):
            row=samples[i]
            row["states"].append(next_state[i].copy());row["actions"].append(sim.previous[i].copy())
            row["previous"].append(before[i]);row["goals"].append(goals[i]);row["failed"].append(sim.failed[i])
            active[i] &= not sim.failed[i]
        if not active.any():
            break
        # Inactive worlds are parked at reset, never appended after a failure.
        sim.reset(ids=np.flatnonzero(~active))
        previous=torch.as_tensor(sim.previous,device=learner.device,dtype=torch.float32)
    return [{k:(v if k=="mode" else np.asarray(v,dtype=np.float32)) for k,v in row.items()}
            for row in samples if len(row["actions"])>=config["model_window"]]


def run(config_path):
    config=json.loads(Path(config_path).read_text())
    root=Path(config["output"]).resolve();root.mkdir(parents=True,exist_ok=True)
    framework=Path(__file__).resolve().parents[1]
    if not root.is_relative_to(framework):
        raise ValueError("all new training output must stay in the independent framework")
    torch.set_num_threads(config["cpu_threads"])
    random.seed(config["seed"]);np.random.seed(config["seed"]);torch.manual_seed(config["seed"])
    device=torch.device(config["device"])
    if device.type=="cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; refusing silent device substitution")
    sim=Simulator(config["asset_dir"],config["num_envs"],dt=config["dt"],timestep=config["timestep"])
    task=Task(sim.model,sim.spec,config["dt"],config["joint_target_speed"],config["clearance_points"]).to(device)
    learner=Learner(task,config);rng=np.random.default_rng(config["seed"]+700)
    started=time.monotonic();stop=False
    state=dict(stage="initializing",pid=os.getpid(),started=utcnow(),format=FORMAT,
        action_dim=6,physical_dofs=6,base_fixed=True,student_policy_qualified=False)
    def status(stage,**extra):
        state.update(stage=stage,heartbeat=utcnow(),actor_updates=learner.actor_updates,
            model_updates=learner.model_updates,elapsed_s=round(time.monotonic()-started,1),**extra)
        atomic_json(root/"status.json",state)
    def signal_stop(*_):
        nonlocal stop
        stop=True
    signal.signal(signal.SIGTERM,signal_stop);signal.signal(signal.SIGINT,signal_stop)
    atomic_json(root/"config.json",config)
    status("generating_references")
    reference_root=framework/"data"/config["dataset_name"]
    references=generate(config["asset_dir"],config,config["reference_clips"],config["seed"]+101,reference_root/"train","train")
    validation=generate(config["asset_dir"],config,config["validation_clips"],config["seed"]+202,reference_root/"validation","validation")
    # Freeze final-test fixtures before any actor/model optimization. They are
    # never passed to replay, BC, critic training or development selection.
    test=generate(config["asset_dir"],config,config["test_clips"],config["seed"]+303,reference_root/"test","test")
    demonstration=Replay(references,device,config["seed"]+1)
    heldout=Replay(validation,device,config["seed"]+2)
    online=Replay([],device,config["seed"]+3,capacity=128)
    pretrain_model=pretrain_actor=0;best=None;best_score=None
    checkpoint=root/"checkpoint_latest.pt"
    if checkpoint.exists():
        ck=torch.load(checkpoint,map_location=device,weights_only=False)
        if ck["format"] != FORMAT or ck["config"] != config:
            raise ValueError("checkpoint format/config differs; use a distinct output directory")
        for name,module in (("actor",learner.actor),("world_model",learner.wm),("critic",learner.critic)):
            module.load_state_dict(ck[name])
        for name,opt in (("actor_opt",learner.actor_opt),("model_opt",learner.wm_opt),("critic_opt",learner.critic_opt)):
            opt.load_state_dict(ck[name])
        learner.actor_updates=ck["actor_updates"];learner.model_updates=ck["model_updates"]
        pretrain_actor=ck["pretrain_actor"];pretrain_model=ck["pretrain_model"]
        online.episodes=ck["online_episodes"]
        for replay,name in ((online,"online_rng"),(demonstration,"demo_rng"),(heldout,"validation_rng")):
            replay.rng.bit_generator.state=ck[name]
        rng.bit_generator.state=ck["rng"];torch.set_rng_state(ck["torch_rng"].cpu())
        if device.type=="cuda":torch.cuda.set_rng_state_all([x.cpu() for x in ck["cuda_rng"]])
        best=ck["best"];best_score=ck["best_score"]
    def save():
        ck=dict(format=FORMAT,config=config,actor=learner.actor.state_dict(),world_model=learner.wm.state_dict(),
            critic=learner.critic.state_dict(),actor_opt=learner.actor_opt.state_dict(),model_opt=learner.wm_opt.state_dict(),
            critic_opt=learner.critic_opt.state_dict(),actor_updates=learner.actor_updates,model_updates=learner.model_updates,
            pretrain_actor=pretrain_actor,pretrain_model=pretrain_model,online_episodes=online.episodes,
            online_rng=online.rng.bit_generator.state,demo_rng=demonstration.rng.bit_generator.state,
            validation_rng=heldout.rng.bit_generator.state,rng=rng.bit_generator.state,
            torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if device.type=="cuda" else [],
            best=best,best_score=best_score)
        temp=checkpoint.with_suffix(".tmp");torch.save(ck,temp);temp.replace(checkpoint)
    def log(**extra):
        record=dict(time=utcnow(),actor_updates=learner.actor_updates,model_updates=learner.model_updates,
                    elapsed_s=round(time.monotonic()-started,1),**extra)
        line=json.dumps(record,allow_nan=False)
        with (root/"metrics.jsonl").open("a") as f:f.write(line+"\n")
        print(line,flush=True)
    try:
        while pretrain_model<config["model_pretrain_updates"] and not stop:
            metrics=learner.update_model(demonstration.windows(config["model_batch"],config["model_window"]))
            pretrain_model+=1
            if pretrain_model%100==0:
                log(phase="model_pretrain",pretrain_model=pretrain_model,**metrics)
                status("pretraining_world_model",pretrain_model=pretrain_model)
            if pretrain_model%1000==0:save()
        while pretrain_actor<config["bc_pretrain_updates"] and not stop:
            bc=learner.pretrain_actor(demonstration.starts(config["bc_batch"]))
            pretrain_actor+=1
            if pretrain_actor%200==0:
                log(phase="bc_pretrain",pretrain_actor=pretrain_actor,bc_loss=bc)
                status("pretraining_actor",pretrain_actor=pretrain_actor)
            if pretrain_actor%2000==0:save()
        while learner.actor_updates<config["actor_updates"] and not stop:
            if learner.actor_updates%config["collection_interval"]==0 or not online.episodes:
                status("collecting_student_rollouts")
                for episode in collect(learner,sim,references,config,rng):online.add(episode)
            for _ in range(config["model_batches"]):
                source=online if online.episodes and rng.random()<.5 else demonstration
                metrics=learner.update_model(source.windows(config["model_batch"],config["model_window"]))
            with torch.no_grad():
                _,validation_error=learner.wm.loss(**heldout.windows(config["model_batch"],config["model_window"]))
            if float(validation_error)>config["max_model_error"]:
                log(phase="actor_gated",validation_error=float(validation_error),**metrics)
                # A finite repair budget prevents an endless model-only loop.
                for _ in range(config["model_repair_updates"]):
                    metrics=learner.update_model(demonstration.windows(config["model_batch"],config["model_window"]))
                with torch.no_grad():
                    _,validation_error=learner.wm.loss(**heldout.windows(config["model_batch"],config["model_window"]))
                if float(validation_error)>config["max_model_error"]:
                    raise RuntimeError("held-out world-model error stayed above actor gate after refit")
            source=online if online.episodes and rng.random()<.5 else demonstration
            horizon=min(config["horizon"],config["initial_horizon"]+learner.actor_updates//config["horizon_ramp"])
            metrics.update(learner.update_actor(source.starts(config["actor_batch"]),
                demonstration.starts(config["bc_batch"]),horizon))
            if learner.actor_updates%config["log_interval"]==0:
                log(phase="supermimic_bptt",model_validation_error=float(validation_error),**metrics)
                status("training_supermimic")
            if learner.actor_updates%config["evaluation_interval"]==0:
                status("evaluating_validation")
                report=evaluate(learner,validation,config["asset_dir"],config,
                                root/f"validation_{learner.actor_updates:07d}.json")
                score=(sum(g["successes"] for g in report["groups"].values()),
                    -sum(g["failures"] for g in report["groups"].values()),
                    -sum(g["mean_position_m"] for g in report["groups"].values() if g["cases"]))
                if best_score is None or score>tuple(best_score):
                    best_score=score;best=learner.actor_updates
                    torch.save(dict(format=FORMAT,config=config,actor=learner.actor.state_dict(),
                                    actor_updates=best),root/"actor_best_validation.pt")
                log(phase="validation",groups=report["groups"])
            if learner.actor_updates%config["checkpoint_interval"]==0:save()
        save()
        if stop:
            status("stopped_by_request",checkpoint=str(checkpoint));return
        status("evaluating_frozen_final_test")
        torch.save(dict(format=FORMAT,config=config,actor=learner.actor.state_dict(),
                        actor_updates=learner.actor_updates),root/"actor_final.pt")
        if best is not None:
            selected=torch.load(root/"actor_best_validation.pt",map_location=device,weights_only=False)
            learner.actor.load_state_dict(selected["actor"])
        report=evaluate(learner,test,config["asset_dir"],config,root/"final_test.json")
        status("training_completed",selected_actor_updates=best or learner.actor_updates,
            final_test_groups=report["groups"],student_policy_qualified=False,
            qualification_note="inspect final-test reach and tracking separately; no full-workspace claim")
    except Exception as error:
        save();status("failed",error=repr(error));raise


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--config",required=True)
    a=p.parse_args();run(a.config)


if __name__=="__main__":main()
