"""Standalone six-action training. Never writes to the legacy run or sources."""
import argparse
import json
import os
from pathlib import Path
import signal
import time
import random
import shutil

os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS","2")
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR","/tmp/fixed_arm_compile_step/inductor")
os.environ.setdefault("TRITON_CACHE_DIR","/tmp/fixed_arm_compile_step/triton")

import mujoco
import numpy as np
import torch

from . import FORMAT
from .data import generate, load, load_additional_training, local_tracking_curriculum, Replay
from .hierarchy import evaluate
from .io import atomic_json, utcnow, sha256
from .learning import Learner, RESIDUAL_ARCH, GLOBAL_ARCH, GOAL_ARCH, STRUCTURED_ARCH
from .robot import Simulator
from .task import Task


def collect(learner, sim, references, config, rng):
    """New student data, including collision failures; no reference actions."""
    sim.reset()
    chosen = [references[rng.integers(len(references))] for _ in range(sim.num_envs)]
    if config.get("initialize_from_reference_state",False):
        initial=np.stack([e["states"][0] for e in chosen])
        sim.restore_reference_state(initial,np.stack([e["previous"][0] for e in chosen]))
    active = np.ones(sim.num_envs,dtype=bool)
    samples = [dict(states=[sim.state()[i]],actions=[],previous=[],goals=[],failed=[],mode=np.array("online"))
               for i in range(sim.num_envs)]
    previous = torch.as_tensor(sim.previous.copy(),device=learner.device,dtype=torch.float32)
    steps=(max(len(e["actions"]) for e in chosen) if config.get("full_reference_collection",False)
           else config["collection_steps"])
    for t in range(steps):
        goals = np.stack([e["goals"][min(t,len(e["goals"])-1)] for e in chosen])
        with torch.no_grad():
            state=torch.as_tensor(sim.state(),device=learner.device)
            goal=torch.as_tensor(goals,device=learner.device)
            action=learner.policy_step(state,previous,goal).cpu().numpy()
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
    episodes=[]
    for row,reference in zip(samples,chosen):
        if len(row["actions"])<config["model_window"]:continue
        row["mode"]=np.array(str(reference["mode"]))
        if bool(reference.get("is_recovery",False)):row["is_recovery"]=np.array(True)
        episodes.append({k:(v if k=="mode" else np.asarray(v,dtype=np.float32)) for k,v in row.items()})
    return episodes


def run(config_path):
    config=json.loads(Path(config_path).read_text())
    guidance_start=(0 if config.get("preserve_guidance_schedule",False) else
                    config.get("guidance_stage_start_updates",config.get("stage_start_updates",0)))
    if guidance_start>config.get("stage_start_updates",0):
        raise ValueError("guidance schedule starts after the source actor counter")
    root=Path(config["output"]).resolve();root.mkdir(parents=True,exist_ok=True)
    framework=Path(__file__).resolve().parents[1]
    if not root.is_relative_to(framework):
        raise ValueError("all new training output must stay in the independent framework")
    torch.set_num_threads(config["cpu_threads"])
    random.seed(config["seed"]);np.random.seed(config["seed"]);torch.manual_seed(config["seed"])
    device=torch.device(config["device"])
    config.setdefault("compile_imagined_step",device.type=="cuda")
    if device.type=="cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; refusing silent device substitution")
    sim=Simulator(config["asset_dir"],config["num_envs"],dt=config["dt"],timestep=config["timestep"])
    task=Task(sim.model,sim.spec,config["dt"],config["joint_target_speed"],config["clearance_points"]).to(device)
    task.velocity_error_penalty=config.get("velocity_error_penalty",0.)
    config.setdefault("proximal_collision_weight",2. if config.get("full_reference_collection",False) else 0.)
    if config["proximal_collision_weight"]:
        from .proximal_collision import ProximalClearance
        task.proximal_clearance=ProximalClearance(sim.model).to(device)
        task.proximal_collision_weight=config["proximal_collision_weight"]
    learner=Learner(task,config);rng=np.random.default_rng(config["seed"]+700)
    pretrain_model=pretrain_actor=0
    started=time.monotonic();stop=False
    state=dict(stage="initializing",pid=os.getpid(),started=utcnow(),format=FORMAT,
        action_dim=6,physical_dofs=6,base_fixed=True,student_policy_qualified=False,
        training_module=str(Path(__file__).resolve()),actor_architecture=config.get("actor_architecture","mlp"),
        trainable_actor_parameters=sum(p.numel() for p in learner.actor.parameters() if p.requires_grad),
        frozen_actor_parameters=sum(p.numel() for p in learner.actor.parameters() if not p.requires_grad))
    def status(stage,**extra):
        state.update(stage=stage,heartbeat=utcnow(),actor_updates=learner.actor_updates,
            model_updates=learner.model_updates,elapsed_s=round(time.monotonic()-started,1),
            pretrain_actor=pretrain_actor,pretrain_model=pretrain_model)
        state.update(extra)
        atomic_json(root/"status.json",state)
    def signal_stop(*_):
        nonlocal stop
        stop=True
    signal.signal(signal.SIGTERM,signal_stop);signal.signal(signal.SIGINT,signal_stop)
    atomic_json(root/"config.json",config)
    status("generating_references")
    reference_root=framework/"data"/config["dataset_name"]
    references=generate(config["asset_dir"],config,config["reference_clips"],config["seed"]+101,reference_root/"train","train")
    original_references=list(references)
    for directory in config.get("extra_train_reference_dirs", []):
        references.extend(load_additional_training(directory))
    if config.get("local_tracking_curriculum",False):
        references,metadata=local_tracking_curriculum(references,task)
        atomic_json(root/"local_tracking_curriculum.json",dict(episodes=metadata,
            uses_validation_or_test_targets=False,physical_reference_files_unchanged=True,
            goal_dim=26,external_tcp_goal_dim=13,planner_reference_dim=12,planned_mode_flag_dim=1))
    validation=generate(config["asset_dir"],config,config["validation_clips"],config["seed"]+202,reference_root/"validation","validation")
    for directory in config.get("extra_validation_reference_dirs",[]):
        directory=Path(directory)
        manifest=json.loads((directory/"manifest.json").read_text())
        if manifest["split"] not in ("development_v2","development_v3"):
            raise ValueError("only a dedicated development split may augment validation")
        for row in manifest["clips"]:
            if sha256(directory/row["file"])!=row["sha256"]:
                raise RuntimeError("extra development reference digest changed")
        validation.extend(load(directory))
    # Freeze final-test fixtures before any actor/model optimization. They are
    # never passed to replay, BC, critic training or development selection.
    test=generate(config["asset_dir"],config,config["test_clips"],config["seed"]+303,reference_root/"test","test")
    recovery_sampling=dict(recovery_prefix_frames=config.get("recovery_prefix_frames",0),
                           recovery_prefix_fraction=config.get("recovery_prefix_fraction",0.))
    demonstration=Replay(references,device,config["seed"]+1,**recovery_sampling)
    heldout=Replay(validation,device,config["seed"]+2)
    online=Replay([],device,config["seed"]+3,capacity=128,**recovery_sampling)
    anchor_replay=None
    if config.get("policy_deviation_weight",0.):
        anchor_path=Path(config["policy_anchor_checkpoint"])
        if config.get("policy_anchor_sha256") and sha256(anchor_path)!=config["policy_anchor_sha256"]:
            raise RuntimeError("frozen source policy checkpoint changed")
        anchor_checkpoint=torch.load(anchor_path,map_location="cpu",weights_only=False)
        for key in ("asset_dir","dt","timestep","joint_target_speed","actor_hidden"):
            if anchor_checkpoint["config"][key]!=config[key]:raise ValueError("source policy physical/input format differs")
        learner.set_policy_anchor(anchor_checkpoint["actor"])
        anchor_replay=Replay(original_references+anchor_checkpoint["online_episodes"],device,config["seed"]+1709)
        atomic_json(root/"policy_anchor.json",dict(checkpoint=str(anchor_path),sha256=sha256(anchor_path),
            source_actor_updates=anchor_checkpoint["actor_updates"],training_episodes=len(anchor_replay.episodes),
            frozen=True,weight=config["policy_deviation_weight"],uses_evaluation_data=False))
        del anchor_checkpoint
    best=None;best_score=None;additional_bc=0
    checkpoint=root/"checkpoint_latest.pt"
    warm_start=config.get("warm_start")
    source=checkpoint if checkpoint.exists() else (Path(warm_start) if warm_start else None)
    if source is None and config.get("actor_architecture") in (RESIDUAL_ARCH,GLOBAL_ARCH,GOAL_ARCH,STRUCTURED_ARCH):
        raise ValueError("a frozen-source residual policy requires a trained six-joint arm checkpoint")
    if source is not None:
        ck=torch.load(source,map_location=device,weights_only=False)
        exact_resume=checkpoint.exists()
        if ck["format"] != FORMAT or (exact_resume and ck["config"] != config):
            raise ValueError("checkpoint format/config differs; use a distinct output directory")
        new_residual=(config.get("actor_architecture")==RESIDUAL_ARCH
                      and ck["config"].get("actor_architecture")!=RESIDUAL_ARCH)
        new_global=(config.get("actor_architecture")==GLOBAL_ARCH
                    and ck["config"].get("actor_architecture")==RESIDUAL_ARCH)
        new_goal=(config.get("actor_architecture")==GOAL_ARCH
                  and ck["config"].get("actor_architecture")!=GOAL_ARCH)
        new_structured=(config.get("actor_architecture")==STRUCTURED_ARCH
                        and ck["config"].get("actor_architecture")!=STRUCTURED_ARCH)
        new_actor=new_residual or new_global or new_goal or new_structured
        new_gain_head=(config.get("actor_architecture")==STRUCTURED_ARCH
                       and ck["config"].get("actor_architecture")==STRUCTURED_ARCH
                       and config.get("independent_velocity_gains",False)
                       and not ck["config"].get("independent_velocity_gains",False))
        changed_trainable_part=(config.get("train_local_correction_only",False)!=ck["config"].get("train_local_correction_only",False))
        fresh_actor_optimizer=new_actor or changed_trainable_part or new_gain_head
        if new_structured:
            if ck["config"].get("actor_architecture")!=GLOBAL_ARCH:
                raise ValueError("structured feedback requires the retained global/local SuperMimic arm")
            learner.actor.source.load_state_dict(ck["actor"])
            encoder_path=Path(config["goal_encoder_checkpoint"])
            if sha256(encoder_path)!=config["goal_encoder_sha256"]:raise RuntimeError("goal encoder changed")
            encoder=torch.load(encoder_path,map_location=device,weights_only=False)
            learner.actor.goal_encoder.load_state_dict(encoder["encoder"])
        elif new_goal:
            architecture=ck["config"].get("actor_architecture")
            if architecture==GLOBAL_ARCH:
                learner.actor.source.load_state_dict({k.removeprefix("source."):v for k,v in ck["actor"].items() if k.startswith("source.")})
            elif architecture==RESIDUAL_ARCH:learner.actor.source.load_state_dict(ck["actor"])
            else:raise ValueError("learned goal representation requires a retained v4 source")
            encoder_path=Path(config["goal_encoder_checkpoint"])
            if sha256(encoder_path)!=config["goal_encoder_sha256"]:
                raise RuntimeError("frozen learned goal representation changed")
            encoder=torch.load(encoder_path,map_location=device,weights_only=False)
            if not (torch.equal(encoder["low"].to(device),task.low) and torch.equal(encoder["high"].to(device),task.high)):
                raise ValueError("goal representation joint ranges differ")
            learner.actor.goal_encoder.load_state_dict(encoder["encoder"])
        elif new_global:
            learner.actor.source.load_state_dict(ck["actor"])
        elif new_residual:
            learner.actor.base.load_state_dict(ck["actor"])
        elif new_gain_head:
            weights=dict(ck["actor"]);last=str(len(learner.actor.gains)-1)
            for suffix in ("weight","bias"):
                key=f"gains.{last}.{suffix}";weights[key]=torch.cat((weights[key],weights[key]),0)
            learner.actor.load_state_dict(weights)
        else:learner.actor.load_state_dict(ck["actor"])
        for name,module in (("world_model",learner.wm),("critic",learner.critic)):
            module.load_state_dict(ck[name])
        for name,opt in (("actor_opt",learner.actor_opt),("model_opt",learner.wm_opt),("critic_opt",learner.critic_opt)):
            if name!="actor_opt" or not fresh_actor_optimizer:opt.load_state_dict(ck[name])
        learner.actor_updates=ck["actor_updates"];learner.model_updates=ck["model_updates"]
        pretrain_actor=ck["pretrain_actor"];pretrain_model=ck["pretrain_model"]
        online.episodes=ck["online_episodes"]
        for replay,name in ((online,"online_rng"),(demonstration,"demo_rng"),(heldout,"validation_rng")):
            replay.rng.bit_generator.state=ck[name]
        rng.bit_generator.state=ck["rng"];torch.set_rng_state(ck["torch_rng"].cpu())
        if exact_resume and anchor_replay is not None and ck.get("anchor_rng") is not None:
            anchor_replay.rng.bit_generator.state=ck["anchor_rng"]
        if device.type=="cuda":torch.cuda.set_rng_state_all([x.cpu() for x in ck["cuda_rng"]])
        if exact_resume:
            best=ck["best"];best_score=ck["best_score"]
            info=root/"warm_start.json"
            if info.exists():additional_bc=json.loads(info.read_text()).get("additional_bc_updates",0)
        else:
            if learner.actor_updates != config["stage_start_updates"]:
                raise ValueError("stage source counter mismatch")
            for opt,key in ((learner.actor_opt,"actor_lr"),(learner.wm_opt,"model_lr"),(learner.critic_opt,"critic_lr")):
                for group in opt.param_groups:group["lr"]=config[key]
            if config.get("reset_online_replay",False):online.episodes=[]
            if config.get("extra_model_pretrain_updates",0):
                pretrain_model=config["model_pretrain_updates"]-config["extra_model_pretrain_updates"]
            if config.get("retain_baseline_best",False):
                best=ck["best"];best_score=ck["best_score"]
                prior_best=Path(warm_start).parent/"actor_best_validation.pt"
                if best is not None:
                    shutil.copy2(prior_best,root/"actor_best_validation.pt")
            additional_bc=max(0,config["bc_pretrain_updates"]-pretrain_actor)
            atomic_json(root/"warm_start.json",dict(source=str(source),source_actor_updates=learner.actor_updates,
                model_updates=learner.model_updates,weights_retained=True,optimizers_retained=not fresh_actor_optimizer,
                source_actor_weights_frozen=new_actor,new_zero_residual=new_actor,
                far_policy_initialization=("collision-checked generated references" if new_structured else
                    "bounded learned joint-goal representation" if new_goal else "retained source"),
                actor_optimizer_retained=not fresh_actor_optimizer,changed_trainable_part=changed_trainable_part,
                actor_optimizer_reset_reason=("new bounded positive task-gain network; prior arm frozen" if new_structured else
                    "independent velocity gain rows added; fresh optimizer from zero-update source" if new_gain_head else
                    "newly trainable local correction; global weights frozen" if changed_trainable_part else None),
                world_model_critic_optimizers_retained=True,
                rng_retained=True,online_replay_retained=not config.get("reset_online_replay",False),
                repeat_bc=bool(additional_bc),additional_bc_updates=additional_bc,
                source_pretrain_actor=pretrain_actor,bc_phase_target=config["bc_pretrain_updates"],
                plan=config.get("adaptation_plan")))
        status("resuming_retained_state",source_checkpoint=str(source),
            pretrain_actor=pretrain_actor,pretrain_model=pretrain_model)
    def save():
        ck=dict(format=FORMAT,config=config,actor=learner.actor.state_dict(),world_model=learner.wm.state_dict(),
            critic=learner.critic.state_dict(),actor_opt=learner.actor_opt.state_dict(),model_opt=learner.wm_opt.state_dict(),
            critic_opt=learner.critic_opt.state_dict(),actor_updates=learner.actor_updates,model_updates=learner.model_updates,
            pretrain_actor=pretrain_actor,pretrain_model=pretrain_model,online_episodes=online.episodes,
            online_rng=online.rng.bit_generator.state,demo_rng=demonstration.rng.bit_generator.state,
            validation_rng=heldout.rng.bit_generator.state,rng=rng.bit_generator.state,
            torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if device.type=="cuda" else [],
            anchor_rng=anchor_replay.rng.bit_generator.state if anchor_replay is not None else None,
            best=best,best_score=best_score)
        temp=checkpoint.with_suffix(".tmp");torch.save(ck,temp);temp.replace(checkpoint)
    def log(**extra):
        record=dict(time=utcnow(),actor_updates=learner.actor_updates,model_updates=learner.model_updates,
                    elapsed_s=round(time.monotonic()-started,1),**extra)
        line=json.dumps(record,allow_nan=False)
        with (root/"metrics.jsonl").open("a") as f:f.write(line+"\n")
        print(line,flush=True)
    try:
        if (config.get("collect_before_model_refit",False) and
            pretrain_model<config["model_pretrain_updates"] and not online.episodes):
            status("collecting_before_model_refit")
            for _ in range(config.get("refit_collection_batches",4)):
                for episode in collect(learner,sim,references,config,rng):online.add(episode)
        while pretrain_model<config["model_pretrain_updates"] and not stop:
            source=(online if config.get("collect_before_model_refit",False) and online.episodes
                    and rng.random()<.5 else demonstration)
            metrics=learner.update_model(source.windows(config["model_batch"],config["model_window"]))
            pretrain_model+=1
            if pretrain_model%100==0:
                log(phase="model_pretrain",pretrain_model=pretrain_model,**metrics)
                status("pretraining_world_model",pretrain_model=pretrain_model)
            if pretrain_model%1000==0:save()
        # Preserve a genuinely model-only snapshot BEFORE any requested
        # supplementary demonstration fitting changes the Actor.
        refit_marker=root/"model_refit_completed.json"
        if (config.get("extra_model_pretrain_updates",0) and not refit_marker.exists()
            and pretrain_model>=config["model_pretrain_updates"] and not stop):
            save();shutil.copy2(checkpoint,root/"checkpoint_after_model_refit.pt")
            atomic_json(refit_marker,dict(actor_updates=learner.actor_updates,model_updates=learner.model_updates,
                pretrain_actor=pretrain_actor,pretrain_model=pretrain_model,
                extra_model_updates=config["extra_model_pretrain_updates"],online_episodes=len(online.episodes),
                snapshot=str(root/"checkpoint_after_model_refit.pt")))
        while pretrain_actor<config["bc_pretrain_updates"] and not stop:
            bc=learner.pretrain_actor(demonstration.starts(config["bc_batch"]))
            pretrain_actor+=1
            if pretrain_actor%200==0:
                log(phase="bc_pretrain",pretrain_actor=pretrain_actor,bc_loss=bc)
                status("pretraining_actor",pretrain_actor=pretrain_actor)
            if pretrain_actor%2000==0:save()
        guidance_marker=root/"guidance_refit_completed.json"
        if additional_bc and pretrain_actor>=config["bc_pretrain_updates"] and not stop and not guidance_marker.exists():
            save();shutil.copy2(checkpoint,root/"checkpoint_after_guidance_refit.pt")
            atomic_json(guidance_marker,dict(actor_updates=learner.actor_updates,model_updates=learner.model_updates,
                pretrain_actor=pretrain_actor,additional_bc_updates=additional_bc,
                snapshot=str(root/"checkpoint_after_guidance_refit.pt")))
        invocation_limit=min(config["actor_updates"],config.get("probe_stop_updates",config["actor_updates"]))
        if config.get("evaluate_initial_candidate",False) and best_score is None and not stop:
            status("evaluating_initial_candidate")
            report=evaluate(learner,validation,config["asset_dir"],config,root/"validation_initial.json")
            groups=list(report["groups"].values())
            rates=[g["successes"]/max(1,g["cases"]) for g in groups]
            failures=[g["failures"]/max(1,g["cases"]) for g in groups]
            tracking=[r["tracking_fraction_after_50_frames"] for r in report["rows"] if r["mode"]=="tracking"]
            best_score=(-max(failures),min(rates),sum(rates),float(np.mean(tracking)),
                -sum(g["mean_position_m"] for g in groups if g["cases"]))
            best=learner.actor_updates
            initial=dict(format=FORMAT,config=config,actor=learner.actor.state_dict(),actor_updates=best)
            torch.save(initial,root/"actor_initial_validation.pt")
            torch.save(initial,root/"actor_best_validation.pt")
            log(phase="initial_validation",groups=report["groups"]);save()
        while learner.actor_updates<invocation_limit and not stop:
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
            preview=horizon if config.get("future_goal_sequences",False) else 0
            mixture=config.get("tracking_sampling_fraction")
            anchor_batch=(anchor_replay.starts(config.get("policy_anchor_batch",64))
                          if anchor_replay is not None else None)
            metrics.update(learner.update_actor(source.starts(config["actor_batch"],preview,mixture),
                demonstration.starts(config["bc_batch"],tracking_fraction=mixture),horizon,anchor_batch))
            if learner.actor_updates%config["log_interval"]==0:
                log(phase="supermimic_bptt",model_validation_error=float(validation_error),**metrics)
                status("training_supermimic")
            if learner.actor_updates%config["evaluation_interval"]==0:
                status("evaluating_validation")
                torch.save(dict(format=FORMAT,config=config,actor=learner.actor.state_dict(),
                    actor_updates=learner.actor_updates),root/f"actor_{learner.actor_updates:07d}.pt")
                report=evaluate(learner,validation,config["asset_dir"],config,
                                root/f"validation_{learner.actor_updates:07d}.json")
                groups=list(report["groups"].values())
                rates=[g["successes"]/max(1,g["cases"]) for g in groups]
                failures=[g["failures"]/max(1,g["cases"]) for g in groups]
                tracking=[r["tracking_fraction_after_50_frames"] for r in report["rows"] if r["mode"]=="tracking"]
                score=(-max(failures),min(rates),sum(rates),float(np.mean(tracking)),
                    -sum(g["mean_position_m"] for g in groups if g["cases"]))
                if config.get("legacy_candidate_selection",False):
                    score=(sum(g["successes"] for g in groups),-sum(g["failures"] for g in groups),
                           -sum(g["mean_position_m"] for g in groups if g["cases"]))
                if best_score is None or score>tuple(best_score):
                    best_score=score;best=learner.actor_updates
                    torch.save(dict(format=FORMAT,config=config,actor=learner.actor.state_dict(),
                                    actor_updates=best),root/"actor_best_validation.pt")
                log(phase="validation",groups=report["groups"])
                save()
                shutil.copy2(checkpoint,root/f"checkpoint_{learner.actor_updates:07d}.pt")
            if learner.actor_updates%config["checkpoint_interval"]==0:save()
        save()
        if stop:
            status("stopped_by_request",checkpoint=str(checkpoint));return
        status("evaluating_frozen_final_test")
        torch.save(dict(format=FORMAT,config=config,actor=learner.actor.state_dict(),
                        actor_updates=learner.actor_updates),root/"actor_final.pt")
        if invocation_limit<config["actor_updates"]:
            status("diagnostic_probe_completed",declared_budget_completed=False,
                   probe_actor_updates=invocation_limit,declared_actor_updates=config["actor_updates"],
                   frozen_final_test_not_used=True)
            return
        if config.get("skip_final_test",False):
            status("training_completed",selected_actor_updates=best or learner.actor_updates,
                frozen_final_test_not_used=True,
                qualification_note="development result only; campaign supervisor owns final-test evaluation")
            return
        if best is not None:
            selected=torch.load(root/"actor_best_validation.pt",map_location=device,weights_only=False)
            learner.actor.load_state_dict(selected["actor"])
        selected_path=root/("actor_best_validation.pt" if best is not None else "actor_final.pt")
        report=evaluate(learner,test,config["asset_dir"],config)
        report.update(actor_updates=best or learner.actor_updates,training_actor_updates=learner.actor_updates,
                      checkpoint=str(selected_path),checkpoint_sha256=sha256(selected_path))
        atomic_json(root/"final_test.json",report)
        status("training_completed",selected_actor_updates=best or learner.actor_updates,
            final_test_groups=report["groups"],student_policy_qualified=False,
            qualification_note="inspect final-test reach and tracking separately; no full-workspace claim")
    except Exception as error:
        save();status("failed",error=repr(error));raise


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--config",required=True)
    a=p.parse_args();run(a.config)


if __name__=="__main__":main()
