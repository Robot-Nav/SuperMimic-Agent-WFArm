"""Independent batched native worlds; same full-pose task qualification."""
from pathlib import Path
import argparse

import numpy as np
import torch

from .data import load
from .evaluate import evaluate as sequential_evaluate
from .learning import Learner
from .robot import Simulator
from .task import Task
from .native_metrics import tcp_error_norms
from .io import atomic_json,sha256


@torch.no_grad()
def evaluate(learner,episodes,asset_dir,config,output=None,batch_size=16):
    rows=[];device=learner.task.low.device
    for begin in range(0,len(episodes),batch_size):
        block=episodes[begin:begin+batch_size];n=len(block)
        sim=Simulator(asset_dir,n,dt=config["dt"],timestep=config["timestep"])
        sim.reset();previous=torch.zeros(n,6,device=device)
        if hasattr(learner,"start_episodes"):learner.start_episodes(block,sim)
        active=np.ones(n,bool);failed=np.zeros(n,bool);hold=np.zeros(n,int);maxhold=hold.copy()
        errors=[[] for _ in block];passing=[[] for _ in block]
        lengths=np.array([len(e["actions"]) for e in block])
        for frame in range(int(lengths.max())):
            goals=np.stack([e["goals"][min(frame,len(e["goals"])-1)] for e in block])
            state=torch.as_tensor(sim.state(),device=device)
            action=learner.inference_action(state,previous,torch.as_tensor(goals,device=device))
            sim.step(action.cpu().numpy())
            previous=torch.as_tensor(sim.previous.copy(),device=device,dtype=torch.float32)
            norms=tcp_error_norms(sim,goals)
            for i in np.flatnonzero(active):
                value=norms[i];bad=bool(sim.failed[i]);failed[i]|=bad
                ok=bool((value<np.array([.03,.15,.08,.2])).all()) and not bad
                errors[i].append(value);passing[i].append(ok)
                hold[i]=hold[i]+1 if ok else 0;maxhold[i]=max(maxhold[i],hold[i])
                if bad or frame+1>=lengths[i]:active[i]=False
            if not active.any():break
            # Independent MjData worlds: parking inactive rows cannot affect
            # remaining trajectories, and no extra row is counted afterwards.
            sim.reset(ids=np.flatnonzero(~active))
            previous=torch.as_tensor(sim.previous.copy(),device=device,dtype=torch.float32)
        for i,e in enumerate(block):
            value=np.array(errors[i]);fraction=sum(passing[i][50:])/max(1,int(lengths[i])-50)
            tracking=str(e["mode"])=="tracking"
            success=not failed[i] and (fraction>=.95 if tracking else hold[i]>=config["hold_steps"])
            rows.append(dict(case=begin+i,mode=str(e["mode"]),success=bool(success),failed=bool(failed[i]),
                requested_frames=int(lengths[i]),executed_frames=len(value),final_hold_steps=int(hold[i]),maximum_hold_steps=int(maxhold[i]),
                tracking_fraction_after_50_frames=float(fraction),position_mean_m=float(value[:,0].mean()),
                position_p95_m=float(np.quantile(value[:,0],.95)),orientation_mean_rad=float(value[:,1].mean()),
                orientation_p95_rad=float(np.quantile(value[:,1],.95)),linear_velocity_error_mean=float(value[:,2].mean()),
                angular_velocity_error_mean=float(value[:,3].mean()),position_terminal_m=float(value[-1,0]),
                orientation_terminal_rad=float(value[-1,1]),linear_velocity_terminal_error=float(value[-1,2]),
                angular_velocity_terminal_error=float(value[-1,3])))
        print(f"native independent worlds {len(rows)}/{len(episodes)}; physical failures {sum(r['failed'] for r in rows)}",flush=True)
    groups={}
    for mode in ("reach","tracking"):
        group=[r for r in rows if r["mode"]==mode];successes=[r for r in group if r["success"]]
        groups[mode]=dict(cases=len(group),successes=len(successes),failures=sum(r["failed"] for r in group),
            mean_position_m=float(np.mean([r["position_mean_m"] for r in group])) if group else None,
            mean_orientation_rad=float(np.mean([r["orientation_mean_rad"] for r in group])) if group else None,
            successful_terminal_position_mean_m=float(np.mean([r["position_terminal_m"] for r in successes])) if successes else None,
            successful_terminal_orientation_mean_rad=float(np.mean([r["orientation_terminal_rad"] for r in successes])) if successes else None)
    report=dict(groups=groups,rows=rows,actor_updates=learner.actor_updates,
        limits=dict(position_m=.03,orientation_rad=.15,linear_speed_error=.08,angular_speed_error=.2,
                    reach_final_hold=config["hold_steps"],tracking_fraction=.95),
        scope="Fixed-base simulation; independently stepped native worlds; sampled goals only",
        hardware_qualified=False,metric_backend="native MuJoCo TCP pose/Jacobians",evaluation_batch_size=batch_size)
    if output:atomic_json(output,report)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint",required=True);p.add_argument("--references",required=True,nargs="+")
    p.add_argument("--output",required=True);p.add_argument("--batch-size",type=int,default=16)
    p.add_argument("--check-sequential",type=int,default=0);args=p.parse_args()
    torch.set_num_threads(2);ck=torch.load(args.checkpoint,map_location="cpu",weights_only=False)
    cfg=dict(ck["config"],device="cpu",compile_imagined_step=False)
    sim=Simulator(cfg["asset_dir"],1,dt=cfg["dt"],timestep=cfg["timestep"])
    task=Task(sim.model,sim.spec,cfg["dt"],cfg["joint_target_speed"],0)
    learner=Learner(task,cfg);learner.actor.load_state_dict(ck["actor"]);learner.actor_updates=ck["actor_updates"]
    episodes=[]
    for directory in args.references:episodes.extend(load(directory))
    if args.check_sequential:
        episodes=episodes[:args.check_sequential]
        baseline=sequential_evaluate(learner,episodes,cfg["asset_dir"],cfg)
    report=evaluate(learner,episodes,cfg["asset_dir"],cfg,batch_size=args.batch_size)
    if args.check_sequential:
        for a,b in zip(baseline["rows"],report["rows"]):
            for key in ("success","failed","requested_frames","executed_frames"):
                assert a[key]==b[key],(a["case"],key,a[key],b[key])
            for key in ("final_hold_steps","maximum_hold_steps"):
                assert abs(a[key]-b[key])<=2,(a["case"],key,a[key],b[key])
            for key in ("position_mean_m","orientation_mean_rad","position_terminal_m","orientation_terminal_rad"):
                assert abs(a[key]-b[key])<2e-4,(a["case"],key,a[key],b[key])
        report["sequential_equivalence_cases"]=len(episodes)
        report["batch_equivalence_tolerances"]=dict(task_success_and_failure="identical",pose_summary_absolute=2e-4,hold_frame_count=2,
            note="CPU matrix multiplication changes rounding with batch size; near-threshold frame counts can differ slightly")
    report.update(checkpoint=str(Path(args.checkpoint).resolve()),checkpoint_sha256=sha256(args.checkpoint))
    atomic_json(args.output,report)


if __name__=="__main__":main()
