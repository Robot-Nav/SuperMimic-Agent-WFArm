"""Physical evaluations with fixed fixtures, separate reach/trajectory metrics."""
import argparse
from pathlib import Path
import numpy as np
import torch

from .data import load
from .io import atomic_json, sha256
from .robot import Simulator
from .task import Task
from .learning import Learner
from .native_metrics import tcp_error_norms


@torch.no_grad()
def evaluate(learner, episodes, asset_dir, config, output=None):
    sim = Simulator(asset_dir, 1, dt=config["dt"], timestep=config["timestep"])
    task = learner.task
    device = task.low.device
    rows = []
    for index, episode in enumerate(episodes):
        sim.reset()
        previous = torch.zeros((1,6), device=device)
        hold, maximum_hold, failed = 0, 0, False
        pe, re, ve, we, successes = [], [], [], [], []
        requested = len(episode["actions"])
        for goal_row in episode["goals"]:
            goal = torch.as_tensor(goal_row[None], device=device)
            state = torch.as_tensor(sim.state(), device=device)
            action = learner.inference_action(state, previous, goal)
            state = torch.as_tensor(sim.step(action.cpu().numpy()), device=device)
            previous = torch.as_tensor(sim.previous, device=device, dtype=torch.float32)
            ep,er,ev,ew=tcp_error_norms(sim,goal_row[None])[0]
            pe.append(float(ep)); re.append(float(er));ve.append(float(ev));we.append(float(ew))
            success = bool(ep<.03 and er<.15 and ev<.08 and ew<.2) and not sim.failed[0]
            successes.append(success); hold = hold+1 if success else 0
            maximum_hold = max(maximum_hold, hold)
            if sim.failed[0]:
                failed = True; break
        tracking = str(episode["mode"]) == "tracking"
        # Unexecuted frames after failure stay in the denominator.
        eligible = max(1, requested-50)
        fraction = sum(successes[50:])/eligible
        qualified = (not failed and (fraction>=.95 if tracking else hold>=config["hold_steps"]))
        rows.append(dict(case=index, mode=str(episode["mode"]), success=qualified,
            failed=failed, requested_frames=requested, executed_frames=len(pe),
            final_hold_steps=hold, maximum_hold_steps=maximum_hold,
            tracking_fraction_after_50_frames=fraction, position_mean_m=float(np.mean(pe)),
            position_p95_m=float(np.quantile(pe,.95)), orientation_mean_rad=float(np.mean(re)),
            orientation_p95_rad=float(np.quantile(re,.95)),
            linear_velocity_error_mean=float(np.mean(ve)), angular_velocity_error_mean=float(np.mean(we)),
            position_terminal_m=pe[-1],orientation_terminal_rad=re[-1],
            linear_velocity_terminal_error=ve[-1],angular_velocity_terminal_error=we[-1]))
        if (index+1)%8==0:
            print(f"native evaluation {index+1}/{len(episodes)} cases; physical failures {sum(r['failed'] for r in rows)}",flush=True)
    groups = {}
    for mode in ("reach", "tracking"):
        group = [r for r in rows if r["mode"]==mode]
        groups[mode] = dict(cases=len(group), successes=sum(r["success"] for r in group),
            failures=sum(r["failed"] for r in group),
            mean_position_m=float(np.mean([r["position_mean_m"] for r in group])) if group else None,
            mean_orientation_rad=float(np.mean([r["orientation_mean_rad"] for r in group])) if group else None,
            successful_terminal_position_mean_m=float(np.mean([r["position_terminal_m"] for r in group if r["success"]])) if any(r["success"] for r in group) else None,
            successful_terminal_orientation_mean_rad=float(np.mean([r["orientation_terminal_rad"] for r in group if r["success"]])) if any(r["success"] for r in group) else None)
    report = dict(groups=groups, rows=rows, actor_updates=learner.actor_updates,
        limits=dict(position_m=.03,orientation_rad=.15,linear_speed_error=.08,angular_speed_error=.2,
                    reach_final_hold=config["hold_steps"], tracking_fraction=.95),
        scope="fixed-base simulation; sampled physically screened goals, not full workspace certification",
        hardware_qualified=False,metric_backend="native MuJoCo TCP pose/Jacobians; FK-equivalence checked")
    if output:
        atomic_json(output,report)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint",required=True);p.add_argument("--references",required=True,nargs="+")
    p.add_argument("--output",required=True);p.add_argument("--batch-size",type=int,default=16);args=p.parse_args()
    ck=torch.load(args.checkpoint,map_location="cpu",weights_only=False)
    config=dict(ck["config"],device="cpu",compile_imagined_step=False)
    config["evaluation_batch_size"]=args.batch_size
    torch.set_num_threads(2)
    sim=Simulator(config["asset_dir"],1,dt=config["dt"],timestep=config["timestep"])
    task=Task(sim.model,sim.spec,config["dt"],config["joint_target_speed"],config["clearance_points"])
    learner=Learner(task,config);learner.actor.load_state_dict(ck["actor"])
    learner.actor_updates=ck["actor_updates"]
    references=[]
    for directory in args.references:references.extend(load(directory))
    if config.get("actor_architecture")=="structured_taskspace_supermimic":
        from .hierarchy import evaluate as hierarchical_evaluate
        report=hierarchical_evaluate(learner,references,config["asset_dir"],config)
    else:report=evaluate(learner,references,config["asset_dir"],config)
    report["reference_directories"]=args.references
    report.update(checkpoint=str(Path(args.checkpoint).resolve()),checkpoint_sha256=sha256(args.checkpoint))
    report.update(actor_architecture=config.get("actor_architecture","mlp"),
                  evaluation_runtime="CPU eager policy; identical native physical model and criteria")
    atomic_json(args.output,report)


if __name__=="__main__":
    main()
