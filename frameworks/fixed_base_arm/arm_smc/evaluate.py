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
            action = task.command(learner.actor, state, previous, goal)
            state = torch.as_tensor(sim.step(action.cpu().numpy()), device=device)
            previous = torch.as_tensor(sim.previous, device=device, dtype=torch.float32)
            ep, er, ev, ew = task.errors(state, goal)
            pe.append(float(ep.norm())); re.append(float(er.norm()))
            ve.append(float(ev.norm())); we.append(float(ew.norm()))
            success = bool(task.success(state, goal)[0]) and not sim.failed[0]
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
            linear_velocity_error_mean=float(np.mean(ve)), angular_velocity_error_mean=float(np.mean(we))))
    groups = {}
    for mode in ("reach", "tracking"):
        group = [r for r in rows if r["mode"]==mode]
        groups[mode] = dict(cases=len(group), successes=sum(r["success"] for r in group),
            failures=sum(r["failed"] for r in group),
            mean_position_m=float(np.mean([r["position_mean_m"] for r in group])) if group else None,
            mean_orientation_rad=float(np.mean([r["orientation_mean_rad"] for r in group])) if group else None)
    report = dict(groups=groups, rows=rows, actor_updates=learner.actor_updates,
        limits=dict(position_m=.03,orientation_rad=.15,linear_speed_error=.08,angular_speed_error=.2,
                    reach_final_hold=config["hold_steps"], tracking_fraction=.95),
        scope="fixed-base simulation; sampled physically screened goals, not full workspace certification",
        hardware_qualified=False)
    if output:
        atomic_json(output,report)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint",required=True);p.add_argument("--references",required=True)
    p.add_argument("--output",required=True);args=p.parse_args()
    ck=torch.load(args.checkpoint,map_location="cpu",weights_only=False)
    config=ck["config"]
    sim=Simulator(config["asset_dir"],1,dt=config["dt"],timestep=config["timestep"])
    task=Task(sim.model,sim.spec,config["dt"],config["joint_target_speed"],config["clearance_points"])
    learner=Learner(task,config);learner.actor.load_state_dict(ck["actor"])
    learner.actor_updates=ck["actor_updates"]
    report=evaluate(learner,load(args.references),config["asset_dir"],config)
    report.update(checkpoint=str(Path(args.checkpoint).resolve()),checkpoint_sha256=sha256(args.checkpoint))
    atomic_json(args.output,report)


if __name__=="__main__":
    main()
