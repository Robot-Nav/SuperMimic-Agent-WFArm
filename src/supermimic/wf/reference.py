"""Procedural whole-body references, executed and screened in real MuJoCo.

References are actuator keyframes, not teleported floating-base poses. A nominal
rollout defines an achievable TCP goal; a separate perturbed rollout must reach
and hold that SAME goal before a clip is admitted. These are synthetic examples,
not demonstrations of an optimal expert or coverage of arbitrary Cartesian goals.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .evaluate import config_from_dict, physical_failure
from .simulator import BatchedSimulator
from .task import WholeBodyTask


def minimum_jerk(u):
    u = np.clip(u, 0., 1.)
    return u**3 * (10. + u * (-15. + 6. * u))


def interpolate_keyframes(times, values, query):
    """C2 interpolation with zero velocity/acceleration at each keyframe."""
    times, values = np.asarray(times, float), np.asarray(values, float)
    if (times.ndim != 1 or len(times) < 2 or values.shape[0] != len(times)
            or not np.isfinite(times).all() or not np.isfinite(values).all()
            or np.any(np.diff(times) <= 0)):
        raise ValueError("keyframes require finite values and strictly increasing times")
    query = np.asarray(query, float)
    i = np.clip(np.searchsorted(times, query, side="right")-1, 0, len(times)-2)
    f = minimum_jerk((query-times[i])/(times[i+1]-times[i]))
    return values[i] + f[..., None]*(values[i+1]-values[i])


def primitive(spec, seed, family, move_seconds=4., hold_seconds=4.):
    """Hand-designed motion families with seeded, bounded keyframe variations."""
    rng = np.random.default_rng(seed)
    final = spec.q0.copy()
    final[16:] += rng.uniform(-1., 1., 6)*np.array([.28, .22, .26, .32, .26, .35])
    sides = np.array([1., -1., 1., -1.])
    front = np.array([1., 1., -1., -1.])
    extent = rng.uniform(.04, .10)*rng.choice([-1., 1.])
    if family in ("height", "pitch", "roll", "drive", "turn"):
        shape = (np.ones(4) if family == "height" else front if family == "pitch"
                 else np.array([1., -1., 1., -1.]) if family == "roll" else rng.uniform(-.4,.4,4))
        final[np.arange(4)*3+1] += sides*extent*shape
        final[np.arange(4)*3+2] -= 2*sides*extent*shape
    nonwheel = np.r_[0:12,16:22]
    final[nonwheel] = np.clip(final[nonwheel], spec.low[nonwheel]+.015, spec.high[nonwheel]-.015)
    v = rng.uniform(.035,.12)*rng.choice([-1.,1.]) if family in ("drive","turn") else 0.
    yaw = rng.uniform(.04,.12)*rng.choice([-1.,1.]) if family == "turn" else 0.
    # Source wheel axes have opposite left/right signs. No lateral base motion
    # is prescribed. Turning is skid steering and is screened for actual slip.
    wheel = np.array([-1.,1.,-1.,1.])*(v + np.array([-1.,1.,-1.,1.])*.20*yaw)/spec.wheel_radius
    total = move_seconds+hold_seconds
    return {"family":family,"seed":int(seed),"duration_s":total,
            "position_times_s":[0.,move_seconds,total],
            "position_joint_order":[spec.joint_names[i] for i in nonwheel],
            "position_values_rad":[spec.q0[nonwheel].tolist(),final[nonwheel].tolist(),final[nonwheel].tolist()],
            "wheel_times_s":[0.,.2*move_seconds,.8*move_seconds,move_seconds,total],
            "wheel_values_rad_s":[np.zeros(4).tolist(),wheel.tolist(),wheel.tolist(),np.zeros(4).tolist(),np.zeros(4).tolist()]}


def commands_from_keyframes(keyframes, spec, dt):
    nonwheel = np.r_[0:12,16:22]
    if keyframes["position_joint_order"] != [spec.joint_names[i] for i in nonwheel]:
        raise ValueError("position keyframe joint order mismatch")
    steps = round(keyframes["duration_s"]/dt)
    if steps < 1 or not np.isclose(steps*dt,keyframes["duration_s"]):
        raise ValueError("reference duration must be an integer number of steps")
    t = (np.arange(steps)+1)*dt
    position = interpolate_keyframes(keyframes["position_times_s"],keyframes["position_values_rad"],t)
    wheel = interpolate_keyframes(keyframes["wheel_times_s"],keyframes["wheel_values_rad_s"],t)
    if position.shape != (steps,18) or wheel.shape != (steps,4):
        raise ValueError("expected 18 position and four wheel velocity keyframes")
    action = np.zeros((steps,22))
    action[:,nonwheel] = (position-spec.q0[nonwheel])/spec.action_scale[nonwheel]
    action[:,12:16] = wheel/spec.action_scale[12:16]
    if np.abs(action).max()>1.+1e-8:
        raise ValueError("keyframes exceed normalized actuator authority")
    return action


@torch.no_grad()
def execute_reference(sim, task, commands, *, settle_steps=100, joint_noise=0., seed=0,
                      final_action=None, base_target=None):
    """Every transition below is produced by actuator-driven mj_step calls."""
    sim.reset()
    if joint_noise:
        state=sim.read_state()
        ids=np.r_[0:12,16:22]
        state[:,7+ids] += np.random.default_rng(seed).normal(0,joint_noise,(1,18))
        sim.write_state(state)
    sim.settle(steps=settle_steps)
    initial=sim.snapshot()
    states=[sim.read_state()[0].copy()]
    actions=[]; safe=[]; contacts=[]; slip=[]; torque=[]; failures=[]
    previous=np.zeros(22)
    for command in commands:
        if final_action is not None:
            # Offline reference-generation teacher ONLY. A bounded servo ramp
            # points toward the final keyframe from the actual command history.
            # It provides corrective labels instead of purely clock-based labels.
            desired=np.asarray(final_action).copy()
            if base_target is not None:
                state=sim.read_state()[0]
                def yaw(q):
                    w,x,y,z=q
                    return np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z))
                heading=yaw(state[3:7]);target_heading=yaw(base_target[3:7])
                delta=base_target[:2]-state[:2]
                dx=np.cos(heading)*delta[0]+np.sin(heading)*delta[1]
                dy=-np.sin(heading)*delta[0]+np.cos(heading)*delta[1]
                angle=np.arctan2(np.sin(target_heading-heading),np.cos(target_heading-heading))
                velocity=np.clip(.8*dx,-.15,.15)
                angular=np.clip(.8*angle+.6*dy*np.sign(velocity),-.2,.2)
                signs=np.array([-1.,1.,-1.,1.])
                desired[12:16]=signs*(velocity+signs*.20*angular)/sim.spec.wheel_radius/sim.spec.action_scale[12:16]
            command=previous+np.clip(desired-previous,-.005,.005)
        next_state=sim.step(command[None])[0].copy()
        previous=sim.applied_action[0].copy()
        metrics=sim.metrics()
        actions.append(sim.applied_action[0].copy())
        states.append(next_state)
        safe.append(bool(metrics["safe"][0]))
        contacts.append(metrics["wheel_contact"][0])
        slip.append(metrics["slip"][0])
        torque.append(metrics["torque_peak"][0])
        failures.append(bool(physical_failure(task,torch.as_tensor(next_state[None],dtype=torch.float32),metrics)[0]))
        if failures[-1]:
            break
    states=np.asarray(states)
    return {"states":states,"actions":np.asarray(actions),"safe":np.asarray(safe),
            "contacts":np.asarray(contacts),"slip":np.asarray(slip),"torque_peak":np.asarray(torque),
            "failed":np.asarray(failures),"xi":sim.xi[0].copy(),
            "initial_mj_state":initial["integration_state"][0],"initial_state_flag":initial["state_flag"]}


@torch.no_grad()
def screen_reference(clip, task, goal, *, hold_steps=50, max_mean_slip=.10, min_contact_fraction=.90):
    states=torch.as_tensor(clip["states"][1:],dtype=torch.float32)
    goal=torch.as_tensor(goal,dtype=torch.float32).expand(len(states),-1)
    if task.cfg.base_command_conditioning:
        initial=torch.as_tensor(clip['states'][:1],dtype=torch.float32).expand(len(states),-1)
        goal=task.transport_goal(initial,states,goal)
    successes=task.success(states,goal).numpy() & clip["safe"] & ~clip["failed"]
    contact=clip["contacts"]
    slip=float(clip["slip"].sum()/max(1,contact.sum()))
    accepted=(len(states)>=hold_steps and not clip["failed"].any()
              and bool(successes[-hold_steps:].all()) and slip<=max_mean_slip
              and float(contact.mean())>=min_contact_fraction)
    return {"accepted":bool(accepted),"final_hold_steps":int(hold_steps),
            "final_hold_pass":bool(len(states)>=hold_steps and successes[-hold_steps:].all()),
            "physical_failure":bool(clip["failed"].any()),"mean_contact_slip_m_s":slip,
            "wheel_contact_fraction":float(contact.mean()),"steps":len(states)}


def generate(config, output, count=32, seed=41000, families=None, attempts=None,feedback=False):
    from .train import model_manifest
    torch.set_num_threads(2)
    output=Path(output)
    if (output/"manifest.json").exists():
        raise FileExistsError("reference library already exists; select a new output directory")
    output.mkdir(parents=True,exist_ok=True)
    c=config.sim
    sim=BatchedSimulator(num_envs=1,seed=seed,dt=c.dt,timestep=c.timestep,asset_dir=config.asset_dir,
        friction=1.,leg_action_scale=c.leg_action_scale,arm_action_scale=c.arm_action_scale,wheel_action_scale=c.wheel_action_scale)
    task=WholeBodyTask(sim.model,sim.spec,reward_cfg=config.reward)
    manifest={"format":"wf_physical_references_v1","provenance":model_manifest(sim.spec),
              "dt":c.dt,"seed":seed,"goal_origin":"nominal physical rollout; independently replay-validated",
              "goal_order":"world xyz, quaternion wxyz","state_order":"WF canonical state57",
              "action_order":"executed normalized mixed position/velocity action22",
              "config":asdict(config),"clips":[],"rejected":[]}
    manifest["reference_teacher"]="offline final-keyframe servo with base-pose correction" if feedback else "open-loop interpolated keyframes"
    families=families or ["arm","height","pitch","roll","drive","turn"]
    for attempt in range(attempts or count*5):
        key=primitive(sim.spec,seed+attempt,families[attempt%len(families)])
        command=commands_from_keyframes(key,sim.spec,c.dt)
        sim.friction_range=(1.,1.)
        nominal=execute_reference(sim,task,command,settle_steps=c.reset_settle_steps)
        if len(nominal["actions"]) != len(command):
            manifest["rejected"].append({"seed":seed+attempt,"reason":"nominal physical failure"})
            continue
        # Calibrate the target ONCE. It is never changed to match validation.
        pose=task.tcp_pose(torch.as_tensor(nominal["states"][-50:],dtype=torch.float32))
        goal=pose[-1].numpy().copy()
        goal[:3]=pose[:,:3].mean(0).numpy()
        nominal_check=screen_reference(nominal,task,goal)
        sim.friction_range=(.85,1.15)
        clip=execute_reference(sim,task,command,settle_steps=c.reset_settle_steps,joint_noise=.005,seed=seed+attempt+500000,
            final_action=command[-1] if feedback else None,
            base_target=nominal["states"][-1,:7] if feedback and key["family"] in ("drive","turn") else None)
        check=screen_reference(clip,task,goal)
        if len(clip["actions"]) != len(command) or not check["accepted"] or not nominal_check["accepted"]:
            manifest["rejected"].append({"seed":seed+attempt,"reason":"independent physical replay failed","screen":check})
            continue
        i=len(manifest["clips"])
        name=f"clip_{i:04d}"
        np.savez_compressed(output/(name+".npz"),**clip,goal=goal,planned_actions=command)
        (output/(name+"_keyframes.json")).write_text(json.dumps(key,indent=2)+"\n")
        record={"file":name+".npz","keyframes":name+"_keyframes.json","family":key["family"],"seed":key["seed"],
                "split":"validation" if i%5==4 else "train","screen":check,
                "sha256":hashlib.sha256((output/(name+".npz")).read_bytes()).hexdigest()}
        manifest["clips"].append(record)
        print(json.dumps({"accepted_clips":len(manifest["clips"]),"attempt":attempt+1,"family":key["family"],**check}),flush=True)
        if len(manifest["clips"])>=count:
            break
    sim.close()
    manifest["requested_clips"]=count
    manifest["complete"]=len(manifest["clips"])==count
    (output/"manifest.json").write_text(json.dumps(manifest,indent=2,allow_nan=False)+"\n")
    if not manifest["complete"] or len(manifest["clips"])<5:
        raise RuntimeError("insufficient physically validated references; inspect rejection records")
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",default="configs/wf_pose_stage1_repaired.json")
    p.add_argument("--output",required=True)
    p.add_argument("--count",type=int,default=32)
    p.add_argument("--seed",type=int,default=41000)
    p.add_argument("--feedback",action="store_true",help="offline corrective reference teacher; never used by the deployed Actor")
    args=p.parse_args()
    if args.count<5: p.error("at least five clips are needed for separate training/validation")
    result=generate(config_from_dict(json.loads(Path(args.config).read_text())),args.output,args.count,args.seed,feedback=args.feedback)
    print(json.dumps({"clips":len(result["clips"]),"rejected":len(result["rejected"]),"manifest":str(Path(args.output)/"manifest.json")}))


if __name__=="__main__":
    main()
