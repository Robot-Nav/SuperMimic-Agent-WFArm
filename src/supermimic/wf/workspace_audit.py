"""Stratified full-arm geometric targets and honest real-physics policy audit.

Targets are selected from collision-free static configurations BEFORE testing
the Actor. Geometric feasibility is not path/dynamic feasibility.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
import torch

from .evaluate import config_from_dict, evaluate_policy
from .networks import Actor
from .simulator import BatchedSimulator
from .task import WholeBodyTask


def sector_masks(points):
    x, y, z = points.T
    return dict(top=(z > .40) & (abs(x) < .35) & (abs(y) < .30),
                left=(y > .30) & (z > .05) & (z < .50),
                right=(y < -.30) & (z > .05) & (z < .50),
                low_side=(z < -.05) & (abs(y) > .22),
                underneath=(z < -.05) & (abs(x) < .23) & (abs(y) < .18),
                rear=(x < -.30) & (abs(y) < .30) & (z > .05) & (z < .50))


def audit(checkpoint, output, samples=30000, per_sector=4, seed=650000):
    from .train import model_manifest
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = config_from_dict(ck["config"])
    sim = BatchedSimulator(num_envs=1, friction=1., dt=cfg.sim.dt, timestep=cfg.sim.timestep,
        leg_action_scale=cfg.sim.leg_action_scale, arm_action_scale=cfg.sim.arm_action_scale,
        wheel_action_scale=cfg.sim.wheel_action_scale)
    try:
        assert model_manifest(sim.spec)["physics_hash"] == ck["manifest"]["physics_hash"]
        sim.settle(steps=cfg.sim.reset_settle_steps)
        spec, model, data = sim.spec, sim.model, sim.data[0]
        initial = sim.read_state()[0].copy()
        origin = data.xpos[spec.base_body].copy()
        rotation = data.xmat[spec.base_body].reshape(3, 3).copy()
        low = np.maximum(spec.low, spec.q0+spec.action_scale*np.asarray(cfg.policy.action_lower))[16:]
        high = np.minimum(spec.high, spec.q0+spec.action_scale*np.asarray(cfg.policy.action_upper))[16:]
        rng = np.random.default_rng(seed)
        angles = rng.uniform(low, high, (samples, 6))
        poses, accepted = [], []
        wheels = set(spec.wheel_geom_ids.tolist())
        for i, q in enumerate(angles):
            data.qpos[spec.qpos_indices[16:]] = q
            mujoco.mj_forward(model, data)
            bad = False
            for contact in data.contact[:data.ncon]:
                if contact.dist > 0:
                    continue
                pair = {int(contact.geom1), int(contact.geom2)}
                if spec.ground_geom in pair and (pair-{spec.ground_geom}).issubset(wheels):
                    continue
                bad = True
                break
            if bad or data.site_xpos[spec.tcp_site, 2] < .1:
                continue
            quaternion = np.zeros(4)
            mujoco.mju_mat2Quat(quaternion, data.site_xmat[spec.tcp_site])
            poses.append(np.r_[data.site_xpos[spec.tcp_site].copy(), quaternion])
            accepted.append(i)
        poses = np.asarray(poses)
        body_points = (poses[:, :3]-origin)@rotation
        masks = sector_masks(body_points)
        records, goals, used = [], [], set()
        for name, mask in masks.items():
            pool = np.flatnonzero(mask)
            pool = rng.permutation(pool)
            chosen = [int(i) for i in pool if int(i) not in used][:per_sector]
            for row in chosen:
                used.add(row)
                records.append(dict(sector=name, candidate_joint_angles=angles[accepted[row]].tolist(),
                    goal=poses[row].tolist(), tcp_position_initial_base_frame=body_points[row].tolist()))
                goals.append(poses[row].tolist())
        protocol = dict(format="wf_sector_audit_v1", seed=seed, samples=samples,
            geometrically_accepted=len(poses), available_by_sector={k:int(v.sum()) for k,v in masks.items()},
            requested_per_sector=per_sector, initial_state=initial.tolist(), records=records,
            checkpoint_sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
            selection="random collision-screened joint configurations; tested Actor not consulted",
            scope="sectors relative to INITIAL chassis; geometric poses, not teacher-verified paths")
        (output/"protocol.json").write_text(json.dumps(protocol, indent=2)+"\n")
        (output/"goals.json").write_text(json.dumps(goals, indent=2)+"\n")
        print(json.dumps(dict(available=protocol["available_by_sector"], selected=len(goals))), flush=True)
        actor = Actor(cfg.policy)
        actor.load_state_dict(ck["actor"])
        task = WholeBodyTask(model, spec, reward_cfg=cfg.reward)
        report = evaluate_policy(actor, task, model, spec, episodes=len(goals), goals=goals,
            steps=cfg.sim.episode_steps, hold_steps=cfg.curriculum.hold_steps, dt=cfg.sim.dt,
            settle_steps=cfg.sim.reset_settle_steps, level=2, seed=seed+1, goal_seed_device="cpu")
        summary = {}
        for name in masks:
            rows = [i for i,r in enumerate(records) if r["sector"] == name]
            measures = [report["records"][i] for i in rows]
            summary[name] = dict(cases=len(rows), final_held_successes=sum(r["final_held_success"] for r in measures),
                physical_failures=sum(r["failed"] for r in measures),
                mean_final_position_error_m=float(np.mean([r["final_position_error_m"] for r in measures])) if rows else None)
        report.update(protocol="protocol.json", sector_summary=summary,
            scope="geometric-pose audit; misses do not prove dynamic reachability or impossibility")
        (output/"report.json").write_text(json.dumps(report, indent=2)+"\n")
        print(json.dumps(summary), flush=True)
        return summary
    finally:
        sim.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--samples", type=int, default=30000)
    p.add_argument("--per-sector", type=int, default=4)
    p.add_argument("--seed", type=int, default=650000)
    a = p.parse_args()
    audit(a.checkpoint, a.output, a.samples, a.per_sector, a.seed)


if __name__ == "__main__":
    main()
