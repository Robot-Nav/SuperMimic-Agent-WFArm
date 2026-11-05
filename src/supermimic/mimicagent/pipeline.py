"""Prompt/plan/code -> FK checks -> diagnostic feedback, at most three attempts.

Run programs supplied by a coding agent with --programs. Each request, source,
error and result is preserved. Generation here is file based: the caller is the
coding agent; this module does not silently substitute templates for an LLM.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import mujoco
import numpy as np

from supermimic.wf.robot import build_model


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def robot_context(spec, *, duration=6., dt=.02):
    if not np.isfinite(duration) or not np.isfinite(dt) or dt <= 0:
        raise ValueError("duration and dt must be finite and dt positive")
    steps = round(duration / dt)
    if not 2 <= steps <= 6000 or not np.isclose(steps * dt, duration):
        raise ValueError("duration must contain 2..6000 whole control intervals")
    q = spec.q0.copy(); q[16:] = 0.
    base = spec.model.qpos0[spec.base_qpos_index:spec.base_qpos_index + 7]
    return dict(format="wf_mimicagent_context_v1", dt=dt, duration_s=duration,
        frames=steps + 1, joint_order=list(spec.joint_names),
        initial_joint_position=q.tolist(), initial_base_pose=base.tolist(),
        joint_lower=[float(v) if np.isfinite(v) else None for v in spec.low],
        joint_upper=[float(v) if np.isfinite(v) else None for v in spec.high],
        joint_velocity_limit=spec.velocity.tolist(), wheel_radius_m=spec.wheel_radius,
        wheel_indices=spec.wheel_indices.tolist(), quaternion_order="wxyz",
        coordinates="base pose in world; joint positions in radians",
        interface="generate(context) -> base_position[frames,3], base_quaternion[frames,4], joint_position[frames,22]",
        initial_condition="standing legs and folded zero arm; fixed gripper",
        scope="kinematic references, never actual dynamics transition labels")


def validate_kinematic(arrays, spec, context, *, aerial=False):
    """Paper-inspired generic checks, plus WF-Arm whole-robot collision checks."""
    n = context["frames"]
    shapes = {"base_position": (n, 3), "base_quaternion": (n, 4), "joint_position": (n, 22)}
    if set(arrays) != set(shapes) or any(np.shape(arrays[k]) != s for k, s in shapes.items()):
        return dict(accepted=False, checks={"shape": False}, diagnostic="return the exact context frame count and field shapes")
    if not all(np.isfinite(v).all() for v in arrays.values()):
        return dict(accepted=False, checks={"finite": False}, diagnostic="remove non-finite trajectory values")
    pos, quat, q = (arrays[k] for k in shapes)
    norms = np.linalg.norm(quat, axis=1)
    quaternion_valid = bool(np.all(np.abs(norms - 1.) <= 1e-5))
    lower, upper = spec.low, spec.high
    violations = int(np.count_nonzero((q < lower - 1e-7) | (q > upper + 1e-7)))
    velocity = np.diff(q, axis=0) / context["dt"]
    checks = dict(finite=True, quaternion=quaternion_valid,
        base_height=bool(np.all((pos[:, 2] >= .1) & (pos[:, 2] <= 2 * spec.default_height))),
        joint_limits=violations == 0,
        joint_velocity=bool(np.all(np.abs(velocity) <= spec.velocity + 1e-6)),
        initial_pose=bool(np.allclose(q[0], context["initial_joint_position"], atol=1e-6)
                          and np.allclose(pos[0], context["initial_base_pose"][:3], atol=1e-6)
                          and abs(np.dot(quat[0], context["initial_base_pose"][3:])) >= 1 - 1e-6))
    if not quaternion_valid:
        return dict(accepted=False, checks=checks, diagnostic="normalize every wxyz quaternion")
    data = mujoco.MjData(spec.model)
    bottoms, forbidden, tcp = [], 0, []
    wheels = set(spec.wheel_geom_ids.tolist())
    for p, r, joints in zip(pos, quat, q):
        data.qpos[spec.base_qpos_index:spec.base_qpos_index + 7] = np.r_[p, r]
        data.qpos[spec.qpos_indices] = joints
        mujoco.mj_forward(spec.model, data)  # FK/contact geometry only: no mj_step.
        bottoms.append(data.geom_xpos[spec.wheel_geom_ids, 2] - spec.wheel_radius)
        tcp_quat = np.empty(4)
        mujoco.mju_mat2Quat(tcp_quat, data.site_xmat[spec.tcp_site])
        tcp.append(np.r_[data.site_xpos[spec.tcp_site].copy(), tcp_quat])
        for c in data.contact[:data.ncon]:
            pair = {int(c.geom1), int(c.geom2)}
            wheel_floor = spec.ground_geom in pair and bool((pair - {spec.ground_geom}) <= wheels)
            forbidden += int(c.dist < -.001 and not wheel_floor)
    bottoms = np.asarray(bottoms)
    support_fraction = float(np.any(np.abs(bottoms) <= .02, axis=1).mean())
    checks.update(ground_penetration=bool(bottoms.min() >= -.01),
                  support_time=bool(aerial or support_fraction >= .5), collision=forbidden == 0)
    return dict(accepted=all(checks.values()), checks=checks,
        joint_violation_count=violations, forbidden_contact_count=forbidden,
        minimum_wheel_height_m=float(bottoms.min()), supported_frame_fraction=support_fraction,
        requested_final_tcp_pose=tcp[-1].tolist(),
        diagnostic="revise only the failed structural checks; do not change the task or claim dynamics feasibility",
        dynamics_verified=False)


def physical_screen(arrays, output, *, dt=.02, frictions=(.6, 1.2), terminal_feedback=False):
    """Replay motor targets. Never impose the proposal's floating-base poses."""
    if len(frictions) != 2 or not all(np.isfinite(f) and f > 0 for f in frictions) or frictions[0] == frictions[1]:
        raise ValueError("physical screening needs two distinct positive friction coefficients")
    import torch
    from supermimic.wf.reference import execute_reference, screen_reference
    from supermimic.wf.simulator import BatchedSimulator
    from supermimic.wf.task import WholeBodyTask
    from supermimic.wf.train import model_manifest

    torch.set_num_threads(2)
    sim = BatchedSimulator(num_envs=1, dt=dt, arm_action_scale=None,
                           arm_initial_position=[0.] * 6, friction=frictions[0])
    try:
        task = WholeBodyTask(sim.model, sim.spec)
        spec = sim.spec
        q = arrays["joint_position"]
        nonwheel = np.r_[0:12, 16:22]
        commands = np.zeros((len(q) - 1, 22))
        commands[:, nonwheel] = (q[1:, nonwheel] - spec.q0[nonwheel]) / spec.action_scale[nonwheel]
        commands[:, 12:16] = np.diff(q[:, 12:16], axis=0) / dt / spec.action_scale[12:16]
        if np.any(np.abs(commands) > 1 + 1e-7):
            return dict(accepted=False, reason="proposal exceeds normalized actuator authority", training_eligible=False)
        # Goal is frozen from the requested geometry BEFORE any physical rollout.
        d = mujoco.MjData(sim.model)
        d.qpos[spec.base_qpos_index:spec.base_qpos_index + 7] = np.r_[arrays["base_position"][-1], arrays["base_quaternion"][-1]]
        d.qpos[spec.qpos_indices] = q[-1]
        mujoco.mj_forward(sim.model, d)
        quat = np.empty(4); mujoco.mju_mat2Quat(quat, d.site_xmat[spec.tcp_site])
        goal = np.r_[d.site_xpos[spec.tcp_site].copy(), quat]
        reports, clips = [], []
        for friction in frictions:
            sim.friction_range = (friction, friction)
            def corrected_commands():
                from supermimic.wf.feedback_reference import corrective_command
                previous = sim.initial_action[0].copy()
                for i, command in enumerate(commands):
                    if i >= int(.65 * len(commands)):
                        command = corrective_command(sim, goal, previous)
                    yield command
                    previous = sim.applied_action[0].copy()

            sequence = corrected_commands() if terminal_feedback else commands
            clip = execute_reference(sim, task, sequence, settle_steps=100)
            clip["initial_action"] = sim.initial_action[0].copy()
            clip["goal"] = goal
            report = screen_reference(clip, task, goal)
            state = torch.as_tensor(clip["states"][-1:], dtype=torch.float32)
            pose = task.tcp_pose(state)[0].numpy()
            report["final_position_error_m"] = float(np.linalg.norm(pose[:3] - goal[:3]))
            report["final_orientation_error_rad"] = float(2 * np.arccos(np.clip(abs(np.dot(pose[3:], goal[3:])), 0, 1)))
            reports.append(dict(friction=friction, **report))
            clips.append(clip)
        accepted = all(r["accepted"] for r in reports)
        output = Path(output); output.mkdir(parents=True, exist_ok=True)
        # Even rejected actual transitions are kept for diagnosis, never admitted
        # by a manifest marked as successful demonstrations.
        for i, clip in enumerate(clips):
            np.savez_compressed(output / f"rollout_{i}.npz", **clip)
        return dict(accepted=accepted, requested_goal=goal.tolist(), checks=reports,
            training_eligible=accepted, provenance=model_manifest(spec), dt=dt,
            rollout_sha256=[hashlib.sha256((output / f"rollout_{i}.npz").read_bytes()).hexdigest() for i in range(len(clips))],
            base_pose_teleported=False, source="actual actuator-driven MuJoCo steps")
    finally:
        sim.close()


def run(prompt, plan, programs, output, *, duration=6., dt=.02, aerial=False,
        physical=False, timeout=30., terminal_feedback=False):
    if not prompt.strip() or not plan.strip():
        raise ValueError("a skill prompt and explicit phase/contact plan are required")
    if not 1 <= len(programs) <= 3:
        raise ValueError("supply one to three coding-agent revisions")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    _, spec = build_model(arm_action_scale=None)
    context = robot_context(spec, duration=duration, dt=dt)
    request = dict(prompt=prompt, motion_plan=plan, robot=context, max_attempts=3,
        source_paper="https://arxiv.org/abs/2609.24145", provider="caller supplied coding-agent programs",
        instructions="Write generate(context). Preserve intent, phase/contact plan and folded-zero start. Return only the specified arrays. Use diagnostic feedback for revisions.")
    write_json(output / "request.json", request)
    write_json(output / "context.json", context)
    attempts, feedback = [], None
    package_root = str(Path(__file__).resolve().parents[2])
    for i, program in enumerate(programs):
        directory = output / f"attempt_{i + 1}"
        directory.mkdir()
        source = directory / "program.py"
        shutil.copyfile(program, source)
        write_json(directory / "request.json", dict(**request, previous_feedback=feedback))
        env = os.environ.copy()
        env["PYTHONPATH"] = package_root + os.pathsep + env.get("PYTHONPATH", "")
        try:
            result = subprocess.run([sys.executable, "-m", "supermimic.mimicagent.worker", str(source),
                str(output / "context.json"), str(directory / "proposal.npz")],
                cwd=directory, env=env, text=True, capture_output=True, timeout=timeout)
            (directory / "execution.log").write_text(result.stdout + result.stderr)
            if result.returncode:
                feedback = dict(accepted=False, checks={"execution": False}, diagnostic=result.stderr[-4000:])
            else:
                with np.load(directory / "proposal.npz", allow_pickle=False) as d:
                    arrays = {k: d[k].copy() for k in d.files}
                feedback = validate_kinematic(arrays, spec, context, aerial=aerial)
                feedback["kinematic_accepted"] = feedback["accepted"]
                if feedback["accepted"] and physical:
                    feedback["physical"] = physical_screen(arrays, directory / "physical", dt=dt,
                                                            terminal_feedback=terminal_feedback)
                    feedback["physical"]["terminal_feedback"] = terminal_feedback
                    feedback["accepted"] = feedback["physical"]["accepted"]
        except subprocess.TimeoutExpired:
            feedback = dict(accepted=False, checks={"execution_timeout": False}, diagnostic="program exceeded execution time budget")
        except (ValueError, KeyError, OSError) as error:
            feedback = dict(accepted=False, checks={"execution_output": False}, diagnostic=str(error))
        write_json(directory / "diagnosis.json", feedback)
        attempts.append(dict(attempt=i + 1, program_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), **feedback))
        if feedback["accepted"]:
            break
    result = dict(format="wf_kinematic_proposals_v1", prompt=prompt,
        accepted=feedback["accepted"], physical_screen_requested=physical,
        training_eligible=bool(feedback.get("physical", {}).get("training_eligible", False)),
        attempts=attempts, policy_improvement_measured=False,
        scope="independent WF-Arm frontend reproduction; not original quadruped benchmark reproduction")
    write_json(output / "report.json", result)
    return result


def export_library(runs, output):
    """Bridge only successful independently screened real rollouts to SuperMimic."""
    runs = [Path(p) for p in runs]
    if len(runs) < 2:
        raise ValueError("at least two distinct requests are needed for a clip-disjoint split")
    loaded = [(p, json.loads((p / "report.json").read_text())) for p in runs]
    if len({p.resolve() for p in runs}) != len(runs):
        raise ValueError("duplicate runs cannot form independent train/validation clips")
    if not all(r["training_eligible"] for _, r in loaded):
        raise ValueError("kinematic-only or physically rejected proposals cannot enter demonstration training")
    provenance = loaded[0][1]["attempts"][-1]["physical"]["provenance"]
    dt = loaded[0][1]["attempts"][-1]["physical"]["dt"]
    records, lengths, goals = [], set(), set()
    for p, r in loaded:
        last = r["attempts"][-1]
        if last["physical"]["provenance"]["physics_hash"] != provenance["physics_hash"] or last["physical"]["dt"] != dt:
            raise ValueError("mixed physics/control periods")
        clip = p / f"attempt_{last['attempt']}" / "physical/rollout_1.npz"
        if hashlib.sha256(clip.read_bytes()).hexdigest() != last["physical"]["rollout_sha256"][1]:
            raise ValueError("screened physical rollout bytes changed")
        with np.load(clip, allow_pickle=False) as a:
            lengths.add(len(a["actions"]))
            goals.add(tuple(np.round(a["goal"], 7)))
        records.append((p, r, clip))
    if len(lengths) != 1 or len(goals) != len(runs):
        raise ValueError("need equal clip lengths and distinct request goals for a disjoint split")
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    clips = []
    for i, (p, r, source) in enumerate(records):
        name = f"clip_{i:04d}.npz"; shutil.copyfile(source, output / name)
        keyframes = f"clip_{i:04d}_request.json"
        shutil.copyfile(p / "request.json", output / keyframes)
        clips.append(dict(file=name, split="validation" if i == len(records)-1 else "train",
            keyframes=keyframes,
            seed=0, level=0,
            family="mimicagent_coding_agent", initial_pose_family="folded_zero", source_run=str(p.resolve()),
            screen=r["attempts"][-1]["physical"]["checks"][-1],
            sha256=hashlib.sha256((output / name).read_bytes()).hexdigest()))
    manifest = dict(format="wf_physical_references_v1", complete=True, dt=dt, seed=0, level=0,
        requested_clips=len(clips), rejected=[],
        provenance=provenance, clips=clips, method="MimicAgent-style generated references + independent physical screening",
        scope="frontend integration smoke data, not benchmark or generalization evidence")
    write_json(output / "manifest.json", manifest)
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--prompt", required=True)
    r.add_argument("--plan", type=Path, required=True)
    r.add_argument("--programs", nargs="+", type=Path, required=True)
    r.add_argument("--output", type=Path, required=True)
    r.add_argument("--duration", type=float, default=6.)
    r.add_argument("--aerial", action="store_true")
    r.add_argument("--physical", action="store_true")
    r.add_argument("--terminal-feedback", action="store_true", help="WF adaptation: use existing offline corrective teacher during final hold")
    e = sub.add_parser("export")
    e.add_argument("--runs", nargs="+", type=Path, required=True)
    e.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.command == "export":
        result = export_library(a.runs, a.output)
        print(json.dumps(dict(clips=len(result["clips"]), output=str(a.output))))
    else:
        if a.terminal_feedback and not a.physical:
            p.error("--terminal-feedback requires --physical")
        result = run(a.prompt, a.plan.read_text(), a.programs, a.output,
                     duration=a.duration, aerial=a.aerial, physical=a.physical,
                     terminal_feedback=a.terminal_feedback)
        print(json.dumps({k: v for k, v in result.items() if k != "attempts"}, ensure_ascii=False))
        if not result["accepted"]:
            raise SystemExit(2)


if __name__ == "__main__":
    main()
