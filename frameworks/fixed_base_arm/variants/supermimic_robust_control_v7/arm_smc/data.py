"""Physical reference generation and episode-safe replay.

References are generated with smooth joint-space excitation and real limited
PD dynamics. Only TCP pose/twist is exposed to the actor. No inverse kinematics
or reference joint lookup is used during inference.
"""
from pathlib import Path
import numpy as np
import torch
import mujoco

from .robot import Simulator
from .task import Task
from .io import atomic_json, sha256


def smooth_segment(start, end, speed, dt, hold=80):
    # max derivative of the quintic blend is 1.875.
    steps = max(100, int(np.ceil(1.875*np.max(np.abs(end-start))/(speed*dt))))
    u = np.arange(1, steps+1)/steps
    blend = 10*u**3-15*u**4+6*u**5
    path = start[None]+blend[:, None]*(end-start)[None]
    return np.concatenate((path, np.tile(end, (hold, 1))))


def sample_target(sim, rng, start, broad):
    s = sim.spec
    data = mujoco.MjData(sim.model)
    for _ in range(300):
        if broad:
            target = rng.uniform(s.low+.04, s.high-.04)
        else:
            center = np.array([0., 1.35, -1.75, 0., .4, 0.]) if rng.random()<.6 else start
            target = np.clip(center+rng.uniform(-.55, .55, 6), s.low+.04, s.high-.04)
        valid = True
        for fraction in np.linspace(0, 1, 25):
            data.qpos[s.qadr] = start+fraction*(target-start)
            mujoco.mj_forward(sim.model, data)
            if sim.collision(data):
                valid = False; break
        if valid:
            return target
    raise RuntimeError("could not find a collision-free six-joint excitation segment")


def generate(asset_dir, config, count, seed, directory, split):
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory/"manifest.json"
    if manifest_path.exists():
        import json
        manifest = json.loads(manifest_path.read_text())
        if manifest["seed"] != seed or manifest["requested"] != count or manifest["split"] != split:
            raise ValueError("existing reference set has different provenance")
        for clip in manifest["clips"]:
            if sha256(directory/clip["file"]) != clip["sha256"]:
                raise RuntimeError("reference data digest changed")
        return load(directory)
    rng = np.random.default_rng(seed)
    sim = Simulator(asset_dir, num_envs=1, dt=config["dt"], timestep=config["timestep"], seed=seed)
    task = Task(sim.model, sim.spec, dt=config["dt"], speed=config["joint_target_speed"], clearance_points=0)
    clips, rows = [], []
    for attempt in range(count*10):
        if len(clips) == count:
            break
        tracking = len(clips)%3 == 2
        sim.reset(); start = np.zeros(6)
        try:
            target = sample_target(sim, rng, start, broad=len(clips)%2 == 0)
            commands = smooth_segment(start, target, config["reference_speed"], sim.dt)
            if tracking:
                second = sample_target(sim, rng, target, broad=True)
                commands = np.concatenate((commands,
                    smooth_segment(target, second, config["reference_speed"], sim.dt)))
        except RuntimeError:
            continue
        states, actions, previous, failures = [sim.state()[0]], [], [], []
        for target in commands:
            previous.append(sim.previous[0].copy())
            state = sim.step(target[None])
            states.append(state[0]); actions.append(sim.previous[0].copy()); failures.append(sim.failed[0])
            if sim.failed[0]:
                break
        if any(failures):
            continue
        states = np.asarray(states, dtype=np.float32)
        with torch.no_grad():
            bodies = task.fk(torch.from_numpy(states))
            poses = bodies["tcp_pose"].numpy()
            if tracking:
                indices = np.minimum(np.arange(len(actions))+3, len(actions))
                goals = np.concatenate((poses[indices], bodies["tcp_v"].numpy()[indices],
                                        bodies["tcp_w"].numpy()[indices]), -1)
                goals[-30:, 7:] = 0
            else:
                goals = np.tile(np.r_[poses[-1], np.zeros(6)], (len(actions), 1))
        clip = dict(states=states, actions=np.asarray(actions, np.float32),
            previous=np.asarray(previous, np.float32), goals=goals.astype(np.float32),
            failed=np.asarray(failures, np.float32), mode=np.array("tracking" if tracking else "reach"))
        filename = f"clip_{len(clips):04d}.npz"
        np.savez_compressed(directory/filename, **clip)
        rows.append(dict(file=filename, sha256=sha256(directory/filename), transitions=len(actions),
                         mode=str(clip["mode"])))
        clips.append(clip)
        if len(clips)%8 == 0:
            print(f"reference {split}: {len(clips)}/{count}", flush=True)
    if len(clips) != count:
        raise RuntimeError(f"physical reference generation only accepted {len(clips)}/{count}")
    atomic_json(manifest_path, dict(split=split, seed=seed, requested=count, clips=rows,
        physical_screening="full executed trajectory; torque/velocity limits; no arm contacts",
        reference_only=True, student_policy_qualified=False))
    return clips


def load(directory):
    import json
    root = Path(directory)
    manifest = json.loads((root/"manifest.json").read_text())
    return [dict(np.load(root/row["file"])) for row in manifest["clips"]]


def load_additional_training(directory):
    import json
    root = Path(directory)
    manifest = json.loads((root/"manifest.json").read_text())
    if manifest["split"] != "additional_train":
        raise ValueError("only the dedicated additional_train split may augment training")
    for row in manifest["clips"]:
        if sha256(root/row["file"]) != row["sha256"]:
            raise RuntimeError("additional training reference digest changed")
    episodes=load(root)
    if manifest.get("deployment_uses_feedback_expert") is False and "feedback" in manifest.get("teacher",""):
        for episode in episodes:episode["is_recovery"]=np.array(True)
    return episodes


def local_tracking_curriculum(episodes,task):
    """TRAIN-only crops and generated reference context for low-level control.

    Large static-goal jumps are handled by the independent planner at runtime.
    Preserve varied local recovery starts and all continuous trajectories.
    Half of episodes use planned reference context; half use the retained
    policy's posture proposal, matching the two runtime modes.
    """
    rng=np.random.default_rng(78204191);result=[];metadata=[]
    with torch.no_grad():
        for index,source in enumerate(episodes):
            episode=dict(source);n=len(source["actions"]);start=0
            recovery=bool(source.get("is_recovery",False))
            if str(source["mode"])=="reach" and not recovery:
                state=torch.as_tensor(source["states"][:-1],device=task.low.device)
                goal=torch.as_tensor(source["goals"],device=task.low.device)
                ep,er,_,_=task.errors(state,goal)
                valid=((ep.norm(dim=-1)<.12)&(er.norm(dim=-1)<.6)).cpu().numpy()
                candidates=np.flatnonzero(valid & (np.arange(n)>=max(0,n-200)))
                start=int(candidates[0]) if len(candidates) else max(0,n-80)
                for key in ("states","actions","previous","goals","failed"):
                    episode[key]=source[key][start:].copy()
                episode["is_recovery"]=np.array(True)
            count=len(episode["actions"])
            if str(source["mode"])=="tracking":
                selected=np.minimum(np.arange(count)+3,count)
                reference=episode["states"][selected].copy()
                reference[-30:,6:]=0
            else:
                reference=np.tile(episode["states"][-1],(count,1));reference[:,6:]=0
            planned=float(rng.random()<.5)
            episode["goals"]=np.concatenate((episode["goals"],reference,
                                              np.full((count,1),planned,np.float32)),axis=-1).astype(np.float32)
            result.append(episode)
            metadata.append(dict(source_train_episode=index,source_frames=n,crop_start=start,
                                 frames=count,mode=str(source["mode"]),planned_reference_context=bool(planned)))
    return result,metadata


class Replay:
    def __init__(self, episodes, device="cpu", seed=0, capacity=256,recovery_prefix_frames=0,recovery_prefix_fraction=0.):
        self.episodes = list(episodes)
        self.device, self.capacity = device, capacity
        self.rng = np.random.default_rng(seed)
        self.recovery_prefix_frames=recovery_prefix_frames
        self.recovery_prefix_fraction=recovery_prefix_fraction

    def add(self, episode):
        if len(episode["actions"]) < 8:
            return
        self.episodes.append(episode)
        if len(self.episodes)>self.capacity:
            self.episodes.pop(0)

    def _tensor(self, array):
        return torch.as_tensor(np.asarray(array), device=self.device, dtype=torch.float32)

    def windows(self, batch, window):
        eligible = [e for e in self.episodes if len(e["actions"])>=window]
        if not eligible:
            raise RuntimeError("no complete replay windows")
        samples = []
        for _ in range(batch):
            e = eligible[self.rng.integers(len(eligible))]
            index = self.rng.integers(len(e["actions"])-window+1)
            samples.append((e, index))
        return dict(states=self._tensor(np.stack([e["states"][j:j+window+1] for e,j in samples], 1)),
            actions=self._tensor(np.stack([e["actions"][j:j+window] for e,j in samples], 1)),
            failed=self._tensor(np.stack([e["failed"][j:j+window] for e,j in samples], 1)))

    def starts(self, batch, horizon=0, tracking_fraction=None):
        samples = []
        pools=None
        if tracking_fraction is not None:
            tracking_pool=[e for e in self.episodes if str(e.get("mode",""))=="tracking"]
            other_pool=[e for e in self.episodes if str(e.get("mode",""))!="tracking"]
            pools={True:tracking_pool or self.episodes,False:other_pool or self.episodes}
        for _ in range(batch):
            pool = self.episodes
            if tracking_fraction is not None:
                tracking = self.rng.random() < tracking_fraction
                pool=pools[tracking]
            e = pool[self.rng.integers(len(pool))]
            upper=len(e["actions"])
            if (self.recovery_prefix_frames and self.recovery_prefix_fraction and bool(e.get("is_recovery",False))
                and self.rng.random()<self.recovery_prefix_fraction):
                upper=min(upper,self.recovery_prefix_frames)
            index = self.rng.integers(upper)
            samples.append((e, index))
        result = {key:self._tensor([e[source][j] for e,j in samples]) for key,source in
            (("state","states"),("previous","previous"),("goal","goals"),("action","actions"))}
        if horizon:
            future = [e["goals"][np.minimum(np.arange(j,j+horizon+1),len(e["goals"])-1)] for e,j in samples]
            result["future_goals"] = self._tensor(np.stack(future,1))
        return result
