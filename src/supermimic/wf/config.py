"""Experiment settings. All actuator values are simulation assumptions, not calibration."""

from dataclasses import dataclass, field
import math

from .task import WholeBodyRewardCfg


@dataclass
class SimConfig:
    num_envs: int = 16
    dt: float = 0.02
    timestep: float = 0.005
    episode_steps: int = 600
    friction_min: float = 0.6
    friction_max: float = 1.2
    exploration_std: float = 0.12
    delta_exploration_std: float = .0003
    exploration_delta_limit: float | None = None
    reset_joint_noise: float = 0.015
    leg_action_scale: float = 0.35
    arm_action_scale: float | None = 0.5  # None enables the complete source joint range
    wheel_action_scale: float = 25.0
    reset_settle_steps: int = 0  # Legacy checkpoints sampled before settling.
    switch_goals_after_hold: int = 0  # Positive: change the goal in place after this many safe held steps.
    arm_initial_position: tuple[float,...] | None = None  # None: legacy working pose; six zeros: URDF folded zero.


@dataclass
class ModelConfig:
    hidden: tuple[int, ...] = (512, 512, 512)
    window: int = 8
    batch_size: int = 128
    batches_per_update: int = 4
    lr: float = 2e-4
    grad_clip: float = 1.0
    validation_fraction: float = 0.2
    actor_start_updates: int = 40
    max_validation_error: float = 2.0
    verify_action_response: bool = False
    response_check_interval: int = 64
    response_min_cosine: float = .9
    response_max_relative_error: float = .5
    excitation_file: str | None = None
    excitation_fraction: float = .5


@dataclass
class PolicyConfig:
    hidden: tuple[int, ...] = (256, 256, 128)
    critic_hidden: tuple[int, ...] = (256, 128)
    latent_dim: int = 32
    horizon: int = 32
    initial_horizon: int = 8
    horizon_ramp_updates: int = 2000
    batch_size: int = 128
    lr: float = 2e-4
    critic_lr: float = 2e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    critic_epochs: int = 4
    grad_clip: float = 1.0
    grad_value_clip: float = 0.1
    obs_noise_std: float = 0.005
    goal_conditioning: str = "encoded"  # "direct" exposes normalized goal errors.
    goal_position_scale: float = 0.1
    goal_relabel_fraction: float = 0.0
    horizon_progress: str = "attempted"  # "accepted" excludes rejected updates.
    goal_skip_lr_scale: float = 1.0
    ignore_previous_action: bool = False
    action_mode: str = "absolute"
    action_delta_limit: float = .01
    architecture: str = 'mlp'
    constraint_observation: bool = False
    gain_bound: float = 0.
    log_damping_init: float = -2.
    feedback_lr_scale: float = 1.  # Matrix damping/stop parameters need their own time scale.
    program_prior_lr_scale: float = .1  # Legacy rate; explicit in continuous-training configs.
    action_lower: tuple[float,...] = ()  # Optional calibrated operating bounds, normalized.
    action_upper: tuple[float,...] = ()
    action_bound_source: str | None = None
    action_bound_sha256: str | None = None
    base_command_conditioning: bool = False
    full_goal_orientation: bool = False
    target_supervision_scale: str = 'absolute'
    feedback_hidden: tuple[int,...] = (256,128)
    feedback_output_scale: float = .05
    feedback_start_phase: float = 0.
    feedback_full_phase: float = 0.
    program_context: bool = False
    increment_feedback: bool = False
    increment_feedback_hidden: tuple[int,...] = (128,128)
    increment_feedback_scale: float = 2.
    increment_feedback_lr_scale: float = 10.


@dataclass
class CurriculumConfig:
    initial_level: int = 0
    max_level: int = 3
    eval_interval: int = 2000
    eval_episodes: int = 16
    promote_success_rate: float = 0.75
    promote_checks: int = 2
    hold_steps: int = 15
    goal_seed_device: str = "policy"  # "cpu" gives identical CPU/CUDA fixtures.
    evaluation_device: str = "policy"


@dataclass
class GuardConfig:
    # New adaptation: candidate actor updates are periodically checked in physics.
    enabled: bool = True
    interval: int = 256
    episodes: int = 8
    score_tolerance: float = 0.01
    allowed_failure_increase: float = 0.0
    min_lr_scale: float = 0.05
    rollback_critic: bool = True
    backtrack_alphas: tuple[float, ...] = ()
    prefer_success: bool = False
    prefer_safety: bool = False
    reference_episodes: int = 0
    reference_anchor_episodes: int = 0


@dataclass
class DemonstrationConfig:
    directory: str | None = None
    pretrain_updates: int = 0
    pretrain_lr: float = 2e-4
    batch_size: int = 256
    model_fraction: float = .5
    start_fraction: float = .5
    bc_weight: float = .1
    tracking_weight: float = .25
    tracking_in_critic: bool = True  # False treats reference tracking as an Actor-only regularizer.
    decay_updates: int = 2000
    observation_noise_std: float = .02
    physical_selection_interval: int = 0
    validation_interval: int = 100
    model_pretrain_updates: int = 0
    hold_weight: float = 0.
    policy_start_steps: int = 0
    task_selection_episodes: int = 0
    query_file: str | None = None
    query_fraction: float = .75
    goal_bank_fraction: float = 0.
    goal_bank_position_noise: float = 0.
    goal_bank_rotation_noise: float = 0.
    evaluate_goal_bank: bool = False
    goal_bank_preserve_command: bool = False
    command_sampling_fraction: float = 0.
    zero_start_sampling_fraction: float = 0.
    broad_anchor_sampling_fraction: float = 0.
    validate_initial_states: bool = False  # Include real recovery/switch initial state and command.


@dataclass
class TrainConfig:
    seed: int = 0
    device: str = "cpu"
    output: str = "logs/wf_goal"
    iterations: int = 100_000
    actor_update_limit: int | None = None  # Optional absolute completed-Actor-update counter.
    warmup_steps: int = 128
    update_every: int = 4
    replay_steps: int = 2048
    log_interval: int = 50
    checkpoint_interval: int = 2000
    keep_actor_checkpoints: bool = False  # Small, immutable snapshots for offline periodic evaluation.
    cpu_threads: int = 4
    compile_fk: bool = False  # Optional numeric kernel optimization; no task/model change.
    compile_clearance: bool = False  # Same sampled geometry and VJP, fewer GPU launches.
    asset_dir: str | None = None
    resume: str | None = None
    warm_start: str | None = None  # Reuse online dynamics/data, initialize a new Actor.
    warm_start_actor: bool = False  # Explicitly retain a self-trained Actor when adapting reward.
    warm_start_actor_path: str | None = None  # Optional separate self-trained Actor; dynamics still use warm_start.
    program_prior: str | None = None  # Learned goal+phase prior, no runtime reference files.
    run_steps: int | None = None  # Bound this invocation without shortening the LR schedule.
    model_only: bool = False  # Invocation mode: collect independent excitation, freeze Actor.
    base_speed_curriculum: tuple[tuple[int,float],...] = ()  # (online step, forward-speed range) boundaries.
    sim: SimConfig = field(default_factory=SimConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    reward: WholeBodyRewardCfg = field(default_factory=WholeBodyRewardCfg)
    guard: GuardConfig = field(default_factory=GuardConfig)
    demonstrations: DemonstrationConfig = field(default_factory=DemonstrationConfig)

    def validate(self):
        if self.actor_update_limit is not None and (isinstance(self.actor_update_limit,bool) or not isinstance(self.actor_update_limit,int) or self.actor_update_limit<1):
            raise ValueError('actor_update_limit must be a positive integer or None')
        if self.model.response_check_interval<1 or not 0<=self.model.response_min_cosine<=1 or self.model.response_max_relative_error<=0:
            raise ValueError("invalid action-response validation parameters")
        d=self.demonstrations
        if not isinstance(d.tracking_in_critic,bool):raise ValueError('tracking_in_critic must be boolean')
        if min(d.pretrain_updates,d.model_pretrain_updates,d.physical_selection_interval,d.policy_start_steps,d.task_selection_episodes)<0 or d.batch_size<1 or d.pretrain_lr<=0 or d.decay_updates<1 or d.validation_interval<1 or d.observation_noise_std<0:
            raise ValueError("invalid reference pretraining parameters")
        if not 0<=d.model_fraction<=1 or not 0<=d.start_fraction<=1 or min(d.bc_weight,d.tracking_weight,d.hold_weight)<0:
            raise ValueError("invalid reference mixture or guidance weights")
        if not 0<=d.command_sampling_fraction<=1:raise ValueError('invalid command-reference sampling fraction')
        if not 0<=d.zero_start_sampling_fraction<=1:raise ValueError('invalid folded-zero reference sampling fraction')
        if not 0<=d.broad_anchor_sampling_fraction<=1:raise ValueError('invalid broad-anchor reference sampling fraction')
        if (d.pretrain_updates or d.model_pretrain_updates) and not d.directory:
            raise ValueError("reference pretraining requires a dataset directory")
        if not 0<=d.query_fraction<=1 or (d.query_file and not d.directory):
            raise ValueError('corrective queries require a parent reference library and a valid mixture fraction')
        if not 0<=d.goal_bank_fraction<=1 or min(self.guard.reference_episodes,self.guard.reference_anchor_episodes,d.goal_bank_position_noise,d.goal_bank_rotation_noise)<0:
            raise ValueError('invalid feasible-goal mixture or reference validation count')
        if self.resume and self.warm_start:
            raise ValueError("select exact --resume or dynamics-only --warm-start")
        if self.warm_start_actor and not (self.warm_start or self.resume):
            raise ValueError("warm_start_actor requires a checkpoint source")
        if self.warm_start_actor_path and not self.warm_start_actor:
            raise ValueError('separate Actor source requires warm_start_actor')
        if self.run_steps is not None and self.run_steps < 1:
            raise ValueError("run_steps must be positive")
        if self.sim.num_envs < 2:
            raise ValueError("num_envs >= 2 is required for disjoint training/validation streams")
        if not 0 < self.model.validation_fraction < 1:
            raise ValueError("validation_fraction must be in (0, 1)")
        if not 1 <= self.policy.initial_horizon <= self.policy.horizon:
            raise ValueError("require 1 <= initial_horizon <= horizon")
        if not 1 <= self.model.window < self.replay_steps:
            raise ValueError("model window must be smaller than replay capacity")
        if self.sim.episode_steps <= self.model.window:
            raise ValueError("episode_steps must exceed the model fitting window")
        if not 0 <= self.curriculum.initial_level <= self.curriculum.max_level <= 3:
            raise ValueError("curriculum levels must be between 0 and 3")
        for name in ("iterations", "warmup_steps", "update_every", "log_interval", "checkpoint_interval"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.sim.friction_min <= 0 or self.sim.friction_max < self.sim.friction_min:
            raise ValueError("invalid friction range")
        if self.model.batch_size < 1 or self.policy.batch_size < 1:
            raise ValueError("batch sizes must be positive")
        if self.model.batches_per_update < 1 or self.policy.critic_epochs < 1:
            raise ValueError("update counts must be positive")
        if self.curriculum.eval_interval < 1 or self.curriculum.hold_steps < 1:
            raise ValueError("evaluation and hold intervals must be positive")
        if self.curriculum.hold_steps > self.sim.episode_steps:
            raise ValueError("hold_steps must not exceed episode_steps")
        if self.cpu_threads < 1 or self.model.actor_start_updates < 0:
            raise ValueError("cpu_threads must be positive and actor_start_updates nonnegative")
        if not isinstance(self.compile_fk,bool):raise ValueError('compile_fk must be boolean')
        if not isinstance(self.compile_clearance,bool):raise ValueError('compile_clearance must be boolean')
        if self.sim.reset_settle_steps < 0:
            raise ValueError("reset_settle_steps must be nonnegative")
        if self.sim.arm_initial_position is not None and (len(self.sim.arm_initial_position)!=6 or not all(math.isfinite(x) for x in self.sim.arm_initial_position)):
            raise ValueError('arm_initial_position must contain six finite joint angles')
        if self.sim.switch_goals_after_hold<0:
            raise ValueError('goal switch hold must be nonnegative')
        if self.policy.goal_conditioning not in ("encoded", "direct") or self.policy.goal_position_scale <= 0:
            raise ValueError("invalid goal conditioning/position scale")
        if self.policy.action_mode not in ("absolute","delta","target") or self.policy.action_delta_limit<=0:
            raise ValueError("invalid policy action mode/rate limit")
        if self.policy.action_mode=='delta' and self.policy.ignore_previous_action:
            raise ValueError("delta policy requires the previous executed command")
        if self.policy.target_supervision_scale not in ('absolute','delta'):
            raise ValueError('unknown target supervision scale')
        if self.policy.architecture not in ('mlp','matrix','residual_target','goal_program'):
            raise ValueError('unknown Actor architecture')
        if self.policy.architecture=='residual_target' and self.policy.action_mode!='target':
            raise ValueError('residual_target architecture requires target action semantics')
        if self.policy.constraint_observation!=self.reward.constraint_observation or self.policy.gain_bound<0:
            raise ValueError('Actor/task geometry observations must match; gain bound must be nonnegative')
        if self.policy.base_command_conditioning != self.reward.base_command_conditioning:
            raise ValueError('Actor and task base-command conditioning must match')
        if self.policy.full_goal_orientation != self.reward.full_goal_orientation:
            raise ValueError('Actor and task full goal orientation features must match')
        if (self.policy.architecture=='goal_program')!=self.reward.goal_program:
            raise ValueError('goal-program Actor and task phase semantics must match')
        if self.policy.program_context!=self.reward.program_context or (self.policy.program_context and not self.reward.goal_program):
            raise ValueError('Actor and task initial program context must match')
        if self.reward.goal_program and (not self.reward.full_goal_orientation or not self.reward.base_command_conditioning or self.policy.action_mode!='target' or self.reward.program_steps<1):
            raise ValueError('goal program requires direct full-pose command features and target action semantics')
        if self.reward.goal_program and self.demonstrations.query_file:
            raise ValueError('goal-program pretraining uses context-bearing physical clips, not unphased queries')
        if self.program_prior and self.policy.architecture!='goal_program':raise ValueError('program prior requires goal-program Actor')
        if self.policy.feedback_output_scale<=0:raise ValueError('feedback output scale must be positive')
        if self.policy.increment_feedback and self.policy.architecture!='goal_program':
            raise ValueError('increment feedback requires the goal-program Actor')
        if min(self.policy.increment_feedback_scale,self.policy.increment_feedback_lr_scale)<=0:
            raise ValueError('increment feedback scales must be positive')
        if not 0<=self.policy.feedback_start_phase<=self.policy.feedback_full_phase<=1:
            raise ValueError('invalid goal-program feedback phase gate')
        if self.reward.base_command_resample_steps < 1:
            raise ValueError('base command resampling interval must be positive')
        if not 0<=self.reward.base_command_fraction<=1 or len(self.reward.base_command_ranges)!=3:
            raise ValueError('invalid base-command mixture or ranges')
        if min(self.reward.base_command_ranges)<0 or min(self.reward.base_linear_scale,self.reward.base_angular_scale,
            self.reward.success_base_linear,self.reward.success_base_angular)<=0:
            raise ValueError('invalid base-command scales or tolerances')
        if not 0<=self.reward.base_straight_fraction<=1 or not 0<=self.reward.base_endpoint_fraction<=1:
            raise ValueError('invalid straight/endpoint command sampling fraction')
        if (len(self.reward.base_linear_scales)!=len(self.reward.base_linear_weights)
            or any(x<=0 or not math.isfinite(x) for x in self.reward.base_linear_scales)
            or any(x<=0 or not math.isfinite(x) for x in self.reward.base_linear_weights)):
            raise ValueError('invalid multiscale base tracking reward')
        schedule=self.base_speed_curriculum
        if schedule and (schedule[0][0]!=0 or any(len(x)!=2 or int(x[0])!=x[0] or x[0]<0
            or not math.isfinite(x[1]) or not 0<x[1]<=self.reward.base_command_ranges[0] for x in schedule)
            or any(a[0]>=b[0] or a[1]>b[1] for a,b in zip(schedule,schedule[1:]))):
            raise ValueError('invalid base speed curriculum')
        if self.policy.architecture=='matrix' and (self.policy.action_mode!='delta' or self.policy.goal_conditioning!='direct' or self.reward.goal_orientation_frame!='tcp'):
            raise ValueError('matrix Actor requires delta actions and TCP-relative direct pose errors')
        if self.policy.goal_skip_lr_scale <= 0:
            raise ValueError("goal_skip_lr_scale must be positive")
        if not math.isfinite(self.policy.feedback_lr_scale) or self.policy.feedback_lr_scale<=0:
            raise ValueError('feedback_lr_scale must be finite and positive')
        if not math.isfinite(self.policy.program_prior_lr_scale) or self.policy.program_prior_lr_scale<=0:
            raise ValueError('program_prior_lr_scale must be finite and positive')
        lo,hi=self.policy.action_lower,self.policy.action_upper
        if bool(lo)!=bool(hi) or (lo and (len(lo)!=22 or len(hi)!=22)):
            raise ValueError('operating action bounds must both have 22 values')
        if lo and any(not math.isfinite(a) or not math.isfinite(b) or not -1<=a<b<=1 for a,b in zip(lo,hi)):
            raise ValueError('invalid operating action bounds')
        if not 0 <= self.policy.goal_relabel_fraction <= 1:
            raise ValueError("goal_relabel_fraction must be in [0,1]")
        if self.policy.horizon_progress not in ("attempted", "accepted"):
            raise ValueError("horizon_progress must be attempted or accepted")
        if self.curriculum.goal_seed_device not in ("policy", "cpu"):
            raise ValueError("goal_seed_device must be policy or cpu")
        if self.curriculum.evaluation_device not in ("policy","cpu"):
            raise ValueError("evaluation_device must be policy or cpu")
        alphas = self.guard.backtrack_alphas
        if any(not 0 < a < 1 for a in alphas) or any(a <= b for a,b in zip(alphas, alphas[1:])):
            raise ValueError("backtracking fractions must strictly decrease in (0,1)")
        if not 0 <= self.sim.exploration_std <= 1 or not 0<=self.sim.delta_exploration_std<=1 or self.sim.reset_joint_noise < 0:
            raise ValueError("invalid exploration/reset noise")
        if self.sim.exploration_delta_limit is not None and (not math.isfinite(self.sim.exploration_delta_limit) or self.sim.exploration_delta_limit<=0):
            raise ValueError('invalid exploration action rate limit')
        if self.sim.leg_action_scale <= 0 or self.sim.wheel_action_scale <= 0:
            raise ValueError("action scales must be positive")
        if self.sim.arm_action_scale is not None and self.sim.arm_action_scale <= 0:
            raise ValueError("arm action scale must be positive or None")
        if self.guard.interval < 1 or self.guard.episodes < 1 or self.guard.score_tolerance < 0:
            raise ValueError("invalid physical policy guard configuration")
        if not 0 < self.guard.min_lr_scale <= 1 or not 0 <= self.guard.allowed_failure_increase <= 1:
            raise ValueError("invalid guard learning-rate scale or failure allowance")
        r = self.reward
        if r.goal_follow_frame not in ("body", "heading"):
            raise ValueError("goal following frame must be body or heading")
        if (not all(math.isfinite(x) for x in (r.geometry_collision_weight,r.geometry_ground_weight,
                    r.geometry_clearance_margin,r.geometry_clearance_scale))
                or min(r.geometry_collision_weight,r.geometry_ground_weight,r.geometry_clearance_margin)<0
                or r.geometry_clearance_scale<=0 or not isinstance(r.geometry_surface_points,int)
                or isinstance(r.geometry_surface_points,bool) or not 16<=r.geometry_surface_points<=512):
            raise ValueError("invalid geometric clearance learning aid")
        if not 0<=r.joint_soft_margin<.5 or min(r.joint_soft_limit_weight,r.self_collision_weight,r.settling_weight)<0:
            raise ValueError("invalid soft constraint/settling reward parameters")
        if min(r.success_position,r.success_orientation,r.success_linear_speed,r.success_angular_speed)<=0:
            raise ValueError("success tolerances must be positive")
        if r.goal_orientation_frame not in ("tcp","base"):
            raise ValueError("goal orientation frame must be tcp or base")
        if len(r.position_scales) != len(r.position_weights) or not r.position_scales or min(r.position_scales) <= 0:
            raise ValueError("position reward scales must be positive and match their weights")
        if len(r.goal_translation_radii) != 4 or len(r.goal_rotation_radii) != 4:
            raise ValueError("goal curricula require four levels")
        if min(r.goal_translation_radii) < 0 or min(r.goal_rotation_radii) < 0:
            raise ValueError("goal radii must be nonnegative")
        if min(r.orientation_scale, r.arrival_speed_scale, r.contact_temperature,
               r.penetration_scale, r.slip_speed_scale, r.collision_scale, r.termination_temperature) <= 0:
            raise ValueError("reward normalization scales must be positive")
        if len(r.orientation_scales) != len(r.orientation_weights):
            raise ValueError("orientation scales and weights must match")
        if r.orientation_scales and min(r.orientation_scales) <= 0:
            raise ValueError("orientation scales must be positive")
        if r.min_base_height >= r.max_base_height:
            raise ValueError("base height bounds must be ordered")
        if r.tilt_barrier_scale<=0 or r.tilt_barrier_weight<0:
            raise ValueError('invalid normalized tilt barrier')
        if (r.wrist_safe_lower is None)!=(r.wrist_safe_upper is None):
            raise ValueError('both wrist operating limits must be specified')
        if r.wrist_safe_lower is not None and r.wrist_safe_lower>=r.wrist_safe_upper:
            raise ValueError('invalid wrist operating range')


def smoke_config() -> TrainConfig:
    """Small real-physics integration check; does not measure task competence."""
    cfg = TrainConfig(iterations=64, warmup_steps=16, update_every=4, replay_steps=128,
                      log_interval=16, checkpoint_interval=64)
    cfg.sim.num_envs = 4
    cfg.sim.episode_steps = 80
    cfg.model.hidden = (64, 64)
    cfg.model.window = 4
    cfg.model.batch_size = 8
    cfg.model.batches_per_update = 1
    cfg.model.actor_start_updates = 2
    cfg.model.max_validation_error = 1e6
    cfg.policy.hidden = (64, 64)
    cfg.policy.critic_hidden = (64, 32)
    cfg.policy.batch_size = 4
    cfg.policy.initial_horizon = cfg.policy.horizon = 4
    cfg.policy.critic_epochs = 1
    cfg.curriculum.eval_interval = 10_000
    cfg.guard.enabled = False
    return cfg
