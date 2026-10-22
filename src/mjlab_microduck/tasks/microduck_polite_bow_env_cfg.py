"""Microduck PoliteBow task — a commanded bow from standing: fold forward, look down, hold,
come back up.

One policy, both directions, driven by a flag in the twist slot, exactly as the sit-stand is:
    cmd = [bow_flag, 0, 0]    bow_flag ∈ {0 = STAND, 1 = BOW}
"Stand" is the all-zero command — the same deployment idle as every other policy. The flag
flips mid-episode with a dwell time of a few seconds, so each episode trains folds, held bows,
rises and standing rest, plus "hold what you're already doing".

Why this shape and not a timed one-shot: a feed-forward policy has no clock, only its body
state, so a bow trained as a fixed-length episode could go down and come up but never *stay*
down for a chosen time. With the flag, the hold is whoever writes the flag — on the robot,
`policy add bow <repo> --hold <s> --command 1,0,0`, which writes 1 for `hold` seconds and then
the manifest's all-zero unwind until it stands (the flamingo's arrangement). The bow's
duration is the owner's knob, and the network only has to know two postures and the way
between them.

Why a trained bow at all: the daemon already bows by running the ground pick partway back,
which is a 30° fold through a crouch, because that is what the pick does; and the gait's own
body-pose command tops out near 12°. This one folds from the hips at standing height, to
BOW_PITCH_DEG (between a polite 30° and an apologetic 45°), and takes its head with it.

Reward design — each term and what it is for:
  bow_pitch_track   — THE task: the trunk's forward pitch (from the root quaternion) tracks
                      a target that is 0 standing, BOW_PITCH bowed, and slews between them
                      over BOW_RAMP_S. Gaussian, std 7°. Being ahead of the ramp pays ~0, so
                      slow IS the argmax (the sit-stand's anti-crash lesson).
  bow_pitch_l1      — the transition driver: a constant per-step cost for resting in the
                      wrong posture, so ignoring the flag is net-negative in both directions.
  bow_head_pose     — the head follows: neck/head pitch track a target that blends from HOME
                      to BOW_HEAD_OVERRIDES with the same slewed blend — looking ahead at the
                      start, at the floor just ahead at the bottom, and back up with the body.
  trunk_roll        — no sideways lean: the fold is sagittal.
  feet_grounded     — both feet stay planted, always on: lifting one costs every step, which
                      rules out stepping or hopping into the bow.
  height_stand      — the trunk stays near standing height (loose std): rules out the
                      crouch-and-nod, the cheapest way to tilt a trunk.
  pose_stand_legs   — legs pulled gently toward HOME (loose std): the fold must stay
                      affordable, but the robot comes back to where it started.
  bow_stillness     — rest quietly AT the commanded pitch once the ramp is done, standing or
                      bowed: a held bow is a still bow, not a wobble.
  regularisers      — action rate (ramped by curriculum), body angular velocity, angular
                      momentum, self-collision: velocity parity, so what transfers to the robot
                      keeps transferring.

The base velocity recipe's `upright` term is removed: it would fight the fold.

DR / noise / observations: copied from the kick env (which copies velocity), so the 61D obs
layout and the sim2real recipe are the ones every other policy on the robot was trained with.
The head and body command slots are zero-padded, as the runtime zeroes them for every skill.
"""

import math
from copy import deepcopy

# Symmetry — off, as for every 61D env (see the velocity env's note).
ENABLE_SYMMETRY = False

# ── Domain randomisation (matched to velocity / kick) ─────────────────────────
ENABLE_COM_RANDOMIZATION             = True
ENABLE_HEAD_COM_RANDOMIZATION        = True
ENABLE_KP_RANDOMIZATION              = False
ENABLE_KD_RANDOMIZATION              = False
ENABLE_MASS_INERTIA_RANDOMIZATION    = True
ENABLE_JOINT_FRICTION_RANDOMIZATION  = True
ENABLE_ARMATURE_RANDOMIZATION        = True
ENABLE_VELOCITY_PUSHES               = True
ENABLE_IMU_ORIENTATION_RANDOMIZATION = True
ENABLE_ENCODER_BIAS                  = True

# ── Ranges (matched to velocity / kick) ──────────────────────────────────────
COM_RANDOMIZATION_RANGE             = 0.003
HEAD_COM_RANDOMIZATION_RANGE        = 0.003
MASS_INERTIA_RANDOMIZATION_RANGE    = (0.95, 1.05)
ARMATURE_RANDOMIZATION_RANGE        = (0.9, 1.1)
JOINT_FRICTION_RANDOMIZATION_RANGE  = (0.9, 1.1)
ENCODER_BIAS_RANGE                  = (-0.015, 0.015)
KP_RANDOMIZATION_RANGE              = (0.85, 1.15)
KD_RANDOMIZATION_RANGE              = (0.9, 1.1)
VELOCITY_PUSH_INTERVAL_S            = (3.0, 6.0)
# A bow is quasi-static like the pick: pushes are gentle (the pick's ±0.15 knocked it over
# at ±0.3 even standing straight).
VELOCITY_PUSH_RANGE                 = (-0.15, 0.15)
IMU_ORIENTATION_RANDOMIZATION_ANGLE = 6.0

# ── Task constants ────────────────────────────────────────────────────────────
# The bow: between a polite 30° and an apologetic 45°.
BOW_PITCH_DEG = 37.5
BOW_PITCH = math.radians(BOW_PITCH_DEG)

# Long enough for several flips: a fold, a held bow, a rise, and standing rest.
EPISODE_LENGTH_S = 8.0
# Seconds in each posture before the flag may flip (the command's resampling time).
BOW_DWELL_S = (2.0, 4.0)
# Probability a resample asks for the bow (vs standing).
BOW_PROB = 0.5
# Seconds for the slewed target to travel the whole way — the bow's own pace. 37.5° over
# 1.2 s is about 30°/s, a deliberate bow rather than a nod.
BOW_RAMP_S = 1.2

# Trunk standing height (measured natural equilibrium at HOME — see standup env).
STAND_Z = 0.115

# Where the head goes at the bottom, as ABSOLUTE joint angles (HOME is +0.349 on both):
# the neck a little forward, the head pitched well down — head_pitch positive is down, and
# +1.0 from home is "looking at the floor 20 cm ahead" per the robot's gaze solver. With the
# trunk itself 37° forward, +0.6 on the head puts the gaze at the floor near the feet.
_NECK_PITCH, _HEAD_PITCH = 5, 6
BOW_HEAD_OVERRIDES = {_NECK_PITCH: 0.15, _HEAD_PITCH: 0.95}

_LEG_JOINTS  = [0, 1, 2, 3, 4, 9, 10, 11, 12, 13]
_NECK_JOINTS = [5, 6, 7, 8]

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import (
    CurriculumTermCfg,
    EventTermCfg,
    ObservationTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlModelCfg,
)
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from mjlab_microduck.robot.microduck_constants import MICRODUCK_STANDUP_ROBOT_CFG
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity_env_cfg import HEAD_BODY_NAMES
from mjlab_microduck.tasks.symmetry import PpoWithSymmetryCfg, SYMMETRY_CFG


def make_microduck_polite_bow_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create the Microduck PoliteBow environment configuration."""

    feet_ground_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(
            mode="geom",
            pattern=r"^(left_foot_collision|right_foot_collision)$",
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )

    self_collision_cfg = ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(mode="subtree", pattern="trunk_base", entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern="trunk_base", entity="robot"),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )

    foot_frictions_geom_names = ("left_foot_collision", "right_foot_collision")

    # ── Base config ───────────────────────────────────────────────────────────
    cfg = make_velocity_env_cfg()

    # Full-collision robot (as standup / ground pick / kick): a deep fold brings the head and
    # trunk near the legs, and the walk model's foot-only collisions would let them pass
    # through each other.
    cfg.scene.entities = {"robot": MICRODUCK_STANDUP_ROBOT_CFG}
    cfg.scene.sensors = (feet_ground_cfg, self_collision_cfg)
    cfg.viewer.body_name = "trunk_base"

    cfg.episode_length_s = EPISODE_LENGTH_S

    # ── Actions ───────────────────────────────────────────────────────────────
    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = 1.0

    # ── Rewards: drop walking-specific terms, and the upright that would fight the fold ──
    for name in [
        "track_linear_velocity",
        "track_angular_velocity",
        "air_time",
        "foot_clearance",
        "foot_swing_height",
        "foot_slip",
        "pose",
        "soft_landing",
        "upright",
    ]:
        if name in cfg.rewards:
            del cfg.rewards[name]

    # ── Rewards: the bow ──────────────────────────────────────────────────────
    cfg.rewards["bow_pitch_track"] = RewardTermCfg(
        func=microduck_mdp.bow_pitch_track,
        weight=4.0,
        params={"command_name": "twist", "bow_pitch": BOW_PITCH, "std": math.radians(7.0)},
    )
    cfg.rewards["bow_pitch_l1"] = RewardTermCfg(
        func=microduck_mdp.bow_pitch_l1,
        weight=-2.0,
        params={"command_name": "twist", "bow_pitch": BOW_PITCH},
    )
    cfg.rewards["bow_head_pose"] = RewardTermCfg(
        func=microduck_mdp.posture_pose_match,
        weight=1.5,
        params={
            "command_name": "twist",
            "sit_overrides": BOW_HEAD_OVERRIDES,
            "joint_indices": _NECK_JOINTS,
            "std": 0.3,
        },
    )
    cfg.rewards["trunk_roll"] = RewardTermCfg(
        func=microduck_mdp.trunk_roll_gaussian,
        weight=1.0,
        params={"std": math.radians(6.0)},
    )
    cfg.rewards["feet_grounded"] = RewardTermCfg(
        func=microduck_mdp.feet_grounded_reward,
        weight=2.0,
        params={"sensor_name": feet_ground_cfg.name},
    )
    cfg.rewards["height_stand"] = RewardTermCfg(
        func=microduck_mdp.height_target_gaussian,
        weight=1.0,
        params={
            "std":           0.03,
            "target_height": STAND_Z,
            "asset_cfg":     SceneEntityCfg("robot", body_names=("trunk_base",)),
        },
    )
    cfg.rewards["pose_stand_legs"] = RewardTermCfg(
        func=microduck_mdp.pose_target_match,
        weight=1.0,
        params={"std": 0.5, "joint_indices": _LEG_JOINTS, "target_overrides": None},
    )
    cfg.rewards["bow_stillness"] = RewardTermCfg(
        func=microduck_mdp.bow_stillness,
        weight=1.0,
        params={"command_name": "twist", "bow_pitch": BOW_PITCH},
    )

    # ── Sim2real regularisers — velocity parity ───────────────────────────────
    cfg.rewards["action_rate_l2"].weight = -0.1  # stage-0; curriculum ramps to -1.0
    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("trunk_base",)
    cfg.rewards["body_ang_vel"].weight = -0.05
    cfg.rewards["angular_momentum"].weight = -0.02

    cfg.rewards["self_collisions"] = RewardTermCfg(
        func=mdp.self_collision_cost,
        weight=-1.0,
        params={"sensor_name": self_collision_cfg.name},
    )

    # ── Observations (unified 61D actor layout) ───────────────────────────────
    del cfg.observations["actor"].terms["base_lin_vel"]

    cfg.observations["critic"].terms["base_lin_vel"] = ObservationTermCfg(
        func=mdp.base_lin_vel, scale=1.0,
    )
    del cfg.observations["critic"].terms["foot_height"]
    del cfg.observations["actor"].terms["height_scan"]
    del cfg.observations["critic"].terms["height_scan"]

    gravity_term_name = "projected_gravity"
    cfg.observations["actor"].terms[gravity_term_name] = deepcopy(
        cfg.observations["actor"].terms[gravity_term_name]
    )
    cfg.observations["actor"].terms["base_ang_vel"] = deepcopy(
        cfg.observations["actor"].terms["base_ang_vel"]
    )

    # IMU obs delay — match velocity's 2026-07 audit values.
    cfg.observations["actor"].terms["base_ang_vel"].delay_min_lag = 0
    cfg.observations["actor"].terms["base_ang_vel"].delay_max_lag = 1
    cfg.observations["actor"].terms["base_ang_vel"].delay_update_period = 64
    cfg.observations["actor"].terms[gravity_term_name].delay_min_lag = 0
    cfg.observations["actor"].terms[gravity_term_name].delay_max_lag = 1
    cfg.observations["actor"].terms[gravity_term_name].delay_update_period = 64

    # Obs noise — matched to the velocity env.
    cfg.observations["actor"].terms["base_ang_vel"].noise    = Unoise(n_min=-0.03, n_max=0.03)
    cfg.observations["actor"].terms[gravity_term_name].noise = Unoise(n_min=-0.01, n_max=0.01)
    cfg.observations["actor"].terms["joint_pos"].noise       = Unoise(n_min=-0.001, n_max=0.001)
    cfg.observations["actor"].terms["joint_vel"].noise       = Unoise(n_min=-0.25, n_max=0.25)

    if ENABLE_IMU_ORIENTATION_RANDOMIZATION:
        av = cfg.observations["actor"].terms["base_ang_vel"]
        av.func = microduck_mdp.base_ang_vel_imu_misaligned
        av.params = {"max_angle_deg": IMU_ORIENTATION_RANDOMIZATION_ANGLE}
        g = cfg.observations["actor"].terms[gravity_term_name]
        g.func = microduck_mdp.projected_gravity_imu_misaligned
        g.params = {"max_angle_deg": IMU_ORIENTATION_RANDOMIZATION_ANGLE}

    # 1-ctrl-step lag on joint_vel (Dynamixel moving-average, see velocity env).
    cfg.observations["actor"].terms["joint_vel"] = deepcopy(
        cfg.observations["actor"].terms["joint_vel"]
    )
    cfg.observations["actor"].terms["joint_vel"].delay_min_lag = 1
    cfg.observations["actor"].terms["joint_vel"].delay_max_lag = 1
    cfg.observations["actor"].terms["joint_vel"].delay_update_period = 0

    passive_excluded = SceneEntityCfg("robot", joint_names=(r"^(?!passive_).*",))
    for grp in ("actor", "critic"):
        for term in ("joint_pos", "joint_vel"):
            cfg.observations[grp].terms[term] = deepcopy(cfg.observations[grp].terms[term])
            cfg.observations[grp].terms[term].params["asset_cfg"] = deepcopy(passive_excluded)

    if ENABLE_ENCODER_BIAS:
        cfg.events["encoder_bias"].params["bias_range"] = ENCODER_BIAS_RANGE
        cfg.observations["actor"].terms["joint_pos"].params["biased"] = True
        cfg.observations["critic"].terms["joint_pos"].params["biased"] = False
    else:
        cfg.events.pop("encoder_bias", None)

    # Command obs slots — unified layout parity: [twist(3), head(4), body(6)]. The twist slot
    # carries the bow flag; head and body are zero-padded, as the runtime feeds every skill.
    for group in ("actor", "critic"):
        cfg.observations[group].terms["head_command"] = ObservationTermCfg(
            func=microduck_mdp.zero_command_padding, params={"dim": 4},
        )
        cfg.observations[group].terms["body_command"] = ObservationTermCfg(
            func=microduck_mdp.zero_command_padding, params={"dim": 6},
        )

    # ── Command: the bow flag in the twist slot ───────────────────────────────
    # cmd = [bow_flag, 0, 0]; dwell-time resampling flips it mid-episode. The term slews an
    # internal target blend over BOW_RAMP_S that the bow rewards track; the OBS stays the raw
    # binary flag, which is what the runtime writes.
    command = cfg.commands["twist"]
    command.rel_standing_envs = 0.0
    command.rel_heading_envs  = 0.0
    command.heading_command   = False
    command.ranges.heading    = None
    command.resampling_time_range = BOW_DWELL_S
    command.debug_vis = False
    cfg.commands["twist"] = microduck_mdp.BowCommandCfg(
        **{
            **vars(command),
            "sit_prob":  BOW_PROB,
            "ramp_s":    BOW_RAMP_S,
            "sit_z":     STAND_Z,   # unused: the bow's blend is pitch, not height
            "stand_z":   STAND_Z,
            "bow_pitch": BOW_PITCH,
        }
    )

    # ── Terminations ──────────────────────────────────────────────────────────
    # fell_over KEPT at the recipe's 70°: a bow that tips over is a failed episode, and the
    # limit is nearly twice the bow.
    assert cfg.terminations["fell_over"].params["limit_angle"] > BOW_PITCH + math.radians(20)
    cfg.terminations["nan_state"] = TerminationTermCfg(
        func=microduck_mdp.robot_state_is_nan,
        time_out=False,
    )

    # ── Events ────────────────────────────────────────────────────────────────
    cfg.events["expand_bam_friction_fields"] = EventTermCfg(
        func=microduck_mdp.expand_bam_friction_fields,
        mode="startup",
    )
    cfg.events["reset_action_history"] = EventTermCfg(
        func=microduck_mdp.reset_action_history,
        mode="reset",
    )
    cfg.events["foot_friction"].params["asset_cfg"].geom_names = foot_frictions_geom_names
    cfg.events["foot_friction"].params["ranges"] = (0.7, 1.3)

    # Joint noise on the standing start: deployment hands off from the gait, whose settled
    # stand won't match HOME exactly.
    cfg.events["reset_robot_joints"].params["position_range"] = (-0.05, 0.05)

    # Standing-only start (the standup env's spawn machinery: random yaw, a little tilt).
    cfg.events["set_ground_state"] = EventTermCfg(
        func=microduck_mdp.set_random_ground_state,
        mode="reset",
        params={
            "face_down_prob":   0.0,
            "face_up_prob":     0.0,
            "sitting_prob":     0.0,
            "standing_prob":    1.0,
            "sitting_tilt_max": math.radians(5),
            "standing_z_min":   0.11,
            "standing_z_max":   0.12,
        },
    )

    if ENABLE_VELOCITY_PUSHES:
        interval = (0.5, 1.0) if play else VELOCITY_PUSH_INTERVAL_S
        cfg.events["push_robot"] = EventTermCfg(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=interval,
            params={
                "velocity_range": {
                    "x": VELOCITY_PUSH_RANGE,
                    "y": VELOCITY_PUSH_RANGE,
                },
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )

    if ENABLE_COM_RANDOMIZATION:
        cfg.events["randomize_com"] = EventTermCfg(
            func=dr.body_ipos,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
                "operation": "add",
                "ranges": (-COM_RANDOMIZATION_RANGE, COM_RANDOMIZATION_RANGE),
            },
        )

    if ENABLE_HEAD_COM_RANDOMIZATION:
        cfg.events["randomize_head_com"] = EventTermCfg(
            func=dr.body_ipos,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=HEAD_BODY_NAMES),
                "operation": "add",
                "ranges": (-HEAD_COM_RANDOMIZATION_RANGE, HEAD_COM_RANDOMIZATION_RANGE),
            },
        )

    if ENABLE_ARMATURE_RANDOMIZATION:
        cfg.events["randomize_armature"] = EventTermCfg(
            func=dr.joint_armature,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=(r".*",)),
                "operation": "scale",
                "ranges": ARMATURE_RANDOMIZATION_RANGE,
            },
        )

    if ENABLE_KP_RANDOMIZATION or ENABLE_KD_RANDOMIZATION:
        kp_range = KP_RANDOMIZATION_RANGE if ENABLE_KP_RANDOMIZATION else (1.0, 1.0)
        kd_range = KD_RANDOMIZATION_RANGE if ENABLE_KD_RANDOMIZATION else (1.0, 1.0)
        cfg.events["randomize_motor_gains"] = EventTermCfg(
            func=microduck_mdp.randomize_delayed_actuator_gains,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "operation": "scale",
                "kp_range": kp_range,
                "kd_range": kd_range,
            },
        )

    if ENABLE_MASS_INERTIA_RANDOMIZATION:
        _mi_lo, _mi_hi = MASS_INERTIA_RANDOMIZATION_RANGE
        cfg.events["randomize_mass_inertia"] = EventTermCfg(
            func=dr.pseudo_inertia,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
                "alpha_range": (math.log(_mi_lo) / 2.0, math.log(_mi_hi) / 2.0),
            },
        )

    if ENABLE_JOINT_FRICTION_RANDOMIZATION:
        cfg.events["randomize_joint_friction"] = EventTermCfg(
            func=microduck_mdp.randomize_bam_friction,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "scale_range": JOINT_FRICTION_RANDOMIZATION_RANGE,
            },
        )

    # ── Terrain: flat only ────────────────────────────────────────────────────
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None

    # ── Curriculum ────────────────────────────────────────────────────────────
    del cfg.curriculum["terrain_levels"]
    del cfg.curriculum["command_vel"]

    # action_rate ramp — velocity's stages, compressed to a 1000–1500 iteration budget: a bow
    # is slow and the ramp is what keeps it from whipping; by iter 900 it is at full strength.
    cfg.curriculum["action_rate_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name":   "action_rate_l2",
            "weight_stages": [
                {"step": 0,         "weight": -0.1},
                {"step": 300 * 24,  "weight": -0.3},
                {"step": 500 * 24,  "weight": -0.5},
                {"step": 700 * 24,  "weight": -0.8},
                {"step": 900 * 24,  "weight": -1.0},
            ],
        },
    )

    if ENABLE_COM_RANDOMIZATION:
        cfg.curriculum["com_range"] = CurriculumTermCfg(
            func=microduck_mdp.com_range_curriculum,
            params={
                "event_name": "randomize_com",
                "range_stages": [
                    {"step": 0,         "range": 0.003},
                    {"step": 300 * 24,  "range": 0.005},
                    {"step": 600 * 24,  "range": 0.01},
                    {"step": 900 * 24,  "range": 0.015},
                ],
            },
        )

    if ENABLE_HEAD_COM_RANDOMIZATION:
        cfg.curriculum["head_com_range"] = CurriculumTermCfg(
            func=microduck_mdp.com_range_curriculum,
            params={
                "event_name": "randomize_head_com",
                "range_stages": [
                    {"step": 0,         "range": 0.003},
                    {"step": 300 * 24,  "range": 0.005},
                    {"step": 600 * 24,  "range": 0.01},
                ],
            },
        )

    if ENABLE_VELOCITY_PUSHES:
        # Pushes come in after the bow has formed: a shove mid-fold at iter 0 taxes the
        # discovery of the fold itself (the standup / kick timing lesson).
        cfg.curriculum["push_magnitude"] = CurriculumTermCfg(
            func=microduck_mdp.push_curriculum,
            params={
                "event_name": "push_robot",
                "push_stages": [
                    {"step": 0,         "velocity_range": {"x": (0.0, 0.0),    "y": (0.0, 0.0)}},
                    {"step": 400 * 24,  "velocity_range": {"x": (-0.08, 0.08), "y": (-0.08, 0.08)}},
                    {"step": 800 * 24,  "velocity_range": {"x": VELOCITY_PUSH_RANGE, "y": VELOCITY_PUSH_RANGE}},
                ],
            },
        )

    return cfg


# ── RL runner config ──────────────────────────────────────────────────────────

MicroduckPoliteBowRlCfg = RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,  # normalizer MUST be baked into ONNX by export.py
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "init_std": 1.0,
            "std_type": "scalar",
        },
    ),
    critic=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
    ),
    algorithm=PpoWithSymmetryCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        symmetry_cfg=SYMMETRY_CFG if ENABLE_SYMMETRY else None,
    ),
    wandb_project="mjlab_microduck",
    experiment_name="polite_bow",
    run_name="polite_bow",
    save_interval=250,
    num_steps_per_env=24,
    max_iterations=1500,
)
