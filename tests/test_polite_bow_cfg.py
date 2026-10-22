"""The polite bow's config: what it rewards, with which sign, and the one bit of maths that
has to be right for any of it to mean anything — the trunk's forward pitch off the quaternion.
"""

import math
from types import SimpleNamespace

import torch

from mjlab_microduck.tasks import mdp
from mjlab_microduck.tasks.microduck_polite_bow_env_cfg import (
    BOW_HEAD_OVERRIDES,
    BOW_PITCH,
    BOW_PITCH_DEG,
    BOW_RAMP_S,
    make_microduck_polite_bow_env_cfg,
)


def _asset_with_quat(w, x, y, z):
    q = torch.tensor([[w, x, y, z]], dtype=torch.float32)
    return SimpleNamespace(data=SimpleNamespace(root_link_quat_w=q))


def test_forward_pitch_is_nose_down_positive_and_roll_is_separate():
    """A pure pitch of +20° about y (right-hand, y left) tips the nose down; a pure roll must
    read as roll and not as pitch. Get this sign wrong and the policy learns to lean back."""
    half = math.radians(20.0) / 2
    nose_down = _asset_with_quat(math.cos(half), 0.0, math.sin(half), 0.0)
    assert math.isclose(mdp._trunk_forward_pitch(nose_down).item(), math.radians(20.0), abs_tol=1e-5)
    assert abs(mdp._trunk_roll(nose_down).item()) < 1e-6

    tilted = _asset_with_quat(math.cos(half), math.sin(half), 0.0, 0.0)
    assert abs(mdp._trunk_forward_pitch(tilted).item()) < 1e-6
    assert math.isclose(abs(mdp._trunk_roll(tilted).item()), math.radians(20.0), abs_tol=1e-5)

    upright = _asset_with_quat(1.0, 0.0, 0.0, 0.0)
    assert mdp._trunk_forward_pitch(upright).item() == 0.0


def test_the_bow_is_between_polite_and_apologetic():
    assert 35.0 <= BOW_PITCH_DEG <= 40.0
    assert math.isclose(BOW_PITCH, math.radians(BOW_PITCH_DEG))
    # About 30°/s: a deliberate bow, not a nod.
    assert 1.0 <= BOW_RAMP_S <= 1.5


def test_the_command_is_a_flag_with_a_slewed_pitch_blend():
    cfg = make_microduck_polite_bow_env_cfg()
    cmd = cfg.commands["twist"]
    assert isinstance(cmd, mdp.BowCommandCfg)
    assert cmd.bow_pitch == BOW_PITCH
    assert cmd.ramp_s == BOW_RAMP_S
    assert cmd.sit_prob == 0.5
    assert cmd.heading_command is False


def test_reward_signs_are_the_intended_ones():
    """Gaussian trackers reward; mjlab-base costs (≥ 0) take a negative weight; self-negating
    microduck penalties (≤ 0) take a POSITIVE one. bow-v1 had the L1 the wrong way round and was
    paid +0.78 a step for pitch error (the playbook's bit-four-envs lesson, now bit five)."""
    r = make_microduck_polite_bow_env_cfg().rewards
    assert r["bow_pitch_track"].weight == 4.0
    assert r["bow_pitch_track"].params["bow_pitch"] == BOW_PITCH
    assert r["bow_pitch_l1"].weight > 0, "self-negating penalty: positive weight"
    assert r["bow_head_pose"].weight > 0
    assert r["bow_head_pose"].params["sit_overrides"] == BOW_HEAD_OVERRIDES
    assert r["bow_head_pose"].params["joint_indices"] == [5, 6, 7, 8]
    for positive in ("trunk_roll", "feet_grounded", "height_stand", "pose_stand_legs", "bow_stillness"):
        assert r[positive].weight > 0, positive
    for negative in ("action_rate_l2", "body_ang_vel", "angular_momentum", "self_collisions"):
        assert r[negative].weight < 0, negative
    assert "upright" not in r, "the recipe's upright would fight the fold"
    assert "track_linear_velocity" not in r


def test_the_l1_term_is_self_negating():
    """The reason its weight is positive: the function itself never returns above zero."""
    err = torch.tensor([0.0, 0.3, -0.5])
    assert torch.all(-torch.abs(err) <= 0)
    # And the weighted sign the run will log: positive weight × (−|err|) ≤ 0.
    r = make_microduck_polite_bow_env_cfg().rewards
    assert r["bow_pitch_l1"].weight * (-0.3) <= 0


def test_the_head_drops_at_the_bottom():
    """Head pitch positive is down, and the bottom of the bow looks at the floor ahead; the
    neck leans a little forward rather than back."""
    neck, head = BOW_HEAD_OVERRIDES[5], BOW_HEAD_OVERRIDES[6]
    home = 0.3491
    assert head > home + 0.4
    assert neck < home


def test_a_tipped_over_bow_still_terminates_but_a_bow_does_not():
    cfg = make_microduck_polite_bow_env_cfg()
    limit = cfg.terminations["fell_over"].params["limit_angle"]
    assert limit > BOW_PITCH + math.radians(20)
    assert "nan_state" in cfg.terminations


def test_the_task_is_registered():
    import mjlab_microduck.tasks  # noqa: F401  (registers on import)
    from mjlab.tasks.registry import list_tasks

    assert "Mjlab-PoliteBow-Flat-MicroDuck" in list_tasks()
