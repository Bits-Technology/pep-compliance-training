"""The viewer's follow camera: `--view` parsing and the keyboard steering, with no window.

A camera spec is typed by hand into an environment variable, so each way of getting it wrong has
to be refused with the spec in the message; and the keys are the only way to move the camera
without a mouse, so each one has to do what the help says.
"""

import pytest

from mjlab_microduck.sim.body_server import DEFAULT_VIEW, FollowCamera, parse_view


class FakeCam:
    """What the viewer opened with: a free overview from 2 m, looking at the floor's middle."""

    def __init__(self):
        self.type = 0
        self.trackbodyid = -1
        self.lookat = [0.3, 0.0, 0.1]
        self.distance = 2.0
        self.azimuth = 90.0
        self.elevation = -45.0


def test_free_and_nothing_mean_the_viewers_own_camera():
    assert parse_view(None) is None
    assert parse_view("") is None
    assert parse_view("free") is None


def test_follow_takes_defaults_and_overrides_in_order():
    assert parse_view("follow") == DEFAULT_VIEW
    assert parse_view("follow:0.5") == (0.5, DEFAULT_VIEW[1], DEFAULT_VIEW[2])
    assert parse_view("follow:0.5,90") == (0.5, 90.0, DEFAULT_VIEW[2])
    assert parse_view(" follow:0.5, 90 ,-30 ") == (0.5, 90.0, -30.0)


@pytest.mark.parametrize("spec", ["chase", "follow:near", "follow:0", "follow:1,2,3,4"])
def test_a_bad_spec_is_refused_naming_it(spec):
    with pytest.raises(SystemExit) as refused:
        parse_view(spec)
    assert spec in str(refused.value)


def test_the_keys_orbit_zoom_and_reset_the_follow_view():
    cam = FakeCam()
    follow = FollowCamera(cam, body_id=7, view=(0.7, 135.0, -20.0))
    assert cam.trackbodyid == 7 and cam.distance == 0.7

    follow.key(263)  # left
    assert cam.azimuth == 125.0
    follow.key(265)  # up: look from higher, i.e. further below the horizon
    assert cam.elevation == -25.0
    follow.key(266)  # page up: closer
    assert cam.distance == pytest.approx(0.7 / 1.2)
    follow.key(268)  # home, while following: the follow view as it started
    assert (cam.distance, cam.azimuth, cam.elevation) == (0.7, 135.0, -20.0)

    # The elevation never passes the pole, where the camera would flip.
    for _ in range(40):
        follow.key(265)
    assert cam.elevation == -89.0


def test_end_gives_back_the_overview_the_viewer_opened_with_and_home_follows_again():
    cam = FakeCam()
    follow = FollowCamera(cam, body_id=7, view=(0.7, 135.0, -20.0))
    follow.key(266)  # zoom the follow view in first, so the two views are told apart

    follow.key(269)  # End: the free overview, exactly as it was
    assert cam.trackbodyid == -1
    assert (list(cam.lookat), cam.distance, cam.azimuth, cam.elevation) == ([0.3, 0.0, 0.1], 2.0, 90.0, -45.0)

    # The same keys steer the free camera, and End again resets only it.
    follow.key(267)  # page down: further
    follow.key(262)  # right
    assert cam.distance == pytest.approx(2.4) and cam.azimuth == 100.0
    follow.key(269)
    assert (cam.distance, cam.azimuth) == (2.0, 90.0)

    follow.key(268)  # Home: back to following, with the zoom it had
    assert cam.trackbodyid == 7
    assert cam.distance == pytest.approx(0.7 / 1.2)


def test_the_viewers_own_keys_are_left_to_it():
    """Letters are display toggles in the viewer — F is contact forces — and must do nothing here."""
    cam = FakeCam()
    follow = FollowCamera(cam, body_id=7, view=(0.7, 135.0, -20.0))
    before = (cam.type, cam.trackbodyid, cam.distance, cam.azimuth, cam.elevation)
    for keycode in [70, 67, 84, 32, 258, 48, 91, 93, 259, 61, 45]:  # F C T Space Tab 0 [ ] Backspace = -
        follow.key(keycode)
    assert (cam.type, cam.trackbodyid, cam.distance, cam.azimuth, cam.elevation) == before
