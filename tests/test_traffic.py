"""Tests for the rounD loader in av_gap.traffic.

These use a tiny made-up scene written to a temporary folder, not rows from
rounD, so no licensed data ends up in the repo.
"""

import numpy as np
import pytest

pd = pytest.importorskip("pandas")

from av_gap.traffic import find_leaders, load_measurements

TRACK_COLUMNS = ["trackId", "frame", "xCenter", "yCenter", "heading",
                 "xVelocity", "yVelocity"]


def test_leader_is_nearest_car_ahead():
    # Three cars in a row heading +x; the middle one leads the last one.
    leader, gap = find_leaders(x=[0, 10, 25], y=[0, 0, 0], heading_deg=[0, 0, 0])
    assert list(leader) == [1, 2, -1]
    assert gap[0] == pytest.approx(10)
    assert gap[1] == pytest.approx(15)
    assert np.isnan(gap[2])


def test_car_behind_is_not_a_leader():
    leader, _ = find_leaders(x=[10, 0], y=[0, 0], heading_deg=[0, 0])
    assert leader[0] == -1


def test_car_outside_cone_is_not_a_leader():
    # 45 degrees off-axis is outside the 20-degree cone.
    leader, _ = find_leaders(x=[0, 10], y=[0, 10], heading_deg=[0, 0])
    assert leader[0] == -1


def test_oncoming_car_is_not_a_leader():
    leader, _ = find_leaders(x=[0, 10], y=[0, 0], heading_deg=[0, 180])
    assert leader[0] == -1


def test_car_too_far_away_is_not_a_leader():
    leader, _ = find_leaders(x=[0, 80], y=[0, 0], heading_deg=[0, 0])
    assert leader[0] == -1


@pytest.fixture
def tiny_recording(tmp_path):
    """One kept frame (0) and one skipped frame (1) of a made-up scene."""
    meta = pd.DataFrame({
        "trackId": [1, 2, 3, 4, 5, 6, 7, 8],
        "class": ["car", "car", "pedestrian", "van", "truck", "car", "car", "car"],
    })
    rows = [
        # Moving pair on y=0: car 1 follows car 2 at 10 m, both at 10 m/s.
        (1, 0, 0.0, 0.0, 0.0, 10.0, 0.0),
        (2, 0, 10.0, 0.0, 0.0, 10.0, 0.0),
        # A pedestrian right in front of car 1 must be ignored.
        (3, 0, 5.0, 0.0, 0.0, 1.0, 0.0),
        # Stopped pair on y=100: van 4 waits 8 m behind truck 5.
        # (Both drive off in frame 1, so they are queued, not parked.)
        (4, 0, 0.0, 100.0, 0.0, 0.0, 0.0),
        (5, 0, 8.0, 100.0, 0.0, 0.0, 0.0),
        # An oncoming car between cars 1 and 2 must be ignored.
        (6, 0, 3.0, 0.5, 180.0, -10.0, 0.0),
        # Parked pair on y=200: never move, so both are dropped entirely.
        (7, 0, 0.0, 200.0, 0.0, 0.0, 0.0),
        (8, 0, 6.0, 200.0, 0.0, 0.0, 0.0),
        (7, 1, 0.0, 200.0, 0.0, 0.0, 0.0),
        (8, 1, 6.0, 200.0, 0.0, 0.0, 0.0),
        # Frame 1 is not a multiple of the frame step, so it is skipped
        # (it still counts when deciding who is parked).
        (1, 1, 0.0, 0.0, 0.0, 10.0, 0.0),
        (2, 1, 4.0, 0.0, 0.0, 10.0, 0.0),
        (4, 1, 0.5, 100.0, 0.0, 5.0, 0.0),
        (5, 1, 8.5, 100.0, 0.0, 5.0, 0.0),
    ]
    tracks = pd.DataFrame(rows, columns=TRACK_COLUMNS)

    tracks_csv = tmp_path / "00_tracks.csv"
    meta_csv = tmp_path / "00_tracksMeta.csv"
    tracks.to_csv(tracks_csv, index=False)
    meta.to_csv(meta_csv, index=False)
    return tracks_csv, meta_csv


def test_load_measurements(tiny_recording):
    out = load_measurements(*tiny_recording, frame_step=5)

    # Only car 1 is moving with a leader: 10 m / 10 m/s = 1 s.
    assert out["time_gaps"] == pytest.approx([1.0])
    # Only van 4 is stopped with a leader, 8 m behind. The parked pair's
    # 6 m gap must not appear.
    assert out["jam_gaps"] == pytest.approx([8.0])
    # Frame 0 keeps the five vehicles (cars, van, truck), not the pedestrian
    # and not the two parked cars.
    assert sorted(out["speeds"]) == pytest.approx([0, 0, 10, 10, 10])
