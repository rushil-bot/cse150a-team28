"""Tests for the rounD loader in av_gap.traffic.

These use a tiny made-up scene written to a temporary folder, not rows from
rounD, so no licensed data ends up in the repo.
"""

import numpy as np
import pytest

pd = pytest.importorskip("pandas")

from highway_env.vehicle.behavior import IDMVehicle

from av_gap.envs import make_env
from av_gap.traffic import (
    ROUND_JAM_GAP,
    ROUND_JAM_GAP_PCTS,
    ROUND_TARGET_SPEED_PCTS,
    ROUND_TIME_GAP,
    ROUND_TIME_GAP_PCTS,
    CalibratedVehicle,
    find_leaders,
    load_location,
    load_measurements,
    summarize,
)

NAN = np.nan
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


def write_recording(folder, prefix, location_id=0):
    """Write a made-up recording: one kept frame (0) and one skipped frame (1)."""
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
        # Frame 1 is skipped (not a multiple of the frame step), but still
        # counts when deciding who is parked.
        (1, 1, 0.0, 0.0, 0.0, 10.0, 0.0),
        (2, 1, 4.0, 0.0, 0.0, 10.0, 0.0),
        (4, 1, 0.5, 100.0, 0.0, 5.0, 0.0),
        (5, 1, 8.5, 100.0, 0.0, 5.0, 0.0),
    ]
    pd.DataFrame(rows, columns=TRACK_COLUMNS).to_csv(
        folder / f"{prefix}_tracks.csv", index=False)
    meta.to_csv(folder / f"{prefix}_tracksMeta.csv", index=False)
    pd.DataFrame({"recordingId": [int(prefix)], "locationId": [location_id]}).to_csv(
        folder / f"{prefix}_recordingMeta.csv", index=False)
    return folder / f"{prefix}_tracks.csv", folder / f"{prefix}_tracksMeta.csv"


def test_load_measurements(tmp_path):
    out = load_measurements(*write_recording(tmp_path, "00"), frame_step=5)

    # Frame 0 rows, in order: car 1, car 2, van 4, truck 5, car 6.
    # The pedestrian and the two parked cars are gone.
    np.testing.assert_allclose(out["speed"], [10, 10, 0, 0, 10])
    np.testing.assert_allclose(out["gap"], [10, NAN, 8, NAN, NAN])
    np.testing.assert_allclose(out["leader_speed"], [10, NAN, 0, NAN, NAN])


def test_load_location_keeps_only_that_location(tmp_path):
    write_recording(tmp_path, "00", location_id=1)
    write_recording(tmp_path, "02", location_id=0)
    write_recording(tmp_path, "03", location_id=0)

    out, recording_ids = load_location(tmp_path, location_id=0)
    assert recording_ids == [2, 3]
    assert len(out["speed"]) == 2 * 5  # five kept rows per recording


def test_load_location_without_recordings_fails(tmp_path):
    write_recording(tmp_path, "00", location_id=1)
    with pytest.raises(ValueError):
        load_location(tmp_path, location_id=0)


def test_summarize():
    measurements = {
        "speed":        np.array([10.0, 10.0, 0.0, 0.0, 0.0, 12.0, 1.5, 3.0]),
        "gap":          np.array([17.0, NAN, 7.0, 30.0, 9.0, NAN, 10.0, 45.0]),
        "leader_speed": np.array([10.0, NAN, 0.0, 5.0, 0.0, NAN, 1.0, 3.0]),
    }
    out = summarize(measurements)

    # Jam gaps: rows 2 and 4 (both cars stopped). Row 3 is stopped, but its
    # leader is driving away, so it is not a queue.
    assert out["jam_gap"]["n"] == 2
    assert out["jam_gap"]["median"] == pytest.approx(8.0)

    # Time gaps use (gap - jam gap) / speed. Row 0: (17 - 8) / 10 = 0.9 s.
    # Row 6 is too slow to count; row 7 gives 12.3 s, over the 10 s limit.
    assert out["time_gap"]["n"] == 1
    assert out["time_gap"]["median"] == pytest.approx(0.9)

    # Target speed: moving cars with no leader, rows 1 and 5.
    assert out["target_speed"]["n"] == 2
    assert out["target_speed"]["median"] == pytest.approx(11.0)
    assert len(out["target_speed"]["percentiles"]) == 5


def background_cars(seed=0, **make_env_kwargs):
    """Reset a roundabout and return its background cars (not the agent)."""
    env = make_env("roundabout", **make_env_kwargs)
    try:
        env.reset(seed=seed)
        sim = env.unwrapped
        return [v for v in sim.road.vehicles if v is not sim.vehicle]
    finally:
        env.close()


def car_settings(car):
    return (car.DISTANCE_WANTED, car.TIME_WANTED, car.target_speed)


def test_calibrated_cars_stay_within_measured_spread():
    for seed in range(5):
        cars = background_cars(seed, calibrated_traffic=True)
        assert cars
        for car in cars:
            assert type(car) is CalibratedVehicle
            jam, time_gap, speed = car_settings(car)
            assert ROUND_JAM_GAP_PCTS[0] <= jam <= ROUND_JAM_GAP_PCTS[-1]
            assert ROUND_TIME_GAP_PCTS[0] <= time_gap <= ROUND_TIME_GAP_PCTS[-1]
            assert ROUND_TARGET_SPEED_PCTS[0] <= speed <= ROUND_TARGET_SPEED_PCTS[-1]


def test_calibrated_cars_differ_from_each_other():
    cars = background_cars(calibrated_traffic=True)
    assert len({car_settings(car) for car in cars}) == len(cars)


def test_calibrated_cars_repeat_with_the_same_seed():
    first = [car_settings(car) for car in background_cars(3, calibrated_traffic=True)]
    again = [car_settings(car) for car in background_cars(3, calibrated_traffic=True)]
    other = [car_settings(car) for car in background_cars(4, calibrated_traffic=True)]
    assert first == again
    assert first != other


def test_per_car_draws_do_not_change_the_class():
    background_cars(calibrated_traffic=True)
    assert CalibratedVehicle.DISTANCE_WANTED == ROUND_JAM_GAP
    assert CalibratedVehicle.TIME_WANTED == ROUND_TIME_GAP


def test_default_traffic_is_unchanged():
    cars = background_cars()
    assert cars
    for car in cars:
        assert type(car) is IDMVehicle
        assert car.TIME_WANTED == 1.5
        assert car.target_speed > 10  # spawn speed ~16 m/s, not the calibrated 7.95


def test_calibrated_traffic_does_not_change_idm_vehicle():
    background_cars(calibrated_traffic=True)
    assert IDMVehicle.TIME_WANTED == 1.5
    assert IDMVehicle.DISTANCE_WANTED == 10.0


def test_calibrated_traffic_only_on_the_roundabout():
    with pytest.raises(ValueError):
        make_env("highway", calibrated_traffic=True)
