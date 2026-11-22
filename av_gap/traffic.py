"""Measure real driving from rounD and apply it to the simulator's background cars.

Gaps are center to center, the same way HighwayEnv measures them.
"""

from pathlib import Path
import numpy as np
from highway_env.vehicle.behavior import IDMVehicle

# Trailers are left out: a towed trailer would look like a leader at ~0 m.
VEHICLE_CLASSES = ("car", "truck", "bus", "van")

FRAME_STEP = 5  # keep every 5th frame (25 fps -> 5 fps)

CONE_DEG = 20.0
SAME_DIRECTION_DEG = 45.0
MAX_LEADER_DIST = 50.0  # in m

PARKED_MAX_SPEED = 1.0  # in m/s tracks that never exceed this are parked cars
MOVING_SPEED = 2.0  # in m/s time gaps and free-flow speeds only while moving
STOPPED_SPEED = 1.0  # in m/s jam gaps only while stopped

MAX_TIME_GAP = 10.0  # in s longer means the cars were not really following
PERCENTILES = (10, 25, 50, 75, 90)


def find_leaders(x, y, heading_deg):
    """Find each vehicle's leader (the car ahead) within one frame.

    The leader is the nearest vehicle that is ahead, inside the cone, and
    driving the same direction. Returns `(leader, gap)`: leader index (-1 if
    none) and distance (NaN if none).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    h = np.radians(np.asarray(heading_deg, dtype=float))
    hx, hy = np.cos(h), np.sin(h)

    # dx[i, j] = x[j] - x[i]: where vehicle j sits relative to vehicle i.
    dx = x[None, :] - x[:, None]
    dy = y[None, :] - y[:, None]
    dist = np.hypot(dx, dy)

    ahead = dx * hx[:, None] + dy * hy[:, None]
    with np.errstate(invalid="ignore", divide="ignore"):
        cos_offaxis = ahead / dist
    cos_heading_diff = hx[:, None] * hx[None, :] + hy[:, None] * hy[None, :]

    candidate = (
        (ahead > 0)
        & (cos_offaxis >= np.cos(np.radians(CONE_DEG)))
        & (cos_heading_diff >= np.cos(np.radians(SAME_DIRECTION_DEG)))
        & (dist <= MAX_LEADER_DIST)
    )
    np.fill_diagonal(candidate, False)

    masked = np.where(candidate, dist, np.inf)
    leader = masked.argmin(axis=1)
    gap = masked.min(axis=1)
    no_leader = np.isinf(gap)
    leader[no_leader] = -1
    gap[no_leader] = np.nan

    return leader, gap


def load_measurements(tracks_csv, tracks_meta_csv, frame_step=FRAME_STEP):
    """Read one rounD recording.

    Returns arrays "speed", "gap" and "leader_speed", one entry per vehicle
    per frame (gap and leader_speed are NaN when there is no leader).
    """
    import pandas as pd  # imported here so make_env never needs pandas

    meta = pd.read_csv(tracks_meta_csv, usecols=["trackId", "class"])
    vehicle_ids = meta.loc[meta["class"].isin(VEHICLE_CLASSES), "trackId"]

    tracks = pd.read_csv(
        tracks_csv,
        usecols=["trackId", "frame", "xCenter", "yCenter", "heading",
                 "xVelocity", "yVelocity"],
    )
    tracks = tracks[tracks["trackId"].isin(vehicle_ids)]
    tracks = tracks.assign(speed=np.hypot(tracks["xVelocity"], tracks["yVelocity"]))

    # Drop parked cars.
    top_speed = tracks.groupby("trackId")["speed"].max()
    tracks = tracks[tracks["trackId"].isin(top_speed.index[top_speed > PARKED_MAX_SPEED])]

    tracks = tracks[tracks["frame"] % frame_step == 0]

    speeds, gaps, leader_speeds = [np.empty(0)], [np.empty(0)], [np.empty(0)]
    for _, frame in tracks.groupby("frame", sort=False):
        leader, gap = find_leaders(
            frame["xCenter"].to_numpy(),
            frame["yCenter"].to_numpy(),
            frame["heading"].to_numpy(),
        )
        v = frame["speed"].to_numpy()
        speeds.append(v)
        gaps.append(gap)
        leader_speeds.append(np.where(leader >= 0, v[leader], np.nan))

    return {
        "speed": np.concatenate(speeds),
        "gap": np.concatenate(gaps),
        "leader_speed": np.concatenate(leader_speeds),
    }


def load_location(data_dir, location_id=0):
    """Load every recording at one location (0 = Neuweiler) and combine them."""
    import pandas as pd

    data_dir = Path(data_dir)
    parts, recording_ids = [], []
    for meta_csv in sorted(data_dir.glob("*_recordingMeta.csv")):
        rec = pd.read_csv(meta_csv).iloc[0]
        if rec["locationId"] != location_id:
            continue
        prefix = meta_csv.name.split("_")[0]
        parts.append(load_measurements(
            data_dir / f"{prefix}_tracks.csv",
            data_dir / f"{prefix}_tracksMeta.csv",
        ))
        recording_ids.append(int(rec["recordingId"]))

    if not parts:
        raise ValueError(f"No recordings for location {location_id} in {data_dir}")
    measurements = {key: np.concatenate([p[key] for p in parts]) for key in parts[0]}

    return measurements, recording_ids


def summarize(measurements):
    """Turn raw measurements into typical values (median and percentiles).

    - jam gap: gap when the car and its leader are both stopped
    - time gap: (gap - jam gap) / speed, matching IDM's jam gap + time gap * speed
    - target speed: speed of moving cars with no leader
    """
    speed = measurements["speed"]
    gap = measurements["gap"]
    leader_speed = measurements["leader_speed"]
    following = ~np.isnan(gap)

    jam = following & (speed < STOPPED_SPEED) & (leader_speed < STOPPED_SPEED)
    jam_gaps = gap[jam]
    jam_gap = np.median(jam_gaps)

    moving = following & (speed > MOVING_SPEED)
    time_gaps = (gap[moving] - jam_gap) / speed[moving]
    time_gaps = time_gaps[(time_gaps > 0) & (time_gaps <= MAX_TIME_GAP)]

    free_speeds = speed[~following & (speed > MOVING_SPEED)]

    def stats(values):
        return {
            "n": len(values),
            "median": float(np.median(values)),
            "percentiles": [float(p) for p in np.percentile(values, PERCENTILES)],
        }

    return {
        "jam_gap": stats(jam_gaps),
        "time_gap": stats(time_gaps),
        "target_speed": stats(free_speeds),
    }


# From summarize(load_location("round-dataset-v1/data", 0)): Neuweiler, recordings 02-23.
# Saved here so the dataset is not needed to run calibrated traffic.
ROUND_JAM_GAP = 6.89  # in m
ROUND_TIME_GAP = 1.30  # in s
ROUND_TARGET_SPEED = 7.95  # in m/s

LEVELS = [0.10, 0.25, 0.50, 0.75, 0.90]
ROUND_JAM_GAP_PCTS = [5.80, 6.26, 6.89, 7.69, 9.74]  # in m
ROUND_TIME_GAP_PCTS = [0.50, 0.80, 1.30, 2.17, 3.76]  # in s
ROUND_TARGET_SPEED_PCTS = [5.58, 6.82, 7.95, 9.30, 11.02]  # in m/s


class CalibratedVehicle(IDMVehicle):
    """A background car with settings measured from rounD.

    A subclass, so the default IDMVehicle stays unchanged.
    """

    DISTANCE_WANTED = ROUND_JAM_GAP
    TIME_WANTED = ROUND_TIME_GAP

    def randomize_behavior(self):
        # Called on every background car when it is created. Each car draws
        # its own values; the episode seed makes the draws repeatable.
        super().randomize_behavior()
        rng = self.road.np_random
        self.DISTANCE_WANTED = _draw(rng, ROUND_JAM_GAP_PCTS)
        self.TIME_WANTED = _draw(rng, ROUND_TIME_GAP_PCTS)
        self.target_speed = _draw(rng, ROUND_TARGET_SPEED_PCTS)


def _draw(rng, values_at_levels):
    """Draw one value from the measured spread (between p10 and p90)."""
    return float(np.interp(rng.uniform(LEVELS[0], LEVELS[-1]), LEVELS, values_at_levels))