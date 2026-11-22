"""Measure real driving from the rounD recordings.

For each vehicle, the car ahead ("leader") is the nearest vehicle that is in
front of it, inside a narrow cone around its heading, driving the same
direction, and within MAX_LEADER_DIST. Gaps are center to center, the same
way HighwayEnv measures them.

The typical values match the three IDM settings HighwayEnv uses:
- jam gap (DISTANCE_WANTED): gap when both the car and its leader are stopped.
- time gap (TIME_WANTED): (gap - jam gap) / speed while following, because
  IDM spaces cars at jam gap + time gap * speed.
- target speed: speed of moving cars with no leader (free flow).
"""

from pathlib import Path
import numpy as np

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
    """Find each vehicle's leader within one frame.

    Returns `(leader, gap)`: `leader[i]` is the index of vehicle i's leader
    (-1 if none) and `gap[i]` the center-to-center distance (NaN if none).
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
    """Extract raw per-row measurements from one rounD recording.

    Returns a dict of aligned arrays, one entry per kept vehicle row:
    "speed" (m/s), "gap" to the leader (m) and "leader_speed" (m/s).
    The last two are NaN when the vehicle has no leader.
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

    # Drop parked cars (judged on the full track, before skipping frames).
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
    """Run `load_measurements` on every recording at one location and combine.

    Returns `(measurements, recording_ids)`. Location 0 is Neuweiler.
    """
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
    """Turn raw measurements into typical values for the three IDM settings.

    Returns {"jam_gap", "time_gap", "target_speed"}, each a dict with
    "n", "median" and "percentiles" (values at PERCENTILES).
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
