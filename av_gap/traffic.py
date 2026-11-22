"""Measure real driving from the rounD recordings.

For each vehicle, the car ahead ("leader") is the nearest vehicle that is in
front of it, inside a narrow cone around its heading, driving the same
direction, and within MAX_LEADER_DIST. Gaps are center to center, the same
way HighwayEnv measures them.
"""

import numpy as np

# Trailers are left out: a towed trailer would look like a leader at ~0 m.
VEHICLE_CLASSES = ("car", "truck", "bus", "van")

FRAME_STEP = 5  # keep every 5th frame (25 fps -> 5 fps)

CONE_DEG = 20.0
SAME_DIRECTION_DEG = 45.0
MAX_LEADER_DIST = 50.0  # m

PARKED_MAX_SPEED = 1.0  # m/s tracks that never exceed this are parked cars
MOVING_SPEED = 2.0  # m/s time gaps only while moving
STOPPED_SPEED = 1.0  # m/s jam gaps only while stopped


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
    """Extract raw measurements from one rounD recording.

    Returns a dict of arrays: "speeds" (m/s), "time_gaps" (gap / speed, s,
    while moving) and "jam_gaps" (gap in m, while stopped).
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
    speed = tracks["speed"].to_numpy()

    time_gaps = [np.empty(0)]
    jam_gaps = [np.empty(0)]
    for _, frame in tracks.groupby("frame", sort=False):
        _, gap = find_leaders(
            frame["xCenter"].to_numpy(),
            frame["yCenter"].to_numpy(),
            frame["heading"].to_numpy(),
        )
        v = frame["speed"].to_numpy()
        has_leader = ~np.isnan(gap)

        moving = has_leader & (v > MOVING_SPEED)
        time_gaps.append(gap[moving] / v[moving])

        stopped = has_leader & (v < STOPPED_SPEED)
        jam_gaps.append(gap[stopped])

    return {
        "time_gaps": np.concatenate(time_gaps),
        "jam_gaps": np.concatenate(jam_gaps),
        "speeds": speed,
    }
