"""Standing/reference history, root decoding and reanchoring."""

from typing import Tuple

import numpy as np
from scipy.spatial.transform import Rotation

from wholebody_umi.config import CONFIG
from wholebody_umi.motion.layout import (
    BASE_MOTION_CHANNELS,
    FOOT_CONTACTS,
    JOINT_ANGLES,
    ROOT_DELTA_X,
    ROOT_DELTA_XY,
    ROOT_DELTA_Y,
    ROOT_HEIGHT,
    ROOT_PITCH_COS_OFFSET,
    ROOT_PITCH_SIN,
    ROOT_ROLL_COS_OFFSET,
    ROOT_ROLL_SIN,
    ROOT_X,
    ROOT_XY,
    ROOT_Y,
    ROOT_YAW_COS,
    ROOT_YAW_DELTA,
    ROOT_YAW_SIN,
)

_STANDING_JOINTS = {
    0: -0.312,  # left_hip_pitch
    3: 0.669,  # left_knee
    4: -0.363,  # left_ankle_pitch
    6: -0.312,  # right_hip_pitch
    9: 0.669,  # right_knee
    10: -0.363,  # right_ankle_pitch
    15: 0.082,  # left_shoulder_pitch
    16: 0.008,  # left_shoulder_roll
    17: -0.041,  # left_shoulder_yaw
    18: -0.179,  # left_elbow
    22: 0.082,  # right_shoulder_pitch
    23: -0.008,  # right_shoulder_roll
    24: 0.041,  # right_shoulder_yaw
    25: -0.179,  # right_elbow
}


def append_abs_root_channels(base_features: np.ndarray) -> np.ndarray:
    """Append cumulative root xy and yaw cos/sin to a heading-aligned reference."""
    frame_count = base_features.shape[0]
    out = np.zeros((frame_count, CONFIG.motion_dim), dtype=np.float32)
    out[:, BASE_MOTION_CHANNELS] = base_features[:, BASE_MOTION_CHANNELS]
    if frame_count > 1:
        out[1:, ROOT_XY] = np.cumsum(base_features[:-1, ROOT_DELTA_XY], axis=0)
        yaw_rel = np.concatenate([[0.0], np.cumsum(base_features[:-1, ROOT_YAW_DELTA])]).astype(
            np.float32
        )
    else:
        yaw_rel = np.zeros(frame_count, dtype=np.float32)
    out[:, ROOT_YAW_COS] = np.cos(yaw_rel)
    out[:, ROOT_YAW_SIN] = np.sin(yaw_rel)
    return out


def build_standing_history(history_length: int) -> np.ndarray:
    """Build the canonical standing seed in the model's motion representation."""
    seed = np.zeros((history_length, CONFIG.motion_dim), dtype=np.float32)
    seed[:, FOOT_CONTACTS] = 1.0
    seed[:, ROOT_HEIGHT] = 0.76
    for joint_index, angle in _STANDING_JOINTS.items():
        seed[:, JOINT_ANGLES.start + joint_index] = angle
    seed[:, ROOT_YAW_COS] = 1.0
    return seed


def build_reference_history(
    reference_features: np.ndarray,
    history_length: int,
    anchor_yaw: float,
) -> np.ndarray:
    """Align raw reference history to its heading and append model root channels."""
    aligned = reference_features[:history_length].copy()
    cosine, sine = np.cos(anchor_yaw), np.sin(anchor_yaw)
    dx, dy = aligned[:, ROOT_DELTA_X].copy(), aligned[:, ROOT_DELTA_Y].copy()
    aligned[:, ROOT_DELTA_X] = cosine * dx + sine * dy
    aligned[:, ROOT_DELTA_Y] = -sine * dx + cosine * dy
    return append_abs_root_channels(aligned[:, BASE_MOTION_CHANNELS])


def decode_motion(
    motion_features: np.ndarray,
    anchor_xy: np.ndarray,
    anchor_yaw: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Decode model motion features into world-frame G1 qpos and yaw."""
    motion_features = motion_features.astype(np.float32, copy=False)
    relative_xy = motion_features[:, ROOT_XY].astype(np.float64)
    relative_yaw = np.arctan2(
        motion_features[:, ROOT_YAW_SIN], motion_features[:, ROOT_YAW_COS]
    ).astype(np.float64)
    cosine, sine = np.cos(anchor_yaw), np.sin(anchor_yaw)
    world_xy = np.stack(
        [
            cosine * relative_xy[:, 0] - sine * relative_xy[:, 1],
            sine * relative_xy[:, 0] + cosine * relative_xy[:, 1],
        ],
        axis=-1,
    )
    xy = (np.asarray(anchor_xy, dtype=np.float64)[None] + world_xy).astype(np.float32)
    yaw = (float(anchor_yaw) + relative_yaw).astype(np.float32)
    position = np.zeros((len(motion_features), 3), dtype=np.float32)
    position[:, :2] = xy
    position[:, 2] = motion_features[:, ROOT_HEIGHT]
    roll = np.arctan2(
        motion_features[:, ROOT_ROLL_SIN],
        np.clip(motion_features[:, ROOT_ROLL_COS_OFFSET] + 1.0, -1.0, 1.0),
    )
    pitch = np.arctan2(
        motion_features[:, ROOT_PITCH_SIN],
        np.clip(motion_features[:, ROOT_PITCH_COS_OFFSET] + 1.0, -1.0, 1.0),
    )
    euler = np.stack([roll, pitch, yaw], axis=-1)
    quaternion = np.roll(Rotation.from_euler("xyz", euler).as_quat(), 1, axis=-1)
    qpos = np.concatenate(
        [position, quaternion.astype(np.float32), motion_features[:, JOINT_ANGLES]], axis=-1
    )
    return qpos.astype(np.float32), yaw


def _rotate_delta_channels(
    motion_features: np.ndarray, src_yaw: float, dst_yaw: float
) -> np.ndarray:
    out = motion_features.astype(np.float32, copy=True)
    delta = float(src_yaw) - float(dst_yaw)
    cos_d = np.cos(delta)
    sin_d = np.sin(delta)
    dx = out[:, ROOT_DELTA_X].copy()
    dy = out[:, ROOT_DELTA_Y].copy()
    out[:, ROOT_DELTA_X] = cos_d * dx - sin_d * dy
    out[:, ROOT_DELTA_Y] = sin_d * dx + cos_d * dy
    return out


def reanchor_absolute_history(
    motion_features: np.ndarray, src_yaw: float, dst_yaw: float
) -> np.ndarray:
    """Move generated history into the next rollout window's root frame."""
    out = _rotate_delta_channels(motion_features, src_yaw=src_yaw, dst_yaw=dst_yaw)
    delta = float(src_yaw) - float(dst_yaw)
    cos_d, sin_d = np.cos(delta), np.sin(delta)
    xy = motion_features[:, ROOT_XY] - motion_features[0:1, ROOT_XY]
    out[:, ROOT_X] = cos_d * xy[:, 0] - sin_d * xy[:, 1]
    out[:, ROOT_Y] = sin_d * xy[:, 0] + cos_d * xy[:, 1]
    yaw_rel = np.arctan2(
        motion_features[:, ROOT_YAW_SIN], motion_features[:, ROOT_YAW_COS]
    ) - np.arctan2(motion_features[0, ROOT_YAW_SIN], motion_features[0, ROOT_YAW_COS])
    out[:, ROOT_YAW_COS] = np.cos(yaw_rel)
    out[:, ROOT_YAW_SIN] = np.sin(yaw_rel)
    return out
