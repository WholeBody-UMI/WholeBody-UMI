"""Hand-window conditioning and masked lookahead for the released model."""

from typing import Tuple

import numpy as np

from wholebody_umi.config import CONFIG
from wholebody_umi.motion.geometry import quat_wxyz_relative_to_first, rot6d_from_quat_wxyz
from wholebody_umi.motion.layout import CONDITION_VELOCITY


def make_relative_hand_segment(
    keypoints: np.ndarray,
    start: int,
    frame_count: int,
    anchor_yaw: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Rebase hand positions and rotate poses into the segment heading."""
    relative_poses = keypoints[start : start + frame_count].copy()
    initial_heights = relative_poses[0, :, 2].copy()
    relative_poses[:, :, :3] -= relative_poses[0:1, :, :3]
    cos_y = np.cos(anchor_yaw)
    sin_y = np.sin(anchor_yaw)
    x = relative_poses[:, :, 0].copy()
    y = relative_poses[:, :, 1].copy()
    relative_poses[:, :, 0] = cos_y * x + sin_y * y
    relative_poses[:, :, 1] = -sin_y * x + cos_y * y
    # Rotate the wxyz quats by R_z(-anchor_yaw): q_heading = q_z(-anchor_yaw) (x) q_world.
    # q_z(-anchor_yaw) = [cos(anchor_yaw/2), 0, 0, -sin(anchor_yaw/2)]. Only its
    # w and z components are non-zero, so the Hamilton product simplifies.
    cw = np.cos(anchor_yaw * 0.5)
    sw = -np.sin(anchor_yaw * 0.5)
    # .copy() so the in-place writes below don't alias the RHS reads.
    qw = relative_poses[:, :, 3].copy()
    qx = relative_poses[:, :, 4].copy()
    qy = relative_poses[:, :, 5].copy()
    qz = relative_poses[:, :, 6].copy()
    relative_poses[:, :, 3] = cw * qw - sw * qz
    relative_poses[:, :, 4] = cw * qx - sw * qy
    relative_poses[:, :, 5] = cw * qy + sw * qx
    relative_poses[:, :, 6] = cw * qz + sw * qw
    return relative_poses, initial_heights


def build_hand_condition(poses: np.ndarray, initial_heights: np.ndarray) -> np.ndarray:
    """Return relative pose 18 + height 2 + orientation 12 + velocity 18."""
    rotations = quat_wxyz_relative_to_first(poses[..., 3:])
    pose_features = (
        np.concatenate(
            [
                poses[..., :3],
                rot6d_from_quat_wxyz(rotations),
            ],
            axis=-1,
        )
        .reshape(len(poses), 18)
        .astype(np.float32)
    )
    heights = np.broadcast_to(initial_heights.astype(np.float32), (len(poses), 2))
    orientation = rot6d_from_quat_wxyz(poses[:1, :, 3:]).reshape(1, 12)
    anchors = np.broadcast_to(orientation.astype(np.float32), (len(poses), 12))
    velocity = np.zeros_like(pose_features)
    velocity[1:] = pose_features[1:] - pose_features[:-1]
    condition = np.concatenate([pose_features, heights, anchors, velocity], axis=-1)
    if len(condition) < CONFIG.window_length:
        hold = condition[-1].copy()
        hold[CONDITION_VELOCITY] = 0.0
        padding = np.repeat(hold[None], CONFIG.window_length - len(condition), axis=0)
        condition = np.concatenate([condition, padding], axis=0)
    return condition.astype(np.float32)


def build_lookahead(condition: np.ndarray, real_frames: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return future hand conditions and a mask that excludes padded frames."""
    preview = np.zeros((CONFIG.lookahead_length, CONFIG.condition_dim), dtype=np.float32)
    valid = np.zeros(CONFIG.lookahead_length, dtype=np.float32)
    available = int(np.clip(real_frames - CONFIG.window_length, 0, CONFIG.lookahead_length))
    if available > 0:
        preview[:available] = condition[CONFIG.window_length : CONFIG.window_length + available]
        hold = preview[available - 1].copy()
    else:
        hold = condition[max(min(CONFIG.window_length, real_frames) - 1, 0)].copy()
    hold[CONDITION_VELOCITY] = 0.0
    preview[available:] = hold
    valid[:available] = 1.0
    return preview, valid
