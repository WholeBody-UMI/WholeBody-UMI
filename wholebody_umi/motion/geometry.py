"""Quaternion and 6D rotation conversions for hand conditioning."""

import numpy as np


def rot6d_from_quat_wxyz(quaternions: np.ndarray) -> np.ndarray:
    """Convert quaternion(s) in wxyz format to Zhou et al. 6D rotation."""
    quaternions = quaternions / np.linalg.norm(quaternions, axis=-1, keepdims=True).clip(1e-8)
    w, x, y, z = quaternions[..., 0], quaternions[..., 1], quaternions[..., 2], quaternions[..., 3]
    rotation_matrix = np.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x * x + z * z),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x * x + y * y),
        ],
        axis=-1,
    ).reshape(quaternions.shape[:-1] + (3, 3))
    return rotation_matrix[..., :2].reshape(quaternions.shape[:-1] + (6,))


def quat_wxyz_relative_to_first(quaternions: np.ndarray) -> np.ndarray:
    """Express each quaternion as inv(quaternions[0]) * quaternions[t] along the time axis."""
    quaternions = quaternions / np.linalg.norm(quaternions, axis=-1, keepdims=True).clip(1e-8)
    initial_inverse = quaternions[:1].copy()
    initial_inverse[..., 1:] *= -1.0

    initial_inverse = np.broadcast_to(initial_inverse, quaternions.shape)
    w1, x1, y1, z1 = [initial_inverse[..., i] for i in range(4)]
    w2, x2, y2, z2 = [quaternions[..., i] for i in range(4)]
    relative_quaternions = np.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        axis=-1,
    )
    return relative_quaternions.astype(np.float32)
