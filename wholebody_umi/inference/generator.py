"""Single-primitive rollout with generated-history carry and cosine overlap."""

from typing import Optional, Tuple

import numpy as np
import torch

from wholebody_umi.config import CONFIG
from wholebody_umi.inference.sampling import sample_primitive
from wholebody_umi.motion.conditioning import (
    build_hand_condition,
    build_lookahead,
    make_relative_hand_segment,
)
from wholebody_umi.motion.layout import QPOS_DIM, QPOS_ROOT_QUATERNION, QPOS_ROOT_XY, REFERENCE_YAW
from wholebody_umi.motion.representation import (
    build_reference_history,
    build_standing_history,
    decode_motion,
    reanchor_absolute_history,
)


def _blend_weights(frame_count: int) -> np.ndarray:
    weights = np.ones(frame_count, dtype=np.float64)
    overlap = min(CONFIG.overlap, (frame_count - CONFIG.history_length) // 2)
    if overlap > 0:
        indices = np.arange(1, overlap + 1, dtype=np.float64)
        theta = np.pi * indices / (overlap + 1)
        weights[CONFIG.history_length : CONFIG.history_length + overlap] = 0.5 * (
            1.0 - np.cos(theta)
        )
        weights[frame_count - overlap :] = 0.5 * (1.0 + np.cos(theta))
    return weights


def generate_clip(
    model: torch.nn.Module,
    keypoints: np.ndarray,
    reference_features: Optional[np.ndarray],
    reference_init: bool,
    device: str,
    ode_steps: int,
) -> Tuple[np.ndarray, np.ndarray, float]:
    history_length = CONFIG.history_length
    step = CONFIG.future_length - CONFIG.overlap
    total_frames = len(keypoints)
    anchor_xy = np.zeros(2, dtype=np.float32)
    anchor_yaw = float(reference_features[0, REFERENCE_YAW]) if reference_init else 0.0
    history = (
        build_reference_history(reference_features, history_length, anchor_yaw)
        if reference_init
        else build_standing_history(history_length)
    )
    qpos_sum = np.zeros((total_frames, QPOS_DIM), dtype=np.float64)
    motion_feature_sum = np.zeros((total_frames, CONFIG.motion_dim), dtype=np.float64)
    weight_sum = np.zeros(total_frames, dtype=np.float64)

    for segment_index, start in enumerate(range(0, total_frames, step)):
        frame_count = min(CONFIG.window_length, total_frames - start)
        real_frames = min(CONFIG.window_length + CONFIG.lookahead_length, total_frames - start)
        relative_poses, heights = make_relative_hand_segment(
            keypoints, start, real_frames, anchor_yaw
        )
        condition = build_hand_condition(relative_poses, heights)
        lookahead, lookahead_valid = build_lookahead(condition, real_frames)
        predicted_features = (
            sample_primitive(
                model,
                torch.from_numpy(condition[: CONFIG.window_length]).unsqueeze(0).to(device),
                torch.from_numpy(history).unsqueeze(0).to(device),
                torch.from_numpy(lookahead).unsqueeze(0).to(device),
                torch.from_numpy(lookahead_valid).unsqueeze(0).to(device),
                ode_steps,
            )
            .squeeze(0)
            .cpu()
            .numpy()[:frame_count]
            .astype(np.float32)
        )
        qpos, world_yaw = decode_motion(predicted_features, anchor_xy, anchor_yaw)
        weights = _blend_weights(frame_count)
        emit_from = 0 if segment_index == 0 else history_length
        output_slice = slice(start + emit_from, start + frame_count)
        qpos_sum[output_slice] += weights[emit_from:, None] * qpos[emit_from:]
        motion_feature_sum[output_slice] += (
            weights[emit_from:, None] * predicted_features[emit_from:]
        )
        weight_sum[output_slice] += weights[emit_from:]
        if start + frame_count >= total_frames:
            break
        next_yaw = float(world_yaw[step])
        history = reanchor_absolute_history(
            predicted_features[step : step + history_length],
            src_yaw=anchor_yaw,
            dst_yaw=next_yaw,
        )
        anchor_xy = qpos[step, QPOS_ROOT_XY].astype(np.float32)
        anchor_yaw = next_yaw

    if not np.all(weight_sum > 0):
        raise RuntimeError("Motion rollout left uncovered frames")
    qpos = (qpos_sum / weight_sum[:, None]).astype(np.float32)
    motion_features = (motion_feature_sum / weight_sum[:, None]).astype(np.float32)
    norm = np.linalg.norm(qpos[:, QPOS_ROOT_QUATERNION], axis=1, keepdims=True)
    qpos[:, QPOS_ROOT_QUATERNION] /= np.clip(norm, 1e-8, None)
    if not np.isfinite(qpos).all() or not np.isfinite(motion_features).all():
        raise RuntimeError("Motion rollout produced non-finite values")
    return qpos, motion_features, float(CONFIG.fps)
