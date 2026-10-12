"""Play generated G1 poses in MuJoCo or record one pass to MP4."""

import time
from pathlib import Path
from typing import Optional, Union

import numpy as np

from wholebody_umi.config import CONFIG
from wholebody_umi.motion.layout import QPOS_DIM, QPOS_ROOT_QUATERNION

_G1_XML = Path(__file__).resolve().parent / "assets/g1/g1_29dof.xml"
_HAND_BODIES = ("left_wrist_yaw_link", "right_wrist_yaw_link")
_HAND_OFFSETS = np.array([[0.0415, 0.0030, 0.0], [0.0415, -0.0030, 0.0]])
_HAND_COLORS = np.array([[0.00, 0.30, 0.90, 1.0], [0.88, 0.05, 0.05, 1.0]], dtype=np.float32)


def _hand_positions(mujoco, model, data, qpos: np.ndarray) -> np.ndarray:
    body_ids = [model.body(name).id for name in _HAND_BODIES]
    positions = np.empty((len(qpos), 2, 3), dtype=np.float64)
    for frame, pose in enumerate(qpos):
        data.qpos[:] = pose
        mujoco.mj_kinematics(model, data)
        for hand, body_id in enumerate(body_ids):
            rotation = data.xmat[body_id].reshape(3, 3)
            positions[frame, hand] = data.xpos[body_id] + rotation @ _HAND_OFFSETS[hand]
    return positions


def _draw_hand_trails(mujoco, scene, positions: np.ndarray, frame: int) -> None:
    """Show the next 60 frames of generated palm FK, left blue and right red."""
    end = min(frame + 59, len(positions) - 1)
    indices = np.unique(np.append(np.arange(frame, end + 1, 3), end))
    for hand, color in enumerate(_HAND_COLORS):
        for first, second in zip(indices[:-1], indices[1:]):
            start, finish = positions[first, hand], positions[second, hand]
            if np.linalg.norm(finish - start) < 1e-8:
                continue
            rgba = color.copy()
            rgba[3] = 0.9 - 0.7 * (second - frame) / max(end - frame, 1)
            geom = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(
                geom,
                mujoco.mjtGeom.mjGEOM_CAPSULE,
                np.full(3, 0.006),
                np.zeros(3),
                np.eye(3).ravel(),
                rgba,
            )
            mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.006, start, finish)
            scene.ngeom += 1
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            geom,
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.full(3, 0.012),
            positions[frame, hand],
            np.eye(3).ravel(),
            color,
        )
        scene.ngeom += 1


def _set_camera(camera, pose: np.ndarray) -> None:
    camera.lookat[:] = (pose[0], pose[1], max(0.65, pose[2] - 0.08))


def _record_video(
    mujoco,
    model,
    data,
    qpos: np.ndarray,
    hand_positions: np.ndarray,
    fps: float,
    video: Path,
    width: int,
    height: int,
) -> None:
    try:
        import imageio.v2 as imageio
        import imageio_ffmpeg
    except ImportError as error:
        raise RuntimeError(
            "Install video dependencies with uv sync --locked --extra visualization"
        ) from error
    imageio_ffmpeg.get_ffmpeg_exe()
    video.parent.mkdir(parents=True, exist_ok=True)
    model.vis.global_.offwidth = width
    model.vis.global_.offheight = height
    camera = mujoco.MjvCamera()
    camera.azimuth = 135.0
    camera.elevation = -18.0
    camera.distance = 2.6
    option = mujoco.MjvOption()
    option.sitegroup[:] = 0
    with mujoco.Renderer(model, height=height, width=width, max_geom=2000) as renderer:
        with imageio.get_writer(
            str(video),
            fps=fps,
            codec="libx264",
            quality=8,
            macro_block_size=2,
            ffmpeg_params=["-pix_fmt", "yuv420p", "-bf", "0", "-movflags", "+faststart"],
        ) as writer:
            for frame, pose in enumerate(qpos):
                data.qpos[:] = pose
                mujoco.mj_forward(model, data)
                _set_camera(camera, pose)
                renderer.update_scene(data, camera=camera, scene_option=option)
                _draw_hand_trails(mujoco, renderer.scene, hand_positions, frame)
                writer.append_data(renderer.render())
    print(f"Saved {len(qpos)} frames at {fps:g} fps to {video}")


def _play_viewer(
    mujoco, model, data, qpos: np.ndarray, hand_positions: np.ndarray, fps: float
) -> None:
    import mujoco.viewer

    data.qpos[:] = qpos[0]
    mujoco.mj_forward(model, data)
    print("MuJoCo playback: left hand blue, right hand red. Close the window to exit.")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        with viewer.lock():
            viewer.cam.azimuth = 135.0
            viewer.cam.elevation = -18.0
            viewer.cam.distance = 2.6
            viewer.opt.sitegroup[:] = 0
        frame = 0
        while viewer.is_running():
            started = time.monotonic()
            with viewer.lock():
                data.qpos[:] = qpos[frame]
                mujoco.mj_forward(model, data)
                _set_camera(viewer.cam, qpos[frame])
                viewer.user_scn.ngeom = 0
                _draw_hand_trails(mujoco, viewer.user_scn, hand_positions, frame)
            viewer.sync()
            frame = (frame + 1) % len(qpos)
            time.sleep(max(0.0, 1.0 / fps - (time.monotonic() - started)))


def visualize_motion(
    qpos: np.ndarray,
    fps: float = CONFIG.fps,
    video: Optional[Union[str, Path]] = None,
    width: int = 1280,
    height: int = 720,
    skip_frames: int = CONFIG.history_length,
) -> None:
    """Loop MuJoCo playback, or save one pass when ``video`` is an MP4 path.

    The initial history is hidden by default. No filtering, pose correction or
    physics stepping is applied; the input qpos array is unchanged.
    """
    qpos = np.asarray(qpos, dtype=np.float64)
    if qpos.ndim != 2 or qpos.shape[1] != QPOS_DIM or not np.isfinite(qpos).all():
        raise ValueError("Expected finite G1 qpos with shape (T, 36)")
    if not np.allclose(
        np.linalg.norm(qpos[:, QPOS_ROOT_QUATERNION], axis=1), 1.0, atol=0.01, rtol=0
    ):
        raise ValueError("Root quaternions must be unit quaternions in wxyz order")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive")
    if not 0 <= skip_frames < len(qpos):
        raise ValueError("skip_frames must leave at least one frame to display")
    if width < 2 or height < 2 or width % 2 or height % 2:
        raise ValueError("Video width and height must be positive even integers")
    video = Path(video).expanduser() if video is not None else None
    if video is not None and video.suffix.lower() != ".mp4":
        raise ValueError("--video must be an .mp4 path")
    try:
        import mujoco
    except ImportError as error:
        raise RuntimeError(
            "MuJoCo could not load. Run uv sync --locked --extra visualization and "
            "check OpenGL; for headless recording set MUJOCO_GL=egl. "
            f"Original error: {error}"
        ) from error
    qpos = qpos[skip_frames:]
    model = mujoco.MjModel.from_xml_path(str(_G1_XML))
    data = mujoco.MjData(model)
    hand_positions = _hand_positions(mujoco, model, data, qpos)
    if video is None:
        _play_viewer(mujoco, model, data, qpos, hand_positions, fps)
    else:
        _record_video(mujoco, model, data, qpos, hand_positions, fps, video, width, height)
