import json
import inspect
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

import hydra
import numpy as np
import torch
from accelerate import PartialState
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from tqdm import tqdm

# rotation utils
import transforms3d as t3d
from scipy.spatial.transform import Rotation

# try:
#     import rootutils

#     rootutils.setup_root(__file__, indicator=".python-version", pythonpath=True)
# except ModuleNotFoundError:
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from experiments.libero.libero_utils import (
    LIBERO_ENV_RESOLUTION,
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    invert_gripper_action,
    quat2axisangle,
    save_prediction_video,
    save_rollout_video,
)
from fastwam.datasets.lerobot.processors.fastwam_processor import FastWAMProcessor
from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
from fastwam.utils.pytorch_utils import set_global_seed
from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from libero.libero import benchmark
from action_ensembler import ActionEnsembler

OmegaConf.register_new_resolver("eval", eval)
OmegaConf.register_new_resolver("max", lambda x: max(x))
OmegaConf.register_new_resolver("split", lambda s, idx: s.split("/")[int(idx)])

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _normalize_mixed_precision(mixed_precision: str) -> str:
    key = str(mixed_precision).strip().lower()
    if key not in {"no", "fp16", "bf16"}:
        raise ValueError(
            f"Unsupported mixed_precision: {mixed_precision}. "
            "Expected one of: ['no', 'fp16', 'bf16']."
        )
    return key


def _mixed_precision_to_model_dtype(mixed_precision: str) -> torch.dtype:
    precision = _normalize_mixed_precision(mixed_precision)
    if precision == "no":
        return torch.float32
    if precision == "fp16":
        return torch.float16
    return torch.bfloat16


def _resolve_eval_device(cfg: DictConfig) -> str:
    eval_device = cfg.EVALUATION.get("device")
    if eval_device is not None:
        return str(eval_device)
    return "cuda" if torch.cuda.is_available() else "cpu"


def _resolve_dataset_stats_path(cfg: DictConfig) -> Path:
    explicit = cfg.EVALUATION.get("dataset_stats_path")
    candidates: list[Path] = []

    if explicit is not None:
        candidates.append(Path(os.path.expanduser(os.path.expandvars(str(explicit)))))

    ckpt = Path(os.path.expanduser(os.path.expandvars(str(cfg.ckpt))))
    for parent in list(ckpt.parents)[:4]:
        candidates.append(parent / "dataset_stats.json")

    seen = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists():
            return resolved

    msg = (
        "Failed to locate dataset_stats.json. Tried explicit "
        "EVALUATION.dataset_stats_path and checkpoint parent directories. "
        "Please pass EVALUATION.dataset_stats_path=/path/to/dataset_stats.json."
    )
    raise FileNotFoundError(msg)


def _load_model_checkpoint(model: torch.nn.Module, ckpt: str) -> None:
    model.load_checkpoint(ckpt)
    logging.info("Loaded checkpoint via model.load_checkpoint: %s", ckpt)
    return

    # deprecated legacy checkpoint loading
    payload = torch.load(ckpt, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Legacy checkpoint payload must be dict, got: {type(payload)}")

    if "mot" in payload and hasattr(model, "mot"):
        missing, unexpected = model.mot.load_state_dict(payload["mot"], strict=False)
        logging.warning(
            "Loaded fallback `mot` state_dict with strict=False. Missing=%d Unexpected=%d",
            len(missing),
            len(unexpected),
        )
        return

    state_dict = None
    for key in ("model_state_dict", "state_dict", "model"):
        value = payload.get(key)
        if isinstance(value, dict):
            state_dict = value
            break
    if state_dict is None and all(torch.is_tensor(v) for v in payload.values()):
        state_dict = payload
    if state_dict is None:
        raise ValueError(f"Cannot parse legacy checkpoint keys from: {ckpt}")

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    logging.warning(
        "Loaded fallback model state_dict with strict=False. Missing=%d Unexpected=%d",
        len(missing),
        len(unexpected),
    )


def _center_crop_resize(image: np.ndarray, width: int, height: int) -> np.ndarray:
    pil_image = Image.fromarray(image)
    src_w, src_h = pil_image.size
    scale = max(width / src_w, height / src_h)
    resized = pil_image.resize((round(src_w * scale), round(src_h * scale)), resample=Image.BILINEAR)
    rw, rh = resized.size
    left = max((rw - width) // 2, 0)
    top = max((rh - height) // 2, 0)
    cropped = resized.crop((left, top, left + width, top + height))
    return np.asarray(cropped, dtype=np.uint8)


def _normalize_proprio(
    proprio: np.ndarray,
    processor: FastWAMProcessor,
) -> torch.Tensor:
    state_meta = processor.shape_meta["state"]
    if len(state_meta) != 1:
        raise ValueError(
            "LIBERO eval currently expects a single merged state key in shape_meta['state']."
        )
    state_key = state_meta[0]["key"]

    state_batch = {"state": {state_key: torch.as_tensor(proprio, dtype=torch.float32).unsqueeze(0)}}
    state_batch = processor.action_state_transform(state_batch)
    state_batch = processor.normalizer.forward(state_batch)
    return state_batch["state"][state_key]


def _obs_to_model_input(
    obs: dict,
    cfg: DictConfig,
    processor: FastWAMProcessor,
    width: int,
    height: int,
    device: str,
    dtype: torch.dtype,
):
    imgs = get_libero_image(obs)
    image_meta = processor.shape_meta["images"]
    if len(image_meta) < int(processor.num_output_cameras):
        raise ValueError(
            f"shape_meta.images has {len(image_meta)} entries, "
            f"but num_output_cameras={processor.num_output_cameras}."
        )

    def _meta_to_hw(meta: dict, camera_idx: int) -> tuple[int, int]:
        shape = meta["shape"]
        if len(shape) != 3:
            raise ValueError(f"shape_meta.images[{camera_idx}].shape must be [C,H,W], got {shape}")
        return int(shape[1]), int(shape[2])

    concatenation = cfg.data.train.get("concat_multi_camera", "horizontal")
    num_cameras = processor.num_output_cameras
    if num_cameras == 1:
        primary_h, primary_w = _meta_to_hw(image_meta[0], camera_idx=0)
        rgb = _center_crop_resize(imgs["image"], width=primary_w, height=primary_h)
    elif num_cameras == 2:
        primary_h, primary_w = _meta_to_hw(image_meta[0], camera_idx=0)
        wrist_h, wrist_w = _meta_to_hw(image_meta[1], camera_idx=1)
        primary = _center_crop_resize(imgs["image"], width=primary_w, height=primary_h)
        wrist = _center_crop_resize(imgs["wrist_image"], width=wrist_w, height=wrist_h)
        if concatenation == "horizontal":
            rgb = np.concatenate([primary, wrist], axis=1)
        elif concatenation == "vertical":
            rgb = np.concatenate([primary, wrist], axis=0)
        else:
            raise ValueError(f"Invalid concat_multi_camera: {concatenation}")
    else:
        raise ValueError(f"LIBERO eval currently supports num_output_cameras in [1, 2], got {num_cameras}.")

    actual_h, actual_w = int(rgb.shape[0]), int(rgb.shape[1])
    expected_h, expected_w = int(height), int(width)
    image_shapes = [meta["shape"] for meta in image_meta]
    assert actual_h == expected_h and actual_w == expected_w, (
        "Input image size mismatch after per-camera resize + concat: "
        f"got (H,W)=({actual_h},{actual_w}), expected (H,W)=({expected_h},{expected_w}) "
        f"from data.train.video_size={[expected_h, expected_w]}; "
        f"shape_meta.images={image_shapes}, concat_multi_camera={concatenation}."
    )

    x = torch.tensor(rgb).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=dtype)
    x = x * (2.0 / 255.0) - 1.0

    proprio = _normalize_proprio(_extract_sim_state(obs), processor)

    return x, proprio, imgs


def _extract_sim_state(obs: dict) -> np.ndarray:
    """Build simulator state from current observation.

    This is used as proprio input for model inference.
    """
    state = np.concatenate(
        (
            obs["robot0_eef_pos"],
            quat2axisangle(obs["robot0_eef_quat"]),
            obs["robot0_gripper_qpos"],
        )
    ).astype(np.float32)

    # logging.info(f"NOTE: state quat: {obs['robot0_eef_quat']}")

    try:
        assert state.shape == (8,)
    except AssertionError:
        logging.info(f"pos: {obs['robot0_eef_pos']}")
        logging.info(f"axis_angle: {quat2axisangle(obs['robot0_eef_quat'])}")
        logging.info(f"gripper: {obs['robot0_gripper_qpos']}")
        logging.error(f"State shape is {state.shape}, expected (8,).")
        raise ValueError(f"State shape is {state.shape}, expected (8,).")

    return state


def _denormalize_action(action: torch.Tensor, processor: FastWAMProcessor) -> np.ndarray:
    if action.ndim == 2:
        action = action.unsqueeze(0)
    if action.ndim != 3:
        raise ValueError(f"Expected action tensor [B, T, D], got {tuple(action.shape)}")

    action_meta = processor.shape_meta["action"]
    if len(action_meta) != 1:
        raise ValueError(
            "LIBERO eval currently expects a single merged action key in shape_meta['action']."
        )

    action_key = action_meta[0]["key"]
    normalizer = processor.normalizer.normalizers["action"][action_key]
    action = action.to(dtype=torch.float32, device="cpu")
    denorm = normalizer.backward(action)
    return denorm.numpy()


def _get_num_video_frames(cfg: DictConfig) -> int:
    return (int(cfg.data.train.num_frames) - 1) // int(cfg.data.train.action_video_freq_ratio) + 1


def _validate_visualize_future_video_cfg(cfg: DictConfig) -> None:
    if not bool(cfg.EVALUATION.get("visualize_future_video", False)):
        return

    action_conditioned = cfg.model.video_dit_config.get("action_conditioned", None)
    if action_conditioned is not False:
        raise ValueError(
            "EVALUATION.visualize_future_video=true requires "
            "model.video_dit_config.action_conditioned=false."
        )


def _select_predicted_future_frames(pred_video: list[Image.Image], cfg: DictConfig) -> list[Image.Image]:
    if len(pred_video) == 0:
        raise ValueError("`infer_joint` returned an empty predicted video.")

    replan_steps = int(cfg.EVALUATION.get("replan_steps", 5))
    action_video_freq_ratio = int(cfg.data.train.action_video_freq_ratio)
    num_future_frames = replan_steps // action_video_freq_ratio
    keep_frames = 1 + num_future_frames
    return list(pred_video[:keep_frames])


def _get_future_frame_capture_steps(cfg: DictConfig) -> list[int]:
    replan_steps = int(cfg.EVALUATION.get("replan_steps", 5))
    action_video_freq_ratio = int(cfg.data.train.action_video_freq_ratio)
    num_future_frames = replan_steps // action_video_freq_ratio
    return [step_idx * action_video_freq_ratio for step_idx in range(num_future_frames + 1)]


def _frame_to_rgb_array(frame: Any) -> np.ndarray:
    if isinstance(frame, dict):
        images = []
        for value in frame.values():
            value_array = np.array(value) if isinstance(value, Image.Image) else np.array(value, copy=True)
            images.append(value_array)
        return np.concatenate(images, axis=1)
    if isinstance(frame, Image.Image):
        return np.array(frame.convert("RGB"))
    return np.array(frame, copy=True)


def _compute_clip_mean_psnr(
    gt_frames: list[Any],
    pred_frames: list[Any],
    eps: float = 1e-8,
) -> Optional[float]:
    if len(gt_frames) == 0 or len(pred_frames) == 0:
        return None
    assert len(gt_frames) == len(pred_frames), (
        "GT/pred frame count mismatch for PSNR: "
        f"len(gt_frames)={len(gt_frames)} len(pred_frames)={len(pred_frames)}. "
        "This indicates temporal misalignment in future-video capture."
    )
    num_frames = len(gt_frames)

    frame_psnr_values = []
    for gt_frame, pred_frame in zip(gt_frames[:num_frames], pred_frames[:num_frames]):
        gt_image = _frame_to_rgb_array(gt_frame)
        pred_image = _frame_to_rgb_array(pred_frame)
        target_h, target_w = pred_image.shape[:2]
        if gt_image.shape[:2] != (target_h, target_w):
            gt_image = np.array(
                Image.fromarray(gt_image).resize((target_w, target_h), resample=Image.BILINEAR)
            )

        gt_f32 = gt_image.astype(np.float32)
        pred_f32 = pred_image.astype(np.float32)
        mse = float(np.mean((pred_f32 - gt_f32) ** 2))
        psnr = 10.0 * np.log10((255.0 * 255.0) / max(mse, eps))
        frame_psnr_values.append(float(psnr))

    if len(frame_psnr_values) == 0:
        return None
    return float(np.mean(frame_psnr_values))


def _predict_action_chunk(
    obs: dict,
    task_description: str,
    model: torch.nn.Module,
    processor: FastWAMProcessor,
    cfg: DictConfig,
    *,
    action_horizon: int,
    input_w: int,
    input_h: int,
    model_device: str,
    accept_signal: Optional[bool] = None,
) -> tuple[np.ndarray, dict, Optional[list[Image.Image]]]:
    num_inference_steps_cfg = cfg.EVALUATION.get("num_inference_steps", None)
    if num_inference_steps_cfg is None:
        num_inference_steps = int(cfg.get("eval_num_inference_steps", 20))
    else:
        num_inference_steps = int(num_inference_steps_cfg)
    prompt_template = DEFAULT_PROMPT
    prompt = prompt_template.format(task=task_description)

    image, proprio, imgs = _obs_to_model_input(
        obs,
        cfg=cfg,
        processor=processor,
        width=input_w,
        height=input_h,
        device=model_device,
        dtype=model.torch_dtype,
    )

    infer_kwargs = {
        "prompt": prompt,
        "input_image": image,
        "action_horizon": action_horizon,
        "negative_prompt": str(cfg.EVALUATION.get("negative_prompt", "")),
        "text_cfg_scale": float(cfg.EVALUATION.get("text_cfg_scale", 1.0)),
        "num_inference_steps": num_inference_steps,
        "proprio": proprio,
        "sigma_shift": (
            None
            if cfg.EVALUATION.get("sigma_shift") is None
            else float(cfg.EVALUATION.get("sigma_shift"))
        ),
        "seed": None if cfg.get("seed") is None else int(cfg.seed),
        "rand_device": str(cfg.EVALUATION.get("rand_device", "cpu")),
        "tiled": bool(cfg.EVALUATION.get("tiled", False)),
    }
    if model.cache_type == 'batchstep':
        infer_kwargs["accept_signal"] = accept_signal
    visualize_future_video = bool(cfg.EVALUATION.get("visualize_future_video", False))
    predicted_future_frames = None
    if visualize_future_video:
        infer_kwargs["num_video_frames"] = _get_num_video_frames(cfg)
    elif "num_video_frames" in inspect.signature(model.infer_action).parameters:
        infer_kwargs["num_video_frames"] = _get_num_video_frames(cfg)

    with torch.no_grad():
        if visualize_future_video:
            pred = model.infer_joint(**infer_kwargs)
            predicted_future_frames = _select_predicted_future_frames(pred["video"], cfg)
        else:
            pred = model.infer_action(**infer_kwargs)
    action = pred["action"]  # [T, D]
    draft_action = pred.get("draft_action", None)
    orig_action = pred.get("orig_action", None)

    action = _denormalize_action(action, processor)[0]  # [T, D]

    # The dataloader flips the sign of the gripper action to align with other datasets
    # (0 = close, 1 = open), so flip it back (-1 = open, +1 = close) before executing the action
    action[..., -1] = action[..., -1] * 2 - 1
    action = invert_gripper_action(action)
    if bool(cfg.EVALUATION.get("binarize_gripper", False)):
        action[..., -1] = np.sign(action[..., -1])
    
    if draft_action is not None:
        draft_action = _denormalize_action(draft_action, processor)[0]  # [T, D]
        draft_action[..., -1] = draft_action[..., -1] * 2 - 1
        draft_action = invert_gripper_action(draft_action)
        if bool(cfg.EVALUATION.get("binarize_gripper", False)):
            draft_action[..., -1] = np.sign(draft_action[..., -1])
        
    if orig_action is not None:
        orig_action = _denormalize_action(orig_action, processor)[0]  # [T, D]
        orig_action[..., -1] = orig_action[..., -1] * 2 - 1
        orig_action = invert_gripper_action(orig_action)
        if bool(cfg.EVALUATION.get("binarize_gripper", False)):
            orig_action[..., -1] = np.sign(orig_action[..., -1])
    
    action_pkg = {"action": action, "draft_action": draft_action, "orig_action": orig_action}

    return action_pkg, imgs, predicted_future_frames


def _get_max_steps(task_suite_name: str) -> int:
    suite_steps = {
        "libero_spatial": 400,
        "libero_object": 400,
        "libero_goal": 400,
        "libero_10": 700,
        "libero_90": 700,
    }
    if task_suite_name not in suite_steps:
        raise ValueError(f"Unknown task suite: {task_suite_name}")
    return suite_steps[task_suite_name]

# rotation_utils
def axis_angle_to_rotation_matrix(axis_angle: np.ndarray) -> np.ndarray:
    axis_angle = np.asarray(axis_angle, dtype=np.float64).reshape(3)
    angle = np.linalg.norm(axis_angle)
    if angle < 1e-12:
        return np.eye(3, dtype=np.float64)

    x, y, z = axis_angle / angle
    skew = np.array(
        [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]],
        dtype=np.float64,
    )
    return (
        np.eye(3, dtype=np.float64)
        + np.sin(angle) * skew
        + (1.0 - np.cos(angle)) * (skew @ skew)
    )
def rotation_matrix_to_axis_angle(rotation_matrix: np.ndarray) -> np.ndarray:
    def project_to_so3(rotation_matrix: np.ndarray) -> np.ndarray:
        u, _, vh = np.linalg.svd(np.asarray(rotation_matrix, dtype=np.float64))
        projected = u @ vh
        if np.linalg.det(projected) < 0:
            u[:, -1] *= -1.0
            projected = u @ vh
        return projected
    rotation_matrix = project_to_so3(rotation_matrix)
    trace = float(np.trace(rotation_matrix))
    cos_angle = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    angle = float(np.arccos(cos_angle))

    if angle < 1e-12:
        return np.zeros(3, dtype=np.float64)

    if abs(np.pi - angle) < 1e-6:
        diag = np.diag(rotation_matrix)
        axis = np.sqrt(np.maximum((diag + 1.0) / 2.0, 0.0))
        if rotation_matrix[0, 1] < 0.0:
            axis[1] = -axis[1]
        if rotation_matrix[0, 2] < 0.0:
            axis[2] = -axis[2]
        if np.linalg.norm(axis) < 1e-12:
            axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        else:
            axis = axis / np.linalg.norm(axis)
        return axis * angle

    axis = np.array(
        [
            rotation_matrix[2, 1] - rotation_matrix[1, 2],
            rotation_matrix[0, 2] - rotation_matrix[2, 0],
            rotation_matrix[1, 0] - rotation_matrix[0, 1],
        ],
        dtype=np.float64,
    ) / (2.0 * np.sin(angle))
    return axis * angle
def euler_to_rotation_matrix(euler: np.ndarray) -> np.ndarray:
    euler = np.asarray(euler, dtype=np.float64).reshape(3)
    return np.asarray(t3d.euler.euler2mat(euler[0], euler[1], euler[2]), dtype=np.float64)
def rotation_matrix_to_euler(rotation_matrix: np.ndarray) -> np.ndarray:
    euler = t3d.euler.mat2euler(np.asarray(rotation_matrix, dtype=np.float64))
    return np.asarray(euler, dtype=np.float64)

def _relative_to_absolute(action_chunk, state=None):
    action_chunk = np.array(action_chunk)
    if not isinstance(action_chunk, np.ndarray):
        raise TypeError("action_chunk should be np.ndarray")
    if action_chunk.ndim != 2:
        raise ValueError(f"action_chunk should be 2D, got shape={action_chunk.shape}")

    # assert isinstance(state, np.ndarray), f"state should be np.ndarray, got {type(state)}, state={state}"
    # assert state.shape == (7,), f"state should be shape (7,), got shape={state.shape}"

    relative_dims = 6  # int(self.relative_action_dims)
    if relative_dims <= 0:
        return action_chunk.copy()
    if action_chunk.shape[1] < relative_dims:
        raise ValueError(
            f"relative_action_dims={relative_dims} exceeds action dim {action_chunk.shape[1]}"
        )

    absolute_chunk = action_chunk.copy()
    position_dims = 3  # min(relative_dims, 3)
    if state is None:
        position_state = np.zeros(position_dims, dtype=np.float64)
    else:
        position_state = np.asarray(state[:3], dtype=np.float64)  # np.zeros(position_dims, dtype=np.float64)
    tail_dims = max(relative_dims - 6, 0)
    if tail_dims > 0:
        if state is None:
            tail_state = np.zeros(tail_dims, dtype=np.float64)
        else:
            tail_state = np.asarray(state[6:relative_dims], dtype=np.float64)  # np.zeros(tail_dims, dtype=np.float64)
    if state is None:
        rotation_state = np.eye(3, dtype=np.float64)
    else:
        rotation_state = axis_angle_to_rotation_matrix(np.asarray(state[3:6], dtype=np.float64))

    # logging.info(f"NOTE: state axis_angle: {state[3:6]}")

    for idx, action in enumerate(action_chunk):
        if position_dims > 0:
            position_state = position_state + np.asarray(action[:position_dims], dtype=np.float64)
            absolute_chunk[idx, :position_dims] = position_state.astype(
                absolute_chunk.dtype, copy=False
            )
        if relative_dims >= 6:
            rotation_state = axis_angle_to_rotation_matrix(
                np.asarray(action[3:6], dtype=np.float64)
            ) @ rotation_state
            absolute_chunk[idx, 3:6] = rotation_matrix_to_axis_angle(
                rotation_state
            ).astype(
                absolute_chunk.dtype, copy=False
            )
        if tail_dims > 0:
            tail_state = tail_state + np.asarray(action[6:relative_dims], dtype=np.float64)
            absolute_chunk[idx, 6:relative_dims] = tail_state.astype(
                absolute_chunk.dtype, copy=False
            )
    # logging.info(f"NOTE: accumulated axis_angle: {absolute_chunk[-1, 3:6]}")
    absolute_chunk = absolute_chunk.tolist()
    return absolute_chunk, absolute_chunk[-1]


def _accumulate_action(action, state):
    """Accumulate action to state."""
    position_state = np.asarray(state[:3], dtype=np.float64)  # np.zeros(position_dims, dtype=np.float64)
    rotation_state = axis_angle_to_rotation_matrix(np.asarray(state[3:6], dtype=np.float64))

    position_state = position_state + np.asarray(action[:3], dtype=np.float64)
    rotation_state = axis_angle_to_rotation_matrix(
        np.asarray(action[3:6], dtype=np.float64)
    ) @ rotation_state
    return position_state.tolist() + rotation_matrix_to_axis_angle(rotation_state).tolist()

def rotation_distance(aa1, aa2):
    r1 = Rotation.from_rotvec(aa1)
    r2 = Rotation.from_rotvec(aa2)
    r_diff = r1.inv() * r2
    return np.linalg.norm(r_diff.as_rotvec())

def rotation_delta(state1_aa, state2_aa):
    r1 = Rotation.from_rotvec(state1_aa)
    r2 = Rotation.from_rotvec(state2_aa)
    # R_delta = R2 @ R1^{-1}
    r_delta = r2 * r1.inv()
    return r_delta.as_rotvec()

def run_single_episode(
    env,
    initial_state,
    task_description: str,
    model: torch.nn.Module,
    processor: FastWAMProcessor,
    cfg: DictConfig,
    episode_idx: int,
    *,
    action_horizon: int,
    input_w: int,
    input_h: int,
    model_device: str,
) -> tuple[bool, list, list[dict[str, Any]], Optional[float], list, list]:
    max_steps = _get_max_steps(cfg.EVALUATION.task_suite_name)
    replan_steps = int(cfg.EVALUATION.get("replan_steps", 5))
    num_steps_wait = int(cfg.EVALUATION.get("num_steps_wait", 5))
    use_action_ensembler = bool(cfg.EVALUATION.get("use_action_ensembler", False))
    visualize_future_video = bool(cfg.EVALUATION.get("visualize_future_video", False))
    capture_steps = set(_get_future_frame_capture_steps(cfg)[1:])

    # record actions (optional)
    record_actions = bool(cfg.EVALUATION.get("record_actions", False))
    record_actions_bias = bool(cfg.EVALUATION.get("record_actions_bias", False))
    use_absolute_actions = bool(cfg.EVALUATION.get("absolute_actions", False))
    action_array, draft_action_array, orig_action_array = [], [], []
    action_state, draft_action_state, orig_action_state = None, None, None
    action_bias_array = []
    prev_extra_actions = []

    use_batchstep = model.cache_type == "batchstep"
    accept_signal = None
    if use_batchstep:
        accept_signal = False
    fix_actions = []

    env.reset()
    obs = env.set_init_state(initial_state)
    if use_action_ensembler:
        ensembler = ActionEnsembler()
        ensembler.reset()

    replay_images = []
    predicted_future_video_clips: list[dict[str, Any]] = []
    episode_future_clip_psnr: list[float] = []
    pending_actions: list[list[float]] = []
    current_predicted_future_clip: Optional[dict[str, Any]] = None
    current_replan_step = 0
    current_replan_idx = -1

    t = 0
    done = False
    pbar = tqdm(total=max_steps + num_steps_wait, desc=f"Episode {episode_idx + 1}")
    while t < max_steps + num_steps_wait:
        pbar.update(1)
        if t < num_steps_wait:
            obs, _, done, _ = env.step(get_libero_dummy_action())
            t += 1
            continue

        if len(pending_actions) == 0:
            action_pkg, imgs, predicted_future_frames = _predict_action_chunk(
                obs=obs,
                task_description=task_description,
                model=model,
                processor=processor,
                cfg=cfg,
                action_horizon=action_horizon,
                input_w=input_w,
                input_h=input_h,
                model_device=model_device,
                accept_signal=accept_signal,
            )

            action_chunk = action_pkg["action"]
            draft_action_chunk = action_pkg["draft_action"]
            orig_action_chunk = action_pkg["orig_action"]

            if predicted_future_frames is not None:
                current_replan_idx += 1
                current_predicted_future_clip = {
                    "replan_idx": current_replan_idx,
                    "gt_frames": [imgs.copy()],
                    "pred_frames": predicted_future_frames,
                }
            else:
                current_predicted_future_clip = None
            current_replan_step = 0
            if use_action_ensembler:
                ensembler.add_actions(action_chunk, t)
                pending_actions = [ensembler.get_action(ts).tolist() for ts in range(t, t + replan_steps)]

                if draft_action_chunk is not None:
                    pending_draft_actions = [draft_action_chunk[ts].tolist() for ts in range(t, t + replan_steps)]
                if orig_action_chunk is not None:
                    pending_orig_actions = [orig_action_chunk[ts].tolist() for ts in range(t, t + replan_steps)]
            else:
                pending_actions = action_chunk[:replan_steps].tolist()

                if draft_action_chunk is not None:
                    pending_draft_actions = draft_action_chunk[:replan_steps].tolist()
                if orig_action_chunk is not None:
                    pending_orig_actions = orig_action_chunk[:replan_steps].tolist()

                # for action bias
                if record_actions_bias:
                    if prev_extra_actions is not None:
                        action_bias = [[a - b for a, b in zip(i, j)] for i, j in zip(pending_actions, prev_extra_actions)]
                        action_bias_array += action_bias
                    assert 2 * replan_steps <= action_chunk.shape[0], (
                        f"Action chunk size {action_chunk.shape[0]} is not enough to predict {replan_steps} steps."
                    )
                    prev_extra_actions = action_chunk[replan_steps:2 * replan_steps].tolist()
            replay_images.append(imgs.copy())

            # if len(fix_actions) > 0:
            #     pending_actions = fix_actions + pending_actions
            
            # if use_absolute_actions:
            #     logging.info("NOTE: record absolute actions")
            #     current_state = _extract_sim_state(obs)[:7]
            pending_actions_absolute, action_state = _relative_to_absolute(pending_actions, state=action_state)
            if draft_action_chunk is not None:
                pending_draft_actions, draft_action_state = _relative_to_absolute(pending_draft_actions, state=draft_action_state)
            if orig_action_chunk is not None:
                pending_orig_actions, orig_action_state = _relative_to_absolute(pending_orig_actions, state=orig_action_state)
            
            if use_batchstep:
                accept_signal = rotation_distance(action_state[3:6], orig_action_state[3:6]) < 0.05
                if not accept_signal:
                    # fix with target model
                    fix_xyz = np.array(orig_action_state[:3]) - np.array(action_state[:3])
                    fix_angle = rotation_delta(action_state[3:6], orig_action_state[3:6])
                    fix_gripper = np.array(pending_actions[-1][6:])
                    fix_action = np.concatenate(
                        (
                            fix_xyz,
                            fix_angle,
                            fix_gripper,
                        )
                    ).astype(np.float32).tolist()
                    assert type(fix_action) == type(pending_actions[-1]), f"type of existing action: {type(pending_actions[-1])}, type of new action: {type(fix_action)}"
                    assert len(fix_action) == len(pending_actions[-1]), f"fix_action {fix_action} and pending_actions[-1] {pending_actions[-1]} have different shape"
                    fix_actions = [fix_action]
                    pending_actions = pending_actions + fix_actions
                    _, action_state = _relative_to_absolute(fix_actions, state=action_state)
                    accept_signal = True
                    logging.info(f"NOTE: fix action here")

            if record_actions:
                action_array += pending_actions_absolute if use_absolute_actions else pending_actions
                if draft_action_chunk is not None:
                    draft_action_array += pending_draft_actions
                if orig_action_chunk is not None:
                    orig_action_array += pending_orig_actions
        else:
            imgs = get_libero_image(obs)
            replay_images.append(imgs.copy())

    # debug
        # current_state = _extract_sim_state(obs)[:7]
        # # logging.info(f"axis_angle_2: {current_state[3:6]}")
        # logging.info(f"real_before_state: {current_state[:6]}")
        # axis_angle_1 = pending_actions[0][3:6]
        # # logging.info(f"axis_angle_1: {axis_angle_1}")
        # logging.info(f"current_action: {pending_actions[0][:6]}")
        # state_after_action = _accumulate_action(pending_actions[0], current_state)
        # logging.info(f"after_state: {state_after_action[:6]}")
        
        obs, _, done, _ = env.step(pending_actions.pop(0))
        
    # debug
        # current_state = _extract_sim_state(obs)[:7]
        # # logging.info(f"axis_angle_2: {current_state[3:6]}")
        # logging.info(f"real_after_state: {current_state[:6]}")

        if visualize_future_video and current_predicted_future_clip is not None:
            current_replan_step += 1
            if current_replan_step in capture_steps:
                current_predicted_future_clip["gt_frames"].append(get_libero_image(obs))
            if done or len(pending_actions) == 0:
                expected_frame_count = 1 + sum(
                    1 for capture_step in capture_steps if capture_step <= current_replan_step
                )
                gt_len = len(current_predicted_future_clip["gt_frames"])
                pred_len = len(current_predicted_future_clip["pred_frames"])
                assert gt_len == expected_frame_count, (
                    "GT future frames do not match expected capture count: "
                    f"gt_len={gt_len} expected={expected_frame_count} "
                    f"episode={episode_idx} replan={current_predicted_future_clip['replan_idx']} "
                    f"current_replan_step={current_replan_step} capture_steps={sorted(capture_steps)}."
                )
                assert pred_len >= expected_frame_count, (
                    "Predicted future frames shorter than expected capture count: "
                    f"pred_len={pred_len} expected={expected_frame_count} "
                    f"episode={episode_idx} replan={current_predicted_future_clip['replan_idx']}."
                )
                if pred_len != expected_frame_count:
                    logging.info(
                        "Align predicted clip length to executed steps: "
                        "episode=%s replan=%s done=%s expected=%s pred_full=%s",
                        episode_idx,
                        current_predicted_future_clip["replan_idx"],
                        done,
                        expected_frame_count,
                        pred_len,
                    )
                current_predicted_future_clip["pred_frames"] = current_predicted_future_clip["pred_frames"][
                    :expected_frame_count
                ]
                assert len(current_predicted_future_clip["gt_frames"]) == len(
                    current_predicted_future_clip["pred_frames"]
                ), (
                    "GT/pred frame count mismatch after alignment: "
                    f"len(gt_frames)={len(current_predicted_future_clip['gt_frames'])} "
                    f"len(pred_frames)={len(current_predicted_future_clip['pred_frames'])} "
                    f"episode={episode_idx} replan={current_predicted_future_clip['replan_idx']}."
                )
                clip_psnr = _compute_clip_mean_psnr(
                    current_predicted_future_clip["gt_frames"],
                    current_predicted_future_clip["pred_frames"],
                )
                if clip_psnr is not None:
                    episode_future_clip_psnr.append(clip_psnr)
                predicted_future_video_clips.append(current_predicted_future_clip)
                current_predicted_future_clip = None
        if done:
            break
        t += 1
    pbar.close()

    if hasattr(model, "reset_episode"):
        model.reset_episode()

    episode_mean_psnr = (
        float(np.mean(episode_future_clip_psnr)) if len(episode_future_clip_psnr) > 0 else None
    )

    action_array_pkg = {
        "action": action_array,
        "draft_action": draft_action_array if len(draft_action_array) > 0 else None,
        "orig_action": orig_action_array if len(orig_action_array) > 0 else None,
    }

    return bool(done), replay_images, predicted_future_video_clips, episode_mean_psnr, action_array_pkg, action_bias_array


def run_single_task(
    task,
    initial_states,
    model: torch.nn.Module,
    processor: FastWAMProcessor,
    cfg: DictConfig,
    video_dir: Path,
    predicted_video_dir: Path,
    *,
    action_horizon: int,
    input_w: int,
    input_h: int,
    model_device: str,
) -> dict:
    env, task_description = get_libero_env(task, LIBERO_ENV_RESOLUTION, cfg.get("seed"))
    visualize_future_video = bool(cfg.EVALUATION.get("visualize_future_video", False))
    results = {
        "successes": 0,
        "failure_episodes": [],
        "success_episodes": [],
        "task_description": task_description,
    }
    if visualize_future_video:
        results["episode_future_video_psnr"] = []
        results["future_video_psnr_mean"] = None
    
    # record actions (optional)
    record_actions = bool(cfg.EVALUATION.get("record_actions", False))
    record_actions_bias = bool(cfg.EVALUATION.get("record_actions_bias", False))
    action_records, draft_action_records, orig_action_records = {}, {}, {}
    action_bias_records = {}

    for trial_idx in range(int(cfg.EVALUATION.num_trials)):
        success, replay_images, predicted_future_video_clips, episode_mean_psnr, action_array_pkg, action_bias_array = run_single_episode(
            env=env,
            initial_state=initial_states[trial_idx],
            task_description=task_description,
            model=model,
            processor=processor,
            cfg=cfg,
            episode_idx=trial_idx,
            action_horizon=action_horizon,
            input_w=input_w,
            input_h=input_h,
            model_device=model_device,
        )
        if success:
            results["successes"] += 1
            results["success_episodes"].append(trial_idx)
        else:
            results["failure_episodes"].append(trial_idx)
        if visualize_future_video:
            results["episode_future_video_psnr"].append(episode_mean_psnr)

        save_rollout_video(
            video_dir,
            replay_images,
            f"task{cfg.EVALUATION.task_id}_trial{trial_idx}",
            success=success,
            task_description=task_description,
        )
        if visualize_future_video:
            if len(predicted_future_video_clips) == 0:
                logging.warning(
                    "No predicted future frames collected for task %s trial %s.",
                    cfg.EVALUATION.task_id,
                    trial_idx,
                )
            else:
                all_gt_frames = []
                all_pred_frames = []
                for clip in predicted_future_video_clips:
                    all_gt_frames.extend(clip["gt_frames"])
                    all_pred_frames.extend(clip["pred_frames"])
                    save_prediction_video(
                        predicted_video_dir,
                        clip["gt_frames"],
                        clip["pred_frames"],
                        f"task{cfg.EVALUATION.task_id}_trial{trial_idx}",
                        clip["replan_idx"],
                        success=success,
                        task_description=task_description,
                    )
                save_prediction_video(
                    predicted_video_dir,
                    all_gt_frames,
                    all_pred_frames,
                    f"task{cfg.EVALUATION.task_id}_trial{trial_idx}",
                    "all",
                    success=success,
                    task_description=task_description,
                )
        if record_actions:
            action_records[f"trial_{trial_idx}"] = action_array_pkg['action']
            if action_array_pkg['draft_action'] is not None:
                draft_action_records[f"trial_{trial_idx}"] = action_array_pkg['draft_action']
            if action_array_pkg['orig_action'] is not None:
                orig_action_records[f"trial_{trial_idx}"] = action_array_pkg['orig_action']
        if record_actions_bias:
            action_bias_records[f"trial_{trial_idx}"] = action_bias_array

    if visualize_future_video:
        valid_episode_psnr = [x for x in results["episode_future_video_psnr"] if x is not None]
        if len(valid_episode_psnr) > 0:
            results["future_video_psnr_mean"] = float(np.mean(valid_episode_psnr))
    results["action_records"] = action_records
    results["draft_action_records"] = draft_action_records
    results["orig_action_records"] = orig_action_records
    results["action_bias_records"] = action_bias_records
    return results


@hydra.main(version_base="1.3", config_path="../../configs", config_name="sim_libero.yaml")
def eval_single_process(cfg: DictConfig):
    start_time = time.time()
    partial_state = PartialState()
    partial_state.config = cfg

    if cfg.get("seed") is not None:
        set_global_seed(int(cfg.seed), get_worker_init_fn=False)

    if cfg.ckpt is None:
        raise ValueError("cfg.ckpt must not be None.")
    _validate_visualize_future_video_cfg(cfg)

    env_num = int(cfg.EVALUATION.get("env_num", 1))
    if env_num != 1:
        raise ValueError(
            "Only env_num=1 is supported in eval_libero_single.py. "
            "Use run_libero_manager/run_libero_parallel_test.sh for multi-GPU task parallelism."
        )

    model_device = _resolve_eval_device(cfg)
    model_dtype = _mixed_precision_to_model_dtype(cfg.get("mixed_precision", "bf16"))
    model = instantiate(cfg.model, model_dtype=model_dtype, device=model_device)
    _load_model_checkpoint(model, str(cfg.ckpt))
    model = model.to(model_device).eval()

    dataset_stats_path = _resolve_dataset_stats_path(cfg)
    dataset_stats = load_dataset_stats_from_json(str(dataset_stats_path))
    processor: FastWAMProcessor = instantiate(cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(dataset_stats)
    logging.info("Using dataset stats: %s", dataset_stats_path)

    action_horizon_cfg = cfg.EVALUATION.get("action_horizon", None)
    if action_horizon_cfg is None:
        action_horizon = int(cfg.data.train.num_frames) - 1
    else:
        action_horizon = int(action_horizon_cfg)
    if action_horizon <= 0:
        raise ValueError(f"EVALUATION.action_horizon must be positive, got {action_horizon}")

    video_size = cfg.data.train.get("video_size", [224, 224])
    if len(video_size) != 2:
        raise ValueError(f"data.train.video_size must be [H, W], got {video_size}")
    input_h = int(video_size[0])
    input_w = int(video_size[1])
    concat_multi_camera = cfg.data.train.get("concat_multi_camera", None)
    shape_meta_images = [meta["shape"] for meta in processor.shape_meta["images"]]

    local_log_dir = Path(cfg.EVALUATION.output_dir)
    local_log_dir.mkdir(parents=True, exist_ok=True)
    video_dir = local_log_dir / cfg.EVALUATION.task_suite_name / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    predicted_video_dir = local_log_dir / cfg.EVALUATION.task_suite_name / "predicted_videos"
    if bool(cfg.EVALUATION.get("visualize_future_video", False)):
        predicted_video_dir.mkdir(parents=True, exist_ok=True)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.EVALUATION.task_suite_name]()
    task = task_suite.get_task(cfg.EVALUATION.task_id)
    initial_states = task_suite.get_task_init_states(cfg.EVALUATION.task_id)

    # record actions (optional)
    record_actions = bool(cfg.EVALUATION.get("record_actions", False))
    action_records_dir = None
    if record_actions:
        action_records_dir = Path(cfg.EVALUATION.output_dir) / "action_records" / cfg.EVALUATION.task_suite_name
        action_records_dir.mkdir(parents=True, exist_ok=True)
        draft_action_records_dir = Path(cfg.EVALUATION.output_dir) / "draft_action_records" / cfg.EVALUATION.task_suite_name
        draft_action_records_dir.mkdir(parents=True, exist_ok=True)
        orig_action_records_dir = Path(cfg.EVALUATION.output_dir) / "orig_action_records" / cfg.EVALUATION.task_suite_name
        orig_action_records_dir.mkdir(parents=True, exist_ok=True)
    record_actions_bias = bool(cfg.EVALUATION.get("record_actions_bias", False))
    action_bias_records_dir = None
    if record_actions_bias:
        action_bias_records_dir = Path(cfg.EVALUATION.output_dir) / "action_bias_records" / cfg.EVALUATION.task_suite_name
        action_bias_records_dir.mkdir(parents=True, exist_ok=True)

    while len(initial_states) < int(cfg.EVALUATION.num_trials):
        initial_states.extend(initial_states[: (int(cfg.EVALUATION.num_trials) - len(initial_states))])

    results = {
        "task_suite": cfg.EVALUATION.task_suite_name,
        "task_id": cfg.EVALUATION.task_id,
        "task_description": None,
        "successes": 0,
        "total_episodes": int(cfg.EVALUATION.num_trials),
        "gpu_id": int(cfg.gpu_id),
        "success_episodes": [],
        "failure_episodes": [],
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "duration": 0,
    }

    logging.info("Running LIBERO evaluation with env_num=1")
    task_results = run_single_task(
        task=task,
        initial_states=initial_states,
        model=model,
        processor=processor,
        cfg=cfg,
        video_dir=video_dir,
        predicted_video_dir=predicted_video_dir,
        action_horizon=action_horizon,
        input_w=input_w,
        input_h=input_h,
        model_device=model_device,
    )
    results.update(task_results)

    results["duration"] = time.time() - start_time
    output_dir = Path(cfg.EVALUATION.output_dir) / cfg.EVALUATION.task_suite_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"gpu{cfg.gpu_id}_task{cfg.EVALUATION.task_id}_results.json"
    
    # record actions (optional)
    def save_actions(action_records, npz_path):
        # Convert action records to numpy arrays
        np_action_records = {str(trial_idx): np.array(actions) for trial_idx, actions in action_records.items()}
        np.savez(str(npz_path), **np_action_records)
        print(f"Saved action records to {npz_path}")
    if record_actions:
        action_records = results["action_records"]
        draft_action_records = results["draft_action_records"]
        orig_action_records = results["orig_action_records"]
        # Save action records as npz file
        npz_filename = f"task{cfg.EVALUATION.task_id}_actions.npz"
        npz_path = action_records_dir / npz_filename
        draft_npz_path = draft_action_records_dir / npz_filename
        orig_npz_path = orig_action_records_dir / npz_filename
        save_actions(action_records, npz_path)
        if len(draft_action_records) > 0:
            save_actions(draft_action_records, draft_npz_path)
        if len(orig_action_records) > 0:
            save_actions(orig_action_records, orig_npz_path)
    if record_actions_bias:
        action_bias_records = results["action_bias_records"]
        # Save action bias records as npz file
        npz_filename = f"task{cfg.EVALUATION.task_id}_actions_bias.npz"
        npz_path = action_bias_records_dir / npz_filename
        # Convert action bias records to numpy arrays
        np_action_bias_records = {str(trial_idx): np.array(actions_bias) for trial_idx, actions_bias in action_bias_records.items()}
        np.savez(str(npz_path), **np_action_bias_records)
        print(f"Saved action bias records to {npz_path}")
    
    del results["action_records"]
    del results["action_bias_records"]
    if 'draft_action_records' in results:
        del results['draft_action_records']
    if 'orig_action_records' in results:
        del results['orig_action_records']
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, cls=NumpyEncoder)

    print(
        f"Task {cfg.EVALUATION.task_id} completed: "
        f"{results['successes']}/{cfg.EVALUATION.num_trials} successes"
    )
    if results.get("future_video_psnr_mean") is not None:
        print(f"Task {cfg.EVALUATION.task_id} future-video PSNR mean: {results['future_video_psnr_mean']:.4f}")
    print(f"Time taken: {results['duration']:.2f} seconds")
    return results


if __name__ == "__main__":
    eval_single_process()
