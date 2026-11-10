"""YAML-driven preprocessing helpers for visualization scripts."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from PIL import Image

import engine.data.transforms  # noqa: F401 - registers transform classes.
from engine.core.workspace import GLOBAL_CONFIG
from engine.data._misc import convert_to_tv_tensor
from engine.solver.sample_adapter import move_samples_to_device


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
NPY_EXTENSIONS = (".npy", ".npz")
OBB_TRANSFORM_NAMES = {"OBBResize", "OBBSanitize", "ConvertOBBBoxes", "MultimodalOBBCompose"}
MULTIMODAL_COMPOSE_NAMES = {"MultimodalCompose", "MultimodalOBBCompose"}


@dataclass(frozen=True)
class VisualInputBundle:
    vis_image: Image.Image
    samples: Any
    model_input: Any
    orig_size: torch.Tensor
    resize_pad: dict[str, torch.Tensor] | None
    box_format: str
    is_seg: bool
    is_obb: bool
    is_multimodal: bool


@dataclass(frozen=True)
class VisualPreprocessor:
    cfg: Any
    transform_cfg: dict[str, Any]
    transform: Any
    is_seg: bool
    is_obb: bool
    is_multimodal: bool

    def prepare(
        self,
        image_path: str | Path,
        device: torch.device | str,
        modality_path: str | Path | None = None,
        model_input_key: str = "rgb",
    ) -> VisualInputBundle:
        return prepare_visual_input(
            self,
            image_path,
            device=device,
            modality_path=modality_path,
            model_input_key=model_input_key,
        )


def get_val_transform_cfg(cfg: Any) -> dict[str, Any]:
    val_dataloader = cfg.yaml_cfg.get("val_dataloader") if hasattr(cfg, "yaml_cfg") else None
    if not isinstance(val_dataloader, dict):
        raise KeyError("Config does not contain val_dataloader.")

    dataset_cfg = val_dataloader.get("dataset")
    if isinstance(dataset_cfg, dict) and "transforms" in dataset_cfg:
        return dataset_cfg["transforms"]

    if "transforms" in val_dataloader:
        return val_dataloader["transforms"]

    raise KeyError("Config val_dataloader must contain dataset.transforms or transforms.")


def build_transform_from_cfg(transform_cfg: dict[str, Any]):
    transform_cfg = copy.deepcopy(transform_cfg)
    transform_type = transform_cfg.pop("type", None)
    if not transform_type:
        raise KeyError("Transform config must contain a 'type' key.")
    if transform_type not in GLOBAL_CONFIG:
        raise KeyError(f"Transform type '{transform_type}' is not registered.")

    schema = GLOBAL_CONFIG[transform_type]
    transform_cls = getattr(schema["_pymodule"], schema["_name"])
    return transform_cls(**transform_cfg)


def build_visual_preprocessor(cfg: Any) -> VisualPreprocessor:
    transform_cfg = copy.deepcopy(get_val_transform_cfg(cfg))
    return VisualPreprocessor(
        cfg=cfg,
        transform_cfg=transform_cfg,
        transform=build_transform_from_cfg(transform_cfg),
        is_seg=is_seg_cfg(cfg),
        is_obb=is_obb_cfg(transform_cfg),
        is_multimodal=is_multimodal_cfg(transform_cfg),
    )


def is_multimodal_cfg(transform_cfg: dict[str, Any]) -> bool:
    return any(name in MULTIMODAL_COMPOSE_NAMES for name in _iter_transform_names(transform_cfg))


def is_obb_cfg(transform_cfg: dict[str, Any]) -> bool:
    return any(name in OBB_TRANSFORM_NAMES for name in _iter_transform_names(transform_cfg))


def is_seg_cfg(cfg: Any) -> bool:
    yaml_cfg = cfg.yaml_cfg if hasattr(cfg, "yaml_cfg") else cfg
    postprocessor = yaml_cfg.get("postprocessor")
    if postprocessor == "SegPostProcessor":
        return True
    if isinstance(postprocessor, dict) and postprocessor.get("type") == "SegPostProcessor":
        return True

    evaluator = yaml_cfg.get("evaluator", {})
    if isinstance(evaluator, dict) and "segm" in evaluator.get("iou_types", []):
        return True
    return False


def prepare_visual_input(
    cfg_or_preprocessor: Any,
    image_path: str | Path,
    device: torch.device | str,
    modality_path: str | Path | None = None,
    model_input_key: str = "rgb",
) -> VisualInputBundle:
    preprocessor = (
        cfg_or_preprocessor
        if isinstance(cfg_or_preprocessor, VisualPreprocessor)
        else build_visual_preprocessor(cfg_or_preprocessor)
    )
    device = torch.device(device)
    image_path = Path(image_path)

    vis_image = Image.open(image_path).convert("RGB")
    width, height = vis_image.size
    target = _make_target(width, height, is_obb=preprocessor.is_obb)

    sample: Any = vis_image
    if preprocessor.is_multimodal:
        dataset_cfg = _get_val_dataset_cfg(preprocessor.cfg)
        resolved_modality_path = resolve_modality_path(
            preprocessor.cfg,
            image_path,
            modality_path=modality_path,
        )
        sample = {
            "rgb": vis_image.copy(),
            "npy": load_second_modality(resolved_modality_path, dataset_cfg=dataset_cfg),
        }

    dataset = SimpleNamespace(epoch=0)
    transformed = preprocessor.transform(sample, target, dataset)
    sample, target = _unpack_transform_output(transformed)

    resize_pad = _extract_resize_pad(target, device)
    samples = move_samples_to_device(_batch_sample(sample), device)
    model_input = samples if preprocessor.is_multimodal else samples
    if preprocessor.is_multimodal and model_input_key != "rgb":
        model_input = samples

    return VisualInputBundle(
        vis_image=vis_image,
        samples=samples,
        model_input=model_input,
        orig_size=torch.tensor([[width, height]], dtype=torch.int64, device=device),
        resize_pad=resize_pad,
        box_format="xywhr" if preprocessor.is_obb else "xyxy",
        is_seg=preprocessor.is_seg,
        is_obb=preprocessor.is_obb,
        is_multimodal=preprocessor.is_multimodal,
    )


def select_cam_input(samples: Any, key: str = "rgb") -> torch.Tensor:
    if torch.is_tensor(samples):
        return samples
    if isinstance(samples, dict):
        if key not in samples:
            raise KeyError(f"Missing CAM input key '{key}'. Available sample keys: {list(samples.keys())}")
        value = samples[key]
        if not torch.is_tensor(value):
            raise TypeError(f"CAM input samples['{key}'] must be a Tensor, got {type(value)}")
        return value
    raise TypeError(f"Unsupported samples type for CAM input: {type(samples)}")


def resolve_modality_path(
    cfg: Any,
    image_path: str | Path,
    modality_path: str | Path | None = None,
) -> Path:
    image_path = Path(image_path)
    dataset_cfg = _get_val_dataset_cfg(cfg)
    modality_format = _get_modality_format(dataset_cfg)

    if modality_path is not None:
        root = Path(modality_path)
        if root.is_file():
            return root
        if root.is_dir():
            return _match_modality_file(root, image_path, dataset_cfg, modality_format)
        raise FileNotFoundError(f"modality_path does not exist: {root}")

    root_key = "modality_folder" if modality_format == "image" else "npy_folder"
    root = dataset_cfg.get(root_key)
    if root is None and modality_format == "image":
        root = dataset_cfg.get("npy_folder")
    if root is None:
        raise ValueError(
            "This multimodal config requires a second modality. Provide modality_path or configure "
            "val_dataloader.dataset.npy_folder/modality_folder."
        )

    root = Path(root)
    if root.is_file():
        return root
    if root.is_dir():
        return _match_modality_file(root, image_path, dataset_cfg, modality_format)

    raise FileNotFoundError(
        f"Could not infer second modality for {image_path}. Inferred {root_key}={root}, but it does not exist. "
        "Pass modality_path explicitly."
    )


def load_second_modality(path: str | Path, dataset_cfg: dict[str, Any] | None = None):
    path = Path(path)
    dataset_cfg = dataset_cfg or {}
    modality_format = _get_modality_format(dataset_cfg)
    if modality_format == "image" or path.suffix.lower() in IMAGE_EXTENSIONS:
        return Image.open(path).convert("L")

    return _load_npy_tensor(path)


def _iter_transform_names(transform_cfg: Any):
    if isinstance(transform_cfg, dict):
        transform_type = transform_cfg.get("type")
        if transform_type:
            yield transform_type
        for op in transform_cfg.get("ops", []) or []:
            yield from _iter_transform_names(op)
    elif isinstance(transform_cfg, list):
        for item in transform_cfg:
            yield from _iter_transform_names(item)


def _get_val_dataset_cfg(cfg: Any) -> dict[str, Any]:
    yaml_cfg = cfg.yaml_cfg if hasattr(cfg, "yaml_cfg") else cfg
    val_dataloader = yaml_cfg.get("val_dataloader", {})
    dataset_cfg = val_dataloader.get("dataset", {})
    return dataset_cfg if isinstance(dataset_cfg, dict) else {}


def _get_modality_format(dataset_cfg: dict[str, Any]) -> str:
    if dataset_cfg.get("modality_format") in {"image", "npy"}:
        return dataset_cfg["modality_format"]
    if dataset_cfg.get("modality_folder") and not dataset_cfg.get("npy_folder"):
        return "image"
    return "npy"


def _make_target(width: int, height: int, is_obb: bool) -> dict[str, torch.Tensor]:
    target = {
        "image_id": torch.tensor([0], dtype=torch.int64),
        "idx": torch.tensor([0], dtype=torch.int64),
        "orig_size": torch.tensor([width, height], dtype=torch.int64),
        "size": torch.tensor([width, height], dtype=torch.int64),
        "labels": torch.zeros((0,), dtype=torch.int64),
        "boxes": convert_to_tv_tensor(
            torch.zeros((0, 4), dtype=torch.float32),
            key="boxes",
            spatial_size=(height, width),
        ),
        "area": torch.zeros((0,), dtype=torch.float32),
        "iscrowd": torch.zeros((0,), dtype=torch.int64),
    }
    if is_obb:
        target["obb_polygons"] = torch.zeros((0, 8), dtype=torch.float32)
    return target


def _unpack_transform_output(transformed: Any):
    if isinstance(transformed, tuple):
        if len(transformed) < 2:
            raise ValueError("Validation transform returned a tuple with fewer than 2 values.")
        return transformed[0], transformed[1]
    return transformed, {}


def _batch_sample(sample: Any):
    if torch.is_tensor(sample):
        return sample.unsqueeze(0)
    if isinstance(sample, dict):
        return {key: _batch_sample(value) for key, value in sample.items()}
    raise TypeError(f"Validation transform must return a Tensor or dict of Tensors, got {type(sample)}")


def _extract_resize_pad(target: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor] | None:
    if not isinstance(target, dict):
        return None
    padding = target.get("padding")
    size = target.get("resize_pad_size")
    if padding is None or size is None:
        return None
    return {
        "padding": torch.as_tensor(padding, dtype=torch.float32, device=device).reshape(1, 4),
        "size": torch.as_tensor(size, dtype=torch.float32, device=device).reshape(1, 2),
    }


def _match_modality_file(
    modality_root: Path,
    image_path: Path,
    dataset_cfg: dict[str, Any],
    modality_format: str,
) -> Path:
    extensions = IMAGE_EXTENSIONS if modality_format == "image" else NPY_EXTENSIONS
    candidates = []

    img_folder = dataset_cfg.get("img_folder")
    if img_folder:
        try:
            rel_path = image_path.resolve().relative_to(Path(img_folder).resolve())
            candidates.extend(_path_with_extensions(modality_root / rel_path, extensions))
        except ValueError:
            pass

    candidates.extend(_path_with_extensions(modality_root / image_path.name, extensions))
    candidates.extend(_path_with_extensions(modality_root / image_path.stem, extensions))

    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists():
            return candidate

    tried = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"Could not find matching second modality for {image_path}. Tried: {tried}. "
        "Pass modality_path explicitly if files do not share the RGB stem."
    )


def _path_with_extensions(path: Path, extensions: tuple[str, ...]) -> list[Path]:
    if path.suffix.lower() in extensions:
        return [path]
    return [path.with_suffix(ext) for ext in extensions]


def _load_npy_tensor(path: Path) -> torch.Tensor:
    raw = np.load(path, allow_pickle=True)
    if type(raw) is np.lib.npyio.NpzFile:
        try:
            keys = list(raw.files)
            if "arr_0" in keys:
                array = raw["arr_0"]
            elif len(keys) == 1:
                array = raw[keys[0]]
            else:
                raise ValueError(f"NPZ file with multiple arrays must contain key 'arr_0', got keys={keys} for {path}")
        finally:
            raw.close()
    else:
        array = raw

    array = np.squeeze(np.asarray(array))
    if array.ndim == 2:
        tensor = torch.from_numpy(array.astype(np.float32, copy=False)).unsqueeze(0)
    elif array.ndim == 3 and array.shape[0] == 1:
        tensor = torch.from_numpy(array.astype(np.float32, copy=False))
    elif array.ndim == 3 and array.shape[-1] == 1:
        tensor = torch.from_numpy(array[..., 0].astype(np.float32, copy=False)).unsqueeze(0)
    else:
        raise ValueError(f"NPY/NPZ modality must be 2D or single-channel, got shape {tuple(array.shape)} for {path}")
    return tensor
