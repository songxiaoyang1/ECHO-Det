"""
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import tqdm
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
from engine.core import YAMLConfig
from engine.prune.checkpoint import load_model_checkpoint
from engine.extre_module.utils import increment_path
from engine.logger_module import get_logger
from engine.misc.modality_utils import normalize_tensor_minmax_per_sample
from tools.inference.utils import draw

logger = get_logger(__name__)

RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
CLASS_NAME = None

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
NPY_EXTENSIONS = (".npy", ".npz")
INFERENCE_SIZE = (640, 640)


def is_image_file(path):
    return Path(path).suffix.lower() in IMAGE_EXTENSIONS


def is_modality_file(path, modality_format="npy"):
    extensions = NPY_EXTENSIONS if modality_format == "npy" else IMAGE_EXTENSIONS
    return Path(path).suffix.lower() in extensions


def resolve_modality_path(rgb_path, modality_dir, modality_format="npy"):
    rgb_path = Path(rgb_path)
    modality_dir = Path(modality_dir)
    extensions = NPY_EXTENSIONS if modality_format == "npy" else IMAGE_EXTENSIONS
    candidates = [modality_dir / f"{rgb_path.stem}{ext}" for ext in extensions]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    tried = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Missing modality file for {rgb_path}. Tried: {tried}")


def iter_multimodal_inputs(rgb_input, modality_input, modality_format="npy"):
    if modality_format not in {"npy", "image"}:
        raise ValueError("modality_format must be one of {'npy', 'image'}")

    rgb_path = Path(rgb_input)
    modality_path = Path(modality_input)
    modality_label = "NPY" if modality_format == "npy" else "image modality"

    if not rgb_path.exists():
        raise FileNotFoundError(f"RGB input does not exist: {rgb_path}")
    if not modality_path.exists():
        raise FileNotFoundError(f"{modality_label} input does not exist: {modality_path}")

    if rgb_path.is_dir():
        if modality_path.is_file():
            raise ValueError(f"When RGB input is a directory, {modality_label} input must also be a directory.")
        rgb_files = sorted(path for path in rgb_path.iterdir() if path.is_file() and is_image_file(path))
        if not rgb_files:
            raise FileNotFoundError(f"No image files found in RGB directory: {rgb_path}")
        return [(path, resolve_modality_path(path, modality_path, modality_format)) for path in rgb_files]

    if not is_image_file(rgb_path):
        raise ValueError(f"RGB input must be an image file or directory, got: {rgb_path}")

    if modality_path.is_dir():
        return [(rgb_path, resolve_modality_path(rgb_path, modality_path, modality_format))]

    if not is_modality_file(modality_path, modality_format):
        expected = ".npy/.npz" if modality_format == "npy" else "/".join(IMAGE_EXTENSIONS)
        raise ValueError(f"{modality_label} input must be a {expected} file or directory, got: {modality_path}")

    return [(rgb_path, modality_path)]


def load_npy_tensor(npy_path):
    raw = np.load(npy_path, allow_pickle=True)
    if type(raw) is np.lib.npyio.NpzFile:
        try:
            keys = list(raw.files)
            if "arr_0" in keys:
                npy = raw["arr_0"]
            elif len(keys) == 1:
                npy = raw[keys[0]]
            else:
                raise ValueError(f"NPZ file with multiple arrays must contain key 'arr_0', got keys={keys} for {npy_path}")
        finally:
            raw.close()
    else:
        npy = raw

    npy = np.squeeze(np.asarray(npy))
    if npy.ndim == 2:
        tensor = torch.from_numpy(npy.astype(np.float32, copy=False)).unsqueeze(0)
    elif npy.ndim == 3 and npy.shape[0] == 1:
        tensor = torch.from_numpy(npy.astype(np.float32, copy=False))
    elif npy.ndim == 3 and npy.shape[-1] == 1:
        tensor = torch.from_numpy(npy[..., 0].astype(np.float32, copy=False)).unsqueeze(0)
    else:
        raise ValueError(f"NPY modality must be 2D or single-channel, got shape {tuple(npy.shape)} for {npy_path}")

    return tensor


def load_image_modality(image_path):
    return Image.open(image_path).convert("L")


def build_multimodal_sample(rgb_path, modality_path, device, modality_format="npy"):
    im_pil = Image.open(rgb_path).convert("RGB")
    w, h = im_pil.size
    orig_size = torch.tensor([[w, h]], device=device)

    rgb_transforms = T.Compose([
        T.Resize(INFERENCE_SIZE),
        T.ToTensor(),
    ])
    rgb_data = rgb_transforms(im_pil).unsqueeze(0).to(device)

    if modality_format == "image":
        modality = load_image_modality(modality_path)
        if modality.size != im_pil.size:
            raise ValueError(f"Image modality/RGB size mismatch for {rgb_path}: modality={modality.size}, rgb={im_pil.size}")
        modality_transforms = T.Compose([
            T.Resize(INFERENCE_SIZE),
            T.ToTensor(),
        ])
        npy_data = modality_transforms(modality).unsqueeze(0)
    else:
        npy_tensor = load_npy_tensor(modality_path)
        npy_h, npy_w = npy_tensor.shape[-2:]
        if (npy_h, npy_w) != (h, w):
            raise ValueError(f"NPY/RGB size mismatch for {rgb_path}: npy=({npy_h}, {npy_w}), rgb=({h}, {w})")
        npy_data = F.interpolate(
            npy_tensor.unsqueeze(0),
            size=INFERENCE_SIZE,
            mode="bilinear",
            align_corners=False,
        )

    npy_data = normalize_tensor_minmax_per_sample(npy_data).to(device)

    return im_pil, {"rgb": rgb_data, "npy": npy_data}, orig_size


def process_image(model, device, rgb_path, modality_path, output_path, thrh, modality_format):
    im_pil, sample, orig_size = build_multimodal_sample(rgb_path, modality_path, device, modality_format=modality_format)

    with torch.no_grad():
        output = model(sample, orig_size)
    if len(output) == 3:
        labels, boxes, scores = output
        masks = None
    elif len(output) == 4:
        labels, boxes, scores, masks = output
    else:
        raise ValueError(f"Expected postprocessor output with 3 or 4 items, got {len(output)}")

    box_format = "xywhr" if boxes.shape[-1] == 5 else "xyxy"
    im_pil = draw([im_pil], labels, boxes, scores, masks=masks, thrh=thrh, class_name=CLASS_NAME, box_format=box_format)
    im_pil.save(output_path / Path(rgb_path).name)


def get_device(device_arg):
    if device_arg == "cpu" or not torch.cuda.is_available():
        return torch.device("cpu")

    cuda_index = str(device_arg).split(",")[0]
    return torch.device(f"cuda:{cuda_index}")


def main(args):
    """Main function"""
    global CLASS_NAME
    cfg = YAMLConfig(args.config, resume=args.resume)

    output_path = increment_path(args.output)
    logger.info(RED + f"output_dir:{str(output_path)}" + RESET)
    output_path.mkdir(parents=True, exist_ok=True)

    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

    if args.resume:
        model, checkpoint = load_model_checkpoint(args.resume, model=cfg.model)
        if checkpoint.get("name", None) != None:
            CLASS_NAME = checkpoint["name"]
    else:
        raise AttributeError("Only support resume to load model.state_dict by now.")

    cfg.model = model

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, sample, orig_target_sizes):
            outputs = self.model(sample)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs

    device = get_device(args.device)
    model = Model().to(device).eval()

    modality_format = "image" if args.input_png is not None else "npy"
    modality_input = args.input_png if modality_format == "image" else args.input_npy
    input_pairs = iter_multimodal_inputs(args.input_rgb, modality_input, modality_format=modality_format)
    for rgb_path, modality_path in tqdm.tqdm(input_pairs, desc="Processing multimodal images"):
        process_image(model, device, rgb_path, modality_path, output_path, args.thrh, modality_format)

    logger.info("Multimodal image processing complete.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True)
    parser.add_argument("-r", "--resume", type=str, required=True)
    parser.add_argument("-i-rgb", "--input-rgb", type=str, required=True)
    modality_group = parser.add_mutually_exclusive_group(required=True)
    modality_group.add_argument("-i-npy", "--input-npy", type=str)
    modality_group.add_argument("-i-png", "--input-png", type=str)
    parser.add_argument("-o", "--output", type=str, default="inference_results/exp")
    parser.add_argument("-t", "--thrh", type=float, default=0.2)
    parser.add_argument("-d", "--device", type=str, default="0")
    args = parser.parse_args()
    main(args)
