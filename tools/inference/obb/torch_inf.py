"""
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
import tqdm
import yaml
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
from engine.core import YAMLConfig
from engine.prune.checkpoint import load_model_checkpoint
from engine.extre_module.utils import increment_path
from engine.logger_module import get_logger
from tools.inference.utils import draw

logger = get_logger(__name__)

RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov"}
DEFAULT_INPUT_SIZE = (640, 640)
DEFAULT_DOTA15_NAMES = [
    "plane",
    "ship",
    "storage tank",
    "baseball diamond",
    "tennis court",
    "basketball court",
    "ground track field",
    "harbor",
    "bridge",
    "large vehicle",
    "small vehicle",
    "helicopter",
    "roundabout",
    "soccer ball field",
    "swimming pool",
    "container crane",
]
CLASS_NAME = DEFAULT_DOTA15_NAMES


def get_device(device_arg):
    if device_arg == "cpu" or not torch.cuda.is_available():
        return torch.device("cpu")
    cuda_index = str(device_arg).split(",")[0]
    return torch.device(f"cuda:{cuda_index}")


def is_image_file(path):
    return Path(path).suffix.lower() in IMAGE_EXTENSIONS


def is_video_file(path):
    return Path(path).suffix.lower() in VIDEO_EXTENSIONS


def get_eval_size(yaml_cfg):
    size = yaml_cfg.get("eval_spatial_size", DEFAULT_INPUT_SIZE)
    if isinstance(size, int):
        return (int(size), int(size))
    if len(size) != 2:
        raise ValueError(f"eval_spatial_size must be int or pair, got: {size}")
    return (int(size[0]), int(size[1]))


def _val_transform_ops(yaml_cfg):
    dataset_cfg = yaml_cfg.get("val_dataloader", {}).get("dataset", {})
    transforms_cfg = dataset_cfg.get("transforms", {})
    return transforms_cfg.get("ops", [])


def get_normalize_transform(yaml_cfg):
    for op in _val_transform_ops(yaml_cfg):
        if op.get("type") == "Normalize":
            return T.Normalize(mean=op["mean"], std=op["std"])
    return None


def resolve_class_names(checkpoint, yaml_cfg):
    ckpt_names = checkpoint.get("name") if isinstance(checkpoint, dict) else None
    if ckpt_names:
        if isinstance(ckpt_names, dict):
            return [ckpt_names[k] for k in sorted(ckpt_names)]
        if isinstance(ckpt_names, (list, tuple)):
            return list(ckpt_names)
        return [str(ckpt_names)]

    data_yaml = yaml_cfg.get("val_dataloader", {}).get("dataset", {}).get("data_yaml")
    if data_yaml and Path(data_yaml).exists():
        with open(data_yaml, "r", encoding="utf-8") as f:
            names = yaml.safe_load(f).get("names", None)
        if isinstance(names, dict):
            return [names[k] for k in sorted(names)]
        if isinstance(names, list):
            return names

    return DEFAULT_DOTA15_NAMES


def letterbox_image(image, size=DEFAULT_INPUT_SIZE, fill=0):
    target_w, target_h = int(size[0]), int(size[1])
    old_w, old_h = image.size
    scale = min(target_w / old_w, target_h / old_h)
    new_w = max(1, int(round(old_w * scale)))
    new_h = max(1, int(round(old_h * scale)))

    resized = image.resize((new_w, new_h), Image.BILINEAR)
    canvas = Image.new("RGB", (target_w, target_h), color=(fill, fill, fill))
    canvas.paste(resized, (0, 0))

    resize_pad = {
        "padding": torch.tensor([[0.0, 0.0, float(target_w - new_w), float(target_h - new_h)]], dtype=torch.float32),
        "size": torch.tensor([[float(target_w), float(target_h)]], dtype=torch.float32),
    }
    return canvas, resize_pad


def move_resize_pad_to_device(resize_pad, device):
    return {key: value.to(device) for key, value in resize_pad.items()}


def build_image_sample(image, device, input_size, normalize=None):
    image = image.convert("RGB")
    width, height = image.size
    orig_size = torch.tensor([[width, height]], dtype=torch.float32, device=device)
    resized, resize_pad = letterbox_image(image, input_size)

    transforms = [T.ToTensor()]
    if normalize is not None:
        transforms.append(normalize)
    image_data = T.Compose(transforms)(resized).unsqueeze(0).to(device)

    return image, image_data, orig_size, move_resize_pad_to_device(resize_pad, device)


def process_image(model, device, file_path, output_path, thrh, input_size, normalize=None):
    image = Image.open(file_path).convert("RGB")
    image, image_data, orig_size, resize_pad = build_image_sample(image, device, input_size, normalize)

    with torch.no_grad():
        labels, boxes, scores = model(image_data, orig_size, resize_pad)

    image = draw([image], labels, boxes, scores, thrh=thrh, class_name=CLASS_NAME, box_format="xywhr")
    image.save(output_path / Path(file_path).name)


def process_video(model, device, file_path, output_path, thrh, input_size, normalize=None):
    cap = cv2.VideoCapture(str(file_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path / Path(file_path).name), fourcc, fps, (orig_w, orig_h))

    if cap.isOpened():
        for _ in tqdm.tqdm(range(total_frames), desc="Processing video frames..."):
            ret, frame = cap.read()
            if not ret:
                break

            frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frame_pil, image_data, orig_size, resize_pad = build_image_sample(frame_pil, device, input_size, normalize)

            with torch.no_grad():
                labels, boxes, scores = model(image_data, orig_size, resize_pad)
            frame_pil = draw([frame_pil], labels, boxes, scores, thrh=thrh, class_name=CLASS_NAME, box_format="xywhr")
            out.write(cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR))

    cap.release()
    out.release()


def main(args):
    global CLASS_NAME
    cfg = YAMLConfig(args.config, resume=args.resume)

    output_path = increment_path(args.output)
    logger.info(RED + f"output_dir:{str(output_path)}" + RESET)
    output_path.mkdir(parents=True, exist_ok=True)

    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

    if not args.resume:
        raise AttributeError("Only support resume to load model.state_dict by now.")

    model, checkpoint = load_model_checkpoint(args.resume, model=cfg.model)
    CLASS_NAME = resolve_class_names(checkpoint, cfg.yaml_cfg)
    cfg.model = model

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig_target_sizes, resize_pad):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes, resize_pad=resize_pad)
            return outputs

    device = get_device(args.device)
    model = Model().to(device).eval()
    input_size = get_eval_size(cfg.yaml_cfg)
    normalize = get_normalize_transform(cfg.yaml_cfg)

    file_path = Path(args.input)
    if file_path.is_dir():
        files = sorted(path for path in file_path.iterdir() if path.is_file() and (is_image_file(path) or is_video_file(path)))
        for path in tqdm.tqdm(files, desc=f"Process {file_path} folder"):
            if is_image_file(path):
                process_image(model, device, path, output_path, args.thrh, input_size, normalize)
            elif is_video_file(path):
                process_video(model, device, path, output_path, args.thrh, input_size, normalize)
    elif is_image_file(file_path):
        process_image(model, device, file_path, output_path, args.thrh, input_size, normalize)
        logger.info("Image processing complete.")
    elif is_video_file(file_path):
        process_video(model, device, file_path, output_path, args.thrh, input_size, normalize)
        logger.info("Video processing complete.")
    else:
        raise ValueError(f"Unsupported input path: {file_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True)
    parser.add_argument("-r", "--resume", type=str, required=True)
    parser.add_argument("-i", "--input", type=str, required=True)
    parser.add_argument("-o", "--output", type=str, default="inference_results/exp")
    parser.add_argument("-t", "--thrh", type=float, default=0.2)
    parser.add_argument("-d", "--device", type=str, default="0")
    args = parser.parse_args()
    main(args)
