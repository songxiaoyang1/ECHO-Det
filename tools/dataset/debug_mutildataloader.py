import argparse
import os
import sys
from pathlib import Path

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from engine.core import YAMLConfig
from engine.logger_module import get_logger
from engine.misc.visualizer import plot_sample

logger = get_logger(__name__)


def _normalize_single_channel_batch(batch: torch.Tensor) -> torch.Tensor:
    """Normalize each sample in a single-channel batch to [0, 1] for visualization."""
    if batch.ndim == 3:
        batch = batch.unsqueeze(1)

    batch = batch.float()
    b = batch.shape[0]
    flat = batch.view(b, -1)
    min_v = flat.min(dim=1).values.view(b, 1, 1, 1)
    max_v = flat.max(dim=1).values.view(b, 1, 1, 1)
    denom = (max_v - min_v).clamp(min=1e-6)
    return (batch - min_v) / denom


def _plot_with_category(samples, targets, dataset, out_path: Path):
    if dataset.remap_mscoco_category:
        plot_sample((samples, targets), dataset.category2name, out_path, dataset.label2category)
    else:
        plot_sample((samples, targets), dataset.category2name, out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default="configs/dataset/multimodal_rgb_npy_detection.yml")
    parser.add_argument("-n", "--num", type=int, default=20)
    parser.add_argument("-s", "--split", type=str, default="train", choices=["train", "val"])
    parser.add_argument("-e", "--epoch", type=int, default=10)
    parser.add_argument("-o", "--output", type=str, default="dataloader_output_multimodal")
    parser.add_argument("--rgb-key", type=str, default="rgb")
    parser.add_argument("--npy-key", type=str, default="npy")
    args = parser.parse_args()

    cfg = YAMLConfig(args.config)
    data_loader = cfg.train_dataloader if args.split == "train" else cfg.val_dataloader

    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)

    if hasattr(data_loader, "set_epoch"):
        data_loader.set_epoch(args.epoch)

    rgb_dir = output_dir / "rgb"
    npy_dir = output_dir / "npy"
    rgb_dir.mkdir(exist_ok=True)
    npy_dir.mkdir(exist_ok=True)

    for idx, (samples, targets) in enumerate(data_loader):
        if isinstance(samples, dict):
            rgb_batch = samples.get(args.rgb_key)
            npy_batch = samples.get(args.npy_key)

            if rgb_batch is not None:
                _plot_with_category(rgb_batch, targets, data_loader.dataset, rgb_dir / f"batch_{idx}.png")
            else:
                logger.warning(f"Missing key '{args.rgb_key}' in samples dict at batch {idx}.")

            if npy_batch is not None:
                npy_vis = _normalize_single_channel_batch(npy_batch)
                _plot_with_category(npy_vis, targets, data_loader.dataset, npy_dir / f"batch_{idx}.png")
            else:
                logger.warning(f"Missing key '{args.npy_key}' in samples dict at batch {idx}.")
        else:
            _plot_with_category(samples, targets, data_loader.dataset, output_dir / f"batch_{idx}.png")

        logger.info(f"index:{idx + 1} plot....")

        if (idx + 1) >= args.num:
            break

    logger.info("done...")
