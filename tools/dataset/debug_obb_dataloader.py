import argparse
import os
import sys
from pathlib import Path

import torch
from PIL import Image, ImageDraw

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from engine.core import YAMLConfig
from engine.logger_module import get_logger


logger = get_logger(__name__)
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def _to_pil_image(image_tensor):
    image = image_tensor.detach().cpu().float()
    if image.ndim != 3:
        raise ValueError(f"Expected CHW image tensor, got shape={tuple(image.shape)}")
    image = image * STD + MEAN
    image = image.clamp(0, 1)
    image = (image * 255).to(torch.uint8).permute(1, 2, 0).numpy()
    return Image.fromarray(image)


def _label_name(dataset, label):
    category2name = getattr(dataset, "category2name", {})
    label2category = getattr(dataset, "label2category", {})
    category = label2category.get(label, label) if isinstance(label2category, dict) else label
    return category2name.get(category, str(label)) if isinstance(category2name, dict) else str(label)


def _validate_target(target):
    assert "labels" in target
    assert "boxes" in target
    assert "obb_polygons" in target
    assert target["boxes"].shape[-1] == 5
    assert torch.isfinite(target["boxes"]).all()


def _draw_target(image, target, dataset):
    draw = ImageDraw.Draw(image)
    polygons = target["obb_polygons"].detach().cpu()
    labels = target["labels"].detach().cpu().tolist()
    for polygon, label in zip(polygons, labels):
        points = [(float(polygon[i]), float(polygon[i + 1])) for i in range(0, 8, 2)]
        draw.line(points + [points[0]], fill=(255, 220, 0), width=2)
        draw.text(points[0], _label_name(dataset, int(label)), fill=(255, 255, 255))
    return image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default="configs/dataset/uavrob_obb_detection.yml")
    parser.add_argument("-n", "--num", type=int, default=20)
    parser.add_argument("-s", "--split", default="train", choices=["train", "val"])
    parser.add_argument("-o", "--output-dir", default="dataloader_obb_output")
    args = parser.parse_args()

    cfg = YAMLConfig(args.config)
    data_loader = cfg.train_dataloader if args.split == "train" else cfg.val_dataloader
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_loader.set_epoch(10)
    for idx, (samples, targets) in enumerate(data_loader):
        for target in targets:
            _validate_target(target)

        canvas = None
        for sample_idx, target in enumerate(targets):
            image = _draw_target(_to_pil_image(samples[sample_idx]), target, data_loader.dataset)
            if canvas is None:
                canvas = Image.new("RGB", (image.width * len(targets), image.height))
            canvas.paste(image, (sample_idx * image.width, 0))

        canvas.save(output_dir / f"batch_{idx}.png")
        logger.info(f"index:{idx + 1} plot....")
        if (idx + 1) >= args.num:
            break

    logger.info("done...")


if __name__ == "__main__":
    main()
