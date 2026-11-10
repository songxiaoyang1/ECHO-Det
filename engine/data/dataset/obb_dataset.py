"""  
YOLO-OBB dataset support for first-stage oriented-box dataloading.
"""

import math
from pathlib import Path    
from typing import Dict, Mapping, Optional, Sequence

import torch    
import yaml    
from PIL import Image  
    
from ._dataset import DetDataset    
from ...core import register
  

@register()    
class OBBDetection(DetDataset):    
    __inject__ = ["transforms"]
    __share__ = ["num_classes", "angle_factor", "text_cache_file"]

    def __init__(
        self,
        img_folder,
        label_folder,
        transforms=None,
        data_yaml=None,
        ann_file=None,
        return_masks=False,
        names=None,
        label_format="yolo_obb",  
        normalized=True,  
        image_extensions=(".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"),   
        filter_empty=False,     
        num_classes=None,
        angle_factor=math.pi,     
        text_cache_file: Optional[str] = None,
    ):
        self.img_folder = str(img_folder)     
        self.label_folder = str(label_folder)     
        self._transforms = transforms
        self.data_yaml = data_yaml  
        self.ann_file = ann_file
        self.return_masks = return_masks
        self.label_format = label_format
        self.normalized = normalized     
        self.filter_empty = filter_empty 
        self.remap_mscoco_category = False
        self.num_classes = num_classes
        self.angle_factor = angle_factor    
        self._text_cache_file = text_cache_file     
        self._text_cache: Optional[Dict[str, torch.Tensor]] = None
        self._category2name = self._load_names(data_yaml, names)
        self.samples = self._build_samples(Path(img_folder), Path(label_folder), image_extensions)   
        self._validate_num_classes()

    def __len__(self):
        return len(self.samples)  

    def __getitem__(self, idx):
        image, target = self.load_item(idx)
        if self._transforms is not None:
            image, target, _ = self._transforms(image, target, self) 
        return image, target 

    @staticmethod
    def _normalize_names(names):   
        if names is None:
            return {}
        if isinstance(names, Mapping):     
            return {int(idx): str(name) for idx, name in names.items()}
        if isinstance(names, Sequence) and not isinstance(names, (str, bytes)):    
            return {idx: str(name) for idx, name in enumerate(names)}
        raise ValueError(f"names must be a mapping or sequence, got {type(names)!r}")
     
    def _load_names(self, data_yaml, names):     
        if names is not None:  
            return self._normalize_names(names)
        if data_yaml is None:   
            return {}    

        with open(data_yaml, "r", encoding="utf-8") as f:   
            data = yaml.safe_load(f) or {} 
        return self._normalize_names(data.get("names"))     
 
    def _build_samples(self, img_folder: Path, label_folder: Path, image_extensions): 
        image_extensions = {ext.lower() for ext in image_extensions}
        if not img_folder.exists():
            raise FileNotFoundError(f"OBB image folder does not exist: {img_folder}")     
        if not label_folder.exists():
            raise FileNotFoundError(f"OBB label folder does not exist: {label_folder}")

        samples = []
        for image_path in sorted(p for p in img_folder.rglob("*") if p.suffix.lower() in image_extensions):  
            rel = image_path.relative_to(img_folder)   
            label_path = label_folder / rel.with_suffix(".txt")
            if self.filter_empty and (not label_path.exists() or label_path.stat().st_size == 0):
                continue
            samples.append((image_path, label_path))

        if not samples:
            raise FileNotFoundError(f"No OBB images found in {img_folder}")
        return samples

    def _read_label_file(self, label_path: Path, width: int, height: int):
        if not label_path.exists() or label_path.stat().st_size == 0:
            return (
                torch.zeros((0,), dtype=torch.int64),     
                torch.zeros((0, 8), dtype=torch.float32),
            ) 
        if self.label_format != "yolo_obb":   
            raise ValueError(f"Unsupported OBB label_format={self.label_format!r}")    
  
        labels = []
        polygons = []     
        with open(label_path, "r", encoding="utf-8") as f:    
            for line_no, raw_line in enumerate(f, start=1):  
                line = raw_line.strip()
                if not line:   
                    continue    
                parts = line.split()    
                if len(parts) != 9:
                    raise ValueError(
                        f"Expected 9 columns in {label_path}:{line_no} for YOLO-OBB, got {len(parts)}"    
                    )
                labels.append(int(float(parts[0])))
                coords = [float(v) for v in parts[1:]] 
                if self.normalized:   
                    coords = [coord * (width if idx % 2 == 0 else height) for idx, coord in enumerate(coords)]  
                polygons.append(coords)     
     
        if not labels:    
            return (    
                torch.zeros((0,), dtype=torch.int64),    
                torch.zeros((0, 8), dtype=torch.float32),
            )
        return torch.tensor(labels, dtype=torch.int64), torch.tensor(polygons, dtype=torch.float32)
  
    @staticmethod
    def _polygon_area(polygons):     
        if polygons.numel() == 0:
            return torch.zeros((0,), dtype=torch.float32)
        points = polygons.reshape(-1, 4, 2)  
        x = points[..., 0]   
        y = points[..., 1]
        return 0.5 * torch.abs(    
            torch.sum(x * torch.roll(y, shifts=-1, dims=1) - y * torch.roll(x, shifts=-1, dims=1), dim=1)
        )

    def _validate_num_classes(self): 
        if self.num_classes is None:     
            return
        configured_num_classes = int(self.num_classes) 
        if self.category2name:
            expected_num_classes = max(self.category2name) + 1  
            if configured_num_classes != expected_num_classes:  
                raise ValueError(
                    "OBB num_classes mismatch: "
                    f"configured num_classes={configured_num_classes}; "    
                    f"expected_num_classes={expected_num_classes} (max(category_id)+1); "
                    f"categories={self.category2name}."
                )

    def load_item(self, idx):
        image_path, label_path = self.samples[idx]
        image = Image.open(image_path).convert("RGB")     
        width, height = image.size  
        labels, polygons = self._read_label_file(label_path, width, height)
        area = self._polygon_area(polygons)
        target = { 
            "image_id": torch.tensor([idx]),
            "idx": torch.tensor([idx]),    
            "labels": labels,  
            "obb_polygons": polygons,
            "area": area, 
            "orig_size": torch.tensor([width, height]),
            "size": torch.tensor([width, height]),  
        }  
        return image, target
     
    @property
    def categories(self):
        return [{"id": idx, "name": name} for idx, name in self.category2name.items()]

    @property 
    def category2name(self):
        return self._category2name

    @property
    def category2label(self):
        return {idx: idx for idx in self.category2name}
   
    @property   
    def label2category(self):
        return {idx: idx for idx in self.category2name}  

    @property 
    def text_list(self):
        return [name for _, name in sorted(self.category2name.items())]    
    
    @property  
    def detect_text_list(self):     
        return self.text_list

    @property 
    def text_cache(self) -> Dict[str, torch.Tensor]:
        if self._text_cache is None:
            if self._text_cache_file is None:     
                raise RuntimeError("OBBDetection.text_cache requires text_cache_file to be set.")
  
            from ...misc.ov_text_cache import load_text_cache_payload 
  
            payload = load_text_cache_payload(  
                torch.load(Path(self._text_cache_file), map_location="cpu"),
                self._text_cache_file,
            )
            categories = payload["categories"]
            text_feats = payload["text_feats"]    
            self._text_cache = {  
                category: text_feats[index] 
                for index, category in enumerate(categories)
            } 
   
        return self._text_cache
