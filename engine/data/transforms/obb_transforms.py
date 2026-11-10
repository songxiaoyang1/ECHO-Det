import math
import random
from typing import Any, Dict, Tuple     

import cv2 
import numpy as np
import torch    
import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as F
from PIL import Image
from torchvision.transforms.v2 import InterpolationMode    
    
from ...core import register  

     
OBB_INSTANCE_CAT_KEYS = {"labels", "obb_polygons", "area", "iscrowd"}


def _get_image_size(image):
    if isinstance(image, Image.Image):
        return image.size    
    height, width = image.shape[-2:]     
    return width, height  

    
def _clone_target(target):
    return {key: value.clone() if isinstance(value, torch.Tensor) else value for key, value in target.items()}

 
def _clone_image(image):   
    if isinstance(image, Image.Image):  
        return image.copy()     
    return image.clone() if hasattr(image, "clone") else image


def _normalize_size(size):
    if isinstance(size, int):
        return (size, size)
    if len(size) != 2:
        raise ValueError(f"size must be int or pair, got {size}")
    return int(size[0]), int(size[1])  

    
def _pack_outputs(image, target, rest):  
    if rest:   
        return (image, target, *rest)    
    return image, target     

 
def _unpack_inputs(inputs):
    if len(inputs) == 1 and isinstance(inputs[0], (tuple, list)):
        sample = tuple(inputs[0]) 
    else:    
        sample = inputs  
    if len(sample) < 2:    
        raise ValueError("OBB transforms expect at least image and target inputs")
    return sample[0], sample[1], sample[2:]     
     

def polygon_area(polygons):     
    if polygons.numel() == 0:    
        return torch.zeros((0,), dtype=torch.float32, device=polygons.device)
    points = polygons.reshape(-1, 4, 2)    
    x = points[..., 0]  
    y = points[..., 1]   
    return 0.5 * torch.abs(    
        torch.sum(x * torch.roll(y, -1, dims=1) - y * torch.roll(x, -1, dims=1), dim=1)  
    )  
   

def _collect_mosaic_source_image_ids(targets):
    image_ids = []   
    for target in targets: 
        if "image_id" not in target:  
            continue
        image_id = target["image_id"]   
        if isinstance(image_id, torch.Tensor):
            image_ids.append(image_id.reshape(-1))    
        else:
            image_ids.append(torch.as_tensor([image_id]))    
    if not image_ids:
        return None   
    return torch.cat(image_ids, dim=0) 
 

def merge_obb_mosaic_targets(targets, placement_offsets, mosaic_height, mosaic_width):    
    adjusted_targets = []    
    for idx, target in enumerate(targets):
        target = _clone_target(target)
        x_offset, y_offset = placement_offsets[idx][:2]  

        if "obb_polygons" in target: 
            offset = torch.as_tensor(   
                [x_offset, y_offset],     
                dtype=target["obb_polygons"].dtype,   
                device=target["obb_polygons"].device,
            ).repeat(4)
            target["obb_polygons"] = target["obb_polygons"] + offset    
            target["area"] = polygon_area(target["obb_polygons"])

        adjusted_targets.append(target) 

    primary_target = adjusted_targets[0]
    merged_target = {}     
    
    for key in primary_target:   
        if key == "boxes":
            continue
        if key in OBB_INSTANCE_CAT_KEYS:
            values = [target[key] for target in adjusted_targets if key in target]     
            if values:
                merged_target[key] = torch.cat(values, dim=0)
        else:
            merged_target[key] = ( 
                primary_target[key].clone()
                if isinstance(primary_target[key], torch.Tensor)   
                else primary_target[key]   
            )  

    source_image_ids = _collect_mosaic_source_image_ids(adjusted_targets) 
    if source_image_ids is not None:
        merged_target["mosaic_source_image_ids"] = source_image_ids
     
    merged_target["orig_size"] = torch.tensor([mosaic_width, mosaic_height], dtype=torch.int64)
    merged_target["size"] = torch.tensor([mosaic_width, mosaic_height], dtype=torch.int64)
    return merged_target     

 
@register()     
class OBBMosaic:    
    def __init__(
        self,
        output_size=320, 
        max_size=None,
        probability=1.0,  
        fill_value=0,
        use_cache=True,
        max_cached_images=50,
        random_pop=True, 
        rotation_range=0,     
        translation_range=(0.0, 0.0),   
        scaling_range=(1.0, 1.0), 
    ) -> None:
        if rotation_range not in (0, 0.0):     
            raise NotImplementedError("OBBMosaic does not support post-mosaic affine rotation")     
        if tuple(translation_range) != (0.0, 0.0):     
            raise NotImplementedError("OBBMosaic does not support post-mosaic affine translation")
        if tuple(scaling_range) != (1.0, 1.0):    
            raise NotImplementedError("OBBMosaic does not support post-mosaic affine scaling")
  
        self.resize = OBBResize(size=output_size, max_size=max_size)  
        self.probability = probability
        self.fill_value = fill_value  
        self.use_cache = use_cache  
        self.mosaic_cache = []
        self.max_cached_images = max_cached_images 
        self.random_pop = random_pop     
   
    def load_samples_from_dataset(self, image, target, dataset):
        image, target = self.resize(image, target)  
        resized_images, resized_targets = [image], [target]
        max_width, max_height = _get_image_size(image) 

        sample_indices = random.choices(range(len(dataset)), k=3)
        for idx in sample_indices:
            sample_image, sample_target = dataset.load_item(idx)    
            sample_image, sample_target = self.resize(sample_image, sample_target)    
            width, height = _get_image_size(sample_image)
            max_width, max_height = max(max_width, width), max(max_height, height) 
            resized_images.append(sample_image)
            resized_targets.append(sample_target) 

        return resized_images, resized_targets, max_height, max_width

    def load_samples_from_cache(self, image, target, cache):    
        image, target = self.resize(image, target)    
        cache.append({"img": image, "labels": target}) 

        if len(cache) > self.max_cached_images:     
            if self.random_pop:
                index = random.randint(0, len(cache) - 2)
            else:  
                index = 0
            cache.pop(index)
     
        sample_indices = random.choices(range(len(cache)), k=3)     
        mosaic_samples = [
            {   
                "img": _clone_image(cache[idx]["img"]), 
                "labels": _clone_target(cache[idx]["labels"]),    
            } 
            for idx in sample_indices 
        ] 
        mosaic_samples = [{"img": _clone_image(image), "labels": _clone_target(target)}] + mosaic_samples
 
        sizes = [_get_image_size(sample["img"]) for sample in mosaic_samples]    
        max_width = max(size[0] for size in sizes)    
        max_height = max(size[1] for size in sizes)
        return mosaic_samples, max_height, max_width 
  
    def create_mosaic_from_cache(self, mosaic_samples, max_height, max_width):
        placement_offsets = [[0, 0], [max_width, 0], [0, max_height], [max_width, max_height]]
        merged_image = Image.new(
            mode=mosaic_samples[0]["img"].mode,
            size=(max_width * 2, max_height * 2),
            color=self.fill_value,
        )
 
        targets = []
        for idx, sample in enumerate(mosaic_samples):    
            merged_image.paste(sample["img"], placement_offsets[idx])   
            targets.append(sample["labels"])  

        merged_target = merge_obb_mosaic_targets(  
            targets,
            placement_offsets,
            mosaic_height=max_height * 2,   
            mosaic_width=max_width * 2,
        )  
        return merged_image, merged_target
    
    def create_mosaic_from_dataset(self, images, targets, max_height, max_width):
        placement_offsets = [[0, 0], [max_width, 0], [0, max_height], [max_width, max_height]]  
        merged_image = Image.new(  
            mode=images[0].mode,
            size=(max_width * 2, max_height * 2),
            color=self.fill_value, 
        )     
        for idx, image in enumerate(images):
            merged_image.paste(image, placement_offsets[idx])  
     
        merged_target = merge_obb_mosaic_targets( 
            targets,
            placement_offsets, 
            mosaic_height=max_height * 2,  
            mosaic_width=max_width * 2,    
        )
        return merged_image, merged_target 

    def __call__(self, *inputs):
        image, target, rest = _unpack_inputs(inputs)
   
        if self.probability < 1.0 and random.random() > self.probability:  
            return _pack_outputs(image, target, rest)
  
        if self.use_cache:  
            mosaic_samples, max_height, max_width = self.load_samples_from_cache(image, target, self.mosaic_cache)
            mosaic_image, mosaic_target = self.create_mosaic_from_cache(mosaic_samples, max_height, max_width)    
        else:   
            if not rest:  
                raise ValueError("OBBMosaic with use_cache=False expects dataset in the transform inputs") 
            resized_images, resized_targets, max_height, max_width = self.load_samples_from_dataset(  
                image, 
                target,  
                rest[0],     
            )     
            mosaic_image, mosaic_target = self.create_mosaic_from_dataset(  
                resized_images,
                resized_targets,
                max_height,
                max_width,  
            )   
     
        return _pack_outputs(mosaic_image, mosaic_target, rest)
     

@register()   
class OBBResize:   
    def __init__( 
        self,
        size,
        interpolation=InterpolationMode.BILINEAR,
        max_size=None,
        antialias=True,
    ):     
        self.resize = T.Resize(
            size=size,    
            interpolation=interpolation,
            max_size=max_size, 
            antialias=antialias,
        )     
        self.size = self.resize.size
        self.interpolation = self.resize.interpolation     
        self.max_size = self.resize.max_size
        self.antialias = self.resize.antialias 
     
    def __call__(self, *inputs):    
        image, target, rest = _unpack_inputs(inputs)
        old_w, old_h = _get_image_size(image)
        out_image = self.resize(image)
        new_w, new_h = _get_image_size(out_image)  
        out_target = _clone_target(target)
        if "obb_polygons" in out_target:    
            polygons = out_target["obb_polygons"]
            scale_xy = polygons.new_tensor([new_w / old_w, new_h / old_h]).repeat(4)   
            out_target["obb_polygons"] = polygons * scale_xy 
            out_target["area"] = polygon_area(out_target["obb_polygons"])     
        out_target["size"] = torch.tensor([new_w, new_h], dtype=torch.int64)  
        return _pack_outputs(out_image, out_target, rest)  

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("   
            f"size={self.size}, "    
            f"interpolation={self.interpolation}, "    
            f"max_size={self.max_size}, "  
            f"antialias={self.antialias})"
        )


@register()
class OBBPadToSize:     
    def __init__(self, size, fill=0, padding_mode="constant"):
        self.size = _normalize_size(size)     
        self.fill = fill   
        self.padding_mode = padding_mode

    def __call__(self, *inputs):
        image, target, rest = _unpack_inputs(inputs)    
        old_w, old_h = _get_image_size(image)
        target_w, target_h = self.size    
        pad_right = max(0, target_w - old_w)
        pad_bottom = max(0, target_h - old_h)
        padding = [0, 0, pad_right, pad_bottom]   
        out_image = F.pad(image, padding=padding, fill=self.fill, padding_mode=self.padding_mode)
        out_target = _clone_target(target)
        out_target["padding"] = torch.tensor(padding, dtype=torch.int64)
        out_target["size"] = torch.tensor([max(old_w, target_w), max(old_h, target_h)], dtype=torch.int64)
        return _pack_outputs(out_image, out_target, rest)    
  

@register()
class OBBRandomHorizontalFlip:     
    def __init__(self, p=0.5): 
        self.p = p
     
    def __call__(self, *inputs):
        image, target, rest = _unpack_inputs(inputs)   
        if torch.rand(1).item() >= self.p:
            return _pack_outputs(image, target, rest)     

        width, _ = _get_image_size(image)    
        out_image = F.hflip(image)     
        out_target = _clone_target(target)
        if "obb_polygons" in out_target:
            polygons = out_target["obb_polygons"].clone()  
            polygons[:, 0::2] = width - polygons[:, 0::2]     
            out_target["obb_polygons"] = polygons     
            out_target["area"] = polygon_area(polygons)     
        return _pack_outputs(out_image, out_target, rest)
   
   
@register()
class OBBSanitize:
    def __init__(self, min_area=1.0):
        self.min_area = float(min_area) 
  
    def __call__(self, *inputs):
        image, target, rest = _unpack_inputs(inputs)    
        out_target = _clone_target(target)
        polygons = out_target.get("obb_polygons")
        if polygons is None:
            return _pack_outputs(image, out_target, rest)

        width, height = _get_image_size(image)     
        areas = polygon_area(polygons)   
        out_target["area"] = areas
        x = polygons[:, 0::2]
        y = polygons[:, 1::2]  
        inside_image = (x >= 0).all(dim=1) & (x <= width).all(dim=1) & (y >= 0).all(dim=1) & (y <= height).all(dim=1)     
        keep = torch.isfinite(polygons).all(dim=1) & torch.isfinite(areas) & (areas >= self.min_area) & inside_image
        for key in ("labels", "obb_polygons", "area", "boxes"):   
            if key in out_target and isinstance(out_target[key], torch.Tensor) and out_target[key].shape[:1] == keep.shape: 
                out_target[key] = out_target[key][keep]    
        return _pack_outputs(image, out_target, rest)


def _canonical_min_area_rect(points: np.ndarray) -> Tuple[float, float, float, float, float]:     
    (cx, cy), (w, h), angle_deg = cv2.minAreaRect(points.astype(np.float32))    
    if h > w:
        w, h = h, w   
        angle_deg += 90.0     
    angle_rad = math.radians(angle_deg)
    angle_rad = ((angle_rad + math.pi / 2) % math.pi) - math.pi / 2
    return float(cx), float(cy), float(w), float(h), float(angle_rad)


@register()     
class ConvertOBBBoxes:
    def __init__(self, normalize=False, angle_factor=math.pi):   
        self.normalize = normalize   
        self.angle_factor = angle_factor     
    
    def __call__(self, *inputs):
        image, target, rest = _unpack_inputs(inputs)
        out_target = _clone_target(target)  
        polygons = out_target.get("obb_polygons")
        if polygons is None:  
            return _pack_outputs(image, out_target, rest)  
        if polygons.numel() == 0:
            out_target["boxes"] = polygons.new_zeros((0, 5))
            return _pack_outputs(image, out_target, rest)   
 
        width, height = _get_image_size(image)
        boxes = [] 
        for polygon in polygons.detach().cpu().numpy().reshape(-1, 4, 2):
            cx, cy, w, h, angle = _canonical_min_area_rect(polygon)
            if self.normalize:  
                cx /= width
                cy /= height     
                w /= width   
                h /= height 
            if self.angle_factor:
                angle /= self.angle_factor    
            boxes.append([cx, cy, w, h, angle])    
        out_target["boxes"] = torch.tensor(boxes, dtype=polygons.dtype, device=polygons.device)
        return _pack_outputs(image, out_target, rest) 


@register()
class OBBRandomRotate:
    def __init__(self, angle_range=180, p=0.5, fill=0, min_area=1.0):
        self.angle_range = float(angle_range)
        self.p = p 
        self.fill = fill
        self.min_area = min_area 
     
    def __call__(self, *inputs):
        image, target, rest = _unpack_inputs(inputs)     
        if torch.rand(1).item() >= self.p:
            return _pack_outputs(image, target, rest)   
     
        angle = float(torch.empty(1).uniform_(-self.angle_range, self.angle_range).item())  
        width, height = _get_image_size(image)    
        out_image = F.rotate(image, angle=angle, interpolation=InterpolationMode.BILINEAR, expand=False, fill=self.fill)
        out_target = _clone_target(target)
        if "obb_polygons" in out_target and out_target["obb_polygons"].numel() > 0:
            polygons = out_target["obb_polygons"]
            points = polygons.reshape(-1, 4, 2)  
            center = polygons.new_tensor([(width - 1) * 0.5, (height - 1) * 0.5])     
            radians = math.radians(-angle)
            cos_a = math.cos(radians)
            sin_a = math.sin(radians)    
            matrix = polygons.new_tensor([[cos_a, -sin_a], [sin_a, cos_a]]) 
            rotated = (points - center) @ matrix.T + center
            out_target["obb_polygons"] = rotated.reshape(-1, 8)
            out_target["area"] = polygon_area(out_target["obb_polygons"])   
            _, out_target = OBBSanitize(min_area=self.min_area)(out_image, out_target)
        return _pack_outputs(out_image, out_target, rest)  
