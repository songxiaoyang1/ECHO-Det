"""OBB-aware multimodal transform container."""
   
import math
import random    
from typing import Any
     
import torch
import torch.nn as nn 
import torchvision    
import torchvision.transforms.v2 as T     
import torchvision.transforms.v2.functional as F  
from torchvision.transforms.v2 import InterpolationMode 
  
from ...core import GLOBAL_CONFIG, register
from ...logger_module import get_logger 
from ._transforms import EmptyTransform
from .multimodal_container import NormalizeNPYMinMax
from .obb_transforms import OBBSanitize, _clone_target, _get_image_size, polygon_area  
    
logger = get_logger(__name__)
torchvision.disable_beta_transforms_warning()


_RGB_ONLY_TRANSFORMS = {
    "RandomPhotometricDistort",
    "ColorJitter",
    "RandomAutocontrast",    
    "RandomEqualize",
    "RandomPosterize",
    "RandomSolarize",
    "RandomAdjustSharpness", 
    "RandomGrayscale", 
    "Normalize",     
}  
_MOSAIC_NAMES = {"MultimodalOBBMosaic"}
_TARGET_ONLY_TRANSFORMS = {"OBBSanitize", "ConvertOBBBoxes"}
_DATASET_AWARE_TRANSFORMS = {"RandomLoadTexts", "LookupTextFeats"}

 
@register()  
class MultimodalOBBCompose(T.Compose):
    def __init__(self, ops, policy=None, mosaic_prob=-0.1) -> None: 
        transforms = []    
        if ops is not None:  
            for op in ops:
                if isinstance(op, dict):  
                    name = op.pop("type")
                    transform = getattr(GLOBAL_CONFIG[name]["_pymodule"], GLOBAL_CONFIG[name]["_name"])(**op)     
                    transforms.append(transform)  
                    op["type"] = name
                    logger.info("     ### Transform @{} ###    ".format(type(transform).__name__))
                elif isinstance(op, nn.Module) or callable(op):
                    transforms.append(op)
                else: 
                    raise ValueError("Unsupported transform config in MultimodalOBBCompose")    
        else:
            transforms = [EmptyTransform()] 
     
        super().__init__(transforms=transforms)     
        self.mosaic_prob = mosaic_prob 
        self.policy = policy or {"name": "default"}
        self.global_samples = 0   
 
    def forward(self, *inputs: Any) -> Any:
        return self.get_forward(self.policy["name"])(*inputs)   

    def get_forward(self, name):     
        return {     
            "default": self.default_forward,
            "stop_epoch": self.stop_epoch_forward,     
            "stop_sample": self.stop_sample_forward,
        }[name]    

    def default_forward(self, *inputs: Any) -> Any:  
        sample, target, dataset = self._normalize_inputs(*inputs)
        for transform in self.transforms:  
            sample, target, dataset = self._apply_transform(transform, sample, target, dataset)
        return sample, target, dataset 

    def stop_epoch_forward(self, *inputs: Any):   
        sample, target, dataset = self._normalize_inputs(*inputs)
        cur_epoch = getattr(dataset, "epoch", 0) 
        policy_ops = self.policy["ops"]
        policy_epoch = self.policy["epoch"]  
  
        if isinstance(policy_epoch, list) and len(policy_epoch) == 3:
            with_mosaic = policy_epoch[0] <= cur_epoch < policy_epoch[1] and random.random() <= self.mosaic_prob
            for transform in self.transforms:
                name = type(transform).__name__   
                if name in policy_ops and cur_epoch < policy_epoch[0]:     
                    continue
                if name in policy_ops and cur_epoch >= policy_epoch[-1]:   
                    continue
                if name in _MOSAIC_NAMES and not with_mosaic:  
                    continue     
                sample, target, dataset = self._apply_transform(transform, sample, target, dataset)   
        else:     
            for transform in self.transforms:    
                name = type(transform).__name__
                if name in policy_ops and cur_epoch >= policy_epoch:
                    continue  
                sample, target, dataset = self._apply_transform(transform, sample, target, dataset)    
        return sample, target, dataset

    def stop_sample_forward(self, *inputs: Any):
        sample, target, dataset = self._normalize_inputs(*inputs)
        policy_ops = self.policy["ops"]  
        policy_sample = self.policy["sample"]
        for transform in self.transforms:     
            if type(transform).__name__ in policy_ops and self.global_samples >= policy_sample:
                continue
            sample, target, dataset = self._apply_transform(transform, sample, target, dataset)
        self.global_samples += 1 
        return sample, target, dataset  

    @staticmethod
    def _normalize_inputs(*inputs):     
        sample = inputs if len(inputs) > 1 else inputs[0]    
        if isinstance(sample, tuple) and len(sample) == 3:
            return sample  
        raise ValueError("MultimodalOBBCompose expects (sample, target, dataset)")
    
    def _apply_transform(self, transform, sample, target, dataset):
        name = type(transform).__name__
        if name == "NormalizeNPYMinMax" or isinstance(transform, NormalizeNPYMinMax):
            return transform(sample, target, dataset)
        if name in _RGB_ONLY_TRANSFORMS:
            rgb, target = self._apply_single(transform, sample["rgb"], target)
            return {**sample, "rgb": rgb}, target, dataset 
        if name == "ConvertPILImage":
            rgb, _ = self._apply_single(transform, sample["rgb"], target)
            return {**sample, "rgb": rgb}, target, dataset   
        if name == "OBBRandomHorizontalFlip":
            return self._apply_horizontal_flip(transform, sample, target, dataset)     
        if name == "OBBRandomRotate":   
            return self._apply_rotate(transform, sample, target, dataset)   
        if name in _TARGET_ONLY_TRANSFORMS:
            rgb, target = self._apply_single(transform, sample["rgb"], target)
            return {**sample, "rgb": rgb}, target, dataset 
        if name in _DATASET_AWARE_TRANSFORMS:    
            return transform((sample, target, dataset))
        if name in _MOSAIC_NAMES:    
            return transform(sample, target, dataset)  
  
        rgb, out_target = self._apply_single(transform, sample["rgb"], target)
        npy, _ = self._apply_single(transform, sample["npy"], target)     
        return {**sample, "rgb": rgb, "npy": npy}, out_target, dataset

    @staticmethod
    def _apply_single(transform, image, target): 
        transformed = transform((image, target))    
        if not isinstance(transformed, tuple):  
            return transformed, target
        return transformed[0], transformed[1] if len(transformed) > 1 else target

    @staticmethod    
    def _apply_horizontal_flip(transform, sample, target, dataset):
        if torch.rand(1).item() >= transform.p:  
            return sample, target, dataset

        width, _ = _get_image_size(sample["rgb"])  
        out_target = _clone_target(target)     
        if "obb_polygons" in out_target:     
            polygons = out_target["obb_polygons"].clone()    
            polygons[:, 0::2] = width - polygons[:, 0::2]  
            out_target["obb_polygons"] = polygons     
            out_target["area"] = polygon_area(polygons)
        return {**sample, "rgb": F.hflip(sample["rgb"]), "npy": F.hflip(sample["npy"])}, out_target, dataset 

    @staticmethod
    def _apply_rotate(transform, sample, target, dataset):  
        if torch.rand(1).item() >= transform.p:   
            return sample, target, dataset
     
        angle = float(torch.empty(1).uniform_(-transform.angle_range, transform.angle_range).item())    
        width, height = _get_image_size(sample["rgb"])
        out_sample = {   
            **sample,
            "rgb": F.rotate(sample["rgb"], angle=angle, interpolation=InterpolationMode.BILINEAR, expand=False, fill=transform.fill),
            "npy": F.rotate(sample["npy"], angle=angle, interpolation=InterpolationMode.BILINEAR, expand=False, fill=transform.fill),  
        }  
        out_target = _clone_target(target)   
        if "obb_polygons" in out_target and out_target["obb_polygons"].numel() > 0:     
            polygons = out_target["obb_polygons"]     
            points = polygons.reshape(-1, 4, 2)     
            center = polygons.new_tensor([(width - 1) * 0.5, (height - 1) * 0.5])
            radians = math.radians(-angle) 
            matrix = polygons.new_tensor(
                [[math.cos(radians), -math.sin(radians)], [math.sin(radians), math.cos(radians)]]  
            )     
            rotated = (points - center) @ matrix.T + center     
            out_target["obb_polygons"] = rotated.reshape(-1, 8)    
            out_target["area"] = polygon_area(out_target["obb_polygons"])
            _, out_target = OBBSanitize(min_area=transform.min_area)(out_sample["rgb"], out_target)  
        return out_sample, out_target, dataset     
