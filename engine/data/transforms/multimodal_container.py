"""
Multimodal transform container for RGB + NPY samples.     
"""

import random
from typing import Any

import torch
import torch.nn as nn 
import torchvision  
import torchvision.transforms.v2 as T    
import torchvision.transforms.v2.functional as F   
from PIL import Image as PILImage  
   
from ._transforms import EmptyTransform     
from .._misc import Image as TVImage
from ...misc.modality_utils import normalize_tensor_minmax_per_sample
from ...core import GLOBAL_CONFIG, register
from ...logger_module import get_logger

logger = get_logger(__name__)
torchvision.disable_beta_transforms_warning()    
  
_COLOR_ONLY_TRANSFORMS = {
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
     
_MOSAIC_NAMES = {"Mosaic", "MultimodalMosaic"}  

  
@register()     
class NormalizeNPYMinMax(nn.Module):
    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps 

    @staticmethod
    def _ensure_single_channel_chw(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim == 2:
            return tensor.unsqueeze(0)  
   
        if tensor.ndim == 3:
            if tensor.shape[0] == 1:
                return tensor
            if tensor.shape[-1] in {1, 3, 4} and tensor.shape[0] not in {1, 3, 4}: 
                tensor = tensor.permute(2, 0, 1)
            if tensor.shape[0] == 1:
                return tensor
            return tensor.float().mean(dim=0, keepdim=True)    

        if tensor.ndim == 4:   
            if tensor.shape[1] == 1:
                return tensor
            if tensor.shape[-1] in {1, 3, 4} and tensor.shape[1] not in {1, 3, 4}:
                tensor = tensor.permute(0, 3, 1, 2)
            if tensor.shape[1] == 1:
                return tensor
            return tensor.float().mean(dim=1, keepdim=True) 
  
        return tensor 
   
    def _normalize_sample(self, sample):
        if not isinstance(sample, dict) or "npy" not in sample:
            return sample     

        npy = sample["npy"] 
        if isinstance(npy, PILImage.Image):
            npy = F.pil_to_tensor(npy)    
        elif type(npy).__name__ == "Image":     
            npy = npy.as_subclass(torch.Tensor)
        elif not torch.is_tensor(npy):
            npy = torch.as_tensor(npy) 
 
        npy = self._ensure_single_channel_chw(npy)     
     
        sample = dict(sample)    
        sample["npy"] = normalize_tensor_minmax_per_sample(npy, eps=self.eps)
        return sample   

    def forward(self, *inputs: Any):     
        if len(inputs) == 1:
            inputs = inputs[0]

        if isinstance(inputs, tuple) and len(inputs) == 3:  
            sample, target, dataset = inputs
            return self._normalize_sample(sample), target, dataset
     
        if isinstance(inputs, tuple) and len(inputs) == 2:     
            sample, target = inputs     
            return self._normalize_sample(sample), target     
   
        raise ValueError("NormalizeNPYMinMax expects (sample, target) or (sample, target, dataset)")   
   

@register()     
class MultimodalCompose(T.Compose):
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
                elif isinstance(op, nn.Module):    
                    transforms.append(op)     
                else:
                    raise ValueError("Unsupported transform config in MultimodalCompose")
        else:  
            transforms = [EmptyTransform()]     

        super().__init__(transforms=transforms)     
     
        self.mosaic_prob = mosaic_prob   
        if policy is None:     
            policy = {"name": "default"}  
        else:
            if self.mosaic_prob > 0:    
                logger.info("     ### Mosaic with Prob.@{} and ZoomOut/IoUCrop existed ### ".format(self.mosaic_prob))
            if "epoch" in policy:
                logger.info("     ### ImgTransforms Epochs: {} ### ".format(policy["epoch"]))   
            if "ops" in policy:
                logger.info("     ### Policy_ops@{} ###".format(policy["ops"]))
        self.global_samples = 0     
        self.policy = policy
   
    def forward(self, *inputs: Any) -> Any:
        return self.get_forward(self.policy["name"])(*inputs) 
 
    def get_forward(self, name):
        forwards = {
            "default": self.default_forward,
            "stop_epoch": self.stop_epoch_forward,  
            "stop_sample": self.stop_sample_forward,
        }    
        return forwards[name]

    def default_forward(self, *inputs: Any) -> Any:  
        sample, target, dataset = self._normalize_inputs(*inputs) 
        for transform in self.transforms:    
            sample, target, dataset = self._apply_transform(transform, sample, target, dataset) 
        return sample, target, dataset

    def stop_epoch_forward(self, *inputs: Any): 
        sample, target, dataset = self._normalize_inputs(*inputs)    
        cur_epoch = dataset.epoch     
        policy_ops = self.policy["ops"]
        policy_epoch = self.policy["epoch"]    

        if isinstance(policy_epoch, list) and len(policy_epoch) == 3:
            if policy_epoch[0] <= cur_epoch < policy_epoch[1]:
                with_mosaic = random.random() <= self.mosaic_prob
            else:
                with_mosaic = False

            for transform in self.transforms:
                t_name = type(transform).__name__  
                if t_name in policy_ops and cur_epoch < policy_epoch[0]: 
                    continue
                if t_name in policy_ops and cur_epoch >= policy_epoch[-1]:  
                    continue     
                if t_name in _MOSAIC_NAMES and not with_mosaic: 
                    continue     
                if t_name in {"RandomZoomOut", "RandomIoUCrop"} and with_mosaic:
                    continue     
                sample, target, dataset = self._apply_transform(transform, sample, target, dataset) 
        else:
            for transform in self.transforms:  
                t_name = type(transform).__name__  
                if t_name in policy_ops and cur_epoch >= policy_epoch:
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
        raise ValueError("MultimodalCompose expects (sample, target, dataset)")    
  
    def _apply_transform(self, transform, sample, target, dataset):
        if type(transform).__name__ == "ConvertPILImage":    
            rgb, target = self._apply_color_only(transform, sample["rgb"], target)
            sample = dict(sample)
            sample["rgb"] = rgb   
            return sample, target, dataset    

        if type(transform).__name__ in _COLOR_ONLY_TRANSFORMS:
            rgb, target = self._apply_color_only(transform, sample["rgb"], target)   
            sample = dict(sample) 
            sample["rgb"] = rgb
            return sample, target, dataset    
     
        sample = self._ensure_npy_image(sample)
        transformed = self._apply_spatial_transform(transform, sample, target, dataset)
        if not isinstance(transformed, tuple):
            return transformed, target, dataset    

        if len(transformed) == 3:
            return transformed
        if len(transformed) == 2: 
            sample, target = transformed
            return sample, target, dataset    
  
        sample = transformed[0]
        return sample, target, dataset    
   
    @staticmethod   
    def _apply_color_only(transform, rgb, target):    
        transformed = transform((rgb, target))   
        if not isinstance(transformed, tuple):    
            return transformed, target

        if len(transformed) >= 2:  
            return transformed[0], transformed[1]
   
        return transformed[0], target

    @staticmethod 
    def _apply_spatial_transform(transform, sample, target, dataset):
        # Most transforms in this repo accept (sample, target, dataset).
        # RandomIoUCrop is strict and expects only (sample, target).
        try:   
            return transform((sample, target, dataset))  
        except TypeError:
            if type(transform).__name__ != "RandomIoUCrop":
                raise
            return transform((sample, target))  

    @staticmethod
    def _ensure_npy_image(sample):
        if not isinstance(sample, dict) or "npy" not in sample:
            return sample
    
        npy = sample["npy"]     
        if type(npy).__name__ == "Image":
            return sample
  
        if isinstance(npy, torch.Tensor): 
            if npy.ndim == 2: 
                npy = npy.unsqueeze(0)
            sample = dict(sample)
            sample["npy"] = TVImage(npy)    

        return sample   
