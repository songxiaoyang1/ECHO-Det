"""OBB-safe multimodal mosaic for RGB plus second modality."""    
 
import random

import torch
import torch.nn.functional as torch_F
import torchvision.transforms.v2.functional as F 
from PIL import Image as PILImage   
   
from ...core import register
from .obb_transforms import OBBResize, _clone_target, merge_obb_mosaic_targets   

 
@register() 
class MultimodalOBBMosaic:
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
            raise NotImplementedError("MultimodalOBBMosaic does not support post-mosaic affine rotation")  
        if tuple(translation_range) != (0.0, 0.0):
            raise NotImplementedError("MultimodalOBBMosaic does not support post-mosaic affine translation")
        if tuple(scaling_range) != (1.0, 1.0):  
            raise NotImplementedError("MultimodalOBBMosaic does not support post-mosaic affine scaling")  
   
        self.resize = OBBResize(size=output_size, max_size=max_size)  
        self.probability = probability
        self.fill_value = fill_value 
        self.use_cache = use_cache
        self.mosaic_cache = []    
        self.max_cached_images = max_cached_images  
        self.random_pop = random_pop
  
    def load_samples_from_dataset(self, sample, target, dataset):
        sample, target = self._resize_sample_and_target(sample, target)
        mosaic_samples = [{"sample": sample, "target": target}]  
        max_width, max_height = self._get_size(sample["rgb"]) 

        for idx in random.choices(range(len(dataset)), k=3):
            sampled, sampled_target = dataset.load_item(idx)
            sampled, sampled_target = self._resize_sample_and_target(sampled, sampled_target)    
            width, height = self._get_size(sampled["rgb"])
            max_width, max_height = max(max_width, width), max(max_height, height)   
            mosaic_samples.append({"sample": sampled, "target": sampled_target})
        return mosaic_samples, max_height, max_width

    def load_samples_from_cache(self, sample, target, cache):
        sample, target = self._resize_sample_and_target(sample, target) 
        cache.append({"sample": self._clone_sample(sample), "target": _clone_target(target)})
        if len(cache) > self.max_cached_images:    
            index = random.randint(0, len(cache) - 2) if self.random_pop else 0   
            cache.pop(index)    

        sample_indices = random.choices(range(len(cache)), k=3)  
        mosaic_samples = [  
            {"sample": self._clone_sample(cache[idx]["sample"]), "target": _clone_target(cache[idx]["target"])}
            for idx in sample_indices
        ]
        mosaic_samples = [{"sample": self._clone_sample(sample), "target": _clone_target(target)}] + mosaic_samples
        sizes = [self._get_size(one["sample"]["rgb"]) for one in mosaic_samples]
        return mosaic_samples, max(size[1] for size in sizes), max(size[0] for size in sizes)
   
    def create_mosaic(self, mosaic_samples, max_height, max_width):  
        placement_offsets = [[0, 0], [max_width, 0], [0, max_height], [max_width, max_height]]
        rgb_ref = mosaic_samples[0]["sample"]["rgb"] 
        npy_ref = mosaic_samples[0]["sample"]["npy"]   
        rgb_ref_tensor = self._to_chw_tensor(rgb_ref)    
        npy_ref_tensor = self._to_chw_tensor(npy_ref)

        merged_rgb = torch.full(
            (rgb_ref_tensor.shape[0], max_height * 2, max_width * 2),
            self.fill_value,
            dtype=rgb_ref_tensor.dtype,   
            device=rgb_ref_tensor.device,   
        )
        merged_npy = torch.full(
            (npy_ref_tensor.shape[0], max_height * 2, max_width * 2),     
            self.fill_value,
            dtype=npy_ref_tensor.dtype,
            device=npy_ref_tensor.device,
        )     
   
        targets = []   
        for idx, one in enumerate(mosaic_samples):   
            x_offset, y_offset = placement_offsets[idx]     
            rgb_tensor = self._to_chw_tensor(one["sample"]["rgb"])
            npy_tensor = self._to_chw_tensor(one["sample"]["npy"])
            height, width = rgb_tensor.shape[-2:]
            merged_rgb[:, y_offset : y_offset + height, x_offset : x_offset + width] = rgb_tensor
            merged_npy[:, y_offset : y_offset + height, x_offset : x_offset + width] = npy_tensor
            targets.append(one["target"])

        merged_sample = { 
            "rgb": self._restore_type(merged_rgb, rgb_ref),
            "npy": self._restore_type(merged_npy, npy_ref),
        } 
        merged_target = merge_obb_mosaic_targets(     
            targets,  
            placement_offsets, 
            mosaic_height=max_height * 2,
            mosaic_width=max_width * 2,    
        )
        return merged_sample, merged_target
  
    def _resize_sample_and_target(self, sample, target):     
        rgb, target = self.resize(sample["rgb"], target)
        width, height = self._get_size(rgb) 
        npy = self._resize_modality(sample["npy"], height, width) 
        return {"rgb": rgb, "npy": npy}, target 

    @staticmethod 
    def _resize_modality(modality, height, width):     
        if isinstance(modality, PILImage.Image):    
            return modality.resize((width, height), resample=PILImage.BILINEAR)

        tensor = MultimodalOBBMosaic._to_chw_tensor(modality)   
        orig_dtype = tensor.dtype 
        tensor = torch_F.interpolate(    
            tensor.unsqueeze(0).to(dtype=torch.float32),     
            size=(height, width),
            mode="bilinear",  
            align_corners=False,
        ).squeeze(0)    
        return tensor.to(dtype=orig_dtype) if orig_dtype != torch.float32 else tensor    

    @staticmethod
    def _clone_sample(sample): 
        cloned = {}  
        for key, value in sample.items():
            if isinstance(value, PILImage.Image):     
                cloned[key] = value.copy() 
            elif hasattr(value, "clone"):
                cloned[key] = value.clone()
            else:
                cloned[key] = value
        return cloned  

    @staticmethod
    def _get_size(image):
        if isinstance(image, PILImage.Image):
            return image.size   
        return image.shape[-1], image.shape[-2]   
     
    @staticmethod
    def _to_chw_tensor(image):
        if isinstance(image, PILImage.Image):
            return F.pil_to_tensor(image) 
        tensor = image.as_subclass(torch.Tensor) if type(image).__name__ == "Image" else torch.as_tensor(image)     
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        return tensor     

    @staticmethod    
    def _restore_type(tensor, reference):
        if isinstance(reference, PILImage.Image):   
            return F.to_pil_image(tensor)
        if type(reference).__name__ == "Image": 
            return reference.__class__(tensor) 
        return tensor
     
    def __call__(self, *inputs):
        if len(inputs) == 1:
            inputs = inputs[0]     
        sample, target, dataset = inputs
        if self.probability < 1.0 and random.random() > self.probability:
            return sample, target, dataset     
    
        if self.use_cache:     
            mosaic_samples, max_height, max_width = self.load_samples_from_cache(sample, target, self.mosaic_cache)    
        else:
            mosaic_samples, max_height, max_width = self.load_samples_from_dataset(sample, target, dataset)
   
        mosaic_sample, mosaic_target = self.create_mosaic(mosaic_samples, max_height, max_width)
        return mosaic_sample, mosaic_target, dataset
