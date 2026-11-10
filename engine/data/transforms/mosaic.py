"""  
DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
"""     

import torch
import torchvision.transforms.v2 as T   
import torchvision.transforms.v2.functional as F    
import random     
from PIL import Image   
     
from .._misc import convert_to_tv_tensor
from ...core import register     
    
   
INSTANCE_CAT_KEYS = {"boxes", "labels", "area", "iscrowd", "masks", "keypoints", "mixup"} 
   

def _clone_value(value):
    return value.clone() if hasattr(value, "clone") else value  

    
def _num_instances(target): 
    for key in ("boxes", "labels", "masks", "keypoints"):  
        if key in target:
            return len(target[key])     
    return 0  


def _offset_keypoints(keypoints, x_offset, y_offset):
    out = keypoints.clone()
    if out.numel() == 0:
        return out
    if out.ndim >= 3 and out.shape[-1] >= 2:
        out[..., 0] += x_offset    
        out[..., 1] += y_offset
    elif out.ndim == 2 and out.shape[1] % 3 == 0: 
        out = out.view(out.shape[0], -1, 3)  
        out[..., 0] += x_offset
        out[..., 1] += y_offset
        out = out.view(keypoints.shape)     
    else:   
        raise ValueError(f"Unsupported keypoints shape for mosaic: {tuple(keypoints.shape)}")
    return out     
     

def _place_masks_on_canvas(masks, x_offset, y_offset, mosaic_height, mosaic_width):
    if not isinstance(masks, torch.Tensor):
        raise NotImplementedError("Please convert masks to torch.Tensor format")
    num_instances = masks.shape[0]    
    full_masks = torch.zeros(
        (num_instances, mosaic_height, mosaic_width),
        dtype=masks.dtype,
        device=masks.device,
    )    
    h, w = masks.shape[-2:]  
    full_masks[:, y_offset:y_offset + h, x_offset:x_offset + w] = masks    
    return full_masks 
    
    
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

    
def merge_mosaic_targets(targets, placement_offsets, mosaic_height, mosaic_width):
    adjusted_targets = []     
    for i, target in enumerate(targets):
        target = {key: _clone_value(value) for key, value in target.items()}
        x_offset, y_offset = placement_offsets[i][:2]

        if "boxes" in target:
            offset = torch.as_tensor(   
                [x_offset, y_offset, x_offset, y_offset],  
                dtype=target["boxes"].dtype,    
                device=target["boxes"].device,
            )
            target["boxes"] = target["boxes"] + offset

        if "masks" in target:
            target["masks"] = _place_masks_on_canvas(
                target["masks"],   
                x_offset,  
                y_offset,   
                mosaic_height,
                mosaic_width,
            )
  
        if "keypoints" in target:
            target["keypoints"] = _offset_keypoints(target["keypoints"], x_offset, y_offset)
   
        adjusted_targets.append(target)
    
    primary_target = adjusted_targets[0]  
    merged_target = {}    
   
    for key in primary_target: 
        if key in INSTANCE_CAT_KEYS:  
            values = []     
            for target in adjusted_targets:  
                if key not in target:
                    continue    
                if key == "mixup" and len(target[key]) != _num_instances(target):    
                    continue
                values.append(target[key])
            if values:
                merged_target[key] = torch.cat(values, dim=0)   
        else: 
            merged_target[key] = _clone_value(primary_target[key])     
     
    source_image_ids = _collect_mosaic_source_image_ids(adjusted_targets)
    if source_image_ids is not None: 
        merged_target["mosaic_source_image_ids"] = source_image_ids     

    merged_target["orig_size"] = torch.as_tensor([mosaic_width, mosaic_height])
    merged_target["size"] = torch.as_tensor([mosaic_height, mosaic_width])
    return merged_target


@register()
class Mosaic(T.Transform):     
    """
    Applies Mosaic augmentation to a batch of images. Combines four randomly selected images  
    into a single composite image with randomized transformations.
    """    

    def __init__(self, output_size=320, max_size=None, rotation_range=0, translation_range=(0.1, 0.1),
                 scaling_range=(0.5, 1.5), probability=1.0, fill_value=114, use_cache=True, max_cached_images=50,
                 random_pop=True) -> None:
        """
        Args:
            output_size (int): Target size for resizing individual images.
            rotation_range (float): Range of rotation in degrees for affine transformation.
            translation_range (tuple): Range of translation for affine transformation.  
            scaling_range (tuple): Range of scaling factors for affine transformation.  
            probability (float): Probability of applying the Mosaic augmentation.
            fill_value (int): Fill value for padding or affine transformations.    
            use_cache (bool): Whether to use cache. Defaults to True.
            max_cached_images (int): The maximum length of the cache.
            random_pop (bool): Whether to randomly pop a result from the cache.
        """
        super().__init__()
        self.resize = T.Resize(size=output_size, max_size=max_size)
        self.probability = probability
        self.affine_transform = T.RandomAffine(degrees=rotation_range, translate=translation_range,
                                               scale=scaling_range, fill=fill_value)
        self.use_cache = use_cache  
        self.mosaic_cache = []   
        self.max_cached_images = max_cached_images
        self.random_pop = random_pop   

    def load_samples_from_dataset(self, image, target, dataset):   
        """Loads and resizes a set of images and their corresponding targets."""
        # Append the main image 
        get_size_func = F.get_size if hasattr(F, "get_size") else F.get_spatial_size  # torchvision >=0.17 is get_size
        image, target = self.resize(image, target)     
        resized_images, resized_targets = [image], [target]
        max_height, max_width = get_size_func(resized_images[0])

        # randomly select 3 images     
        sample_indices = random.choices(range(len(dataset)), k=3) 
        for idx in sample_indices: 
            # image, target = dataset.load_item(idx)    
            image, target = self.resize(dataset.load_item(idx))
            height, width = get_size_func(image)
            max_height, max_width = max(max_height, height), max(max_width, width)
            resized_images.append(image)
            resized_targets.append(target)

        return resized_images, resized_targets, max_height, max_width    

    def load_samples_from_cache(self, image, target, cache): 
        image, target = self.resize(image, target)
        cache.append(dict(img=image, labels=target)) 

        if len(cache) > self.max_cached_images:
            if self.random_pop:   
                index = random.randint(0, len(cache) - 2)  # do not remove last image
            else:    
                index = 0    
            cache.pop(index)
        sample_indices = random.choices(range(len(cache)), k=3)
        mosaic_samples = [dict(img=cache[idx]["img"].copy(), labels=self._clone(cache[idx]["labels"])) for idx in
                          sample_indices]  # sample 3 images
        mosaic_samples = [dict(img=image.copy(), labels=self._clone(target))] + mosaic_samples  

        get_size_func = F.get_size if hasattr(F, "get_size") else F.get_spatial_size 
        sizes = [get_size_func(mosaic_samples[idx]["img"]) for idx in range(4)]
        max_height = max(size[0] for size in sizes)
        max_width = max(size[1] for size in sizes)
  
        return mosaic_samples, max_height, max_width  
   
    # def create_mosaic_from_cache(self, mosaic_samples, max_height, max_width):    
    #     placement_offsets = [[0, 0], [max_width, 0], [0, max_height], [max_width, max_height]]   
    #     merged_image = Image.new(mode=mosaic_samples[0]["img"].mode, size=(max_width * 2, max_height * 2), color=0)     
    #     offsets = torch.tensor([[0, 0], [max_width, 0], [0, max_height], [max_width, max_height]]).repeat(1, 2)

    #     mosaic_target = []
    #     for i, sample in enumerate(mosaic_samples):    
    #         img = sample["img"]     
    #         target = sample["labels"] 
    
    #         merged_image.paste(img, placement_offsets[i])    
    #         target['boxes'] = target['boxes'] + offsets[i] 
    #         mosaic_target.append(target)

    #     merged_target = {}
    #     for key in mosaic_target[0]:
    #         merged_target[key] = torch.cat([target[key] for target in mosaic_target])    
     
    #     return merged_image, merged_target    
 
    # def create_mosaic_from_dataset(self, images, targets, max_height, max_width):     
    #     """Creates a mosaic image by combining multiple images.""" 
    #     placement_offsets = [[0, 0], [max_width, 0], [0, max_height], [max_width, max_height]]
    #     merged_image = Image.new(mode=images[0].mode, size=(max_width * 2, max_height * 2), color=0)     
    #     for i, img in enumerate(images):     
    #         merged_image.paste(img, placement_offsets[i])

    #     """Merges targets into a single target dictionary for the mosaic.""" 
    #     offsets = torch.tensor([[0, 0], [max_width, 0], [0, max_height], [max_width, max_height]]).repeat(1, 2)     
    #     merged_target = {}
    #     for key in targets[0]:
    #         if key == 'boxes':    
    #             values = [target[key] + offsets[i] for i, target in enumerate(targets)]
    #         else:
    #             values = [target[key] for target in targets]

    #         merged_target[key] = torch.cat(values, dim=0) if isinstance(values[0], torch.Tensor) else values
     
    #     return merged_image, merged_target
    
    def create_mosaic_from_cache(self, mosaic_samples, max_height, max_width):    
        placement_offsets = [[0, 0], [max_width, 0], [0, max_height], [max_width, max_height]] 
        merged_image = Image.new(mode=mosaic_samples[0]["img"].mode, size=(max_width * 2, max_height * 2), color=0)  
     
        mosaic_target = []
        for i, sample in enumerate(mosaic_samples):   
            img = sample["img"]
            target = sample["labels"]   
   
            merged_image.paste(img, placement_offsets[i])    
            mosaic_target.append(target)    
 
        merged_target = merge_mosaic_targets(   
            mosaic_target,
            placement_offsets,     
            mosaic_height=max_height * 2,   
            mosaic_width=max_width * 2,     
        )

        return merged_image, merged_target     

    def create_mosaic_from_dataset(self, images, targets, max_height, max_width):     
        """Creates a mosaic image by combining multiple images."""
        placement_offsets = [[0, 0], [max_width, 0], [0, max_height], [max_width, max_height]]
        merged_image = Image.new(mode=images[0].mode, size=(max_width * 2, max_height * 2), color=0)
        for i, img in enumerate(images):
            merged_image.paste(img, placement_offsets[i])  
   
        """Merges targets into a single target dictionary for the mosaic.""" 
        merged_target = merge_mosaic_targets( 
            targets,    
            placement_offsets,  
            mosaic_height=max_height * 2,
            mosaic_width=max_width * 2,  
        )
   
        return merged_image, merged_target    

    @staticmethod
    def _clone(tensor_dict):  
        return {key: value.clone() for (key, value) in tensor_dict.items()}

    def forward(self, *inputs): 
        """ 
        Args:   
            inputs (tuple): Input tuple containing (image, target, dataset).

        Returns:
            tuple: Augmented (image, target, dataset).     
        """
        if len(inputs) == 1:
            inputs = inputs[0]
        image, target, dataset = inputs    
 
        # Skip mosaic augmentation with probability 1 - self.probability
        if self.probability < 1.0 and random.random() > self.probability:   
            return image, target, dataset
  
        # Prepare mosaic components 
        if self.use_cache:
            mosaic_samples, max_height, max_width = self.load_samples_from_cache(image, target, self.mosaic_cache)    
            mosaic_image, mosaic_target = self.create_mosaic_from_cache(mosaic_samples, max_height, max_width)
        else:
            resized_images, resized_targets, max_height, max_width = self.load_samples_from_dataset(image, target,dataset)    
            mosaic_image, mosaic_target = self.create_mosaic_from_dataset(resized_images, resized_targets, max_height, max_width)  
 
        # Clamp boxes and convert target formats   
        if 'boxes' in mosaic_target:
            mosaic_target['boxes'] = convert_to_tv_tensor(mosaic_target['boxes'], 'boxes', box_format='xyxy',
                                                          spatial_size=mosaic_image.size[::-1])
        if 'masks' in mosaic_target:     
            mosaic_target['masks'] = convert_to_tv_tensor(mosaic_target['masks'], 'masks')     
     
        # Apply affine transformations
        mosaic_image, mosaic_target = self.affine_transform(mosaic_image, mosaic_target)     

        return mosaic_image, mosaic_target, dataset
