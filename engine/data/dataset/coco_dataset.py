"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py  

Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""     
import os, math, cv2, tqdm, random, psutil, contextlib    
from copy import deepcopy  
from pathlib import Path
from multiprocessing.pool import ThreadPool  
import numpy as np
import torch

import torchvision

from PIL import Image
from ._dataset import DetDataset    
from .._misc import convert_to_tv_tensor   
from ...core import register
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from torchvision.datasets.vision import VisionDataset    

torchvision.disable_beta_transforms_warning()     
Image.MAX_IMAGE_PIXELS = None 
NUM_THREADS = min(8, max(1, os.cpu_count() - 1))  # number of YOLO multiprocessing threads   
LOCAL_RANK = int(os.getenv("LOCAL_RANK", -1))  # https://pytorch.org/docs/stable/elastic/run.html 

__all__ = ['CocoDetection']

class CocoDetection_Cache(torchvision.datasets.CocoDetection):    
    def __init__(
        self, 
        root: Union[str, Path],  
        annFile: str, 
        transform: Optional[Callable[..., Any]] = None, 
        target_transform: Optional[Callable[..., Any]] = None, 
        transforms: Optional[Callable[..., Any]] = None, 
        cache_imgsz: Optional[int] = None,     
        ram_cache: bool = False, 
    ) -> None:   
        with open(os.devnull, 'w') as devnull:
            with contextlib.redirect_stdout(devnull):
                super().__init__(root, annFile, transform, target_transform, transforms)

        self.cache_imgsz = cache_imgsz 
        self.ram_cache = ram_cache  
        self.cache_flag = False    
        self.ni = len(self.ids)
        self.ims, self.labs = {i:None for i in self.ids}, {i:None for i in self.ids}   
        
        if self.ram_cache and self.check_cache_ram():
            self._populate_ram_cache()     
 
    def _populate_ram_cache(self):
        b, gb = 0, 1 << 30  # bytes of cached images, bytes per gigabytes
        with ThreadPool(NUM_THREADS) as pool:
            results = pool.imap(self.load_image_and_label_from_coco, self.ids) 
            pbar = tqdm.tqdm(enumerate(results), total=self.ni, disable=LOCAL_RANK > 0) 
            for i, x in pbar:
                d_id, d_ims, d_labs = x  # im, hw_orig, hw_resized = load_image(self, i)    
                b += np.array(self.ims[d_id]).nbytes     
                pbar.desc = f"Caching images ({b / gb:.1f}GB Ram"
            pbar.close()   
        self.cache_flag = True   

    def _cache_resize_ratio(self, h, w):  
        if self.cache_imgsz is None:
            return 1.0
        return float(self.cache_imgsz) / max(h, w)
 
    def load_image_and_label_from_coco(self, i):     
        im = self.ims[i]   
        im_info = self.coco.loadImgs(i)[0]
        path = im_info["file_name"]
        label = self.coco.loadAnns(self.coco.getAnnIds(i)) 
        if im is None:
            im = Image.open(os.path.join(self.root, path)).convert("RGB")
            im = np.array(im)
  
            h0, w0 = im.shape[:2]  # orig hw
            # resize long side to imgsz while maintaining aspect ratio
            r = self._cache_resize_ratio(h0, w0)  # ratio
            new_w, new_h = w0, h0
            if r != 1:  # if sizes are not equal  
                
                new_w = min(math.ceil(w0 * r), self.cache_imgsz)
                new_h = min(math.ceil(h0 * r), self.cache_imgsz)
                im = cv2.resize(im, (new_w, new_h), interpolation=cv2.INTER_LINEAR)    
    
                # ✅ 更新 COCO 内部的图像尺寸信息 
                img_id = im_info['id']
                # 方法1: 直接修改 imgs 字典   
                self.coco.imgs[img_id]['width'] = new_w
                self.coco.imgs[img_id]['height'] = new_h 
            
            # 对 label 进行缩放    
            for ann in label:
                # 缩放边界框
                if 'bbox' in ann:   
                    bbox = ann['bbox']  
                    ann['bbox'] = [   
                        bbox[0] * r,  # x   
                        bbox[1] * r,  # y  
                        bbox[2] * r,  # width 
                        bbox[3] * r   # height    
                    ]   
 
                # 缩放实例分割标注     
                if 'segmentation' in ann:
                    seg = ann['segmentation'] 
                    
                    # 情况1: Polygon 格式 (list of lists)  
                    if isinstance(seg, list):     
                        scaled_seg = []   
                        for poly in seg:
                            # poly 是 [x1, y1, x2, y2, ..., xn, yn] 格式
                            scaled_poly = []
                            for j in range(0, len(poly), 2):
                                scaled_poly.append(poly[j] * r)      # x 坐标
                                scaled_poly.append(poly[j+1] * r)    # y 坐标     
                            scaled_seg.append(scaled_poly)
                        ann['segmentation'] = scaled_seg 
                    
                    # 情况2: RLE 格式 (dict with 'counts' and 'size')
                    elif isinstance(seg, dict) and 'counts' in seg: 
                        # RLE 需要解码、缩放、再编码
                        from pycocoeval import mask as mask_util     
                        
                        # 解码 RLE 到二值 mask
                        mask = mask_util.decode(seg)  # (h0, w0)
  
                        # 缩放 mask 
                        mask_resized = cv2.resize(
                            mask, 
                            (new_w, new_h),  # 注意 cv2.resize 是 (width, height)  
                            interpolation=cv2.INTER_NEAREST
                        )     
  
                        # 重新编码为 RLE
                        # 确保是 Fortran 顺序（列优先）     
                        mask_resized = np.asfortranarray(mask_resized.astype(np.uint8))    
                        ann['segmentation'] = mask_util.encode(mask_resized)     
  
                # 缩放面积 
                if 'area' in ann:     
                    ann['area'] = ann['area'] * (r ** 2) 
            
            self.ims[i] = Image.fromarray(im)    
            self.labs[i] = label 

        return i, self.ims[i], self.labs[i]

    def _estimate_image_cache_bytes(self, sample_ids):
        b = 0     
        for image_id in sample_ids:
            im = cv2.imread(os.path.join(self.root, self.coco.loadImgs(image_id)[0]["file_name"]))  # sample image   
            ratio = self._cache_resize_ratio(im.shape[0], im.shape[1])  # max(h, w)  # ratio     
            b += im.nbytes * ratio**2 
        return b
    
    def check_cache_ram(self, safety_margin=0.5):
        """Check image caching requirements vs available memory."""
        gb = 1 << 30  # bytes per gigabyte
        n = min(self.ni, 30)  # extrapolate from 30 random images
        sample_ids = random.sample(self.ids, n)
        b = self._estimate_image_cache_bytes(sample_ids)
        mem_required = b * self.ni / n * (1 + safety_margin)  # GB required to cache dataset into RAM 
        mem = psutil.virtual_memory()
        cache = mem_required < mem.available  # to cache or not to cache, that is the question
        if not cache:    
            print( 
                f'{mem_required / gb:.1f}GB RAM required to cache images '
                f'with {int(safety_margin * 100)}% safety margin but only '
                f'{mem.available / gb:.1f}/{mem.total / gb:.1f}GB available, '     
                f"{'caching images ✅' if cache else 'not caching images ⚠️'}"
            )
        return cache 
   
    def __getitem__(self, index: int) -> Tuple[Any, Any]:     
        id = self.ids[index]   
        if self.cache_flag:    
            image = self.ims[id]
            target = self.labs[id]  
        else:
            image = self._load_image(id)
            target = self._load_target(id)    

        if self.transforms is not None:     
            image, target = self.transforms(image, target)     
    
        return image, target

@register()
# class CocoDetection(torchvision.datasets.CocoDetection, DetDataset):
class CocoDetection(CocoDetection_Cache, DetDataset):     
    __inject__ = ['transforms', ]    
    __share__ = ['remap_mscoco_category', 'cache_imgsz', 'ram_cache', 'num_classes', 'text_cache_file']     
   
    def __init__( 
        self,     
        img_folder,
        ann_file,
        transforms, 
        return_masks=False,     
        remap_mscoco_category=False,
        cache_imgsz=None,   
        ram_cache=False,   
        num_classes=None,  
        text_cache_file: Optional[str] = None,
    ):    
        super(CocoDetection, self).__init__(img_folder, ann_file, cache_imgsz=cache_imgsz, ram_cache=ram_cache)     
        # super(CocoDetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)
        self.img_folder = img_folder
        self.ann_file = ann_file 
        self.return_masks = return_masks     
        self.remap_mscoco_category = remap_mscoco_category
        self.num_classes = num_classes    
        self._text_cache_file = text_cache_file    
        self._text_cache: Optional[Dict[str, torch.Tensor]] = None
        self._validate_num_classes()   
 
    def _validate_num_classes(self):
        if self.num_classes is None:
            return 

        # When remapping to contiguous labels is enabled, downstream class space is    
        # determined by the remapping table instead of raw json category ids.
        if self.remap_mscoco_category:
            return
 
        categories = self.coco.dataset.get('categories', [])
        if not categories: 
            raise ValueError(
                f"COCO annotation has no categories in {self.ann_file}, "    
                f"but configured num_classes={self.num_classes}."
            )

        try:
            category_ids = sorted({int(cat['id']) for cat in categories})
        except Exception as e:
            raise ValueError(    
                f"Failed to parse category ids from {self.ann_file}. "   
                f"Raw categories={categories}"
            ) from e  

        min_id = min(category_ids)
        max_id = max(category_ids)   
        expected_num_classes = max_id + 1   
        configured_num_classes = int(self.num_classes)    
    
        category_pairs = sorted(  
            {(int(cat['id']), str(cat.get('name', ''))) for cat in categories}, 
            key=lambda x: x[0],
        )  

        if configured_num_classes != expected_num_classes:
            raise ValueError(
                "COCO num_classes mismatch: "
                f"ann_file={self.ann_file}; "
                f"configured num_classes={configured_num_classes}; "
                f"expected_num_classes={expected_num_classes} (max(category_id)+1); " 
                f"min_category_id={min_id}; max_category_id={max_id}; "  
                f"unique_category_count={len(category_ids)}; "
                f"category_ids={category_ids}; "
                f"categories(id,name)={category_pairs}. "   
                "Hint: if category ids are 1-based (e.g., 1..N), set num_classes to N+1 "   
                "to account for index 0."     
            )
 
    def __getitem__(self, idx):
        img, target = self.load_item(idx)   
        if self._transforms is not None:   
            img, target, _ = self._transforms(img, target, self)   
        return img, target  
   
    def load_item(self, idx):
        image, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]  
        target = {'image_id': image_id, 'annotations': target}
 
        if self.remap_mscoco_category:
            image, target = self.prepare(image, target, category2label=mscoco_category2label) 
        else:
            image, target = self.prepare(image, target)  

        target['idx'] = torch.tensor([idx]) 

        if 'boxes' in target:
            target['boxes'] = convert_to_tv_tensor(target['boxes'], key='boxes', spatial_size=image.size[::-1])   
    
        if 'masks' in target: 
            target['masks'] = convert_to_tv_tensor(target['masks'], key='masks') 
    
        return image, target   
   
    def extra_repr(self) -> str:
        s = f' img_folder: {self.img_folder}\n ann_file: {self.ann_file}\n'
        s += f' return_masks: {self.return_masks}\n'  
        if hasattr(self, '_transforms') and self._transforms is not None:   
            s += f' transforms:\n   {repr(self._transforms)}'   
        if hasattr(self, '_preset') and self._preset is not None:
            s += f' preset:\n   {repr(self._preset)}'
        return s   
  
    @property    
    def categories(self, ):
        return self.coco.dataset['categories']
 
    @property    
    def category2name(self, ):
        return {cat['id']: cat['name'] for cat in self.categories}  
     
    @property    
    def category2label(self, ):    
        return {cat['id']: i for i, cat in enumerate(self.categories)}

    @property     
    def label2category(self, ):
        return {i: cat['id'] for i, cat in enumerate(self.categories)}  

    @property
    def text_list(self) -> List[str]:    
        return sorted(cat['name'] for cat in self.categories)    

    @property  
    def detect_text_list(self) -> List[str]:
        return self.text_list    

    @property    
    def text_cache(self) -> Dict[str, torch.Tensor]:     
        if self._text_cache is None:
            if self._text_cache_file is None:
                raise RuntimeError("CocoDetection.text_cache requires text_cache_file to be set.")     
  
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


def convert_coco_poly_to_mask(segmentations, height, width): 
    masks = []
    valid_mask_flags = []  # 记录哪些 mask 是有效的
    
    for polygons in segmentations:  
        if not polygons:    
            # 空标注，创建全零掩码
            valid_mask_flags.append(False)   
            masks.append(torch.zeros((height, width), dtype=torch.uint8))     
            continue

        # 检查 polygon 格式是否有效
        if isinstance(polygons, list):  
            is_valid = False
            for poly in polygons:
                if isinstance(poly, list) and len(poly) >= 6:  # 至少3个点
                    is_valid = True
                    break  
    
            if not is_valid:  
                valid_mask_flags.append(False)
                masks.append(torch.zeros((height, width), dtype=torch.uint8))  
                continue    
        
        try:
            # import pycocoeval.mask as coco_mask     
            # rles = coco_mask.frPyObjects(polygons, height, width)
            # mask = coco_mask.decode(rles)
   
            mask = np.zeros((height, width), dtype=np.uint8)   
            for polygon in polygons:
                pts = np.array(polygon).reshape(-1, 2).astype(np.int32)   
                cv2.fillPoly(mask, [pts], 1)

            if len(mask.shape) < 3:     
                mask = mask[..., None]
            mask = torch.as_tensor(mask, dtype=torch.uint8)   
            mask = mask.any(dim=2)
     
            # 检查 mask 是否全为0（无效mask）     
            if mask.sum() == 0:
                valid_mask_flags.append(False)    
            else:
                valid_mask_flags.append(True)

            masks.append(mask)   
        except Exception as e:
            valid_mask_flags.append(False)
            masks.append(torch.zeros((height, width), dtype=torch.uint8))     
    if masks: 
        masks = torch.stack(masks, dim=0)    
    else:   
        masks = torch.zeros((0, height, width), dtype=torch.uint8) 
     
    return masks, torch.tensor(valid_mask_flags, dtype=torch.bool)


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):   
        self.return_masks = return_masks 

    def __call__(self, image: Image.Image, target, **kwargs): 
        w, h = image.size
    
        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]   
 
        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]   
    
        boxes = [obj["bbox"] for obj in anno]  
        # guard against no boxes via resizing   
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]   
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)    
     
        category2label = kwargs.get('category2label', None)
        if category2label is not None:   
            labels = [category2label[obj["category_id"]] for obj in anno]   
        else:  
            labels = [obj["category_id"] for obj in anno]
    
        labels = torch.tensor(labels, dtype=torch.int64)
 
        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]     
            masks, valid_mask_flags = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:    
            keypoints = [obj["keypoints"] for obj in anno]   
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)  
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)
  
        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        # 如果需要返回 masks，额外过滤掉没有有效 segmentation 的目标 
        if self.return_masks:  
            keep = keep & valid_mask_flags 
        boxes = boxes[keep] 
        labels = labels[keep] 
        if self.return_masks:
            masks = masks[keep] 
        if keypoints is not None:     
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes 
        target["labels"] = labels 
        if self.return_masks:    
            target["masks"] = masks
        target["image_id"] = image_id    
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api  
        area = torch.tensor([obj["area"] for obj in anno])   
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]     
        target["iscrowd"] = iscrowd[keep] 

        target["orig_size"] = torch.as_tensor([int(w), int(h)])
        # target["size"] = torch.as_tensor([int(w), int(h)])

        return image, target     

 
mscoco_category2name = { 
    1: 'person',  
    2: 'bicycle',
    3: 'car',
    4: 'motorcycle',
    5: 'airplane',
    6: 'bus',
    7: 'train',
    8: 'truck',  
    9: 'boat',   
    10: 'traffic light',
    11: 'fire hydrant',
    13: 'stop sign',     
    14: 'parking meter',
    15: 'bench',
    16: 'bird',   
    17: 'cat',
    18: 'dog',   
    19: 'horse',
    20: 'sheep',    
    21: 'cow',  
    22: 'elephant', 
    23: 'bear',
    24: 'zebra',
    25: 'giraffe',    
    27: 'backpack',   
    28: 'umbrella',  
    31: 'handbag',
    32: 'tie',  
    33: 'suitcase',
    34: 'frisbee',  
    35: 'skis',
    36: 'snowboard',
    37: 'sports ball',     
    38: 'kite',
    39: 'baseball bat',
    40: 'baseball glove',   
    41: 'skateboard',
    42: 'surfboard',     
    43: 'tennis racket',
    44: 'bottle',
    46: 'wine glass', 
    47: 'cup',  
    48: 'fork',    
    49: 'knife',  
    50: 'spoon',
    51: 'bowl',    
    52: 'banana',
    53: 'apple',
    54: 'sandwich',     
    55: 'orange',
    56: 'broccoli',   
    57: 'carrot',
    58: 'hot dog',
    59: 'pizza', 
    60: 'donut',
    61: 'cake',
    62: 'chair',
    63: 'couch',
    64: 'potted plant', 
    65: 'bed',   
    67: 'dining table',   
    70: 'toilet',    
    72: 'tv',   
    73: 'laptop',
    74: 'mouse',
    75: 'remote',
    76: 'keyboard',  
    77: 'cell phone', 
    78: 'microwave',
    79: 'oven',  
    80: 'toaster',     
    81: 'sink',
    82: 'refrigerator', 
    84: 'book',
    85: 'clock',
    86: 'vase',
    87: 'scissors',     
    88: 'teddy bear',     
    89: 'hair drier',
    90: 'toothbrush'     
}    
   
mscoco_category2label = {k: i for i, k in enumerate(mscoco_category2name.keys())}  
mscoco_label2category = {v: k for k, v in mscoco_category2label.items()}
