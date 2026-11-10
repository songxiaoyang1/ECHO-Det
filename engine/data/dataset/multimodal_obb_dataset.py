"""Multimodal YOLO-OBB dataset for RGB plus image/NPY second modality."""    

import math
from pathlib import Path
  
import numpy as np     
import torch
import torch.nn.functional as F 
from PIL import Image
   
from ...core import register   
from .obb_dataset import OBBDetection
     

@register()  
class MultimodalOBBDetection(OBBDetection):
    __inject__ = ["transforms"]     
    __share__ = ["num_classes", "angle_factor", "text_cache_file"]
 
    def __init__(
        self,  
        img_folder,
        label_folder,  
        modality_folder=None,
        npy_folder=None,  
        modality_format="image", 
        modality_size_mismatch="error",
        transforms=None, 
        data_yaml=None, 
        names=None,
        label_format="yolo_obb",
        normalized=True,    
        image_extensions=(".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"),
        filter_empty=False,  
        num_classes=None, 
        angle_factor=None,
        text_cache_file=None,    
    ):
        if modality_format not in {"image", "npy"}: 
            raise ValueError("modality_format must be one of {'image', 'npy'}")     
        if modality_size_mismatch not in {"error", "resize"}:   
            raise ValueError("modality_size_mismatch must be one of {'error', 'resize'}")   
        if modality_format == "image" and modality_folder is None:
            raise ValueError("modality_folder is required when modality_format='image'")   
        if modality_format == "npy" and npy_folder is None:
            raise ValueError("npy_folder is required when modality_format='npy'") 
  
        super().__init__(
            img_folder=img_folder,
            label_folder=label_folder,   
            transforms=transforms,   
            data_yaml=data_yaml,
            names=names,
            label_format=label_format,   
            normalized=normalized,     
            image_extensions=image_extensions,
            filter_empty=filter_empty,
            num_classes=num_classes,
            angle_factor=angle_factor if angle_factor is not None else math.pi,  
            text_cache_file=text_cache_file,
        )     
        self.modality_folder = str(modality_folder) if modality_folder is not None else None  
        self.npy_folder = str(npy_folder) if npy_folder is not None else None
        self.modality_format = modality_format   
        self.modality_size_mismatch = modality_size_mismatch
        self._image_extensions = tuple(image_extensions)     
  
    def load_item(self, idx):
        rgb, target = super().load_item(idx)
        modality = self._load_modality(idx, rgb)   
        return {"rgb": rgb, "npy": modality}, target
   
    def _relative_image_path(self, idx: int) -> Path:  
        image_path = Path(self.samples[idx][0])
        return image_path.relative_to(Path(self.img_folder))
   
    def _load_modality(self, idx: int, rgb: Image.Image):  
        if self.modality_format == "image":     
            return self._load_image_modality(idx, rgb)
        return self._load_npy_modality(idx, rgb)

    def _load_image_modality(self, idx: int, rgb: Image.Image) -> Image.Image:
        rel = self._relative_image_path(idx)
        modality_root = Path(self.modality_folder)
        modality_path = modality_root / rel
        if not modality_path.exists():
            candidates = [modality_root / rel.with_suffix(ext) for ext in self._image_extensions]
            modality_path = next((path for path in candidates if path.exists()), modality_path)
        if not modality_path.exists():
            raise FileNotFoundError(f"Missing image modality for {self.samples[idx][0]}. Tried: {modality_path}")     
   
        modality = Image.open(modality_path).convert("L")
        if modality.size != rgb.size:
            if self.modality_size_mismatch == "error":
                raise ValueError(
                    f"Image modality/RGB size mismatch for {self.samples[idx][0]} and {modality_path}: "
                    f"modality={modality.size}, rgb={rgb.size}"
                )  
            modality = modality.resize(rgb.size, resample=Image.BILINEAR)
        return modality

    def _resolve_npy_path(self, idx: int) -> Path:
        rel = self._relative_image_path(idx)     
        npy_root = Path(self.npy_folder)
        npy_path = npy_root / rel.with_suffix(".npy")
        npz_path = npy_root / rel.with_suffix(".npz")
        if npy_path.exists():
            return npy_path   
        if npz_path.exists(): 
            return npz_path
        raise FileNotFoundError(f"Missing NPY/NPZ modality for {self.samples[idx][0]}. Tried: {npy_path} and {npz_path}")

    @staticmethod
    def _read_npy(path: Path) -> np.ndarray:
        raw = np.load(path, allow_pickle=True) 
        if type(raw) is np.lib.npyio.NpzFile:     
            try:  
                keys = list(raw.files)   
                if "arr_0" in keys:     
                    array = raw["arr_0"]    
                elif len(keys) == 1:     
                    array = raw[keys[0]] 
                else:
                    raise ValueError(f"NPZ file with multiple arrays must contain key 'arr_0', got keys={keys} for {path}")    
            finally:   
                raw.close()   
        else:
            array = raw
        return np.asarray(array)

    def _load_npy_modality(self, idx: int, rgb: Image.Image) -> torch.Tensor:
        npy_path = self._resolve_npy_path(idx)
        array = np.squeeze(self._read_npy(npy_path))
        if array.ndim == 2:   
            tensor = torch.from_numpy(array.astype(np.float32, copy=False)).unsqueeze(0)    
        elif array.ndim == 3 and array.shape[0] == 1:
            tensor = torch.from_numpy(array.astype(np.float32, copy=False))    
        elif array.ndim == 3 and array.shape[-1] == 1:
            tensor = torch.from_numpy(array[..., 0].astype(np.float32, copy=False)).unsqueeze(0)
        else: 
            raise ValueError(f"NPY modality must be 2D or single-channel, got shape {tuple(array.shape)} for {npy_path}")
   
        rgb_w, rgb_h = rgb.size
        npy_h, npy_w = tensor.shape[-2:]    
        if (npy_h, npy_w) != (rgb_h, rgb_w):
            if self.modality_size_mismatch == "error":
                raise ValueError(     
                    f"NPY/RGB size mismatch for {self.samples[idx][0]} and {npy_path}: " 
                    f"npy=({npy_h}, {npy_w}), rgb=({rgb_h}, {rgb_w})"
                )
            tensor = F.interpolate(    
                tensor.unsqueeze(0),    
                size=(rgb_h, rgb_w),
                mode="bilinear",
                align_corners=False, 
            ).squeeze(0)
        return tensor
