import math
from typing import Optional
     
import numpy as np
import torch


def _get_covariance_matrix(boxes: torch.Tensor):    
    gbbs = torch.cat((boxes[..., 2:4].pow(2) / 12, boxes[..., 4:5]), dim=-1)     
    a, b, c = gbbs.split(1, dim=-1)
    cos = c.cos()
    sin = c.sin() 
    cos2 = cos.pow(2)
    sin2 = sin.pow(2)
    return a * cos2 + b * sin2, a * sin2 + b * cos2, (a - b) * cos * sin
    

def probiou(obb1: torch.Tensor, obb2: torch.Tensor, CIoU: bool = False, eps: float = 1e-7) -> torch.Tensor:
    x1, y1 = obb1[..., :2].split(1, dim=-1)    
    x2, y2 = obb2[..., :2].split(1, dim=-1)     
    a1, b1, c1 = _get_covariance_matrix(obb1)    
    a2, b2, c2 = _get_covariance_matrix(obb2)  
 
    denom = (a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps
    t1 = (((a1 + a2) * (y1 - y2).pow(2) + (b1 + b2) * (x1 - x2).pow(2)) / denom) * 0.25
    t2 = (((c1 + c2) * (x2 - x1) * (y1 - y2)) / denom) * 0.5   
    det = ((a1 * b1 - c1.pow(2)).clamp(0) * (a2 * b2 - c2.pow(2)).clamp(0)).sqrt() 
    t3 = (((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2)) / (4 * det + eps) + eps).log() * 0.5
    bd = (t1 + t2 + t3).clamp(eps, 100.0)
    hd = (1.0 - (-bd).exp() + eps).sqrt()     
    iou = 1 - hd
    if CIoU: 
        w1, h1 = obb1[..., 2:4].split(1, dim=-1)
        w2, h2 = obb2[..., 2:4].split(1, dim=-1)
        v = (4 / math.pi**2) * ((w2 / h2).atan() - (w1 / h1).atan()).pow(2)   
        with torch.no_grad():
            alpha = v / (v - iou + (1 + eps))
        return iou - v * alpha
    return iou  
    

def batch_probiou(obb1: torch.Tensor | np.ndarray, obb2: torch.Tensor | np.ndarray, eps: float = 1e-7) -> torch.Tensor:  
    obb1 = torch.from_numpy(obb1) if isinstance(obb1, np.ndarray) else obb1  
    obb2 = torch.from_numpy(obb2) if isinstance(obb2, np.ndarray) else obb2     
    if obb1.numel() == 0 or obb2.numel() == 0:
        return obb1.new_zeros((obb1.shape[0], obb2.shape[0]))  

    x1, y1 = obb1[..., :2].split(1, dim=-1)
    x2, y2 = (x.squeeze(-1)[None] for x in obb2[..., :2].split(1, dim=-1))     
    a1, b1, c1 = _get_covariance_matrix(obb1) 
    a2, b2, c2 = (x.squeeze(-1)[None] for x in _get_covariance_matrix(obb2))     
 
    denom = (a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps
    t1 = (((a1 + a2) * (y1 - y2).pow(2) + (b1 + b2) * (x1 - x2).pow(2)) / denom) * 0.25   
    t2 = (((c1 + c2) * (x2 - x1) * (y1 - y2)) / denom) * 0.5
    det = ((a1 * b1 - c1.pow(2)).clamp(0) * (a2 * b2 - c2.pow(2)).clamp(0)).sqrt()
    t3 = (((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2)) / (4 * det + eps) + eps).log() * 0.5     
    bd = (t1 + t2 + t3).clamp(eps, 100.0)
    hd = (1.0 - (-bd).exp() + eps).sqrt()    
    return 1 - hd  
  
    
def scale_obb_to_image(boxes: torch.Tensor, image_sizes: torch.Tensor, angle_factor: float) -> torch.Tensor:  
    sizes = image_sizes.to(device=boxes.device, dtype=boxes.dtype) 
    xywh_scale = sizes[:, None, [0, 1, 0, 1]]
    out = boxes.clone() 
    out[..., :4] = out[..., :4] * xywh_scale
    out[..., 4] = out[..., 4] * float(angle_factor)
    return out


def scale_obb_from_resize_pad(
    boxes: torch.Tensor,    
    orig_target_sizes: torch.Tensor,  
    resize_pad: Optional[dict],
    angle_factor: float, 
) -> torch.Tensor:
    if resize_pad is None:     
        return scale_obb_to_image(boxes, orig_target_sizes, angle_factor)     

    input_sizes = resize_pad["size"].to(device=boxes.device, dtype=boxes.dtype)  
    padding = resize_pad["padding"].to(device=boxes.device, dtype=boxes.dtype)
    orig_sizes = orig_target_sizes.to(device=boxes.device, dtype=boxes.dtype) 

    out = boxes.clone()
    out[..., :4] = out[..., :4] * input_sizes[:, None, [0, 1, 0, 1]]     
    out[..., 0] = out[..., 0] - padding[:, None, 0]
    out[..., 1] = out[..., 1] - padding[:, None, 1]
    
    resized_sizes = input_sizes - padding[:, [0, 1]] - padding[:, [2, 3]]    
    scales = resized_sizes / orig_sizes
    out[..., :4] = out[..., :4] / scales[:, None, [0, 1, 0, 1]].clamp(min=1e-12)    
    out[..., 4] = out[..., 4] * float(angle_factor)   
    return out   


def xywhr_to_poly(boxes: torch.Tensor) -> torch.Tensor:   
    cx, cy, w, h, angle = boxes.unbind(-1)
    cos = angle.cos()  
    sin = angle.sin()  
    dw = w * 0.5   
    dh = h * 0.5   
    local = boxes.new_tensor([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
    x = local[:, 0] * dw[..., None]
    y = local[:, 1] * dh[..., None] 
    px = x * cos[..., None] - y * sin[..., None] + cx[..., None]   
    py = x * sin[..., None] + y * cos[..., None] + cy[..., None]
    return torch.stack((px, py), dim=-1).reshape(*boxes.shape[:-1], 8)

     
def xywhr_to_corners(boxes: torch.Tensor) -> torch.Tensor:    
    return xywhr_to_poly(boxes).reshape(*boxes.shape[:-1], 4, 2)

    
def pairwise_chamfer_cost(pred_pixel: torch.Tensor, tgt_pixel: torch.Tensor, image_size: torch.Tensor) -> torch.Tensor:
    if pred_pixel.numel() == 0 or tgt_pixel.numel() == 0:
        return pred_pixel.new_zeros((pred_pixel.shape[0], tgt_pixel.shape[0])) 

    scale = image_size.to(device=pred_pixel.device, dtype=pred_pixel.dtype)[[0, 1, 0, 1, 0]]     
    scale = scale.clone()
    scale[-1] = 1.0     
    pred_corners = xywhr_to_corners(pred_pixel / scale)
    tgt_corners = xywhr_to_corners(tgt_pixel.to(device=pred_pixel.device, dtype=pred_pixel.dtype) / scale)

    pred_points = pred_corners[:, None, :, None, :]
    tgt_points = tgt_corners[None, :, None, :, :]  
    point_dist = torch.linalg.vector_norm(pred_points - tgt_points, dim=-1)
    pred_to_tgt = point_dist.min(dim=3).values.mean(dim=2)   
    tgt_to_pred = point_dist.min(dim=2).values.mean(dim=2)
    return pred_to_tgt + tgt_to_pred

  
def normalized_obb_to_pixel(boxes: torch.Tensor, image_sizes: torch.Tensor, angle_factor: float) -> torch.Tensor:
    if boxes.numel() == 0: 
        return boxes.clone()     
    sizes = image_sizes.to(device=boxes.device, dtype=boxes.dtype)  
    scale = sizes[[0, 1, 0, 1]]
    out = boxes.clone()
    out[..., :4] = out[..., :4] * scale    
    out[..., 4] = out[..., 4] * float(angle_factor)
    return out
