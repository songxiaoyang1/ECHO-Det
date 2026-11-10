"""Optional Deformable-DETR CUDA backend for DEIM default deformable attention."""
   
from __future__ import annotations   

from functools import lru_cache  
from typing import Sequence 

import torch  
from torch.autograd import Function
from torch.autograd.function import once_differentiable    
 
from ..deim.utils import deformable_attention_core_func_v2

    
def _load_msda_extension(): 
    try:     
        import MultiScaleDeformableAttention as msda    
    except (ImportError, OSError):
        return None    
    if not hasattr(msda, "ms_deform_attn_forward") or not hasattr(msda, "ms_deform_attn_backward"):
        return None  
    return msda 
 
   
class _MSDeformAttnFunction(Function):     
    @staticmethod
    def forward(ctx, value, spatial_shapes, level_start_index, sampling_locations, attention_weights, im2col_step):
        msda = _load_msda_extension()  
        if msda is None:
            raise RuntimeError("MultiScaleDeformableAttention extension is not available") 
        ctx.im2col_step = im2col_step  
        output = msda.ms_deform_attn_forward(  
            value,
            spatial_shapes,
            level_start_index,    
            sampling_locations,  
            attention_weights,   
            im2col_step, 
        )  
        ctx.save_for_backward(value, spatial_shapes, level_start_index, sampling_locations, attention_weights)     
        return output
   
    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        msda = _load_msda_extension()
        if msda is None:
            raise RuntimeError("MultiScaleDeformableAttention extension is not available")   
        value, spatial_shapes, level_start_index, sampling_locations, attention_weights = ctx.saved_tensors
        grad_value, grad_sampling_loc, grad_attn_weight = msda.ms_deform_attn_backward(    
            value,
            spatial_shapes,
            level_start_index,
            sampling_locations,  
            attention_weights,
            grad_output,
            ctx.im2col_step,
        )
        return grad_value, None, None, grad_sampling_loc, grad_attn_weight, None 
    
     
def _as_spatial_shapes_tensor(value_spatial_shapes, device):
    if torch.is_tensor(value_spatial_shapes):     
        return value_spatial_shapes.to(device=device, dtype=torch.long) 
    return torch.tensor(value_spatial_shapes, device=device, dtype=torch.long)


def _value_tuple_to_msda_value(value):
    if torch.is_tensor(value):  
        if value.dim() != 4:     
            raise ValueError(f"Expected value tensor [B, S, H, D], got {tuple(value.shape)}")
        return value.contiguous()
    if not isinstance(value, (tuple, list)) or not value:   
        raise ValueError("Expected value to be a non-empty tuple/list or a [B, S, H, D] tensor")  
    return torch.cat(value, dim=-1).permute(0, 3, 1, 2).contiguous()


def _level_start_index(spatial_shapes):   
    area = spatial_shapes[:, 0] * spatial_shapes[:, 1]
    return torch.cat([area.new_zeros((1,)), area.cumsum(0)[:-1]])


# @lru_cache(maxsize=8)   
def _msda_extension_matches_default_core(device_type: str, dtype: torch.dtype):
    if device_type != "cuda" or dtype != torch.float32:  
        return False  
    if _load_msda_extension() is None or not torch.cuda.is_available():
        return False
    return True     

    # device = torch.device("cuda")     
    # generator = torch.Generator(device=device)    
    # generator.manual_seed(20260629)
    # batch, queries, heads, head_dim = 1, 3, 2, 4
    # num_points_list = [2, 2] 
    # spatial_shapes = torch.tensor([[2, 2], [1, 1]], device=device, dtype=torch.long)    
    # total = int((spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum().item())
    # value = torch.randn(batch, total, heads, head_dim, device=device, dtype=dtype, generator=generator)    
    # value_tuple = value.permute(0, 2, 3, 1).split((spatial_shapes[:, 0] * spatial_shapes[:, 1]).tolist(), dim=-1)    
    # total_points = sum(num_points_list)    
    # sampling_locations = torch.rand(
    #     batch,    
    #     queries,     
    #     heads,
    #     total_points, 
    #     2,   
    #     device=device,
    #     dtype=dtype,    
    #     generator=generator,   
    # )     
    # attention_weights = torch.randn(
    #     batch,
    #     queries,     
    #     heads,
    #     total_points,     
    #     device=device, 
    #     dtype=dtype,    
    #     generator=generator, 
    # ).softmax(-1)
 
    # fallback = _fallback_default_core(     
    #     value_tuple,     
    #     spatial_shapes,
    #     sampling_locations, 
    #     attention_weights,   
    #     num_points_list,
    # )
    # packed_locations, packed_weights = pack_flattened_points_by_level(  
    #     sampling_locations,
    #     attention_weights, 
    #     num_points_list,   
    # ) 
    # accelerated = _MSDeformAttnFunction.apply(
    #     value,
    #     spatial_shapes,  
    #     _level_start_index(spatial_shapes),     
    #     packed_locations, 
    #     packed_weights, 
    #     64,
    # )     
    # return torch.allclose(accelerated, fallback, atol=1e-6, rtol=1e-6)    

    
def should_use_deformable_detr_msda(value, sampling_locations, attention_weights):
    if (
        not sampling_locations.is_cuda
        or not attention_weights.is_cuda
        or _load_msda_extension() is None
    ):
        return False 

    msda_value = _value_tuple_to_msda_value(value)
    return (
        msda_value.is_cuda    
        and _msda_extension_matches_default_core(msda_value.device.type, torch.float32)   
    )

     
def _expand_reference_levels(reference_points, num_levels):
    if reference_points.shape[2] == 1:     
        return reference_points.repeat(1, 1, num_levels, 1)  
    if reference_points.shape[2] != num_levels: 
        raise ValueError(
            f"Expected reference point levels to be 1 or {num_levels}, got {reference_points.shape[2]}"
        )     
    return reference_points   
   

def _repeat_per_level(tensor, num_points_list): 
    return torch.cat(
        [tensor[:, :, level : level + 1].repeat(1, 1, points, 1) for level, points in enumerate(num_points_list)],    
        dim=2,
    )
     

def _fallback_default_core(value, spatial_shapes, sampling_locations, attention_weights, num_points_list):
    if torch.is_tensor(value):    
        return deformable_attention_core_func_v2(     
            value,
            spatial_shapes.tolist(),  
            sampling_locations,  
            attention_weights,
            list(num_points_list),
            method="default",    
            value_shape="reshape",
        )
    return deformable_attention_core_func_v2(  
        value,
        spatial_shapes.tolist(),    
        sampling_locations, 
        attention_weights,
        list(num_points_list), 
        method="default",
    )   

   
def build_default_sampling_locations(
    query,
    reference_points,
    sampling_offsets,
    value_spatial_shapes,     
    num_points_list: Sequence[int],
    num_levels: int,  
    offset_scale: float,
):     
    batch, queries = query.shape[:2]
    heads = sampling_offsets.shape[2]
    total_points = sum(num_points_list)
    if len(num_points_list) != num_levels:     
        raise ValueError(f"Expected {num_levels} num_points entries, got {len(num_points_list)}")
    if sampling_offsets.shape != (batch, queries, heads, total_points, 2):     
        raise ValueError("sampling_offsets has an unexpected shape")    

    spatial_shapes = _as_spatial_shapes_tensor(value_spatial_shapes, query.device).to(dtype=query.dtype)
    num_points_scale = torch.tensor(
        [1 / n for n in num_points_list for _ in range(n)],
        device=query.device,  
        dtype=query.dtype,
    ).reshape(1, 1, 1, total_points, 1) 
 
    if reference_points.shape[-1] == 2:
        reference_points = _expand_reference_levels(reference_points, num_levels)     
        offset_normalizer = spatial_shapes.flip([1]).reshape(1, 1, 1, num_levels, 1, 2)
        sampling_locations = (    
            reference_points.reshape(batch, queries, 1, num_levels, 1, 2)
            + sampling_offsets.reshape(batch, queries, heads, num_levels, -1, 2) / offset_normalizer
        )
        return sampling_locations.reshape(batch, queries, heads, total_points, 2)     
   
    if reference_points.shape[-1] == 4: 
        if reference_points.shape[2] == 1:     
            offset = sampling_offsets * num_points_scale * reference_points[:, :, None, :, 2:] * offset_scale
            return reference_points[:, :, None, :, :2] + offset
        reference_points = _expand_reference_levels(reference_points, num_levels)     
        reference_xy = _repeat_per_level(reference_points[..., :2], num_points_list)     
        reference_wh = _repeat_per_level(reference_points[..., 2:], num_points_list)    
        offset = sampling_offsets * num_points_scale * reference_wh[:, :, None] * offset_scale    
        return reference_xy[:, :, None] + offset    
     
    if reference_points.shape[-1] == 5:     
        reference_points = _expand_reference_levels(reference_points, num_levels)
        cos_angle = torch.cos(reference_points[..., 4:])  
        sin_angle = torch.sin(reference_points[..., 4:])    
        rot = torch.cat([cos_angle, -sin_angle, sin_angle, cos_angle], dim=-1)
        rot = rot.reshape(batch, queries, num_levels, 2, 2)
        wh = reference_points[..., 2:4] * offset_scale    
        rotated_wh = torch.einsum("bqlij,bqlj->bqli", rot, wh)
        rotated_wh = _repeat_per_level(rotated_wh, num_points_list)
        reference_xy = _repeat_per_level(reference_points[..., :2], num_points_list) 
        offset = sampling_offsets * num_points_scale * rotated_wh[:, :, None]
        return reference_xy[:, :, None] + offset 
    
    raise ValueError(f"Last dim of reference_points must be 2, 4, or 5, got {reference_points.shape[-1]}")
  
    
def pack_flattened_points_by_level(sampling_locations, attention_weights, num_points_list: Sequence[int]):
    batch, queries, heads, total_points, xy = sampling_locations.shape  
    if xy != 2:
        raise ValueError(f"Expected sampling_locations last dim 2, got {xy}")  
    if attention_weights.shape != (batch, queries, heads, total_points):   
        raise ValueError("attention_weights shape must match flattened sampling locations")
    if total_points != sum(num_points_list):     
        raise ValueError(f"Expected {sum(num_points_list)} points, got {total_points}")  

    num_levels = len(num_points_list)     
    if len(set(num_points_list)) == 1: 
        points = num_points_list[0]    
        return (   
            sampling_locations.reshape(batch, queries, heads, num_levels, points, 2).contiguous(),   
            attention_weights.reshape(batch, queries, heads, num_levels, points).contiguous(),
        )
     
    max_points = max(num_points_list)
    packed_locations = sampling_locations.new_zeros(batch, queries, heads, num_levels, max_points, 2)
    packed_weights = attention_weights.new_zeros(batch, queries, heads, num_levels, max_points)     
    start = 0
    for level, points in enumerate(num_points_list):
        end = start + points
        packed_locations[:, :, :, level, :points] = sampling_locations[:, :, :, start:end]
        packed_weights[:, :, :, level, :points] = attention_weights[:, :, :, start:end]    
        start = end    
    return packed_locations.contiguous(), packed_weights.contiguous()

     
def deformable_detr_msda_core_default(
    value,
    value_spatial_shapes,  
    sampling_locations,
    attention_weights,
    num_points_list: Sequence[int],
    im2col_step: int = 64,
): 
    spatial_shapes = _as_spatial_shapes_tensor(value_spatial_shapes, sampling_locations.device)    

    if (
        not sampling_locations.is_cuda
        or not attention_weights.is_cuda
        or _load_msda_extension() is None   
    ):
        return _fallback_default_core(
            value,
            spatial_shapes,  
            sampling_locations,     
            attention_weights,
            num_points_list,  
        )
     
    msda_value = _value_tuple_to_msda_value(value)     
    if (    
        not msda_value.is_cuda
        or not _msda_extension_matches_default_core(msda_value.device.type, torch.float32)     
    ):
        return _fallback_default_core(
            value,  
            spatial_shapes, 
            sampling_locations,  
            attention_weights,
            num_points_list,
        )   
  
    output_dtype = msda_value.dtype 
    msda_value = msda_value.to(dtype=torch.float32)
    sampling_locations = sampling_locations.to(dtype=torch.float32)    
    attention_weights = attention_weights.to(dtype=torch.float32)    
    packed_locations, packed_weights = pack_flattened_points_by_level(
        sampling_locations,
        attention_weights,    
        list(num_points_list),  
    )  
    level_start_index = _level_start_index(spatial_shapes)   
    output = _MSDeformAttnFunction.apply(
        msda_value, 
        spatial_shapes,  
        level_start_index,
        packed_locations,
        packed_weights,
        im2col_step,    
    )     
    return output.to(dtype=output_dtype)   
