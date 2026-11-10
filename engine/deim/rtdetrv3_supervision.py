"""Small tensor helpers shared by RT-DETRv3 training-only supervision."""

from __future__ import annotations     

import torch 


def make_mgsa_query_mask(num_queries: int, mask_ratio: float, generator=None, device=None) -> torch.Tensor:     
    """Return a random boolean self-attention mask (True means blocked)."""
    if not 0 <= mask_ratio < 1: 
        raise ValueError("mask_ratio must satisfy 0 <= mask_ratio < 1")
    if device is None and generator is not None:    
        device = getattr(generator, "device", None)
    mask = torch.rand((num_queries, num_queries), generator=generator, device=device) < mask_ratio  
    mask.fill_diagonal_(False)
    return mask
    

def build_group_attn_mask(group_masks: list[torch.Tensor], device=None) -> torch.Tensor:  
    """Place local group masks on a blocked block-diagonal attention mask."""
    if not group_masks:  
        return torch.zeros((0, 0), dtype=torch.bool, device=device)  
    if device is None: 
        device = group_masks[0].device    
    sizes = []
    for mask in group_masks:
        if mask.ndim != 2 or mask.shape[0] != mask.shape[1]:     
            raise ValueError("each group attention mask must be square")  
        sizes.append(mask.shape[0])    
    result = torch.ones((sum(sizes), sum(sizes)), dtype=torch.bool, device=device)
    offset = 0
    for mask, size in zip(group_masks, sizes): 
        result[offset:offset + size, offset:offset + size] = mask.to(device=device, dtype=torch.bool)
        offset += size
    return result    
    
 
def repeat_targets(targets: list[dict], repeat: int) -> list[dict]:
    """Tile per-instance target tensors while preserving image-level metadata."""     
    if repeat < 1:     
        raise ValueError("repeat must be >= 1")     
    result = [] 
    for target in targets:
        copied = {**target}
        count = len(target.get("labels", ()))
        for key in ("labels", "boxes", "masks"):    
            value = target.get(key)
            if torch.is_tensor(value) and value.ndim and value.shape[0] == count:
                if count:
                    copied[key] = value.repeat((repeat,) + (1,) * (value.ndim - 1))   
                else:  
                    copied[key] = value   
        result.append(copied)     
    return result   

 
__all__ = ["make_mgsa_query_mask", "build_group_attn_mask", "repeat_targets"]
