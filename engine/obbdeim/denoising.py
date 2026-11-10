import torch
  
from ..deim.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh
from ..deim.utils import inverse_sigmoid  

    
def get_contrastive_denoising_training_group(   
    targets,  
    num_classes,   
    num_queries,
    class_embed,
    num_denoising=100,     
    label_noise_ratio=0.5,   
    box_noise_scale=1.0,   
    angle_noise_scale=1.0,
    noise_mode="xyxy",
): 
    if noise_mode not in ("xyxy", "xyxya"):  
        raise ValueError(f"noise_mode must be 'xyxy' or 'xyxya', got {noise_mode!r}")
   
    if num_denoising <= 0:
        return None, None, None, None

    raw_num_gts = [len(t["labels"]) for t in targets]     
    device = targets[0]["labels"].device
    if max(raw_num_gts) == 0:
        return None, None, None, None
    max_gt_num = min(max(raw_num_gts), max(int(num_denoising), 1))  
    num_gts = [min(n, max_gt_num) for n in raw_num_gts]

    num_group = max(num_denoising // max_gt_num, 1)  
    bs = len(num_gts)    
    input_query_class = torch.full([bs, max_gt_num], num_classes, dtype=torch.int64, device=device)
    input_query_bbox = torch.zeros([bs, max_gt_num, 5], device=device)   
    pad_gt_mask = torch.zeros([bs, max_gt_num], dtype=torch.bool, device=device)

    for i, num_gt in enumerate(num_gts):
        if num_gt > 0:
            input_query_class[i, :num_gt] = targets[i]["labels"][:num_gt]
            input_query_bbox[i, :num_gt] = targets[i]["boxes"][:num_gt]
            pad_gt_mask[i, :num_gt] = True
    
    input_query_class = input_query_class.tile([1, 2 * num_group])  
    input_query_bbox = input_query_bbox.tile([1, 2 * num_group, 1])  
    input_query_bbox[..., 4:5] = input_query_bbox[..., 4:5].remainder(1.0)    
    pad_gt_mask = pad_gt_mask.tile([1, 2 * num_group])   

    negative_gt_mask = torch.zeros([bs, max_gt_num * 2, 1], device=device)  
    negative_gt_mask[:, max_gt_num:] = 1
    negative_gt_mask = negative_gt_mask.tile([1, num_group, 1])
    positive_gt_mask = (1 - negative_gt_mask).squeeze(-1).bool() & pad_gt_mask
    dn_positive_idx = torch.nonzero(positive_gt_mask)[:, 1]  
    dn_positive_idx = torch.split(dn_positive_idx, [n * num_group for n in num_gts])     
    num_denoising = int(max_gt_num * 2 * num_group)    
   
    if label_noise_ratio > 0:
        mask = torch.rand_like(input_query_class, dtype=torch.float) < (label_noise_ratio * 0.5)
        new_label = torch.randint_like(input_query_class, 0, num_classes)
        input_query_class = torch.where(mask & pad_gt_mask, new_label, input_query_class)
 
    if box_noise_scale > 0:  
        known_bbox = box_cxcywh_to_xyxy(input_query_bbox[..., :4])    
        diff = torch.tile(input_query_bbox[..., 2:4] * 0.5, [1, 1, 2]) * box_noise_scale
        rand_sign = torch.randint_like(input_query_bbox[..., :4], 0, 2) * 2.0 - 1.0
        rand_part = torch.rand_like(input_query_bbox[..., :4])
        rand_part = (rand_part + 1.0) * negative_gt_mask + rand_part * (1 - negative_gt_mask)     
        known_bbox = (known_bbox + rand_sign * rand_part * diff).clamp(0.0, 1.0)
        input_query_bbox[..., :4] = box_xyxy_to_cxcywh(known_bbox).clamp(0.0, 1.0)
 
    if noise_mode == "xyxya" and angle_noise_scale > 0:
        rand_sign_angle = torch.randint_like(input_query_bbox[..., 4:5], 0, 2) * 2.0 - 1.0
        rand_part_angle = torch.rand_like(input_query_bbox[..., 4:5])   
        rand_part_angle = (rand_part_angle + 1.0) * negative_gt_mask + rand_part_angle * (1 - negative_gt_mask)
        angle_noise = rand_sign_angle * rand_part_angle * input_query_bbox[..., 4:5] * angle_noise_scale / 18.0     
        input_query_bbox[..., 4:5] = (input_query_bbox[..., 4:5] + angle_noise).remainder(1.0)

    input_query_bbox[..., :4] = input_query_bbox[..., :4].clamp(min=1e-4, max=1 - 1e-4)   
    input_query_bbox[..., 4:5] = input_query_bbox[..., 4:5].remainder(1.0).clamp(min=1e-4, max=1 - 1e-4)     
    input_query_bbox_unact = inverse_sigmoid(input_query_bbox)     
    input_query_logits = class_embed(input_query_class)
 
    tgt_size = num_denoising + num_queries  
    attn_mask = torch.full([tgt_size, tgt_size], False, dtype=torch.bool, device=device)  
    attn_mask[num_denoising:, :num_denoising] = True
    for i in range(num_group):
        start = max_gt_num * 2 * i   
        end = max_gt_num * 2 * (i + 1)
        attn_mask[start:end, end:num_denoising] = True  
        attn_mask[start:end, :start] = True  
 
    dn_meta = {
        "dn_positive_idx": dn_positive_idx,     
        "dn_num_group": num_group, 
        "dn_num_split": [num_denoising, num_queries],
        "dn_num_gts": num_gts, 
    }     
    return input_query_logits, input_query_bbox_unact, attn_mask, dn_meta  
