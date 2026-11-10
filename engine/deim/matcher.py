"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved    
Modules to compute the matching cost and solve the corresponding LSAP.

Copyright (c) 2024 The D-FINE Authors All Rights Reserved.    
""" 
   
import os    
   
import torch
import torch.nn as nn
import torch.nn.functional as F 

from scipy.optimize import linear_sum_assignment
from typing import Dict
  
from .box_ops import (
    batch_dice_loss,
    batch_sigmoid_ce_loss, 
    box_cxcywh_to_xyxy,
    generalized_box_iou,     
    box_iou,
    generalized_box_inner_iou,     
)
from .utils import point_sample
     
from ..core import register
import numpy as np  

from ..logger_module import get_logger   
    
logger = get_logger(__name__)     

VIS_MATCH = False
VIS_MATCH_DIR = "vis_match"
VIS_MATCH_EVERY_N_EPOCHS = 1
_vis_match_call_idx = 0   

@register()
class HungarianMatcher(nn.Module):   
    """This class computes an assignment between the targets and the predictions of the network   
 
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions, 
    while the others are un-matched (and thus treated as non-objects).  
    """   

    __share__ = ['use_focal_loss', ]  

    def __init__(self, weight_dict, use_focal_loss=False, alpha=0.25, gamma=2.0,   
                 change_matcher=False, iou_order_alpha=1.0, matcher_change_epoch=10000,
                 mask_point_sample_ratio=None): 
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost     
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """ 
        super().__init__()    
        self.cost_class = weight_dict['cost_class']
        self.cost_bbox = weight_dict['cost_bbox']
        self.cost_giou = weight_dict['cost_giou']
        self.cost_mask_ce = weight_dict.get('cost_mask_ce', 0)     
        self.cost_mask_dice = weight_dict.get('cost_mask_dice', 0)
        self.mask_point_sample_ratio = mask_point_sample_ratio   
 
        self.change_matcher = change_matcher
        self.iou_order_alpha = iou_order_alpha 
        self.matcher_change_epoch = matcher_change_epoch
        if self.change_matcher: 
            logger.info(f"Using the new matching cost with iou_order_alpha = {iou_order_alpha} at epoch {matcher_change_epoch}")    
     
        self.use_focal_loss = use_focal_loss
        self.alpha = alpha     
        self.gamma = gamma   
        self.epoch = -1   
  
        assert (   
            self.cost_class != 0
            or self.cost_bbox != 0   
            or self.cost_giou != 0   
            or self.cost_mask_ce != 0
            or self.cost_mask_dice != 0     
        ), "all costs cant be 0"
  
    def _has_mask_cost(self, outputs, targets):    
        return (  
            self.mask_point_sample_ratio
            and 'pred_masks' in outputs
            and len(targets) > 0    
            and all('masks' in target for target in targets)
            and sum(len(target['masks']) for target in targets) > 0  
            and (self.cost_mask_ce != 0 or self.cost_mask_dice != 0)   
        )
 
    def _compute_mask_cost(self, outputs, targets):   
        out_masks = outputs['pred_masks'].flatten(0, 1)     
        tgt_masks = torch.cat([target['masks'] for target in targets]).to(out_masks.dtype)     
        num_points = max(
            out_masks.shape[-2],
            out_masks.shape[-2] * out_masks.shape[-1] // self.mask_point_sample_ratio,
        )  
        point_coords = torch.rand(1, num_points, 2, device=out_masks.device)
        pred_masks_logits = point_sample(  
            out_masks.unsqueeze(1),
            point_coords.repeat(out_masks.shape[0], 1, 1),
            align_corners=False,   
        ).squeeze(1)
        tgt_masks_flat = point_sample(   
            tgt_masks.unsqueeze(1),
            point_coords.repeat(tgt_masks.shape[0], 1, 1), 
            align_corners=False,
            mode='nearest',
        ).squeeze(1)
        return (
            batch_sigmoid_ce_loss(pred_masks_logits, tgt_masks_flat),
            batch_dice_loss(pred_masks_logits, tgt_masks_flat),
        )   
    
    @torch.no_grad()  
    def forward(self, outputs: Dict[str, torch.Tensor], targets, return_topk=False, epoch=0, num_queries_list=None):   
        """ Performs the matching

        Params:  
            outputs: This is a dict that contains at least these entries:  
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates 

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:  
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates  

        Returns:     
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order) 
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)    
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch   
        if self.use_focal_loss:     
            out_prob = F.sigmoid(outputs["pred_logits"].flatten(0, 1))
        else:  
            out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes] 

        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4] 
        has_mask_cost = self._has_mask_cost(outputs, targets)
        if has_mask_cost:  
            cost_mask_ce, cost_mask_dice = self._compute_mask_cost(outputs, targets) 
  
        # Also concat the target labels and boxes    
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])
     
        # is_small_obj = self.identify_small_objects(tgt_bbox)  
        # is_small_obj = is_small_obj.unsqueeze(0).repeat(out_bbox.size(0), 1)   

        if self.change_matcher and epoch >= self.matcher_change_epoch:
            # Compute the class_score     
            class_score = out_prob[:, tgt_ids]  # shape = [batch_size * num_queries, gt num within a batch]  

            # # Compute iou
            bbox_iou, _ = box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))     

            # Final cost matrix     
            C = (-1) * (class_score * torch.pow(bbox_iou, self.iou_order_alpha))   
            if has_mask_cost: 
                C = C + self.cost_mask_ce * cost_mask_ce + self.cost_mask_dice * cost_mask_dice  
        else:
            # Compute the classification cost. Contrary to the loss, we don't use the NLL,
            # but approximate it in 1 - proba[target class].
            # The 1 is a constant that doesn't change the matching, it can be ommitted.
            if self.use_focal_loss:    
                out_prob = out_prob[:, tgt_ids]
                neg_cost_class = (1 - self.alpha) * (out_prob ** self.gamma) * (-(1 - out_prob + 1e-8).log())
                pos_cost_class = self.alpha * ((1 - out_prob) ** self.gamma) * (-(out_prob + 1e-8).log())
                cost_class = pos_cost_class - neg_cost_class    
            else:
                # cost_class = -out_prob[:, tgt_ids]
   
                out_prob = out_prob[:, tgt_ids]     
                neg_cost_class = out_prob * (-(1 - out_prob + 1e-8).log())   
                pos_cost_class = (1 - out_prob) * (-(out_prob + 1e-8).log())
                cost_class = pos_cost_class - neg_cost_class
     
            # Compute the L1 cost between boxes   
            cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
 
            # Compute the giou cost betwen boxes   
            cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))  
            # cost_inner_giou = -generalized_box_inner_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))  

            # cost_giou[is_small_obj] = cost_inner_giou[is_small_obj]
    
            # Final cost matrix 3 * self.cost_bbox + 2 * self.cost_class + self.cost_giou 
            C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou    
            if has_mask_cost:
                C = C + self.cost_mask_ce * cost_mask_ce + self.cost_mask_dice * cost_mask_dice
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        # FIXME，RT-DETR, different way to set NaN
        C = torch.nan_to_num(C, nan=1.0)
        if num_queries_list:    
            for i, qn in enumerate(num_queries_list):
                C[i, qn:] = 10000.0  
        indices_pre = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))] 
        indices = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices_pre]     

        # Compute topk indices
        if return_topk:   
            return {'indices_o2m': self.get_top_k_matches(C, sizes=sizes, k=return_topk, initial_indices=indices_pre)}
   
        if VIS_MATCH and (epoch % VIS_MATCH_EVERY_N_EPOCHS == 0):
            self._visualize_matching(outputs, targets, indices)

        return {'indices': indices} # , 'indices_o2m': C.min(-1)[1]}

    def get_top_k_matches(self, C, sizes, k=1, initial_indices=None):   
        indices_list = []     
        # C_original = C.clone() 
        for i in range(k):
            indices_k = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))] if i > 0 else initial_indices
            indices_list.append([
                (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
                for i, j in indices_k 
            ])    
            for c, idx_k in zip(C.split(sizes, -1), indices_k):
                idx_k = np.stack(idx_k) 
                c[:, idx_k] = 1e6 
        indices_list = [(torch.cat([indices_list[i][j][0] for i in range(k)], dim=0),    
                        torch.cat([indices_list[i][j][1] for i in range(k)], dim=0)) for j in range(len(sizes))]     
        # C.copy_(C_original)
        return indices_list    
     
    def set_epoch(self, epoch):   
        self.epoch = epoch

    @torch.no_grad()    
    def _visualize_matching(self, outputs, targets, indices):     
        global _vis_match_call_idx 
        from PIL import Image, ImageDraw  

        os.makedirs(VIS_MATCH_DIR, exist_ok=True)     
        pred_boxes = outputs["pred_boxes"]  # (bs, num_queries, 4) in cxcywh normalized
  
        for img_idx, (target, (pred_idx, tgt_idx)) in enumerate(zip(targets, indices)):
            if len(target["boxes"]) == 0:
                continue  

            # orig_size is [H, W] in DEIM   
            if "orig_size" in target:
                H_raw, W_raw = int(target["orig_size"][0].item()), int(target["orig_size"][1].item())     
            elif "size" in target:
                H_raw, W_raw = int(target["size"][0].item()), int(target["size"][1].item())    
            else:
                H_raw, W_raw = 640, 640   

            scale_factor = max(1.0, 512.0 / max(W_raw, H_raw))
            draw_W = max(1, int(W_raw * scale_factor))
            draw_H = max(1, int(H_raw * scale_factor)) 
    
            def to_xyxy_pixel(boxes): 
                cx = boxes[:, 0] * W_raw * scale_factor    
                cy = boxes[:, 1] * H_raw * scale_factor     
                hw = boxes[:, 2] * W_raw * scale_factor / 2    
                hh = boxes[:, 3] * H_raw * scale_factor / 2   
                return torch.stack([cx - hw, cy - hh, cx + hw, cy + hh], dim=-1).cpu()     
  
            pred_xyxy = to_xyxy_pixel(pred_boxes[img_idx])
            tgt_boxes_dev = target["boxes"].to(pred_boxes.device, pred_boxes.dtype)     
            tgt_xyxy = to_xyxy_pixel(tgt_boxes_dev)    
            pred_cx = (pred_boxes[img_idx, :, 0] * W_raw * scale_factor).cpu()
            pred_cy = (pred_boxes[img_idx, :, 1] * H_raw * scale_factor).cpu()
            tgt_cx = (tgt_boxes_dev[:, 0] * W_raw * scale_factor).cpu()     
            tgt_cy = (tgt_boxes_dev[:, 1] * H_raw * scale_factor).cpu()  
   
            canvas = Image.new("RGB", (draw_W, draw_H), color=(255, 255, 255))
            draw = ImageDraw.Draw(canvas)
            matched_pred_set = set(pred_idx.tolist())   
            num_queries = pred_xyxy.shape[0]
 
            # Unmatched pred boxes (gray)   
            for pi in range(num_queries):     
                if pi in matched_pred_set:
                    continue
                x1, y1, x2, y2 = pred_xyxy[pi].tolist() 
                draw.rectangle([x1, y1, x2, y2], outline=(180, 180, 180), width=1)

            # GT boxes (green)   
            for j in range(len(tgt_xyxy)):    
                x1, y1, x2, y2 = tgt_xyxy[j].tolist()
                draw.rectangle([x1, y1, x2, y2], outline=(0, 200, 0), width=2)
                draw.text((tgt_cx[j].item(), tgt_cy[j].item()), f"gt_{j}", fill=(0, 150, 0))   
     
            # Matched pred boxes (red) + connecting lines (blue)  
            for pi, ti in zip(pred_idx.tolist(), tgt_idx.tolist()):  
                x1, y1, x2, y2 = pred_xyxy[pi].tolist()
                draw.rectangle([x1, y1, x2, y2], outline=(220, 50, 50), width=2)
                draw.text((pred_cx[pi].item(), pred_cy[pi].item()), f"pred_{pi}", fill=(180, 30, 30))
                draw.line(   
                    [(pred_cx[pi].item(), pred_cy[pi].item()), (tgt_cx[ti].item(), tgt_cy[ti].item())],
                    fill=(50, 100, 220),
                    width=1,
                )

            save_path = os.path.join(VIS_MATCH_DIR, f"call{_vis_match_call_idx:04d}_img{img_idx}.png")    
            canvas.save(save_path)
    
        _vis_match_call_idx += 1
   
    def identify_small_objects(self, boxes):
        """
        Identify small objects based on box area     

        Args:
            boxes: Tensor of shape [N, 4] in cxcywh format   

        Returns:
            Tensor of shape [N] with boolean values indicating small objects
        """ 
        self.small_object_threshold = 0.0005
        # Calculate area (w * h) for boxes in cxcywh format 
        areas = boxes[:, 2] * boxes[:, 3]
        # Objects with area less than threshold are considered small     
        is_small = areas < self.small_object_threshold 
        # logger.debug(f'is-small:{torch.sum(is_small) / is_small.size(0):.2f}')
        return is_small
