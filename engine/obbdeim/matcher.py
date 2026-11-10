import os  

import torch    
import torch.nn as nn 
import torch.nn.functional as F     
from scipy.optimize import linear_sum_assignment
     
from ..core import register  

from .box_ops import batch_probiou, normalized_obb_to_pixel, xywhr_to_corners
  
VIS_MATCH = False  
VIS_MATCH_DIR = "vis_match"    
VIS_MATCH_EVERY_N_EPOCHS = 1   
_vis_match_call_idx = 0
     
@register()     
class OBBHungarianMatcher(nn.Module):
    __share__ = ["use_focal_loss", "angle_factor"]

    def __init__(  
        self,
        weight_dict=None,
        use_focal_loss=True,    
        alpha=0.25,     
        gamma=2.0,     
        angle_factor=3.141592653589793,     
    ): 
        super().__init__()     
        weight_dict = weight_dict or {"cost_class": 2, "cost_bbox": 5, "cost_probiou": 2}  
        self.cost_class = weight_dict.get("cost_class", 0)  
        self.cost_bbox = weight_dict.get("cost_bbox", 0) 
        self.cost_probiou = weight_dict.get("cost_probiou", 0)
        self.use_focal_loss = use_focal_loss
        self.alpha = alpha  
        self.gamma = gamma 
        self.angle_factor = angle_factor

    @staticmethod    
    def _target_size(target, device, dtype):
        if "orig_size" in target:
            return target["orig_size"].to(device=device, dtype=dtype) 
        if "size" in target:
            return target["size"].to(device=device, dtype=dtype) 
        return torch.tensor([1.0, 1.0], device=device, dtype=dtype)     

    @torch.no_grad()
    def forward(self, outputs, targets, return_topk=False, epoch=0, num_queries_list=None):   
        del return_topk     
        bs, num_queries = outputs["pred_logits"].shape[:2]    
        device = outputs["pred_logits"].device     

        sizes = [len(v["boxes"]) for v in targets]    
        if sum(sizes) == 0:    
            empty = [
                (
                    torch.zeros(0, dtype=torch.int64, device=device),    
                    torch.zeros(0, dtype=torch.int64, device=device),
                )    
                for _ in targets 
            ]
            return {"indices": empty}
   
        out_prob_all = outputs["pred_logits"]    
        if self.use_focal_loss:
            out_prob = out_prob_all.sigmoid()  
        else:   
            out_prob = out_prob_all.softmax(-1)     
        out_bbox = outputs["pred_boxes"]

        cost_matrices = []     
        for img_idx, (pred_prob, pred_bbox, target) in enumerate(zip(out_prob, out_bbox, targets)):
            num_targets = len(target["boxes"])    
            if num_targets == 0: 
                cost = pred_bbox.new_zeros((num_queries, 0))
                cost_matrices.append(cost.cpu())
                continue    
  
            target_ids = target["labels"]     
            if self.use_focal_loss:   
                pred_prob_for_targets = pred_prob[:, target_ids]
                neg_cost_class = (1 - self.alpha) * (pred_prob_for_targets ** self.gamma) * (-(1 - pred_prob_for_targets + 1e-8).log())
                pos_cost_class = self.alpha * ((1 - pred_prob_for_targets) ** self.gamma) * (-(pred_prob_for_targets + 1e-8).log())     
                cost_class = pos_cost_class - neg_cost_class 
            else:
                cost_class = -pred_prob[:, target_ids]     

            target_boxes = target["boxes"].to(pred_bbox.device, pred_bbox.dtype)
            if self.cost_bbox:  
                cost_bbox = torch.cdist(pred_bbox[..., :4], target_boxes[..., :4], p=1)
            else:
                cost_bbox = pred_bbox.new_zeros((num_queries, num_targets))

            if self.cost_probiou:
                image_size = self._target_size(target, pred_bbox.device, pred_bbox.dtype)
                pred_pixel = normalized_obb_to_pixel(pred_bbox, image_size, self.angle_factor)    
                tgt_pixel = normalized_obb_to_pixel(target_boxes, image_size, self.angle_factor)
                cost_probiou = -batch_probiou(pred_pixel, tgt_pixel)
            else:     
                cost_probiou = pred_bbox.new_zeros((num_queries, num_targets))     

            cost = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_probiou * cost_probiou     
            cost = torch.nan_to_num(cost, nan=1.0, posinf=1e6, neginf=-1e6).cpu()     
            if num_queries_list:    
                cost[int(num_queries_list[img_idx]) :] = 10000.0
            cost_matrices.append(cost)

        indices_pre = [linear_sum_assignment(cost) for cost in cost_matrices]
        indices = [
            (
                torch.as_tensor(i, dtype=torch.int64, device=device),    
                torch.as_tensor(j, dtype=torch.int64, device=device), 
            )
            for i, j in indices_pre
        ] 
        if VIS_MATCH and (epoch % VIS_MATCH_EVERY_N_EPOCHS == 0):
            self._visualize_matching(outputs, targets, indices)    
        return {"indices": indices} 

    @torch.no_grad() 
    def _visualize_matching(self, outputs, targets, indices):
        global _vis_match_call_idx
        from PIL import Image, ImageDraw
     
        os.makedirs(VIS_MATCH_DIR, exist_ok=True)
        pred_boxes = outputs["pred_boxes"]  # (bs, num_queries, 5)
 
        for img_idx, (target, (pred_idx, tgt_idx)) in enumerate(zip(targets, indices)):
            if len(target["boxes"]) == 0:    
                continue

            image_size = self._target_size(target, pred_boxes.device, pred_boxes.dtype)  # [W, H]  
            W_raw, H_raw = int(image_size[0].item()), int(image_size[1].item())   
            scale_factor = max(1.0, 512.0 / max(W_raw, H_raw)) 
            draw_W = max(1, int(W_raw * scale_factor))
            draw_H = max(1, int(H_raw * scale_factor))

            pred_pixel = normalized_obb_to_pixel(pred_boxes[img_idx], image_size, self.angle_factor)
            tgt_pixel = normalized_obb_to_pixel(
                target["boxes"].to(pred_boxes.device, pred_boxes.dtype),   
                image_size,  
                self.angle_factor,
            )

            pred_corners = xywhr_to_corners(pred_pixel) * scale_factor   # (num_queries, 4, 2)
            tgt_corners = xywhr_to_corners(tgt_pixel) * scale_factor     # (num_gt, 4, 2)     
            pred_cx = (pred_pixel[:, 0] * scale_factor).cpu()
            pred_cy = (pred_pixel[:, 1] * scale_factor).cpu()   
            tgt_cx = (tgt_pixel[:, 0] * scale_factor).cpu()
            tgt_cy = (tgt_pixel[:, 1] * scale_factor).cpu()
    
            canvas = Image.new("RGB", (draw_W, draw_H), color=(255, 255, 255))     
            draw = ImageDraw.Draw(canvas)  
 
            matched_pred_set = set(pred_idx.tolist())   
            num_queries = pred_corners.shape[0]
    
            # Unmatched pred OBBs (gray)    
            for pi in range(num_queries):   
                if pi in matched_pred_set:  
                    continue
                pts = [tuple(p) for p in pred_corners[pi].cpu().tolist()] 
                draw.line(pts + [pts[0]], fill=(180, 180, 180), width=1)

            # GT OBBs (green)
            for j in range(len(tgt_corners)):
                pts = [tuple(p) for p in tgt_corners[j].cpu().tolist()]
                draw.line(pts + [pts[0]], fill=(0, 200, 0), width=2)
                draw.text((tgt_cx[j].item(), tgt_cy[j].item()), f"gt_{j}", fill=(0, 150, 0))

            # Matched pred OBBs (red) + connecting lines (blue)    
            for pi, ti in zip(pred_idx.tolist(), tgt_idx.tolist()):    
                pts = [tuple(p) for p in pred_corners[pi].cpu().tolist()]
                draw.line(pts + [pts[0]], fill=(220, 50, 50), width=2)
                draw.text((pred_cx[pi].item(), pred_cy[pi].item()), f"pred_{pi}", fill=(180, 30, 30))
                draw.line(
                    [(pred_cx[pi].item(), pred_cy[pi].item()), (tgt_cx[ti].item(), tgt_cy[ti].item())],   
                    fill=(50, 100, 220),    
                    width=1,
                )     
 
            save_path = os.path.join(VIS_MATCH_DIR, f"call{_vis_match_call_idx:04d}_img{img_idx}.png")     
            canvas.save(save_path)

        _vis_match_call_idx += 1     
    
    def set_epoch(self, epoch):
        self.epoch = epoch     
