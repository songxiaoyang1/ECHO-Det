import copy
import os    
     
import torch
import torch.nn as nn
import torch.nn.functional as F  
import torchvision 

from ..deim.box_ops import box_iou, box_cxcywh_to_xyxy
from ..deim.dfine_utils import bbox2distance
from ..deim.o2m_c import O2MCOBBLoss
from ..deim.rtdetrv3_supervision import repeat_targets
from ..core import register
from ..logger_module import get_logger   
from ..misc.dist_utils import get_world_size, is_dist_available_and_initialized
from .box_ops import normalized_obb_to_pixel, probiou, xywhr_to_corners   
    
try:  
    from mmcv.ops import box_iou_rotated as _mmcv_box_iou_rotated    
    _MMCV_AVAILABLE = True 
except ImportError:
    _MMCV_AVAILABLE = False
   
 
VIS_LOSS_MATCH = False
VIS_LOSS_MATCH_DIR = "vis_loss_match"
VIS_LOSS_MATCH_EVERY_N_EPOCHS = 1
_vis_loss_match_call_idx = 0   
logger = get_logger(__name__)
RED, RESET = "\033[91m", "\033[0m"


@register()
class OBBDEIMCriterion(nn.Module):
    __inject__ = ["matcher"]
    __share__ = ["num_classes", "angle_factor"]    

    def __init__(
        self, 
        matcher,   
        weight_dict, 
        losses,
        alpha=0.75,
        gamma=2.0,
        num_classes=80,     
        angle_factor=3.141592653589793,
        vfl_iou_type="hbox_iou", 
        no_weight_vfl_epoch=-1,
        no_weight_vfl_warmup_iter=500, 
        reg_max=32,
        o2m_t_loss_weight=1.0,  
    ):
        super().__init__()  
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses     
        self.alpha = alpha   
        self.gamma = gamma   
        self.num_classes = num_classes
        self.angle_factor = angle_factor
        if vfl_iou_type not in ("hbox_iou", "rbox_iou", "prob_iou"):    
            raise ValueError(f"Unsupported vfl_iou_type {vfl_iou_type!r}")  
        self.vfl_iou_type = vfl_iou_type
        self.model = None
        self.epoch = 0
        self.no_weight_vfl_epoch = no_weight_vfl_epoch
        self.no_weight_vfl_warmup_iter = no_weight_vfl_warmup_iter
        self.reg_max = reg_max 
        self.o2m_t_loss_weight = o2m_t_loss_weight  
        self.num_pos, self.num_neg = None, None     
        self.o2m_c_loss = O2MCOBBLoss(num_classes, angle_factor)
 
        if self.no_weight_vfl_epoch != -1:
            logger.info(RED + f"no_weight_vfl_epoch set {self.no_weight_vfl_epoch}" + RESET)    
            logger.info(RED + f"no_weight_vfl_warmup_iter set {self.no_weight_vfl_warmup_iter}" + RESET)     
   
    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])   
        return batch_idx, src_idx
  
    @staticmethod   
    def _target_size(target, device, dtype):    
        if "orig_size" in target:  
            return target["orig_size"].to(device=device, dtype=dtype)
        if "size" in target:
            return target["size"].to(device=device, dtype=dtype)
        return torch.tensor([1.0, 1.0], device=device, dtype=dtype)   
   
    def _matched_target_boxes(self, targets, indices, device, dtype): 
        boxes = []
        sizes = []     
        for target, (_, target_idx) in zip(targets, indices):
            if target_idx.numel():
                boxes.append(target["boxes"][target_idx].to(device=device, dtype=dtype))
            sizes.append(self._target_size(target, device, dtype)) 
        if boxes:    
            return torch.cat(boxes, dim=0), sizes
        return torch.zeros(0, 5, device=device, dtype=dtype), sizes

    def _matched_pixel_boxes(self, src_boxes, target_boxes, targets, indices):     
        src_pixel = [] 
        tgt_pixel = []
        start = 0
        for target, (src_idx, target_idx) in zip(targets, indices):
            count = len(target_idx)
            if count:
                image_size = self._target_size(target, src_boxes.device, src_boxes.dtype)    
                src_pixel.append(normalized_obb_to_pixel(src_boxes[start : start + count], image_size, self.angle_factor))
                tgt_pixel.append(normalized_obb_to_pixel(target_boxes[start : start + count], image_size, self.angle_factor))  
            start += count    
        if src_pixel:
            return torch.cat(src_pixel, dim=0), torch.cat(tgt_pixel, dim=0)  
        return src_boxes, target_boxes 
     
    @staticmethod
    def _cxcywh_to_xyxy(boxes):   
        cx, cy, w, h = boxes[..., :4].unbind(-1)
        return torch.stack((cx - w * 0.5, cy - h * 0.5, cx + w * 0.5, cy + h * 0.5), dim=-1)
     
    @staticmethod     
    def _polygon_signed_area(poly): 
        if poly.numel() == 0:
            return poly.new_tensor(0.0)
        x = poly[:, 0]    
        y = poly[:, 1]
        return 0.5 * (x * torch.roll(y, -1) - y * torch.roll(x, -1)).sum()

    @classmethod
    def _clip_polygon(cls, subject, clip):     
        output = [point for point in subject]
        clip_sign = torch.sign(cls._polygon_signed_area(clip))
        if clip_sign == 0:
            return []
   
        def inside(point, edge_start, edge_end):   
            edge = edge_end - edge_start  
            rel = point - edge_start   
            cross = edge[0] * rel[1] - edge[1] * rel[0]
            return cross * clip_sign >= -1e-7    

        def intersection(p1, p2, edge_start, edge_end):    
            dp = p2 - p1
            de = edge_end - edge_start
            denom = dp[0] * de[1] - dp[1] * de[0]    
            if torch.abs(denom) < 1e-12:    
                return p2
            diff = edge_start - p1     
            t = (diff[0] * de[1] - diff[1] * de[0]) / denom
            return p1 + t * dp  

        for i in range(clip.shape[0]):
            edge_start = clip[i]  
            edge_end = clip[(i + 1) % clip.shape[0]]
            input_points = output     
            output = []
            if not input_points:    
                break
            prev = input_points[-1] 
            for curr in input_points:
                curr_inside = inside(curr, edge_start, edge_end) 
                prev_inside = inside(prev, edge_start, edge_end)  
                if curr_inside:
                    if not prev_inside:  
                        output.append(intersection(prev, curr, edge_start, edge_end))  
                    output.append(curr)    
                elif prev_inside:
                    output.append(intersection(prev, curr, edge_start, edge_end))
                prev = curr 
        return output  

    @classmethod
    def _aligned_rbox_iou(cls, boxes1, boxes2):  
        if boxes1.numel() == 0: 
            return boxes1.new_zeros((0,))    
        if _MMCV_AVAILABLE:
            # clamp w/h to avoid degenerate boxes in the CUDA kernel
            b1 = boxes1.detach().clone()
            b2 = boxes2.detach().clone()   
            b1[:, 2:4].clamp_(min=1e-3)
            b2[:, 2:4].clamp_(min=1e-3)     
            return _mmcv_box_iou_rotated(b1, b2, mode="iou", aligned=True).squeeze(-1)   
        # fallback: pure-Python Sutherland-Hodgman polygon clipping    
        corners1 = xywhr_to_corners(boxes1)
        corners2 = xywhr_to_corners(boxes2)   
        ious = []
        for poly1, poly2 in zip(corners1, corners2):
            area1 = cls._polygon_signed_area(poly1).abs()
            area2 = cls._polygon_signed_area(poly2).abs()
            clipped = cls._clip_polygon(poly1, poly2)
            if len(clipped) < 3:
                inter = boxes1.new_tensor(0.0)    
            else:
                inter = cls._polygon_signed_area(torch.stack(clipped)).abs()     
            union = (area1 + area2 - inter).clamp(min=1e-12)
            ious.append(inter / union) 
        return torch.stack(ious)

    def _matched_vfl_scores(self, src_boxes, target_boxes, targets, indices):
        if self.vfl_iou_type == "hbox_iou":   
            ious, _ = box_iou(box_cxcywh_to_xyxy(src_boxes[..., :4]), box_cxcywh_to_xyxy(target_boxes[..., :4]))
            return torch.diag(ious).detach()

        src_pixel, tgt_pixel = self._matched_pixel_boxes(src_boxes, target_boxes, targets, indices)    
        if self.vfl_iou_type == "rbox_iou":
            return self._aligned_rbox_iou(src_pixel.detach(), tgt_pixel).detach().clamp(min=0)    
        return probiou(src_pixel, tgt_pixel).squeeze(-1).detach().clamp(min=0)
   
    def _get_vfl_weight_warmup_factor(self):
        if self.no_weight_vfl_epoch == -1:
            return 1.0  
        if self.epoch < self.no_weight_vfl_epoch:     
            return 0.0   
        if self.no_weight_vfl_warmup_iter <= 0:
            return 1.0
   
        step = getattr(self, "step", None)
        epoch_step = getattr(self, "epoch_step", None)
        if step is None or epoch_step is None:
            return 1.0
  
        epoch_step = max(int(epoch_step), 1)
        transition_iter = (self.epoch - self.no_weight_vfl_epoch) * epoch_step + int(step)
        transition_iter = max(transition_iter, 0)   
        return min(transition_iter / float(self.no_weight_vfl_warmup_iter), 1.0)

    def _vfl_weighted_bce_loss(self, src_logits, target_score, weight):     
        warmup_factor = self._get_vfl_weight_warmup_factor() 
        if warmup_factor <= 0:
            return F.binary_cross_entropy_with_logits(src_logits, target_score, reduction="none")
    
        loss_weighted = F.binary_cross_entropy_with_logits(src_logits, target_score, weight=weight, reduction="none")
        if warmup_factor >= 1:   
            return loss_weighted

        loss_no_weight = F.binary_cross_entropy_with_logits(src_logits, target_score, reduction="none")
        return loss_no_weight * (1 - warmup_factor) + loss_weighted * warmup_factor

    def loss_labels_vfl(self, outputs, targets, indices, num_boxes, values=None):
        assert 'pred_boxes' in outputs   
        idx = self._get_src_permutation_idx(indices)
        if values is None:     
            src_boxes = outputs['pred_boxes'][idx]   
            target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)   
            ious = self._matched_vfl_scores(src_boxes, target_boxes, targets, indices)  
        else:
            ious = values    

        src_logits = outputs['pred_logits']
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])     
        target_classes = torch.full(src_logits.shape[:2], self.num_classes, 
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o 
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]
 
        target_score_o = torch.zeros_like(target_classes, dtype=src_logits.dtype)
        target_score_o[idx] = ious.to(target_score_o.dtype)
        target_score = target_score_o.unsqueeze(-1) * target
    
        pred_score = F.sigmoid(src_logits).detach()  
        weight = (self.alpha * pred_score.pow(self.gamma)) * (1 - target) + target_score

        loss = self._vfl_weighted_bce_loss(src_logits, target_score, weight)  
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes    
        return {'loss_vfl': loss}

    def loss_boxes(self, outputs, targets, indices, num_boxes):     
        src_boxes = outputs["pred_boxes"][self._get_src_permutation_idx(indices)] 
        target_boxes, _ = self._matched_target_boxes(targets, indices, src_boxes.device, src_boxes.dtype)
        losses = {}

        loss_bbox = F.l1_loss(src_boxes[..., :4], target_boxes[..., :4], reduction="none")    
        angle_delta = (src_boxes[..., 4] - target_boxes[..., 4]) * self.angle_factor
        loss_angle = torch.atan2(torch.sin(2.0 * angle_delta), torch.cos(2.0 * angle_delta)).abs() / self.angle_factor
        losses['loss_bbox'] = (loss_bbox.sum() + loss_angle.sum()) / num_boxes   
  
        src_pixel, tgt_pixel = self._matched_pixel_boxes(src_boxes, target_boxes, targets, indices)    
        losses["loss_probiou"] = (1 - probiou(src_pixel, tgt_pixel).squeeze(-1)).sum() / num_boxes

        return losses

    def _ddf_query_weights(self, outputs, targets, indices, idx):  
        weight_targets_local = outputs["teacher_logits"].sigmoid().max(dim=-1)[0]   
  
        mask = torch.zeros_like(weight_targets_local, dtype=torch.bool)     
        mask[idx] = True
    
        if mask.any():
            src_boxes = outputs["pred_boxes"][idx]
            target_boxes, _ = self._matched_target_boxes(targets, indices, src_boxes.device, src_boxes.dtype) 
            quality = self._matched_vfl_scores(src_boxes, target_boxes, targets, indices)
            weight_targets_local[idx] = quality.reshape_as(weight_targets_local[idx]).to(weight_targets_local.dtype)
 
        if "is_dn" not in outputs or self.num_pos is None or self.num_neg is None:    
            batch_scale = 8 / outputs["pred_boxes"].shape[0] 
            self.num_pos = (mask.sum() * batch_scale).sqrt()
            self.num_neg = ((~mask).sum() * batch_scale).sqrt() 

        return mask, weight_targets_local.detach() 
   
    def _aggregate_ddf_loss(self, loss, mask):    
        pos_loss = loss[mask].mean() if mask.any() else loss.sum() * 0   
        neg_loss = loss[~mask].mean() if (~mask).any() else loss.sum() * 0
        return (pos_loss * self.num_pos + neg_loss * self.num_neg) / (self.num_pos + self.num_neg)     

    def loss_local(self, outputs, targets, indices, num_boxes, T=5):
        losses = {}
        if "pred_corners" not in outputs:
            return losses

        idx = self._get_src_permutation_idx(indices)    
        pred_corners = outputs["pred_corners"][idx]
        if pred_corners.numel() == 0:
            losses["loss_fgl"] = outputs["pred_corners"].sum() * 0
        else:    
            target_boxes, _ = self._matched_target_boxes(targets, indices, pred_corners.device, pred_corners.dtype)  
            target_xywh = target_boxes[..., :4]
            ref_points = outputs["ref_points"][idx][..., :4].detach()   
            pred_corners = pred_corners.reshape(-1, self.reg_max + 1)

            with torch.no_grad():
                target_corners, weight_right, weight_left = bbox2distance(  
                    ref_points,  
                    box_cxcywh_to_xyxy(target_xywh),
                    self.reg_max,    
                    outputs["reg_scale"],   
                    outputs["up"],
                )     

            ious = torch.diag(    
                box_iou( 
                    box_cxcywh_to_xyxy(outputs["pred_boxes"][idx][..., :4]),
                    box_cxcywh_to_xyxy(target_xywh),  
                )[0] 
            )  
            weight_targets = ious[:, None].repeat(1, 4).reshape(-1).detach()
            losses["loss_fgl"] = self.unimodal_distribution_focal_loss(
                pred_corners,     
                target_corners, 
                weight_right,
                weight_left,    
                weight_targets,
                avg_factor=num_boxes,
            )
        if "teacher_logits" not in outputs:
            return losses
  
        mask, query_weights = self._ddf_query_weights(outputs, targets, indices, idx) 
 
        if "teacher_corners" in outputs:
            pred_corners_all = outputs["pred_corners"].reshape(-1, self.reg_max + 1) 
            target_corners_all = outputs["teacher_corners"].reshape(-1, self.reg_max + 1)  
            corner_weights = query_weights.unsqueeze(-1).repeat(1, 1, 4).reshape(-1) 
            corner_mask = mask.unsqueeze(-1).repeat(1, 1, 4).reshape(-1) 
            loss_ddf = ( 
                corner_weights   
                * (T ** 2)
                * nn.KLDivLoss(reduction="none")(  
                    F.log_softmax(pred_corners_all / T, dim=1),
                    F.softmax(target_corners_all.detach() / T, dim=1),    
                ).sum(-1)
            )  
            losses["loss_ddf"] = self._aggregate_ddf_loss(loss_ddf, corner_mask)
    
        if "teacher_boxes" in outputs:
            pred_angle = outputs["pred_boxes"][..., 4]     
            teacher_angle = outputs["teacher_boxes"][..., 4].detach()
            angle_delta = (pred_angle - teacher_angle) * self.angle_factor
            periodic = (     
                torch.atan2(torch.sin(2.0 * angle_delta), torch.cos(2.0 * angle_delta)).abs()
                / self.angle_factor  
            )     
            losses["loss_ddf_angle"] = self._aggregate_ddf_loss(query_weights * periodic, mask)     
        return losses 

    def unimodal_distribution_focal_loss( 
        self,
        pred,    
        label,
        weight_right,  
        weight_left,    
        weight=None,
        reduction="sum",
        avg_factor=None,
    ): 
        dis_left = label.long()   
        dis_right = dis_left + 1

        loss = (
            F.cross_entropy(pred, dis_left, reduction="none") * weight_left.reshape(-1)   
            + F.cross_entropy(pred, dis_right, reduction="none") * weight_right.reshape(-1)
        )
    
        if weight is not None:     
            loss = loss * weight.float()
  
        if avg_factor is not None:
            loss = loss.sum() / avg_factor
        elif reduction == "mean":
            loss = loss.mean()
        elif reduction == "sum":    
            loss = loss.sum()  
        return loss 

    @torch.no_grad()   
    def _build_match_loss_labels(self, outputs, targets, indices): 
        pred_logits = outputs["pred_logits"]
        pred_boxes = outputs["pred_boxes"]
        labels_by_image = []

        for img_idx, (target, (pred_idx, tgt_idx)) in enumerate(zip(targets, indices)): 
            image_labels = []
            if pred_idx.numel() == 0:    
                labels_by_image.append(image_labels)
                continue  
 
            src_logits = pred_logits[img_idx, pred_idx]  
            src_boxes = pred_boxes[img_idx, pred_idx] 
            target_classes = target["labels"][tgt_idx].to(device=src_logits.device)   
            target_boxes = target["boxes"][tgt_idx].to(device=src_boxes.device, dtype=src_boxes.dtype) 
   
            target_onehot = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1].to(dtype=src_logits.dtype)
            target_scores = self._matched_vfl_scores(     
                src_boxes,    
                target_boxes,
                [targets[img_idx]],
                [(torch.arange(pred_idx.numel(), device=pred_idx.device), torch.arange(tgt_idx.numel(), device=tgt_idx.device))],     
            ).to(dtype=src_logits.dtype) 
            cls_loss = F.binary_cross_entropy_with_logits( 
                src_logits, 
                target_scores.unsqueeze(-1) * target_onehot,   
                reduction="none",  
            ).mean(-1)    

            box_loss = F.l1_loss(src_boxes, target_boxes, reduction="none").sum(-1) 
            src_pixel = []  
            tgt_pixel = []   
            image_size = self._target_size(targets[img_idx], src_boxes.device, src_boxes.dtype)  
            if src_boxes.numel():
                src_pixel = normalized_obb_to_pixel(src_boxes, image_size, self.angle_factor)
                tgt_pixel = normalized_obb_to_pixel(target_boxes, image_size, self.angle_factor) 

            prob_loss = 1 - probiou(src_pixel, tgt_pixel).squeeze(-1)   
     
            for cls_value, box_value, prob_value in zip(cls_loss, box_loss, prob_loss):
                image_labels.append(    
                    { 
                        "cls": float(cls_value.detach().cpu().item()),
                        "box": float(box_value.detach().cpu().item()),
                        "prob": float(prob_value.detach().cpu().item()), 
                    }
                )
            labels_by_image.append(image_labels)
        return labels_by_image

    @torch.no_grad() 
    def _visualize_loss_matching(self, outputs, targets, indices):
        global _vis_loss_match_call_idx    
        from PIL import Image, ImageDraw

        os.makedirs(VIS_LOSS_MATCH_DIR, exist_ok=True) 
        pred_boxes = outputs["pred_boxes"]
        labels_by_image = self._build_match_loss_labels(outputs, targets, indices)   

        for img_idx, (target, (pred_idx, tgt_idx)) in enumerate(zip(targets, indices)):    
            if len(target["boxes"]) == 0:   
                continue
    
            image_size = self._target_size(target, pred_boxes.device, pred_boxes.dtype)     
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

            pred_corners = xywhr_to_corners(pred_pixel) * scale_factor  
            tgt_corners = xywhr_to_corners(tgt_pixel) * scale_factor
            pred_cx = (pred_pixel[:, 0] * scale_factor).cpu()
            pred_cy = (pred_pixel[:, 1] * scale_factor).cpu() 
            tgt_cx = (tgt_pixel[:, 0] * scale_factor).cpu()
            tgt_cy = (tgt_pixel[:, 1] * scale_factor).cpu()

            canvas = Image.new("RGB", (draw_W, draw_H), color=(255, 255, 255))
            draw = ImageDraw.Draw(canvas)
   
            matched_pred_set = set(pred_idx.tolist()) 
            num_queries = pred_corners.shape[0]
   
            for pi in range(num_queries):
                if pi in matched_pred_set:
                    continue
                pts = [tuple(p) for p in pred_corners[pi].cpu().tolist()]   
                draw.line(pts + [pts[0]], fill=(180, 180, 180), width=1)  
 
            for j in range(len(tgt_corners)):
                pts = [tuple(p) for p in tgt_corners[j].cpu().tolist()] 
                draw.line(pts + [pts[0]], fill=(0, 200, 0), width=2)     
                draw.text((tgt_cx[j].item(), tgt_cy[j].item()), f"gt_{j}", fill=(0, 150, 0)) 
     
            for label, pi, ti in zip(labels_by_image[img_idx], pred_idx.tolist(), tgt_idx.tolist()):
                pts = [tuple(p) for p in pred_corners[pi].cpu().tolist()]  
                draw.line(pts + [pts[0]], fill=(220, 50, 50), width=2)
                draw.text((pred_cx[pi].item(), pred_cy[pi].item()), f"pred_{pi}", fill=(180, 30, 30))
                line_start = (pred_cx[pi].item(), pred_cy[pi].item())   
                line_end = (tgt_cx[ti].item(), tgt_cy[ti].item())
                draw.line([line_start, line_end], fill=(50, 100, 220), width=1) 
                mid_x = (line_start[0] + line_end[0]) * 0.5     
                mid_y = (line_start[1] + line_end[1]) * 0.5
                geometry_items = [(k, v) for k, v in label.items() if k not in ("cls", "box")]
                geometry_text = " ".join(f"{k}={v:.3f}" for k, v in geometry_items)  
                draw.text(
                    (mid_x, mid_y), 
                    f"cls={label['cls']:.3f} box={label['box']:.3f} {geometry_text}",
                    fill=(0, 0, 0),    
                )  

            save_path = os.path.join(VIS_LOSS_MATCH_DIR, f"call{_vis_loss_match_call_idx:04d}_img{img_idx}.png") 
            canvas.save(save_path)
  
        _vis_loss_match_call_idx += 1
  
    def get_loss(self, loss, outputs, targets, indices, num_boxes):  
        loss_map = {
            "vfl": self.loss_labels_vfl,
            "boxes": self.loss_boxes,   
            "local": self.loss_local,
        }     
        return loss_map[loss](outputs, targets, indices, num_boxes)

    def _with_local_params(self, child_outputs, parent_outputs):   
        if "local" not in self.losses or "pred_corners" not in child_outputs:   
            return child_outputs
        if "up" not in parent_outputs or "reg_scale" not in parent_outputs: 
            return child_outputs     

        child_outputs = {**child_outputs}
        child_outputs.setdefault("up", parent_outputs["up"]) 
        child_outputs.setdefault("reg_scale", parent_outputs["reg_scale"])
        return child_outputs
  
    def _num_boxes(self, targets, device):
        num_boxes = sum(len(t["labels"]) for t in targets) 
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=device)     
        if is_dist_available_and_initialized():
            torch.distributed.all_reduce(num_boxes)     
        return torch.clamp(num_boxes / get_world_size(), min=1).item()     
     
    def _compute_losses_for_outputs(self, outputs, targets, indices, num_boxes, suffix=""):
        losses = {}  
        for loss in self.losses:   
            l_dict = self.get_loss(loss, outputs, targets, indices, num_boxes)     
            l_dict = {k: v * self.weight_dict[k] for k, v in l_dict.items() if k in self.weight_dict}    
            if suffix:
                l_dict = {k + suffix: v for k, v in l_dict.items()}
            losses.update(l_dict)     
        return losses   
   
    def forward(self, outputs, targets, **kwargs):
        if not kwargs.pop('_in_group_forward', False) and ('mgsa_outputs' in outputs or 'o2m_t_outputs' in outputs):   
            group_count = 1 + len(outputs.get('mgsa_outputs', []))
            base = {k: v for k, v in outputs.items() if k not in ('mgsa_outputs', 'o2m_t_outputs', 'o2m_obb_c')}
            losses = {k: v / group_count for k, v in self.forward(base, targets, _in_group_forward=True, **kwargs).items()}
            for i, group in enumerate(outputs.get('mgsa_outputs', []), 1):
                group_losses = self.forward(group, targets, _in_group_forward=True, **kwargs)
                losses.update({f'{k}_mgsa_{i}': v / group_count for k, v in group_losses.items()}) 
            if 'o2m_t_outputs' in outputs:
                o2m = outputs['o2m_t_outputs']
                copied_targets = repeat_targets(targets, int(o2m.get('o2m_repeat', 1)))
                o2m_losses = self.forward(o2m, copied_targets, _in_group_forward=True, **kwargs)    
                losses.update({f'{k}_o2m_t': v * self.o2m_t_loss_weight for k, v in o2m_losses.items()})
            if 'o2m_obb_c' in outputs:
                losses['loss_o2m_obb_c'] = self.o2m_c_loss(outputs['o2m_obb_c'], targets) * self.weight_dict.get('loss_o2m_obb_c', 1)
            return losses
    
        self.num_pos, self.num_neg = None, None  
        epoch = kwargs.get("epoch", 0)   
        device = outputs["pred_logits"].device 
        num_boxes = self._num_boxes(targets, device)
        outputs_without_aux = { 
            k: v 
            for k, v in outputs.items()
            if k not in ("aux_outputs", "dn_outputs", "dn_pre_outputs", "dn_meta", "o2m_obb_c", "mgsa_outputs", "o2m_t_outputs")
        }
        indices = self.matcher(outputs_without_aux, targets, epoch=epoch, num_queries_list=outputs.get("num_queries_list"))["indices"]
        if VIS_LOSS_MATCH and (epoch % VIS_LOSS_MATCH_EVERY_N_EPOCHS == 0):
            self._visualize_loss_matching(outputs_without_aux, targets, indices)   
     
        losses = self._compute_losses_for_outputs(outputs, targets, indices, num_boxes)   
        for i, aux_outputs in enumerate(outputs.get("aux_outputs", [])):
            aux_outputs = self._with_local_params(aux_outputs, outputs)
            aux_indices = self.matcher(aux_outputs, targets, epoch=epoch, num_queries_list=outputs.get("num_queries_list"))["indices"]   
            losses.update(self._compute_losses_for_outputs(aux_outputs, targets, aux_indices, num_boxes, suffix=f"_aux_{i}"))  

        if "pre_outputs" in outputs:   
            pre_indices = self.matcher(outputs["pre_outputs"], targets, epoch=epoch, num_queries_list=outputs.get("num_queries_list"))["indices"] 
            losses.update(self._compute_losses_for_outputs(outputs["pre_outputs"], targets, pre_indices, num_boxes, suffix="_pre"))
   
        if "dn_outputs" in outputs:   
            indices_dn = self.get_cdn_matched_indices(outputs["dn_meta"], targets)
            dn_num_boxes = num_boxes * outputs["dn_meta"]["dn_num_group"]
            for i, aux_outputs in enumerate(outputs["dn_outputs"]):  
                aux_outputs = self._with_local_params(aux_outputs, outputs)  
                if "local" in self.losses:
                    aux_outputs = {**aux_outputs, "is_dn": True}
                losses.update(self._compute_losses_for_outputs(aux_outputs, targets, indices_dn, dn_num_boxes, suffix=f"_dn_{i}"))
            if "dn_pre_outputs" in outputs:   
                losses.update(self._compute_losses_for_outputs(outputs["dn_pre_outputs"], targets, indices_dn, dn_num_boxes, suffix="_dn_pre"))
   
        if "enc_aux_outputs" in outputs:   
            assert "enc_meta" in outputs  
            class_agnostic = outputs["enc_meta"]["class_agnostic"]  
            if class_agnostic:    
                orig_num_classes = self.num_classes   
                self.num_classes = 1
                enc_targets = [{**t, "labels": torch.zeros_like(t["labels"])} for t in targets]     
            else:
                enc_targets = targets
            for i, aux_outputs in enumerate(outputs["enc_aux_outputs"]): 
                enc_indices = self.matcher(aux_outputs, enc_targets, epoch=epoch, num_queries_list=outputs.get("num_queries_list"))["indices"]   
                losses.update(self._compute_losses_for_outputs(aux_outputs, enc_targets, enc_indices, num_boxes, suffix=f"_enc_{i}"))
            if class_agnostic:  
                self.num_classes = orig_num_classes    
  
        if "o2m_obb_c" in outputs:   
            losses["loss_o2m_obb_c"] = self.o2m_c_loss(outputs["o2m_obb_c"], targets) * self.weight_dict.get(
                "loss_o2m_obb_c", 1
            )

        return {k: torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0) for k, v in losses.items()}   

    def set_epoch(self, epoch):
        self.epoch = epoch
        if hasattr(self.matcher, "set_epoch"):  
            self.matcher.set_epoch(epoch)
        if self.no_weight_vfl_epoch != -1 and self.epoch == self.no_weight_vfl_epoch:
            logger.info(
                RED    
                + ( 
                    f"Epoch:[{self.epoch}]>=[{self.no_weight_vfl_epoch}] "   
                    f"warming up weight in vfl for {self.no_weight_vfl_warmup_iter} iters." 
                )
                + RESET
            )

    def set_model(self, model):  
        self.model = model  

    def reset_model(self):
        self.model = None
  
    @staticmethod
    def get_cdn_matched_indices(dn_meta, targets):   
        dn_positive_idx, dn_num_group = dn_meta["dn_positive_idx"], dn_meta["dn_num_group"]    
        dn_num_gts = dn_meta.get("dn_num_gts")  
        device = targets[0]["labels"].device  
        dn_match_indices = []
        for i, target in enumerate(targets):  
            num_gt = dn_num_gts[i] if dn_num_gts is not None else len(target["labels"]) 
            if num_gt > 0:
                gt_idx = torch.arange(num_gt, dtype=torch.int64, device=device).tile(dn_num_group)  
                dn_match_indices.append((dn_positive_idx[i].to(device), gt_idx)) 
            else:     
                dn_match_indices.append((torch.zeros(0, dtype=torch.int64, device=device), torch.zeros(0, dtype=torch.int64, device=device)))
        return dn_match_indices   
