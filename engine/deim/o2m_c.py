import math     

import torch
import torch.distributed as dist     
import torch.nn as nn
import torch.nn.functional as F     
     

def _box_iou(boxes1, boxes2): 
    lt = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2]) 
    rb = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    inter = (rb - lt).clamp_min(0).prod(-1)    
    area1 = (boxes1[:, 2:] - boxes1[:, :2]).clamp_min(0).prod(-1)     
    area2 = (boxes2[:, 2:] - boxes2[:, :2]).clamp_min(0).prod(-1)
    return inter / (area1[:, None] + area2[None, :] - inter).clamp_min(1e-9)    
 

class O2MCHead(nn.Module):
    def __init__(self, feat_channels, feat_strides, num_classes, reg_max=16):   
        super().__init__()
        if len(feat_channels) != len(feat_strides): 
            raise ValueError("feat_channels and feat_strides must have the same length")
        self.feat_strides = tuple(feat_strides)
        self.num_classes = num_classes
        self.reg_max = reg_max 
        self.stems = nn.ModuleList(
            nn.Sequential(nn.Conv2d(c, c, 3, padding=1, bias=False), nn.BatchNorm2d(c), nn.SiLU())
            for c in feat_channels
        )    
        self.cls_heads = nn.ModuleList(nn.Conv2d(c, num_classes, 1) for c in feat_channels)     
        self.reg_heads = nn.ModuleList(nn.Conv2d(c, 4 * (reg_max + 1), 1) for c in feat_channels)     
        for head in self.cls_heads:
            nn.init.constant_(head.bias, -4.595)   
     
    def forward(self, features):     
        if len(features) != len(self.stems):
            raise ValueError(f"expected {len(self.stems)} feature maps, got {len(features)}")
 
        logits, distri, shapes = [], [], []   
        for feature, stem, cls_head, reg_head in zip(features, self.stems, self.cls_heads, self.reg_heads):
            feature = stem(feature)
            shapes.append(tuple(feature.shape[-2:])) 
            logits.append(cls_head(feature).flatten(2).transpose(1, 2))
            distri.append(reg_head(feature).flatten(2).transpose(1, 2))
   
        return { 
            "pred_logits": torch.cat(logits, dim=1), 
            "pred_distri": torch.cat(distri, dim=1),    
            "feature_shapes": shapes,
            "feat_strides": self.feat_strides,
            "reg_max": self.reg_max,
        } 

 
class O2MCOBBHead(O2MCHead):
    def __init__(self, feat_channels, feat_strides, num_classes, reg_max=16, angle_factor=math.pi):  
        super().__init__(feat_channels, feat_strides, num_classes, reg_max)
        self.angle_factor = angle_factor  
        self.angle_heads = nn.ModuleList(nn.Conv2d(c, 1, 1) for c in feat_channels)

    def forward(self, features):    
        if len(features) != len(self.stems):  
            raise ValueError(f"expected {len(self.stems)} feature maps, got {len(features)}")
 
        logits, distri, angles, shapes = [], [], [], []
        for feature, stem, cls_head, reg_head, angle_head in zip(
            features, self.stems, self.cls_heads, self.reg_heads, self.angle_heads
        ):   
            feature = stem(feature)
            shapes.append(tuple(feature.shape[-2:]))
            logits.append(cls_head(feature).flatten(2).transpose(1, 2)) 
            distri.append(reg_head(feature).flatten(2).transpose(1, 2))
            angle = (angle_head(feature).sigmoid() - 0.25) * self.angle_factor
            angles.append(angle.flatten(2).transpose(1, 2))
        return {   
            "pred_logits": torch.cat(logits, dim=1),
            "pred_distri": torch.cat(distri, dim=1),   
            "pred_angle": torch.cat(angles, dim=1),
            "feature_shapes": shapes,
            "feat_strides": self.feat_strides,   
            "reg_max": self.reg_max,
            "angle_factor": self.angle_factor,
        }     
 

class TaskAlignedAssigner(nn.Module):
    def __init__(self, topk=10, alpha=0.5, beta=6.0):
        super().__init__() 
        self.topk = topk 
        self.alpha = alpha
        self.beta = beta 

    @torch.no_grad()    
    def forward(self, pred_scores, pred_boxes, anchor_points, gt_labels, gt_boxes, mask_gt=None):    
        batch_size, num_anchors, num_classes = pred_scores.shape    
        gt_labels = gt_labels.squeeze(-1).long() if gt_labels.ndim == 3 else gt_labels.long()  
        if mask_gt is None:
            mask_gt = torch.ones_like(gt_labels, dtype=torch.bool)
        else:
            mask_gt = mask_gt.squeeze(-1).bool() if mask_gt.ndim == 3 else mask_gt.bool() 
   
        target_labels = torch.full(
            (batch_size, num_anchors), num_classes, dtype=torch.long, device=pred_scores.device
        )  
        target_boxes = torch.zeros_like(pred_boxes)  
        target_scores = pred_scores.new_zeros((batch_size, num_anchors, num_classes))
        foreground = torch.zeros((batch_size, num_anchors), dtype=torch.bool, device=pred_scores.device)   
        target_gt_idx = torch.zeros((batch_size, num_anchors), dtype=torch.long, device=pred_scores.device)
 
        for batch_idx in range(batch_size):  
            valid_idx = mask_gt[batch_idx].nonzero(as_tuple=False).flatten()
            if valid_idx.numel() == 0:  
                continue    
    
            labels = gt_labels[batch_idx, valid_idx]
            boxes = gt_boxes[batch_idx, valid_idx]  
            deltas = torch.cat(
                (anchor_points[None] - boxes[:, None, :2], boxes[:, None, 2:] - anchor_points[None]), dim=-1
            )  
            inside = deltas.amin(-1) > 0
            ious = _box_iou(boxes, pred_boxes[batch_idx]).clamp_min(0)
            class_scores = pred_scores[batch_idx][:, labels].transpose(0, 1)  
            metric = class_scores.pow(self.alpha) * ious.pow(self.beta) * inside
     
            topk = min(self.topk, num_anchors)     
            values, indices = metric.topk(topk, dim=-1)   
            positive = torch.zeros_like(metric, dtype=torch.bool)    
            positive.scatter_(1, indices, values > 0)
   
            conflicts = positive.sum(0) > 1
            if conflicts.any():
                conflict_idx = conflicts.nonzero(as_tuple=False).flatten()
                best_gt = ious[:, conflict_idx].argmax(0) 
                positive[:, conflict_idx] = False    
                positive[best_gt, conflict_idx] = True

            gt_for_anchor = positive.float().argmax(0)
            fg = positive.any(0)
            if not fg.any(): 
                continue
    
            anchor_idx = fg.nonzero(as_tuple=False).flatten()
            assigned_gt = gt_for_anchor[fg]   
            assigned_labels = labels[assigned_gt]
            quality = ious[assigned_gt, anchor_idx]
            foreground[batch_idx, fg] = True
            target_gt_idx[batch_idx, fg] = valid_idx[assigned_gt]
            target_labels[batch_idx, fg] = assigned_labels   
            target_boxes[batch_idx, fg] = boxes[assigned_gt]  
            target_scores[batch_idx, anchor_idx, assigned_labels] = quality
  
        return target_labels, target_boxes, target_scores, foreground, target_gt_idx 

    
class RotatedTaskAlignedAssigner(TaskAlignedAssigner):
    def __init__(self, topk=10, alpha=0.5, beta=6.0, feat_strides=(8, 16, 32)):
        super().__init__(topk, alpha, beta)
        self.min_stride = feat_strides[0]  
        self.stride_val = feat_strides[1] if len(feat_strides) > 1 else feat_strides[0]

    @staticmethod
    def _anchors_in_boxes(anchor_points, boxes):     
        from ..obbdeim.box_ops import xywhr_to_corners   
     
        corners = xywhr_to_corners(boxes)
        a, b, _, d = corners.unbind(-2)     
        ab, ad = b - a, d - a 
        ap = anchor_points[None] - a[:, None]    
        ap_ab = (ap * ab[:, None]).sum(-1)    
        ap_ad = (ap * ad[:, None]).sum(-1)
        return (     
            (ap_ab >= 0) 
            & (ap_ab <= ab.square().sum(-1, keepdim=True))   
            & (ap_ad >= 0)
            & (ap_ad <= ad.square().sum(-1, keepdim=True)) 
        )  
   
    @torch.no_grad() 
    def forward(self, pred_scores, pred_boxes, anchor_points, gt_labels, gt_boxes, mask_gt=None):    
        from ..obbdeim.box_ops import batch_probiou

        batch_size, num_anchors, num_classes = pred_scores.shape
        gt_labels = gt_labels.squeeze(-1).long() if gt_labels.ndim == 3 else gt_labels.long()   
        if mask_gt is None:
            mask_gt = torch.ones_like(gt_labels, dtype=torch.bool)
        else:
            mask_gt = mask_gt.squeeze(-1).bool() if mask_gt.ndim == 3 else mask_gt.bool()     
  
        target_labels = torch.full(
            (batch_size, num_anchors), num_classes, dtype=torch.long, device=pred_scores.device 
        )
        target_boxes = torch.zeros_like(pred_boxes) 
        target_scores = pred_scores.new_zeros((batch_size, num_anchors, num_classes))
        foreground = torch.zeros((batch_size, num_anchors), dtype=torch.bool, device=pred_scores.device)   
        target_gt_idx = torch.zeros((batch_size, num_anchors), dtype=torch.long, device=pred_scores.device) 
     
        for batch_idx in range(batch_size): 
            valid_idx = mask_gt[batch_idx].nonzero(as_tuple=False).flatten()
            if valid_idx.numel() == 0:
                continue   
 
            labels = gt_labels[batch_idx, valid_idx]   
            boxes = gt_boxes[batch_idx, valid_idx]
            candidate_boxes = boxes.clone()    
            candidate_boxes[:, 2:4] = torch.where( 
                candidate_boxes[:, 2:4] < self.min_stride,  
                candidate_boxes.new_tensor(self.stride_val),
                candidate_boxes[:, 2:4],   
            )
            inside = self._anchors_in_boxes(anchor_points, candidate_boxes)    
            overlaps = batch_probiou(boxes, pred_boxes[batch_idx]).clamp_min(0)
            class_scores = pred_scores[batch_idx][:, labels].transpose(0, 1)  
            metric = class_scores.pow(self.alpha) * overlaps.pow(self.beta) * inside
     
            topk = min(self.topk, num_anchors)  
            values, indices = metric.topk(topk, dim=-1) 
            positive = torch.zeros_like(metric, dtype=torch.bool) 
            positive.scatter_(1, indices, values > 0)

            conflicts = positive.sum(0) > 1
            if conflicts.any():
                conflict_idx = conflicts.nonzero(as_tuple=False).flatten()
                best_gt = overlaps[:, conflict_idx].argmax(0)     
                positive[:, conflict_idx] = False  
                positive[best_gt, conflict_idx] = True  
 
            fg = positive.any(0)
            if not fg.any():    
                continue   
   
            assigned_gt = positive.float().argmax(0)[fg]
            anchor_idx = fg.nonzero(as_tuple=False).flatten()     
            assigned_labels = labels[assigned_gt]
            masked_metric = metric * positive  
            max_metric = masked_metric.amax(-1, keepdim=True)
            max_overlap = (overlaps * positive).amax(-1, keepdim=True)  
            quality = (masked_metric * max_overlap / (max_metric + 1e-9)).amax(0)[fg]
   
            foreground[batch_idx, fg] = True  
            target_gt_idx[batch_idx, fg] = valid_idx[assigned_gt]
            target_labels[batch_idx, fg] = assigned_labels   
            target_boxes[batch_idx, fg] = boxes[assigned_gt]     
            target_scores[batch_idx, anchor_idx, assigned_labels] = quality     

        return target_labels, target_boxes, target_scores, foreground, target_gt_idx     
     
     
class O2MCLoss(nn.Module):  
    def __init__(self, num_classes, topk=10, alpha=0.5, beta=6.0, vfl_alpha=0.75, vfl_gamma=2.0):     
        super().__init__()    
        self.num_classes = num_classes
        self.vfl_alpha = vfl_alpha   
        self.vfl_gamma = vfl_gamma 
        self.assigner = TaskAlignedAssigner(topk=topk, alpha=alpha, beta=beta)
  
    @staticmethod    
    def _anchors(feature_shapes, feat_strides, device, dtype):    
        points, strides = [], []   
        for (height, width), stride in zip(feature_shapes, feat_strides):     
            y, x = torch.meshgrid(
                torch.arange(height, device=device, dtype=dtype),
                torch.arange(width, device=device, dtype=dtype),
                indexing="ij",
            )    
            points.append(torch.stack((x, y), dim=-1).reshape(-1, 2).add_(0.5).mul_(stride))   
            strides.append(torch.full((height * width, 1), stride, device=device, dtype=dtype))    
        return torch.cat(points), torch.cat(strides)  
  
    @staticmethod     
    def _decode(pred_distri, anchor_points, anchor_strides, reg_max):  
        bins = reg_max + 1
        distribution = pred_distri.reshape(*pred_distri.shape[:2], 4, bins).softmax(-1) 
        distances = distribution.matmul(torch.arange(bins, device=pred_distri.device, dtype=pred_distri.dtype)) 
        distances = distances * anchor_strides[None]
        points = anchor_points[None] 
        return torch.cat((points - distances[..., :2], points + distances[..., 2:]), dim=-1)

    @staticmethod  
    def _normalizer(target_scores):
        value = target_scores.sum().detach()
        if dist.is_available() and dist.is_initialized():  
            dist.all_reduce(value)
            value /= dist.get_world_size()  
        return value.clamp_min(1.0)

    def forward(self, outputs, targets):
        pred_logits = outputs["pred_logits"]     
        pred_distri = outputs["pred_distri"]    
        feature_shapes = outputs["feature_shapes"]  
        feat_strides = outputs["feat_strides"]   
        reg_max = outputs["reg_max"]
        device, dtype = pred_logits.device, pred_logits.dtype
        batch_size = pred_logits.shape[0]
    
        anchor_points, anchor_strides = self._anchors(feature_shapes, feat_strides, device, dtype)
        pred_boxes = self._decode(pred_distri, anchor_points, anchor_strides, reg_max) 
        image_height = max(height * stride for (height, _), stride in zip(feature_shapes, feat_strides))  
        image_width = max(width * stride for (_, width), stride in zip(feature_shapes, feat_strides))   
   
        max_gt = max((len(target["labels"]) for target in targets), default=0) 
        gt_labels = torch.zeros((batch_size, max_gt), dtype=torch.long, device=device)    
        gt_boxes = torch.zeros((batch_size, max_gt, 4), dtype=dtype, device=device) 
        mask_gt = torch.zeros((batch_size, max_gt), dtype=torch.bool, device=device)     
        scale = pred_logits.new_tensor((image_width, image_height, image_width, image_height))    
        for batch_idx, target in enumerate(targets):     
            count = len(target["labels"])
            if count == 0:
                continue 
            labels = target["labels"].to(device=device, dtype=torch.long)     
            boxes = target["boxes"].to(device=device, dtype=dtype)   
            xyxy = torch.cat((boxes[:, :2] - boxes[:, 2:] / 2, boxes[:, :2] + boxes[:, 2:] / 2), dim=-1)
            gt_labels[batch_idx, :count] = labels  
            gt_boxes[batch_idx, :count] = xyxy * scale
            mask_gt[batch_idx, :count] = True    
   
        _, target_boxes, target_scores, foreground, _ = self.assigner(
            pred_logits.detach().sigmoid(), pred_boxes.detach(), anchor_points, gt_labels, gt_boxes, mask_gt    
        )     
        normalizer = self._normalizer(target_scores)    

        pred_scores = pred_logits.sigmoid() 
        vfl_weight = self.vfl_alpha * pred_scores.pow(self.vfl_gamma) * (1 - target_scores) + target_scores
        loss_vfl = (F.binary_cross_entropy_with_logits(pred_logits, target_scores, reduction="none") * vfl_weight).sum()
        loss_vfl = loss_vfl / normalizer

        if not foreground.any():
            return loss_vfl
  
        points = anchor_points[None].expand(batch_size, -1, -1)
        strides = anchor_strides[None].expand(batch_size, -1, -1)
        distances = torch.cat( 
            ((points - target_boxes[..., :2]) / strides, (target_boxes[..., 2:] - points) / strides), dim=-1
        ).clamp(0, reg_max - 1e-2)  
        pred_pos = pred_distri.reshape(batch_size, -1, 4, reg_max + 1)[foreground]   
        target_pos = distances[foreground]  
        left = target_pos.long()
        right = left + 1
        loss_left = F.cross_entropy(pred_pos.reshape(-1, reg_max + 1), left.reshape(-1), reduction="none")     
        loss_right = F.cross_entropy(pred_pos.reshape(-1, reg_max + 1), right.reshape(-1), reduction="none")   
        loss_dfl = loss_left * (right - target_pos).reshape(-1) + loss_right * (target_pos - left).reshape(-1)  
        quality = target_scores.sum(-1)[foreground].repeat_interleave(4)
        loss_dfl = (loss_dfl * quality).sum() / normalizer  
        return loss_vfl + loss_dfl 
 

def _dist2rbox(distance, angle, anchor_points):   
    left_top, right_bottom = distance.split(2, dim=-1)
    offset_x, offset_y = ((right_bottom - left_top) / 2).split(1, dim=-1)
    cos, sin = angle.cos(), angle.sin() 
    center = torch.cat((offset_x * cos - offset_y * sin, offset_x * sin + offset_y * cos), dim=-1)    
    return torch.cat((center + anchor_points, left_top + right_bottom), dim=-1)
     

def _rbox2dist(boxes, anchor_points, angle, reg_max):
    center, size = boxes.split(2, dim=-1)     
    offset_x, offset_y = (center - anchor_points).split(1, dim=-1)
    cos, sin = angle.cos(), angle.sin()    
    x = offset_x * cos + offset_y * sin     
    y = -offset_x * sin + offset_y * cos
    width, height = size.split(1, dim=-1)
    return torch.cat((width / 2 - x, height / 2 - y, width / 2 + x, height / 2 + y), dim=-1).clamp(     
        0, reg_max - 1e-2 
    )

  
class O2MCOBBLoss(O2MCLoss):   
    def __init__(
        self,
        num_classes,
        angle_factor=math.pi,
        topk=10,
        alpha=0.5,   
        beta=6.0,
        vfl_alpha=0.75,
        vfl_gamma=2.0, 
    ): 
        super().__init__(num_classes, topk, alpha, beta, vfl_alpha, vfl_gamma)     
        self.angle_factor = angle_factor
        self.assigner = RotatedTaskAlignedAssigner(topk, alpha, beta)

    def forward(self, outputs, targets):
        pred_logits = outputs["pred_logits"]
        pred_distri = outputs["pred_distri"]
        pred_angle = outputs["pred_angle"]    
        feature_shapes = outputs["feature_shapes"]   
        feat_strides = outputs["feat_strides"]
        reg_max = outputs["reg_max"]
        device, dtype = pred_logits.device, pred_logits.dtype
        batch_size = pred_logits.shape[0]
  
        anchor_points, anchor_strides = self._anchors(feature_shapes, feat_strides, device, dtype)
        grid_points = anchor_points / anchor_strides 
        bins = reg_max + 1
        distribution = pred_distri.reshape(batch_size, -1, 4, bins).softmax(-1)
        distances = distribution.matmul(torch.arange(bins, device=device, dtype=dtype))
        pred_boxes = _dist2rbox(distances, pred_angle, grid_points[None])
        pred_boxes[..., :4] *= anchor_strides[None]
        pred_boxes = torch.cat((pred_boxes, pred_angle), dim=-1) 
     
        image_height = max(height * stride for (height, _), stride in zip(feature_shapes, feat_strides))     
        image_width = max(width * stride for (_, width), stride in zip(feature_shapes, feat_strides))
        max_gt = max((len(target["labels"]) for target in targets), default=0)     
        gt_labels = torch.zeros((batch_size, max_gt), dtype=torch.long, device=device)   
        gt_boxes = torch.zeros((batch_size, max_gt, 5), dtype=dtype, device=device) 
        mask_gt = torch.zeros((batch_size, max_gt), dtype=torch.bool, device=device)
        scale = pred_logits.new_tensor((image_width, image_height, image_width, image_height))
        for batch_idx, target in enumerate(targets):
            count = len(target["labels"])     
            if count == 0:     
                continue
            boxes = target["boxes"].to(device=device, dtype=dtype) 
            gt_labels[batch_idx, :count] = target["labels"].to(device=device, dtype=torch.long)    
            gt_boxes[batch_idx, :count, :4] = boxes[:, :4] * scale
            gt_boxes[batch_idx, :count, 4] = boxes[:, 4] * self.angle_factor
            mask_gt[batch_idx, :count] = True

        self.assigner.min_stride = feat_strides[0]
        self.assigner.stride_val = feat_strides[1] if len(feat_strides) > 1 else feat_strides[0]
        _, target_boxes, target_scores, foreground, _ = self.assigner(    
            pred_logits.detach().sigmoid(), pred_boxes.detach(), anchor_points, gt_labels, gt_boxes, mask_gt
        )     
        normalizer = self._normalizer(target_scores)
 
        pred_scores = pred_logits.sigmoid()     
        vfl_weight = self.vfl_alpha * pred_scores.pow(self.vfl_gamma) * (1 - target_scores) + target_scores
        loss_vfl = (   
            F.binary_cross_entropy_with_logits(pred_logits, target_scores, reduction="none") * vfl_weight
        ).sum() / normalizer
        if not foreground.any():    
            return loss_vfl + (pred_distri.sum() + pred_angle.sum()) * 0
 
        from ..obbdeim.box_ops import probiou   
    
        quality = target_scores.sum(-1)[foreground]
        loss_probiou = ((1 - probiou(pred_boxes[foreground], target_boxes[foreground]).squeeze(-1)) * quality).sum()
        loss_probiou = loss_probiou / normalizer     

        strides = anchor_strides[None].expand(batch_size, -1, -1) 
        target_grid = target_boxes[..., :4] / strides
        target_distri = _rbox2dist(
            target_grid,
            grid_points[None].expand(batch_size, -1, -1),  
            target_boxes[..., 4:5],
            reg_max,
        )
        pred_pos = pred_distri.reshape(batch_size, -1, 4, bins)[foreground]
        target_pos = target_distri[foreground]
        left = target_pos.long()
        right = left + 1   
        loss_left = F.cross_entropy(pred_pos.reshape(-1, bins), left.reshape(-1), reduction="none")
        loss_right = F.cross_entropy(pred_pos.reshape(-1, bins), right.reshape(-1), reduction="none")
        loss_dfl = loss_left * (right - target_pos).reshape(-1) + loss_right * (target_pos - left).reshape(-1)
        loss_dfl = (loss_dfl * quality.repeat_interleave(4)).sum() / normalizer   
    
        target_pos_boxes = target_boxes[foreground]  
        log_ar = torch.log((target_pos_boxes[:, 2] + 1e-9) / (target_pos_boxes[:, 3] + 1e-9)) 
        angle_weight = torch.exp(-(log_ar.square()) / 9)
        angle_delta = pred_boxes[foreground][:, 4] - target_pos_boxes[:, 4]     
        angle_delta = angle_delta - torch.round(angle_delta / math.pi) * math.pi
        loss_angle = (angle_weight * torch.sin(2 * angle_delta).square() * quality).sum() / normalizer
        return loss_vfl + loss_probiou + loss_dfl + loss_angle     
