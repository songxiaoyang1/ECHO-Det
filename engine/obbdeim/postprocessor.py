import torch     
import torch.nn as nn
import torch.nn.functional as F    

from ..core import register    
from .box_ops import scale_obb_from_resize_pad

    
@register()
class OBBPostProcessor(nn.Module):
    __share__ = ["num_classes", "use_focal_loss", "num_top_queries", "angle_factor", "score_threshold"]     
    
    def __init__(
        self,    
        num_classes=80,  
        use_focal_loss=True,
        num_top_queries=300,
        angle_factor=3.141592653589793,
        score_threshold=0.001    
    ): 
        super().__init__()   
        self.num_classes = int(num_classes)
        self.use_focal_loss = use_focal_loss
        self.num_top_queries = num_top_queries
        self.angle_factor = angle_factor
        self.score_threshold = score_threshold
        self.deploy_mode = False   

    def forward(self, outputs, orig_target_sizes: torch.Tensor, for_eval=False, resize_pad=None):  
        del for_eval
        logits, boxes = outputs["pred_logits"], outputs["pred_boxes"] 
        bbox_pred = scale_obb_from_resize_pad(boxes, orig_target_sizes, resize_pad, self.angle_factor)  
        topk = min(self.num_top_queries, logits.shape[1] * self.num_classes)
     
        if self.use_focal_loss: 
            scores = F.sigmoid(logits)
            scores, index = torch.topk(scores.flatten(1), topk, dim=-1) 
            labels = index % self.num_classes
            box_index = index // self.num_classes
            boxes_out = bbox_pred.gather(dim=1, index=box_index.unsqueeze(-1).repeat(1, 1, bbox_pred.shape[-1]))
        else:
            scores = F.softmax(logits, dim=-1)[..., : self.num_classes]    
            scores, labels = scores.max(dim=-1)
            boxes_out = bbox_pred    
            if scores.shape[1] > self.num_top_queries:   
                scores, box_index = torch.topk(scores, self.num_top_queries, dim=-1)     
                labels = labels.gather(dim=1, index=box_index)
                boxes_out = boxes_out.gather(dim=1, index=box_index.unsqueeze(-1).repeat(1, 1, boxes_out.shape[-1])) 

        if self.deploy_mode:
            return labels, boxes_out, scores 

        results = []
        for lab, box, sco in zip(labels, boxes_out, scores):   
            keep = sco > self.score_threshold
            results.append(dict(labels=lab[keep], boxes=box[keep], scores=sco[keep]))    
 
        return results  
   
    def deploy(self):   
        self.eval()
        self.deploy_mode = True
        return self
