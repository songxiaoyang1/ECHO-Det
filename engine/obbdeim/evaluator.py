import contextlib
import io 
from types import SimpleNamespace  
from collections.abc import Mapping, Sequence

import torch

from ..core import register
from ..logger_module import get_logger
from .box_ops import scale_obb_to_image


logger = get_logger(__name__)

   
@register()
class OBBEvaluator:   
    __share__ = ["num_classes", "angle_factor"]

    def __init__( 
        self, 
        metrics=None,     
        num_classes=80,    
        angle_factor=3.141592653589793,
        category2name=None,
        categories=None,
        **kwargs,   
    ):    
        del kwargs
        self.metrics = metrics or ["yolo_obb"]
        self.num_classes = num_classes
        self.angle_factor = angle_factor
        self.iou_types = ("obb",)
        self.category2name = self._normalize_category_names(category2name, categories)
        self.cleanup()
 
    @staticmethod
    def _normalize_category_names(category2name=None, categories=None):
        if category2name is not None:  
            return {int(category_id): str(name) for category_id, name in category2name.items()} 
   
        if categories is None: 
            return {}

        if isinstance(categories, Mapping):
            return {int(category_id): str(name) for category_id, name in categories.items()} 

        if isinstance(categories, Sequence) and not isinstance(categories, (str, bytes)):
            normalized = {}    
            for index, category in enumerate(categories):     
                if isinstance(category, Mapping):
                    category_id = int(category.get("id", index))
                    name = str(category.get("name", category_id))
                else:
                    category_id = index     
                    name = str(category)
                normalized[category_id] = name
            return normalized

        raise ValueError(f"categories must be a mapping or sequence, got {type(categories)!r}") 
  
    def _coco_categories(self):
        return [
            {"id": label + 1, "name": self.category2name.get(label, str(label))}
            for label in range(self.num_classes)  
        ]  
    
    def cleanup(self):  
        self.predictions = {}
        self.targets = {} 
        self.stats = {}
        self.coco_eval = {}
        self.cocoGt = None
        self.coco_pred_json = []
        return self

    def update(self, predictions, targets=None):
        for image_id, pred in predictions.items():   
            self.predictions[int(image_id)] = {
                k: (v.detach().cpu() if torch.is_tensor(v) else v)
                for k, v in pred.items()
            }
        if targets is not None:    
            for target in targets:  
                image_id = int(target["image_id"].flatten()[0].item())   
                self.targets[image_id] = {   
                    k: (v.detach().cpu() if torch.is_tensor(v) else v)  
                    for k, v in target.items()
                }

    def synchronize_between_processes(self): 
        return None
    
    def _target_boxes_to_image(self, target):
        boxes = target.get("boxes", torch.zeros(0, 5)).float() 
        if boxes.numel() == 0:
            return boxes.reshape(0, 5)  

        image_size = target.get("orig_size", target.get("size"))     
        if image_size is None:
            return boxes     
    
        # Dataloader targets are normalized; some unit tests pass pixel xywhr.    
        if boxes[..., :4].max() <= 2.0: 
            return scale_obb_to_image(boxes.unsqueeze(0), image_size.reshape(1, 2).float(), self.angle_factor)[0]   
        return boxes
    
    def _build_yolo_arrays(self): 
        preds = []
        gts = [] 
        for image_id in sorted(set(self.predictions) | set(self.targets)):   
            pred = self.predictions.get(image_id, {}) 
            target = self.targets.get(image_id, {})   
            pred_boxes = pred.get("boxes", torch.zeros(0, 5)).float()     
            pred_scores = pred.get("scores", torch.zeros(0)).float() 
            pred_labels = pred.get("labels", torch.zeros(0, dtype=torch.long)).float()
            gt_boxes = self._target_boxes_to_image(target)
            gt_labels = target.get("labels", torch.zeros(0, dtype=torch.long)).float()
            if pred_boxes.numel(): 
                image_col = torch.full((pred_boxes.shape[0], 1), image_id, dtype=pred_boxes.dtype) 
                preds.append(torch.cat([image_col, pred_boxes, pred_scores[:, None], pred_labels[:, None]], dim=1))
            if gt_boxes.numel():  
                image_col = torch.full((gt_boxes.shape[0], 1), image_id, dtype=gt_boxes.dtype)
                gts.append(torch.cat([image_col, gt_labels[:, None], gt_boxes], dim=1))  
        pred_tensor = torch.cat(preds, dim=0) if preds else torch.zeros(0, 8)    
        gt_tensor = torch.cat(gts, dim=0) if gts else torch.zeros(0, 7)
        return pred_tensor, gt_tensor     

    def _image_size(self, image_id, target, pred):
        image_size = target.get("orig_size", target.get("size"))
        if image_size is not None:  
            width, height = image_size.flatten()[:2].tolist()
            return float(width), float(height)    

        pred_boxes = pred.get("boxes", torch.zeros(0, 5)).float()   
        if pred_boxes.numel():
            x_extent = pred_boxes[:, 0].max() + pred_boxes[:, 2].max()
            y_extent = pred_boxes[:, 1].max() + pred_boxes[:, 3].max()  
            return float(max(x_extent.item(), 1.0)), float(max(y_extent.item(), 1.0))
        return 1.0, 1.0
     
    def _build_coco_inputs(self): 
        from pycocoeval.coco import COCO 
    
        with contextlib.redirect_stdout(io.StringIO()):     
            image_ids = sorted(set(self.predictions) | set(self.targets))     
            images = [] 
            annotations = [] 
            detections = []   
            ann_id = 1  

            for image_id in image_ids:   
                pred = self.predictions.get(image_id, {})
                target = self.targets.get(image_id, {})    
                width, height = self._image_size(image_id, target, pred)
                images.append({"id": int(image_id), "width": width, "height": height, "file_name": f"{image_id}.jpg"})

                gt_boxes = self._target_boxes_to_image(target)  
                gt_labels = target.get("labels", torch.zeros(0, dtype=torch.long)).long()
                for box, label in zip(gt_boxes.tolist(), gt_labels.tolist()):
                    area = max(float(box[2]) * float(box[3]), 0.0)
                    annotations.append( 
                        {
                            "id": ann_id,  
                            "image_id": int(image_id),   
                            "category_id": int(label) + 1, 
                            "bbox": [float(v) for v in box],     
                            "rbox": [float(v) for v in box],
                            "area": area,
                            "iscrowd": 0, 
                        }
                    )
                    ann_id += 1 

                pred_boxes = pred.get("boxes", torch.zeros(0, 5)).float()
                pred_scores = pred.get("scores", torch.zeros(0)).float()   
                pred_labels = pred.get("labels", torch.zeros(0, dtype=torch.long)).long()
                for box, score, label in zip(pred_boxes.tolist(), pred_scores.tolist(), pred_labels.tolist()):
                    detections.append(
                        {
                            "image_id": int(image_id),
                            "category_id": int(label) + 1, 
                            "bbox": [float(v) for v in box],   
                            "rbox": [float(v) for v in box],     
                            "score": float(score),     
                        } 
                    ) 

            categories = self._coco_categories()
            coco_gt = COCO()
            coco_gt.dataset = {
                "images": images,
                "annotations": annotations,   
                "categories": categories,
            }
            coco_gt.createIndex()

            if detections:     
                coco_dt = coco_gt.loadRes(detections)    
            else:    
                coco_dt = COCO()  
                coco_dt.dataset = {
                    "images": list(images), 
                    "annotations": [],     
                    "categories": categories, 
                } 
                coco_dt.createIndex()

            return coco_gt, coco_dt, detections, image_ids

    def accumulate(self):
        coco_gt = coco_dt = None  
        coco_pred_json = []
        image_ids = []     
        if "coco_obb" in self.metrics or "yolo_obb" in self.metrics:  
            coco_gt, coco_dt, coco_pred_json, image_ids = self._build_coco_inputs()    
            self.cocoGt = coco_gt
            self.coco_pred_json = coco_pred_json
            self.coco_eval = {"bbox": SimpleNamespace(cocoGt=coco_gt)}
     
        if "yolo_obb" in self.metrics: 
            from pycocoeval.yoloeval import get_yolo_obb_metrice
            self.stats["yolo_eval_obb"] = get_yolo_obb_metrice(logger, self, coco_pred_json)  
 
        # if "coco_obb" in self.metrics:
        #     try:
        #         from pycocoeval.rotated_cocoeval import RotatedCOCOeval    
        #         with contextlib.redirect_stdout(io.StringIO()):
        #             coco_eval = RotatedCOCOeval(coco_gt, coco_dt, iouType="bbox")
        #             coco_eval.params.imgIds = image_ids     
        #             coco_eval.params.catIds = [i + 1 for i in range(self.num_classes)]
        #             self.coco_eval["bbox"] = coco_eval     
        #             coco_eval.evaluate() 
        #             coco_eval.accumulate() 
        #             coco_eval.summarize()
        #         self.stats["coco_eval_obb"] = coco_eval.stats.tolist()
        #     except Exception as exc:
        #         self.stats["coco_eval_obb_error"] = str(exc)    
        #         self.stats["coco_eval_obb"] = [0.0] * 12     

    def summarize(self):
        return self.stats  
