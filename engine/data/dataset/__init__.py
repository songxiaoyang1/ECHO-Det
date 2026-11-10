"""  
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)  
Copyright(c) 2023 lyuwenyu. All Rights Reserved.   
"""    
   
# from ._dataset import DetDataset   
from .afss_dataset import AFSSDataset     
from .coco_dataset import CocoDetection
from .multimodal_coco_dataset import MultimodalCocoDetection    
from .multimodal_obb_dataset import MultimodalOBBDetection
from .obb_dataset import OBBDetection     
from .coco_dataset import (  
    mscoco_category2name,
    mscoco_category2label,
    mscoco_label2category,   
)
from .coco_eval import CocoEvaluator   
from .coco_utils import get_coco_api_from_dataset     
from .voc_detection import VOCDetection    
from .voc_eval import VOCEvaluator
