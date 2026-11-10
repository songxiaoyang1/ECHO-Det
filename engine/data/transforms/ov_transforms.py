import random
from typing import Any, Dict, List, Tuple
    
import torch
import torch.nn as nn   

from ...core import register  
   

def _build_label_to_name(dataset: Any, labels: List[int]) -> Dict[int, str]:     
    category2name = getattr(dataset, "category2name", None)  
    if category2name is None:    
        category2name = {    
            int(category_id): cat["name"] 
            for category_id, cat in getattr(dataset.coco, "cats", {}).items()   
        }   

    label2category = getattr(dataset, "label2category", None)
    label_index_to_name = {} 
    if label2category is not None: 
        label_index_to_name = {    
            int(label): category2name[int(category_id)]     
            for label, category_id in label2category.items()     
            if int(category_id) in category2name 
        }
    elif hasattr(dataset, "category2label"):
        label_index_to_name = {
            int(label): category2name[int(category_id)]   
            for category_id, label in dataset.category2label.items()     
            if int(category_id) in category2name
        }   
     
    raw_category_ids = set(category2name.keys())
    uses_label_indices = bool(getattr(dataset, "remap_mscoco_category", False)) or any(     
        int(label) not in raw_category_ids for label in labels    
    )
   
    if uses_label_indices and label_index_to_name:
        return label_index_to_name
    return {int(category_id): name for category_id, name in category2name.items()}  


@register() 
class RandomLoadTexts(nn.Module):
    def __init__(self, num_classes=80, blank_text: bool = False):    
        super().__init__()   
        self.num_classes = num_classes     
        self.blank_text = blank_text
    
    def forward(
        self, sample: Tuple[Any, Dict[str, Any], Any]
    ) -> Tuple[Any, Dict[str, Any], Any]:
        img, target, dataset = sample    
    
        labels = target["labels"]  
        label_list = labels.tolist() if isinstance(labels, torch.Tensor) else list(labels) 
        label_to_name = _build_label_to_name(dataset, label_list)

        positive_texts: List[str] = []     
        for label in label_list:    
            text = label_to_name.get(int(label))
            if text is not None and text not in positive_texts:  
                positive_texts.append(text)

        class_texts_source = ( 
            getattr(dataset, "detect_text_list", None) 
            if not self.blank_text  
            else getattr(dataset, "text_list", None)
        )   
        class_texts = list(class_texts_source) if class_texts_source is not None else []
    
        remaining = [text for text in class_texts if text not in positive_texts]  
        selected_texts = list(positive_texts)     
        num_remaining_slots = max(self.num_classes - len(selected_texts), 0)
 
        if not self.blank_text and num_remaining_slots > 0:  
            random.shuffle(remaining) 
            selected_texts.extend(remaining[:num_remaining_slots])
            num_remaining_slots = self.num_classes - len(selected_texts)

        if num_remaining_slots > 0:    
            selected_texts.extend([" "] * num_remaining_slots)     
    
        text2id = {text: index for index, text in enumerate(selected_texts)}
        unmapped_labels = [
            int(label)   
            for label in label_list     
            if int(label) not in label_to_name or label_to_name[int(label)] not in text2id 
        ]
        if unmapped_labels:
            raise ValueError(
                "RandomLoadTexts cannot map labels "  
                f"{sorted(set(unmapped_labels))} to loaded texts. "
                f"Available label ids: {sorted(label_to_name.keys())}; "
                f"selected texts: {selected_texts}"
            )  
     
        remapped_labels = [text2id[label_to_name[int(label)]] for label in label_list]     

        label_device = labels.device if isinstance(labels, torch.Tensor) else None   
        target["labels"] = torch.as_tensor(remapped_labels, dtype=torch.int64, device=label_device) 
        target["texts"] = selected_texts
        return img, target, dataset


@register()  
class LookupTextFeats(nn.Module):
    def forward(
        self, sample: Tuple[Any, Dict[str, Any], Any]    
    ) -> Tuple[Any, Dict[str, Any], Any]:
        img, target, dataset = sample   
        texts: List[str] = target["texts"]
        cache = dataset.text_cache 
  
        if not cache:   
            raise RuntimeError("LookupTextFeats requires dataset.text_cache to contain at least one entry.")

        blank_feat = torch.zeros_like(next(iter(cache.values())))  
        target["text_feats"] = torch.stack([cache.get(text, blank_feat) for text in texts], dim=0)    
        return img, target, dataset
