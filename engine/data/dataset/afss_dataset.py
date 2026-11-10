import torch.utils.data as data

from ...core import register


@register()
class AFSSDataset(data.Dataset): 
    __inject__ = ["dataset"]   

    def __init__(self, dataset, **kwargs):
        self.dataset = dataset   
        self._all_indices = list(range(len(dataset)))    
        self._active_indices = self._all_indices.copy()
        self._epoch = -1
    
    def __len__(self):     
        return len(self._active_indices)

    def __getitem__(self, idx):   
        base_idx = self._active_indices[idx]
        return self.dataset[base_idx]

    def __getattr__(self, name):
        attributes = object.__getattribute__(self, "__dict__")  
        if "dataset" not in attributes:
            raise AttributeError(f"{type(self).__name__!s} object has no attribute {name!r}")   
        return getattr(attributes["dataset"], name)
 
    def set_epoch(self, epoch):   
        self._epoch = epoch
        if hasattr(self.dataset, "set_epoch"):
            self.dataset.set_epoch(epoch)  

    def set_active_indices(self, indices):
        normalized = [int(i) for i in indices]
        if not normalized:    
            raise ValueError("AFSSDataset requires a non-empty active index list.")
        self._active_indices = normalized 

    def reset_active_indices(self):   
        self._active_indices = self._all_indices.copy()    

    def build_distributed_sampler(self, shuffle, num_replicas, rank):  
        from ..afss_sampler import AFSSDistributedSampler     

        return AFSSDistributedSampler( 
            self, 
            num_replicas=num_replicas,   
            rank=rank,
            shuffle=shuffle,  
            drop_last=False,
        )  
 
    @property 
    def active_indices(self):  
        return self._active_indices 
   
    @property    
    def all_indices(self):
        return self._all_indices 
 
    @property     
    def base_dataset(self):
        return self.dataset
