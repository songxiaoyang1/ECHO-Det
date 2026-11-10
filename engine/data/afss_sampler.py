import math
import random   
     
from torch.utils.data import Sampler     


class AFSSDistributedSampler(Sampler):    
    def __init__(self, dataset, num_replicas=1, rank=0, shuffle=True, drop_last=False, seed=0):    
        self.dataset = dataset     
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle     
        self.drop_last = drop_last     
        self.seed = seed   
        self.epoch = 0

    def __iter__(self):
        indices = list(range(len(self.dataset.active_indices)))  
        if self.shuffle:    
            rng = random.Random(self.seed + self.epoch)
            rng.shuffle(indices)
 
        if self.drop_last: 
            total_size = (len(indices) // self.num_replicas) * self.num_replicas
            indices = indices[:total_size]
        else:
            total_size = math.ceil(max(len(indices), 1) / self.num_replicas) * self.num_replicas
            if len(indices) < total_size: 
                indices = indices + indices[: total_size - len(indices)]     

        return iter(indices[self.rank:total_size:self.num_replicas])    
  
    def __len__(self):
        if self.drop_last:   
            return len(self.dataset.active_indices) // self.num_replicas    
        return math.ceil(max(len(self.dataset.active_indices), 1) / self.num_replicas)

    def set_epoch(self, epoch):
        self.epoch = epoch
