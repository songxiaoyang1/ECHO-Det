"""   
Copyright (c) 2024 The DEIM Authors. All Rights Reserved. 
"""
     
import torch.nn as nn     
from ..core import register
from .o2m_c import O2MCHead 


__all__ = ['DEIM', ]
    
    
@register() 
class DEIM(nn.Module):
    __inject__ = ['backbone', 'encoder', 'decoder', ]
    __share__ = ['num_classes', ]
     
    def __init__(self, \
        backbone: nn.Module,   
        encoder: nn.Module,
        decoder: nn.Module,     
        o2m_c=False,
        num_classes=80,     
    ):  
        super().__init__()   
        self.backbone = backbone
        self.decoder = decoder
        self.encoder = encoder
        if o2m_c and not all( 
            hasattr(encoder, attr) for attr in ('out_channels', 'out_strides')
        ):
            raise ValueError('o2m_c requires encoder.out_channels and encoder.out_strides')    
        self.o2m_c = ( 
            O2MCHead(encoder.out_channels, encoder.out_strides, num_classes)
            if o2m_c else None    
        )
     
    def forward(self, x, targets=None):
        features = self.encoder(self.backbone(x))     
        outputs = self.decoder(features, targets)   
        if self.training and self.o2m_c is not None:
            outputs['o2m_c'] = self.o2m_c(features)
     
        return outputs

    def deploy(self, ):
        self.eval()  
        self.o2m_c = None
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):  
                m.convert_to_deploy()
        return self
