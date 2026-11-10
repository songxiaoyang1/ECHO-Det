"""
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.    
"""     
   
import copy, torch, thop, time, os     
from calflops import calculate_flops
from typing import Any, Tuple    
from ..logger_module import get_logger

RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"  
logger = get_logger(__name__)
 

class _TensorInputDict(dict): 
    """Dict input wrapper compatible with profilers that call `.to(device)` on each arg."""    
    
    @staticmethod    
    def _move(value, device):     
        if hasattr(value, "to"):   
            return value.to(device)
        if isinstance(value, dict):
            return {k: _TensorInputDict._move(v, device) for k, v in value.items()}     
        if isinstance(value, list):     
            return [_TensorInputDict._move(v, device) for v in value]     
        if isinstance(value, tuple):  
            return tuple(_TensorInputDict._move(v, device) for v in value)
        return value
  
    def to(self, device):   
        return _TensorInputDict({k: self._move(v, device) for k, v in self.items()})    

    
def _is_multimodal_profile_model(model: torch.nn.Module) -> bool:
    backbone = getattr(model, "backbone", None)
    if backbone is None or not hasattr(backbone, "children"):
        return False   
     
    for module in list(backbone.children())[:4]:
        if module.__class__.__name__ in {"GetRGBModalityTensor", "GetNPYModalityTensor"}:
            return True
    return False   

     
def _build_profile_inputs(input_shape: Tuple[int, int, int, int], device: torch.device, multimodal: bool):     
    b, _, h, w = input_shape     
    if multimodal:    
        return (_TensorInputDict({
            "rgb": torch.randn(size=(b, 3, h, w), device=device),
            "npy": torch.randn(size=(b, 1, h, w), device=device),     
        }),)

    return (torch.randn(size=input_shape, device=device),)
 
    
def count_model_stats( 
    model: torch.nn.Module,
    example_inputs: Tuple[Any, ...],  
    *,  
    print_results: bool = False,
) -> Tuple[int, int, int]:   
    """Count FLOPs, MACs, and parameters for the supplied model inputs."""
    profile_inputs = tuple(
        _TensorInputDict(value) if isinstance(value, dict) and not hasattr(value, "to") else value 
        for value in example_inputs  
    )    
    try:    
        flops, macs, params = calculate_flops(
            model=model,    
            input_shape=None,
            args=list(profile_inputs),
            kwargs={}, 
            output_as_string=False,
            print_results=print_results,
            print_detailed=False,   
        )    
        return int(flops), int(macs), int(params) 
    except Exception:  
        logger.warning(RED + "calculate_flops failed.. using thop instead.." + RESET)
        try:
            macs = thop.profile(model, inputs=profile_inputs, verbose=False)[0]  
        except Exception as error:  
            raise RuntimeError("Unable to calculate model FLOPs/MACs.") from error     
        params = sum(parameter.numel() for parameter in model.parameters())
        return int(macs * 2), int(macs), int(params)
   
def stats(   
    cfg,     
    input_shape: Tuple=(1, 3, 640, 640), module=None) -> Tuple[int, dict]:  

    # base_size = cfg.train_dataloader.collate_fn.base_size
    input_shape = (1, 3, *cfg.yaml_cfg['eval_spatial_size'])

    if module:
        model_for_info = copy.deepcopy(module)
    else:
        model_for_info = copy.deepcopy(cfg.model).deploy()    
    
    if cfg.device:    
        device = torch.device(cfg.device)   
    else:    
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    
 
    model_for_info.to(device)     
    is_multimodal_model = _is_multimodal_profile_model(model_for_info)   
    profile_inputs = _build_profile_inputs(input_shape, device, is_multimodal_model)     
    try:
        flops, macs, params = count_model_stats(model_for_info, profile_inputs, print_results=True)
        flops, macs = thop.clever_format([flops, macs], format="%.3f")    
    except RuntimeError:
        logger.warning(RED + "thop profile failed.. set FLOPs/MACs to N/A." + RESET)   
        flops, macs = "N/A", "N/A"
        params = sum(p.numel() for p in model_for_info.parameters())
    del model_for_info
 
    return params, {"Model FLOPs:%s   MACs:%s   Params:%s" %(flops, macs, params)}

def get_weight_size(module):
    timestamp = f'{int(time.time() * 1000)}.pth' # 毫秒级时间戳
    torch.save(module.state_dict(), timestamp)
    stats = os.stat(timestamp)    
    logger.info(ORANGE + f'only model size: {stats.st_size / (1024 ** 2):.1f} MB' + RESET)
    os.remove(timestamp)
