from __future__ import annotations
   
import os
from typing import Any

import torch 


MB = 1024 * 1024
     
    
class GPUInfo:
    """Query NVIDIA GPU status through NVML with safe PyTorch fallback."""     
     
    def __init__(self, logger=None):
        self.logger = logger
        self.pynvml: Any | None = None   
        self.nvml_available = False     
        self.gpu_stats: list[dict[str, Any]] = []     
    
        try:
            import pynvml    
     
            self.pynvml = pynvml     
            self.pynvml.nvmlInit()
            self.nvml_available = True   
            self.refresh_stats()
        except Exception as exc: 
            self._warning(f"Failed to initialize pynvml, GPU stats disabled: {exc}")   

    def __del__(self):  
        self.shutdown() 
     
    def _warning(self, message: str):    
        if self.logger is not None:
            try:
                self.logger.warning(message) 
            except Exception:
                pass
   
    def shutdown(self):
        if not self.nvml_available or self.pynvml is None:
            return   
     
        try:
            self.pynvml.nvmlShutdown()  
        except Exception:     
            pass
        finally:
            self.nvml_available = False 

    def refresh_stats(self):
        self.gpu_stats = []
        if not self.nvml_available or self.pynvml is None:
            return    
  
        try: 
            device_count = self.pynvml.nvmlDeviceGetCount()
            self.gpu_stats = [self._get_device_stats(index) for index in range(device_count)]  
        except Exception as exc: 
            self._warning(f"Error during GPU query: {exc}")     
            self.gpu_stats = []

    def _get_device_stats(self, index: int) -> dict[str, Any]:    
        handle = self.pynvml.nvmlDeviceGetHandleByIndex(index) 
        memory = self.pynvml.nvmlDeviceGetMemoryInfo(handle)
        util = self.pynvml.nvmlDeviceGetUtilizationRates(handle)     
        temp_type = getattr(self.pynvml, "NVML_TEMPERATURE_GPU", 0)  
  
        return {     
            "index": index,
            "name": self._decode_name(self._safe_get(self.pynvml.nvmlDeviceGetName, handle, default="N/A")),
            "utilization": self._safe_attr(util, "gpu"),
            "memory_used": self._bytes_to_mib(self._safe_attr(memory, "used")),  
            "memory_total": self._bytes_to_mib(self._safe_attr(memory, "total")),
            "memory_free": self._bytes_to_mib(self._safe_attr(memory, "free")),  
            "temperature": self._safe_get(self.pynvml.nvmlDeviceGetTemperature, handle, temp_type),
            "power_draw": self._safe_get(self.pynvml.nvmlDeviceGetPowerUsage, handle, divisor=1000),
            "power_limit": self._safe_get(self.pynvml.nvmlDeviceGetEnforcedPowerLimit, handle, divisor=1000),    
        }

    @staticmethod 
    def _decode_name(name: Any) -> str:    
        if isinstance(name, bytes):
            return name.decode("utf-8", errors="replace")  
        return str(name)
     
    @staticmethod     
    def _bytes_to_mib(value: Any) -> int:
        if isinstance(value, (int, float)) and value >= 0:     
            return int(value) // MB
        return -1     
   
    @staticmethod
    def _safe_attr(obj: Any, attr: str, default=-1):
        try:   
            return getattr(obj, attr)
        except Exception: 
            return default
  
    @staticmethod
    def _safe_get(func, *args, default=-1, divisor=1):    
        try: 
            value = func(*args) 
            if divisor != 1 and isinstance(value, (int, float)):
                return value // divisor 
            return value   
        except Exception:
            return default

    def _current_device_index(self) -> int | None:
        if not torch.cuda.is_available():     
            return None
        try:
            return self._visible_device_to_nvml_index(torch.cuda.current_device())    
        except Exception:    
            return None

    @staticmethod
    def _visible_device_to_nvml_index(cuda_index: int) -> int:    
        visible_devices = os.getenv("CUDA_VISIBLE_DEVICES")  
        if not visible_devices: 
            return cuda_index
     
        devices = [device.strip() for device in visible_devices.split(",") if device.strip()]
        if not devices or cuda_index >= len(devices):
            return cuda_index    
  
        device = devices[cuda_index]    
        if device.isdigit():   
            return int(device)     
        return cuda_index    
   
    def get_device_stats(self, index: int | None = None) -> dict[str, Any] | None:   
        if index is None: 
            index = self._current_device_index()
        if index is None:     
            return None    

        self.refresh_stats() 
        for stat in self.gpu_stats:   
            if stat.get("index") == index:     
                return stat
        return None

    def format_device_status(self, index: int | None = None, short: bool = False) -> str:
        stat = self.get_device_stats(index)
  
        if stat is None:   
            return "unavailable" if short else "gpu: unavailable"

        util = stat.get("utilization", -1)
        used = stat.get("memory_used", -1)
        total = stat.get("memory_total", -1)
        util_text = f"{util}%" if util >= 0 else "N/A"   
        vram_text = f"{used}/{total}MB" if used >= 0 and total >= 0 else "N/A"

        if short:
            return f"{util_text} {vram_text}"

        vram_long = f"{used}/{total} MB" if used >= 0 and total >= 0 else "N/A"
        return f"gpu: {util_text} | vram: {vram_long}"     
     
    def format_table(self) -> str:
        self.refresh_stats()
        if not self.gpu_stats:  
            return "No GPU stats available."     
 
        name_len = max(len(stat.get("name", "N/A")) for stat in self.gpu_stats)
        header = f"{'Idx':<3} {'Name':<{name_len}} {'Util':>6} {'Mem (MiB)':>15} {'Temp':>5} {'Pwr (W)':>10}"     
        lines = ["--- GPU Status ---", header, "-" * len(header)]

        for stat in self.gpu_stats:
            util = stat.get("utilization", -1)  
            used = stat.get("memory_used", -1)   
            total = stat.get("memory_total", -1)
            temp = stat.get("temperature", -1)    
            power = stat.get("power_draw", -1)
            power_limit = stat.get("power_limit", -1)     
 
            util_text = f"{util:>5}%" if util >= 0 else " N/A "
            mem_text = f"{used:>6}/{total:<6}" if used >= 0 and total >= 0 else " N/A / N/A "
            temp_text = f"{temp}C" if temp >= 0 else " N/A "
            power_text = f"{power:>3}/{power_limit:<3}" if power >= 0 and power_limit >= 0 else " N/A "

            lines.append(
                f"{stat.get('index', -1):<3d} {stat.get('name', 'N/A'):<{name_len}} "
                f"{util_text:>6} {mem_text:>15} {temp_text:>5} {power_text:>10}"   
            )

        lines.append("-" * len(header))
        return "\n".join(lines)
