"""Optional FlashAttention-backed self-attention wrapper."""     
    
import torch 
import torch.nn as nn
import torch.nn.functional as F
  
try: 
    from flash_attn import flash_attn_func
except (ImportError, OSError):    
    flash_attn_func = None     

class FlashSelfAttention(nn.MultiheadAttention):
    """Self-attention with nn.MultiheadAttention-compatible parameter names."""   
     
    def __init__(
        self,     
        embed_dim,  
        num_heads,     
        dropout=0.0,
        bias=True,     
        batch_first=True,     
        device=None,     
        dtype=None,    
    ):    
        if not batch_first: 
            raise ValueError("FlashSelfAttention currently requires batch_first=True") 
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__( 
            embed_dim,    
            num_heads, 
            dropout=dropout,  
            bias=bias,
            batch_first=batch_first,     
            **factory_kwargs,     
        )

    def _project_qkv(self, query, key, value):    
        q_weight, k_weight, v_weight = self.in_proj_weight.chunk(3, dim=0) 
        if self.in_proj_bias is None:     
            q_bias = k_bias = v_bias = None   
        else:     
            q_bias, k_bias, v_bias = self.in_proj_bias.chunk(3, dim=0)
        q = F.linear(query, q_weight, q_bias)
        k = F.linear(key, k_weight, k_bias)
        v = F.linear(value, v_weight, v_bias)     
        return q, k, v

    def _shape_projection(self, x):
        batch_size, seq_len, _ = x.shape     
        return x.reshape(batch_size, seq_len, self.num_heads, self.head_dim) 

    def _can_use_flash(self, q, k, v, attn_mask, key_padding_mask, need_weights):
        return (  
            flash_attn_func is not None 
            and attn_mask is None
            and key_padding_mask is None
            and not need_weights    
            and q.is_cuda
            and k.is_cuda   
            and v.is_cuda     
        )    

    def forward(
        self,   
        query, 
        key,
        value,  
        key_padding_mask=None,  
        need_weights=True,
        attn_mask=None,
        average_attn_weights=True,
        is_causal=False,
    ):
        if not query.dim() == key.dim() == value.dim() == 3:
            attn_output, attn_weights = super().forward( 
                query,    
                key,
                value,
                key_padding_mask=key_padding_mask,
                need_weights=False,     
                attn_mask=attn_mask,
                average_attn_weights=average_attn_weights,
                is_causal=is_causal,  
            )   
            return attn_output, attn_weights if need_weights else None    
   
        q, k, v = self._project_qkv(query, key, value)
        batch_size, len_q, _ = q.shape
        q = self._shape_projection(q)
        k = self._shape_projection(k) 
        v = self._shape_projection(v)
        dropout_p = self.dropout if self.training else 0.0

        if self._can_use_flash(q, k, v, attn_mask, key_padding_mask, need_weights):     
            ori_dtype = q.dtype     
            attn_output = flash_attn_func(     
                q.half().contiguous(),
                k.half().contiguous(),
                v.half().contiguous(),  
                dropout_p=dropout_p,    
                causal=is_causal,  
            ).to(ori_dtype)
            attn_output = attn_output.reshape(batch_size, len_q, self.embed_dim)
            return self.out_proj(attn_output), None    

        attn_output, attn_weights = super().forward(    
            query,
            key,
            value, 
            key_padding_mask=key_padding_mask,   
            need_weights=False,
            attn_mask=attn_mask,
            average_attn_weights=average_attn_weights,
            is_causal=is_causal,
            )    
        return attn_output, attn_weights if need_weights else None 
