import copy    
import functools
import math
from collections import OrderedDict 
from typing import List 
   
import torch
import torch.nn as nn 
import torch.nn.functional as F  
import torch.nn.init as init 

from ..core import register
from ..deim.dfine_utils import distance2bbox, weighting_function
from ..deim.utils import bias_init_with_prob, deformable_attention_core_func_v2, get_activation, inverse_sigmoid
from ..misc.deformable_detr_msda import deformable_detr_msda_core_default
from ..misc.flash_attention import FlashSelfAttention    
from .denoising import get_contrastive_denoising_training_group   
from ..deim.rtdetrv3_supervision import build_group_attn_mask, make_mgsa_query_mask    
    
__all__ = ["MLP", "MSDeformableAttention", "OBBRTDETRTransformerv2", "OBBDFINETransformer"]


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, act="relu"):
        super().__init__() 
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)     
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))    
        self.act = get_activation(act)
  
    def forward(self, x):    
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < self.num_layers - 1 else layer(x) 
        return x


class MSDeformableAttention(nn.Module):
    def __init__(
        self,
        embed_dim=256,    
        num_heads=8,
        num_levels=4,
        num_points=4,
        method="default",    
        offset_scale=0.5, 
        value_shape="reshape",  
    ):
        super().__init__()   
        self.embed_dim = embed_dim  
        self.num_heads = num_heads
        self.num_levels = num_levels 
        self.offset_scale = offset_scale     
        self.num_points_list = num_points if isinstance(num_points, list) else [num_points for _ in range(num_levels)] 
        if len(self.num_points_list) != num_levels:
            raise ValueError("num_points list length must match num_levels")

        num_points_scale = [1 / n for n in self.num_points_list for _ in range(n)] 
        self.register_buffer("num_points_scale", torch.tensor(num_points_scale, dtype=torch.float32))    

        self.total_points = num_heads * sum(self.num_points_list)     
        self.method = method
        self.head_dim = embed_dim // num_heads
        if self.head_dim * num_heads != embed_dim:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.sampling_offsets = nn.Linear(embed_dim, self.total_points * 2)
        self.attention_weights = nn.Linear(embed_dim, self.total_points)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim) 
        self.ms_deformable_attn_core = functools.partial(
            deformable_attention_core_func_v2,
            method=self.method,
            value_shape=value_shape,    
        )   

        self._reset_parameters()     
        if method == "discrete": 
            for p in self.sampling_offsets.parameters():
                p.requires_grad = False     

    def _reset_parameters(self):
        init.constant_(self.sampling_offsets.weight, 0)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (2.0 * math.pi / self.num_heads)     
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = grid_init / grid_init.abs().max(-1, keepdim=True).values
        grid_init = grid_init.reshape(self.num_heads, 1, 2).tile([1, sum(self.num_points_list), 1])
        scaling = torch.concat([torch.arange(1, n + 1) for n in self.num_points_list]).reshape(1, -1, 1)  
        grid_init *= scaling     
        self.sampling_offsets.bias.data[...] = grid_init.flatten()
        init.constant_(self.attention_weights.weight, 0)     
        init.constant_(self.attention_weights.bias, 0)
        init.xavier_uniform_(self.value_proj.weight)
        init.constant_(self.value_proj.bias, 0) 
        init.xavier_uniform_(self.output_proj.weight)   
        init.constant_(self.output_proj.bias, 0)

    def forward(     
        self,
        query: torch.Tensor,
        reference_points: torch.Tensor,
        value: torch.Tensor,    
        value_spatial_shapes: List[int],
        value_mask: torch.Tensor = None, 
    ):
        bs, len_q = query.shape[:2]
        len_v = value.shape[1]
     
        value = self.value_proj(value)
        if value_mask is not None:
            value = value * value_mask.to(value.dtype).unsqueeze(-1)   
        value = value.reshape(bs, len_v, self.num_heads, self.head_dim)
   
        sampling_offsets = self.sampling_offsets(query)
        sampling_offsets = sampling_offsets.reshape(bs, len_q, self.num_heads, sum(self.num_points_list), 2)
        attention_weights = self.attention_weights(query).reshape(bs, len_q, self.num_heads, sum(self.num_points_list))  
        attention_weights = F.softmax(attention_weights, dim=-1)

        if reference_points.shape[-1] == 2:    
            offset_normalizer = torch.tensor(value_spatial_shapes, device=query.device, dtype=query.dtype)    
            offset_normalizer = offset_normalizer.flip([1]).reshape(1, 1, 1, self.num_levels, 1, 2)
            sampling_locations = (    
                reference_points.reshape(bs, len_q, 1, self.num_levels, 1, 2)  
                + sampling_offsets.reshape(bs, len_q, self.num_heads, self.num_levels, -1, 2) / offset_normalizer     
            )  
            sampling_locations = sampling_locations.flatten(3, 4)
        elif reference_points.shape[-1] == 4:
            num_points_scale = self.num_points_scale.to(device=query.device, dtype=query.dtype).unsqueeze(-1) 
            ref_wh = torch.cat( 
                [
                    reference_points[:, :, level : level + 1, 2:].repeat(1, 1, n, 1)
                    for level, n in enumerate(self.num_points_list)  
                ],   
                dim=2,    
            )
            offset = sampling_offsets * num_points_scale * ref_wh[:, :, None] * self.offset_scale
            sampling_locations = reference_points[:, :, None, :, :2].repeat_interleave(
                torch.tensor(self.num_points_list, device=query.device), dim=3    
            ) + offset
        elif reference_points.shape[-1] == 5: 
            cos_angle = torch.cos(reference_points[..., 4:])
            sin_angle = torch.sin(reference_points[..., 4:])
            rot = torch.cat([cos_angle, -sin_angle, sin_angle, cos_angle], dim=-1)
            rot = rot.reshape(bs, len_q, self.num_levels, 2, 2)
            wh = reference_points[..., 2:4] * self.offset_scale     
            rotated_wh = torch.einsum("bqlij,bqlj->bqli", rot, wh)
            rotated_wh = torch.cat(
                [rotated_wh[:, :, level : level + 1].repeat(1, 1, n, 1) for level, n in enumerate(self.num_points_list)],
                dim=2,     
            ) 
            reference_xy = torch.cat(     
                [     
                    reference_points[:, :, level : level + 1, :2].repeat(1, 1, n, 1)   
                    for level, n in enumerate(self.num_points_list)    
                ],
                dim=2,    
            )
            num_points_scale = self.num_points_scale.to(device=query.device, dtype=query.dtype).unsqueeze(-1)
            offset = sampling_offsets * num_points_scale * rotated_wh[:, :, None]
            sampling_locations = reference_xy[:, :, None] + offset
        else:
            raise ValueError(f"Last dim of reference_points must be 2, 4, or 5, got {reference_points.shape[-1]}")
   
        output = self.ms_deformable_attn_core(
            value,  
            value_spatial_shapes,    
            sampling_locations,
            attention_weights,   
            self.num_points_list,  
        )     
        return self.output_proj(output)   


class TransformerDecoderLayer(nn.Module):
    def __init__(   
        self, 
        d_model=256,
        n_head=8,  
        dim_feedforward=1024, 
        dropout=0.0,
        activation="relu",     
        n_levels=4,     
        n_points=4,
        cross_attn_method="default",
        value_shape="reshape",
    ): 
        super().__init__()
        self.self_attn = FlashSelfAttention(d_model, n_head, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)  
        self.cross_attn = MSDeformableAttention(    
            d_model, 
            n_head,
            n_levels,   
            n_points,     
            method=cross_attn_method,  
            value_shape=value_shape,
        )
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = get_activation(activation)    
        self.dropout3 = nn.Dropout(dropout)   
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)
        self._reset_parameters()     

    def _reset_parameters(self):
        init.xavier_uniform_(self.linear1.weight)    
        init.xavier_uniform_(self.linear2.weight)    
    
    @staticmethod
    def with_pos_embed(tensor, pos):    
        return tensor if pos is None else tensor + pos  
     
    def forward_ffn(self, target):  
        return self.linear2(self.dropout3(self.activation(self.linear1(target))))

    def forward(  
        self, 
        target,  
        reference_points,
        memory,     
        memory_spatial_shapes,   
        attn_mask=None,     
        memory_mask=None,  
        query_pos_embed=None, 
    ):   
        q = k = self.with_pos_embed(target, query_pos_embed)
        target2, _ = self.self_attn(q, k, value=target, attn_mask=attn_mask, need_weights=False)
        target = self.norm1(target + self.dropout1(target2))

        target2 = self.cross_attn(
            self.with_pos_embed(target, query_pos_embed),     
            reference_points,   
            memory,
            memory_spatial_shapes,
            memory_mask,
        ) 
        target = self.norm2(target + self.dropout2(target2))
        target = self.norm3(target + self.dropout4(self.forward_ffn(target)))     
        return target   


class TransformerDecoder(nn.Module):
    def __init__(self, hidden_dim, decoder_layer, num_layers, eval_idx=-1, angle_factor=math.pi):    
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])     
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx    
        self.angle_factor = angle_factor     
 
    # def forward(  
    #     self,
    #     target, 
    #     ref_points_unact,
    #     memory,
    #     memory_spatial_shapes,
    #     bbox_head,     
    #     score_head,
    #     query_pos_head,
    #     attn_mask=None,
    #     memory_mask=None,
    # ): 
    #     dec_out_bboxes = []    
    #     dec_out_logits = []   
    #     ref_points_detach = F.sigmoid(ref_points_unact[..., :4])
    #     output = target   
  
    #     for i, layer in enumerate(self.layers):     
    #         ref_points_input = ref_points_detach.unsqueeze(2).repeat(1, 1, len(memory_spatial_shapes), 1)
    #         query_pos_embed = query_pos_head(ref_points_detach)
 
    #         output = layer(output, ref_points_input, memory, memory_spatial_shapes, attn_mask, memory_mask, query_pos_embed) 

    #         box_cxcywhr = bbox_head[i](output)   
    #         inter_ref_bbox = box_cxcywhr[..., :4] + inverse_sigmoid(ref_points_detach)   
    #         inter_ref_bbox = F.sigmoid(torch.cat([inter_ref_bbox, box_cxcywhr[..., 4:]], dim=-1))   

    #         if self.training:
    #             dec_out_logits.append(score_head[i](output))
    #             if i == 0:     
    #                 dec_out_bboxes.append(inter_ref_bbox)     
    #             else: 
    #                 box_cxcywhr = bbox_head[i](output)
    #                 inter_ref_bbox = box_cxcywhr[..., :4] + inverse_sigmoid(ref_points) 
    #                 inter_ref_bbox = F.sigmoid(torch.cat([inter_ref_bbox, box_cxcywhr[..., 4:]], dim=-1))  
    #                 dec_out_bboxes.append(inter_ref_bbox)
    #         elif i == self.eval_idx:     
    #             dec_out_logits.append(score_head[i](output))
    #             dec_out_bboxes.append(inter_ref_bbox)
    #             break 
    
    #         ref_points = inter_ref_bbox[..., :4]    
    #         ref_points_detach = inter_ref_bbox[..., :4].detach() 

    #     return torch.stack(dec_out_bboxes), torch.stack(dec_out_logits)
 
    def forward(    
        self,   
        target,
        ref_points_unact, 
        memory,    
        memory_spatial_shapes,
        bbox_head,
        score_head,
        query_pos_head,    
        attn_mask=None,    
        memory_mask=None,
    ):
        dec_out_bboxes = []     
        dec_out_logits = []
        ref_points_detach = F.sigmoid(ref_points_unact)     
        output = target

        for i, layer in enumerate(self.layers):   
            ref_points_input = ref_points_detach.unsqueeze(2).repeat(1, 1, len(memory_spatial_shapes), 1)
            ref_points_input[..., -1] *= self.angle_factor 
            query_pos_embed = query_pos_head(ref_points_detach) 
  
            output = layer(output, ref_points_input, memory, memory_spatial_shapes, attn_mask, memory_mask, query_pos_embed) 
   
            inter_ref_bbox = F.sigmoid(bbox_head[i](output) + inverse_sigmoid(ref_points_detach, eps=1e-3))
     
            if self.training:
                dec_out_logits.append(score_head[i](output))     
                if i == 0:   
                    dec_out_bboxes.append(inter_ref_bbox)
                else:     
                    dec_out_bboxes.append(F.sigmoid(bbox_head[i](output) + inverse_sigmoid(ref_points)))    
            elif i == self.eval_idx:    
                dec_out_logits.append(score_head[i](output))
                dec_out_bboxes.append(inter_ref_bbox)
                break 

            ref_points = inter_ref_bbox 
            ref_points_detach = inter_ref_bbox.detach()   

        return torch.stack(dec_out_bboxes), torch.stack(dec_out_logits)
 
 
class OBBDFINEMSDeformableAttention(nn.Module):
    def __init__(
        self, 
        embed_dim=256,
        num_heads=8,
        num_levels=4,
        num_points=4,    
        method="default",     
        offset_scale=0.5,  
    ):
        super().__init__()
        self.embed_dim = embed_dim    
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.offset_scale = offset_scale
        self.num_points_list = num_points if isinstance(num_points, list) else [num_points for _ in range(num_levels)]
        if len(self.num_points_list) != num_levels:  
            raise ValueError("num_points list length must match num_levels")

        num_points_scale = [1 / n for n in self.num_points_list for _ in range(n)]     
        self.register_buffer("num_points_scale", torch.tensor(num_points_scale, dtype=torch.float32))
 
        self.total_points = num_heads * sum(self.num_points_list)
        self.method = method   
        self.head_dim = embed_dim // num_heads
        if self.head_dim * num_heads != embed_dim:    
            raise ValueError("embed_dim must be divisible by num_heads")

        self.sampling_offsets = nn.Linear(embed_dim, self.total_points * 2)
        self.attention_weights = nn.Linear(embed_dim, self.total_points)    
        self.ms_deformable_attn_core = functools.partial(deformable_attention_core_func_v2, method=self.method) 
   
        self._reset_parameters()
        if method == "discrete":  
            for p in self.sampling_offsets.parameters():     
                p.requires_grad = False    
    
    def _reset_parameters(self):  
        init.constant_(self.sampling_offsets.weight, 0)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)     
        grid_init = grid_init / grid_init.abs().max(-1, keepdim=True).values 
        grid_init = grid_init.reshape(self.num_heads, 1, 2).tile([1, sum(self.num_points_list), 1])
        scaling = torch.concat([torch.arange(1, n + 1) for n in self.num_points_list]).reshape(1, -1, 1)
        grid_init *= scaling   
        self.sampling_offsets.bias.data[...] = grid_init.flatten()     
        init.constant_(self.attention_weights.weight, 0)
        init.constant_(self.attention_weights.bias, 0)     

    def _expand_reference_levels(self, reference_points):   
        if reference_points.shape[2] == 1:   
            return reference_points.repeat(1, 1, self.num_levels, 1)
        if reference_points.shape[2] != self.num_levels:    
            raise ValueError(     
                f"Expected 1 or {self.num_levels} reference levels, got {reference_points.shape[2]}"
            ) 
        return reference_points 

    def forward(     
        self,
        query: torch.Tensor,
        reference_points: torch.Tensor, 
        value: torch.Tensor, 
        value_spatial_shapes: List[int],
    ):   
        bs, len_q = query.shape[:2]     

        sampling_offsets = self.sampling_offsets(query)     
        sampling_offsets = sampling_offsets.reshape(bs, len_q, self.num_heads, sum(self.num_points_list), 2)   
        attention_weights = self.attention_weights(query).reshape(bs, len_q, self.num_heads, sum(self.num_points_list))
        attention_weights = F.softmax(attention_weights, dim=-1)
    
        if reference_points.shape[-1] == 2: 
            reference_points = self._expand_reference_levels(reference_points) 
            offset_normalizer = torch.tensor(value_spatial_shapes, device=query.device, dtype=query.dtype)
            offset_normalizer = offset_normalizer.flip([1]).reshape(1, 1, 1, self.num_levels, 1, 2)    
            sampling_locations = (
                reference_points.reshape(bs, len_q, 1, self.num_levels, 1, 2)
                + sampling_offsets.reshape(bs, len_q, self.num_heads, self.num_levels, -1, 2) / offset_normalizer 
            )     
            sampling_locations = sampling_locations.flatten(3, 4)
        elif reference_points.shape[-1] == 4:
            reference_points = self._expand_reference_levels(reference_points)     
            num_points_scale = self.num_points_scale.to(device=query.device, dtype=query.dtype).unsqueeze(-1)
            ref_wh = torch.cat(  
                [    
                    reference_points[:, :, level : level + 1, 2:].repeat(1, 1, n, 1)
                    for level, n in enumerate(self.num_points_list)     
                ],
                dim=2,
            )   
            reference_xy = torch.cat(
                [
                    reference_points[:, :, level : level + 1, :2].repeat(1, 1, n, 1)   
                    for level, n in enumerate(self.num_points_list)     
                ],
                dim=2,   
            )    
            offset = sampling_offsets * num_points_scale * ref_wh[:, :, None] * self.offset_scale  
            sampling_locations = reference_xy[:, :, None] + offset 
        elif reference_points.shape[-1] == 5:
            reference_points = self._expand_reference_levels(reference_points)
            num_points_scale = self.num_points_scale.to(device=query.device, dtype=query.dtype).unsqueeze(-1)
            cos_angle = torch.cos(reference_points[..., 4:])
            sin_angle = torch.sin(reference_points[..., 4:])   
            rot = torch.cat([cos_angle, -sin_angle, sin_angle, cos_angle], dim=-1)
            rot = rot.reshape(bs, len_q, self.num_levels, 2, 2)   
            wh = reference_points[..., 2:4] * self.offset_scale   
            rotated_wh = torch.einsum("bqlij,bqlj->bqli", rot, wh)  
            rotated_wh = torch.cat(
                [   
                    rotated_wh[:, :, level : level + 1].repeat(1, 1, n, 1)
                    for level, n in enumerate(self.num_points_list)    
                ],
                dim=2,
            )   
            reference_xy = torch.cat( 
                [    
                    reference_points[:, :, level : level + 1, :2].repeat(1, 1, n, 1)    
                    for level, n in enumerate(self.num_points_list)
                ],   
                dim=2,
            )    
            offset = sampling_offsets * num_points_scale * rotated_wh[:, :, None] 
            sampling_locations = reference_xy[:, :, None] + offset  
        else: 
            raise ValueError(f"Last dim of reference_points must be 2, 4, or 5, got {reference_points.shape[-1]}")

        if self.method == "default":
            return deformable_detr_msda_core_default(    
                value,    
                value_spatial_shapes,
                sampling_locations, 
                attention_weights,
                self.num_points_list,
            )
  
        return self.ms_deformable_attn_core(
            value,
            value_spatial_shapes,   
            sampling_locations,  
            attention_weights,    
            self.num_points_list,   
        )


class Gate(nn.Module):     
    def __init__(self, d_model):
        super().__init__()
        self.gate = nn.Linear(2 * d_model, 2 * d_model)
        bias = bias_init_with_prob(0.5)
        init.constant_(self.gate.bias, bias)   
        init.constant_(self.gate.weight, 0)     
        self.norm = nn.LayerNorm(d_model)   

    def forward(self, x1, x2):
        gate_input = torch.cat([x1, x2], dim=-1)  
        gates = torch.sigmoid(self.gate(gate_input))   
        gate1, gate2 = gates.chunk(2, dim=-1)    
        return self.norm(gate1 * x1 + gate2 * x2)    
 

class Integral(nn.Module):
    def __init__(self, reg_max=32):    
        super().__init__()   
        self.reg_max = reg_max    
 
    def forward(self, x, project):  
        shape = x.shape 
        x = F.softmax(x.reshape(-1, self.reg_max + 1), dim=1)     
        x = F.linear(x, project.to(x.device)).reshape(-1, 4)
        return x.reshape(list(shape[:-1]) + [-1])   
    

class LQE(nn.Module):
    def __init__(self, k, hidden_dim, num_layers, reg_max, act="relu"):
        super().__init__()    
        self.k = k 
        self.reg_max = reg_max
        self.reg_conf = MLP(4 * (k + 1), hidden_dim, 1, num_layers, act=act)
        init.constant_(self.reg_conf.layers[-1].bias, 0) 
        init.constant_(self.reg_conf.layers[-1].weight, 0)   

    def forward(self, scores, pred_corners):
        batch_size, length, _ = pred_corners.size()    
        prob = F.softmax(pred_corners.reshape(batch_size, length, 4, self.reg_max + 1), dim=-1)    
        prob_topk, _ = prob.topk(self.k, dim=-1)    
        stat = torch.cat([prob_topk, prob_topk.mean(dim=-1, keepdim=True)], dim=-1)
        quality_score = self.reg_conf(stat.reshape(batch_size, length, -1))
        return scores + quality_score


class OBBDFINETransformerDecoderLayer(nn.Module):    
    def __init__(   
        self, 
        d_model=256,
        n_head=8,   
        dim_feedforward=1024,
        dropout=0.0,
        activation="relu",     
        n_levels=4,
        n_points=4,
        cross_attn_method="default",
        layer_scale=None,
    ):
        super().__init__()
        if layer_scale is not None:     
            dim_feedforward = round(layer_scale * dim_feedforward)
            d_model = round(layer_scale * d_model)

        self.self_attn = FlashSelfAttention(d_model, n_head, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)    
        self.norm1 = nn.LayerNorm(d_model)
        self.cross_attn = OBBDFINEMSDeformableAttention(     
            d_model, 
            n_head,
            n_levels,    
            n_points,
            method=cross_attn_method,
        )     
        self.dropout2 = nn.Dropout(dropout)
        self.gateway = Gate(d_model)
        self.linear1 = nn.Linear(d_model, dim_feedforward)     
        self.activation = get_activation(activation)  
        self.dropout3 = nn.Dropout(dropout)     
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)
        self._reset_parameters()

    def _reset_parameters(self):    
        init.xavier_uniform_(self.linear1.weight)     
        init.xavier_uniform_(self.linear2.weight)

    @staticmethod  
    def with_pos_embed(tensor, pos):    
        return tensor if pos is None else tensor + pos    

    def forward_ffn(self, target):
        return self.linear2(self.dropout3(self.activation(self.linear1(target))))     

    def forward( 
        self,
        target,    
        reference_points,
        value,  
        spatial_shapes,
        attn_mask=None,
        query_pos_embed=None,   
    ): 
        q = k = self.with_pos_embed(target, query_pos_embed) 
        target2, _ = self.self_attn(q, k, value=target, attn_mask=attn_mask, need_weights=False) 
        target = self.norm1(target + self.dropout1(target2))  

        target2 = self.cross_attn(  
            self.with_pos_embed(target, query_pos_embed),    
            reference_points,
            value,
            spatial_shapes,
        ) 
        target = self.gateway(target, self.dropout2(target2))     
        target = self.norm3((target + self.dropout4(self.forward_ffn(target))).clamp(min=-65504, max=65504))
        return target    
   
 
class OBBDFINETransformerDecoder(nn.Module): 
    def __init__(
        self,   
        hidden_dim,   
        decoder_layer,
        decoder_layer_wide,
        num_layers,
        num_head,
        reg_max,
        reg_scale, 
        up,     
        eval_idx=-1,
        layer_scale=2,
        act="relu",  
    ): 
        super().__init__()     
        self.hidden_dim = hidden_dim  
        self.num_layers = num_layers     
        self.layer_scale = layer_scale     
        self.num_head = num_head
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx  
        self.up, self.reg_scale, self.reg_max = up, reg_scale, reg_max
        self.layers = nn.ModuleList(
            [copy.deepcopy(decoder_layer) for _ in range(self.eval_idx + 1)]
            + [copy.deepcopy(decoder_layer_wide) for _ in range(num_layers - self.eval_idx - 1)]     
        ) 
        self.lqe_layers = nn.ModuleList([copy.deepcopy(LQE(4, 64, 2, reg_max, act=act)) for _ in range(num_layers)]) 
   
    def value_op(self, memory, value_proj, value_scale, memory_mask, memory_spatial_shapes):
        value = value_proj(memory) if value_proj is not None else memory  
        value = F.interpolate(memory, size=value_scale) if value_scale is not None else value     
        if memory_mask is not None:
            value = value * memory_mask.to(value.dtype).unsqueeze(-1)  
        value = value.reshape(value.shape[0], value.shape[1], self.num_head, -1)
        split_shape = [h * w for h, w in memory_spatial_shapes]    
        return value.permute(0, 2, 3, 1).split(split_shape, dim=-1)  

    def convert_to_deploy(self):
        self.project = weighting_function(self.reg_max, self.up, self.reg_scale, deploy=True)
        self.layers = self.layers[: self.eval_idx + 1]     
        self.lqe_layers = nn.ModuleList([nn.Identity()] * self.eval_idx + [self.lqe_layers[self.eval_idx]])  

    def forward(    
        self,
        target,
        ref_points_unact,    
        memory,
        spatial_shapes,  
        bbox_head,   
        angle_head,
        score_head,     
        query_pos_head,    
        pre_bbox_head,
        integral,
        up,    
        reg_scale,
        angle_factor,    
        attn_mask=None,   
        memory_mask=None,     
        dn_meta=None,
    ):
        del dn_meta  
        output = target
        output_detach = pred_corners_undetach = 0   
        value = self.value_op(memory, None, None, memory_mask, spatial_shapes)   
     
        dec_out_bboxes = []     
        dec_out_logits = []    
        dec_out_pred_corners = []
        dec_out_refs = []
        project = self.project if hasattr(self, "project") else weighting_function(self.reg_max, up, reg_scale)

        ref_points_detach = F.sigmoid(ref_points_unact)
  
        for i, layer in enumerate(self.layers):
            ref_points_input = ref_points_detach.unsqueeze(2).repeat(1, 1, len(spatial_shapes), 1)
            ref_points_input[..., -1] *= angle_factor
            query_pos_embed = query_pos_head(ref_points_detach).clamp(min=-10, max=10)
 
            if i >= self.eval_idx + 1 and self.layer_scale > 1:    
                query_pos_embed = F.interpolate(query_pos_embed, scale_factor=self.layer_scale)
                value = self.value_op(memory, None, query_pos_embed.shape[-1], memory_mask, spatial_shapes)   
                output = F.interpolate(output, size=query_pos_embed.shape[-1])
                output_detach = output.detach()
    
            output = layer(output, ref_points_input, value, spatial_shapes, attn_mask, query_pos_embed)     
     
            if i == 0:   
                pre_bboxes = F.sigmoid(pre_bbox_head(output) + inverse_sigmoid(ref_points_detach, eps=1e-3))  
                pre_scores = score_head[0](output)
                ref_points_initial = pre_bboxes.detach()  
   
            pred_corners = bbox_head[i](output + output_detach) + pred_corners_undetach    
            refined_xywh = distance2bbox( 
                ref_points_initial[..., :4],     
                integral(pred_corners, project),     
                reg_scale,   
            )
            refined_angle = F.sigmoid(    
                angle_head[i](output + output_detach)
                + inverse_sigmoid(ref_points_detach[..., 4:5], eps=1e-3)
            )
            inter_ref_bbox = torch.cat([refined_xywh, refined_angle], dim=-1)     

            if self.training or i == self.eval_idx: 
                scores = score_head[i](output)
                scores = self.lqe_layers[i](scores, pred_corners)  
                dec_out_logits.append(scores)
                dec_out_bboxes.append(inter_ref_bbox)
                dec_out_pred_corners.append(pred_corners)
                dec_out_refs.append(ref_points_initial)
 
                if not self.training:     
                    break     

            pred_corners_undetach = pred_corners
            ref_points_detach = inter_ref_bbox.detach()   
            output_detach = output.detach()     
 
        return (
            torch.stack(dec_out_bboxes),
            torch.stack(dec_out_logits),    
            torch.stack(dec_out_pred_corners),    
            torch.stack(dec_out_refs),  
            pre_bboxes,   
            pre_scores,
        )  


@register() 
class OBBRTDETRTransformerv2(nn.Module): 
    __share__ = ["num_classes", "eval_spatial_size"]

    def __init__(  
        self,
        num_classes=80,
        hidden_dim=256,    
        num_queries=300,
        feat_channels=[512, 1024, 2048],
        feat_strides=[8, 16, 32],
        num_levels=3,  
        num_points=4,
        nhead=8,
        num_layers=6,
        dim_feedforward=1024,     
        dropout=0.0, 
        activation="relu",  
        num_denoising=100, 
        label_noise_ratio=0.5,
        box_noise_scale=1.0,   
        learn_query_content=False,     
        eval_spatial_size=None,
        eval_idx=-1,   
        eps=1e-2,  
        aux_loss=True,
        cross_attn_method="default", 
        query_select_method="default", 
        value_shape="reshape",   
        mlp_act="relu",
        angle_factor=math.pi,
        denoising_type='xyxya', 
        mgsa_groups=0, mgsa_num_queries=None, mgsa_num_denoising=None, mgsa_mask_ratio=0.1,   
        o2m_t=False, num_queries_o2m=450, o2m_repeat=4,     
        **kwargs, 
    ):  
        super().__init__()   
        kwargs.pop("reg_max", None)
        kwargs.pop("reg_scale", None)
        kwargs.pop("layer_scale", None)    
        if kwargs:    
            raise TypeError(f"Unexpected OBBRTDETRTransformerv2 args: {sorted(kwargs)}")     
        feat_channels = list(feat_channels)  
        feat_strides = list(feat_strides)
        if len(feat_channels) > num_levels:    
            raise ValueError("feat_channels length must be <= num_levels")
        if len(feat_strides) != len(feat_channels):    
            raise ValueError("feat_strides length must match feat_channels")  

        for _ in range(num_levels - len(feat_strides)):     
            feat_strides.append(feat_strides[-1] * 2)   

        self.hidden_dim = hidden_dim
        self.nhead = nhead
        self.feat_strides = feat_strides
        self.num_levels = num_levels
        self.num_classes = num_classes     
        self.num_queries = num_queries
        self.eps = eps
        self.num_layers = num_layers
        self.eval_spatial_size = eval_spatial_size
        self.aux_loss = aux_loss
        self.angle_factor = angle_factor
        self.denoising_type = denoising_type    
        if mgsa_groups < 0 or not 0 <= mgsa_mask_ratio < 1 or o2m_repeat < 1 or num_queries_o2m <= 0:     
            raise ValueError('invalid RT-DETRv3 supervision parameters')    
        if mgsa_num_queries and (len(mgsa_num_queries) != mgsa_groups or any(q <= 0 for q in mgsa_num_queries)):
            raise ValueError('mgsa_num_queries must match mgsa_groups and contain positive values')
        self.mgsa_groups, self.mgsa_num_queries = int(mgsa_groups), list(mgsa_num_queries or [])
        self.mgsa_num_denoising, self.mgsa_mask_ratio = mgsa_num_denoising, mgsa_mask_ratio     
        self.o2m_t, self.num_queries_o2m, self.o2m_repeat = bool(o2m_t), int(num_queries_o2m), int(o2m_repeat)   

        if query_select_method not in ("default", "one2many", "agnostic"):   
            raise ValueError(f"Unsupported query_select_method {query_select_method!r}")
        if cross_attn_method not in ("default", "discrete"):
            raise ValueError(f"Unsupported cross_attn_method {cross_attn_method!r}")  
        self.cross_attn_method = cross_attn_method
        self.query_select_method = query_select_method   

        self._build_input_proj_layer(feat_channels)    
  
        decoder_layer = TransformerDecoderLayer(   
            hidden_dim,    
            nhead,
            dim_feedforward, 
            dropout,
            activation,
            num_levels,  
            num_points,  
            cross_attn_method=cross_attn_method,
            value_shape=value_shape,
        ) 
        self.decoder = TransformerDecoder(hidden_dim, decoder_layer, num_layers, eval_idx, angle_factor) 
  
        self.num_denoising = num_denoising
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale
        if num_denoising > 0:  
            self.denoising_class_embed = nn.Embedding(num_classes + 1, hidden_dim, padding_idx=num_classes)  
            init.normal_(self.denoising_class_embed.weight[:-1]) 
    
        self.learn_query_content = learn_query_content    
        if learn_query_content:    
            self.tgt_embed = nn.Embedding(num_queries, hidden_dim)

        self.query_pos_head = MLP(5, 2 * hidden_dim, hidden_dim, 2, act=mlp_act) 
        self.enc_output = nn.Sequential(
            OrderedDict(
                [ 
                    ("proj", nn.Linear(hidden_dim, hidden_dim)),
                    ("norm", nn.LayerNorm(hidden_dim)),     
                ]    
            )
        )
        self.enc_score_head = nn.Linear(hidden_dim, 1 if query_select_method == "agnostic" else num_classes) 
        self.enc_bbox_head = MLP(hidden_dim, hidden_dim, 5, 3, act=mlp_act)    
        extra_count = self.mgsa_groups + int(self.o2m_t)
        self.map_memory = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim)) if extra_count else None
        if extra_count: 
            self.extra_enc_output = nn.ModuleList([copy.deepcopy(self.enc_output) for _ in range(extra_count)])
            score_dim = 1 if query_select_method == 'agnostic' else num_classes  
            self.extra_enc_score_head = nn.ModuleList([nn.Linear(hidden_dim, score_dim) for _ in range(extra_count)]) 
            self.extra_enc_bbox_head = nn.ModuleList([MLP(hidden_dim, hidden_dim, 5, 3, act=mlp_act) for _ in range(extra_count)]) 
        else:     
            self.extra_enc_output = nn.ModuleList() 
            self.extra_enc_score_head = nn.ModuleList()
            self.extra_enc_bbox_head = nn.ModuleList()   
        self.dec_score_head = nn.ModuleList([nn.Linear(hidden_dim, num_classes) for _ in range(num_layers)])
        self.dec_bbox_head = nn.ModuleList([MLP(hidden_dim, hidden_dim, 5, 3, act=mlp_act) for _ in range(num_layers)])
 
        if self.eval_spatial_size:
            anchors, valid_mask = self._generate_anchors()
            self.register_buffer("anchors", anchors)    
            self.register_buffer("valid_mask", valid_mask)   
  
        self._reset_parameters()
        for output, score, bbox in zip(self.extra_enc_output, self.extra_enc_score_head, self.extra_enc_bbox_head):    
            init.xavier_uniform_(output[0].weight); init.constant_(score.bias, bias_init_with_prob(0.01)); init.constant_(bbox.layers[-1].weight, 0); init.constant_(bbox.layers[-1].bias, 0)
     
    def _reset_parameters(self):
        bias = bias_init_with_prob(0.01)   
        init.constant_(self.enc_score_head.bias, bias)
        init.constant_(self.enc_bbox_head.layers[-1].weight, 0)  
        init.constant_(self.enc_bbox_head.layers[-1].bias, 0)
        for cls_head, reg_head in zip(self.dec_score_head, self.dec_bbox_head):
            init.constant_(cls_head.bias, bias)     
            init.constant_(reg_head.layers[-1].weight, 0)
            init.constant_(reg_head.layers[-1].bias, 0)
        init.xavier_uniform_(self.enc_output[0].weight)
        if self.learn_query_content:   
            init.xavier_uniform_(self.tgt_embed.weight)
        init.xavier_uniform_(self.query_pos_head.layers[0].weight)
        init.xavier_uniform_(self.query_pos_head.layers[1].weight)
        for module in self.input_proj:
            init.xavier_uniform_(module[0].weight)
     
    def _build_input_proj_layer(self, feat_channels):     
        self.input_proj = nn.ModuleList()
        for in_channels in feat_channels:
            self.input_proj.append(
                nn.Sequential(
                    OrderedDict(  
                        [     
                            ("conv", nn.Conv2d(in_channels, self.hidden_dim, 1, bias=False)),     
                            ("norm", nn.BatchNorm2d(self.hidden_dim)),
                        ]  
                    )  
                ) 
            )
   
        in_channels = feat_channels[-1]
        for _ in range(self.num_levels - len(feat_channels)):
            self.input_proj.append(   
                nn.Sequential( 
                    OrderedDict(  
                        [    
                            ("conv", nn.Conv2d(in_channels, self.hidden_dim, 3, 2, padding=1, bias=False)),
                            ("norm", nn.BatchNorm2d(self.hidden_dim)),
                        ] 
                    )
                )
            )
            in_channels = self.hidden_dim
 
    def _get_encoder_input(self, feats: List[torch.Tensor]):
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]    
        if self.num_levels > len(proj_feats): 
            len_srcs = len(proj_feats)
            for i in range(len_srcs, self.num_levels):
                proj_feats.append(self.input_proj[i](feats[-1] if i == len_srcs else proj_feats[-1]))

        feat_flatten = []
        spatial_shapes = []
        for feat in proj_feats:
            _, _, h, w = feat.shape
            feat_flatten.append(feat.flatten(2).permute(0, 2, 1))
            spatial_shapes.append([h, w])  
        return torch.concat(feat_flatten, 1), spatial_shapes
    
    def _generate_anchors(self, spatial_shapes=None, grid_size=0.05, dtype=torch.float32, device="cpu"):     
        if spatial_shapes is None:
            spatial_shapes = []
            eval_h, eval_w = self.eval_spatial_size    
            for stride in self.feat_strides:     
                spatial_shapes.append([int(eval_h / stride), int(eval_w / stride)])

        anchors = []     
        for lvl, (h, w) in enumerate(spatial_shapes):  
            grid_y, grid_x = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij")
            grid_xy = torch.stack([grid_x, grid_y], dim=-1).to(dtype)   
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / torch.tensor([w, h], dtype=dtype, device=device)
            wh = torch.ones_like(grid_xy) * grid_size * (2.0**lvl)   
            angle = torch.full_like(grid_xy[..., :1], 0.5)
            anchors.append(torch.concat([grid_xy, wh, angle], dim=-1).reshape(-1, h * w, 5))   
     
        anchors = torch.concat(anchors, dim=1).to(device)    
        valid_mask = ((anchors > self.eps) * (anchors < 1 - self.eps)).all(-1, keepdim=True)
        anchors = torch.log(anchors / (1 - anchors))
        anchors = torch.where(valid_mask, anchors, torch.inf)  
        return anchors, valid_mask
 
    def _get_decoder_input(self, memory, spatial_shapes, denoising_logits=None, denoising_bbox_unact=None): 
        if self.training or self.eval_spatial_size is None:
            anchors, valid_mask = self._generate_anchors(spatial_shapes, device=memory.device)     
        else:     
            anchors = self.anchors    
            valid_mask = self.valid_mask
 
        memory = valid_mask.to(memory.dtype) * memory    
        output_memory = self.enc_output(memory)
        enc_outputs_logits = self.enc_score_head(output_memory) 
        enc_outputs_coord_unact = self.enc_bbox_head(output_memory) + anchors    
        enc_topk_memory, enc_topk_logits, enc_topk_bbox_unact = self._select_topk(   
            output_memory,
            enc_outputs_logits,     
            enc_outputs_coord_unact, 
            self.num_queries,     
        )

        enc_topk_bboxes_list, enc_topk_logits_list = [], []     
        if self.training:
            enc_topk_bboxes_list.append(F.sigmoid(enc_topk_bbox_unact))     
            enc_topk_logits_list.append(enc_topk_logits) 

        content = self.tgt_embed.weight.unsqueeze(0).tile([memory.shape[0], 1, 1]) if self.learn_query_content else enc_topk_memory.detach()     
        enc_topk_bbox_unact = enc_topk_bbox_unact.detach()
        if denoising_bbox_unact is not None:   
            enc_topk_bbox_unact = torch.concat([denoising_bbox_unact, enc_topk_bbox_unact], dim=1)
            content = torch.concat([denoising_logits, content], dim=1)

        return content, enc_topk_bbox_unact, enc_topk_bboxes_list, enc_topk_logits_list, enc_outputs_logits

    def _select_topk(self, memory, outputs_logits, outputs_coords_unact, topk):    
        if self.query_select_method == "default":
            _, topk_ind = torch.topk(outputs_logits.max(-1).values, topk, dim=-1)
        elif self.query_select_method == "one2many":    
            _, topk_ind = torch.topk(outputs_logits.flatten(1), topk, dim=-1)
            topk_ind = topk_ind // outputs_logits.shape[-1]
        elif self.query_select_method == "agnostic": 
            _, topk_ind = torch.topk(outputs_logits.squeeze(-1), topk, dim=-1)
 
        topk_coords = outputs_coords_unact.gather(dim=1, index=topk_ind.unsqueeze(-1).repeat(1, 1, outputs_coords_unact.shape[-1]))
        topk_logits = outputs_logits.gather(dim=1, index=topk_ind.unsqueeze(-1).repeat(1, 1, outputs_logits.shape[-1]))   
        topk_memory = memory.gather(dim=1, index=topk_ind.unsqueeze(-1).repeat(1, 1, memory.shape[-1]))   
        return topk_memory, topk_logits, topk_coords

    def _forward_rtdetrv3(self, memory, spatial_shapes, targets): 
        anchors, valid_mask = self._generate_anchors(spatial_shapes, device=memory.device)
        if memory.shape[0] > 1 and anchors.shape[0] == 1: anchors = anchors.repeat(memory.shape[0], 1, 1)   
        memory = valid_mask.to(memory.dtype) * memory; main = self.enc_output(memory); mapped = self.map_memory(memory.detach())    
        groups, masks = [], [] 
        sizes = [self.num_queries] + (self.mgsa_num_queries or [self.num_queries] * self.mgsa_groups) 
        for index, query_size in enumerate(sizes):
            dn_count = self.num_denoising if index == 0 else (self.mgsa_num_denoising or self.num_denoising)   
            if dn_count > 0:
                dn_logits, dn_boxes, dn_mask, dn_meta = get_contrastive_denoising_training_group(targets, self.num_classes, query_size, self.denoising_class_embed, num_denoising=dn_count, label_noise_ratio=self.label_noise_ratio, box_noise_scale=self.box_noise_scale, noise_mode=self.denoising_type)   
            else: dn_logits = dn_boxes = dn_mask = dn_meta = None
            proposal = main if index == 0 else self.extra_enc_output[index - 1](mapped); score = self.enc_score_head if index == 0 else self.extra_enc_score_head[index - 1]; bbox = self.enc_bbox_head if index == 0 else self.extra_enc_bbox_head[index - 1]   
            logits = score(proposal); top_memory, top_logits, top_anchors = self._select_topk(proposal, logits, bbox(proposal) + anchors, query_size); top_boxes = top_anchors
            content = self.tgt_embed.weight.unsqueeze(0).tile([memory.shape[0], 1, 1]) if self.learn_query_content and index == 0 else top_memory.detach(); ref = top_boxes.detach()    
            if dn_boxes is not None: ref = torch.cat([dn_boxes, ref], 1); content = torch.cat([dn_logits, content], 1)  
            groups.append((content, ref, [top_logits], [torch.sigmoid(top_boxes)], dn_meta, query_size)); local = torch.zeros((ref.shape[1], ref.shape[1]), dtype=torch.bool, device=memory.device) if dn_mask is None else dn_mask
            if index:  
                dn_size = int(dn_meta['dn_num_split'][0]) if dn_meta else 0; local = local.clone(); local[dn_size:, dn_size:] = make_mgsa_query_mask(query_size, self.mgsa_mask_ratio, device=memory.device) 
            masks.append(local)
        if self.o2m_t:
            extra = self.mgsa_groups; proposal = self.extra_enc_output[extra](mapped); logits = self.extra_enc_score_head[extra](proposal); top_memory, top_logits, top_anchors = self._select_topk(proposal, logits, self.extra_enc_bbox_head[extra](proposal) + anchors, self.num_queries_o2m)
            groups.append((top_memory.detach(), top_anchors.detach(), [top_logits], [torch.sigmoid(top_anchors)], None, self.num_queries_o2m)); masks.append(torch.zeros((self.num_queries_o2m, self.num_queries_o2m), dtype=torch.bool, device=memory.device))
        content = torch.cat([g[0] for g in groups], 1); refs = torch.cat([g[1] for g in groups], 1)
        ob, ol = self.decoder(content, refs, memory, spatial_shapes, self.dec_bbox_head, self.dec_score_head, self.query_pos_head, attn_mask=build_group_attn_mask(masks, device=memory.device))   
        bundles, offset = [], 0
        for _, _, enc_logits, enc_boxes, dn_meta, query_size in groups:
            dn_size = int(dn_meta['dn_num_split'][0]) if dn_meta else 0; total = dn_size + query_size; gl, gb = ol[:, :, offset:offset + total], ob[:, :, offset:offset + total]; offset += total
            bundle = {'pred_logits': gl[-1, :, dn_size:], 'pred_boxes': gb[-1, :, dn_size:]}     
            if self.aux_loss: 
                bundle['aux_outputs'] = self._set_aux_loss(gl[:-1, :, dn_size:], gb[:-1, :, dn_size:]); bundle['enc_aux_outputs'] = self._set_aux_loss(enc_logits, enc_boxes); bundle['enc_meta'] = {'class_agnostic': self.query_select_method == 'agnostic'} 
                if dn_meta: bundle['dn_outputs'] = self._set_aux_loss(gl[:, :, :dn_size], gb[:, :, :dn_size]); bundle['dn_meta'] = dn_meta
            bundles.append(bundle) 
        out = bundles[0]
        if self.mgsa_groups: out['mgsa_outputs'] = bundles[1:1 + self.mgsa_groups]
        if self.o2m_t: out['o2m_t_outputs'] = {**bundles[-1], 'o2m_repeat': self.o2m_repeat}     
        return out     

    def convert_to_deploy(self):   
        self.mgsa_groups = 0
        self.mgsa_num_queries = [] 
        self.o2m_t = False
        self.map_memory = None
        self.extra_enc_output = nn.ModuleList()   
        self.extra_enc_score_head = nn.ModuleList()
        self.extra_enc_bbox_head = nn.ModuleList()

    def forward(self, feats, targets=None):  
        memory, spatial_shapes = self._get_encoder_input(feats)
        if self.training and (self.mgsa_groups or self.o2m_t):
            return self._forward_rtdetrv3(memory, spatial_shapes, targets)
        if self.training and self.num_denoising > 0:
            denoising_logits, denoising_bbox_unact, attn_mask, dn_meta = get_contrastive_denoising_training_group(  
                targets,   
                self.num_classes,   
                self.num_queries,
                self.denoising_class_embed, 
                num_denoising=self.num_denoising,
                label_noise_ratio=self.label_noise_ratio,  
                box_noise_scale=self.box_noise_scale,     
                noise_mode=self.denoising_type
            )     
        else:   
            denoising_logits, denoising_bbox_unact, attn_mask, dn_meta = None, None, None, None

        init_ref_contents, init_ref_points_unact, enc_topk_bboxes_list, enc_topk_logits_list, enc_outputs_logits = self._get_decoder_input(
            memory,  
            spatial_shapes,
            denoising_logits,
            denoising_bbox_unact,
        )
        out_bboxes, out_logits = self.decoder(     
            init_ref_contents,
            init_ref_points_unact,    
            memory,   
            spatial_shapes,
            self.dec_bbox_head,
            self.dec_score_head,   
            self.query_pos_head,
            attn_mask=attn_mask,   
        ) 
  
        if self.training and dn_meta is not None:  
            dn_out_bboxes, out_bboxes = torch.split(out_bboxes, dn_meta["dn_num_split"], dim=2)   
            dn_out_logits, out_logits = torch.split(out_logits, dn_meta["dn_num_split"], dim=2)   
     
        out = {"pred_logits": out_logits[-1], "pred_boxes": out_bboxes[-1]}
        if self.training and self.aux_loss:   
            out["aux_outputs"] = self._set_aux_loss(out_logits[:-1], out_bboxes[:-1])
            out["enc_aux_outputs"] = self._set_aux_loss(enc_topk_logits_list, enc_topk_bboxes_list)   
            out["enc_meta"] = {"class_agnostic": self.query_select_method == "agnostic"}
            if dn_meta is not None:
                out["dn_outputs"] = self._set_aux_loss(dn_out_logits, dn_out_bboxes)     
                out["dn_meta"] = dn_meta   
            out["enc_outputs_logits"] = enc_outputs_logits
        return out    

    @torch.jit.unused   
    def _set_aux_loss(self, outputs_class, outputs_coord):
        return [{"pred_logits": a, "pred_boxes": b} for a, b in zip(outputs_class, outputs_coord)]  

     
@register()
class OBBDFINETransformer(nn.Module): 
    __share__ = ["num_classes", "eval_spatial_size"]
  
    def __init__(
        self,
        num_classes=80,
        hidden_dim=256, 
        num_queries=300,   
        feat_channels=[512, 1024, 2048],
        feat_strides=[8, 16, 32],   
        num_levels=3,     
        num_points=4,   
        nhead=8,
        num_layers=6,
        dim_feedforward=1024,  
        dropout=0.0,   
        activation="relu",
        num_denoising=100,     
        label_noise_ratio=0.5,
        box_noise_scale=1.0,     
        learn_query_content=False,
        eval_spatial_size=None,   
        eval_idx=-1,   
        eps=1e-2,
        aux_loss=True,  
        cross_attn_method="default",
        query_select_method="default",
        reg_max=32,
        reg_scale=4.0,
        layer_scale=1,
        mlp_act="relu",
        angle_factor=math.pi,
        denoising_type="xyxya",
        mgsa_groups=0, mgsa_num_queries=None, mgsa_num_denoising=None, mgsa_mask_ratio=0.1,
        o2m_t=False, num_queries_o2m=450, o2m_repeat=4,
    ): 
        super().__init__()
        feat_channels = list(feat_channels)  
        feat_strides = list(feat_strides)
        if len(feat_channels) > num_levels:  
            raise ValueError("feat_channels length must be <= num_levels")
        if len(feat_strides) != len(feat_channels):  
            raise ValueError("feat_strides length must match feat_channels")
    
        for _ in range(num_levels - len(feat_strides)):   
            feat_strides.append(feat_strides[-1] * 2)
 
        self.hidden_dim = hidden_dim 
        scaled_dim = round(layer_scale * hidden_dim)   
        self.nhead = nhead
        self.feat_strides = feat_strides    
        self.num_levels = num_levels
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.eps = eps
        self.num_layers = num_layers 
        self.eval_spatial_size = eval_spatial_size
        self.aux_loss = aux_loss
        self.reg_max = reg_max
        self.angle_factor = angle_factor
        self.denoising_type = denoising_type    
        if mgsa_groups < 0 or not 0 <= mgsa_mask_ratio < 1 or o2m_repeat < 1 or num_queries_o2m <= 0:   
            raise ValueError('invalid RT-DETRv3 supervision parameters')    
        if mgsa_num_queries and (len(mgsa_num_queries) != mgsa_groups or any(q <= 0 for q in mgsa_num_queries)):     
            raise ValueError('mgsa_num_queries must match mgsa_groups and contain positive values') 
        self.mgsa_groups, self.mgsa_num_queries = int(mgsa_groups), list(mgsa_num_queries or [])
        self.mgsa_num_denoising, self.mgsa_mask_ratio = mgsa_num_denoising, mgsa_mask_ratio
        self.o2m_t, self.num_queries_o2m, self.o2m_repeat = bool(o2m_t), int(num_queries_o2m), int(o2m_repeat) 
   
        if query_select_method not in ("default", "one2many", "agnostic"):
            raise ValueError(f"Unsupported query_select_method {query_select_method!r}") 
        if cross_attn_method not in ("default", "discrete"):
            raise ValueError(f"Unsupported cross_attn_method {cross_attn_method!r}")
        self.cross_attn_method = cross_attn_method     
        self.query_select_method = query_select_method    

        self._build_input_proj_layer(feat_channels)     

        self.up = nn.Parameter(torch.tensor([0.5]), requires_grad=False)
        self.reg_scale = nn.Parameter(torch.tensor([reg_scale]), requires_grad=False)

        decoder_layer = OBBDFINETransformerDecoderLayer( 
            hidden_dim,
            nhead,  
            dim_feedforward, 
            dropout,
            activation,
            num_levels,
            num_points,
            cross_attn_method=cross_attn_method,
        )    
        decoder_layer_wide = OBBDFINETransformerDecoderLayer(
            hidden_dim,    
            nhead,
            dim_feedforward,
            dropout,
            activation, 
            num_levels,
            num_points,
            cross_attn_method=cross_attn_method,
            layer_scale=layer_scale,
        )
        self.decoder = OBBDFINETransformerDecoder(  
            hidden_dim, 
            decoder_layer,
            decoder_layer_wide,
            num_layers,
            nhead,
            reg_max,  
            self.reg_scale,    
            self.up,
            eval_idx,  
            layer_scale,    
            act=activation, 
        )     
    
        self.num_denoising = num_denoising
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale
        if num_denoising > 0:
            self.denoising_class_embed = nn.Embedding(num_classes + 1, hidden_dim, padding_idx=num_classes)
            init.normal_(self.denoising_class_embed.weight[:-1])
   
        self.learn_query_content = learn_query_content   
        if learn_query_content:
            self.tgt_embed = nn.Embedding(num_queries, hidden_dim) 

        self.query_pos_head = MLP(5, 2 * hidden_dim, hidden_dim, 2, act=mlp_act)     
        self.enc_output = nn.Sequential(
            OrderedDict(     
                [ 
                    ("proj", nn.Linear(hidden_dim, hidden_dim)),     
                    ("norm", nn.LayerNorm(hidden_dim)),
                ]
            )  
        )
        self.enc_score_head = nn.Linear(hidden_dim, 1 if query_select_method == "agnostic" else num_classes)
        self.enc_bbox_head = MLP(hidden_dim, hidden_dim, 5, 3, act=mlp_act)
        extra_count = self.mgsa_groups + int(self.o2m_t) 
        self.map_memory = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim)) if extra_count else None   
        if extra_count:
            self.extra_enc_output = nn.ModuleList([copy.deepcopy(self.enc_output) for _ in range(extra_count)])
            score_dim = 1 if query_select_method == 'agnostic' else num_classes 
            self.extra_enc_score_head = nn.ModuleList([nn.Linear(hidden_dim, score_dim) for _ in range(extra_count)])
            self.extra_enc_bbox_head = nn.ModuleList([MLP(hidden_dim, hidden_dim, 5, 3, act=mlp_act) for _ in range(extra_count)])
        else:
            self.extra_enc_output = nn.ModuleList()
            self.extra_enc_score_head = nn.ModuleList()
            self.extra_enc_bbox_head = nn.ModuleList()

        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx   
        self.dec_score_head = nn.ModuleList(    
            [nn.Linear(hidden_dim, num_classes) for _ in range(self.eval_idx + 1)]
            + [nn.Linear(scaled_dim, num_classes) for _ in range(num_layers - self.eval_idx - 1)] 
        )
        self.pre_bbox_head = MLP(hidden_dim, hidden_dim, 5, 3, act=mlp_act)
        self.dec_bbox_head = nn.ModuleList(
            [MLP(hidden_dim, hidden_dim, 4 * (self.reg_max + 1), 3, act=mlp_act) for _ in range(self.eval_idx + 1)]
            + [
                MLP(scaled_dim, scaled_dim, 4 * (self.reg_max + 1), 3, act=mlp_act) 
                for _ in range(num_layers - self.eval_idx - 1) 
            ]
        )    
        self.dec_angle_head = nn.ModuleList(
            [MLP(hidden_dim, hidden_dim, 1, 3, act=mlp_act) for _ in range(self.eval_idx + 1)]     
            + [MLP(scaled_dim, scaled_dim, 1, 3, act=mlp_act) for _ in range(num_layers - self.eval_idx - 1)]  
        )    
        self.integral = Integral(self.reg_max)
   
        if self.eval_spatial_size:
            anchors, valid_mask = self._generate_anchors()
            self.register_buffer("anchors", anchors)     
            self.register_buffer("valid_mask", valid_mask) 
   
        self._reset_parameters(feat_channels)
        for output, score, bbox in zip(self.extra_enc_output, self.extra_enc_score_head, self.extra_enc_bbox_head):
            init.xavier_uniform_(output[0].weight); init.constant_(score.bias, bias_init_with_prob(0.01)); init.constant_(bbox.layers[-1].weight, 0); init.constant_(bbox.layers[-1].bias, 0)

    def convert_to_deploy(self):     
        self.mgsa_groups = 0
        self.mgsa_num_queries = []
        self.o2m_t = False    
        self.map_memory = None
        self.extra_enc_output = nn.ModuleList() 
        self.extra_enc_score_head = nn.ModuleList()    
        self.extra_enc_bbox_head = nn.ModuleList()
        self.dec_score_head = nn.ModuleList([nn.Identity()] * self.eval_idx + [self.dec_score_head[self.eval_idx]])
        self.dec_bbox_head = nn.ModuleList(  
            [self.dec_bbox_head[i] if i <= self.eval_idx else nn.Identity() for i in range(len(self.dec_bbox_head))]
        )
        self.dec_angle_head = nn.ModuleList(  
            [self.dec_angle_head[i] if i <= self.eval_idx else nn.Identity() for i in range(len(self.dec_angle_head))]
        ) 

    def _reset_parameters(self, feat_channels):   
        bias = bias_init_with_prob(0.01)    
        init.constant_(self.enc_score_head.bias, bias)
        init.constant_(self.enc_bbox_head.layers[-1].weight, 0)  
        init.constant_(self.enc_bbox_head.layers[-1].bias, 0)   
        init.constant_(self.pre_bbox_head.layers[-1].weight, 0)
        init.constant_(self.pre_bbox_head.layers[-1].bias, 0)
     
        for cls_head, reg_head in zip(self.dec_score_head, self.dec_bbox_head):
            init.constant_(cls_head.bias, bias)
            if hasattr(reg_head, "layers"):  
                init.constant_(reg_head.layers[-1].weight, 0)
                init.constant_(reg_head.layers[-1].bias, 0)
        for angle_head in self.dec_angle_head:   
            init.constant_(angle_head.layers[-1].weight, 0)     
            init.constant_(angle_head.layers[-1].bias, 0)  
    
        init.xavier_uniform_(self.enc_output[0].weight)
        if self.learn_query_content:
            init.xavier_uniform_(self.tgt_embed.weight)
        init.xavier_uniform_(self.query_pos_head.layers[0].weight)
        init.xavier_uniform_(self.query_pos_head.layers[1].weight)     
        for module, in_channels in zip(self.input_proj, feat_channels):
            if in_channels != self.hidden_dim and isinstance(module, nn.Sequential):     
                init.xavier_uniform_(module[0].weight) 

    def _build_input_proj_layer(self, feat_channels):    
        self.input_proj = nn.ModuleList()
        for in_channels in feat_channels:
            if in_channels == self.hidden_dim:
                self.input_proj.append(nn.Identity())
            else:     
                self.input_proj.append(
                    nn.Sequential(  
                        OrderedDict(
                            [
                                ("conv", nn.Conv2d(in_channels, self.hidden_dim, 1, bias=False)), 
                                ("norm", nn.BatchNorm2d(self.hidden_dim)),
                            ]
                        )   
                    )  
                )    
   
        in_channels = feat_channels[-1]    
        for _ in range(self.num_levels - len(feat_channels)):     
            if in_channels == self.hidden_dim:   
                self.input_proj.append(nn.Identity())    
            else:     
                self.input_proj.append(
                    nn.Sequential(
                        OrderedDict(
                            [ 
                                ("conv", nn.Conv2d(in_channels, self.hidden_dim, 3, 2, padding=1, bias=False)), 
                                ("norm", nn.BatchNorm2d(self.hidden_dim)),   
                            ]  
                        )  
                    )
                )  
                in_channels = self.hidden_dim
 
    def _get_encoder_input(self, feats: List[torch.Tensor]):  
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]   
        if self.num_levels > len(proj_feats):     
            len_srcs = len(proj_feats)
            for i in range(len_srcs, self.num_levels):  
                proj_feats.append(self.input_proj[i](feats[-1] if i == len_srcs else proj_feats[-1]))

        feat_flatten = []
        spatial_shapes = []     
        for feat in proj_feats:
            _, _, h, w = feat.shape
            feat_flatten.append(feat.flatten(2).permute(0, 2, 1))
            spatial_shapes.append([h, w])
        return torch.concat(feat_flatten, 1), spatial_shapes

    def _generate_anchors(self, spatial_shapes=None, grid_size=0.05, dtype=torch.float32, device="cpu"):
        if spatial_shapes is None:  
            spatial_shapes = []
            eval_h, eval_w = self.eval_spatial_size  
            for stride in self.feat_strides:
                spatial_shapes.append([int(eval_h / stride), int(eval_w / stride)])     
     
        anchors = []   
        for lvl, (h, w) in enumerate(spatial_shapes):
            grid_y, grid_x = torch.meshgrid(    
                torch.arange(h, device=device),    
                torch.arange(w, device=device),     
                indexing="ij",    
            )
            grid_xy = torch.stack([grid_x, grid_y], dim=-1).to(dtype)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / torch.tensor([w, h], dtype=dtype, device=device)
            wh = torch.ones_like(grid_xy) * grid_size * (2.0**lvl)
            angle = torch.full_like(grid_xy[..., :1], 0.5)
            anchors.append(torch.concat([grid_xy, wh, angle], dim=-1).reshape(-1, h * w, 5))   
    
        anchors = torch.concat(anchors, dim=1).to(device)  
        valid_mask = ((anchors > self.eps) * (anchors < 1 - self.eps)).all(-1, keepdim=True)  
        anchors = torch.log(anchors / (1 - anchors)) 
        anchors = torch.where(valid_mask, anchors, torch.inf)  
        return anchors, valid_mask   
 
    def _get_decoder_input(self, memory, spatial_shapes, denoising_logits=None, denoising_bbox_unact=None):
        if self.training or self.eval_spatial_size is None:
            anchors, valid_mask = self._generate_anchors(spatial_shapes, device=memory.device)
        else:
            anchors = self.anchors
            valid_mask = self.valid_mask
        if memory.shape[0] > 1:    
            anchors = anchors.repeat(memory.shape[0], 1, 1)    

        memory = valid_mask.to(memory.dtype) * memory     
        output_memory = self.enc_output(memory) 
        enc_outputs_logits = self.enc_score_head(output_memory)   
        enc_topk_memory, enc_topk_logits, enc_topk_anchors = self._select_topk(
            output_memory,
            enc_outputs_logits,
            anchors,
            self.num_queries,    
        )
        enc_topk_bbox_unact = self.enc_bbox_head(enc_topk_memory) + enc_topk_anchors 
    
        enc_topk_bboxes_list, enc_topk_logits_list = [], []
        if self.training:   
            enc_topk_bboxes_list.append(F.sigmoid(enc_topk_bbox_unact))
            enc_topk_logits_list.append(enc_topk_logits)

        content = self.tgt_embed.weight.unsqueeze(0).tile([memory.shape[0], 1, 1]) if self.learn_query_content else enc_topk_memory.detach()     
        enc_topk_bbox_unact = enc_topk_bbox_unact.detach()
        if denoising_bbox_unact is not None:
            enc_topk_bbox_unact = torch.concat([denoising_bbox_unact, enc_topk_bbox_unact], dim=1)
            content = torch.concat([denoising_logits, content], dim=1)     
   
        return content, enc_topk_bbox_unact, enc_topk_bboxes_list, enc_topk_logits_list, enc_outputs_logits
  
    def _select_topk(self, memory, outputs_logits, outputs_anchors_unact, topk):
        if self.query_select_method == "default":  
            _, topk_ind = torch.topk(outputs_logits.max(-1).values, topk, dim=-1)
        elif self.query_select_method == "one2many":
            _, topk_ind = torch.topk(outputs_logits.flatten(1), topk, dim=-1)
            topk_ind = topk_ind // outputs_logits.shape[-1]
        elif self.query_select_method == "agnostic":
            _, topk_ind = torch.topk(outputs_logits.squeeze(-1), topk, dim=-1)   
 
        topk_anchors = outputs_anchors_unact.gather(
            dim=1,
            index=topk_ind.unsqueeze(-1).repeat(1, 1, outputs_anchors_unact.shape[-1]),
        )
        topk_logits = outputs_logits.gather(   
            dim=1,
            index=topk_ind.unsqueeze(-1).repeat(1, 1, outputs_logits.shape[-1]),
        ) if self.training else None    
        topk_memory = memory.gather(dim=1, index=topk_ind.unsqueeze(-1).repeat(1, 1, memory.shape[-1]))
        return topk_memory, topk_logits, topk_anchors 
 
    def _forward_rtdetrv3(self, memory, spatial_shapes, targets):
        anchors, valid_mask = self._generate_anchors(spatial_shapes, device=memory.device)
        if memory.shape[0] > 1 and anchors.shape[0] == 1: anchors = anchors.repeat(memory.shape[0], 1, 1)
        memory = valid_mask.to(memory.dtype) * memory; main = self.enc_output(memory); mapped = self.map_memory(memory.detach())
        groups, masks = [], []
        sizes = [self.num_queries] + (self.mgsa_num_queries or [self.num_queries] * self.mgsa_groups)  
        for index, query_size in enumerate(sizes):
            dn_count = self.num_denoising if index == 0 else (self.mgsa_num_denoising or self.num_denoising)
            if dn_count > 0:    
                dn_logits, dn_boxes, dn_mask, dn_meta = get_contrastive_denoising_training_group(targets, self.num_classes, query_size, self.denoising_class_embed, num_denoising=dn_count, label_noise_ratio=self.label_noise_ratio, box_noise_scale=self.box_noise_scale, noise_mode=self.denoising_type)
            else: dn_logits = dn_boxes = dn_mask = dn_meta = None
            if index == 0: proposal, score, bbox = main, self.enc_score_head, self.enc_bbox_head    
            else:
                extra = index - 1; proposal = self.extra_enc_output[extra](mapped); score, bbox = self.extra_enc_score_head[extra], self.extra_enc_bbox_head[extra]
            logits = score(proposal); top_memory, top_logits, top_anchors = self._select_topk(proposal, logits, anchors, query_size); top_boxes = bbox(top_memory) + top_anchors
            content = self.tgt_embed.weight.unsqueeze(0).tile([memory.shape[0], 1, 1]) if self.learn_query_content and index == 0 else top_memory.detach(); ref = top_boxes.detach()     
            if dn_boxes is not None: ref = torch.cat([dn_boxes, ref], 1); content = torch.cat([dn_logits, content], 1)
            groups.append((content, ref, [top_logits], [torch.sigmoid(top_boxes)], dn_meta, query_size))
            local = torch.zeros((ref.shape[1], ref.shape[1]), dtype=torch.bool, device=memory.device) if dn_mask is None else dn_mask
            if index:   
                dn_size = int(dn_meta['dn_num_split'][0]) if dn_meta else 0; local = local.clone(); local[dn_size:, dn_size:] = make_mgsa_query_mask(query_size, self.mgsa_mask_ratio, device=memory.device)    
            masks.append(local)  
        if self.o2m_t: 
            extra = self.mgsa_groups; proposal = self.extra_enc_output[extra](mapped); logits = self.extra_enc_score_head[extra](proposal); top_memory, top_logits, top_anchors = self._select_topk(proposal, logits, anchors, self.num_queries_o2m); top_boxes = self.extra_enc_bbox_head[extra](top_memory) + top_anchors 
            groups.append((top_memory.detach(), top_boxes.detach(), [top_logits], [torch.sigmoid(top_boxes)], None, self.num_queries_o2m)); masks.append(torch.zeros((self.num_queries_o2m, self.num_queries_o2m), dtype=torch.bool, device=memory.device))
        content = torch.cat([g[0] for g in groups], 1); refs = torch.cat([g[1] for g in groups], 1)   
        ob, ol, oc, ors, pb, pl = self.decoder(content, refs, memory, spatial_shapes, self.dec_bbox_head, self.dec_angle_head, self.dec_score_head, self.query_pos_head, self.pre_bbox_head, self.integral, self.up, self.reg_scale, self.angle_factor, attn_mask=build_group_attn_mask(masks, device=memory.device), dn_meta=None)     
        bundles, offset = [], 0
        for _, _, enc_logits, enc_boxes, dn_meta, query_size in groups:  
            dn_size = int(dn_meta['dn_num_split'][0]) if dn_meta else 0; total = dn_size + query_size
            gl, gb, gc, gr = (x[:, :, offset:offset + total] for x in (ol, ob, oc, ors)); ppb, ppl = pb[:, offset:offset + total], pl[:, offset:offset + total]; offset += total
            bundle = {'pred_logits': gl[-1, :, dn_size:], 'pred_boxes': gb[-1, :, dn_size:], 'pred_corners': gc[-1, :, dn_size:], 'ref_points': gr[-1, :, dn_size:], 'up': self.up, 'reg_scale': self.reg_scale}  
            if self.aux_loss:     
                bundle['aux_outputs'] = self._set_aux_loss2(gl[:-1, :, dn_size:], gb[:-1, :, dn_size:], gc[:-1, :, dn_size:], gr[:-1, :, dn_size:], gc[-1, :, dn_size:], gl[-1, :, dn_size:], gb[-1, :, dn_size:]); bundle['enc_aux_outputs'] = self._set_aux_loss(enc_logits, enc_boxes); bundle['enc_meta'] = {'class_agnostic': self.query_select_method == 'agnostic'}   
                if dn_meta:
                    bundle['dn_outputs'] = self._set_aux_loss2(gl[:, :, :dn_size], gb[:, :, :dn_size], gc[:, :, :dn_size], gr[:, :, :dn_size], gc[-1, :, :dn_size], gl[-1, :, :dn_size], gb[-1, :, :dn_size]); bundle['dn_meta'] = dn_meta; bundle['dn_pre_outputs'] = {'pred_logits': ppl[:, :dn_size], 'pred_boxes': ppb[:, :dn_size]}    
                bundle['pre_outputs'] = {'pred_logits': ppl[:, dn_size:], 'pred_boxes': ppb[:, dn_size:]}
            bundles.append(bundle)    
        out = bundles[0]    
        if self.mgsa_groups: out['mgsa_outputs'] = bundles[1:1 + self.mgsa_groups]
        if self.o2m_t: out['o2m_t_outputs'] = {k: v for k, v in bundles[-1].items() if k not in ('pre_outputs', 'dn_outputs', 'dn_pre_outputs', 'dn_meta')} | {'o2m_repeat': self.o2m_repeat}
        return out
  
    def forward(self, feats, targets=None):
        memory, spatial_shapes = self._get_encoder_input(feats) 
        if self.training and (self.mgsa_groups or self.o2m_t):
            return self._forward_rtdetrv3(memory, spatial_shapes, targets)
        if self.training and self.num_denoising > 0:   
            denoising_logits, denoising_bbox_unact, attn_mask, dn_meta = get_contrastive_denoising_training_group(
                targets,    
                self.num_classes,
                self.num_queries,  
                self.denoising_class_embed,
                num_denoising=self.num_denoising,
                label_noise_ratio=self.label_noise_ratio,     
                box_noise_scale=self.box_noise_scale,
                noise_mode=self.denoising_type,
            )
        else:
            denoising_logits, denoising_bbox_unact, attn_mask, dn_meta = None, None, None, None
 
        init_ref_contents, init_ref_points_unact, enc_topk_bboxes_list, enc_topk_logits_list, enc_outputs_logits = (
            self._get_decoder_input(memory, spatial_shapes, denoising_logits, denoising_bbox_unact)
        )
   
        out_bboxes, out_logits, out_corners, out_refs, pre_bboxes, pre_logits = self.decoder(
            init_ref_contents, 
            init_ref_points_unact,    
            memory,
            spatial_shapes,
            self.dec_bbox_head,
            self.dec_angle_head,    
            self.dec_score_head,  
            self.query_pos_head,   
            self.pre_bbox_head,  
            self.integral,
            self.up,
            self.reg_scale,     
            self.angle_factor,  
            attn_mask=attn_mask,
            dn_meta=dn_meta,
        )

        if self.training and dn_meta is not None: 
            dn_pre_logits, pre_logits = torch.split(pre_logits, dn_meta["dn_num_split"], dim=1)  
            dn_pre_bboxes, pre_bboxes = torch.split(pre_bboxes, dn_meta["dn_num_split"], dim=1)    
            dn_out_logits, out_logits = torch.split(out_logits, dn_meta["dn_num_split"], dim=2)   
            dn_out_bboxes, out_bboxes = torch.split(out_bboxes, dn_meta["dn_num_split"], dim=2)
            dn_out_corners, out_corners = torch.split(out_corners, dn_meta["dn_num_split"], dim=2)     
            dn_out_refs, out_refs = torch.split(out_refs, dn_meta["dn_num_split"], dim=2)
 
        if self.training:
            out = {
                "pred_logits": out_logits[-1],
                "pred_boxes": out_bboxes[-1],    
                "pred_corners": out_corners[-1],
                "ref_points": out_refs[-1],     
                "up": self.up, 
                "reg_scale": self.reg_scale,     
            }  
        else:
            out = {"pred_logits": out_logits[-1], "pred_boxes": out_bboxes[-1]}
   
        if self.training and self.aux_loss:   
            out["aux_outputs"] = self._set_aux_loss2(
                out_logits[:-1],
                out_bboxes[:-1],
                out_corners[:-1],
                out_refs[:-1],    
                out_corners[-1],
                out_logits[-1],
                out_bboxes[-1], 
            )    
            out["enc_aux_outputs"] = self._set_aux_loss(enc_topk_logits_list, enc_topk_bboxes_list)
            out["pre_outputs"] = {"pred_logits": pre_logits, "pred_boxes": pre_bboxes}
            out["enc_meta"] = {"class_agnostic": self.query_select_method == "agnostic"}   
   
            if dn_meta is not None:     
                out["dn_outputs"] = self._set_aux_loss2(    
                    dn_out_logits,
                    dn_out_bboxes, 
                    dn_out_corners,     
                    dn_out_refs,     
                    dn_out_corners[-1],
                    dn_out_logits[-1],
                    dn_out_bboxes[-1],
                )
                out["dn_pre_outputs"] = {"pred_logits": dn_pre_logits, "pred_boxes": dn_pre_bboxes}
                out["dn_meta"] = dn_meta

            out["enc_outputs_logits"] = enc_outputs_logits

        return out
    
    @torch.jit.unused   
    def _set_aux_loss(self, outputs_class, outputs_coord):   
        return [{"pred_logits": a, "pred_boxes": b} for a, b in zip(outputs_class, outputs_coord)]
     
    @torch.jit.unused  
    def _set_aux_loss2(     
        self,  
        outputs_class,   
        outputs_coord,  
        outputs_corners,  
        outputs_ref,     
        teacher_corners=None,
        teacher_logits=None,     
        teacher_boxes=None,   
    ):  
        return [
            {    
                "pred_logits": a,
                "pred_boxes": b,     
                "pred_corners": c,  
                "ref_points": d,
                "teacher_corners": teacher_corners,
                "teacher_logits": teacher_logits,  
                "teacher_boxes": teacher_boxes,
            }     
            for a, b, c, d in zip(outputs_class, outputs_coord, outputs_corners, outputs_ref)
        ]   
