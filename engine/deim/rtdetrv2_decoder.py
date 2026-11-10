"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
Modifications Copyright (c) 2024 The DEIM Authors. All Rights Reserved. 
"""

import math 
import copy 
import functools
from collections import OrderedDict

import torch 
import torch.nn as nn 
import torch.nn.functional as F     
import torch.nn.init as init 
from typing import List 

from .denoising import get_contrastive_denoising_training_group     
from .rtdetrv3_supervision import build_group_attn_mask, make_mgsa_query_mask    
from .utils import bias_init_with_prob, get_activation, inverse_sigmoid
from .utils import deformable_attention_core_func_v2

from ..core import register   
from ..misc.flash_attention import FlashSelfAttention
 
__all__ = ['RTDETRTransformerv2']    
  

class MLP(nn.Module):     
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, act='relu'):
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
        method='default',     
        offset_scale=0.5,    
        value_shape='default',    
    ):
        """Multi-Scale Deformable Attention 
        """   
        super(MSDeformableAttention, self).__init__()
        self.embed_dim = embed_dim  
        self.num_heads = num_heads
        self.num_levels = num_levels  
        self.offset_scale = offset_scale
   
        if isinstance(num_points, list):    
            assert len(num_points) == num_levels, ''
            num_points_list = num_points
        else:
            num_points_list = [num_points for _ in range(num_levels)]

        self.num_points_list = num_points_list  
        
        num_points_scale = [1/n for n in num_points_list for _ in range(n)]  
        self.register_buffer('num_points_scale', torch.tensor(num_points_scale, dtype=torch.float32))   

        self.total_points = num_heads * sum(num_points_list)  
        self.method = method
   
        self.head_dim = embed_dim // num_heads    
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads" 
    
        self.sampling_offsets = nn.Linear(embed_dim, self.total_points * 2)   
        self.attention_weights = nn.Linear(embed_dim, self.total_points)
        self.value_proj = nn.Linear(embed_dim, embed_dim) 
        self.output_proj = nn.Linear(embed_dim, embed_dim)
     
        self.ms_deformable_attn_core = functools.partial(deformable_attention_core_func_v2, 
                                                    method=self.method, value_shape=value_shape) 
 
        self._reset_parameters()  
    
        if method == 'discrete':    
            for p in self.sampling_offsets.parameters():
                p.requires_grad = False

    def _reset_parameters(self):
        # sampling_offsets
        init.constant_(self.sampling_offsets.weight, 0) 
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = grid_init / grid_init.abs().max(-1, keepdim=True).values 
        grid_init = grid_init.reshape(self.num_heads, 1, 2).tile([1, sum(self.num_points_list), 1])    
        scaling = torch.concat([torch.arange(1, n + 1) for n in self.num_points_list]).reshape(1, -1, 1)  
        grid_init *= scaling    
        self.sampling_offsets.bias.data[...] = grid_init.flatten()

        # attention_weights  
        init.constant_(self.attention_weights.weight, 0)
        init.constant_(self.attention_weights.bias, 0)    

        # proj    
        init.xavier_uniform_(self.value_proj.weight)
        init.constant_(self.value_proj.bias, 0)  
        init.xavier_uniform_(self.output_proj.weight)
        init.constant_(self.output_proj.bias, 0)    

   
    def forward(self,
                query: torch.Tensor,
                reference_points: torch.Tensor,
                value: torch.Tensor,
                value_spatial_shapes: List[int],  
                value_mask: torch.Tensor=None):   
        """
        Args:   
            query (Tensor): [bs, query_length, C]
            reference_points (Tensor): [bs, query_length, n_levels, 2], range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area    
            value (Tensor): [bs, value_length, C]    
            value_spatial_shapes (List): [n_levels, 2], [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
            value_mask (Tensor): [bs, value_length], True for non-padding elements, False for padding elements

        Returns: 
            output (Tensor): [bs, Length_{query}, C]
        """  
        bs, Len_q = query.shape[:2]
        Len_v = value.shape[1]  
    
        value = self.value_proj(value)
        if value_mask is not None:     
            value = value * value_mask.to(value.dtype).unsqueeze(-1)  
  
        value = value.reshape(bs, Len_v, self.num_heads, self.head_dim)
     
        sampling_offsets: torch.Tensor = self.sampling_offsets(query) 
        sampling_offsets = sampling_offsets.reshape(bs, Len_q, self.num_heads, sum(self.num_points_list), 2)
   
        attention_weights = self.attention_weights(query).reshape(bs, Len_q, self.num_heads, sum(self.num_points_list))
        attention_weights = F.softmax(attention_weights, dim=-1).reshape(bs, Len_q, self.num_heads, sum(self.num_points_list))
   
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.tensor(value_spatial_shapes)
            offset_normalizer = offset_normalizer.flip([1]).reshape(1, 1, 1, self.num_levels, 1, 2)
            sampling_locations = reference_points.reshape(bs, Len_q, 1, self.num_levels, 1, 2) + sampling_offsets / offset_normalizer   
        elif reference_points.shape[-1] == 4:
            # reference_points [8, 480, None, 1,  4]     
            # sampling_offsets [8, 480, 8,    12, 2]
            num_points_scale = self.num_points_scale.to(dtype=query.dtype).unsqueeze(-1)
            offset = sampling_offsets * num_points_scale * reference_points[:, :, None, :, 2:] * self.offset_scale
            sampling_locations = reference_points[:, :, None, :, :2] + offset 
        else:   
            raise ValueError(
                "Last dim of reference_points must be 2 or 4, but get {} instead.".
                format(reference_points.shape[-1]))  

        output = self.ms_deformable_attn_core(value, value_spatial_shapes, sampling_locations, attention_weights, self.num_points_list)  
 
        output = self.output_proj(output) 

        return output

     
class TransformerDecoderLayer(nn.Module):
    def __init__(self,  
                 d_model=256,
                 n_head=8,
                 dim_feedforward=1024,    
                 dropout=0.,
                 activation='relu', 
                 n_levels=4,    
                 n_points=4,
                 cross_attn_method='default',
                 value_shape='default',
                 ):
        super(TransformerDecoderLayer, self).__init__()

        # self attention
        self.self_attn = FlashSelfAttention(d_model, n_head, dropout=dropout, batch_first=True) 
        self.dropout1 = nn.Dropout(dropout)   
        self.norm1 = nn.LayerNorm(d_model)

        # cross attention
        self.cross_attn = MSDeformableAttention(d_model, n_head, n_levels, n_points, method=cross_attn_method, value_shape=value_shape)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn   
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

    def with_pos_embed(self, tensor, pos): 
        return tensor if pos is None else tensor + pos
  
    def forward_ffn(self, tgt):
        return self.linear2(self.dropout3(self.activation(self.linear1(tgt))))   

    def forward(self,
                target,
                reference_points,  
                memory, 
                memory_spatial_shapes,     
                attn_mask=None,     
                memory_mask=None,
                query_pos_embed=None):
        # self attention
        q = k = self.with_pos_embed(target, query_pos_embed)
   
        target2, _ = self.self_attn(q, k, value=target, attn_mask=attn_mask, need_weights=False)   
        target = target + self.dropout1(target2)   
        target = self.norm1(target)
 
        # cross attention    
        target2 = self.cross_attn(\
            self.with_pos_embed(target, query_pos_embed),    
            reference_points, 
            memory,    
            memory_spatial_shapes, 
            memory_mask)   
        target = target + self.dropout2(target2)  
        target = self.norm2(target)
 
        # ffn   
        target2 = self.forward_ffn(target)    
        target = target + self.dropout4(target2) 
        target = self.norm3(target)  

        return target


class TransformerDecoder(nn.Module):
    def __init__(self, hidden_dim, decoder_layer, num_layers, eval_idx=-1): 
        super(TransformerDecoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])   
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx   

    def forward(self,   
                target,   
                ref_points_unact,
                memory,    
                memory_spatial_shapes,   
                bbox_head,  
                score_head,    
                query_pos_head,
                attn_mask=None,
                memory_mask=None):
        dec_out_bboxes = []
        dec_out_logits = []    
        ref_points_detach = F.sigmoid(ref_points_unact)   

        output = target    
        for i, layer in enumerate(self.layers):   
            ref_points_input = ref_points_detach.unsqueeze(2)   
            query_pos_embed = query_pos_head(ref_points_detach)     
 
            output = layer(output, ref_points_input, memory, memory_spatial_shapes, attn_mask, memory_mask, query_pos_embed)     

            inter_ref_bbox = F.sigmoid(bbox_head[i](output) + inverse_sigmoid(ref_points_detach))

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

  
@register()
class RTDETRTransformerv2(nn.Module):
    __share__ = ['num_classes', 'eval_spatial_size']
    
    def __init__(self,
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
                 dropout=0.,   
                 activation="relu",  
                 num_denoising=100,
                 label_noise_ratio=0.5,
                 box_noise_scale=1.0,
                 learn_query_content=False,
                 eval_spatial_size=None,
                 eval_idx=-1,
                 eps=1e-2, 
                 aux_loss=True,  
                 cross_attn_method='default', 
                 query_select_method='default',  
                 value_shape='reshape',    
                 mlp_act='relu', 
                 query_pos_method='default',
                 mgsa_groups=0, mgsa_num_queries=None, mgsa_num_denoising=None, mgsa_mask_ratio=0.1, 
                 o2m_t=False, num_queries_o2m=450, o2m_repeat=4, 
                 ):
        super().__init__()     
        assert len(feat_channels) <= num_levels
        assert len(feat_strides) == len(feat_channels) 
        if mgsa_groups < 0 or not 0 <= mgsa_mask_ratio < 1 or o2m_repeat < 1 or num_queries_o2m <= 0: 
            raise ValueError('invalid RT-DETRv3 supervision parameters') 
        if mgsa_num_queries and (len(mgsa_num_queries) != mgsa_groups or any(q <= 0 for q in mgsa_num_queries)):
            raise ValueError('mgsa_num_queries must match mgsa_groups and contain positive values') 
  
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
        self.mgsa_groups, self.mgsa_num_queries = int(mgsa_groups), list(mgsa_num_queries or [])   
        self.mgsa_num_denoising, self.mgsa_mask_ratio = mgsa_num_denoising, mgsa_mask_ratio
        self.o2m_t, self.num_queries_o2m, self.o2m_repeat = bool(o2m_t), int(num_queries_o2m), int(o2m_repeat)     

        assert query_select_method in ('default', 'one2many', 'agnostic'), ''     
        assert cross_attn_method in ('default', 'discrete'), ''    
        self.cross_attn_method = cross_attn_method
        self.query_select_method = query_select_method    

        # backbone feature projection   
        self._build_input_proj_layer(feat_channels)
  
        # Transformer module
        decoder_layer = TransformerDecoderLayer(hidden_dim, nhead, dim_feedforward, dropout, \
            activation, num_levels, num_points, cross_attn_method=cross_attn_method, value_shape=value_shape)  
        self.decoder = TransformerDecoder(hidden_dim, decoder_layer, num_layers, eval_idx)    

        # denoising     
        self.num_denoising = num_denoising  
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale 
        if num_denoising > 0: 
            self.denoising_class_embed = nn.Embedding(num_classes+1, hidden_dim, padding_idx=num_classes)
            init.normal_(self.denoising_class_embed.weight[:-1])
    
        # decoder embedding
        self.learn_query_content = learn_query_content
        if learn_query_content:     
            self.tgt_embed = nn.Embedding(num_queries, hidden_dim)   
   
        if query_pos_method == 'as_reg':
            self.query_pos_head = MLP(4, hidden_dim, hidden_dim, 3, act=mlp_act)     
            print("     ### Query Position Embedding@{} ###     ".format(query_pos_method))
        else:
            self.query_pos_head = MLP(4, 2 * hidden_dim, hidden_dim, 2, act=mlp_act)     

        # if num_select_queries != self.num_queries:     
        #     layer = TransformerEncoderLayer(hidden_dim, nhead, dim_feedforward, activation='gelu')     
        #     self.encoder = TransformerEncoder(layer, 1)    
    
        self.enc_output = nn.Sequential(OrderedDict([
            ('proj', nn.Linear(hidden_dim, hidden_dim)),
            ('norm', nn.LayerNorm(hidden_dim,)),
        ])) 

        if query_select_method == 'agnostic':
            self.enc_score_head = nn.Linear(hidden_dim, 1)
        else:   
            self.enc_score_head = nn.Linear(hidden_dim, num_classes)
 
        self.enc_bbox_head = MLP(hidden_dim, hidden_dim, 4, 3, act=mlp_act)     
        extra_count = self.mgsa_groups + int(self.o2m_t)
        self.map_memory = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim)) if extra_count else None     
        if extra_count:    
            self.extra_enc_output = nn.ModuleList([copy.deepcopy(self.enc_output) for _ in range(extra_count)])  
            score_dim = 1 if query_select_method == 'agnostic' else num_classes     
            self.extra_enc_score_head = nn.ModuleList([nn.Linear(hidden_dim, score_dim) for _ in range(extra_count)])
            self.extra_enc_bbox_head = nn.ModuleList([MLP(hidden_dim, hidden_dim, 4, 3, act=mlp_act) for _ in range(extra_count)])   
        else:
            self.extra_enc_output = nn.ModuleList()
            self.extra_enc_score_head = nn.ModuleList()  
            self.extra_enc_bbox_head = nn.ModuleList() 
  
        # decoder head     
        self.dec_score_head = nn.ModuleList([   
            nn.Linear(hidden_dim, num_classes) for _ in range(num_layers)  
        ])
        self.dec_bbox_head = nn.ModuleList([
            MLP(hidden_dim, hidden_dim, 4, 3, act=mlp_act) for _ in range(num_layers)     
        ])

        # init encoder output anchors and valid_mask
        if self.eval_spatial_size: 
            anchors, valid_mask = self._generate_anchors()
            self.register_buffer('anchors', anchors) 
            self.register_buffer('valid_mask', valid_mask)
    
        self._reset_parameters()
        for output, score, bbox in zip(self.extra_enc_output, self.extra_enc_score_head, self.extra_enc_bbox_head): 
            init.xavier_uniform_(output[0].weight)
            init.constant_(score.bias, bias_init_with_prob(0.01))
            init.constant_(bbox.layers[-1].weight, 0)   
            init.constant_(bbox.layers[-1].bias, 0)
        
    def _reset_parameters(self):    
        bias = bias_init_with_prob(0.01)  
        init.constant_(self.enc_score_head.bias, bias)
        init.constant_(self.enc_bbox_head.layers[-1].weight, 0)    
        init.constant_(self.enc_bbox_head.layers[-1].bias, 0)

        for _cls, _reg in zip(self.dec_score_head, self.dec_bbox_head):
            init.constant_(_cls.bias, bias)
            init.constant_(_reg.layers[-1].weight, 0)   
            init.constant_(_reg.layers[-1].bias, 0)    
        
        init.xavier_uniform_(self.enc_output[0].weight)    
        if self.learn_query_content:   
            init.xavier_uniform_(self.tgt_embed.weight)
        init.xavier_uniform_(self.query_pos_head.layers[0].weight)   
        init.xavier_uniform_(self.query_pos_head.layers[1].weight)    
        for m in self.input_proj:   
            init.xavier_uniform_(m[0].weight)   

    def _build_input_proj_layer(self, feat_channels):
        self.input_proj = nn.ModuleList() 
        for in_channels in feat_channels:     
            self.input_proj.append(    
                nn.Sequential(OrderedDict([ 
                    ('conv', nn.Conv2d(in_channels, self.hidden_dim, 1, bias=False)),    
                    ('norm', nn.BatchNorm2d(self.hidden_dim,))])
                )  
            )

        in_channels = feat_channels[-1]  

        for _ in range(self.num_levels - len(feat_channels)):     
            self.input_proj.append(
                nn.Sequential(OrderedDict([
                    ('conv', nn.Conv2d(in_channels, self.hidden_dim, 3, 2, padding=1, bias=False)),  
                    ('norm', nn.BatchNorm2d(self.hidden_dim))])
                )  
            )
            in_channels = self.hidden_dim

    def _get_encoder_input(self, feats: List[torch.Tensor]):     
        # get projection features    
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)] 
        if self.num_levels > len(proj_feats):
            len_srcs = len(proj_feats)
            for i in range(len_srcs, self.num_levels):
                if i == len_srcs:
                    proj_feats.append(self.input_proj[i](feats[-1]))
                else:
                    proj_feats.append(self.input_proj[i](proj_feats[-1]))

        # get encoder inputs 
        feat_flatten = []
        spatial_shapes = []    
        for i, feat in enumerate(proj_feats):
            _, _, h, w = feat.shape   
            # [b, c, h, w] -> [b, h*w, c]  
            feat_flatten.append(feat.flatten(2).permute(0, 2, 1))     
            # [num_levels, 2]     
            spatial_shapes.append([h, w]) 
        # [b, l, c]
        feat_flatten = torch.concat(feat_flatten, 1)
        return feat_flatten, spatial_shapes

    def _generate_anchors(self,
                          spatial_shapes=None,
                          grid_size=0.05,  
                          dtype=torch.float32,     
                          device='cpu'):   
        if spatial_shapes is None:    
            spatial_shapes = []     
            eval_h, eval_w = self.eval_spatial_size
            for s in self.feat_strides:    
                spatial_shapes.append([int(eval_h / s), int(eval_w / s)])   

        anchors = []
        for lvl, (h, w) in enumerate(spatial_shapes):
            grid_y, grid_x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij') 
            grid_xy = torch.stack([grid_x, grid_y], dim=-1)  
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / torch.tensor([w, h], dtype=dtype)   
            wh = torch.ones_like(grid_xy) * grid_size * (2.0 ** lvl)
            lvl_anchors = torch.concat([grid_xy, wh], dim=-1).reshape(-1, h * w, 4)    
            anchors.append(lvl_anchors)     

        anchors = torch.concat(anchors, dim=1).to(device)
        valid_mask = ((anchors > self.eps) * (anchors < 1 - self.eps)).all(-1, keepdim=True)   
        anchors = torch.log(anchors / (1 - anchors))    
        anchors = torch.where(valid_mask, anchors, torch.inf)   

        return anchors, valid_mask    
  
     
    def _get_decoder_input(self,
                           memory: torch.Tensor, 
                           spatial_shapes, 
                           denoising_logits=None,
                           denoising_bbox_unact=None):  

        # prepare input for decoder
        if self.training or self.eval_spatial_size is None:
            anchors, valid_mask = self._generate_anchors(spatial_shapes, device=memory.device)
        else:     
            anchors = self.anchors
            valid_mask = self.valid_mask
    
        # memory = torch.where(valid_mask, memory, 0)    
        # TODO fix type error for onnx export 
        memory = valid_mask.to(memory.dtype) * memory  

        output_memory :torch.Tensor = self.enc_output(memory) 
        enc_outputs_logits :torch.Tensor = self.enc_score_head(output_memory)     
        enc_outputs_coord_unact :torch.Tensor = self.enc_bbox_head(output_memory) + anchors   

        enc_topk_bboxes_list, enc_topk_logits_list = [], []
        enc_topk_memory, enc_topk_logits, enc_topk_bbox_unact = \
            self._select_topk(output_memory, enc_outputs_logits, enc_outputs_coord_unact, self.num_queries)     
            
        if self.training:     
            enc_topk_bboxes = F.sigmoid(enc_topk_bbox_unact)
            enc_topk_bboxes_list.append(enc_topk_bboxes)
            enc_topk_logits_list.append(enc_topk_logits)   
   
        # if self.num_select_queries != self.num_queries: 
        #     raise NotImplementedError('')

        if self.learn_query_content:     
            content = self.tgt_embed.weight.unsqueeze(0).tile([memory.shape[0], 1, 1])  
        else:  
            content = enc_topk_memory.detach()
     
        enc_topk_bbox_unact = enc_topk_bbox_unact.detach()   
        
        if denoising_bbox_unact is not None:     
            enc_topk_bbox_unact = torch.concat([denoising_bbox_unact, enc_topk_bbox_unact], dim=1)
            content = torch.concat([denoising_logits, content], dim=1)
  
        return content, enc_topk_bbox_unact, enc_topk_bboxes_list, enc_topk_logits_list

    def _select_topk(self, memory: torch.Tensor, outputs_logits: torch.Tensor, outputs_coords_unact: torch.Tensor, topk: int):
        if self.query_select_method == 'default':  
            _, topk_ind = torch.topk(outputs_logits.max(-1).values, topk, dim=-1)     

        elif self.query_select_method == 'one2many':     
            _, topk_ind = torch.topk(outputs_logits.flatten(1), topk, dim=-1)     
            topk_ind = topk_ind // outputs_logits.shape[-1]

        elif self.query_select_method == 'agnostic':
            _, topk_ind = torch.topk(outputs_logits.squeeze(-1), topk, dim=-1)
        
        topk_ind: torch.Tensor     
     
        topk_coords = outputs_coords_unact.gather(dim=1, \
            index=topk_ind.unsqueeze(-1).repeat(1, 1, outputs_coords_unact.shape[-1])) 
        
        topk_logits = outputs_logits.gather(dim=1, \
            index=topk_ind.unsqueeze(-1).repeat(1, 1, outputs_logits.shape[-1]))
   
        topk_memory = memory.gather(dim=1, \
            index=topk_ind.unsqueeze(-1).repeat(1, 1, memory.shape[-1]))
   
        return topk_memory, topk_logits, topk_coords     
    
    def _forward_rtdetrv3(self, memory, spatial_shapes, targets):   
        anchors, valid_mask = self._generate_anchors(spatial_shapes, device=memory.device)
        if memory.shape[0] > 1 and anchors.shape[0] == 1:
            anchors = anchors.repeat(memory.shape[0], 1, 1)    
        memory = valid_mask.to(memory.dtype) * memory
        main_memory = self.enc_output(memory)    
        mapped = self.map_memory(memory.detach())
        sizes = [self.num_queries] + (self.mgsa_num_queries or [self.num_queries] * self.mgsa_groups)    
        groups, masks = [], [] 
        for index, query_size in enumerate(sizes):     
            dn_count = self.num_denoising if index == 0 else (self.mgsa_num_denoising or self.num_denoising)
            if dn_count > 0:
                dn_logits, dn_boxes, dn_mask, dn_meta = get_contrastive_denoising_training_group(  
                    targets, self.num_classes, query_size, self.denoising_class_embed,
                    num_denoising=dn_count, label_noise_ratio=self.label_noise_ratio, box_noise_scale=self.box_noise_scale)  
            else:    
                dn_logits = dn_boxes = dn_mask = dn_meta = None     
            if index == 0:    
                proposal = main_memory; score = self.enc_score_head; bbox = self.enc_bbox_head
            else:
                extra = index - 1; proposal = self.extra_enc_output[extra](mapped); score = self.extra_enc_score_head[extra]; bbox = self.extra_enc_bbox_head[extra]
            logits = score(proposal)  
            top_memory, top_logits, top_anchors = self._select_topk(proposal, logits, anchors, query_size)
            top_boxes = bbox(top_memory) + top_anchors
            content = self.tgt_embed.weight.unsqueeze(0).tile([memory.shape[0], 1, 1]) if self.learn_query_content and index == 0 else top_memory.detach() 
            ref = top_boxes.detach() 
            if dn_boxes is not None:
                ref = torch.cat([dn_boxes, ref], 1); content = torch.cat([dn_logits, content], 1) 
            groups.append((content, ref, [top_logits] if self.training else [], [torch.sigmoid(top_boxes)] if self.training else [], dn_meta, query_size))
            local = torch.zeros((ref.shape[1], ref.shape[1]), dtype=torch.bool, device=memory.device) if dn_mask is None else dn_mask
            if index:
                dn_size = int(dn_meta['dn_num_split'][0]) if dn_meta else 0 
                local = local.clone(); local[dn_size:, dn_size:] = make_mgsa_query_mask(query_size, self.mgsa_mask_ratio, device=memory.device)
            masks.append(local)    
        if self.o2m_t:  
            extra = self.mgsa_groups; proposal = self.extra_enc_output[extra](mapped); score = self.extra_enc_score_head[extra]; bbox = self.extra_enc_bbox_head[extra]     
            logits = score(proposal); top_memory, top_logits, top_anchors = self._select_topk(proposal, logits, anchors, self.num_queries_o2m)
            top_boxes = bbox(top_memory) + top_anchors
            groups.append((top_memory.detach(), top_boxes.detach(), [top_logits], [torch.sigmoid(top_boxes)], None, self.num_queries_o2m))
            masks.append(torch.zeros((self.num_queries_o2m, self.num_queries_o2m), dtype=torch.bool, device=memory.device))    
        content = torch.cat([g[0] for g in groups], 1); refs = torch.cat([g[1] for g in groups], 1)   
        out_bboxes, out_logits = self.decoder(content, refs, memory, spatial_shapes, self.dec_bbox_head, self.dec_score_head, self.query_pos_head,  
                                              attn_mask=build_group_attn_mask(masks, device=memory.device))
        bundles, offset = [], 0  
        for group in groups:   
            content, ref, enc_logits, enc_boxes, dn_meta, query_size = group
            dn_size = int(dn_meta['dn_num_split'][0]) if dn_meta else 0; total = dn_size + query_size   
            gl, gb = out_logits[:, :, offset:offset + total], out_bboxes[:, :, offset:offset + total]; offset += total
            bundle = {'pred_logits': gl[-1, :, dn_size:], 'pred_boxes': gb[-1, :, dn_size:]}
            if self.aux_loss:  
                bundle['aux_outputs'] = self._set_aux_loss(gl[:-1, :, dn_size:], gb[:-1, :, dn_size:])
                bundle['enc_aux_outputs'] = self._set_aux_loss(enc_logits, enc_boxes)   
                bundle['enc_meta'] = {'class_agnostic': self.query_select_method == 'agnostic'}  
                if dn_meta:    
                    bundle['dn_outputs'] = self._set_aux_loss(gl[:, :, :dn_size], gb[:, :, :dn_size]); bundle['dn_meta'] = dn_meta
            bundles.append(bundle)
        out = bundles[0]    
        if self.mgsa_groups: out['mgsa_outputs'] = bundles[1:1 + self.mgsa_groups]
        if self.o2m_t: 
            out['o2m_t_outputs'] = bundles[-1]; out['o2m_t_outputs']['o2m_repeat'] = self.o2m_repeat    
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
        # input projection and embedding 
        memory, spatial_shapes = self._get_encoder_input(feats)   
        if self.training and (self.mgsa_groups or self.o2m_t):  
            return self._forward_rtdetrv3(memory, spatial_shapes, targets) 
        
        # prepare denoising training
        if self.training and self.num_denoising > 0:
            denoising_logits, denoising_bbox_unact, attn_mask, dn_meta = \
                get_contrastive_denoising_training_group(targets, \
                    self.num_classes, 
                    self.num_queries, 
                    self.denoising_class_embed, 
                    num_denoising=self.num_denoising,     
                    label_noise_ratio=self.label_noise_ratio, 
                    box_noise_scale=self.box_noise_scale, )
        else:
            denoising_logits, denoising_bbox_unact, attn_mask, dn_meta = None, None, None, None

        init_ref_contents, init_ref_points_unact, enc_topk_bboxes_list, enc_topk_logits_list = \
            self._get_decoder_input(memory, spatial_shapes, denoising_logits, denoising_bbox_unact)

        # decoder  
        out_bboxes, out_logits = self.decoder(
            init_ref_contents,
            init_ref_points_unact, 
            memory,
            spatial_shapes,
            self.dec_bbox_head,   
            self.dec_score_head,
            self.query_pos_head,
            attn_mask=attn_mask)

        if self.training and dn_meta is not None:    
            dn_out_bboxes, out_bboxes = torch.split(out_bboxes, dn_meta['dn_num_split'], dim=2)
            dn_out_logits, out_logits = torch.split(out_logits, dn_meta['dn_num_split'], dim=2)
   
        out = {'pred_logits': out_logits[-1], 'pred_boxes': out_bboxes[-1]}    
  
        if self.training and self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(out_logits[:-1], out_bboxes[:-1])
            out['enc_aux_outputs'] = self._set_aux_loss(enc_topk_logits_list, enc_topk_bboxes_list)  
            out['enc_meta'] = {'class_agnostic': self.query_select_method == 'agnostic'}  

            if dn_meta is not None:
                out['dn_outputs'] = self._set_aux_loss(dn_out_logits, dn_out_bboxes)
                out['dn_meta'] = dn_meta  
   
        return out  

 
    @torch.jit.unused   
    def _set_aux_loss(self, outputs_class, outputs_coord): 
        # this is a workaround to make torchscript happy, as torchscript    
        # doesn't support dictionary with non-homogeneous values, such     
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}    
                for a, b in zip(outputs_class, outputs_coord)]     
