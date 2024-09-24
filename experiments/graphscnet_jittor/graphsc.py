from typing import Optional, Union

import ipdb
""" import torch
import torch.nn as nn
from torch import Tensor """
import numpy as np
from PCR_Jittor.jittor.modules.transformer.vanilla_transformer import TransformerLayer
from ops import index_select, spatial_consistency
import jittor as jt
from jittor import nn
from jittor.contrib import concat
from jittor import linalg
class FourierEmbedding(nn.Module):
    """Fourier positional embedding.

    Emb(x) = [sin(2^k Pi x), cos(2^k Pi x), sin(2^(k+1) Pi x), cos(2^(k+1) Pi x), ..., sin(2^(k+L-1) Pi x), cos(2^(k+L-1) Pi x)],
    where x is the input tensor.
    """

    def __init__(self, length: int, k0: float = 0.0, use_pi: bool = True, use_input: bool = False) -> None:
        """Initialize a Fourier embedding function.

        Args:
            length (float): the length of the embedding.
            k0 (float): the starting exponential of the embedding. Default: 0.
            use_pi (bool): if True, use pi in the embedding. Default: True.
            use_input (bool): if True, return the input vector in the embedding. Default: False.
        """
        super().__init__()
        self.length = length
        self.k0 = k0
        self.use_pi = use_pi
        self.use_input = use_input

    def execute(self, inputs):
        """Fourier embedding execute.

        Args:
            inputs (Tensor): the input tensor in the shape of (*, N).

        Returns:
            A Tensor of the embeddings in the shape of (*, Lx2xN) or (*, (2L+1)xN) if use_input.
        """
        batch_shape = inputs.shape[:-1]
        num_inputs = inputs.shape[-1]
        inputs = inputs.view(-1, 1, num_inputs)  # (B, 1, N)
        factors = (2.0 ** jt.arange(self.k0, self.k0 + self.length).float()).view(1, -1, 1)  # (1, L, 1)
        if self.use_pi:
            factors = factors * np.pi
        thetas = factors * inputs  # (B, L, N)
        sin_values = jt.sin(thetas)  # (B, L, N)
        cos_values = jt.cos(thetas)  # (B, L, N)
        embeddings =concat([sin_values, cos_values], dim=-1)  # (B, L, 2xN)
        output_shape = batch_shape + (2 * self.length * num_inputs,)
        embeddings = embeddings.view(*output_shape)  # (*, Lx2xN)
        if self.use_input:
            input_shape = batch_shape + (num_inputs,)
            inputs = inputs.view(*input_shape)
            embeddings = concat([inputs, embeddings], dim=-1)  # (*, (2L+1)xN)
        return embeddings

def find_optimal_num_groups(num_channels: int) -> int:
    """Find the optimal number of groups for GroupNorm."""
    # strategy: at most 32 groups, at least 8 channels per group
    num_groups = 32
    while num_groups > 1:
        if num_channels % num_groups == 0:
            num_channels_per_group = num_channels // num_groups
            if num_channels_per_group >= 8:
                break
        num_groups = num_groups // 2
    assert num_groups != 1, (
        f"Cannot find 'num_groups' in GroupNorm with 'num_channels={num_channels}' automatically. "
        "Please manually specify 'num_groups'."
    )
    return num_groups
class GraphSCModule(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        num_heads: int,
        num_blocks: int,
        num_layers_per_block: int,
        sigma_d: float,
        embedding_k: int = 0,
        embedding_dim: int = 10,

    ):
        super().__init__()

        self.sigma_d = sigma_d
        self.eps = 1e-7

        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        self.num_layers_per_block = num_layers_per_block

        self.embedding = FourierEmbedding(length=embedding_dim, k0=embedding_k, use_pi=False, use_input=True)
        print('in_proj',input_dim * (2 * embedding_dim + 1))
        print(find_optimal_num_groups(hidden_dim),hidden_dim)
        self.in_proj1 = nn.Sequential(nn.Conv1d(input_dim * (2 * embedding_dim + 1),hidden_dim,kernel_size=1,bias=True),
                                      nn.GroupNorm(find_optimal_num_groups(hidden_dim),hidden_dim),
                                    nn.LeakyReLU(),     
                                    nn.Identity()                                   
                                        
        )
        self.in_proj2 = nn.Sequential(
                                        nn.Conv1d(hidden_dim,hidden_dim,kernel_size=1,bias=True),
                                        nn.GroupNorm(find_optimal_num_groups(hidden_dim),hidden_dim),
                                        nn.LeakyReLU(),
                                        nn.Identity()
                                        
        )
        self.in_proj3 = nn.Sequential( nn.Conv1d(hidden_dim,hidden_dim,kernel_size=1,bias=True),
                                        nn.Identity()
        )
        """ self.in_proj = nn.Sequential(nn.Conv1d(input_dim * (2 * embedding_dim + 1),hidden_dim,kernel_size=1,bias=True),
                                        nn.GroupNorm(find_optimal_num_groups(hidden_dim),hidden_dim),
                                        nn.LeakyReLU(),
                                        nn.Identity(),
                                        nn.Conv1d(hidden_dim,hidden_dim,kernel_size=1,bias=True),
                                        nn.GroupNorm(find_optimal_num_groups(hidden_dim),hidden_dim),
                                        nn.LeakyReLU(),
                                        nn.Identity(),
                                        nn.Conv1d(hidden_dim,hidden_dim,kernel_size=1,bias=True),
                                        nn.Identity()
        ) """

        self.out_proj = nn.Linear(hidden_dim, output_dim)

        layers = []
        for i in range(num_layers_per_block * num_blocks):
            layers.append(TransformerLayer(hidden_dim, num_heads, dropout=None, activation_fn="ReLU"))
        self.transformer = nn.ModuleList(layers)

    def execute(
        self,
        src_corr_points,
        tgt_corr_points,
        local_corr_indices,
        local_corr_weights,
        local_corr_masks,
    ):
        """LOCSC Transformer Module.

        Args:
            src_corr_points (Tensor): The correspondence points in source point cloud (C, 3).
            tgt_corr_points (Tensor): The correspondence points in target point cloud (C, 3).
            local_corr_indices (LongTensor): The local indices for the correspondences (M, k).
            local_corr_weights (Tensor): The local weights for the correspondences (M, k).
            local_corr_masks (BoolTensor): The local masks for the correspondences (M, k).
        """
        num_correspondences = src_corr_points.shape[0]

        # 1. input projection
        src_corr_points_norm = src_corr_points - src_corr_points.mean(dim=0, keepdim=True)  # (C, 3)
        tgt_corr_points_norm = tgt_corr_points - tgt_corr_points.mean(dim=0, keepdim=True)  # (C, 3)
        src_corr_embeddings = self.embedding(src_corr_points_norm)  # (C, 6L)
        tgt_corr_embeddings = self.embedding(tgt_corr_points_norm)  # (C, 6L)
        
        corr_embeddings = concat([src_corr_embeddings, tgt_corr_embeddings], dim=-1)  # (C, 12L)

        corr_embeddings=corr_embeddings.transpose(0, 1).unsqueeze(0).contiguous()
        corr_feats = self.in_proj1(corr_embeddings).reshape(1,self.hidden_dim,num_correspondences) 
        corr_feats= self.in_proj2(corr_feats).reshape(1,self.hidden_dim,num_correspondences)
        corr_feats= self.in_proj3(corr_feats).reshape(1,self.hidden_dim,num_correspondences)# (C, d)
        corr_feats=corr_feats.squeeze(0).transpose(0, 1)

        # 2. spatial consistency
        local_src_corr_points = index_select(src_corr_points_norm, local_corr_indices, dim=0)  # (M, k, 3)
        local_tgt_corr_points = index_select(tgt_corr_points_norm, local_corr_indices, dim=0)  # (M, k, 3)
        sc_weights = spatial_consistency(local_src_corr_points, local_tgt_corr_points, self.sigma_d)  # (M, k, k)
        # 3. prepare for aggregation
        flat_local_corr_indices = local_corr_indices.view(-1)  # (Mxk)
        flat_local_corr_weights = local_corr_weights.view(-1)  # (Mxk)
        corr_sum_weights =[] #jt.zeros((num_correspondences,)).view(-1)  # (C,)
        unique_local_corr_indices=jt.unique(flat_local_corr_indices)
        count=0
        for i in unique_local_corr_indices:
            if count==i:
                select_indices=jt.equal(flat_local_corr_indices,i)
                corr_sum_weights.append(flat_local_corr_weights[select_indices].sum())
            else:
                while count!=i:
                    corr_sum_weights.append(jt.Var(0.0))
                    count+=1
                select_indices=jt.equal(flat_local_corr_indices,i)
                corr_sum_weights.append(flat_local_corr_weights[select_indices].sum())
            count+=1
        if count<num_correspondences:
            for i in range(num_correspondences-count):
                corr_sum_weights.append(jt.Var(0.0))
        corr_sum_weights=jt.stack(corr_sum_weights,dim=0).view(-1) 
        #jt.index_add_(corr_sum_weights,0, flat_local_corr_indices, flat_local_corr_weights)  # (C,) danger
        flat_local_corr_sum_weights = corr_sum_weights[flat_local_corr_indices]  # (Mxk)
        flat_local_corr_weights = flat_local_corr_weights / (flat_local_corr_sum_weights + self.eps)  # (Mxk)
        flat_local_corr_weights = flat_local_corr_weights.unsqueeze(1).expand(-1, self.hidden_dim)  # (Mxk, d)
        # 4. transformer module
        local_corr_masks = jt.logical_not(local_corr_masks)
        for block_idx in range(self.num_blocks):
            # 4.1 grouping
            local_corr_feats = index_select(corr_feats, local_corr_indices, dim=0)  # (M, k, d)
            # 4.2 transformer
            for layer_idx in range(self.num_layers_per_block):
                index = block_idx * self.num_layers_per_block + layer_idx
                local_corr_feats = self.transformer[index](
                    local_corr_feats,
                    local_corr_feats,
                    attention_factors=sc_weights,
                    memory_masks=None,
                )

            # 4.3 aggregate
            flat_local_corr_feats = local_corr_feats.view(-1, self.hidden_dim) 
            flat_local_corr_feats = flat_local_corr_feats * flat_local_corr_weights  # (Mxk, d)
            corr_feats = [] # (C, d)
            count=0
            for j in unique_local_corr_indices:
                if count==j:
                    select_indices=jt.equal(flat_local_corr_indices,j)
                    corr_feats.append(flat_local_corr_feats[select_indices].sum(dim=0))
                else:
                    while count!=j:
                        corr_feats.append(jt.zeros(flat_local_corr_feats.shape[-1]))
                        count+=1
                    select_indices=jt.equal(flat_local_corr_indices,j)
                    corr_feats.append(flat_local_corr_feats[select_indices].sum(dim=0))
                count+=1
            if count<num_correspondences:
                for j in range(num_correspondences-count):
                    corr_feats.append(jt.zeros(self.hidden_dim))
            
            corr_feats=jt.stack(corr_feats,dim=0)
            #print(block_idx,corr_feats.shape)
            
            #jt.index_add_(corr_feats,0, flat_local_corr_indices[:,0],flat_local_corr_feats)  # (C, d) danger
            """ for j in range(len(flat_local_corr_indices)):
                corr_feats[flat_local_corr_indices[j]]+=flat_local_corr_feats[j] """
                    # 5. output projection
        #print()
        corr_feats = self.out_proj(corr_feats)
        corr_masks = jt.greater(corr_sum_weights, 0.0)  # (C,)

        return corr_feats, corr_masks
