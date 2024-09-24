from typing import Optional, Tuple, Union

""" import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor
from torch.nn import functional as F """
import jittor as jt
from jittor import nn
from jittor.contrib import concat
from jittor import linalg


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        q_embed_proj: bool = False,
        k_embed_proj: bool = False,
        v_embed_proj: bool = False,
        qk_embed_proj: bool = False,
        qv_embed_proj: bool = False,
        dropout: Optional[float] = None,
    ):
        super().__init__()

        assert d_model % num_heads == 0, f"'d_model={d_model}' is not divisible by 'num_heads={num_heads}'."

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_model_per_head = d_model // num_heads

        self.q_token_layer = nn.Linear(self.d_model, self.d_model)
        self.k_token_layer = nn.Linear(self.d_model, self.d_model)
        self.v_token_layer = nn.Linear(self.d_model, self.d_model)

        self.has_q_embed_proj = q_embed_proj
        if self.has_q_embed_proj:
            self.q_embed_layer = nn.Linear(self.d_model, self.d_model)

        self.has_k_embed_proj = k_embed_proj
        if self.has_k_embed_proj:
            self.k_embed_layer = nn.Linear(self.d_model, self.d_model)

        self.has_v_embed_proj = v_embed_proj
        if self.has_v_embed_proj:
            self.v_embed_layer = nn.Linear(self.d_model, self.d_model)

        self.has_qk_embed_proj = qk_embed_proj
        if self.has_qk_embed_proj:
            self.qk_embed_layer = nn.Linear(self.d_model, self.d_model)

        self.has_qv_embed_proj = qv_embed_proj
        if self.has_qv_embed_proj:
            self.qv_embed_layer = nn.Linear(self.d_model, self.d_model)

        self.dropout = nn.Identity()
    def _transpose_for_scores(self, x):
        x = x.view(x.shape[0], x.shape[1], self.num_heads, self.d_model_per_head)
        x = x.permute(0, 2, 1, 3)
        return x
    def execute(
        self,
        q_tokens,
        k_tokens,
        v_tokens,
        q_embeds= None,
        k_embeds = None,
        v_embeds = None,
        qk_embeds= None,
        qv_embeds= None,
        k_weights= None,
        k_masks= None,
        qk_weights = None,
        qk_masks = None,
    ):
        """Multi-Head Attention execute propagation.

        Args:
            q_tokens (Tensor): query tokens (B, N, C)
            k_tokens (Tensor): key tokens (B, M, C)
            v_tokens (Tensor): value tokens (B, M, C)
            q_embeds (Tensor): query embeddings (B, N, C)
            k_embeds (Tensor): key embeddings (B, M, C)
            v_embeds (Tensor): value embeddings (B, M, C)
            qk_embeds (Tensor): query-key embeddings (B, N, M, C)
            qv_embeds (Tensor): query-value embeddings (B, N, M, C)
            k_weights (Tensor): key weights (B, M)
            k_masks (BoolTensor): key masks. If True, ignored. (B, M)
            qk_weights (Tensor): query-key weights (B, N, M)
            qk_masks (BoolTensor): query-key masks. If True, ignored. (B, N, M)

        Returns:
            hidden_tokens (Tensor): output tokens (B, C, N)
            attention_scores (Tensor): attention scores (after dropout) (B, H, N, M)
        """
        # input check
        if self.has_q_embed_proj:
            assert q_embeds is not None, "No 'q_embeds' but 'q_embed_proj' is set."
        if self.has_k_embed_proj:
            assert k_embeds is not None, "No 'k_embeds' but 'k_embed_proj' is set."
        if self.has_v_embed_proj:
            assert v_embeds is not None, "No 'v_embeds' but 'v_embed_proj' is set."
        if self.has_qk_embed_proj:
            assert qk_embeds is not None, "No 'qk_embeds' but 'qk_embed_proj' is set."
        if self.has_qv_embed_proj:
            assert qv_embeds is not None, "No 'qv_embeds' but 'qv_embed_proj' is set."

        # compute query and key tokens
        q_tokens = self.q_token_layer(q_tokens)

        q_tokens = self._transpose_for_scores(q_tokens)
        k_tokens = self.k_token_layer(k_tokens)
        k_tokens =  self._transpose_for_scores(k_tokens)

        # compute attention scores
        attention_scores =jt.linalg.einsum("bhnc,bhmc->bhnm", q_tokens, k_tokens)
        attention_scores = attention_scores / self.d_model_per_head ** 0.5
        if qk_weights is not None:
            attention_scores = attention_scores * qk_weights.unsqueeze(1)
        if k_masks is not None:
            attention_scores = attention_scores.masked_fill(k_masks.unsqueeze(1).unsqueeze(1), float("-inf"))

        attention_scores = nn.softmax(attention_scores, dim=-1)
        attention_scores = self.dropout(attention_scores)

        # compute output tokens
        v_tokens = self.v_token_layer(v_tokens)
        v_tokens =  self._transpose_for_scores(v_tokens)
        
        hidden_tokens = jt.matmul(attention_scores, v_tokens)
        hidden_tokens = hidden_tokens.permute(0, 2, 1, 3).contiguous()
        hidden_tokens = hidden_tokens.view(hidden_tokens.shape[0], hidden_tokens.shape[1], self.d_model)

        return hidden_tokens, attention_scores


class AttentionLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        q_embed_proj: bool = False,
        k_embed_proj: bool = False,
        v_embed_proj: bool = False,
        qk_embed_proj: bool = False,
        qv_embed_proj: bool = False,
        dropout: Optional[float] = None,
    ):
        super().__init__()
        self.attention = MultiHeadAttention(
            d_model,
            num_heads,
            q_embed_proj=q_embed_proj,
            k_embed_proj=k_embed_proj,
            v_embed_proj=v_embed_proj,
            qk_embed_proj=qk_embed_proj,
            qv_embed_proj=qv_embed_proj,
            dropout=dropout,
        )
        self.linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Identity()
        self.norm = nn.LayerNorm(d_model)

    def execute(
        self,
        q_tokens,
        k_tokens,
        v_tokens,
        q_embeds = None,
        k_embeds= None,
        v_embeds= None,
        qk_embeds = None,
        qv_embeds= None,
        k_weights= None,
        k_masks= None,
        qk_weights = None,
        qk_masks= None,
    ) :
        hidden_tokens, attention_scores = self.attention(
            q_tokens,
            k_tokens,
            v_tokens,
            q_embeds=q_embeds,
            k_embeds=k_embeds,
            v_embeds=v_embeds,
            qk_embeds=qk_embeds,
            qv_embeds=qv_embeds,
            k_weights=k_weights,
            k_masks=k_masks,
            qk_weights=qk_weights,
            qk_masks=qk_masks,
        )
        hidden_tokens = self.linear(hidden_tokens)
        hidden_tokens = self.dropout(hidden_tokens)
        output_tokens = self.norm(hidden_tokens + q_tokens)
        return output_tokens, attention_scores


class AttentionOutput(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.expand = nn.Linear(d_model, d_model * 2)
        self.activation = nn.ReLU()
        self.squeeze = nn.Linear(d_model * 2, d_model)
        self.dropout = nn.Identity()
        self.norm = nn.LayerNorm(d_model)

    def execute(self, input_tokens):
        hidden_tokens = self.expand(input_tokens)
        hidden_tokens = self.activation(hidden_tokens)
        hidden_tokens = self.squeeze(hidden_tokens)
        hidden_tokens = self.dropout(hidden_tokens)
        output_tokens = self.norm(input_tokens + hidden_tokens)
        return output_tokens


class TransformerLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        q_embed_proj: bool = False,
        k_embed_proj: bool = False,
        v_embed_proj: bool = False,
        qk_embed_proj: bool = False,
        qv_embed_proj: bool = False,
    ):
        super().__init__()
        self.attention = AttentionLayer(
            d_model,
            num_heads,
            q_embed_proj=q_embed_proj,
            k_embed_proj=k_embed_proj,
            v_embed_proj=v_embed_proj,
            qk_embed_proj=qk_embed_proj,
            qv_embed_proj=qv_embed_proj,
        )
        self.output = AttentionOutput(d_model)

    def execute(
        self,
        q_tokens,
        k_tokens,
        v_tokens,
        q_embeds = None,
        k_embeds = None,
        v_embeds= None,
        qk_embeds = None,
        qv_embeds = None,
        k_weights = None,
        k_masks= None,
        qk_weights = None,
        qk_masks= None,
        return_attention_score = False,
    ):
        hidden_tokens, attention_scores = self.attention(
            q_tokens,
            k_tokens,
            v_tokens,
            q_embeds=q_embeds,
            k_embeds=k_embeds,
            v_embeds=v_embeds,
            qk_embeds=qk_embeds,
            qv_embeds=qv_embeds,
            k_weights=k_weights,
            k_masks=k_masks,
            qk_weights=qk_weights,
            qk_masks=qk_masks,
        )
        output_tokens = self.output(hidden_tokens)

        if return_attention_score:
            return output_tokens, attention_scores
        return output_tokens
