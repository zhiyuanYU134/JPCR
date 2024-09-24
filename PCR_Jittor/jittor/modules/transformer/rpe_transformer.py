r"""Transformer with Relative Positional Embeddings.

Relative positional embedding is further projected in each multi-head attention layer.

The shape of input tensor should be (B, N, C). Implemented with `nn.Linear` and `nn.LayerNorm` (with affine).
"""

import jittor as jt
import jittor.nn as nn
from jittor import linalg

from einops import rearrange
from IPython import embed

from PCR_Jittor.jittor.modules.layers import build_dropout_layer
from PCR_Jittor.jittor.modules.transformer.output_layer import AttentionOutput


""" class RPEMultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=None):
        super(RPEMultiHeadAttention, self).__init__()
        if d_model % num_heads != 0:
            raise ValueError('`d_model` ({}) must be a multiple of `num_heads` ({}).'.format(d_model, num_heads))

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_model_per_head = d_model // num_heads

        self.proj_q = nn.Linear(self.d_model, self.d_model)
        self.proj_k = nn.Linear(self.d_model, self.d_model)
        self.proj_v = nn.Linear(self.d_model, self.d_model)
        self.proj_p = nn.Linear(self.d_model, self.d_model)

        self.dropout = build_dropout_layer(dropout)

    
    def _transpose_for_scores(self, x):
        x = x.view(x.shape[0], x.shape[1], self.num_heads, self.d_model_per_head)
        x = x.permute(0, 2, 1, 3)
        return x

    def _transpose_rpe_for_scores(self, x):
        x = x.view(x.shape[0], x.shape[1], x.shape[2], self.num_heads, self.d_model_per_head)
        x = x.permute(0, 3, 1, 2, 4)
        return x

    def execute(self, input_q, input_k, input_v, embed_qk, key_weights=None, key_masks=None, attention_factors=None):

        q = self.proj_q(input_q)
        k = self.proj_k(input_k)
        v = self.proj_v(input_v)
        p = self.proj_p(embed_qk)


        q = self._transpose_for_scores(q)
        k = self._transpose_for_scores(k)
        v = self._transpose_for_scores(v)
        p = self._transpose_rpe_for_scores(p)

        attention_scores_p = linalg.einsum('bhnc,bhnmc->bhnm', q, p)
        attention_scores_e = linalg.einsum('bhnc,bhmc->bhnm', q, k)
        attention_scores = (attention_scores_e + attention_scores_p) / self.d_model_per_head ** 0.5
        if attention_factors is not None:
            attention_scores = attention_factors.unsqueeze(1) * attention_scores
        if key_weights is not None:
            attention_scores = attention_scores * key_weights.unsqueeze(1).unsqueeze(1)
        if key_masks is not None:
            #attention_scores = attention_scores.masked_fill(key_masks.unsqueeze(1).unsqueeze(1), float('-inf'))
            attention_scores = jt.masked_fill(attention_scores,key_masks.unsqueeze(1).unsqueeze(1), float('-inf'))
        attention_scores = nn.softmax(attention_scores, dim=-1)
        attention_scores = self.dropout(attention_scores)

        hidden_states = jt.matmul(attention_scores, v)

        hidden_states = hidden_states.permute(0, 2, 1, 3).contiguous()
        hidden_states = hidden_states.view(hidden_states.shape[0], hidden_states.shape[1], self.d_model)


        return hidden_states, attention_scores """


class RPEMultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=None):
        super(RPEMultiHeadAttention, self).__init__()
        if d_model % num_heads != 0:
            raise ValueError('`d_model` ({}) must be a multiple of `num_heads` ({}).'.format(d_model, num_heads))

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_model_per_head = d_model // num_heads

        self.proj_q = nn.Linear(self.d_model, self.d_model)
        self.proj_k = nn.Linear(self.d_model, self.d_model)
        self.proj_v = nn.Linear(self.d_model, self.d_model)
        self.proj_p = nn.Linear(self.d_model, self.d_model)

        self.dropout = build_dropout_layer(dropout)

    
    def _transpose_for_scores(self, x):
        x = x.view(x.shape[0], x.shape[1], self.num_heads, self.d_model_per_head)
        x = x.permute(0, 2, 1, 3)
        return x

    def _transpose_rpe_for_scores(self, x):
        x = x.view(x.shape[0], x.shape[1], x.shape[2], self.num_heads, self.d_model_per_head)
        x = x.permute(0, 3, 1, 2, 4)
        return x

    def execute(self, input_q, input_k, input_v, embed_qk, key_weights=None, key_masks=None, attention_factors=None,mask_mode=None):
        r"""Scaled Dot-Product Attention with Pre-computed Relative Positional Embedding (forward)

        Args:
            input_q: torch.Tensor (B, N, C)
            input_k: torch.Tensor (B, M, C)
            input_v: torch.Tensor (B, M, C)
            embed_qk: torch.Tensor (B, N, M, C), relative positional embedding
            key_weights: torch.Tensor (B, M), soft masks for the keys
            key_masks: torch.Tensor (B, M), True if ignored, False if preserved
            attention_factors: torch.Tensor (B, N, M)

        Returns:
            hidden_states: torch.Tensor (B, C, N)
            attention_scores: torch.Tensor (B, H, N, M)
        """
        q = self.proj_q(input_q)
        k = self.proj_k(input_k)
        v = self.proj_v(input_v)
        p = self.proj_p(embed_qk)


        q = self._transpose_for_scores(q)
        k = self._transpose_for_scores(k)
        v = self._transpose_for_scores(v)
        p = self._transpose_rpe_for_scores(p)

        attention_scores_p = linalg.einsum('bhnc,bhnmc->bhnm', q, p)
        attention_scores_e = linalg.einsum('bhnc,bhmc->bhnm', q, k)
        attention_scores = (attention_scores_e + attention_scores_p) / self.d_model_per_head ** 0.5
        if attention_factors is not None:
            attention_scores = attention_factors.unsqueeze(1) * attention_scores
        if key_weights is not None:
            attention_scores = attention_scores * key_weights.unsqueeze(1).unsqueeze(1)
        if key_masks is not None:
            #attention_scores = attention_scores.masked_fill(key_masks.unsqueeze(1).unsqueeze(1), float('-inf'))
            if mask_mode is not None:  
                key_masks=key_masks.unsqueeze(1).unsqueeze(1)
                key_masks=key_masks.expand(key_masks.shape[0], self.num_heads, key_masks.shape[2],key_masks.shape[3] )
                attention_scores = attention_scores.masked_fill(key_masks, float('-inf'))
            else:
                attention_scores = jt.masked_fill(attention_scores,key_masks.unsqueeze(1).unsqueeze(1), float('-inf'))
        attention_scores = nn.softmax(attention_scores, dim=-1)
        attention_scores = self.dropout(attention_scores)
        hidden_states = jt.matmul(attention_scores, v)

        hidden_states = hidden_states.permute(0, 2, 1, 3).contiguous()
        hidden_states = hidden_states.view(hidden_states.shape[0], hidden_states.shape[1], self.d_model)


        return hidden_states, attention_scores


class RPEAttentionLayer(nn.Module):
    def __init__(self, d_model, num_heads, dropout=None):
        super(RPEAttentionLayer, self).__init__()
        self.attention = RPEMultiHeadAttention(d_model, num_heads, dropout=dropout)
        self.linear = nn.Linear(d_model, d_model)
        self.dropout = build_dropout_layer(dropout)
        self.norm = nn.LayerNorm(d_model)

    def execute(
        self,
        input_states,
        memory_states,
        position_states,
        memory_weights=None,
        memory_masks=None,
        attention_factors=None,
        mask_mode=None
    ):
        hidden_states, attention_scores = self.attention(
            input_states,
            memory_states,
            memory_states,
            position_states,
            key_weights=memory_weights,
            key_masks=memory_masks,
            attention_factors=attention_factors,
            mask_mode=mask_mode
        )
        hidden_states = self.linear(hidden_states)
        hidden_states = self.dropout(hidden_states)
        output_states = self.norm(hidden_states + input_states)
        return output_states, attention_scores


class RPETransformerLayer(nn.Module):
    def __init__(self, d_model, num_heads, dropout=None, activation_fn='ReLU'):
        super(RPETransformerLayer, self).__init__()
        self.attention = RPEAttentionLayer(d_model, num_heads, dropout=dropout)
        self.output = AttentionOutput(d_model, dropout=dropout, activation_fn=activation_fn)

    def execute(
        self,
        input_states,
        memory_states,
        position_states,
        memory_weights=None,
        memory_masks=None,
        attention_factors=None,
        mask_mode=None
    ):
        hidden_states, attention_scores = self.attention(
            input_states,
            memory_states,
            position_states,
            memory_weights=memory_weights,
            memory_masks=memory_masks,
            attention_factors=attention_factors,
            mask_mode=mask_mode
        )
        output_states = self.output(hidden_states)
        return output_states, attention_scores
