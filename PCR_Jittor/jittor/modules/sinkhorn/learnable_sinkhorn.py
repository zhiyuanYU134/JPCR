from cmath import isnan
import numpy as np
import jittor as jt
import jittor.nn as nn
from jittor.contrib import concat
def log_sum_exp(data,dim,keepdim=False):
    eps = 1e-20
    m = jt.max(data, dim=dim, keepdims=True)
    data=data-m
    data=data.exp().sum(dim, keepdim)+ eps
    data=data.log()+m.squeeze(dim)
    return data

class LearnableLogOptimalTransport(nn.Module):
    def __init__(self, num_iterations, inf=1e12):
        r"""Sinkhorn Optimal transport with dustbin parameter (SuperGlue style)."""
        super(LearnableLogOptimalTransport, self).__init__()
        self.num_iterations = num_iterations
        self.alpha=jt.nn.Parameter(jt.Var(1.0))
        self.inf = inf

    def log_sinkhorn_normalization(self, scores, log_mu, log_nu):
        u, v = jt.zeros_like(log_mu), jt.zeros_like(log_nu)
        for i in range(self.num_iterations):
            u = log_mu - log_sum_exp(scores + v.unsqueeze(1), dim=2)
            v = log_nu -log_sum_exp(scores + u.unsqueeze(2), dim=1)
        return scores + u.unsqueeze(2) + v.unsqueeze(1)

    def execute(self, scores, row_masks=None, col_masks=None):
        r"""Sinkhorn Optimal Transport (SuperGlue style) forward.

        Args:
            scores: torch.Tensor (B, M, N)
            row_masks: torch.Tensor (B, M)
            col_masks: torch.Tensor (B, N)

        Returns:
            matching_scores: torch.Tensor (B, M+1, N+1)
        """
        batch_size, num_row, num_col = scores.shape
        if row_masks is None:
            row_masks = jt.ones((batch_size, num_row), dtype='bool')
        if col_masks is None:
            col_masks = jt.ones((batch_size, num_col), dtype='bool')

        padded_row_masks = jt.zeros((batch_size, num_row + 1), dtype='bool')
        padded_row_masks[:, :num_row] = jt.logical_not(row_masks)
        padded_col_masks = jt.zeros((batch_size, num_col + 1), dtype='bool')
        padded_col_masks[:, :num_col] = jt.logical_not(col_masks)
        padded_score_masks = jt.logical_or(padded_row_masks.unsqueeze(2), padded_col_masks.unsqueeze(1))


        padded_col = self.alpha.expand(batch_size, num_row, 1)
        padded_row = self.alpha.expand(batch_size, 1, num_col + 1)
        padded_scores = concat([concat([scores, padded_col], dim=-1), padded_row], dim=1)

        padded_scores=jt.masked_fill(padded_scores,padded_score_masks, -self.inf)

        num_valid_row = row_masks.float().sum(1)
        num_valid_col = col_masks.float().sum(1)
        norm = -jt.log(num_valid_row + num_valid_col)  # (B,)
        """ print(num_valid_row)
        print(num_valid_col)
        print(jt.log(num_valid_col))
        print(norm) """
        log_mu = jt.empty((batch_size, num_row + 1))
        log_mu[:, :num_row] = norm.unsqueeze(1)
        log_mu[:, num_row] = jt.log(num_valid_col) + norm
        log_mu[padded_row_masks] = -self.inf

        log_nu = jt.empty((batch_size, num_col + 1))
        log_nu[:, :num_col] = norm.unsqueeze(1)
        log_nu[:, num_col] = jt.log(num_valid_row) + norm
        log_nu[padded_col_masks] = -self.inf
        """ print(jt.isnan(padded_scores))
        print(jt.isnan(log_mu))
        print(jt.isnan(log_nu)) """

        outputs = self.log_sinkhorn_normalization(padded_scores, log_mu, log_nu)
        outputs = outputs - norm.unsqueeze(1).unsqueeze(2)

        return outputs

    def __repr__(self):
        format_string = self.__class__.__name__ + '(num_iterations={})'.format(self.num_iterations)
        return format_string
