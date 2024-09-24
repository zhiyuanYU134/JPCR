import ipdb
import jittor as jt
from jittor import nn
from jittor.contrib import concat
def log_sum_exp(data,dim,keepdim=False):
    eps = 1e-20
    m = jt.max(data, dim=dim, keepdims=True)
    data=data-m
    data=data.exp().sum(dim, keepdim)+ eps
    data=data.log()+m.squeeze(dim)
    return data

def circle_loss(
    pos_masks,
    neg_masks,
    feat_dists,
    pos_margin,
    neg_margin,
    pos_optimal,
    neg_optimal,
    log_scale,
):
    # get anchors that have both positive and negative pairs
    row_masks = (jt.greater(pos_masks.sum(-1), 0) & jt.greater(neg_masks.sum(-1), 0)).detach()
    col_masks = (jt.greater(pos_masks.sum(-2), 0) &jt.greater(neg_masks.sum(-2), 0)).detach()

    # get alpha for both positive and negative pairs
    pos_weights = feat_dists - 1e5 * (jt.logical_not(pos_masks)).float()  # mask the non-positive
    pos_weights = pos_weights - pos_optimal  # mask the uninformative positive
    pos_weights = jt.maximum(jt.zeros_like(pos_weights), pos_weights).detach()

    neg_weights = feat_dists + 1e5 * (jt.logical_not(neg_masks)).float()  # mask the non-negative
    neg_weights = neg_optimal - neg_weights  # mask the uninformative negative
    neg_weights = jt.maximum(jt.zeros_like(neg_weights), neg_weights).detach()

    loss_pos_row =log_sum_exp(log_scale * (feat_dists - pos_margin) * pos_weights, dim=-1)
    loss_pos_col = log_sum_exp(log_scale * (feat_dists - pos_margin) * pos_weights, dim=-2)

    loss_neg_row = log_sum_exp(log_scale * (neg_margin - feat_dists) * neg_weights, dim=-1)
    loss_neg_col =log_sum_exp(log_scale * (neg_margin - feat_dists) * neg_weights, dim=-2)

    loss_row = nn.softplus(loss_pos_row + loss_neg_row) / log_scale
    loss_col = nn.softplus(loss_pos_col + loss_neg_col) / log_scale

    loss = (loss_row[row_masks].mean() + loss_col[col_masks].mean()) / 2

    return loss


def weighted_circle_loss(
    pos_masks,
    neg_masks,
    feat_dists,
    pos_margin,
    neg_margin,
    pos_optimal,
    neg_optimal,
    log_scale,
    pos_scales=None,
    neg_scales=None,
):
    # get anchors that have both positive and negative pairs
    row_masks = (jt.greater(pos_masks.sum(-1), 0) & jt.greater(neg_masks.sum(-1), 0)).detach()
    col_masks = (jt.greater(pos_masks.sum(-2), 0) & jt.greater(neg_masks.sum(-2), 0)).detach()

    # get alpha for both positive and negative pairs
    pos_weights = feat_dists - 1e5 * (jt.logical_not(pos_masks)).float()  # mask the non-positive
    pos_weights = pos_weights - pos_optimal  # mask the uninformative positive
    pos_weights = jt.maximum(jt.zeros_like(pos_weights), pos_weights)
    if pos_scales is not None:
        pos_weights = pos_weights * pos_scales
    pos_weights = pos_weights.detach()

    neg_weights = feat_dists + 1e5 * (jt.logical_not(neg_masks)).float()  # mask the non-negative
    neg_weights = neg_optimal - neg_weights  # mask the uninformative negative
    neg_weights = jt.maximum(jt.zeros_like(neg_weights), neg_weights)
    if neg_scales is not None:
        neg_weights = neg_weights * neg_scales
    neg_weights = neg_weights.detach()

    loss_pos_row = log_sum_exp(log_scale * (feat_dists - pos_margin) * pos_weights, dim=-1)
    loss_pos_col = log_sum_exp(log_scale * (feat_dists - pos_margin) * pos_weights, dim=-2)

    loss_neg_row =log_sum_exp(log_scale * (neg_margin - feat_dists) * neg_weights, dim=-1)
    loss_neg_col =log_sum_exp(log_scale * (neg_margin - feat_dists) * neg_weights, dim=-2)

    loss_row = nn.softplus(loss_pos_row + loss_neg_row) / log_scale
    loss_col = nn.softplus(loss_pos_col + loss_neg_col) / log_scale
    loss = (loss_row[row_masks].mean() + loss_col[col_masks].mean()) / 2

    return loss


class CircleLoss(nn.Module):
    def __init__(self, pos_margin, neg_margin, pos_optimal, neg_optimal, log_scale):
        super(CircleLoss, self).__init__()
        self.pos_margin = pos_margin
        self.neg_margin = neg_margin
        self.pos_optimal = pos_optimal
        self.neg_optimal = neg_optimal
        self.log_scale = log_scale

    def forward(self, pos_masks, neg_masks, feat_dists):
        return circle_loss(
            pos_masks,
            neg_masks,
            feat_dists,
            self.pos_margin,
            self.neg_margin,
            self.pos_optimal,
            self.neg_optimal,
            self.log_scale,
        )


class WeightedCircleLoss(nn.Module):
    def __init__(self, pos_margin, neg_margin, pos_optimal, neg_optimal, log_scale):
        super(WeightedCircleLoss, self).__init__()
        self.pos_margin = pos_margin
        self.neg_margin = neg_margin
        self.pos_optimal = pos_optimal
        self.neg_optimal = neg_optimal
        self.log_scale = log_scale

    def execute(self, pos_masks, neg_masks, feat_dists, pos_scales=None, neg_scales=None):
        return weighted_circle_loss(
            pos_masks,
            neg_masks,
            feat_dists,
            self.pos_margin,
            self.neg_margin,
            self.pos_optimal,
            self.neg_optimal,
            self.log_scale,
            pos_scales=pos_scales,
            neg_scales=neg_scales,
        )
