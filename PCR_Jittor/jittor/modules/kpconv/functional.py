import jittor as jt
from jittor import nn
from jittor.contrib import concat
from PCR_Jittor.jittor.modules.ops import index_select
import numpy as np

def gather(x, idx, method=2):
    """
    implementation of a custom gather operation for faster backwards.
    :param x: input with shape [N, D_1, ... D_d]
    :param idx: indexing with shape [n_1, ..., n_m]
    :param method: Choice of the method
    :return: x[idx] with shape [n_1, ..., n_m, D_1, ... D_d]
    """

    if method == 0:
        return x[idx]
    elif method == 1:
        x = x.unsqueeze(1)
        x = x.expand((-1, idx.shape[-1], -1))
        idx = idx.unsqueeze(2)
        idx = idx.expand((-1, -1, x.shape[-1]))
        return x.gather(0, idx)
    elif method == 2:
        for i, ni in enumerate(idx.size()[1:]):
            x = x.unsqueeze(i+1)
            new_s = list(x.size())
            new_s[i+1] = ni
            x = x.expand(new_s)
        n = len(idx.size())
        # print("forloop size 2:", len(x.size()[n:]))
        for i, di in enumerate(x.size()[n:]):
            idx = idx.unsqueeze(i+n)
            new_s = list(idx.size())
            new_s[i+n] = di
            idx = idx.expand(new_s)
        return x.gather(0, idx)
    else:
        raise ValueError('Unkown method')


def radius_gaussian(sq_r, sig, eps=1e-9):
    """
    Compute a radius gaussian (gaussian of distance)
    :param sq_r: input radiuses [dn, ..., d1, d0]
    :param sig: extents of gaussians [d1, d0] or [d0] or float
    :return: gaussian of sq_r [dn, ..., d1, d0]
    """
    return jt.exp(-sq_r / (2 * sig**2 + eps))

def maxpool(x, inds):
    """
    Pools features with the maximum values.
    :param x: [n1, d] features matrix
    :param inds: [n2, max_num] pooling indices
    :return: [n2, d] pooled features matrix
    """
    
    # Add a last row with minimum features for shadow pools
    x = jt.concat((x, jt.zeros_like(x[:1, :])), 0)

    # Get all features for each pooling location [n2, max_num, d]
    pool_features = gather(x, inds)

    # Pool the maximum [n2, d]
    max_features= jt.max(pool_features, 1)
    return max_features


def global_average(x, batch_lengths):
    """
    Block performing a global average over batch pooling
    :param x: [N, D] input features
    :param batch_lengths: [B] list of batch lengths
    :return: [B, D] averaged features
    """

    # Loop over the clouds of the batch
    averaged_features = []
    i0 = 0
    for b_i, length in enumerate(batch_lengths):
        # length.shape: [1]
        # Average features for each batch cloud
        averaged_features.append(jt.mean(x[i0:i0 + length.data[0]], dim=0))

        # Increment for next cloud
        i0 += length.data[0]

    # Average features in each batch
    return jt.stack(averaged_features)



def nearest_upsample(x, upsample_indices):
    """Pools features from the closest neighbors.

    WARNING: this function assumes the neighbors are ordered.

    Args:
        x: [n1, d] features matrix
        upsample_indices: [n2, max_num] Only the first column is used for pooling

    Returns:
        x: [n2, d] pooled features matrix
    """
    # Add a last row with minimum features for shadow pools
    slack_row = jt.array(np.zeros_like(x[:1, :]))
    x = concat((x, slack_row), 0)
    # Get features for each pooling location [n2, d]
    x = index_select(x, upsample_indices[:, 0], dim=0)
    return x


def knn_interpolate(s_feats, q_points, s_points, neighbor_indices, k, eps=1e-8):
    r"""K-NN interpolate.

    WARNING: this function assumes the neighbors are ordered.

    Args:
        s_feats (Tensor): (M, C)
        q_points (Tensor): (N, 3)
        s_points (Tensor): (M, 3)
        neighbor_indices (LongTensor): (N, X)
        k (int)
        eps (float)

    Returns:
        q_feats (Tensor): (N, C)
    """
    s_points =concat((s_points, jt.zeros_like(s_points[:1, :])), 0)  # (M + 1, 3)
    s_feats =concat((s_feats, jt.zeros_like(s_feats[:1, :])), 0)  # (M + 1, C)
    knn_indices = neighbor_indices[:, :k].contiguous()
    knn_points = index_select(s_points, knn_indices, dim=0)  # (N, k, 3)
    knn_feats = index_select(s_feats, knn_indices, dim=0)  # (N, k, C)
    knn_sq_distances = (q_points.unsqueeze(1) - knn_points).pow(2).sum(dim=-1)  # (N, k)
    knn_masks = jt.not_equal(knn_indices, s_points.shape[0] - 1).float()  # (N, k)
    knn_weights = knn_masks / (knn_sq_distances + eps)  # (N, k)
    knn_weights = knn_weights / (knn_weights.sum(dim=1, keepdim=True) + eps)  # (N, k)
    q_feats = (knn_feats * knn_weights.unsqueeze(-1)).sum(dim=1)  # (N, C)
    return q_feats


def max_pool(x, neighbor_indices):
    """Max pooling from neighbors.

    Args:
        x: [n1, d] features matrix
        neighbor_indices: [n2, max_num] pooling indices

    Returns:
        pooled_feats: [n2, d] pooled features matrix
    """
    x =concat((x, jt.zeros_like(x[:1, :])), 0)
    neighbor_feats = index_select(x, neighbor_indices, dim=0)
    pooled_feats= jt.max(neighbor_feats, dim=1)
    #pooled_feats = neighbor_feats.max(1)
    return pooled_feats


def global_avgpool(x, batch_lengths):
    """Global average pooling over batch.

    Args:
        x: [N, D] input features
        batch_lengths: [B] list of batch lengths

    Returns:
        x: [B, D] averaged features
    """
    # Loop over the clouds of the batch
    averaged_features = []
    i0 = 0
    for b_i, length in enumerate(batch_lengths):
        # Average features for each batch cloud
        averaged_features.append(jt.mean(x[i0 : i0 + length], dim=0))
        # Increment for next cloud
        i0 += length
    # Average features in each batch
    x = jt.stack(averaged_features)
    return x
