import jittor as jt
import jittor.nn as nn
from jittor import linalg
from jittor import init
import ipdb


def weighted_procrustes(
    src_points,
    ref_points,
    weights=None,
    weight_thresh=0.0,
    eps=1e-5,
    return_transform=False,
):
    r"""Compute rigid transformation from `src_points` to `ref_points` using weighted SVD.

    Modified from [PointDSC](https://github.com/XuyangBai/PointDSC/blob/master/models/common.py).

    Args:
        src_points: torch.Tensor (B, N, 3) or (N, 3)
        ref_points: torch.Tensor (B, N, 3) or (N, 3)
        weights: torch.Tensor (B, N) or (N,) (default: None)
        weight_thresh: float (default: 0.)
        eps: float (default: 1e-5)
        return_transform: bool (default: False)

    Returns:
        R: torch.Tensor (B, 3, 3) or (3, 3)
        t: torch.Tensor (B, 3) or (3,)
        transform: torch.Tensor (B, 4, 4) or (4, 4)
    """
    if src_points.ndim == 2:
        src_points = src_points.unsqueeze(0)
        ref_points = ref_points.unsqueeze(0)
        if weights is not None:
            weights = weights.unsqueeze(0)
        squeeze_first = True
    else:
        squeeze_first = False

    batch_size = src_points.shape[0]
    if weights is None:
        weights = jt.ones_like(src_points[:, :, 0])
    
    weights = jt.where(jt.less(weights, weight_thresh), jt.zeros_like(weights), weights)
    
    weights = weights / (jt.sum(weights, dim=1, keepdims=True) + eps)

    weights = weights.unsqueeze(2)  # (B, N, 1)

    src_centroid = jt.sum(src_points * weights, dim=1, keepdims=True)  # (B, 1, 3)
    ref_centroid = jt.sum(ref_points * weights, dim=1, keepdims=True)  # (B, 1, 3)
    src_points_centered = src_points - src_centroid  # (B, N, 3)
    ref_points_centered = ref_points - ref_centroid  # (B, N, 3)


    """ src_points_centered=src_points_centered.astype(jt.float64)
    weights=weights.astype(jt.float64)
    ref_points_centered=ref_points_centered.astype(jt.float64) """
    
    #H = src_points_centered.permute(0, 2, 1) @ (weights * ref_points_centered)
    H=jt.matmul(src_points_centered.permute(0, 2, 1),weights * ref_points_centered)
    """ print(H)
    print(H.shape) """
    U, _, V = linalg.svd(H)  # H = USV^T
    Ut, V = U.transpose(1, 2), V
    eye = init.eye(3).unsqueeze(0).repeat(batch_size, 1, 1)
    eye[:, -1, -1] = nn.sign(linalg.det(V @ Ut))
    R = V @ eye @ Ut

    t = ref_centroid.permute(0, 2, 1) - R @ src_centroid.permute(0, 2, 1)
    t = t.squeeze(2)
    """ print(t)
    print(t.shape) """

    if return_transform:
        transform = init.eye(4).unsqueeze(0).repeat(batch_size, 1, 1)
        transform[:, :3, :3] = R
        transform[:, :3, 3] = t
        if squeeze_first:
            transform = transform.squeeze(0)
        return transform.astype(jt.float32),src_points_centered,jt.sum(weights, dim=1, keepdims=True),ref_points_centered
    else:
        if squeeze_first:
            R = R.squeeze(0).astype(jt.float32)
            t = t.squeeze(0).astype(jt.float32)
        return R, t


class WeightedProcrustes(nn.Module):
    def __init__(self, weight_thresh=0.0, eps=1e-5, return_transform=False):
        super(WeightedProcrustes, self).__init__()
        self.weight_thresh = weight_thresh
        self.eps = eps
        self.return_transform = return_transform

    def execute(self, src_points, tgt_points, weights=None):
        return weighted_procrustes(
            src_points,
            tgt_points,
            weights=weights,
            weight_thresh=self.weight_thresh,
            eps=self.eps,
            return_transform=self.return_transform,
        )
