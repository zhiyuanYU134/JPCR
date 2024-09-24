from typing import Optional

import jittor as jt
import jittor.nn as nn
from jittor import init


def apply_transform(points, transform, normals= None):
    r"""Rigid transform to points and normals (optional).

    Given a point cloud P(3, N), normals V(3, N) and a transform matrix T in the form of
      | R t |
      | 0 1 |,
    the output point cloud Q = RP + t, V' = RV.

    In the implementation, P and V are (N, 3), so R should be transposed: Q = PR^T + t, V' = VR^T.

    There are two cases supported:
    1. points and normals are (*, 3), transform is (4, 4), the output points are (*, 3).
       In this case, the transform is applied to all points.
    2. points and normals are (B, N, 3), transform is (B, 4, 4), the output points are (B, N, 3).
       In this case, the transform is applied batch-wise. The points can be broadcast if B=1.

    Args:
        points (Tensor): (*, 3) or (B, N, 3)
        normals (optional[Tensor]=None): same shape as points.
        transform (Tensor): (4, 4) or (B, 4, 4)

    Returns:
        points (Tensor): same shape as points.
        normals (Tensor): same shape as points.
    """
    if normals is not None:
        assert points.shape == normals.shape
    if transform.ndim == 2:
        rotation = transform[:3, :3]
        translation = transform[:3, 3]
        points_shape = points.shape
        points = points.reshape(-1, 3)
        points = jt.matmul(points, rotation.transpose(-1, -2)) + translation
        points = points.reshape(*points_shape)
        if normals is not None:
            normals = normals.reshape(-1, 3)
            normals = jt.matmul(normals, rotation.transpose(-1, -2))
            normals = normals.reshape(*points_shape)
    elif transform.ndim == 3 and points.ndim == 3:
        rotation = transform[:, :3, :3]  # (B, 3, 3)
        translation = transform[:, None, :3, 3]  # (B, 1, 3)
        points=points.expand(rotation.shape[0],points.shape[1], points.shape[2]) 

        points = jt.matmul(points, rotation.transpose(-1, -2)) + translation
        if normals is not None:
            normals = jt.matmul(normals, rotation.transpose(-1, -2))
    else:
        raise ValueError(
            'Incompatible shapes between points {} and transform {}.'.format(
                tuple(points.shape), tuple(transform.shape)
            )
        )
    if normals is not None:
        return points, normals
    else:
        return points


def apply_rotation(points, rotation, normals= None):
    r"""Rotate points and normals (optional) along the origin.

    Given a point cloud P(3, N), normals V(3, N) and a rotation matrix R, the output point cloud Q = RP, V' = RV.

    In the implementation, P and V are (N, 3), so R should be transposed: Q = PR^T, V' = VR^T.

    There are two cases supported:
    1. points and normals are (*, 3), rotation is (3, 3), the output points are (*, 3).
       In this case, the rotation is applied to all points.
    2. points and normals are (B, N, 3), transform is (B, 3, 3), the output points are (B, N, 3).
       In this case, the rotation is applied batch-wise. The points can be broadcast if B=1.

    Args:
        points (Tensor): (*, 3) or (B, N, 3)
        normals (optional[Tensor]=None): same shape as points.
        rotation (Tensor): (3, 3) or (B, 3, 3)

    Returns:
        points (Tensor): same shape as points.
        normals (Tensor): same shape as points.
    """
    if normals is not None:
        assert points.shape == normals.shape
    if rotation.ndim == 2:
        points_shape = points.shape
        points = points.reshape(-1, 3)
        points = jt.matmul(points, rotation.transpose(-1, -2))
        points = points.reshape(*points_shape)
        if normals is not None:
            normals = normals.reshape(-1, 3)
            normals = jt.matmul(normals, rotation.transpose(-1, -2))
            normals = normals.reshape(*points_shape)
    elif rotation.ndim == 3 and points.ndim == 3:
        points = jt.matmul(points, rotation.transpose(-1, -2))
        if normals is not None:
            normals = jt.matmul(normals, rotation.transpose(-1, -2))
    else:
        raise ValueError(
            'Incompatible shapes between points {} and rotation{}.'.format(tuple(points.shape), tuple(rotation.shape))
        )
    if normals is not None:
        return points, normals
    else:
        return points


def get_rotation_translation_from_transform(transform):
    r"""Decompose transformation matrix into rotation matrix and translation vector.

    Args:
        transform (Tensor): (*, 4, 4)

    Returns:
        rotation (Tensor): (*, 3, 3)
        translation (Tensor): (*, 3)
    """
    rotation = transform[..., :3, :3]
    translation = transform[..., :3, 3]
    return rotation, translation


def get_transform_from_rotation_translation(rotation, translation):
    r"""Compose transformation matrix from rotation matrix and translation vector.

    Args:
        rotation (Tensor): (*, 3, 3)
        translation (Tensor): (*, 3)

    Returns:
        transform (Tensor): (*, 4, 4)
    """
    input_shape = rotation.shape
    rotation = rotation.view(-1, 3, 3)
    translation = translation.view(-1, 3)
    transform = init.eye(4).to(rotation).unsqueeze(0).repeat(rotation.shape[0], 1, 1)
    transform[:, :3, :3] = rotation
    transform[:, :3, 3] = translation
    output_shape = input_shape[:-2] + (4, 4)
    transform = transform.view(*output_shape)
    return transform


def inverse_transform(transform):
    r"""Inverse rigid transform.

    Args:
        transform (Tensor): (*, 4, 4)

    Return:
        inv_transform (Tensor): (*, 4, 4)
    """
    rotation, translation = get_rotation_translation_from_transform(transform)  # (*, 3, 3), (*, 3)
    inv_rotation = rotation.transpose(-1, -2)  # (*, 3, 3)
    inv_translation = -jt.matmul(inv_rotation, translation.unsqueeze(-1)).squeeze(-1)  # (*, 3)
    inv_transform = get_transform_from_rotation_translation(inv_rotation, inv_translation)  # (*, 4, 4)
    return inv_transform


def skew_symmetric_matrix(inputs):
    r"""Compute Skew-symmetric Matrix.

    [v]_{\times} =  0 -z  y
                    z  0 -x
                   -y  x  0

    Args:
        inputs (Tensor): input vectors (*, c)

    Returns:
        skews (Tensor): output skew-symmetric matrix (*, 3, 3)
    """
    input_shape = inputs.shape
    output_shape = input_shape[:-1] + (3, 3)
    skews = jt.zeros(size=output_shape)
    skews[..., 0, 1] = -inputs[..., 2]
    skews[..., 0, 2] = inputs[..., 1]
    skews[..., 1, 0] = inputs[..., 2]
    skews[..., 1, 2] = -inputs[..., 0]
    skews[..., 2, 0] = -inputs[..., 1]
    skews[..., 2, 1] = inputs[..., 0]
    return skews


def rodrigues_rotation_matrix(axes, angles):
    r"""Compute Rodrigues Rotation Matrix.

    R = I + \sin{\theta} K + (1 - \cos{\theta}) K^2,
    where K is the skew-symmetric matrix of the axis vector.

    Args:
        axes (Tensor): axis vectors (*, 3)
        angles (Tensor): rotation angles in right-hand direction in rad. (*)

    Returns:
        rotations (Tensor): Rodrigues rotation matrix (*, 3, 3)
    """
    input_shape = axes.shape
    axes = axes.view(-1, 3)
    angles = angles.view(-1)
    axes = jt.normalize(axes, p=2, dim=1)
    skews = skew_symmetric_matrix(axes)  # (B, 3, 3)
    sin_values = jt.sin(angles).view(-1, 1, 1)  # (B,)
    cos_values = jt.cos(angles).view(-1, 1, 1)  # (B,)
    eyes = init.eye(3).unsqueeze(0).expand_as(skews)  # (B, 3, 3)
    rotations = eyes + sin_values * skews + (1.0 - cos_values) * jt.matmul(skews, skews)
    output_shape = input_shape[:-1] + (3, 3)
    rotations = rotations.view(*output_shape)
    return rotations


def rodrigues_alignment_matrix(src_vectors, tgt_vectors):
    r"""Compute the Rodrigues rotation matrix aligning source vectors to target vectors.

    Args:
        src_vectors (Tensor): source vectors (*, 3)
        tgt_vectors (Tensor): target vectors (*, 3)

    Returns:
        rotations (Tensor): rotation matrix (*, 3, 3)
    """
    input_shape = src_vectors.shape
    src_vectors = src_vectors.view(-1, 3)  # (B, 3)
    tgt_vectors = tgt_vectors.view(-1, 3)  # (B, 3)

    # compute axes
    src_vectors = jt.normalize(src_vectors, dim=-1, p=2)  # (B, 3)
    tgt_vectors = jt.normalize(tgt_vectors, dim=-1, p=2)  # (B, 3)
    src_skews = skew_symmetric_matrix(src_vectors)  # (B, 3, 3)
    axes = jt.matmul(src_skews, tgt_vectors.unsqueeze(-1)).squeeze(-1)  # (B, 3)

    # compute rodrigues rotation matrix
    sin_values = jt.norm(axes, dim=-1)  # (B,)
    cos_values = (src_vectors * tgt_vectors).sum(dim=-1)  # (B,)
    axes = jt.normalize(axes, dim=-1, p=2)  # (B, 3)
    skews = skew_symmetric_matrix(axes)  # (B, 3, 3)
    eyes = init.eye(3).unsqueeze(0).expand_as(skews)  # (B, 3, 3)
    sin_values = sin_values.view(-1, 1, 1)
    cos_values = cos_values.view(-1, 1, 1)
    rotations = eyes + sin_values * skews + (1.0 - cos_values) * jt.matmul(skews, skews)

    # handle opposite direction
    sin_values = sin_values.view(-1)
    cos_values = cos_values.view(-1)
    masks = jt.logical_and(jt.equal(sin_values, 0.0), jt.less(cos_values, 0.0))
    rotations[masks] *= -1

    output_shape = input_shape[:-1] + (3, 3)
    rotations = rotations.view(*output_shape)

    return rotations
