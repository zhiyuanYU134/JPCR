import torch
from torch import Tensor
from typing import Optional, Tuple, Union



def apply_transform(
    points: Tensor, transform: Tensor, normals: Optional[Tensor] = None
) -> Union[Tensor, Tuple[Tensor, ...]]:
    """Rigid transform to points and normals (optional).

    Given a point cloud P(3, N), normals V(3, N) and a transform matrix T in the form of
      | R t |
      | 0 1 |,
    the output point cloud Q = RP + t, V' = RV.

    In the implementation, P and V are (N, 3), so R should be transposed: Q = PR^T + t, V' = VR^T.

    There are three cases supported:
    1. points and normals are (*, 3), transform is (4, 4), the output points are (*, 3).
       In this case, the transform is applied to all points.
    2. points and normals are (B, N, 3), transform is (B, 4, 4), the output points are (B, N, 3).
       In this case, the transform is applied batch-wise. The points are automatically broadcast if B=1.
    3. points and normals are (B, 3), transform is (B, 4, 4), the output points are (B, 3).
       In this case, the points are automatically broadcast to (B, 1, 3) and the transform is applied batch-wise. The
       first dim of points/normals and transform must be the same.

    Args:
        points (Tensor): (*, 3) or (B, N, 3) or (B, 3).
        normals (optional[Tensor]=None): same shape as points.
        transform (Tensor): (4, 4) or (B, 4, 4)

    Returns:
        points (Tensor): same shape as points.
        normals (Tensor): same shape as points.
    """
    assert transform.dim() == 2 or (
        transform.dim() == 3 and points.dim() in [2, 3]
    ), f"Incompatible shapes between points {tuple(points.shape)} and transform {tuple(transform.shape)}."

    if normals is not None:
        assert (
            points.shape == normals.shape
        ), f"The shapes of points {tuple(points.shape)} and normals {tuple(normals.shape)} must be the same."

    if transform.dim() == 2:
        # case 1: (*, 3) x (4, 4)
        input_shape = points.shape
        rotation = transform[:3, :3]  # (3, 3)
        translation = transform[None, :3, 3]  # (1, 3)
        points = points.reshape(-1, 3)
        points = torch.matmul(points, rotation.transpose(-1, -2)) + translation
        points = points.reshape(*input_shape)
        if normals is not None:
            normals = normals.reshape(-1, 3)
            normals = torch.matmul(normals, rotation.transpose(-1, -2))
            normals = normals.reshape(*input_shape)
    elif transform.dim() == 3 and points.dim() == 3:
        # case 2: (B, N, 3) x (B, 4, 4)
        rotation = transform[:, :3, :3]  # (B, 3, 3)
        translation = transform[:, None, :3, 3]  # (B, 1, 3)
        points = torch.matmul(points, rotation.transpose(-1, -2)) + translation
        if normals is not None:
            normals = torch.matmul(normals, rotation.transpose(-1, -2))
    elif transform.dim() == 3 and points.dim() == 2:
        # case 3: (B, 3) x (B, 4, 4)
        rotation = transform[:, :3, :3]  # (B, 3, 3)
        translation = transform[:, None, :3, 3]  # (B, 1, 3)
        points = points.unsqueeze(1)
        points = torch.matmul(points, rotation.transpose(-1, -2)) + translation
        points = points.squeeze(1)
        if normals is not None:
            normals = normals.unsqueeze(1)
            normals = torch.matmul(normals, rotation.transpose(-1, -2))
            normals = normals.squeeze(1)

    if normals is not None:
        return points, normals

    return points
def apply_deformation(
    points: Tensor,
    nodes: Tensor,
    transforms: Tensor,
    anchor_indices: Tensor,
    anchor_weights: Tensor,
    eps: float = 1e-6,
) -> Tensor:
    """Apply Embedded Deformation Warping Function.

    Args:
        points (Tensor): The points to warp in the shape of (N, 3).
        nodes (Tensor): The graph nodes in the shape of (M, 3).
        transforms (Tensor): The associated transformations for each node in the shape of (M, 4, 4).
        anchor_indices (LongTensor): The indices of the anchor nodes for the points in the shape of (N, K). If an index
            is -1, the corresponding anchor does not exist.
        anchor_weights (Tensor): The skinning weights of the anchor nodes for the point in the shape of (N, K).
        eps (float=1e-6): A safe number for division.

    Returns:
        warped_points (Tensor): The warped points in the shape of (N, 3).
    """
    anchor_weights = anchor_weights / (anchor_weights.sum(dim=1, keepdim=True) + eps)  # (N, K)
    anchor_masks = torch.ne(anchor_indices, -1)  # (N, K)
    p_indices, col_indices = torch.nonzero(anchor_masks, as_tuple=True)  # (C), (C)
    n_indices = anchor_indices[p_indices, col_indices]  # (C)
    weights = anchor_weights[p_indices, col_indices]  # (C)
    sel_points = points[p_indices]  # (C, 3)
    sel_nodes = nodes[n_indices]  # (C, 3)
    sel_transforms = transforms[n_indices]  # (C, 4, 4)
    sel_warped_points = apply_transform(sel_points - sel_nodes, sel_transforms) + sel_nodes  # (C, 3)
    sel_warped_points = sel_warped_points * weights.unsqueeze(1)  # (C, 3)
    warped_points = torch.zeros_like(points)  # (N, 3)
    p_indices = p_indices.unsqueeze(1).expand_as(sel_warped_points)  # (C, 3)
    warped_points.scatter_add_(dim=0, index=p_indices, src=sel_warped_points)  # (N, 3)
    return warped_points

def apply_rotation(
    points: Tensor, rotation: Tensor, normals: Optional[Tensor] = None
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """Rotate points and normals (optional) along the origin.

    Given a point cloud P(3, N), normals V(3, N) and a rotation matrix R, the output point cloud Q = RP, V' = RV.

    In the implementation, P and V are (N, 3), so R should be transposed: Q = PR^T, V' = VR^T.

    There are three cases supported:
    1. points and normals are (*, 3), rotation is (3, 3), the output points are (*, 3).
       In this case, the rotation is applied to all points.
    2. points and normals are (B, N, 3), rotation is (B, 3, 3), the output points are (B, N, 3).
       In this case, the rotation is applied batch-wise. The points can be broadcast if B=1.
    3. points and normals are (B, 3), rotation is (B, 3, 3), the output points are (B, 3).
       In this case, the points are automatically broadcast to (B, 1, 3) and the rotation is applied batch-wise. The
       first dim of points/normals and rotation must be the same.

    Args:
        points (Tensor): (*, 3) or (B, N, 3), or (B, 3)
        normals (Tensor=None): same shape as points.
        rotation (Tensor): (3, 3) or (B, 3, 3)

    Returns:
        points (Tensor): same shape as points.
        normals (Tensor): same shape as points.
    """
    assert rotation.dim() == 2 or (
        rotation.dim() == 3 and points.dim() in (2, 3)
    ), f"Incompatible shapes between points {tuple(points.shape)} and rotation {tuple(rotation.shape)}."

    if normals is not None:
        assert (
            points.shape == normals.shape
        ), f"The shapes of points {tuple(points.shape)} and normals {tuple(normals.shape)} must be the same."

    if rotation.dim() == 2:
        # case 1
        input_shape = points.shape
        points = points.reshape(-1, 3)
        points = torch.matmul(points, rotation.transpose(-1, -2))
        points = points.reshape(*input_shape)
        if normals is not None:
            normals = normals.reshape(-1, 3)
            normals = torch.matmul(normals, rotation.transpose(-1, -2))
            normals = normals.reshape(*input_shape)
    elif rotation.dim() == 3 and points.dim() == 3:
        # case 2
        points = torch.matmul(points, rotation.transpose(-1, -2))
        if normals is not None:
            normals = torch.matmul(normals, rotation.transpose(-1, -2))
    elif rotation.dim() == 3 and points.dim() == 2:
        # case 3
        points = points.unsqueeze(1)
        points = torch.matmul(points, rotation.transpose(-1, -2))
        points = points.squeeze(1)
        if normals is not None:
            normals = normals.unsqueeze(1)
            normals = torch.matmul(normals, rotation.transpose(-1, -2))
            normals = normals.squeeze(1)

    if normals is not None:
        return points, normals

    return points

def axis_angle_to_rotation_matrix(phi: Tensor) -> Tensor:
    """Convert axis angle to rotation matrix.

    A.k.a., exponential map on Lie group, which is implemented with Rodrigues' rotation formula.

    Note:
        If phi is a zero vector, the rotation matrix is an identity matrix.

    Args:
        phi (Tensor): The so(3) exponential coordinates in the shape of (*, 3).

    Returns:
        rotation (Tensor): The SO(3) rotation matrix in the shape of (*, 3, 3).
    """
    theta = torch.linalg.norm(phi, dim=-1)
    omega = safe_divide(phi, theta.unsqueeze(-1))
    rotation = rodrigues_rotation_formula(omega, theta)
    return rotation
def safe_divide(a: Union[Tensor, float], b: Tensor, eps: float = 1e-6):
    b = torch.clamp(b, min=eps)
    return a / b

def skew_symmetric_matrix(vector: Tensor) -> Tensor:
    """Compute Skew-symmetric Matrix.

    [v]_{\times} =  0 -z  y
                    z  0 -x
                   -y  x  0

    Note: Use matrix multiplication to make the computation differentiable.

    Args:
        vector (Tensor): input vectors (*, 3)

    Returns:
        skew (Tensor): output skew-symmetric matrix (*, 3, 3)
    """
    vector_shape = vector.shape
    matrix_shape = vector_shape[:-1] + (9, 3)
    vector_to_skew = torch.zeros(size=matrix_shape).cuda()  # (*, 9, 3)
    vector_to_skew[..., 1, 2] = -1.0
    vector_to_skew[..., 2, 1] = 1.0
    vector_to_skew[..., 3, 2] = 1.0
    vector_to_skew[..., 5, 0] = -1.0
    vector_to_skew[..., 6, 1] = -1.0
    vector_to_skew[..., 7, 0] = 1.0
    skew_shape = vector_shape[:-1] + (3, 3)
    skew = torch.matmul(vector_to_skew, vector.unsqueeze(-1)).view(*skew_shape)
    return skew


def rodrigues_rotation_formula(omega: Tensor, theta: Tensor) -> Tensor:
    """Compute rotation matrix from axis-angle with Rodrigues' Rotation Formula.

    R = I + \sin{\theta} K + (1 - \cos{\theta}) K^2,
    where K is the skew-symmetric matrix of the axis vector.

    Note:
        If omega is a zero vector, the rotation matrix is always an identity matrix.

    Args:
        omega (Tensor): The unit rotation axis vector in the shape of (*, 3).
        theta (Tensor): The rotation angles (rad) in right-hand direction in the shape of (*).

    Returns:
        rotations (Tensor): The SO(3) rotation matrix in the shape of (*, 3, 3).
    """
    input_shape = omega.shape
    omega = omega.view(-1, 3)
    theta = theta.view(-1)
    skew = skew_symmetric_matrix(omega)  # (B, 3, 3)
    sin_value = torch.sin(theta).view(-1, 1, 1)  # (B, 1, 1)
    cos_value = torch.cos(theta).view(-1, 1, 1)  # (B, 1, 1)
    eye = torch.eye(3).cuda().unsqueeze(0).expand_as(skew)  # (B, 3, 3)
    rotation = eye + sin_value * skew + (1.0 - cos_value) * torch.matmul(skew, skew)
    output_shape = input_shape[:-1] + (3, 3)
    rotation = rotation.view(*output_shape)
    return rotation



def get_transform_from_rotation_translation(rotation: Tensor, translation: Tensor) -> Tensor:
    """Compose transformation matrix from rotation matrix and translation vector.

    Args:
        rotation (Tensor): (*, 3, 3)
        translation (Tensor): (*, 3)

    Returns:
        transform (Tensor): (*, 4, 4)
    """
    input_shape = rotation.shape
    rotation = rotation.view(-1, 3, 3)
    translation = translation.view(-1, 3)
    transform = torch.eye(4).to(rotation).unsqueeze(0).repeat(rotation.shape[0], 1, 1)
    transform[:, :3, :3] = rotation
    transform[:, :3, 3] = translation
    output_shape = input_shape[:-2] + (4, 4)
    transform = transform.view(*output_shape)
    return transform
