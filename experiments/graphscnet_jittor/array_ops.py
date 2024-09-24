from typing import Optional, Tuple, Union

import numpy as np
from numpy import ndarray
from scipy.spatial.transform import Rotation
import random
from scipy.spatial import cKDTree

def apply_transform(
    points: ndarray, transform: ndarray, normals: Optional[ndarray] = None
) -> Union[ndarray, Tuple[ndarray, ndarray]]:
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    points = np.matmul(points, rotation.T) + translation
    if normals is not None:
        normals = np.matmul(normals, rotation.T)
        return points, normals
    return points


def compose_transforms(*transforms: ndarray) -> ndarray:
    """
    Compose transforms from the first one to the last one.
    T = T_{n_1} \circ T_{n_2} \circ ... \circ T_1 \circ T_0
    """
    final_transform = transforms[0]
    for transform in transforms[1:]:
        final_transform = np.matmul(transform, final_transform)
    return final_transform


def get_transform_from_rotation_translation(rotation: Optional[ndarray], translation: Optional[ndarray]) -> ndarray:
    """Get rigid transform matrix from rotation matrix and translation vector.

    Args:
        rotation (array): (3, 3)
        translation (array): (3,)

    Returns:
        transform: (4, 4)
    """
    transform = np.eye(4)
    if rotation is not None:
        transform[:3, :3] = rotation
    if translation is not None:
        transform[:3, 3] = translation
    return transform


def get_rotation_translation_from_transform(transform: ndarray) -> Tuple[ndarray, ndarray]:
    """Get rotation matrix and translation vector from rigid transform matrix.

    Args:
        transform (array): (4, 4)

    Returns:
        rotation (array): (3, 3)
        translation (array): (3,)
    """
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    return rotation, translation


def inverse_transform(transform: ndarray) -> ndarray:
    """Inverse rigid transform.

    Args:
        transform (array): (4, 4)

    Return:
        inv_transform (array): (4, 4)
    """
    rotation, translation = get_rotation_translation_from_transform(transform)  # (3, 3), (3,)
    inv_rotation = rotation.T  # (3, 3)
    inv_translation = -np.matmul(inv_rotation, translation)  # (3,)
    inv_transform = get_transform_from_rotation_translation(inv_rotation, inv_translation)  # (4, 4)
    return inv_transform
def normalize_points(points: ndarray) -> ndarray:
    """Normalize point cloud to a unit sphere at origin."""
    points = points - points.mean(axis=0)
    points = points / np.max(np.linalg.norm(points, axis=1))
    return points


def normalize_points_on_xy_plane(points: ndarray) -> ndarray:
    """Normalize point cloud along x-y plane in place."""
    barycenter_2d = np.mean(points[:, :2], axis=0)
    points[:, :2] -= barycenter_2d
    return points


def sample_points(points: ndarray, num_samples: int, normals: Optional[ndarray] = None):
    """Sample the first K points."""
    points = points[:num_samples]
    if normals is None:
        return points
    normals = normals[:num_samples]
    return points, normals


def random_sample_points(points: ndarray, num_samples: int, normals: Optional[ndarray] = None):
    """Randomly sample points."""
    num_points = points.shape[0]
    sel_indices = np.random.permutation(num_points)
    if num_points > num_samples:
        sel_indices = sel_indices[:num_samples]
    elif num_points < num_samples:
        num_iterations = num_samples // num_points
        num_paddings = num_samples % num_points
        all_sel_indices = [sel_indices for _ in range(num_iterations)]
        if num_paddings > 0:
            all_sel_indices.append(sel_indices[:num_paddings])
        sel_indices = np.concatenate(all_sel_indices, axis=0)
    points = points[sel_indices]
    if normals is None:
        return points
    normals = normals[sel_indices]
    return points, normals


def random_scale_points(points: ndarray, low: float = 0.8, high: float = 1.2) -> ndarray:
    """Randomly rescale point cloud."""
    scale = random.uniform(low, high)
    points = points * scale
    return points


def random_shift_points(points: ndarray, shift: float = 0.2) -> ndarray:
    bias = np.random.uniform(low=-shift, high=shift, size=(1, 3))
    points = points + bias
    return points


def random_scale_shift_points(
    points: ndarray,
    low: float = 2.0 / 3.0,
    high: float = 3.0 / 2.0,
    shift: float = 0.2,
    normals: Optional[ndarray] = None,
):
    """Randomly sigma and shift point cloud."""
    scale = np.random.uniform(low=low, high=high, size=(1, 3))
    bias = np.random.uniform(low=-shift, high=shift, size=(1, 3))
    points = points * scale + bias
    if normals is None:
        return points
    normals = normals * scale
    normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
    return points, normals


def random_rotate_points_along_up_axis(
    points: ndarray,
    rotation_scale: float = 1.0,
    normals: Optional[ndarray] = None,
):
    """Randomly rotate point cloud along z-axis."""
    theta = np.random.rand() * 2.0 * np.pi * rotation_scale
    # fmt: off
    rotation_t = np.array([
        [np.cos(theta), np.sin(theta), 0],
        [-np.sin(theta), np.cos(theta), 0],
        [0, 0, 1],
    ])
    # fmt: on
    points = np.matmul(points, rotation_t)
    if normals is None:
        return points
    normals = np.matmul(normals, rotation_t)
    return points, normals


def random_jitter_points(points: ndarray, sigma: float = 0.01, scale: float = 0.05) -> ndarray:
    """Randomly jitter point cloud."""
    noises = np.clip(np.random.normal(scale=sigma, size=points.shape), a_min=-scale, a_max=scale)
    points = points + noises
    return points


def random_shuffle_points(points: ndarray, normals: Optional[ndarray] = None):
    """Randomly permute point cloud."""
    indices = np.random.permutation(points.shape[0])
    points = points[indices]
    if normals is None:
        return points
    normals = normals[indices]
    return points, normals


def random_dropout_points(points: ndarray, max_p: float) -> ndarray:
    """Randomly dropout point cloud proposed in PointNet++."""
    num_points = points.shape[0]
    p = np.random.rand(num_points) * max_p
    masks = np.random.rand(num_points) < p
    points[masks] = points[0]
    return points


def random_jitter_features(feats: ndarray, mu: float = 0.0, sigma: float = 0.01) -> ndarray:
    """Randomly jitter features in the original implementation of FCGF."""
    if random.random() < 0.95:
        feats = feats + np.random.normal(loc=mu, scale=sigma, size=feats.shape).astype(feats.dtype)
    return feats


# Normals


def regularize_normals(points: ndarray, normals: ndarray, positive: bool = True) -> ndarray:
    """Regularize the normals towards the positive/negative direction to the origin point.

    positive: the origin point is on positive direction of the normals.
    negative: the origin point is on negative direction of the normals.
    """
    dot_products = -(points * normals).sum(axis=1, keepdims=True)
    direction = dot_products > 0
    if positive:
        normals = normals * direction - normals * (1 - direction)
    else:
        normals = normals * (1 - direction) - normals * direction
    return normals


# Cropping


def random_sample_direction() -> ndarray:
    """Random sample a plane passing the origin and return its normal."""
    phi = np.random.uniform(0.0, 2 * np.pi)  # longitude
    theta = np.random.uniform(0.0, np.pi)  # latitude

    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    normal = np.asarray([x, y, z])

    return normal


def random_crop_points_with_plane(
    points: ndarray, p_normal: Optional[ndarray] = None, keep_ratio: float = 0.7, normals: Optional[ndarray] = None
):
    """Random crop a point cloud with a plane and keep num_samples points."""
    num_samples = int(np.floor(points.shape[0] * keep_ratio + 0.5))
    if p_normal is None:
        p_normal = random_sample_direction()  # (3,)
    distances = np.dot(points, p_normal)
    sel_indices = np.argsort(-distances)[:num_samples]  # select the largest K points
    points = points[sel_indices]
    if normals is None:
        return points
    normals = normals[sel_indices]
    return points, normals


def random_sample_viewpoint(radius: float = 500) -> ndarray:
    """Randomly sample observing point from 8 directions."""
    return np.random.rand(3) + np.array([radius, radius, radius]) * np.random.choice([1.0, -1.0], size=3)


def random_crop_points_from_viewpoint(
    points: ndarray, viewpoint: Optional[ndarray] = None, keep_ratio: float = 0.7, normals: Optional[ndarray] = None
):
    """Random crop point cloud from the observing point."""
    num_samples = int(np.floor(points.shape[0] * keep_ratio + 0.5))
    if viewpoint is None:
        viewpoint = random_sample_viewpoint()
    distances = np.linalg.norm(viewpoint - points, axis=1)
    sel_indices = np.argsort(distances)[:num_samples]
    points = points[sel_indices]
    if normals is None:
        return points
    normals = normals[sel_indices]
    return points, normals


# Sample SE(3), SO(3)


def random_sample_rotation(rotation_factor: float = 1.0) -> np.ndarray:
    """Random sample a rotation matrix for strong/standard data augmentation.

    Steps:
        1. random sample three euler angles from [0, 2pi / factor].
        2. compute the rotation matrix from the euler angles.
    """
    # angle_z, angle_y, angle_x
    euler = np.random.rand(3) * np.pi * 2 / rotation_factor  # (0, 2 * pi / rotation_range)
    rotation = euler_to_rotation_matrix(euler, "zyx")
    return rotation


def random_sample_rotation_norm() -> np.ndarray:
    axis = np.random.rand(3) - 0.5
    axis = axis / np.linalg.norm(axis) + 1e-8
    theta = np.pi * np.random.rand()
    euler = axis * theta
    rotation = euler_to_rotation_matrix(euler, "zyx")
    return rotation


def random_sample_transform(rotation_magnitude: float, translation_magnitude: float) -> np.ndarray:
    euler = np.random.rand(3) * np.pi * rotation_magnitude / 180.0  # (0, rot_mag)
    rotation = euler_to_rotation_matrix(euler, "zyx")
    translation = np.random.uniform(-translation_magnitude, translation_magnitude, 3)
    transform = get_transform_from_rotation_translation(rotation, translation)
    return transform


def random_sample_small_transform(scale: float = 0.1) -> ndarray:
    """Random sample a small transform for weak data augmentation.

    Steps:
        1. random sample an axis.
        2. random sample a rotation angle: [0, sigma * pi / sqrt(3)]
        3. compute the rotation matrix with Rodrigues' equation.
        4. random sample a translation vector: [-sigma / sqrt(3), sigma / sqrt(3)].
        5. compose transformation matrix.
    """
    axis = random_sample_direction()
    theta = np.random.rand() * scale * np.pi / np.sqrt(3.0)
    rotation = rodrigues_rotation_formula(axis, theta)
    translation = np.random.randn(3) * scale / np.sqrt(3.0)
    transform = get_transform_from_rotation_translation(rotation, translation)
    return transform

def get_rotation_along_axis(scale: float, axis: int = "z") -> ndarray:
    assert axis in ["x", "y", "z"]
    theta = 2.0 * np.pi * scale
    if axis == "x":
        rotation = np.asarray(
            [[1.0, 0.0, 0.0], [0.0, np.cos(theta), -np.sin(theta)], [0.0, np.sin(theta), np.cos(theta)]]
        )
    elif axis == "y":
        rotation = np.asarray(
            [[np.cos(theta), 0.0, np.sin(theta)], [0.0, 1.0, 1.0], [-np.sin(theta), 0.0, np.cos(theta)]]
        )
    else:
        rotation = np.asarray(
            [[np.cos(theta), -np.sin(theta), 0.0], [np.sin(theta), np.cos(theta), 0.0], [0.0, 0.0, 1.0]]
        )
    return rotation


def apply_rotation(
    points: ndarray, rotation: ndarray, normals: Optional[ndarray] = None
) -> Union[ndarray, Tuple[ndarray, ndarray]]:
    points = np.matmul(points, rotation.T)
    if normals is not None:
        normals = np.matmul(normals, rotation.T)
        return points, normals
    return points


def skew_symmetric_matrix(vector: ndarray) -> ndarray:
    """Compute Skew-symmetric Matrix.

    [v]_{\times} =  0 -z  y
                    z  0 -x
                   -y  x  0

    Args:
        vector (ndarray): input vectors (3)

    Returns:
        skews (ndarray): output skew-symmetric matrix (3, 3)
    """
    skews = np.zeros(shape=(3, 3))
    skews[0, 1] = -vector[2]
    skews[0, 2] = vector[1]
    skews[1, 0] = vector[2]
    skews[1, 2] = -vector[0]
    skews[2, 0] = -vector[1]
    skews[2, 1] = vector[0]
    return skews


def rodrigues_rotation_formula(axis: ndarray, angle: float) -> ndarray:
    """Compute Rodrigues Rotation Matrix.

    R = I + \sin{\theta} K + (1 - \cos{\theta}) K^2,
    where K is the skew-symmetric matrix of the axis vector.

    Args:
        axis (array<float>): normalized axis vectors (3)
        angle (float): rotation angles in right-hand direction in rad.

    Returns:
        rotation (array<float>): Rodrigues rotation matrix (3, 3)
    """
    skews = skew_symmetric_matrix(axis)  # (3, 3)
    rotation = np.eye(3) + np.sin(angle) * skews + (1.0 - np.cos(angle)) * np.matmul(skews, skews)
    return rotation


def axis_angle_to_rotation_matrix(phi: ndarray) -> ndarray:
    rotation = Rotation.from_rotvec(phi).as_matrix()
    return rotation


def axis_angle_to_quaternion(phi: ndarray) -> ndarray:
    q = Rotation.from_rotvec(phi).as_quat()
    q = q[..., [3, 0, 1, 2]]  # (xyzw) -> (wxyz)
    return q


def quaternion_to_axis_angle(q: ndarray) -> ndarray:
    q = q[..., [1, 2, 3, 0]]  # (xyzw) -> (wxyz)
    phi = Rotation.from_quat(q).as_rotvec()
    return phi


def quaternion_to_rotation_matrix(q: ndarray) -> ndarray:
    q = q[..., [1, 2, 3, 0]]  # (xyzw) -> (wxyz)
    rotation = Rotation.from_quat(q).as_matrix()
    return rotation


def rotation_matrix_to_axis_angle(rotation: ndarray) -> ndarray:
    phi = Rotation.from_matrix(rotation).as_rotvec()
    return phi


def rotation_matrix_to_quaternion(rotation: ndarray) -> ndarray:
    q = Rotation.from_matrix(rotation).as_quat()
    q = q[..., [3, 0, 1, 2]]  # (xyzw) -> (wxyz)
    return q


def rotation_matrix_to_euler(rotation: ndarray, order: str, use_degree: bool = False) -> ndarray:
    euler = Rotation.from_matrix(rotation).as_euler(order, degrees=use_degree)
    return euler


def euler_to_rotation_matrix(euler: ndarray, order: str) -> ndarray:
    rotation = Rotation.from_euler(order, euler).as_matrix()
    return rotation



def knn(q_points: ndarray, s_points: ndarray, k=1, return_distance: bool = False):
    """Compute the nearest neighbor for the query points in support points.

    Note:
        If k=1, the return arrays are squeezed.
    """
    s_tree = cKDTree(s_points)
    knn_distances, knn_indices = s_tree.query(q_points, k=k)
    if return_distance:
        return knn_distances, knn_indices
    else:
        return knn_indices
def knn_interpolate(
    q_points: ndarray,
    s_points: ndarray,
    s_feats: ndarray,
    k: int = 3,
    distance_limit: float = 0.1,
    eps: float = 1e-10,
    inf: float = 1e10,
) -> ndarray:
    distances, indices = knn(q_points, s_points, k=k, return_distance=True)  # (N, 3)
    if distance_limit is not None:
        distances[distances > distance_limit] = inf
    weights = 1.0 / (distances + eps)
    weights = weights / weights.sum(axis=1, keepdims=True)  # (N, 3)
    knn_feats = s_feats[indices]  # (N, 3, C)
    q_feats = (knn_feats * weights[:, :, None]).sum(axis=1)
    return q_feats