import os.path as osp
import pickle
import random
from typing import Dict

import torch
import numpy as np
from jittor.dataset.dataset import Dataset
import jittor as jt
from jittor import nn
from jittor.contrib import concat
from pathlib import Path
from PCR_Jittor.utils.pointcloud import (
    random_sample_rotation,
    random_sample_rotation_v2,
    get_transform_from_rotation_translation,
)
from PCR_Jittor.utils.registration import get_correspondences
#from PCR_Jittor.jittor.modules.ops import grid_subsample, radius_search

import cpp_wrappers.cpp_subsampling.grid_subsampling as cpp_subsampling
import cpp_wrappers.cpp_neighbors.radius_neighbors as cpp_neighbors
BASE_DIR = Path(__file__).parent

def create_3D_rotations(axis, angle):
    """
    Create rotation matrices from a list of axes and angles. Code from wikipedia on quaternions
    :param axis: float32[N, 3]
    :param angle: float32[N,]
    :return: float32[N, 3, 3]
    """

    t1 = np.cos(angle)
    t2 = 1 - t1
    t3 = axis[:, 0] * axis[:, 0]
    t6 = t2 * axis[:, 0]
    t7 = t6 * axis[:, 1]
    t8 = np.sin(angle)
    t9 = t8 * axis[:, 2]
    t11 = t6 * axis[:, 2]
    t12 = t8 * axis[:, 1]
    t15 = axis[:, 1] * axis[:, 1]
    t19 = t2 * axis[:, 1] * axis[:, 2]
    t20 = t8 * axis[:, 0]
    t24 = axis[:, 2] * axis[:, 2]
    R = np.stack([t1 + t2 * t3,
                  t7 - t9,
                  t11 + t12,
                  t7 + t9,
                  t1 + t2 * t15,
                  t19 - t20,
                  t11 - t12,
                  t19 + t20,
                  t1 + t2 * t24], axis=1)

    return np.reshape(R, (-1, 3, 3))

def grid_subsampling(points, features=None, labels=None, sampleDl=0.1, verbose=0):
    """
    CPP wrapper for a grid subsampling (method = barycenter for points and features)
    :param points: (N, 3) matrix of input points
    :param features: optional (N, d) matrix of features (floating number)
    :param labels: optional (N,) matrix of integer labels
    :param sampleDl: parameter defining the size of grid voxels
    :param verbose: 1 to display
    :return: subsampled points, with features and/or labels depending of the input
    """

    if (features is None) and (labels is None):
        return cpp_subsampling.subsample(points,
                                         sampleDl=sampleDl,
                                         verbose=verbose)
    elif (labels is None):
        return cpp_subsampling.subsample(points,
                                         features=features,
                                         sampleDl=sampleDl,
                                         verbose=verbose)
    elif (features is None):
        return cpp_subsampling.subsample(points,
                                         classes=labels,
                                         sampleDl=sampleDl,
                                         verbose=verbose)
    else:
        return cpp_subsampling.subsample(points,
                                         features=features,
                                         classes=labels,
                                         sampleDl=sampleDl,
                                         verbose=verbose)


def batch_grid_subsampling(
        points,
        batches_len,
        features=None,
        labels=None,
        sampleDl=0.1,
        max_p=0,
        verbose=0
):
    """
    CPP wrapper for a grid subsampling (method = barycenter for points and features)
    """
    if features is None and labels is None:
        s_points, s_len = cpp_subsampling.subsample_batch(
            points, batches_len, sampleDl=sampleDl, max_p=max_p, verbose=verbose
        )
        return s_points, s_len
    elif labels is None:
        s_points, s_len, s_features = cpp_subsampling.subsample_batch(
            points, batches_len, features=features, sampleDl=sampleDl, max_p=max_p, verbose=verbose
        )
        return s_points,s_len, s_features
    elif features is None:
        s_points, s_len, s_labels = cpp_subsampling.subsample_batch(
            points, batches_len, classes=labels, sampleDl=sampleDl, max_p=max_p, verbose=verbose
        )
        return s_points,s_len,s_labels
    else:
        s_points, s_len, s_features, s_labels = cpp_subsampling.subsample_batch(
            points, batches_len, features=features, classes=labels, sampleDl=sampleDl, max_p=max_p, verbose=verbose
        )
        return s_points, s_len, s_features, s_labels

def batch_neighbors(queries, supports, q_batches, s_batches, radius, max_neighbors):
        """
        Computes neighbors for a batch of queries and supports
        :param queries: (N1, 3) the query points
        :param supports: (N2, 3) the support points
        :param q_batches: (B) the list of lengths of batch elements in queries
        :param s_batches: (B)the list of lengths of batch elements in supports
        :param radius: float32
        :return: neighbors indices
        """
        neighbors = cpp_neighbors.batch_query(queries, supports, q_batches, s_batches, radius=radius)
        if max_neighbors > 0:
            neighbors = neighbors[:, :max_neighbors]
        return neighbors


class JittorThreeDMatchPairDataset(Dataset):
    def __init__(
        self,
        config,
        split_data,
        dataset_root,
        subset,
        neighbor_limits=None,
        point_limit=None,
        use_augmentation=False,
        augmentation_noise=0.005,
        augmentation_rotation=1,
        overlap_threshold=None,
        return_corr_indices=False,
        matching_radius=None,
        rotated=False,
    ):
        super(JittorThreeDMatchPairDataset, self).__init__()

        self.config=config

        self.dataset_root = dataset_root
        self.metadata_root = osp.join(self.dataset_root, 'metadata')
        self.data_root = osp.join(self.dataset_root, 'data')
        self.neighbor_limits=neighbor_limits

        self.subset = subset
        self.point_limit = point_limit
        self.overlap_threshold = overlap_threshold
        self.rotated = rotated

        self.return_corr_indices = return_corr_indices
        self.matching_radius = matching_radius
        if self.return_corr_indices and self.matching_radius is None:
            raise ValueError('"matching_radius" is None but "return_corr_indices" is set.')

        self.use_augmentation = use_augmentation
        self.aug_noise = augmentation_noise
        self.aug_rotation = augmentation_rotation
        
        with open(osp.join(self.metadata_root, f'{subset}.pkl'), 'rb') as f:
            self.metadata_list = pickle.load(f)
            if self.overlap_threshold is not None:
                self.metadata_list = [x for x in self.metadata_list if x['overlap'] > self.overlap_threshold]
        self.num=len(self.metadata_list)
        self.start=0
        """ if split_data:
            self.num=11000
            self.start=0
        else:
            self.num=len(self.metadata_list)-11000
            self.start=11000 """

    def __len__(self):
        return self.num

    def _load_point_cloud(self, file_name):
        points = torch.load(osp.join(self.data_root, file_name))
        # NOTE: setting "point_limit" with "num_workers" > 1 will cause nondeterminism.
        if self.point_limit is not None and points.shape[0] > self.point_limit:
            indices = np.random.permutation(points.shape[0])[: self.point_limit]
            points = points[indices]
        return points

    def _augment_point_cloud(self, ref_points, src_points, rotation, translation):
        r"""Augment point clouds.

        ref_points = src_points @ rotation.T + translation

        1. Random rotation to one point cloud.
        2. Random noise.
        """
        print('jinru')
        aug_rotation = random_sample_rotation(self.aug_rotation)
        print('random_sample_rotation')
        if random.random() > 0.5:
            print('random')
            ref_points = np.matmul(ref_points, aug_rotation.T)
            rotation = np.matmul(aug_rotation, rotation)
            translation = np.matmul(aug_rotation, translation)
            print('matmul')
        else:
            print('random')
            src_points = np.matmul(src_points, aug_rotation.T)
            rotation = np.matmul(rotation, aug_rotation.T)
            print('matmul')
        ref_points += (np.random.rand(ref_points.shape[0], 3) - 0.5) * self.aug_noise
        src_points += (np.random.rand(src_points.shape[0], 3) - 0.5) * self.aug_noise
        print('aug_noise')
        return ref_points, src_points, rotation, translation

    def __getitem__(self, index):
        data_dict = {}
        # metadata
        metadata: Dict = self.metadata_list[index+self.start]#+16898
        data_dict['scene_name'] = metadata['scene_name']
        data_dict['ref_frame'] = metadata['frag_id0']
        data_dict['src_frame'] = metadata['frag_id1']
        data_dict['overlap'] = metadata['overlap']

        # get transformation
        rotation = metadata['rotation']
        translation = metadata['translation']

        # get point cloud
        ref_points = self._load_point_cloud(metadata['pcd0'])
        src_points = self._load_point_cloud(metadata['pcd1'])
        # augmentation
        """ if self.use_augmentation:
            ref_points, src_points, rotation, translation = self._augment_point_cloud(
                ref_points, src_points, rotation, translation
            )
        print('use_augmentation') """
        if self.rotated:
            ref_rotation = random_sample_rotation_v2()
            ref_points = np.matmul(ref_points, ref_rotation.T)
            rotation = np.matmul(ref_rotation, rotation)
            translation = np.matmul(ref_rotation, translation)

            src_rotation = random_sample_rotation_v2()
            src_points = np.matmul(src_points, src_rotation.T)
            rotation = np.matmul(rotation, src_rotation.T)
        transform = get_transform_from_rotation_translation(rotation, translation)

        # get correspondences
        if self.return_corr_indices:
            corr_indices = get_correspondences(ref_points, src_points, transform, self.matching_radius)
            data_dict['corr_indices'] = corr_indices

        data_dict['transform'] = transform.astype(np.float32)
        data_dict['features']=np.concatenate([np.ones((ref_points.shape[0], 1), dtype=np.float32) , np.ones((src_points.shape[0], 1), dtype=np.float32)], axis=0)

        stacked_points = np.concatenate([ ref_points.astype(np.float32) , src_points.astype(np.float32)], axis=0)
        stacked_lengths =np.array([ref_points.shape[0], src_points.shape[0]])
        stacked_lengths=stacked_lengths.astype(np.int32)
        input_dict= self.kpconv_inputs(stacked_points, stacked_lengths)
        data_dict.update(input_dict)
        del input_dict,stacked_points,stacked_lengths
        return data_dict
    
    def kpconv_inputs(self,
                              stacked_points,
                              stack_lengths):

        # Starting radius of convolutions
        r_normal = self.config.backbone.init_voxel_size * self.config.backbone.base_radius

        # Starting layer
        layer_blocks = []
        layer = 0

        # Lists of inputs
        input_points = []
        input_neighbors = []
        input_pools = []
        input_stack_lengths = []
        input_upsamples = []

        ######################
        # Loop over the blocks
        ######################

        arch = self.config.backbone.architecture

        for block_i, block in enumerate(arch):

            # Get all blocks of the layer
            if 'global' in block or 'upsample' in block:
                break
            if not ('pool' in block or 'strided' in block):
                layer_blocks += [block]
                if block_i < len(self.config.backbone.architecture) - 1 and not ('upsample' in self.config.backbone.architecture[block_i + 1]):
                    continue

            # Convolution neighbors indices
            # *****************************

            if layer_blocks:
                # Convolutions are done in this layer, compute the neighbors with the good radius
                if np.any(['deformable' in blck for blck in layer_blocks[:-1]]):
                    r = r_normal * self.config.backbone.base_sigma / self.config.backbone.base_radius
                else:
                    r = r_normal
                conv_i = batch_neighbors(stacked_points, stacked_points, stack_lengths, stack_lengths, r,self.neighbor_limits[layer])

            else:
                # This layer only perform pooling, no neighbors required
                conv_i = np.zeros((0, 1), dtype=np.int32)

            # Pooling neighbors indices
            # *************************

            # If end of layer is a pooling operation
            if 'pool' in block or 'strided' in block:

                # New subsampling length
                dl = 2 * r_normal / self.config.backbone.base_radius

                # Subsampled points
                pool_p, pool_b = batch_grid_subsampling(stacked_points, stack_lengths, sampleDl=dl)

                # Radius of pooled neighbors
                if 'deformable' in block:
                    r = r_normal *  self.config.backbone.base_sigma / self.config.backbone.base_radius
                else:
                    r = r_normal

                # Subsample indices
                pool_i = batch_neighbors(pool_p, stacked_points, pool_b, stack_lengths, r, self.neighbor_limits[layer])
                up_i = batch_neighbors(
                stacked_points, pool_p, stack_lengths, pool_b, 2 * r, self.neighbor_limits[layer]
            )

            else:
                # No pooling in the end of this layer, no pooling indices required
                pool_i = np.zeros((0, 1), dtype=np.int32)
                pool_p = np.zeros((0, 1), dtype=np.float32)
                pool_b = np.zeros((0,), dtype=np.int32)
                up_i = np.zeros((0, 1), dtype=np.int64)


            # Updating input lists
            
            input_points += [stacked_points]
            input_neighbors += [conv_i.astype(np.int64)]
            input_pools += [pool_i.astype(np.int64)]
            input_upsamples += [up_i.astype(np.int64)]
            input_stack_lengths += [stack_lengths]

            # New points for next layer
            stacked_points = pool_p
            stack_lengths = pool_b

            # Update radius and reset blocks
            r_normal *= 2
            layer += 1
            layer_blocks = []
        return {
        'points': input_points,
        'lengths': input_stack_lengths,
        'neighbors': input_neighbors,
        'subsampling': input_pools,
        'upsampling': input_upsamples,
        }



class Neighbor_limits(Dataset):
    def __init__(
        self,
        config,
        dataset_root,
        subset,
        point_limit=None,
        use_augmentation=False,
        augmentation_noise=0.005,
        augmentation_rotation=1,
        overlap_threshold=None,
        return_corr_indices=False,
        matching_radius=None,
        rotated=False,
    ):
        super(Neighbor_limits, self).__init__()

        self.config=config

        self.dataset_root = dataset_root
        self.metadata_root = osp.join(self.dataset_root, 'metadata')
        self.data_root = osp.join(self.dataset_root, 'data')

        self.subset = subset
        self.point_limit = point_limit
        self.overlap_threshold = overlap_threshold
        self.rotated = rotated
        
        self.start=np.random

        self.return_corr_indices = return_corr_indices
        self.matching_radius = matching_radius
        if self.return_corr_indices and self.matching_radius is None:
            raise ValueError('"matching_radius" is None but "return_corr_indices" is set.')

        self.use_augmentation = use_augmentation
        self.aug_noise = augmentation_noise
        self.aug_rotation = augmentation_rotation

        with open(osp.join(self.metadata_root, f'{subset}.pkl'), 'rb') as f:
            self.metadata_list = pickle.load(f)
            if self.overlap_threshold is not None:
                self.metadata_list = [x for x in self.metadata_list if x['overlap'] > self.overlap_threshold]

    def __len__(self):
        return len(self.metadata_list)

    def _load_point_cloud(self, file_name):
        points = torch.load(osp.join(self.data_root, file_name))
        # NOTE: setting "point_limit" with "num_workers" > 1 will cause nondeterminism.
        if self.point_limit is not None and points.shape[0] > self.point_limit:
            indices = np.random.permutation(points.shape[0])[: self.point_limit]
            points = points[indices]
        return points

    def _augment_point_cloud(self, ref_points, src_points, rotation, translation):
        r"""Augment point clouds.

        ref_points = src_points @ rotation.T + translation

        1. Random rotation to one point cloud.
        2. Random noise.
        """
        aug_rotation = random_sample_rotation(self.aug_rotation)
        if random.random() > 0.5:
            ref_points = np.matmul(ref_points, aug_rotation.T)
            rotation = np.matmul(aug_rotation, rotation)
            translation = np.matmul(aug_rotation, translation)
        else:
            src_points = np.matmul(src_points, aug_rotation.T)
            rotation = np.matmul(rotation, aug_rotation.T)

        ref_points += (np.random.rand(ref_points.shape[0], 3) - 0.5) * self.aug_noise
        src_points += (np.random.rand(src_points.shape[0], 3) - 0.5) * self.aug_noise

        return ref_points, src_points, rotation, translation

    def __getitem__(self, index):
        data_dict = {}

        # metadata
        metadata: Dict = self.metadata_list[index]
        data_dict['scene_name'] = metadata['scene_name']
        data_dict['ref_frame'] = metadata['frag_id0']
        data_dict['src_frame'] = metadata['frag_id1']
        data_dict['overlap'] = metadata['overlap']

        # get transformation
        rotation = metadata['rotation']
        translation = metadata['translation']

        # get point cloud
        ref_points = self._load_point_cloud(metadata['pcd0'])
        src_points = self._load_point_cloud(metadata['pcd1'])

        # augmentation
        if self.use_augmentation:
            ref_points, src_points, rotation, translation = self._augment_point_cloud(
                ref_points, src_points, rotation, translation
            )

        if self.rotated:
            ref_rotation = random_sample_rotation_v2()
            ref_points = np.matmul(ref_points, ref_rotation.T)
            rotation = np.matmul(ref_rotation, rotation)
            translation = np.matmul(ref_rotation, translation)

            src_rotation = random_sample_rotation_v2()
            src_points = np.matmul(src_points, src_rotation.T)
            rotation = np.matmul(rotation, src_rotation.T)

        transform = get_transform_from_rotation_translation(rotation, translation)

        # get correspondences
        if self.return_corr_indices:
            corr_indices = get_correspondences(ref_points, src_points, transform, self.matching_radius)
            data_dict['corr_indices'] = corr_indices

        data_dict['ref_points'] = ref_points.astype(np.float32)
        data_dict['src_points'] = src_points.astype(np.float32)
        data_dict['ref_feats'] = np.ones((ref_points.shape[0], 1), dtype=np.float32)
        data_dict['src_feats'] = np.ones((src_points.shape[0], 1), dtype=np.float32)
        data_dict['transform'] = transform.astype(np.float32)
        return data_dict

    
