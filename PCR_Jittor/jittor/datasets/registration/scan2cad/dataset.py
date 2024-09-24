import os.path as osp
import pickle
import random
from typing import Dict

import numpy as np
import torch
import torch.utils.data
import numpy as np
from jittor.dataset.dataset import Dataset
from pathlib import Path

import pinocchio as pin
import quaternion

from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
import time

import json
from pathlib import Path
from PIL import Image
import os

import cpp_wrappers.cpp_subsampling.grid_subsampling as cpp_subsampling
import cpp_wrappers.cpp_neighbors.radius_neighbors as cpp_neighbors
BASE_DIR = Path(__file__).parent

def pairwise_distance(points0, points1, normalized=False, clamp=False):
    r"""
    [PyTorch/Numpy] Pairwise distance of two point clouds.

    :param points0: torch.Tensor (d0, ..., dn, num_point0, num_feature)
    :param points1: torch.Tensor (d0, ..., dn, num_point1, num_feature)
    :param normalized: bool (default: False)
        If True, the points are normalized, so a2 and b2 both 1. This enables us to use 2 instead of a2 + b2 for
        simplicity.
    :param clamp: bool (default: False)
        If True, all value will be assured to be non-negative.
    :return: dist: torch.Tensor (d0, ..., dn, num_point0, num_point1)
    """
    if isinstance(points0, torch.Tensor):
        if len(points0.shape)==3:
            
            dist2 = (points0.unsqueeze(2)-points1.unsqueeze(1)).abs().pow(2).sum(-1)
        else:
            dist2 = (points0.unsqueeze(1)-points1.unsqueeze(0)).abs().pow(2).sum(-1)
    else:
        ab = np.matmul(points0, points1.transpose(-1, -2))
        if normalized:
            dist2 = 2 - 2 * ab
        else:
            a2 = np.expand_dims(np.sum(points0 ** 2, axis=-1), axis=-1)
            b2 = np.expand_dims(np.sum(points1 ** 2, axis=-1), axis=-2)
            dist2 = a2 - 2 * ab + b2
        if clamp:
            dist2 = np.maximum(dist2, np.zeros_like(dist2))
    return dist2

def make_M_from_tqs(t, q, s):
    q = np.quaternion(q[0], q[1], q[2], q[3])
    T = np.eye(4)
    T[0:3, 3] = t
    R = np.eye(4)
    R[0:3, 0:3] = quaternion.as_rotation_matrix(q)
    S = np.eye(4)
    S[0:3, 0:3] = np.diag(s)

    M = T.dot(R).dot(S)
    return M 


def calc_Mbbox(model):
    trs_obj = model["trs"]
    bbox_obj = np.asarray(model["bbox"], dtype=np.float64)
    center_obj = np.asarray(model["center"], dtype=np.float64)
    trans_obj = np.asarray(trs_obj["translation"], dtype=np.float64)
    rot_obj = np.asarray(trs_obj["rotation"], dtype=np.float64)
    q_obj = np.quaternion(rot_obj[0], rot_obj[1], rot_obj[2], rot_obj[3])
    scale_obj = np.asarray(trs_obj["scale"], dtype=np.float64)

    tcenter1 = np.eye(4)
    tcenter1[0:3, 3] = center_obj
    trans1 = np.eye(4)
    trans1[0:3, 3] = trans_obj
    rot1 = np.eye(4)
    rot1[0:3, 0:3] = quaternion.as_rotation_matrix(q_obj)
    scale1 = np.eye(4)
    scale1[0:3, 0:3] = np.diag(scale_obj)
    bbox1 = np.eye(4)
    bbox1[0:3, 0:3] = np.diag(bbox_obj)
    M = trans1.dot(rot1).dot(scale1).dot(tcenter1).dot(bbox1)
    return M


def vanish(Mbbox, scan_warped):
    Mbbox_inverse = np.linalg.inv(Mbbox)
    scan_warped_warped = np.dot(Mbbox_inverse, scan_warped.T).T[:, :3]
    idx = ((np.multiply((scan_warped_warped < 1.1), (scan_warped_warped > -1.1))).sum(-1) < 3).nonzero()[0]

    return idx

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

class Scan2cadKPConvDataset(Dataset):
    def __init__(self,
            config,
                 scan2cad_root,
                 split,
                 matching_radius,
                 neighbor_limits=None,
                 max_point=30000,
                 use_augmentation=True,
                 augmentation_noise=0.005,
                 rotation_factor=1,
                 overlap_thresh=None,
                 return_correspondences=True,
                 suffix=None,
                 aligned=False,
                 rotated=False):
        super(Scan2cadKPConvDataset, self).__init__()

        self.scan2cad_root = scan2cad_root
        self.partition = split
        self.matching_radius = matching_radius
        self.neighbor_limits=neighbor_limits
        self.max_point = max_point
        self.return_correspondences = return_correspondences
        self.suffix = suffix
        self.aligned = aligned
        self.rotated = rotated
        self.train_num = 1528
        self.val_num = 218
        self.test_num = 438
        self.config=config
        if self.partition == 'train':
            self.num = self.train_num
            self.start = 0
            self.use_augmentation = True
            self.augmentation_noise = augmentation_noise
            self.rotation_factor = rotation_factor

        elif self.partition == 'val':
            self.num = self.val_num
            self.start = self.train_num
            self.use_augmentation = False

        elif self.partition == 'test':
            self.num = self.test_num
            self.start = self.train_num + self.val_num
            self.use_augmentation = False
        else:
            print('gg')

    def __len__(self):
        return self.num

    def __getitem__(self, id):
        
        # metadata
        frame_id=self.start+id
        dataset=np.load(self.scan2cad_root+ 'data{:05d}.npz'.format(frame_id))
        points0=dataset['points0']
        points1=dataset['points1']
        if self.max_point is not None and points0.shape[0] > self.max_point:
            indices = np.random.permutation(points0.shape[0])[: self.max_point]
            points0 = points0[indices]
        if self.max_point is not None and points1.shape[0] > self.max_point:
            indices = np.random.permutation(points1.shape[0])[: self.max_point]
            points1 = points1[indices]
        Rts=dataset['trans']
        frag_id1=dataset['frag_id1']
        scene_name=dataset['scene_name']
        sym=dataset['sym']
        correspondences=[]
        data_dict = {}
        if sym=="__SYM_NONE":
            data_dict['sym'] = 1
        else:
            data_dict['sym'] = 0

        """ data_dict['scene_name'] = sym
        data_dict['ref_frame'] = frag_id1
        data_dict['src_frame'] = scene_name """
        data_dict['overlap'] = 0.1

        data_dict['features']=np.concatenate([np.ones((points0.shape[0], 1), dtype=np.float32) , np.ones((points1.shape[0], 1), dtype=np.float32)], axis=0)
        data_dict['transform'] =Rts.astype(np.float32)
        stacked_points = np.concatenate([ points0.astype(np.float32) , points1.astype(np.float32)], axis=0)
        stacked_lengths =np.array([points0.shape[0], points1.shape[0]])
        stacked_lengths=stacked_lengths.astype(np.int32)
        start_time=time.time()
        input_dict= self.kpconv_inputs(stacked_points, stacked_lengths)
        data_dict.update(input_dict)
        print("dataset process",time.time()-start_time)
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
    def __init__(self,
            config,
                 scan2cad_root,
                 split,
                 matching_radius,
                 max_point=30000,
                 use_augmentation=True,
                 augmentation_noise=0.005,
                 rotation_factor=1,
                 overlap_thresh=None,
                 return_correspondences=True,
                 suffix=None,
                 aligned=False,
                 rotated=False):
        super(Neighbor_limits, self).__init__()

        self.scan2cad_root = scan2cad_root
        self.partition = split
        self.matching_radius = matching_radius
        
        self.max_point = max_point
        self.return_correspondences = return_correspondences
        self.suffix = suffix
        self.aligned = aligned
        self.rotated = rotated
        self.train_num = 1528
        self.val_num = 218
        self.test_num = 438
        self.config=config
        if self.partition == 'train':
            self.num = self.train_num
            self.start = 0
            self.use_augmentation = True
            self.augmentation_noise = augmentation_noise
            self.rotation_factor = rotation_factor

        elif self.partition == 'val':
            self.num = self.val_num
            self.start = self.train_num
            self.use_augmentation = False

        elif self.partition == 'test':
            self.num = self.test_num
            self.start = self.train_num + self.val_num
            self.use_augmentation = False
        else:
            print('gg')

    def __len__(self):
        return self.num

    def __getitem__(self, id):
        
        # metadata
        frame_id=self.start+id
        dataset=np.load(self.scan2cad_root+ 'data{:05d}.npz'.format(frame_id))
        points0=dataset['points0']
        points1=dataset['points1']
        if self.max_point is not None and points0.shape[0] > self.max_point:
            indices = np.random.permutation(points0.shape[0])[: self.max_point]
            points0 = points0[indices]
        if self.max_point is not None and points1.shape[0] > self.max_point:
            indices = np.random.permutation(points1.shape[0])[: self.max_point]
            points1 = points1[indices]
        Rts=dataset['trans']
        frag_id1=dataset['frag_id1']
        scene_name=dataset['scene_name']
        sym=dataset['sym']
        feats0 = np.ones((points0.shape[0], 1), dtype=np.float32)
        feats1 = np.ones((points1.shape[0], 1), dtype=np.float32)
        correspondences=[]
        data_dict = {}
        if sym=="__SYM_NONE":
            data_dict['sym'] = 1
        else:
            data_dict['sym'] = 0
        data_dict['scene_name'] = Rts.astype(np.float32)
        data_dict['ref_frame'] = Rts.astype(np.float32)
        data_dict['src_frame'] = Rts.astype(np.float32)
        data_dict['overlap'] = 0.1
        data_dict['ref_points'] = points0.astype(np.float32)
        data_dict['src_points'] = points1.astype(np.float32)
        data_dict['ref_feats'] = feats0
        data_dict['src_feats'] = feats1
        data_dict['transform'] =Rts.astype(np.float32)
    
        return data_dict