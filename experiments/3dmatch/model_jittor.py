from re import T
import jittor as jt
import jittor.nn as nn
from jittor.contrib import concat
from jittor import linalg
from IPython import embed
import torch
from PCR_Jittor.jittor.modules.ops import point_to_node_partition, index_select
from PCR_Jittor.jittor.modules.registration import get_node_correspondences
from PCR_Jittor.jittor.modules.sinkhorn import LearnableLogOptimalTransport
from PCR_Jittor.jittor.modules.geotransformer import (
    GeometricTransformer,
    SuperPointMatching,
    SuperPointTargetGenerator,
    LocalGlobalRegistration_jittor
)

from PCR_Jittor.modules.geotransformer import (
    LocalGlobalRegistration
)

from backbone_jittor import KPConvFPN
import time
import numpy as np
def to_tensor(array):
    """
    Convert array to tensor
    """
    if(not isinstance(array,torch.Tensor)):
        return torch.from_numpy(array).float()
    else:
        return array
class GeoTransformer(nn.Module):
    def __init__(self, cfg):
        super(GeoTransformer, self).__init__()
        self.num_points_in_patch = cfg.model.num_points_in_patch
        self.matching_radius = cfg.model.ground_truth_matching_radius

        self.backbone = KPConvFPN(
            cfg.backbone.input_dim,
            cfg.backbone.output_dim,
            cfg.backbone.init_dim,
            cfg.backbone.kernel_size,
            cfg.backbone.init_radius,
            cfg.backbone.init_sigma,
            cfg.backbone.group_norm,
        )

        self.transformer = GeometricTransformer(
            cfg.geotransformer.input_dim,
            cfg.geotransformer.output_dim,
            cfg.geotransformer.hidden_dim,
            cfg.geotransformer.num_heads,
            cfg.geotransformer.blocks,
            cfg.geotransformer.sigma_d,
            cfg.geotransformer.sigma_a,
            cfg.geotransformer.angle_k,
            reduction_a=cfg.geotransformer.reduction_a,
        )

        self.coarse_target = SuperPointTargetGenerator(
            cfg.coarse_matching.num_targets, cfg.coarse_matching.overlap_threshold
        )

        self.coarse_matching = SuperPointMatching(
            cfg.coarse_matching.num_correspondences, cfg.coarse_matching.dual_normalization
        )

        self.fine_matching = LocalGlobalRegistration(
            cfg.fine_matching.topk,
            cfg.fine_matching.acceptance_radius,
            mutual=cfg.fine_matching.mutual,
            confidence_threshold=cfg.fine_matching.confidence_threshold,
            use_dustbin=cfg.fine_matching.use_dustbin,
            use_global_score=cfg.fine_matching.use_global_score,
            correspondence_threshold=cfg.fine_matching.correspondence_threshold,
            correspondence_limit=cfg.fine_matching.correspondence_limit,
            num_refinement_steps=cfg.fine_matching.num_refinement_steps,
        )

        self.fine_matching_jittor = LocalGlobalRegistration_jittor(
            cfg.fine_matching.topk,
            cfg.fine_matching.acceptance_radius,
            mutual=cfg.fine_matching.mutual,
            confidence_threshold=cfg.fine_matching.confidence_threshold,
            use_dustbin=cfg.fine_matching.use_dustbin,
            use_global_score=cfg.fine_matching.use_global_score,
            correspondence_threshold=cfg.fine_matching.correspondence_threshold,
            correspondence_limit=cfg.fine_matching.correspondence_limit,
            num_refinement_steps=cfg.fine_matching.num_refinement_steps,
        )

        self.optimal_transport = LearnableLogOptimalTransport(cfg.model.num_sinkhorn_iterations)
        self.training=False

    def execute(self, data_dict):
        start_time=time.time() 
        output_dict = {}

        feats = data_dict['features'].detach()
        transform = data_dict['transform'].detach()

        ref_length_c = data_dict['lengths'][-1][0].item()
        ref_length_f = data_dict['lengths'][1][0].item()
        ref_length = data_dict['lengths'][0][0].item()
        points_c = data_dict['points'][-1].detach()
        points_f = data_dict['points'][1].detach()
        points = data_dict['points'][0].detach()

        ref_points_c = points_c[:ref_length_c]
        src_points_c = points_c[ref_length_c:]
        ref_points_f = points_f[:ref_length_f]
        src_points_f = points_f[ref_length_f:]
        ref_points = points[:ref_length]
        src_points= points[ref_length:]

        output_dict['ref_points'] = ref_points
        output_dict['src_points'] = src_points

        output_dict['ref_points_c'] = ref_points_c
        output_dict['src_points_c'] = src_points_c
        output_dict['ref_points_f'] = ref_points_f
        output_dict['src_points_f'] = src_points_f

        """ print(ref_points.shape)
        print(src_points.shape)
        print(ref_points_c.shape)
        print(src_points_c.shape)
        print(ref_points_f.shape)
        print(src_points_f.shape) """
        
        # 1. Generate ground truth node correspondences
        
        with jt.no_grad():
            _, ref_node_masks, ref_node_knn_indices, ref_node_knn_masks = point_to_node_partition(
                ref_points_f, ref_points_c, self.num_points_in_patch
            )
            _, src_node_masks, src_node_knn_indices, src_node_knn_masks = point_to_node_partition(
                src_points_f, src_points_c, self.num_points_in_patch
            )
        
        """ print(ref_node_masks)
        print(ref_node_knn_indices)
        print(ref_node_knn_masks) """
        ref_padded_points_f =concat([ref_points_f, jt.zeros_like(ref_points_f[:1])], 0)
        src_padded_points_f =concat([src_points_f, jt.zeros_like(src_points_f[:1])], 0)
        ref_node_knn_points = index_select(ref_padded_points_f, ref_node_knn_indices, dim=0)
        src_node_knn_points = index_select(src_padded_points_f, src_node_knn_indices, dim=0)

                
        loading_time = time.time() - start_time
        print("Generate_gt_corr",loading_time)
        start_time=time.time() 
        # 2. KPFCNN Encoder
        feats_list = self.backbone(feats, data_dict)

        loading_time = time.time() - start_time
        print("KPFCNN",loading_time)


        return feats_list


def create_model(config):
    model = GeoTransformer(config)
    return model


def main():
    from config import make_cfg
    cfg = make_cfg()
    model = create_model(cfg)



if __name__ == '__main__':
    main()
