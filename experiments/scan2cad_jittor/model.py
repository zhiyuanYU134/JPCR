import jittor as jt
import jittor.nn as nn
from jittor.contrib import concat
from jittor import linalg
from IPython import embed
from PCR_Jittor.jittor.modules.ops import pairwise_distance,apply_transform
from PCR_Jittor.jittor.modules.ops import point_to_node_partition, index_select
from PCR_Jittor.jittor.modules.registration import get_node_correspondences
from PCR_Jittor.jittor.modules.sinkhorn import LearnableLogOptimalTransport
from PCR_Jittor.jittor.modules.geotransformer import (
    GeometricTransformer,
    CoarseMaskTransformer2,
    CoarseMatching,
    CoarseTargetGenerator,
    LocalGlobalRegistration_jittor
)

from PCR_Jittor.modules.geotransformer import (
    InstanceAwarePointMatching
)
from backbone_jittor import KPConvFPN
import random
import numpy as np
import torch

def to_tensor(array):
    """
    Convert array to tensor
    """
    if(not isinstance(array,torch.Tensor)):
        return torch.from_numpy(array).float()
    else:
        return array
def get_point2trans_index(ref_points,src_points,trans,dist_thre):
    point2trans_index=jt.full([len(ref_points)],-1)
    min_dists=jt.ones([len(trans),len(ref_points)])
    for i in range(len(trans)):
        tran=trans[i]
        src_points_tran=apply_transform(src_points,tran)
        dist= pairwise_distance(ref_points, src_points_tran)
        min_dists[i,:]=jt.argmin(dist,dim=1)[1]
    min_dists_indices,min_dist=jt.argmin(min_dists,dim=0)
    min_dist_mask=jt.less(min_dist,dist_thre*dist_thre)
    point2trans_index[min_dist_mask]=min_dists_indices[min_dist_mask]
    return point2trans_index

@jt.no_grad()
def farthest_point_sample(data,npoints):

    N,D = data.shape #N是点数，D是维度
    xyz = data[:,:3] #只需要坐标
    centroids = jt.zeros((npoints,))#最终的采样点index
    dictance = jt.ones((N,))*1e10 #距离列表,一开始设置的足够大,保证第一轮肯定能更新dictance
    farthest = jt.randint(low=0,high=N,shape=(1,))#随机选一个采样点的index
    for i in range(npoints):
        centroids[i] = farthest
        centroid = xyz[farthest,:]
        dict = ((xyz-centroid)**2).sum(dim=-1)
        mask = dict < dictance
        dictance[mask] = dict[mask]
        farthest = jt.argmax(dictance,dim=-1)[0]

    #data= data[centroids.type(torch.long)]
    return centroids.long()


class MIRETR(nn.Module):
    def __init__(self, cfg):
        super(MIRETR, self).__init__()
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

        self.transformer = CoarseMaskTransformer2(
            cfg.geotransformer.input_dim,
            cfg.geotransformer.output_dim,
            cfg.geotransformer.hidden_dim,
            cfg.geotransformer.num_heads,
            cfg.geotransformer.blocks,
            cfg.geotransformer.sigma_d,
            cfg.geotransformer.sigma_a,
            cfg.geotransformer.angle_k,
            cfg.max_neighboor,
            cfg.geodesic_radis
        )

        self.coarse_target = CoarseTargetGenerator(
            cfg.coarse_matching.num_targets, cfg.coarse_matching.overlap_threshold
        )

        self.coarse_matching = CoarseMatching(
            cfg.coarse_matching.num_correspondences, cfg.coarse_matching.dual_normalization
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

        self.fine_matching = InstanceAwarePointMatching(
            cfg.cluster_thre,
            cfg.cluster_refine,
            cfg.fine_matching_max_num_corr,
            cfg.fine_matching.topk,
            mutual=cfg.fine_matching.mutual,
            with_slack=cfg.fine_matching.use_dustbin,
            threshold=cfg.fine_matching.confidence_threshold,
            conditional=cfg.fine_matching.use_global_score,
            matching_radius=cfg.fine_matching.acceptance_radius,
            min_num_corr=cfg.fine_matching.correspondence_threshold,
            num_registration_iter=cfg.fine_matching.num_refinement_steps,
        )

        self.optimal_transport = LearnableLogOptimalTransport(cfg.model.num_sinkhorn_iterations)
        self.instance_mask_thre=cfg.instance_mask_thre
        self.eval_iou=False
        self.finematch_max_point=cfg.finematch_max_point
        self.max_neighboor=cfg.max_neighboor
        self.max_sample_neighboor=cfg.max_sample_neighboor
        self.final_feats_dim = cfg.backbone.output_dim
        self.pos_radius=cfg.model.ground_truth_matching_radius
        self.training=False
    def execute(self, data_dict):
        output_dict = {}

        # Downsample point clouds
        feats = data_dict['features'].detach()
        transforms= data_dict['transform'].detach()
        transform=transforms[0]

        ref_length_c = data_dict['lengths'][-1][0].item()
        ref_length_f = data_dict['lengths'][1][0].item()
        ref_length = data_dict['lengths'][0][0].item()
        points_c = data_dict['points'][-1].detach()
        points_f = data_dict['points'][1].detach()
        points = data_dict['points'][0].detach()
        sym=data_dict['sym']

        ref_points_c = points_c[:ref_length_c]
        src_points_c = points_c[ref_length_c:]
        ref_points_f = points_f[:ref_length_f]
        src_points_f = points_f[ref_length_f:]
        ref_points = points[:ref_length]
        src_points = points[ref_length:]

        output_dict['ref_points_c'] = ref_points_c
        output_dict['src_points_c'] = src_points_c
        output_dict['ref_points_f'] = ref_points_f
        output_dict['src_points_f'] = src_points_f
        output_dict['ref_points'] = ref_points
        output_dict['src_points'] = src_points

        # 1. Generate ground truth node correspondences
        _, ref_node_masks, ref_node_knn_indices, ref_node_knn_masks = point_to_node_partition(
            ref_points_f, ref_points_c, self.num_points_in_patch
        )
        _, src_node_masks, src_node_knn_indices, src_node_knn_masks = point_to_node_partition(
            src_points_f, src_points_c, self.num_points_in_patch
        )
        

        ref_padded_points_f = concat([ref_points_f, jt.zeros_like(ref_points_f[:1])], 0)
        src_padded_points_f = concat([src_points_f, jt.zeros_like(src_points_f[:1])], 0)
        ref_node_knn_points = index_select(ref_padded_points_f, ref_node_knn_indices, dim=0)
        src_node_knn_points = index_select(src_padded_points_f, src_node_knn_indices, dim=0)
        point2trans_index=get_point2trans_index(ref_points_c,src_points_c,transforms,self.pos_radius*4)
        src_R=pairwise_distance(src_points_f,src_points_f).max()  
        gt_node_corr_indices, gt_node_corr_overlaps = get_node_correspondences(
            ref_points_c,
            src_points_c,
            ref_node_knn_points,
            src_node_knn_points,
            transform,
            self.matching_radius,
            ref_masks=ref_node_masks,
            src_masks=src_node_masks,
            ref_knn_masks=ref_node_knn_masks,
            src_knn_masks=src_node_knn_masks,
        )
        for i in range(len(transforms)-1):
            transform=transforms[i+1]

            gt_node_corr_indices_tmp, gt_node_corr_overlaps_tmp = get_node_correspondences(
                ref_points_c,
                src_points_c,
                ref_node_knn_points,
                src_node_knn_points,
                transform,
                self.matching_radius,
                ref_masks=ref_node_masks,
                src_masks=src_node_masks,
                ref_knn_masks=ref_node_knn_masks,
                src_knn_masks=src_node_knn_masks,
            )
            gt_node_corr_indices=concat([gt_node_corr_indices,gt_node_corr_indices_tmp], 0)
            gt_node_corr_overlaps=concat([gt_node_corr_overlaps,gt_node_corr_overlaps_tmp], 0)
        gt_node_corr_indices_key,index,couts=jt.unique(gt_node_corr_indices, return_inverse=True,return_counts=True,dim=0)
        gt_node_corr_overlaps_new=jt.zeros(len(gt_node_corr_indices_key))
        for i in range(len(index)):
            gt_node_corr_overlaps_new[index[i]]=gt_node_corr_overlaps[i]
        gt_node_corr_indices=gt_node_corr_indices_key
        gt_node_corr_overlaps=gt_node_corr_overlaps_new
        output_dict['gt_node_corr_indices'] = gt_node_corr_indices
        output_dict['gt_node_corr_overlaps'] = gt_node_corr_overlaps


        # 2. KPFCNN Encoder
        feats_list = self.backbone(feats, data_dict)

        feats_c = feats_list[-1]
        feats_f = feats_list[0]

        # 3. Conditional Transformer
        ref_feats_c = feats_c[:ref_length_c]
        src_feats_c = feats_c[ref_length_c:]
        # 5. Head for fine level matching
        ref_feats_f = feats_f[:ref_length_f]
        src_feats_f = feats_f[ref_length_f:]
        output_dict['ref_feats_f'] = ref_feats_f
        output_dict['src_feats_f'] = src_feats_f
        ref_feats_node, src_feats_node ,ref_node_neighbor_indices,src_node_neighbor_indices,geo_dist,pred_masks_list,attn_masks_list, mask_attention_score_list= self.transformer(
            ref_points_c, src_points_c, ref_feats_c, src_feats_c,jt.sqrt(src_R),point2trans_index,gt_corr_indices=gt_node_corr_indices
        )
        ref_feats_node_norm = jt.normalize(ref_feats_node, p=2, dim=1)
        src_feats_node_norm = jt.normalize(src_feats_node, p=2, dim=1)

        output_dict['ref_feats_c'] = ref_feats_node_norm
        output_dict['src_feats_c'] = src_feats_node_norm
        with jt.no_grad():         
            # 7 Random select ground truth node correspondences during training
            if self.training:
                gt_node_corr_trans_index=point2trans_index[ref_node_neighbor_indices]#(num_proposal, max_point)
                sample_proposal_tran_index=point2trans_index
                sample_proposal_tran_index=sample_proposal_tran_index.unsqueeze(1)
                sample_proposal_tran_index=sample_proposal_tran_index.expand(sample_proposal_tran_index.shape[-2],ref_node_neighbor_indices.shape[-1])
                gt_masks=jt.equal(gt_node_corr_trans_index,sample_proposal_tran_index)

                ref_node_corr_indices, src_node_corr_indices, node_corr_scores,sel_indices,gt_corr_masks = self.coarse_target(
                        gt_node_corr_indices, gt_node_corr_overlaps
                    )#num_proposal

                ref_seed_neighbor_indices=ref_node_neighbor_indices[ref_node_corr_indices]
                src_seed_neighbor_indices=src_node_neighbor_indices[src_node_corr_indices]# (num_proposal, max_point)
                
                ref_node_neighbor_mask = gt_masks[ref_node_corr_indices]# (num_proposal, max_point)
                src_node_neighbor_num=ref_node_neighbor_mask.sum(1)
                src_node_neighbor_mask=jt.zeros_like(src_seed_neighbor_indices).bool()
                for i in range(len(src_node_neighbor_mask)):
                    src_node_neighbor_mask[i][:src_node_neighbor_num[i]]=True

                output_dict['pred_masks_list']=pred_masks_list
                output_dict['attn_masks_list']=attn_masks_list
                output_dict['gt_masks']=gt_masks
                output_dict['ref_node_corr_indices']=ref_node_corr_indices
                output_dict['src_node_corr_indices']=src_node_corr_indices
                output_dict['ref_node_neighbor_indices']=ref_node_neighbor_indices
                output_dict['src_node_neighbor_indices']=src_node_neighbor_indices
            else :
                ref_node_corr_indices, src_node_corr_indices, node_corr_scores = self.coarse_matching(
                    ref_feats_node_norm, src_feats_node_norm
                )
                ref_seed_neighbor_indices=ref_node_neighbor_indices[ref_node_corr_indices]
                src_seed_neighbor_indices=src_node_neighbor_indices[src_node_corr_indices]# (num_proposal, max_point)

                ref_node_neighbor_mask=pred_masks_list[-1]
                ref_node_neighbor_mask = (ref_node_neighbor_mask.sigmoid() > self.instance_mask_thre).bool()# (num_proposal, max_point)
                ref_node_neighbor_mask=ref_node_neighbor_mask[ref_node_corr_indices]
                src_node_neighbor_num=ref_node_neighbor_mask.sum(1)
                src_node_neighbor_mask=jt.zeros_like(src_seed_neighbor_indices).bool()
                for i in range(len(src_node_neighbor_mask)):
                    src_node_neighbor_mask[i][:src_node_neighbor_num[i]]=True
                output_dict['ref_node_knn_points']=ref_node_knn_points
                output_dict['src_node_knn_points']=src_node_knn_points
                output_dict['node_knn_indices']=ref_seed_neighbor_indices
                output_dict['pred_masks_list']=pred_masks_list
                output_dict['attn_masks_list']=attn_masks_list
                output_dict['geo_dist']=geo_dist
                output_dict['mask_attention_score_list']=mask_attention_score_list

        # 7.2 Generate batched node points & feats
        output_dict['ref_node_corr_indices']=ref_node_corr_indices
        output_dict['src_node_corr_indices']=src_node_corr_indices
        output_dict['ref_node_neighbor_indices']=ref_node_neighbor_indices
        output_dict['src_node_neighbor_indices']=src_node_neighbor_indices

        ref_node_neighbor_mask=ref_node_neighbor_mask.unsqueeze(-1).expand(ref_node_neighbor_mask.shape[0], ref_node_neighbor_mask.shape[1],ref_node_knn_masks.shape[1])# (num_proposal, max_point, K)
        ref_node_corr_knn_masks = ref_node_knn_masks[ref_seed_neighbor_indices]  # (num_proposal, max_point, K)
        src_node_corr_knn_masks = src_node_knn_masks[src_seed_neighbor_indices]  # (num_proposal, max_point, K)
        ref_node_corr_knn_points = ref_node_knn_points[ref_seed_neighbor_indices]  # (num_proposal, max_point, K,3)
        src_node_corr_knn_points = src_node_knn_points[src_seed_neighbor_indices]  # (num_proposal, max_point, K,3)
        ref_node_corr_knn_indices = ref_node_knn_indices[ref_seed_neighbor_indices]  # (num_proposal, max_point, K)
        src_node_corr_knn_indices= src_node_knn_indices[src_seed_neighbor_indices]  # (num_proposal, max_point, K)
        
        
        ref_neighbor_corr_knn_feats = jt.zeros((len(ref_node_neighbor_mask),self.finematch_max_point, self.final_feats_dim))
        src_neighbor_corr_knn_feats = jt.zeros((len(ref_node_neighbor_mask),self.finematch_max_point, self.final_feats_dim))
        ref_neighbor_corr_knn_points = jt.zeros((len(ref_node_neighbor_mask),self.finematch_max_point, 3))
        src_neighbor_corr_knn_points = jt.zeros((len(ref_node_neighbor_mask),self.finematch_max_point, 3))
        ref_neighbor_knn_masks=jt.full((len(ref_node_neighbor_mask),self.finematch_max_point),False).bool()    
        src_neighbor_knn_masks=jt.full((len(ref_node_neighbor_mask),self.finematch_max_point),False).bool()    
        sentinel_feat = jt.zeros((1, self.final_feats_dim))
        ref_padded_feats_f = jt.cat([ref_feats_f, sentinel_feat], dim=0)
        src_padded_feats_f = jt.cat([src_feats_f, sentinel_feat], dim=0)

        ref_node_corr_knn_feats = ref_padded_feats_f[ref_node_corr_knn_indices]  # (num_proposal, max_point, K, C)
        src_node_corr_knn_feats = src_padded_feats_f[src_node_corr_knn_indices]  # (num_proposal, max_point, K, C)

        ref_node_corr_knn_masks=jt.logical_and(ref_node_corr_knn_masks,ref_node_neighbor_mask)

        for i in range(len(ref_node_neighbor_mask)):
            ref_tmp_points=ref_node_corr_knn_points[i]
            src_tmp_points=src_node_corr_knn_points[i]
            ref_tmp_masks=ref_node_corr_knn_masks[i]
            src_tmp_masks=src_node_corr_knn_masks[i]
            ref_tmp_feats=ref_node_corr_knn_feats[i]
            src_tmp_feats=src_node_corr_knn_feats[i]


            ref_tmp_points=ref_tmp_points.reshape(-1,ref_tmp_points.shape[-1])
            src_tmp_points=src_tmp_points.reshape(-1,src_tmp_points.shape[-1])
            ref_tmp_feats=ref_tmp_feats.reshape(-1,ref_tmp_feats.shape[-1])
            src_tmp_feats=src_tmp_feats.reshape(-1,src_tmp_feats.shape[-1])
            ref_tmp_points=ref_tmp_points[ref_tmp_masks.reshape(-1)]
            src_tmp_points=src_tmp_points[src_tmp_masks.reshape(-1)]
            ref_tmp_feats=ref_tmp_feats[ref_tmp_masks.reshape(-1)]
            src_tmp_feats=src_tmp_feats[src_tmp_masks.reshape(-1)]
            if len(ref_tmp_points)>=self.finematch_max_point:
                #inds = jt.LongTensor(random.sample(range(len(ref_tmp_points)), self.finematch_max_point))
                inds = np.random.choice(range(len(ref_tmp_points)), self.finematch_max_point, replace=False)
                inds=jt.Var(inds)
                ref_neighbor_corr_knn_feats[i] =ref_tmp_feats[inds] 
                ref_neighbor_corr_knn_points[i]=ref_tmp_points[inds] 
                ref_neighbor_knn_masks[i,:]=True

            else:
                ref_neighbor_corr_knn_feats[i,:len(ref_tmp_points)]=ref_tmp_feats
                ref_neighbor_corr_knn_points[i,:len(ref_tmp_points)]=ref_tmp_points
                ref_neighbor_knn_masks[i,:len(ref_tmp_points)]=True

            if len(src_tmp_points)>=self.finematch_max_point:
                #inds = torch.LongTensor(random.sample(range(len(src_tmp_points)), self.finematch_max_point)).cuda()
                inds = np.random.choice(range(len(src_tmp_points)), self.finematch_max_point, replace=False)
                inds=jt.Var(inds)
                src_neighbor_corr_knn_feats[i] =src_tmp_feats[inds] 
                src_neighbor_corr_knn_points[i]=src_tmp_points[inds] 
                src_neighbor_knn_masks[i,:]=True
            else:
                src_neighbor_corr_knn_feats[i,:len(src_tmp_points)]=src_tmp_feats
                src_neighbor_corr_knn_points[i,:len(src_tmp_points)]=src_tmp_points
                src_neighbor_knn_masks[i,:len(src_tmp_points)]=True

        # 8. Optimal transport

        matching_scores = linalg.einsum('bnd,bmd->bnm', ref_neighbor_corr_knn_feats, src_neighbor_corr_knn_feats)  # (P, K, K)
        matching_scores = matching_scores / self.final_feats_dim ** 0.5
        matching_scores = self.optimal_transport(matching_scores, ref_neighbor_knn_masks, src_neighbor_knn_masks)
        output_dict['matching_scores'] = matching_scores
        output_dict['ref_node_corr_knn_points']=ref_neighbor_corr_knn_points
        output_dict['src_node_corr_knn_points']=src_neighbor_corr_knn_points
        output_dict['ref_node_corr_knn_masks']=ref_neighbor_knn_masks
        output_dict['src_node_corr_knn_masks']=src_neighbor_knn_masks
        output_dict['ref_node_corr_knn_feats']=ref_neighbor_corr_knn_feats
        output_dict['src_node_corr_knn_feats']=src_neighbor_corr_knn_feats
        # 9. Generate final correspondences during testing
        """ with torch.no_grad():
            if not self.fine_matching.use_dustbin:
                matching_scores = matching_scores[:, :-1, :-1]

            ref_corr_points, src_corr_points, corr_scores= self.fine_matching(
                ref_node_corr_knn_points,
                src_node_corr_knn_points,
                ref_node_corr_knn_masks,
                src_node_corr_knn_masks,
                matching_scores,
                node_corr_scores,
            )#, estimated_transform 

            output_dict['ref_corr_points'] = ref_corr_points
            output_dict['src_corr_points'] = src_corr_points
            output_dict['corr_scores'] = corr_scores
            #output_dict['estimated_transform'] = estimated_transform """
        
        if not self.training:
            if not self.fine_matching.with_slack:
                matching_scores = matching_scores[:, :-1, :-1]
            ref_neighbor_corr_knn_points=to_tensor(ref_neighbor_corr_knn_points.cpu().numpy()).cuda()
            src_neighbor_corr_knn_points=to_tensor(src_neighbor_corr_knn_points.cpu().numpy()).cuda()
            ref_neighbor_knn_masks=torch.from_numpy(ref_neighbor_knn_masks.cpu().numpy()).cuda()
            src_neighbor_knn_masks=torch.from_numpy(src_neighbor_knn_masks.cpu().numpy()).cuda()
            matching_scores=to_tensor(matching_scores.cpu().numpy()).cuda()
            node_corr_scores=to_tensor(node_corr_scores.cpu().numpy()).cuda()
            ref_points_f=to_tensor(ref_points_f.cpu().numpy()).cuda()
            src_points_f=to_tensor(src_points_f.cpu().numpy()).cuda()
            estimated_transforms,all_ref_corr_points,all_src_corr_points,all_corr_scores= self.fine_matching(
                    ref_neighbor_corr_knn_points, src_neighbor_corr_knn_points,
                    ref_neighbor_knn_masks, src_neighbor_knn_masks,
                    matching_scores, node_corr_scores,ref_points_f,src_points_f
                )
            estimated_transforms=jt.Var(estimated_transforms.cpu().numpy())
            all_ref_corr_points=jt.Var(all_ref_corr_points.cpu().numpy()) 
            all_src_corr_points=jt.Var(all_src_corr_points.cpu().numpy())
            all_corr_scores=jt.Var(all_corr_scores.cpu().numpy()) 
            output_dict['estimated_transforms'] = estimated_transforms
            output_dict['all_ref_corr_points'] = all_ref_corr_points 
            output_dict['all_src_corr_points'] = all_src_corr_points
            output_dict['all_corr_scores'] = all_corr_scores


        return output_dict


def create_model(config):
    model = MIRETR(config)
    return model


def main():
    from config import make_cfg

    cfg = make_cfg()
    model = create_model(cfg)
    print(model.state_dict().keys())
    print(model)


if __name__ == '__main__':
    main()