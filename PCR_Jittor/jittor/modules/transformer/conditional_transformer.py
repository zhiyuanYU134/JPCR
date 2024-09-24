import jittor.nn as nn

from PCR_Jittor.jittor.modules.transformer.rpe_transformer import RPETransformerLayer
from PCR_Jittor.jittor.modules.transformer.vanilla_transformer import TransformerLayer
from jittor import linalg
from jittor.contrib import concat

from PCR_Jittor.jittor.modules.ops import pairwise_distance
import jittor as jt
def _check_block_type(block):
    if block not in ['self', 'cross']:
        raise ValueError('Unsupported block type "{}".'.format(block))



class RPEConditionalTransformer(nn.Module):
    def __init__(
        self,
        blocks,
        d_model,
        num_heads,
        dropout=None,
        activation_fn='ReLU',
        return_attention_scores=False,
        parallel=False,
    ):
        super(RPEConditionalTransformer, self).__init__()
        self.blocks = blocks
        layers = []
        for block in self.blocks:
            _check_block_type(block)
            if block == 'self':
                layers.append(RPETransformerLayer(d_model, num_heads, dropout=dropout, activation_fn=activation_fn))
            else:
                layers.append(TransformerLayer(d_model, num_heads, dropout=dropout, activation_fn=activation_fn))
        self.layers = nn.ModuleList(layers)
        self.return_attention_scores = return_attention_scores
        self.parallel = parallel

    def execute(self, feats0, feats1, embeddings0, embeddings1, masks0=None, masks1=None):
        attention_scores = []
        for i, block in enumerate(self.blocks):
            if block == 'self':
                feats0, scores0 = self.layers[i](feats0, feats0, embeddings0, memory_masks=masks0)
                feats1, scores1 = self.layers[i](feats1, feats1, embeddings1, memory_masks=masks1)
            else:
                if self.parallel:
                    new_feats0, scores0 = self.layers[i](feats0, feats1, memory_masks=None)
                    new_feats1, scores1 = self.layers[i](feats1, feats0, memory_masks=None)
                    feats0 = new_feats0
                    feats1 = new_feats1
                else:
                    feats0, scores0 = self.layers[i](feats0, feats1, memory_masks=None)
                    feats1, scores1 = self.layers[i](feats1, feats0, memory_masks=None)
            if self.return_attention_scores:
                attention_scores.append([scores0, scores1])
        if self.return_attention_scores:
            return feats0, feats1, attention_scores
        else:
            return feats0, feats1



class Mask_RPETransformer2(nn.Module):
    def __init__(self, blocks, d_model, num_head, dropout=0.1, activation_fn='ReLU'):
        super(Mask_RPETransformer2, self).__init__()
        self.blocks = blocks
        layers = []
        for block in self.blocks:
            if block == 'self':
                layers.append(RPETransformerLayer(d_model, num_head, dropout=dropout, activation_fn=activation_fn))
            elif block == 'cross':
                layers.append(TransformerLayer(d_model, num_head, dropout=dropout, activation_fn=activation_fn))
            elif block == 'mask':
                layers.append(RPETransformerLayer(d_model, num_head, dropout=dropout, activation_fn=activation_fn))
            else:
                raise ValueError('Unsupported block type "{}" in `RPEConditionalTransformer`.'.format(block))
        self.layers = nn.ModuleList(layers)
        self.mask_proj = nn.Sequential(
        nn.Linear(2*d_model, d_model),nn.LayerNorm(d_model), nn.ReLU() , 
        nn.Linear(d_model, 1))

        self.out_proj = nn.Sequential(nn.Linear(d_model, d_model),nn.LayerNorm(d_model))


    def prediction_head(self, query, mask_feats,cross_position_embeddings):
        #(num_proposal,1 ,C) ,(num_proposal, max_point, C)
        query=self.out_proj(query)
        all_ref_node_mask_features=concat((mask_feats-query,cross_position_embeddings.squeeze(1)),-1)
        pred_masks=self.mask_proj(all_ref_node_mask_features).squeeze(-1)
        pred_masks=pred_masks.sigmoid() 
        attn_mask = (pred_masks < 0.5).bool()
        attn_mask[jt.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
        attn_mask = attn_mask.detach()
        return pred_masks, attn_mask

    def execute(self, feats0, feats1, embeddings0, embeddings1, cross_position_embeddings,ref_node_knn_indices,src_node_knn_indices, masks0=None, masks1=None):
        #(N, C),(M, C), (N,1, max_point, C), (M,1, max_point, C), (N,1, max_point, C), (N, max_point),(M, max_point), (N, max_point),(M, max_point)
        pred_masks_list, attn_masks_list,mask_attention_score_list=[],[],[]
        for i, block in enumerate(self.blocks):
            if block == 'self':
                #(N, 1，C),(N,max_point,  C), (N,1, max_point, C),(N,1, max_point, C)
                feats0, _ = self.layers[i](feats0.unsqueeze(1), feats0[ref_node_knn_indices], embeddings0, memory_masks=masks0,mask_mode='MIRETR')
                feats1, _ = self.layers[i](feats1.unsqueeze(1), feats1[src_node_knn_indices], embeddings1, memory_masks=masks1,mask_mode='MIRETR')
                feats0=feats0.squeeze(1)
                feats1=feats1.squeeze(1)
                """ print(feats0.shape)
                print(feats1.shape) """
            elif block == 'cross' :
                feats0=feats0.unsqueeze(0)
                feats1=feats1.unsqueeze(0)
                feats0, _ = self.layers[i](feats0, feats1, memory_masks=None)#masks1
                feats1, _ = self.layers[i](feats1, feats0, memory_masks=None)#masks0
                feats0=feats0.squeeze(0)
                feats1=feats1.squeeze(0)
            else:
                #(N,1 ,C) ,(N, max_point, C), (N,1, max_point, C),(N, max_point)
                ref_support_feature, attention_scores = self.layers[i](feats0.unsqueeze(1), feats0[ref_node_knn_indices], cross_position_embeddings,memory_masks=masks0,mask_mode='MIRETR')#masks1
                mask_attention_score_list.append(attention_scores)
                ref_support_feature=ref_support_feature.squeeze(1)
                pred_masks, attn_masks=self.prediction_head(ref_support_feature.unsqueeze(1),ref_support_feature[ref_node_knn_indices],cross_position_embeddings)
                masks0=attn_masks
                pred_masks_list.append(pred_masks)
                attn_masks_list.append(attn_masks)
        return feats0, feats1,pred_masks_list,attn_masks_list,mask_attention_score_list