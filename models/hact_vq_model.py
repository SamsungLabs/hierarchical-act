from typing import Optional

import gin
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .registry import register
from losses.hact_vq_loss import HactVqLoss
from models.module import position_encoding_v2 as position_encoding
from models.module.cnn_backbone import ResnetBackbone
from models.module.act_transformer_v2 import TransformerEncoderAlone, TransformerDecoderAlone, Transformer
from models.module.vq_vae import VQEmbedding
from ops.reparametrize import reparametrize_normal



def get_actions_decoder_mask(len_input):
    # mask : [source, target]
    mask = np.tril(np.ones((len_input, len_input)))
    return torch.FloatTensor(mask)

def get_actions_mask(len_input):
    # mask : [source, target]
    mask = np.tril(np.ones((len_input, len_input)))
    return torch.FloatTensor(mask)


@register('hact_vq')
@gin.configurable()
class HactVqModel(nn.Module):
    def __init__(self,
        cnn_backbone_type='resnet18',
        state_dim=14,
        action_dim=14,
        context_dim=512,
        hidden_dim=512,
        input_image_num=2,
        image_encoder_head_num=8,
        image_encoder_layer_num=4,
        image_encoder_feedforward_dim=2048,
        subgoal_encoder_head_num =8,
        subgoal_encoder_layer_num=4,
        subgoal_encoder_feedforward_dim=2048,
        actions_encoder_head_num=8,
        actions_encoder_layer_num=4,
        actions_encoder_feedforward_dim=2048,
        style_encoder_head_num=8,
        style_encoder_layer_num=4,
        style_encoder_feedforward_dim=2048,
        actions_decoder_head_num=8,
        actions_decoder_layer_num=7,
        actions_decoder_feedforward_dim=3200,
        plan_decoder_head_num=8,
        plan_decoder_layer_num=10,
        plan_decoder_feedforward_dim=3200, 
        chunk_size=100,
        hcode_dim=511,
        zcode_dim=513,
        zcode_num=512,
        hcode_num=1024,
        max_timestep=400,
        timestep_bin_size=2,
        ):
        super(HactVqModel, self).__init__()

        self.cnn_backbone_type=cnn_backbone_type
        self.state_dim=state_dim
        self.action_dim=action_dim
        self.context_dim=context_dim
        self.hidden_dim=hidden_dim
        self.input_image_num = input_image_num
        self.image_encoder_head_num = image_encoder_head_num
        self.image_encoder_layer_num = image_encoder_layer_num
        self.image_encoder_feedforward_dim = image_encoder_feedforward_dim
        self.subgoal_encoder_head_num = subgoal_encoder_head_num
        self.subgoal_encoder_layer_num = subgoal_encoder_layer_num
        self.subgoal_encoder_feedforward_dim = subgoal_encoder_feedforward_dim
        self.actions_encoder_head_num = actions_encoder_head_num
        self.actions_encoder_layer_num = actions_encoder_layer_num
        self.actions_encoder_feedforward_dim = actions_encoder_feedforward_dim
        self.style_encoder_head_num = style_encoder_head_num
        self.style_encoder_layer_num = style_encoder_layer_num
        self.style_encoder_feedforward_dim = style_encoder_feedforward_dim
        self.actions_decoder_head_num = actions_decoder_head_num
        self.actions_decoder_layer_num = actions_decoder_layer_num
        self.actions_decoder_feedforward_dim = actions_decoder_feedforward_dim
        self.plan_decoder_head_num = plan_decoder_head_num
        self.plan_decoder_layer_num = plan_decoder_layer_num
        self.plan_decoder_feedforward_dim = plan_decoder_feedforward_dim
        self.chunk_size = chunk_size
        self.hcode_dim = hcode_dim
        self.zcode_dim = zcode_dim
        self.hcode_num = hcode_num
        self.zcode_num = zcode_num
        self.max_timestep = max_timestep
        self.timestep_bin_size = timestep_bin_size
        self.time_cost_dim = self.max_timestep//self.timestep_bin_size
        if self.cnn_backbone_type=='resnet18':
            self.cnn_feature_shape = [15, 20]
            self.cnn_feature_num = self.input_image_num \
                                    * self.cnn_feature_shape[0] * self.cnn_feature_shape[1]

        self.cnn_backbone = ResnetBackbone(
            name = self.cnn_backbone_type, 
            train_backbone=True, 
            return_interm_layers=False, 
            dilation=False)

        self.image_encoder = TransformerEncoderAlone(
            d_model=self.hidden_dim,
            nhead=self.image_encoder_head_num,
            dim_feedforward=self.image_encoder_feedforward_dim,
            num_encoder_layers=self.image_encoder_layer_num,
            dropout=0.1,
            normalize_before=False
        )
        
        self.subgoal_encoder = TransformerEncoderAlone(
            d_model=self.hidden_dim,
            nhead=self.subgoal_encoder_head_num,
            dim_feedforward=self.subgoal_encoder_feedforward_dim,
            num_encoder_layers=self.subgoal_encoder_layer_num,
            dropout=0.1,
            normalize_before=False
        )

        self.actions_encoder = TransformerEncoderAlone(
            d_model=self.hidden_dim,
            nhead=self.actions_encoder_head_num,
            dim_feedforward=self.actions_encoder_feedforward_dim,
            num_encoder_layers=self.actions_encoder_layer_num,
            dropout=0.1,
            normalize_before=False
        )
        
        self.style_encoder = TransformerEncoderAlone(
            d_model=self.hidden_dim,
            nhead=self.style_encoder_head_num,
            dim_feedforward=self.style_encoder_feedforward_dim,
            num_encoder_layers=self.style_encoder_layer_num,
            dropout=0.1,
            normalize_before=False
        )
        self.actions_decoder = TransformerDecoderAlone(
            d_model=self.hidden_dim,
            nhead=self.actions_decoder_head_num,
            dim_feedforward=self.actions_decoder_feedforward_dim,
            num_decoder_layers=self.actions_decoder_layer_num,
            dropout=0.1,
            normalize_before=False,
            return_intermediate=False)
            
        self.plan_decoder = TransformerDecoderAlone(
            d_model=self.hidden_dim,
            nhead=self.plan_decoder_head_num,
            dim_feedforward=self.plan_decoder_feedforward_dim,
            num_decoder_layers=self.plan_decoder_layer_num,
            dropout=0.1,
            normalize_before=False,
            return_intermediate=False)
            
        self.image_positional_encoding = position_encoding.PositionalEncoding(
            d_model=self.hidden_dim, 
            dropout=0.1, 
            max_len=self.cnn_feature_num,
            )
        
        self.actions_encoder_positional_encoding = position_encoding.PositionalEncoding(
            d_model=self.hidden_dim, 
            dropout=0.1, 
            max_len=self.chunk_size,
            )

        self.subgoal_positional_encoding = position_encoding.PositionalEncoding(
            d_model=self.hidden_dim, 
            dropout=0.1, 
            max_len=1+self.cnn_feature_num,
            )

        self.style_positional_encoding = position_encoding.PositionalEncoding(
            d_model=self.hidden_dim, 
            dropout=0.1, 
            max_len=2+self.cnn_feature_num+self.chunk_size,
            )

        self.actions_decoder_positional_encoding = position_encoding.PositionalEncoding(
            d_model=self.hidden_dim, 
            dropout=0.1, 
            max_len=self.chunk_size,
            )

        self.actions_memory_positional_encoding = position_encoding.PositionalEncoding(
            d_model=self.hidden_dim, 
            dropout=0.1, 
            max_len=3+self.cnn_feature_num,
            )

        self.plan_decoder_positional_encoding = position_encoding.PositionalEncoding(
            d_model=self.hidden_dim, 
            dropout=0.1, 
            max_len=3+self.cnn_feature_num,
            )        

        self.input_cnn_proj = nn.Conv2d(self.cnn_backbone.num_channels, self.hidden_dim, kernel_size=1)
        self.input_image_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.input_actions_proj = nn.Linear(self.action_dim, self.hidden_dim)
        self.input_context_proj = nn.Linear(self.context_dim, self.hidden_dim)
        self.input_z_proj = nn.Linear(self.zcode_dim, self.hidden_dim)
        self.input_z_plan_proj = nn.Linear(self.zcode_dim, self.hidden_dim)
        self.input_h_style_proj = nn.Linear(self.hcode_dim, self.hidden_dim)
        self.input_h_act_proj = nn.Linear(self.hcode_num, self.hidden_dim)
        self.input_h_plan_proj = nn.Linear(self.hcode_num, self.hidden_dim)
        self.input_qpos_proj = nn.Linear(self.state_dim, self.hidden_dim)

        self.output_h_proj = nn.Linear(self.hidden_dim, self.hcode_dim)
        self.output_z_proj = nn.Linear(self.hidden_dim, self.zcode_dim)
        self.output_h_plan_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.output_z_plan_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.output_time_cost_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.h_head = nn.Linear(self.hidden_dim, self.hcode_num)
        self.z_head = nn.Linear(self.hidden_dim, self.zcode_num)
        self.action_head = nn.Linear(self.hidden_dim, self.action_dim)
        self.is_pad_head = nn.Linear(self.hidden_dim, 1)
        self.time_cost_head = nn.Linear(self.hidden_dim, self.time_cost_dim)

        self.actions_query = nn.Embedding(self.chunk_size, self.hidden_dim).weight
        self.h_query = nn.Embedding(1, self.hidden_dim).weight
        self.z_query = nn.Embedding(1, self.hidden_dim).weight
        self.time_cost_query = nn.Embedding(1, self.hidden_dim).weight

        self.z_codebook = VQEmbedding(self.zcode_num, self.zcode_dim)
        self.h_codebook = VQEmbedding(self.hcode_num, self.hcode_dim)
        self.register_buffer('actions_mask', get_actions_mask(self.chunk_size))
        self.register_buffer('actions_decoder_mask', get_actions_decoder_mask(self.chunk_size))
        self.loss_fn = HactVqLoss()

    def set_data_statistics(self, data_statistics):
        self._data_statistics = {
            'action_mean': torch.from_numpy(data_statistics.action_mean),
            'action_std': torch.from_numpy(data_statistics.action_std)
        }


    def forward(self,
                qpos, 
                images, 
                contexts,
                num_samples:int=10):
        images_embedding = self.encode_images(images, training=False)
        pred_h_logit, pred_z_logit = self.predict_plan(images_embedding, contexts)

        ## todo: change gumble softmax sampling
        pred_h_index_list = []
        pred_z_index_list = []
        pred_time_cost_list = []
        for _ in range(num_samples):
            pred_h_index = F.gumbel_softmax(pred_h_logit, tau=0.1, dim=-1, hard=True)
            pred_h_index = torch.argmax(pred_h_index, dim=-1)

            pred_z_index = F.gumbel_softmax(pred_z_logit, tau=0.1, dim=-1, hard=True)
            pred_z_index = torch.argmax(pred_z_index, dim=-1)

            pred_time_cost_logit = self.predict_time_cost(images_embedding, 
                                                            pred_h_index, 
                                                            pred_z_index)
            pred_time_cost = self._get_time_cost_from_logit(pred_time_cost_logit)
            
            pred_h_index_list.append(pred_h_index)
            pred_z_index_list.append(pred_z_index)
            pred_time_cost_list.append(pred_time_cost)
        
        pred_h_index_list = torch.stack(pred_h_index_list, dim=1)
        pred_z_index_list = torch.stack(pred_z_index_list, dim=1)
        pred_time_cost_list = torch.stack(pred_time_cost_list, dim=1)

        argmin_index = torch.argmin(pred_time_cost_list,dim=-1)
        opt_pred_time_cost = pred_time_cost_list[:, argmin_index][:,0]
        opt_pred_h_index = pred_h_index_list[:,argmin_index][:,0]
        opt_pred_z_index = pred_z_index_list[:,argmin_index][:,0]
        
        h_q = self._get_h_q(opt_pred_h_index)
        z_q = self._get_z_q(opt_pred_z_index)

        pred_a, pred_is_pad_logit = self.predict_actions(qpos, images_embedding,
                                                    h_q, z_q)


        model_outputs = {
            'pred_h_index': opt_pred_h_index,
            'pred_z_index': opt_pred_z_index,
            'time_cost': opt_pred_time_cost,
            'h_q': h_q,
            'z_q': z_q,
            'pred_a': pred_a,
            'pred_is_pad': torch.sigmoid(pred_is_pad_logit)

        }
        return model_outputs


    def forward_train(self,
            qpos,
            images,
            goal_images,
            actions,
            is_pad,
            time_cost,
            context):
        images_embedding = self.encode_images(images, training=True)
        goal_images_embedding = self.encode_images(goal_images, training=True)
        actions_embedding = self.encode_actions(actions, is_pad)
        
        h_e, h_q, h_q_sample, h_index = self.encode_subgoal(goal_images_embedding)
        z_e, z_q, z_q_sample, z_index = self.encode_style(images_embedding, 
                                                    h_q_sample, actions_embedding, is_pad)

        pred_a, pred_is_pad_logit = self.predict_actions(qpos, images_embedding,
                                                    h_q_sample, z_q_sample)


        pred_h_logit, pred_z_logit = self.predict_plan(images_embedding, context)
        pred_time_cost_logit = self.predict_time_cost(images_embedding, 
                                                    h_index.detach(), 
                                                    z_index.detach())
        time_cost_index = self.compute_time_cost_index(time_cost)

        model_outputs={
            'pred_a': pred_a,
            'pred_is_pad_logit': pred_is_pad_logit,
            'h_e': h_e,
            'h_q': h_q,
            'h_q_sample': h_q_sample,
            'h_index': h_index,

            'z_e': z_e,
            'z_q': z_q,
            'z_q_sample': z_q_sample,
            'z_index': z_index,
            'time_cost_index': time_cost_index,

            'pred_h_logit': pred_h_logit,
            'pred_z_logit': pred_z_logit,
            'pred_time_cost_logit': pred_time_cost_logit,

        }
        return model_outputs


    def encode_images(self, images, training:bool=True):
        bs = images.shape[0]
        num_images = images.shape[1]
        num_patches = self.cnn_feature_num

        if training:
            cnn_feature_list = self._get_cnn_feature(images)
        else:
            with torch.no_grad():
                cnn_feature_list = self._get_cnn_feature(images)
        
        feature_embedding = []
        for cnn_feature in cnn_feature_list:
            feature_embedding.append(self.input_cnn_proj(cnn_feature))
        images_inputs= torch.cat(feature_embedding, dim=3).flatten(2).permute(0,2,1)
        images_pos = self.image_positional_encoding(images_inputs)
        # Maybe due to torch internel problem, setting explicit shape is required
        images_inputs = images_inputs.reshape((bs, num_patches, self.hidden_dim))
        images_pos = images_pos.reshape((bs, num_patches, self.hidden_dim))
        images_embedding = self.image_encoder(images_inputs, images_pos)
        images_embedding=images_embedding.reshape((bs, num_patches, self.hidden_dim))
        return images_embedding


    def encode_actions(self, actions, is_pad):
        actions_inputs = self.input_actions_proj(actions)
        actions_pos = self.actions_encoder_positional_encoding(actions_inputs)
        actions_embedding= self.actions_encoder(actions_inputs, actions_pos, 
                                            mask=self.actions_mask, src_key_padding_mask=is_pad)
        return actions_embedding


    def encode_subgoal(self, goal_images_embedding):
        bs = goal_images_embedding.shape[0]
        device = goal_images_embedding.device
        num_image_patches = goal_images_embedding.shape[1]

        h_query = torch.unsqueeze(self.h_query, dim=0).repeat(bs, 1, 1)
        goal_encoder_input = torch.cat([h_query,
                                        goal_images_embedding], dim=1)
        goal_encoder_pos = self.subgoal_positional_encoding(goal_encoder_input)

        goal_encoder_input = goal_encoder_input.reshape([bs, 1+num_image_patches, self.hidden_dim])
        goal_encoder_pos = goal_encoder_pos.reshape([bs, 1+num_image_patches, self.hidden_dim])
        
        goal_encoder_output = self.subgoal_encoder(goal_encoder_input,
                                                    goal_encoder_pos)
        
        h_e = self.output_h_proj(goal_encoder_output[:,0,:])
        h_e_reshaped = h_e.unsqueeze(dim=2).unsqueeze(dim=3)
        h_q_sample, h_q, h_indices = self.h_codebook.straight_through(h_e_reshaped)
        h_q = h_q.squeeze(dim=3).squeeze(dim=2)
        h_q_sample = h_q_sample.squeeze(dim=3).squeeze(dim=2)
        return h_e, h_q, h_q_sample, h_indices


    def encode_style(self, images_embedding, h_sample, actions_embedding, is_pad):
        bs = images_embedding.shape[0]
        device = images_embedding.device
        num_image_patches = images_embedding.shape[1]

        h_embedding = self.input_h_style_proj(h_sample).unsqueeze(dim=1)
        
        z_query = torch.unsqueeze(self.z_query, dim=0).repeat(bs, 1, 1)
        style_input = torch.cat([z_query,
                                images_embedding,
                                h_embedding,
                                actions_embedding], dim=1)
        style_pos = self.style_positional_encoding(style_input)

        style_input = style_input.reshape([bs, self.chunk_size+2+num_image_patches, self.hidden_dim])
        style_pos = style_pos.reshape([bs, self.chunk_size+2+num_image_patches, self.hidden_dim])
        src_key_padding_mask = torch.zeros((bs, 2+num_image_patches), dtype=torch.bool).to(device)
        src_key_padding_mask = torch.cat([src_key_padding_mask,
                                            is_pad], dim=1)
        style_output = self.style_encoder(style_input, style_pos, src_key_padding_mask=src_key_padding_mask)
        z_e = self.output_z_proj(style_output[:,0,:])
        z_e_reshaped = z_e.unsqueeze(dim=2).unsqueeze(dim=3)
        z_q_sample, z_q, z_indices = self.z_codebook.straight_through(z_e_reshaped)
        z_q = z_q.squeeze(dim=3).squeeze(dim=2)
        z_q_sample = z_q_sample.squeeze(dim=3).squeeze(dim=2)
        return z_e, z_q, z_q_sample, z_indices

    def predict_actions(self, qpos, images_embedding, h_sample, z_sample):
        bs = qpos.shape[0]
        num_image_patches = images_embedding.shape[1]
        proprio_embedding = self.input_qpos_proj(qpos).unsqueeze(dim=1)
        z_embedding = self.input_z_proj(z_sample).unsqueeze(dim=1)
        h_embedding = self.input_h_act_proj(h_sample).unsqueeze(dim=1)

        memory = torch.concat([images_embedding, 
                                h_embedding, 
                                proprio_embedding, 
                                z_embedding], dim=1)
        memory_pos = self.actions_memory_positional_encoding(memory)

        actions_query = self.actions_query.repeat(bs, 1, 1)
        actions_query_pos = self.actions_decoder_positional_encoding(actions_query)
        actions_query = actions_query.reshape((bs, self.chunk_size, self.hidden_dim))
        actions_query_pos = actions_query_pos.reshape((bs, self.chunk_size, self.hidden_dim))
        memory = memory.reshape((bs, num_image_patches+3, self.hidden_dim))
        memory_pos = memory_pos.reshape((bs,num_image_patches+3, self.hidden_dim))

        hs = self.actions_decoder(
                src = actions_query,
                pos = actions_query_pos,
                memory = memory,
                memory_pos = memory_pos,
                mask = self.actions_decoder_mask
            )

        a_hat = self.action_head(hs)
        is_pad_hat_logit = self.is_pad_head(hs)
        #is_pad_hat = torch.sigmoid(is_pad_hat_logit)
        return a_hat, is_pad_hat_logit

    def predict_plan(self, images_embedding, context):
        bs = images_embedding.shape[0]
        num_patches = self.cnn_feature_num + 3
        images_embed = self.input_image_proj(images_embedding) # (bs, num_patches, dim)
        context_embed= self.input_context_proj(context).unsqueeze(dim=1)

        h_query = torch.unsqueeze(self.h_query, dim=0).repeat(bs, 1, 1)
        z_query = torch.unsqueeze(self.z_query, dim=0).repeat(bs, 1, 1)
        plan_decoder_input = torch.cat([images_embed,
                                        context_embed,
                                            h_query,
                                            z_query], dim=1)
        plan_decoder_pos = self.plan_decoder_positional_encoding(plan_decoder_input)
        
        plan_decoder_input = plan_decoder_input.reshape((bs, num_patches, self.hidden_dim))
        plan_decoder_pos = plan_decoder_pos.reshape((bs, num_patches, self.hidden_dim))
        plan_decoder_output = self.plan_decoder(plan_decoder_input, plan_decoder_pos)
        h_ouput = self.output_h_plan_proj(plan_decoder_output[:, -2, :])
        h_ouput = F.relu(h_ouput)
        pred_h_logit = self.h_head(h_ouput)

        z_ouput = self.output_z_plan_proj(plan_decoder_output[:, -1, :])
        z_ouput = F.relu(z_ouput)
        pred_z_logit = self.z_head(z_ouput)
        return pred_h_logit, pred_z_logit

    def predict_time_cost(self, images_embedding, pred_h_index, pred_z_index):
        bs = images_embedding.shape[0]
        images_embed = self.input_image_proj(images_embedding)

        pred_z_onehot = F.one_hot(pred_z_index, num_classes=self.zcode_num).float()
        pred_h_onehot = F.one_hot(pred_h_index, num_classes=self.hcode_num).float()
        pred_h_embed = self.input_h_plan_proj(pred_h_onehot).unsqueeze(dim=1)
        pred_z_embed = self.input_z_plan_proj(pred_z_onehot).unsqueeze(dim=1)

        h_query = torch.unsqueeze(self.h_query, dim=0).repeat(bs, 1, 1)
        z_query = torch.unsqueeze(self.z_query, dim=0).repeat(bs, 1, 1)
        time_cost_query = torch.unsqueeze(self.time_cost_query, dim=0).repeat(bs, 1, 1)
        
        plan_decoder_input = torch.cat([images_embed,
                                            h_query,
                                            z_query,
                                            pred_h_embed,
                                            pred_z_embed,
                                            time_cost_query], dim=1)
        plan_decoder_output = self.plan_decoder(plan_decoder_input)
        time_cost_ouput = self.output_time_cost_proj(plan_decoder_output[:, -1, :])
        time_cost_ouput = F.relu(time_cost_ouput)
        pred_time_cost_logit = self.time_cost_head(time_cost_ouput)
        return pred_time_cost_logit

    def compute_time_cost_index(self, time_cost):
        time_cost_index = (time_cost/self.timestep_bin_size).to(torch.int64)
        time_cost_index = torch.clip(time_cost_index, 0, self.time_cost_dim-1)
        return time_cost_index

    def _get_cnn_feature(self, images):
        feature_list = []
        num_camera = images.shape[1]

        for i in range(num_camera):
            features = self.cnn_backbone(images[:, i, :, :])
            
            cnn_feature_list = []
            for key, feature in features.items():
                cnn_feature_list.append(feature)
            picked_features = cnn_feature_list[0] # take the last layer feature
            feature_list.append(picked_features)
        return feature_list

    def _get_time_cost_from_logit(self, time_cost_logit):
        bs = time_cost_logit.shape[0]
        device = time_cost_logit.device
        arange = torch.arange(self.time_cost_dim).to(device)
        arange = torch.unsqueeze(arange, dim=0).repeat(bs,1)
        time_cost_weight = F.softmax(time_cost_logit, dim=-1)
        time_cost = torch.sum(arange*time_cost_weight, dim = -1)
        return time_cost

    def _get_h_q(self, h_index):
        return self.h_codebook.sample_by_indices(h_index)
    
    def _get_z_q(self, z_index):
        return self.z_codebook.sample_by_indices(z_index)

    @torch.jit.unused
    def compute_train_loss(self, data):
        qpos_batch = data['qpos']
        images_batch = data['images']
        goal_images_batch = data['goal_images']
        actions_batch = data['actions']
        is_pad_batch = data['is_pad']
        context_batch = data['contexts']
        time_cost_batch = data['time_cost']
        
        device = qpos_batch.device
        action_mean = self._data_statistics['action_mean'].to(device).float()
        action_std = self._data_statistics['action_std'].to(device).float()

        model_outputs = self.forward_train(
                                qpos_batch,
                                images_batch,
                                goal_images_batch,
                                actions_batch,
                                is_pad_batch,
                                time_cost_batch,
                                context_batch)

        loss_outputs = self.loss_fn(data, 
                               model_outputs,
                               action_mean, 
                               action_std)

        loss = loss_outputs['total_loss']
        return loss, loss_outputs

    @torch.jit.unused
    def compute_val_loss(self, data):
        qpos_batch = data['qpos']
        images_batch = data['images']
        goal_images_batch = data['goal_images']
        actions_batch = data['actions']
        is_pad_batch = data['is_pad']
        context_batch = data['contexts']
        time_cost_batch = data['time_cost']

        device = qpos_batch.device
        action_mean = self._data_statistics['action_mean'].to(device)
        action_std = self._data_statistics['action_std'].to(device)

        model_outputs = self.forward_train(
                                qpos_batch,
                                images_batch,
                                goal_images_batch,
                                actions_batch,
                                is_pad_batch,
                                time_cost_batch,
                                context_batch)

        loss_outputs = self.loss_fn(data, 
                               model_outputs,
                               action_mean, 
                               action_std)

        loss = loss_outputs['total_loss']
        return loss, loss_outputs