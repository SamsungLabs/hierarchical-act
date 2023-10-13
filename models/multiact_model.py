from typing import Optional

import gin
import numpy as np
import torch
from torch import nn


from .registry import register
from losses.act_loss import ActLoss
from models.module import position_encoding
from models.module.cnn_backbone import ResnetBackbone
from models.module.act_transformer import Transformer, TransformerEncoderAlone
from ops.reparametrize import reparametrize_normal

def get_sinusoid_encoding_table(n_position, d_hid):
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]
    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
    return torch.FloatTensor(sinusoid_table).unsqueeze(0)

@register('multiact')
@gin.configurable()
class MultiActModel(nn.Module):
    def __init__(self,
        cnn_backbone_type='resnet18',
        cnn_backbone_lr=1e-5,
        state_dim=14,
        action_dim=14,
        context_dim=512,
        hidden_dim=512,
        latent_dim=32,
        transformer_dropout=0.1,
        transformer_nhead=8,
        dim_feedforward=3200,
        num_encoder_layers=4,
        num_decoder_layers=7,
        num_vae_encoder_layers=4,
        chunk_size=100,
        ):
        super(MultiActModel, self).__init__()

        self.cnn_backbone_type=cnn_backbone_type
        self.cnn_backbone_lr=cnn_backbone_lr
        self.state_dim=state_dim
        self.action_dim=action_dim
        self.hidden_dim=hidden_dim
        self.context_dim=context_dim
        self.latent_dim=latent_dim
        self.transformer_dropout=transformer_dropout
        self.transformer_nhead=transformer_nhead
        self.dim_feedforward=dim_feedforward
        self.num_vae_encoder_layers = num_vae_encoder_layers
        self.num_encoder_layers=num_encoder_layers
        self.num_decoder_layers=num_decoder_layers
        self.chunk_size=chunk_size
        
        self.position_embedding = position_encoding.PositionEmbeddingSine(
            num_pos_feats=self.hidden_dim//2, 
            temperature=10000, 
            normalize=False, 
            scale=None)
        #self.position_embedding = position_encoding.PositionEmbeddingLearned(
        #    num_pos_feats=self.num_pos_feats)

        self.cnn_backbone = ResnetBackbone(
            name = self.cnn_backbone_type, 
            train_backbone=True, 
            return_interm_layers=False, 
            dilation=False)

        self.transformer = Transformer(
            d_model=self.hidden_dim,
            nhead=self.transformer_nhead,
            dropout=0.1,
            dim_feedforward=3200,
            num_encoder_layers=self.num_encoder_layers,
            num_decoder_layers=self.num_decoder_layers,
            normalize_before=False,
            return_intermediate_dec=True)

        self.encoder = TransformerEncoderAlone(
            d_model=self.hidden_dim,
            nhead=self.transformer_nhead,
            dim_feedforward=2048,
            dropout=0.1,
            num_encoder_layers=self.num_vae_encoder_layers,
            normalize_before=False
        )

        self.action_head = nn.Linear(self.hidden_dim, self.action_dim)
        self.is_pad_head = nn.Linear(self.hidden_dim, 1)
        self.query_embed = nn.Embedding(self.chunk_size, self.hidden_dim)
        self.input_proj = nn.Conv2d(self.cnn_backbone.num_channels, self.hidden_dim, kernel_size=1)
        self.input_proj_robot_state = nn.Linear(self.state_dim, self.hidden_dim)
        self.input_proj_context = nn.Linear(self.context_dim, self.hidden_dim)
        self.cls_embed = nn.Embedding(1, self.hidden_dim) # extra cls token embedding
        self.encoder_proj = nn.Linear(self.action_dim, self.hidden_dim) # project state to embedding
        self.encoder_joint_proj = nn.Linear(self.state_dim, self.hidden_dim)
        self.latent_proj = nn.Linear(self.hidden_dim, self.latent_dim*2) # project hidden state to latent std, var
        self.register_buffer('pos_table', get_sinusoid_encoding_table(self.chunk_size+2, self.hidden_dim))
        self.latent_out_proj = nn.Linear(self.latent_dim, self.hidden_dim) # project latent sample to embedding
        self.additional_pos_embed = nn.Embedding(3, self.hidden_dim) # learned position embedding for proprio and latent

        self.act_loss = ActLoss()


    def set_data_statistics(self, data_statistics):
        self._data_statistics = {
            'action_mean': torch.from_numpy(data_statistics.action_mean),
            'action_std': torch.from_numpy(data_statistics.action_std)
        }

    def forward(self, qpos, images, context,
                actions: Optional[torch.Tensor]=None, 
                is_pad: Optional[torch.Tensor]=None):
        bs, _ = qpos.shape
        device = qpos.device
        ### Obtain latent z from action sequence
        if (actions is not None) and (is_pad is not None):
            # project action sequence to embedding dim, and concat with a CLS token
            action_embed = self.encoder_proj(actions) # (bs, seq, hidden_dim)
            qpos_embed = self.encoder_joint_proj(qpos).unsqueeze(1)
            cls_embed = self.cls_embed.weight # (1, hidden_dim)
            cls_embed = torch.unsqueeze(cls_embed, dim=0).repeat(bs, 1, 1) # (bs, 1, hidden_dim)
            encoder_input = torch.cat([cls_embed, qpos_embed, action_embed], dim=1) # (bs, seq+1, hidden_dim)
            # do not mask cls token
            cls_is_pad = torch.zeros((bs, 1), dtype=torch.bool).to(device) # False: not a padding
            qpos_is_pad = torch.zeros((bs, 1), dtype=torch.bool).to(device) # False: not a padding
            is_pad = torch.cat([cls_is_pad, qpos_is_pad, is_pad], dim=1)  # (bs, seq+2)
            # obtain position embedding
            pos_embed = self.pos_table.clone().detach()
            # query model
            encoder_output = self.encoder(encoder_input, pos_embed, src_key_padding_mask=is_pad)
            encoder_output = encoder_output[:,0,:] # take cls output only
            latent_info = self.latent_proj(encoder_output)
            mu = latent_info[:, :self.latent_dim]
            logvar = latent_info[:, self.latent_dim:]
            latent_sample = reparametrize_normal(mu, logvar, training=True)
            latent_input = self.latent_out_proj(latent_sample)

            all_cam_features, all_cam_pos = self.get_all_cam_features(images)
        else:
            mu = logvar = torch.zeros([bs, self.latent_dim], dtype=torch.float32).to(device)
            latent_sample = torch.zeros([bs, self.latent_dim], dtype=torch.float32).to(device)
            latent_input = self.latent_out_proj(latent_sample)
            with torch.no_grad():
                all_cam_features, all_cam_pos = self.get_all_cam_features(images)

        # proprioception features
        proprio_input = self.input_proj_robot_state(qpos) #[batch_size, 512]
        context_input = self.input_proj_context(context) #[batch_size, 512]
        # fold camera dimension into width dimension
        src = torch.cat(all_cam_features, dim=3)
        pos = torch.cat(all_cam_pos, dim=3)

        hs = self.transformer(src, self.query_embed.weight, pos, None, 
                              latent_input, proprio_input, context_input,
                              additional_pos_embed=self.additional_pos_embed.weight)[0]

        a_hat = self.action_head(hs)
        is_pad_hat = self.is_pad_head(hs)

        model_outputs={
            'a_hat': a_hat,
            'is_pad_hat': is_pad_hat,
            'mu': mu,
            'logvar': logvar
        }
        return model_outputs
    
    def get_all_cam_features(self, images):
        # Image observation features and position embeddings
        all_cam_features = []
        all_cam_pos = []
        num_camera = images.shape[1]

        for i in range(num_camera):
            features = self.cnn_backbone(images[:, i, :, :])
            
            feature_list = []
            pos_list = []
            for key, feature in features.items():
                pos = self.position_embedding(feature)
                feature_list.append(feature)
                pos_list.append(pos)
            picked_features = feature_list[0] # take the last layer feature
            picked_pos = pos_list[0]
            all_cam_features.append(self.input_proj(picked_features))
            all_cam_pos.append(picked_pos)
        return all_cam_features, all_cam_pos

    @torch.jit.unused
    def compute_train_loss(self, data):
        images = data['images']
        qpos = data['qpos']
        contexts = data['contexts']
        actions = data['actions']
        is_pad = data['is_pad']
        device = qpos.device

        action_mean = self._data_statistics['action_mean'].to(device)
        action_std = self._data_statistics['action_std'].to(device)
        
        model_outputs = self(qpos, images, contexts, actions=actions, is_pad=is_pad)
        loss_outputs = self.act_loss(data, 
                                model_outputs, 
                                action_mean, 
                                action_std)
        loss = loss_outputs['total_loss']
        return loss, loss_outputs

    @torch.jit.unused
    def compute_val_loss(self, data):
        images = data['images']
        qpos = data['qpos']
        contexts = data['contexts']
        actions = data['actions']
        is_pad = data['is_pad']
        device = qpos.device

        action_mean = self._data_statistics['action_mean'].to(device)
        action_std = self._data_statistics['action_std'].to(device)

        model_outputs = self(qpos, images, contexts, actions=actions, is_pad=is_pad)
        loss_outputs = self.act_loss(data, 
                                model_outputs, 
                                action_mean, 
                                action_std)
        loss = loss_outputs['total_loss']
        return loss, loss_outputs