
from functools import partial

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets.swin_unetr import SwinTransformer

from utils.atlas import *


class SwinViT(SwinTransformer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mask_token = nn.Parameter(torch.zeros(1, self.embed_dim, 1, 1))
        nn.init.normal_(self.mask_token, mean=0., std=0.02)

    def forward(self, x, mask=None, normalize=True):
        x = self.patch_embed(x)

        # mask
        assert mask is not None
        B, C, H, W = x.shape
        mask_tokens = self.mask_token.expand(B, -1, H, W)
        w = mask.unsqueeze(1).repeat(1, C, 1, 1).type_as(mask_tokens)
        x0 = x * (1 - w) + mask_tokens * w

        x0_out = self.proj_out(x0, normalize)
        if self.use_v2:
            x0 = self.layers1c[0](x0.contiguous())

        x1 = self.layers1[0](x0.contiguous())
        x1_out = self.proj_out(x1, normalize)
        if self.use_v2:
            x1 = self.layers2c[0](x1.contiguous())

        x2 = self.layers2[0](x1.contiguous())
        x2_out = self.proj_out(x2, normalize)
        if self.use_v2:
            x2 = self.layers3c[0](x2.contiguous())

        x3 = self.layers3[0](x2.contiguous())
        x3_out = self.proj_out(x3, normalize)
        if self.use_v2:
            x3 = self.layers4c[0](x3.contiguous())

        x4 = self.layers4[0](x3.contiguous())
        x4_out = self.proj_out(x4, normalize)

        return [x0_out, x1_out, x2_out, x3_out, x4_out]


class EchoCareHead(nn.Module):
    def __init__(self, configs, n_hiera_class, upsample='vae', dim=2048):
        super().__init__()
        self.n_hiera_class = n_hiera_class

        self.encoder = SwinViT(
            in_chans=configs.in_channels,
            embed_dim=configs.feature_size,
            window_size=[8] * configs.spatial_dims,
            patch_size=[configs.patch_size] * configs.spatial_dims,
            depths=[2, 2, 18, 2],
            num_heads=[4, 8, 16, 32],
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=configs.dropout_path_rate,
            norm_layer=nn.LayerNorm,
            use_checkpoint=configs.use_checkpoint,
            spatial_dims=configs.spatial_dims,
            use_v2=True
        )

        # image decoder
        if upsample == "large_kernel_deconv":
            self.pixel_decoder = nn.ConvTranspose2d(dim, configs.in_channels, kernel_size=(32, 32), stride=(32, 32))
        elif upsample == "deconv":
            self.pixel_decoder = nn.Sequential(
                nn.ConvTranspose2d(dim, dim // 2, kernel_size=(2, 2), stride=(2, 2)),
                nn.ConvTranspose2d(dim // 2, dim // 4, kernel_size=(2, 2), stride=(2, 2)),
                nn.ConvTranspose2d(dim // 4, dim // 8, kernel_size=(2, 2), stride=(2, 2)),
                nn.ConvTranspose2d(dim // 8, dim // 16, kernel_size=(2, 2), stride=(2, 2)),
                nn.ConvTranspose2d(dim // 16, configs.in_channels, kernel_size=(2, 2), stride=(2, 2)),
            )
        elif upsample == "vae":
            self.pixel_decoder = nn.Sequential(
                nn.Conv2d(dim, dim // 2, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm2d(dim // 2),
                nn.LeakyReLU(),
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(dim // 2, dim // 4, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm2d(dim // 4),
                nn.LeakyReLU(),
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(dim // 4, dim // 8, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm2d(dim // 8),
                nn.LeakyReLU(),
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(dim // 8, dim // 16, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm2d(dim // 16),
                nn.LeakyReLU(),
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(dim // 16, dim // 16, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm2d(dim // 16),
                nn.LeakyReLU(),
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(dim // 16, configs.in_channels, kernel_size=1, stride=1),
            )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.hiera_pre = nn.Linear(dim, dim)
        self.hiera_head = nn.Linear(dim, n_hiera_class)

        self.patch_size = configs.patch_size

    def forward(self, image, mask):
        _, _, _, _, z = self.encoder(image.contiguous(), mask)
        image_rec = self.pixel_decoder(z)
        x = self.avgpool(z)
        x = x.view(x.size(0), -1)
        image_hiera = self.hiera_head(self.hiera_pre(x))
        mask = mask.repeat_interleave(self.patch_size, 1).repeat_interleave(self.patch_size, 2).unsqueeze(1).contiguous()
        return image_rec, image_hiera, mask

    @classmethod
    def loss(cls, image_ori, image_label, mask, image_rec, image_prediction, eps=1e-7):
        loss_rec = F.l1_loss(image_ori, image_rec, reduction='none')
        loss_recon = (loss_rec * mask).sum() / (mask.sum() + 1e-5)

        bs = image_prediction.shape[0]
        image_prediction = F.sigmoid(image_prediction)

        # atlas
        n_hiera_class = image_prediction.size(1)
        bottom_nodes, middle_nodes, top_nodes = AtlasNode.get_leaves(), AtlasNode.get_middle_level_nodes(), AtlasNode.get_roots()

        # target
        image_target = []
        for per_image_label in image_label:
            one_batch_target = torch.zeros((n_hiera_class,), dtype=per_image_label.dtype, device=per_image_label.device)
            one_batch_target[per_image_label] = 1
            # all father node
            for farther_label in AtlasNode.get_parents_by_id(per_image_label):
                one_batch_target[farther_label] = 1
            image_target.append(one_batch_target)
        image_target = torch.stack(image_target, dim=0)

        # hierarchical constraints
        # bottom level
        MCMA = image_prediction[:, :len(bottom_nodes)]
        # middle level
        MCMB = torch.zeros((bs, len(middle_nodes)), device=image_prediction.device, dtype=image_prediction.dtype)
        for i, node_id in enumerate(middle_nodes):
            subclass_ids = AtlasNode.get_direct_children_by_id(node_id)
            max_sub_score = torch.max(torch.cat([MCMA[:, sub_id:sub_id+1] for sub_id in subclass_ids], dim=1), 1, keepdim=True)[0]
            middle_score = image_prediction[:, node_id:node_id + 1]
            MCMB[:, i:i+1] = torch.max(max_sub_score, middle_score)
        # top level
        MCMC = torch.zeros((bs, len(top_nodes)), device=image_prediction.device, dtype=image_prediction.dtype)
        for i, node_id in enumerate(top_nodes):
            subclass_ids = AtlasNode.get_children_by_id(node_id)
            max_sub_score = torch.max(torch.cat([image_prediction[:, sub_id:sub_id+1] for sub_id in subclass_ids], dim=1), 1, keepdim=True)[0]
            top_score = image_prediction[:, node_id:node_id + 1]
            MCMC[:, i:i+1] = torch.max(max_sub_score, top_score)

        # top level
        offset = len(bottom_nodes) + len(middle_nodes)
        MCLC = image_prediction[:, offset:]
        # middle level
        MCLB = torch.zeros((bs, len(middle_nodes)), device=image_prediction.device, dtype=image_prediction.dtype)
        for i, node_id in enumerate(middle_nodes):
            superclass_id = AtlasNode.get_direct_parent_by_id(node_id)
            top_score = image_prediction[:, superclass_id:superclass_id+1]
            middle_score = image_prediction[:, node_id:node_id + 1]
            MCLB[:, i:i+1] = torch.min(top_score, middle_score)
        # bottom level
        MCLA = torch.zeros((bs, len(bottom_nodes)), device=image_prediction.device, dtype=image_prediction.dtype)
        for i, node_id in enumerate(bottom_nodes):
            superclass_ids = AtlasNode.get_parents_by_id(node_id)
            top_score = torch.min(torch.cat([image_prediction[:, sub_id:sub_id+1] for sub_id in superclass_ids], dim=1), 1, keepdim=True)[0]
            bottom_score = image_prediction[:, node_id:node_id + 1]
            MCLA[:, i:i+1] = torch.min(top_score, bottom_score)

        # meta-classification hierarchical loss
        loss_hiera = 0
        bottom_target = image_target[:, :len(bottom_nodes)]
        middle_target = image_target[:, len(bottom_nodes):len(bottom_nodes)+len(middle_nodes)]
        top_target = image_target[:, len(bottom_nodes)+len(middle_nodes):]

        loss_hiera += torch.mean((-bottom_target*torch.log(MCLA+eps)-(1-bottom_target)*torch.log(1-MCMA+eps)))
        loss_hiera += torch.mean((-middle_target*torch.log(MCLB+eps)-(1-middle_target)*torch.log(1-MCMB+eps)))
        loss_hiera += torch.mean((-top_target*torch.log(MCLC+eps)-(1-top_target)*torch.log(1-MCMC+eps)))

        return loss_recon, loss_hiera

