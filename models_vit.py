# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# MAE: https://github.com/facebookresearch/mae
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn

import timm.models.vision_transformer


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

        self.global_pool = global_pool
        self.softmax = nn.Softmax(dim=-1)
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

    def patchify_map(self, map):
        """
        map: (B, 1, H, W)
        x: (B, patch_size**2)
        """
        p = self.patch_embed.patch_size[0]

        h = w = map.shape[2] // p
        map = map.squeeze(1)
        x = map.reshape(shape=(map.shape[0], h, p, w, p))
        x = torch.einsum('bhpwq->bhwpq', x)
        x = x.reshape(shape=(map.shape[0], h * w, p ** 2))
        return x

    def masking_throwing(self, x, mask_ratio, throw_ratio, mask_weights):
        """
        Perform per-sample attention-driven masking and throwing.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim

        len_mask_tail = int(L * mask_ratio)
        len_keep_head = int(L * (mask_ratio + throw_ratio))

        mask_weights = self.patchify_map(mask_weights)
        mask_weights = mask_weights.sum(-1)

        ids_shuffle = torch.multinomial(mask_weights, L)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, len_keep_head:]

        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is masked, -1 is thrown
        mask = torch.ones([N, L], device=x.device)
        mask[:, len_keep_head:] = 0
        mask[:, len_mask_tail:len_keep_head] = -1

        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward(self, x, mask_weights, mask_ratio, throw_ratio):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        x, _, _ = self.masking_throwing(x, mask_ratio, throw_ratio, mask_weights)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_drop(x)

        # apply Transformer blocks
        for i, blk in enumerate(self.blocks):
            x = blk(x)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        outcome = self.head(outcome)
        return outcome

    def forward_test(self, x):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_drop(x)

        # apply Transformer blocks
        for i, blk in enumerate(self.blocks):
            x = blk(x)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        outcome = self.head(outcome)
        return outcome


    def forward_encoder_test(self,x):
        x = self.patch_embed(x)

    # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

    # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_drop(x)

    # apply Transformer blocks
        for i, blk in enumerate(self.blocks):
            x = blk(x)

        cam = self.blocks[-1].attn.return_attn()
        return cam

def vit_base_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_huge_patch14(**kwargs):
    model = VisionTransformer(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

