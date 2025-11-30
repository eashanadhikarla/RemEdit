"""
Codebase for "Improved Denoising Diffusion Probabilistic Models".
"""


from abc import abstractmethod

import math

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from .fp16_util import convert_module_to_f16, convert_module_to_f32
from .nn import (
    checkpoint,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
)

def slerp(t,v0,v1):
    _shape = v0.shape

    v0_origin = v0.clone()
    v1_origin = v1.clone()

    v0_copy = v0.view(_shape[0], -1)
    v1_copy = v1.view(_shape[0], -1)

    # Normalize the vectors to get the directions and angles
    v0 = v0 / th.norm(v0_copy, dim=1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    v1 = v1 / th.norm(v1_copy, dim=1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

    v0_copy = v0.view(_shape[0], -1)
    v1_copy = v1.view(_shape[0], -1)

    # Dot product with the normalized vectors (can't use np.dot in W)
    dot = th.sum(v0_copy * v1_copy, dim=1, keepdim=True).squeeze(-1)
    # If absolute value of dot product is almost 1, vectors are ~colineal, so use lerp
    # if torch.abs(dot) > 0.9995:
    #     return lerp(t, v0, v1)
    # Calculate initial angle between v0 and v1
    theta_0 = th.acos(dot)
    sin_theta_0 = th.sin(theta_0)
    # Angle at timestep t
    theta_t = theta_0 * t
    sin_theta_t = th.sin(theta_t)
    # Finish the slerp algorithm
    s0 = th.sin(theta_0 - theta_t) / sin_theta_0
    s1 = sin_theta_t / sin_theta_0
    s0 = s0.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    s1 = s1.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    v2 = s0 * v0_origin + s1 * v1_origin
    # v2 = v2.view(_shape)
    return v2



class AttentionPool2d(nn.Module):
    """
    Adapted from CLIP: https://github.com/openai/CLIP/blob/main/clip/model.py
    """

    def __init__(
        self,
        spacial_dim: int,
        embed_dim: int,
        num_heads_channels: int,
        output_dim: int = None,
    ):
        super().__init__()
        self.positional_embedding = nn.Parameter(
            th.randn(embed_dim, spacial_dim ** 2 + 1) / embed_dim ** 0.5
        )
        self.qkv_proj = conv_nd(1, embed_dim, 3 * embed_dim, 1)
        self.c_proj = conv_nd(1, embed_dim, output_dim or embed_dim, 1)
        self.num_heads = embed_dim // num_heads_channels
        self.attention = QKVAttention(self.num_heads)

    def forward(self, x):
        b, c, *_spatial = x.shape
        x = x.reshape(b, c, -1)  # NC(HW)
        x = th.cat([x.mean(dim=-1, keepdim=True), x], dim=-1)  # NC(HW+1)
        x = x + self.positional_embedding[None, :, :].to(x.dtype)  # NC(HW+1)
        x = self.qkv_proj(x)
        x = self.attention(x)
        x = self.c_proj(x)
        return x[:, :, 0]


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(
                dims, self.channels, self.out_channels, 3, stride=stride, padding=1
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.

    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        )

    def _forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.

    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
        use_new_attention_order=False,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        if use_new_attention_order:
            # split qkv before split heads
            self.attention = QKVAttention(self.num_heads)
        else:
            # split heads before split qkv
            self.attention = QKVAttentionLegacy(self.num_heads)

        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x):
        return checkpoint(self._forward, (x,), self.parameters(), True)

    def _forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


def count_flops_attn(model, _x, y):
    """
    A counter for the `thop` package to count the operations in an
    attention operation.
    Meant to be used like:
        macs, params = thop.profile(
            model,
            inputs=(inputs, timestamps),
            custom_ops={QKVAttention: QKVAttention.count_flops},
        )
    """
    b, c, *spatial = y[0].shape
    num_spatial = int(np.prod(spatial))
    # We perform two matmuls with the same number of ops.
    # The first computes the weight matrix, the second computes
    # the combination of the value vectors.
    matmul_ops = 2 * b * (num_spatial ** 2) * c
    model.total_ops += th.DoubleTensor([matmul_ops])


class QKVAttentionLegacy(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention and splits in a different order.
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class UNetModel(nn.Module):
    """
    The full UNet model with attention and timestep embedding.

    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    """

    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        ch = input_ch = int(channel_mult[0] * model_channels)
        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, in_channels, ch, 3, padding=1))]
        )
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(mult * model_channels)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                use_new_attention_order=use_new_attention_order,
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=int(model_channels * mult),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(model_channels * mult)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, input_ch, out_channels, 3, padding=1)),
        )

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)

    def forward(self, x, timesteps, y=None, index=None, t_edit=400, hs_coeff=(1.0, 1.0), delta_h=None, ignore_timestep=False, use_mask=False):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
        h = self.middle_block(h, emb)

        middle_h = h
        h2 = None
        
        if index is not None:
            # assert len(hs_coeff) == index + 1 + 1
            # check t_edit 
            if timesteps[0] >= t_edit:
                # use DeltaBlock
                if delta_h is None: #RemEdit
                    h2 = h * hs_coeff[0]
                    for i in range(index+1):
                        delta_h = getattr(self, f"layer_{i}")(h, None if ignore_timestep else emb)
                        h2 += delta_h * hs_coeff[i+1]
                # use input delta_h  : even tough you does not use DeltaBlock, you need to use index is 0.
                else:  #DiffStyle # DiffStyle; Just ignore this code. We will update about it in README.md later.
                    if use_mask:
                        mask = th.zeros_like(h)
                        mask[:,:,4:-1,3:5] = 1.0
                        inverted_mask = 1 - mask

                        masked_delta_h = delta_h * mask
                        masked_h = h * mask

                        partial_h2 = slerp(1-hs_coeff[0], masked_h, masked_delta_h)
                        h2 = partial_h2 + inverted_mask * h


                    else:
                        h_shape = h.shape
                        h_copy = h.clone().view(h_shape[0],-1)
                        delta_h_copy = delta_h.clone().view(h_shape[0],-1)

                        h_norm = th.norm(h_copy, dim=1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                        delta_h_norm = th.norm(delta_h_copy, dim=1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                        normalized_delta_h = h_norm * delta_h / delta_h_norm
                        
                        h2 = slerp(1.0-hs_coeff[0], h, normalized_delta_h)

            # when t[0] < t_edit : pass the delta_h
            else:
                h2 = h

            hs_index = -1
            for module in self.output_blocks:
                h2 = th.cat([h2, hs[hs_index]], dim=1)
                hs_index -= 1
                h2 = module(h2, emb)
            h2 = h2.type(x.dtype)
            h2 = self.out(h2)

        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
        h = h.type(x.dtype)

        h = self.out(h)
        return h, h2, delta_h, middle_h


    def setattr_layers(self, nums):
        ch = int(self.channel_mult[0] * self.model_channels)

        for level, mult in enumerate(self.channel_mult):
            for _ in range(self.num_res_blocks):
                ch = int(mult * self.model_channels)

        # for i in range(nums):
            # setattr(self, f"layer_{i}", DeltaBlock(in_channels=ch,
            #                            out_channels=ch,
            #                            temb_channels=self.model_channels * 4,
            #                            dropout=0.0)
            # )
            # setattr(self, f"layer_{i}", DeltaBlock(channels=ch,
            #                            emb_channels=self.model_channels * 4,
            #                            dropout=0.0
            #                            )
            # )
        for i in range(nums):
            setattr(
                self,
                f"layer_{i}",
                DeltaBlock(
                    in_channels=block_in,
                    out_channels=block_in,
                    temb_channels=self.temb_ch,
                    dropout=0.0,
                    layer_type=self.db_layer_type,
                    nheads=self.db_nheads,
                    num_layers=self.db_num_layers,
                    dim_feedforward=self.db_dim_feedforward,
                    emb_type=self.db_emb_type,
                    # use_midblock=self.use_midblock
                ),
            )


class DeltaBlock(nn.Module):
    def __init__(
        self, *, 
        in_channels, 
        out_channels=None, 
        conv_shortcut=False, 
        dropout=0.1, 
        temb_channels=512, 
        layer_type="conv",
        nheads=1, 
        num_layers=1, 
        dim_feedforward=2048, 
        emb_type="add", 
        use_midblock=False
    ):

class CLIPWrapper(nn.Module):
    def __init__(self, model_name="ViT-B/16", device="cuda"):
        super().__init__()
        self.model, _ = clip.load(model_name, device=device)
        self.model.eval()
        self.device = device

        # Normalize transform for CLIP input
        self.clip_mean = torch.tensor([0.4815, 0.4578, 0.4082], device=device).view(1, 3, 1, 1)
        self.clip_std = torch.tensor([0.2686, 0.2613, 0.2758], device=device).view(1, 3, 1, 1)

    def normalize(self, img):
        return (img - self.clip_mean) / self.clip_std

    def encode_image(self, img_tensor):
        """
        img_tensor: [B, 3, H, W] float32 in [0, 1] range
        Returns: [B, 512] CLIP image embeddings
        """
        assert img_tensor.shape[1] == 3, "CLIP expects 3-channel RGB images"
        img_resized = F.interpolate(img_tensor, size=224, mode='bicubic', align_corners=False)
        img_norm = self.normalize(img_resized)
        return self.model.encode_image(img_norm)

# Initialization (once)
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_encoder = CLIPWrapper("ViT-B/32", device=device)

class RiemannianBlock(nn.Module):
    """
    A DeltaBlock variant that learns a Riemannian geodesic update:
    1) Projects features and time embedding via 1×1 conv + linear.
    2) Normalizes via GroupNorm.
    3) Applies the Riemannian exponential map to compute Δh.
    """
    def __init__(self, in_channels, out_channels, temb_channels, num_groups=32):
        super().__init__()
        # 1×1 convolution to project input features to the geodesic space
        self.conv_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)
        # Linear layer for the timestep embedding injection
        self.temb_proj = nn.Linear(temb_channels, out_channels)
        # GroupNorm over channels (spatial-agnostic)
        self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels, eps=1e-6, affine=True)
        # ExponentialMap module: true geodesic integration
        self.exp_map = ExponentialMapTrue(out_channels, temb_channels)

    def forward(self, h, temb):
        """
        Args:
            h    (torch.Tensor): Input feature map, shape [B, C_in, H, W].
            temb (torch.Tensor): Timestep embedding, shape [B, temb_channels].
        Returns:
            delta_h (torch.Tensor): Geodesic update, shape [B, C_out, H, W].
        """
        # Project spatial features via 1×1 conv
        h_proj = self.conv_proj(h)  # [B, C_out, H, W]

        # Project time embedding and broadcast over spatial dimensions
        t_proj = self.temb_proj(temb)[:, :, None, None]  # [B, C_out, 1, 1]

        # Combine and normalize
        h_comb = h_proj + t_proj                         # [B, C_out, H, W]
        h_norm = self.norm1(h_comb)                      # GroupNorm across channels

        # Compute Riemannian geodesic update Δh
        delta_h = self.exp_map(h_norm, temb)              # [B, C_out, H, W]
        return delta_h


class DeltaBlock(nn.Module):
    def __init__(
        self, *, 
        in_channels, 
        out_channels=None, 
        conv_shortcut=False, 
        dropout=0.1, 
        temb_channels=512, 
        layer_type="conv",
        nheads=1, 
        num_layers=1, 
        dim_feedforward=2048, 
        emb_type="add", 
        use_midblock=False
    ):

        super().__init__()
        # self.use_midblock = use_midblock
        self.emb_type = emb_type

        # if use_midblock:
        #     self.model = UNetMidBlock2DCrossAttn(512, 512, cross_attention_dim=512)
        # else:
        out_channels = out_channels or in_channels

        # self.mamba_in = Mamba(d_model=in_channels, d_state=32, d_conv=4, expand=4)
        self.in_layer = nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0)
        # self.mamba_out = Mamba(d_model=out_channels, d_state=32, d_conv=4, expand=4)
        self.out_layer = nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0)

        self.temb_proj = nn.Linear(temb_channels, out_channels)
        self.norm2 = Normalize(out_channels)
        self.final_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1)

        if self.emb_type == "adagn":
            self.adagn = AdaGroupNorm(embedding_dim=512, out_dim=512, num_groups=32)

        # Explicit Riemannian Block at the end
        self.riemannian_block = RiemannianBlock(out_channels, out_channels, temb_channels)

    # def forward(self, x, temb=None):
    def forward(self, x, temb=None, text_emb=None):
        # if self.use_midblock:
        #     return self.model(x, temb)

        h = self.in_layer(x)

        # batch, channels, height, width = x.shape

        # Input projection
        # h_flat = x.view(batch, channels, height * width).permute(0, 2, 1)
        # h = self.mamba_in(h_flat).permute(0, 2, 1).view(batch, channels, height, width)

        # Temporal embedding projection
        if temb is not None:
            temb_proj = self.temb_proj(nonlinearity(temb))[:, :, None, None]

            if self.emb_type == "add":
                h = self.norm2(h + temb_proj)
            elif self.emb_type == "mult":
                h = self.norm2(h * temb_proj)
            elif self.emb_type == "adagn":
                h = self.adagn(h, temb)

            h = nonlinearity(h)

        h = self.out_layer(h)

        # Output projection
        # h_flat = h.view(batch, channels, height * width).permute(0, 2, 1)
        # h = self.mamba_out(h_flat).permute(0, 2, 1).view(batch, channels, height, width)

        delta_h = self.riemannian_block(h, temb)
        # delta_h = self.riemannian_block(h, temb, text_emb=text_emb)
        h = h + delta_h

        h = self.final_conv(h)

        return h
