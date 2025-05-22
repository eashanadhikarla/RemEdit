import math
import torch
import torch.nn as nn
# from diffusers.models.attention import AdaGroupNorm
from diffusers.models.attention import AdaLayerNorm as AdaGroupNorm
from diffusers.models.unet_2d_blocks import UNetMidBlock2DCrossAttn

from mamba_ssm import Mamba, Mamba2
import torch.nn.functional as F


def slerp(t, v0, v1):
    _shape = v0.shape

    v0_origin = v0.clone()
    v1_origin = v1.clone()

    v0_copy = v0.view(_shape[0], -1)
    v1_copy = v1.view(_shape[0], -1)

    # Normalize the vectors to get the directions and angles
    v0 = v0 / torch.norm(v0_copy, dim=1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    v1 = v1 / torch.norm(v1_copy, dim=1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

    v0_copy = v0.view(_shape[0], -1)
    v1_copy = v1.view(_shape[0], -1)

    # Dot product with the normalized vectors (can't use np.dot in W)
    dot = torch.sum(v0_copy * v1_copy, dim=1, keepdim=True).squeeze(-1)

    # Calculate initial angle between v0 and v1
    theta_0 = torch.acos(dot)
    sin_theta_0 = torch.sin(theta_0)

    # Angle at timestep t
    theta_t = theta_0 * t
    sin_theta_t = torch.sin(theta_t)

    # Finish the slerp algorithm
    s0 = torch.sin(theta_0 - theta_t) / sin_theta_0
    s1 = sin_theta_t / sin_theta_0
    s0 = s0.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    s1 = s1.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    v2 = s0 * v0_origin + s1 * v1_origin
    return v2


def get_timestep_embedding(timesteps, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


def nonlinearity(x):
    # swish
    return x * torch.sigmoid(x)


def Normalize(in_channels):
    return torch.nn.GroupNorm(
        num_groups=32, num_channels=in_channels, eps=1e-6, affine=True
    )


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=1, padding=1
            )

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=2, padding=0
            )

    def forward(self, x):
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):
    def __init__(
        self,
        *,
        in_channels,
        out_channels=None,
        conv_shortcut=False,
        dropout,
        temb_channels=512,
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.temb_proj = torch.nn.Linear(temb_channels, out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(
                    in_channels, out_channels, kernel_size=3, stride=1, padding=1
                )
            else:
                self.nin_shortcut = torch.nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=1, padding=0
                )

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.k = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.v = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.proj_out = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w)
        q = q.permute(0, 2, 1)  # b,hw,c
        k = k.reshape(b, c, h * w)  # b,c,hw
        w_ = torch.bmm(q, k)  # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c) ** (-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h * w)
        w_ = w_.permute(0, 2, 1)  # b,hw,hw (first hw of k, second of q)
        # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = torch.bmm(v, w_)
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        return x + h_


class DeltaBlock_global(nn.Module):
    def __init__(
        self,
        *,
        in_channels,
        out_channels=None,
        conv_shortcut=False,
        dropout,
        temb_channels=512,
        clip_channels=512,
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.conv1 = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.temb_proj = torch.nn.Linear(temb_channels, out_channels)
        self.clip_proj = torch.nn.Linear(clip_channels, out_channels)
        self.clip_proj_2 = torch.nn.Linear(clip_channels, 512 * 8 * 8)
        self.norm2 = Normalize(out_channels)
        self.conv2 = torch.nn.Conv2d(
            out_channels, out_channels, kernel_size=1, stride=1, padding=0
        )
        self.norm3 = Normalize(out_channels)
        self.conv3 = torch.nn.Conv2d(
            out_channels, out_channels, kernel_size=1, stride=1, padding=0
        )

        self.norm4 = Normalize(out_channels)
        self.conv4 = torch.nn.Conv2d(
            out_channels, out_channels, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x, temb, clip_direction):
        h = x

        h = self.conv1(h)
        h = (
            h
            + self.temb_proj(nonlinearity(temb))[:, :, None, None]
            + self.clip_proj(clip_direction)[:, :, None, None]
        )
        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.conv2(h)
        clip_pro = self.clip_proj_2(clip_direction).reshape(1, 512, 8, 8)
        h = h + clip_pro
        h = self.norm3(h)
        h = nonlinearity(h)
        h = self.conv3(h)
        h = self.norm4(h)
        h = nonlinearity(h)
        h = self.conv4(h)
        return h


class DDPM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        ch, out_ch, ch_mult = (
            config.model.ch,
            config.model.out_ch,
            tuple(config.model.ch_mult),
        )
        num_res_blocks = config.model.num_res_blocks
        attn_resolutions = config.model.attn_resolutions
        dropout = config.model.dropout
        in_channels = config.model.in_channels
        resolution = config.data.image_size
        resamp_with_conv = config.model.resamp_with_conv

        self.ch = ch
        self.temb_ch = self.ch * 4
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # timestep embedding
        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList(
            [
                torch.nn.Linear(self.ch, self.temb_ch),
                torch.nn.Linear(self.temb_ch, self.temb_ch),
            ]
        )

        # downsampling
        self.conv_in = torch.nn.Conv2d(
            in_channels, self.ch, kernel_size=3, stride=1, padding=1
        )

        curr_res = resolution
        in_ch_mult = (1,) + ch_mult
        self.down = nn.ModuleList()
        block_in = None
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            skip_in = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                if i_block == self.num_res_blocks:
                    skip_in = ch * in_ch_mult[i_level]
                block.append(
                    ResnetBlock(
                        in_channels=block_in + skip_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(
            block_in, out_ch, kernel_size=3, stride=1, padding=1
        )

    def setattr_layers(self, nums):
        ch, ch_mult = self.config.model.ch, tuple(self.config.model.ch_mult)
        block_in = None
        for i_level in range(self.num_resolutions):
            block_in = ch * ch_mult[i_level]

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

    def setattr_global_layer(self, nums):
        ch, ch_mult = self.config.model.ch, tuple(self.config.model.ch_mult)
        block_in = None
        for i_level in range(self.num_resolutions):
            block_in = ch * ch_mult[i_level]

        setattr(
            self,
            "layer_0",
            DeltaBlock_global(
                in_channels=block_in,
                out_channels=block_in,
                temb_channels=self.temb_ch,
                dropout=0.0,
            ),
        )

    def get_temb(self, t):
        # timestep embedding
        temb = get_timestep_embedding(t, self.ch)
        temb = self.temb.dense[0](temb)
        temb = nonlinearity(temb)
        temb = self.temb.dense[1](temb)
        return temb

    # def forward(self, x, t, index=None, t_edit=400, hs_coeff=(1.0, 1.0), delta_h=None, ignore_timestep=False, use_mask=False):
    def forward(self, x, t, index=None, t_edit=400, hs_coeff=(1.0, 1.0), delta_h=None, ignore_timestep=False, use_mask=False, clip_direction=None):
        # print(f"[DEBUG] Input shape to model: {x.shape}, expected resolution: {self.resolution}")
        assert x.shape[2] == x.shape[3] == self.resolution

        # timestep embedding
        temb = get_timestep_embedding(t, self.ch)
        temb = self.temb.dense[0](temb)
        temb = nonlinearity(temb)
        temb = self.temb.dense[1](temb)
        cnt = 0

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)

            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)
        middle_h = h
        h2 = None

        if index is not None:
            # assert len(hs_coeff) == index + 1 + 1
            # check t_edit
            if t[0] >= t_edit:
                # use DeltaBlock
                if delta_h is None:  # Asyrp
                    h2 = h * hs_coeff[0]
                    for i in range(index + 1):
                        # delta_h = getattr(self, f"layer_{i}")(
                        #     h, None if ignore_timestep else temb
                        # )
                        delta_h = getattr(self, f"layer_{i}")(
                            h, None if ignore_timestep else temb, text_emb=clip_direction
                        )
                        h2 += delta_h * hs_coeff[i + 1]

                # use input delta_h  : even tough you does not use DeltaBlock, you need to use index is 0.
                else:  # DiffStyle; Just ignore this code. We will update about it in README.md later.
                    if use_mask:
                        mask = torch.zeros_like(h)
                        mask[:, :, 4:-1, 3:5] = 1.0
                        inverted_mask = 1 - mask

                        masked_delta_h = delta_h * mask
                        masked_h = h * mask

                        partial_h2 = slerp(1 - hs_coeff[0], masked_h, masked_delta_h)
                        h2 = partial_h2 + inverted_mask * h

                    else:
                        h_shape = h.shape
                        h_copy = h.clone().view(h_shape[0], -1)
                        delta_h_copy = delta_h.clone().view(h_shape[0], -1)

                        h_norm = (torch.norm(h_copy, dim=1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1))
                        delta_h_norm = (torch.norm(delta_h_copy, dim=1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1))
                        normalized_delta_h = h_norm * delta_h / delta_h_norm
                        h2 = slerp(1.0 - hs_coeff[0], h, normalized_delta_h)
            # when t[0] < t_edit : pass the delta_h
            else:
                h2 = h

            hs_index = -1

            for i_level in reversed(range(self.num_resolutions)):
                for i_block in range(self.num_res_blocks + 1):
                    h2 = self.up[i_level].block[i_block](
                        torch.cat([h2, hs[hs_index]], dim=1), temb
                    )
                    hs_index -= 1
                    if len(self.up[i_level].attn) > 0:
                        h2 = self.up[i_level].attn[i_block](h2)
                if i_level != 0:
                    h2 = self.up[i_level].upsample(h2)

            # end
            h2 = self.norm_out(h2)
            h2 = nonlinearity(h2)
            h2 = self.conv_out(h2)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](
                    torch.cat([h, hs.pop()], dim=1), temb
                )
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)

            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h, h2, delta_h, middle_h

    def forward_layer_check(self, x, t, index=None, t_edit=400, hs_coeff=(1.0, 1.0), delta_h=None, ignore_timestep=False):
        assert x.shape[2] == x.shape[3] == self.resolution

        # timestep embedding
        temb = get_timestep_embedding(t, self.ch)
        temb = self.temb.dense[0](temb)
        temb = nonlinearity(temb)
        temb = self.temb.dense[1](temb)
        cnt = 0

        print(f"{cnt}<-x.shape:{x.shape}")
        cnt += 1

        # downsampling
        hs = [self.conv_in(x)]
        print(f"{cnt}<-h.shape:{hs[-1].shape}")
        cnt += 1
        for i_level in range(self.num_resolutions):
            if i_level > 0.1:
                print(f"{cnt}<-h.shape:{h.shape},i_level:{i_level}")
                cnt += 1
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)

            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        print(f"{cnt}<-mid,h.shape:{h.shape}")
        cnt += 1
        h = self.mid.block_1(h, temb)
        print(f"{cnt}<-mid,h.shape:{h.shape}")
        cnt += 1
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)
        middle_h = h
        h2 = None

        if index is not None:
            assert len(hs_coeff) == index + 1 + 1
            # check t_edit
            if t[0] >= t_edit:
                # use DeltaBlock
                if delta_h is None:
                    h2 = h * hs_coeff[0]
                    for i in range(index + 1):
                        delta_h = getattr(self, f"layer_{i}")(
                            h, None if ignore_timestep else temb
                        )
                        # delta_h = getattr(self, f"layer_{i}")(
                        #     h, None if ignore_timestep else temb, text_emb=clip_direction
                        # )
                        h2 += delta_h * hs_coeff[i + 1]
                # use input delta_h  : even tough you does not use DeltaBlock, you need to use index is 0.
                else:
                    h2 = h * hs_coeff[0] + delta_h * hs_coeff[1]
            # when t[0] < t_edit : pass the delta_h
            else:
                h2 = h

            hs_index = -1

            for i_level in reversed(range(self.num_resolutions)):
                for i_block in range(self.num_res_blocks + 1):
                    h2 = self.up[i_level].block[i_block](
                        torch.cat([h2, hs[hs_index]], dim=1), temb
                    )
                    hs_index -= 1
                    if len(self.up[i_level].attn) > 0:
                        h2 = self.up[i_level].attn[i_block](h2)
                if i_level != 0:
                    h2 = self.up[i_level].upsample(h2)

            # end
            h2 = self.norm_out(h2)
            h2 = nonlinearity(h2)
            h2 = self.conv_out(h2)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            print(f"{cnt}<-h.shape:{h.shape},i_level:{i_level}")
            cnt += 1
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](
                    torch.cat([h, hs.pop()], dim=1), temb
                )
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)

            if i_level != 0:
                h = self.up[i_level].upsample(h)

        print(f"{cnt}<-,h.shape:{h.shape}")
        cnt += 1
        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        print(f"{cnt}<-h.shape:{h.shape}")
        cnt += 1

        import pdb
        pdb.set_trace()

        return h, h2, delta_h, middle_h

    def multiple_attr(self, x, t, index=None, maintain=400, rambda=(1.0, 1.0)):
        assert x.shape[2] == x.shape[3] == self.resolution

        # timestep embedding
        temb = get_timestep_embedding(t, self.ch)
        temb = self.temb.dense[0](temb)
        temb = nonlinearity(temb)
        temb = self.temb.dense[1](temb)

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        if index is not None:
            if t[0] >= maintain:
                delta_h_sum = None
                for i in range(index):
                    delta_h = getattr(self, f"layer_{i}")(h, temb)
                    if i == 0:
                        delta_h_sum = delta_h * rambda[0]
                    else:
                        delta_h_sum = delta_h_sum + delta_h * rambda[i]

                h2 = h + delta_h_sum / (index) ** (1 / 2)
            else:
                h2 = h

            hs_index = -1

            for i_level in reversed(range(self.num_resolutions)):
                for i_block in range(self.num_res_blocks + 1):
                    h2 = self.up[i_level].block[i_block](
                        torch.cat([h2, hs[hs_index]], dim=1), temb
                    )
                    hs_index -= 1
                    if len(self.up[i_level].attn) > 0:
                        h2 = self.up[i_level].attn[i_block](h2)
                if i_level != 0:
                    h2 = self.up[i_level].upsample(h2)

            # end
            h2 = self.norm_out(h2)
            h2 = nonlinearity(h2)
            h2 = self.conv_out(h2)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](
                    torch.cat([h, hs.pop()], dim=1), temb
                )
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)

        if index is not None:
            return h, h2
        else:
            return h

    def interpolation2(self, x, t, index=None, maintain=400, alpha=None):
        assert x.shape[2] == x.shape[3] == self.resolution

        # timestep embedding
        temb = get_timestep_embedding(t, self.ch)
        temb = self.temb.dense[0](temb)
        temb = nonlinearity(temb)
        temb = self.temb.dense[1](temb)

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        if index is not None:
            if t[0] >= maintain:
                h_index_0 = torch.stack([h[0] for i in range(h.shape[0])])
                h_index_last = torch.stack([h[-1] for i in range(h.shape[0])])
                alpha = alpha.unsqueeze(1).unsqueeze(2).unsqueeze(3)
                h2 = (1 - alpha) * h_index_0 + alpha * h_index_last
            else:
                h2 = h

            hs_index = -1

            for i_level in reversed(range(self.num_resolutions)):
                for i_block in range(self.num_res_blocks + 1):
                    h2 = self.up[i_level].block[i_block](
                        torch.cat([h2, hs[hs_index]], dim=1), temb
                    )
                    hs_index -= 1
                    if len(self.up[i_level].attn) > 0:
                        h2 = self.up[i_level].attn[i_block](h2)
                if i_level != 0:
                    h2 = self.up[i_level].upsample(h2)

            # end
            h2 = self.norm_out(h2)
            h2 = nonlinearity(h2)
            h2 = self.conv_out(h2)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](
                    torch.cat([h, hs.pop()], dim=1), temb
                )
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)

        if index is not None:
            return h, h2
        else:
            return h

    def forward_at(self, x, t, index=None):
        assert x.shape[2] == x.shape[3] == self.resolution

        # timestep embedding
        temb = get_timestep_embedding(t, self.ch)
        temb = self.temb.dense[0](temb)
        temb = nonlinearity(temb)
        temb = self.temb.dense[1](temb)

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        if index is not None:
            delta_h = getattr(self, f"layer_{index}")(h, temb)  # .roll(1, dims=3)
            h2 = h + delta_h

            hs_index = -1

            for i_level in reversed(range(self.num_resolutions)):
                for i_block in range(self.num_res_blocks + 1):
                    h2 = self.up[i_level].block[i_block](
                        torch.cat([h2, hs[hs_index]], dim=1), temb
                    )
                    hs_index -= 1
                    if len(self.up[i_level].attn) > 0:
                        h2 = self.up[i_level].attn[i_block](h2)
                if i_level != 0:
                    h2 = self.up[i_level].upsample(h2)

            # end
            h2 = self.norm_out(h2)
            h2 = nonlinearity(h2)
            h2 = self.conv_out(h2)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](
                    torch.cat([h, hs.pop()], dim=1), temb
                )
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)

        if index is not None:
            return h, h2
        else:
            return h

    def forward_global(self, x, t, index=None, maintain=400, direction=None):
        assert x.shape[2] == x.shape[3] == self.resolution

        # timestep embedding
        temb = get_timestep_embedding(t, self.ch)
        temb = self.temb.dense[0](temb)
        temb = nonlinearity(temb)
        temb = self.temb.dense[1](temb)

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        if index is not None:
            if t[0] >= maintain:
                delta_h = getattr(self, "layer_0")(
                    h, temb, direction
                )  # .roll(1, dims=3)
                h2 = h + delta_h
            else:
                h2 = h

            hs_index = -1

            for i_level in reversed(range(self.num_resolutions)):
                for i_block in range(self.num_res_blocks + 1):
                    h2 = self.up[i_level].block[i_block](
                        torch.cat([h2, hs[hs_index]], dim=1), temb
                    )
                    hs_index -= 1
                    if len(self.up[i_level].attn) > 0:
                        h2 = self.up[i_level].attn[i_block](h2)
                if i_level != 0:
                    h2 = self.up[i_level].upsample(h2)

            # end
            h2 = self.norm_out(h2)
            h2 = nonlinearity(h2)
            h2 = self.conv_out(h2)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](
                    torch.cat([h, hs.pop()], dim=1), temb
                )
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)

        if index is not None:
            return h, h2
        else:
            return h

# ==================================================
## Exponential Map code moved to another file
# ==================================================
from models.ddpm.expomap import ExponentialMapVanilla, ExponentialMapVanilla2
# from models.ddpm.expomap import ODEExponentialMap
# from models.ddpm.expomap import HyperScaleExponentialMap

import clip
import torchvision.transforms as T

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
    def __init__(self, in_channels, out_channels, temb_channels):
        super(RiemannianBlock, self).__init__()
        
        # Linear projection to transform latent space into geodesic calculation space
        self.linear = nn.Linear(in_channels, out_channels)
        self.temb_proj = nn.Linear(temb_channels, out_channels)
        self.norm1 = nn.LayerNorm(out_channels)

        # # ExponentialMap for accurate geodesic computation
        # self.exp_map = ExponentialMapVanilla(out_channels, temb_channels)
        # self.exp_map = ExponentialMap(out_channels, temb_channels, text_dim=512)
        # self.exp_map = ExponentialMap(out_channels, temb_channels, text_dim=512, clip_model=clip_encoder)
        self.exp_map = ExponentialMapVanilla2(out_channels, temb_channels, text_dim=1024) #, clip_model=clip_encoder)

        # # Explicitly use ODEExponentialMap
        # self.exp_map = ODEExponentialMap(out_channels, temb_channels)
        # self.exp_map = ODEExponentialMap(out_channels, temb_channels, text_dim=512)
        # self.exp_map = ODEExponentialMap(out_channels, temb_channels, text_dim=512, clip_model=clip_encoder)

        # Hyperscale
        # self.exp_map = HyperScaleExponentialMap(out_channels, temb_channels)

    # def forward(self, h, temb):
    def forward(self, h, temb, text_emb=None):
        batch_size, channels, height, width = h.shape
        h_flat = h.view(batch_size, channels, -1).permute(0, 2, 1)  # [batch, height*width, channels]

        # Project input features and temporal embeddings
        h_proj = self.linear(h_flat)  # [batch, height*width, out_channels]
        temb_proj = self.temb_proj(temb).unsqueeze(1)  # [batch, 1, out_channels]

        # Apply transformation and normalization
        h_proj = self.norm1(h_proj + temb_proj)

        # Reshape back to spatial dimensions
        h_proj = h_proj.permute(0, 2, 1).view(batch_size, -1, height, width)

        # Compute explicit geodesic movement via exponential map
        # delta_h = self.exp_map(h_proj, temb)
        delta_h = self.exp_map(h_proj, temb, text_emb=text_emb)
        # delta_h = self.exp_map(h_proj, temb, text_emb=text_emb, use_geodesic=(text_emb is not None))

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

        # delta_h = self.riemannian_block(h, temb)
        delta_h = self.riemannian_block(h, temb, text_emb=text_emb)
        h = h + delta_h

        h = self.final_conv(h)

        return h
