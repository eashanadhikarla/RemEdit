import math
import torch
import torch.nn as nn
# from diffusers.models.attention import AdaGroupNorm
from diffusers.models.attention import AdaLayerNorm as AdaGroupNorm
from diffusers.models.unet_2d_blocks import UNetMidBlock2DCrossAttn

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
    # If absolute value of dot product is almost 1, vectors are ~colineal, so use lerp
    # if torch.abs(dot) > 0.9995:
    #     return lerp(t, v0, v1)
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
    # v2 = v2.view(_shape)
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

    def forward(self, x, t, index=None, t_edit=400, hs_coeff=(1.0, 1.0), delta_h=None, ignore_timestep=False, use_mask=False):
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
                        delta_h = getattr(self, f"layer_{i}")(
                            h, None if ignore_timestep else temb
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



from mamba_ssm import Mamba, Mamba2
import torch.nn.functional as F

# class EnhancedRiemannianBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, temb_channels, geodesic_scale=2.0):
#         super(EnhancedRiemannianBlock, self).__init__()
#         self.linear = nn.Linear(in_channels, out_channels)
#         self.temb_proj = nn.Linear(temb_channels, out_channels)
#         self.norm1 = nn.LayerNorm(out_channels)
#         self.geodesic_scale = geodesic_scale

#     def forward(self, h, temb):
#         batch_size, channels, height, width = h.shape
#         h_flat = h.view(batch_size, channels, -1).permute(0, 2, 1)

#         h_proj = self.linear(h_flat)
#         temb_proj = self.temb_proj(temb).unsqueeze(1)

#         h_proj = self.norm1(h_proj + temb_proj)
#         h_proj = h_proj.permute(0, 2, 1).view(batch_size, -1, height, width)

#         delta_h = self.compute_geodesic(h_proj)
#         return delta_h

#     def compute_geodesic(self, h_proj):
#         norm = torch.norm(h_proj, dim=1, keepdim=True)
#         geodesic_direction = h_proj / (norm + 1e-5)
#         geodesic_magnitude = torch.log(1 + norm) * self.geodesic_scale

#         return geodesic_direction * geodesic_magnitude

class RiemannianBlock(nn.Module):
    def __init__(self, in_channels, out_channels, temb_channels):
        super(RiemannianBlock, self).__init__()
        # Linear projection for geodesic calculation
        self.linear = nn.Linear(in_channels, out_channels)
        self.temb_proj = nn.Linear(temb_channels, out_channels)
        # Layer to compute geodesic direction (instead of straight Euclidean shifts)
        self.norm1 = nn.LayerNorm(out_channels)

    def forward(self, h, temb):
        # Reshape h to (batch_size, channels, height * width)
        batch_size, channels, height, width = h.shape
        h_flat = h.view(batch_size, channels, -1).permute(0, 2, 1)  # (batch_size, height * width, channels)

        # # Project the input h-space features and temporal embeddings by applying linear layer
        h_proj = self.linear(h_flat)  # (batch_size, height * width, out_channels)
        # Project temporal embedding
        temb_proj = self.temb_proj(temb).unsqueeze(1)  # (batch_size, 1, out_channels)

        # Apply Riemannian-based transformation
        h_proj = self.norm1(h_proj + temb_proj)
        # Reshape back to (batch_size, out_channels, height, width)
        h_proj = h_proj.permute(0, 2, 1).view(batch_size, -1, height, width)

        # Geodesic distance approximation in latent space
        delta_h = self.compute_geodesic(h_proj)

        return delta_h

    def compute_geodesic(self, h_proj):
        # Compute the geodesic distance along the manifold
        norm = torch.norm(h_proj, dim=1, keepdim=True)
        geodesic_direction = h_proj / (norm + 1e-5)  # normalize
        geodesic_magnitude = torch.log(1 + norm)  # use log map for geodesic distance
        
        return geodesic_direction * geodesic_magnitude


class DeltaBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False, dropout, temb_channels=512,
                 layer_type="mamba", nheads=1, num_layers=1, dim_feedforward=2048, emb_type="add",
                 use_clip=True, use_contrastive=True, geodesic_scale=2.0, adaptive_scaling=True):

        super().__init__()
        self.emb_type = emb_type
        self.layer_type = layer_type
        self.use_clip = use_clip
        self.use_contrastive = use_contrastive
        self.adaptive_scaling = adaptive_scaling
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels

        self.in_layer = torch.nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0)

        # Initialize Mamba within DeltaBlock
        self.mamba_layer = Mamba(d_model=512, d_state=32, d_conv=3, expand=2)
        # self.mamba_layer = Mamba2(d_model=512, d_state=64, d_conv=4, expand=2)

        # Temporal Embedding Projection
        self.temb_proj = nn.Linear(temb_channels, self.out_channels)
        self.norm2 = Normalize(self.out_channels)

        if emb_type == "adagn":
            self.adagn = AdaGroupNorm(embedding_dim=512, out_dim=512, num_groups=32)

        # Enhanced Riemannian Block for advanced delta_h manipulation
        self.riemannian_block = RiemannianBlock(in_channels, out_channels, temb_channels)

        if use_clip:
            # self.clip_proj = nn.Linear(512, self.out_channels)  # CLIP projection layer
            self.clip_proj = Mamba(d_model=512, d_state=64, d_conv=4, expand=2)

        if use_contrastive:
            self.contrastive_proj = nn.Linear(self.in_channels, 128)
            self.temperature = 0.07  # Temperature for contrastive loss scaling

    def forward(self, x, temb=None, clip_embedding=None):
        h = x
        # h = self.mamba_layer(h)
        
        batch, channels, height, width = h.shape
        h_flat = h.view(batch, channels, height * width).permute(0, 2, 1)  # [batch, sequence, channels]
        h = self.mamba_layer(h_flat)
        h = h.permute(0, 2, 1).view(batch, channels, height, width)

        # h = self.in_layer(h)

        # print(f"\nh Shape debugging: {h.shape}\n")
        # batch, channels, height, width = h.shape
        # h_flat_0 = h.view(batch, channels, height * width).permute(0, 2, 1)
        # h = self.riemannian_block(h_flat_0, h_flat_0)

        if temb is not None:
            h = self.apply_embedding(h, temb)
        
        # Flatten
        batch, channels, height, width = h.shape
        h_flat = h.view(batch, channels, height * width).permute(0, 2, 1)  # [batch, sequence, channels]
        h = self.mamba_layer(h_flat)
        h = h.permute(0, 2, 1).view(batch, channels, height, width)

        # Enhanced Riemannian manifold block to compute delta_h
        delta_h = self.riemannian_block(h, temb)

        # Contrastive loss for disentanglement (if enabled)
        if self.use_contrastive:
            # print(f"\nShape of h: {h.shape}\n, Shape of delta_h: {delta_h.shape}\n")
            delta_h = self.apply_contrastive_loss(h, delta_h)

        # CLIP embedding modulation if enabled
        if self.use_clip and clip_embedding is not None:
            delta_h += self.clip_proj(clip_embedding)

        # print(f"\nShape of delta_h: {delta_h.shape}\n")
        return delta_h

    def apply_embedding(self, h, temb):
        if self.emb_type == "add":
            h += self.temb_proj(F.relu(temb))[:, :, None, None]
        elif self.emb_type == "mult":
            h *= self.temb_proj(F.relu(temb))[:, :, None, None]
        elif self.emb_type == "adagn":
            h = self.adagn(h, temb)
        return self.norm2(h)

    def apply_contrastive_loss(self, h, delta_h):
        batch_size, channels, height, width = h.shape

        # Reshape h and delta_h to [batch_size, channels, height * width]
        h_flat = h.view(batch_size, channels, -1)  # [9, 512, 64]
        delta_h_flat = delta_h.view(batch_size, channels, -1)  # [9, 512, 64]

        # Apply contrastive projection at the channel level
        h_proj = self.contrastive_proj(h_flat.permute(0, 2, 1))  # [batch_size, height * width, 128]
        delta_h_proj = self.contrastive_proj(delta_h_flat.permute(0, 2, 1))  # [batch_size, height * width, 128]

        # Normalize projections
        h_proj = h_proj / h_proj.norm(dim=-1, keepdim=True)
        delta_h_proj = delta_h_proj / delta_h_proj.norm(dim=-1, keepdim=True)

        # Compute contrastive loss (returns scalar)
        contrastive_loss = self.compute_contrastive_loss(h_proj, delta_h_proj)

        # Broadcast scalar loss to match delta_h shape
        contrastive_loss = contrastive_loss * torch.ones_like(delta_h)  # Expands to [batch_size, channels, height, width]
        
        return delta_h + 0.05*contrastive_loss

    def compute_contrastive_loss(self, h_proj, delta_h_proj):
        sim_matrix = torch.einsum('bik,bjk->bij', h_proj, delta_h_proj) / self.temperature
        batch_size, seq_len, _ = h_proj.shape
        labels = torch.arange(seq_len).repeat(batch_size, 1).cuda()
        
        loss = nn.CrossEntropyLoss()(sim_matrix.view(-1, seq_len), labels.view(-1))
    
        # Ensure loss is a scalar
        return loss.mean()  # Returns a single scalar value


    ## =============================================================
    # def apply_contrastive_loss(self, h, delta_h):
    #     batch_size, channels, height, width = h.shape

    #     # Reshape to [batch_size, channels, height * width]
    #     h_flat = h.view(batch_size, channels, -1)  # [9, 512, 64]
    #     delta_h_flat = delta_h.view(batch_size, channels, -1)  # [9, 512, 64]

    #     # Apply contrastive projection at the channel level
    #     h_proj = self.contrastive_proj(h_flat.permute(0, 2, 1))  # [batch_size, height * width, 128]
    #     delta_h_proj = self.contrastive_proj(delta_h_flat.permute(0, 2, 1))  # [batch_size, height * width, 128]

    #     # Normalize projections
    #     h_proj = h_proj / h_proj.norm(dim=-1, keepdim=True)
    #     delta_h_proj = delta_h_proj / delta_h_proj.norm(dim=-1, keepdim=True)

    #     # Compute contrastive loss
    #     contrastive_loss = self.compute_contrastive_loss(h_proj, delta_h_proj)

    #     # Reshape contrastive loss to match delta_h shape
    #     contrastive_loss = contrastive_loss.view(batch_size, 1, height, width)
    #     return delta_h + contrastive_loss 

    # def compute_contrastive_loss(self, h_proj, delta_h_proj):
    #     # Compute cosine similarity correctly along the feature dimension
    #     sim_matrix = torch.einsum('bik,bjk->bij', h_proj, delta_h_proj) / self.temperature  # [batch, height*width, height*width]

    #     # Ensure labels are correctly shaped
    #     batch_size, seq_len, _ = h_proj.shape
    #     labels = torch.arange(seq_len).repeat(batch_size, 1).cuda()  # Align label shape with similarity matrix

    #     loss = nn.CrossEntropyLoss()(sim_matrix.view(-1, seq_len), labels.view(-1))
    #     return loss
    ## =============================================================



    ## =============================================================
    # def apply_contrastive_loss(self, h, delta_h):
    #     # Flatten h and delta_h to [batch_size, features] for linear projection
    #     batch_size, channels, height, width = h.shape
    #     h_flat = h.reshape(batch_size, -1)  # [batch_size, channels * height * width]
    #     delta_h_flat = delta_h.reshape(batch_size, -1)  # [batch_size, channels * height * width]

    #     # Apply contrastive projection
    #     h_proj = self.contrastive_proj(h_flat)  # [batch_size, 128]
    #     delta_h_proj = self.contrastive_proj(delta_h_flat)  # [batch_size, 128]

    #     # Normalize projections
    #     h_proj = h_proj / h_proj.norm(dim=-1, keepdim=True)
    #     delta_h_proj = delta_h_proj / delta_h_proj.norm(dim=-1, keepdim=True)

    #     # Calculate contrastive loss and add it to delta_h
    #     contrastive_loss = self.compute_contrastive_loss(h_proj, delta_h_proj)
        
    #     # Reshape contrastive loss to match delta_h shape
    #     contrastive_loss = contrastive_loss.view(batch_size, channels, height, width)
    #     return delta_h + contrastive_loss

    # def compute_contrastive_loss(self, h_proj, delta_h_proj):
    #     sim_matrix = torch.matmul(h_proj, delta_h_proj.T) / self.temperature
    #     labels = torch.arange(h_proj.size(0)).cuda()
    #     loss = nn.CrossEntropyLoss()(sim_matrix, labels)
    #     return loss
    ## =============================================================


# def get_dh_layer(layer_name, nheads, num_layers, dim_feedforward=2048, dropout=0.1):
#     # defaults were
#     # nheads = 8, dim_feedforward = 2048, dropout = 0.1, num_encoder_layers = 1 (we're not using the decoder layers)
#     layer = None
 
#     # do both of the options in sequence
#     if layer_name == "pc_transformer_simple":
#         layer = DualTransformerSimple(nheads, num_layers, dim_feedforward, dropout, "pc")

#     elif layer_name == "cp_transformer_simple":
#         layer = DualTransformerSimple(nheads, num_layers, dim_feedforward, dropout, "cp")

#     elif layer_name == "c_transformer_simple":
#         layer = TransformerSimple(nheads, num_layers, dim_feedforward, dropout, "channel")

#     elif layer_name == "p_transformer_simple":
#         layer = TransformerSimple(nheads, num_layers, dim_feedforward, dropout, "pixel")

#     elif layer_name == "conv":
#         layer = torch.nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0)

#     else:
#         raise NotImplementedError(f"No layer implemented with name: {layer_name}")
#     return layer


# class TransformerSimple(nn.Module):
#     def __init__(self, nheads, num_layers, dim_feedforward, dropout, model_type="pixel"):
#         super().__init__()
#         if model_type == "pixel":
#             n_features = 512
#         elif model_type == "channel":
#             n_features = 64
#         else:
#             raise NotImplementedError(f"No implementation for type {model_type} of TransformerSimple")
#         self.model_type = model_type

#         transformer = nn.Transformer(
#             d_model=n_features,
#             nhead=nheads,
#             num_encoder_layers=num_layers,
#             num_decoder_layers=0,
#             dim_feedforward=dim_feedforward,
#             batch_first=True,
#             dropout=dropout,
#         )
#         del transformer.decoder
#         self.t_layer = transformer.encoder

#     def forward(self, h):
#         h = torch.reshape(h, (-1, 512, 64))
#         if self.model_type == "pixel":
#             h = h.permute(0, 2, 1)
#             h = self.t_layer(h)
#             h = h.permute(0, 2, 1)
#         elif self.model_type == "channel":
#             h = self.t_layer(h)
#         else:
#             raise NotImplementedError(f"No implementation for type {self.model_type} of TransformerSimple")

#         h = torch.reshape(h, (-1, 512, 8, 8))
#         return h


# class DualTransformerSimple(nn.Module):
#     def __init__(self, nheads, num_layers, dim_feedforward, dropout, order):
#         super().__init__()
#         transformer_in_64 = nn.Transformer(
#             d_model=64,
#             nhead=nheads,
#             num_encoder_layers=num_layers,
#             num_decoder_layers=0,
#             dim_feedforward=dim_feedforward,
#             batch_first=True,
#             dropout=dropout,
#         )
#         del transformer_in_64.decoder
#         self.t_channel = transformer_in_64.encoder

#         transformer_in_512 = nn.Transformer(
#             d_model=512,
#             nhead=nheads,
#             num_encoder_layers=num_layers,
#             num_decoder_layers=0,
#             dim_feedforward=dim_feedforward,
#             batch_first=True,
#             dropout=dropout,
#         )
#         del transformer_in_512.decoder
#         self.t_pixel = transformer_in_512.encoder
#         if order not in ["pc", "cp"]:
#             raise ValueError(f"Order of argument should be 'cp' or 'pc', not '{order}'")
#         self.order = order

#     def forward(self, h):
#         # change input sizes for use with transformer
#         if self.order == "cp":
#             h = torch.reshape(h, (-1, 512, 64))
#             h = self.t_channel(h)
#             h = h.permute(0, 2, 1)
#             h = self.t_pixel(h)
#             h = h.permute(0, 2, 1)
#             h = torch.reshape(h, (-1, 512, 8, 8))
#         elif self.order == "pc":
#             h = torch.reshape(h, (-1, 512, 64))
#             h = h.permute(0, 2, 1)
#             h = self.t_pixel(h)
#             h = h.permute(0, 2, 1)
#             h = self.t_channel(h)
#             h = torch.reshape(h, (-1, 512, 8, 8))
#         else:
#             # this can not happen because of the assert in the init but who knows.
#             raise ValueError(f"Order of argument should be 'cp' or 'pc', not '{self.order}'")
#         return h


# class EnhancedRiemannianBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, temb_channels, geodesic_scale=1.2):
#         super(EnhancedRiemannianBlock, self).__init__()
#         self.linear = nn.Linear(in_channels, out_channels)
#         self.temb_proj = nn.Linear(temb_channels, out_channels)
#         self.norm1 = nn.LayerNorm(out_channels)
#         self.geodesic_scale = geodesic_scale  # Scaling factor for stronger manipulation

#     def forward(self, h, temb):
#         # Reshape h to (batch_size, channels, height * width)
#         batch_size, channels, height, width = h.shape
#         h_flat = h.view(batch_size, channels, -1).permute(0, 2, 1)  # (batch_size, height * width, channels)

#         # Project the input h-space features and temporal embeddings by applying linear layer
#         h_proj = self.linear(h_flat)  # (batch_size, height * width, out_channels)
#         temb_proj = self.temb_proj(temb).unsqueeze(1)  # (batch_size, 1, out_channels)

#         # Apply Riemannian-based transformation
#         h_proj = self.norm1(h_proj + temb_proj)
#         h_proj = h_proj.permute(0, 2, 1).view(batch_size, -1, height, width)

#         # Geodesic distance approximation with stronger scaling
#         delta_h = self.compute_geodesic(h_proj)

#         return delta_h

#     def compute_geodesic(self, h_proj):
#         # Compute the geodesic direction
#         norm = torch.norm(h_proj, dim=1, keepdim=True)
#         geodesic_direction = h_proj / (norm + 1e-5)
#         geodesic_magnitude = torch.log(1 + norm) * self.geodesic_scale  # Amplified scaling

#         return geodesic_direction * geodesic_magnitude

# class DeltaBlock(nn.Module):
#     def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False, dropout, temb_channels=512,
#         layer_type="conv", nheads=1, num_layers=1, dim_feedforward=2048, emb_type="add", use_midblock=False,
#         use_clip=True, use_contrastive=True, geodesic_scale=2.0, adaptive_scaling=True):
#         super().__init__()
#         self.use_midblock = use_midblock
#         self.emb_type = emb_type
#         self.layer_type = layer_type
#         self.use_clip = use_clip
#         self.use_contrastive = use_contrastive
#         self.adaptive_scaling = adaptive_scaling

#         if use_midblock:
#             self.model = UNetMidBlock2DCrossAttn(512, 512, cross_attention_dim=512)
#         else:
#             self.in_channels = in_channels
#             out_channels = in_channels if out_channels is None else out_channels
#             self.out_channels = out_channels
#             self.use_conv_shortcut = conv_shortcut
#             self.layer_type = layer_type
#             self.in_layer = get_dh_layer(
#                 layer_type, nheads, num_layers, dim_feedforward, dropout
#             )

#             self.temb_proj = torch.nn.Linear(temb_channels, out_channels)
#             self.norm2 = Normalize(out_channels)
#             self.out_layer = get_dh_layer(
#                 layer_type, nheads, num_layers, dim_feedforward, dropout
#             )
#             if emb_type == "adagn":
#                 # num groups is kept the same as in Normalize
#                 self.adagn = AdaGroupNorm(embedding_dim=512, out_dim=512, num_groups=32)

#             if layer_type == 'conv':
#                 self.conv1 = self.in_layer
#                 self.conv2 = self.out_layer
#                 del self.in_layer
#                 del self.out_layer

#         # Enhanced Riemannian Block for improved delta_h manipulation
#         self.riemannian_block = EnhancedRiemannianBlock(in_channels, out_channels, temb_channels, geodesic_scale)

#         if use_clip:
#             self.clip_proj = nn.Linear(512, out_channels)  # CLIP projection

#         if use_contrastive:
#             self.contrastive_proj = nn.Linear(in_channels, 128)
#             self.temperature = 0.07  # Temperature for contrastive loss

#     def forward(self, x, temb=None, clip_embedding=None):
#         if self.use_midblock:
#             h = self.model(x, temb)
#         else:
#             h = x
#             h = self.conv1(h) if self.layer_type == "conv" else self.in_layer(h)

#             if temb is not None:
#                 h = self.apply_embedding(h, temb)

#             h = self.conv2(h) if self.layer_type == "conv" else self.out_layer(h)

#         # Enhanced Riemannian manifold block to generate delta_h
#         delta_h = self.riemannian_block(h, temb)

#         # Apply contrastive loss if enabled
#         if self.use_contrastive:
#             delta_h = self.apply_contrastive_loss(h, delta_h)

#         # CLIP embedding modulation if enabled
#         if self.use_clip and clip_embedding is not None:
#             delta_h += self.clip_proj(clip_embedding)

#         return delta_h

#     def apply_embedding(self, h, temb):
#         if self.emb_type == "add":
#             h += self.temb_proj(nonlinearity(temb))[:, :, None, None]
#         elif self.emb_type == "mult":
#             h *= self.temb_proj(nonlinearity(temb))[:, :, None, None]
#         elif self.emb_type == "adagn":
#             h = self.adagn(h, temb)
#         return self.norm2(h)

#     def apply_contrastive_loss(self, h, delta_h):
#         h_proj = self.contrastive_proj(h)
#         delta_h_proj = self.contrastive_proj(delta_h)
#         h_proj = h_proj / h_proj.norm(dim=-1, keepdim=True)
#         delta_h_proj = delta_h_proj / delta_h_proj.norm(dim=-1, keepdim=True)
#         contrastive_loss = self.compute_contrastive_loss(h_proj, delta_h_proj)
#         return delta_h + contrastive_loss

#     def compute_contrastive_loss(self, h_proj, delta_h_proj):
#         sim_matrix = torch.matmul(h_proj, delta_h_proj.T) / self.temperature
#         labels = torch.arange(h_proj.size(0)).cuda()
#         loss = nn.CrossEntropyLoss()(sim_matrix, labels)
#         return loss