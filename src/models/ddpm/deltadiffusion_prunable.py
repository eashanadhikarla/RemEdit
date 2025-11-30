import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.ddpm.expomap import ExponentialMapTrue
from flashfftconv import FlashFFTConv
from diffusers.models.attention import AdaLayerNorm as AdaGroupNorm

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
            self.conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x

class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)
    def forward(self, x):
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = F.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = F.avg_pool2d(x, kernel_size=2, stride=2)
        return x

class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False, dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.temb_proj = torch.nn.Linear(temb_channels, out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
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

class PruningHead(nn.Module):
    def __init__(self, token_dim, edit_vector_dim, hidden_dim: int = 256):
        super().__init__()
        # still exactly three Linear layers
        self.layer1 = nn.Linear(token_dim + edit_vector_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.layer3 = nn.Linear(hidden_dim // 2, 1)
        self.act = nn.SiLU()

        # remember the split index once
        self._split_idx = token_dim          # weight[:, :split] → tokens
                                              # weight[:, split:] → edit vec

    def forward(self, tokens: torch.Tensor, edit_vec: torch.Tensor | None):
        """
        tokens   : [B, N, C_token]
        edit_vec : [B, C_edit]  or None
        returns  : [B, N, 1]
        """
        B, N, _ = tokens.shape
        if edit_vec is None:
            edit_vec = tokens.new_zeros(B, self.layer1.in_features - tokens.size(-1))

        # ------- fast manual linear: token part per-token, edit part once -------
        W = self.layer1.weight                                     # [H, C_tot]
        b = self.layer1.bias                                       # [H]

        W_tok  = W[:, :self._split_idx]                            # [H, C_token]
        W_edit = W[:, self._split_idx:]                            # [H, C_edit]

        # token contribution  : (B·N , C_token) @ W_tok.T  →  [B,N,H]
        tok_proj = torch.matmul(tokens.reshape(-1, tokens.size(-1)), W_tok.T) \
                        .reshape(B, N, -1)

        # edit contribution (once per image) : [B,H] → broadcast
        edit_proj = torch.matmul(edit_vec, W_edit.T) + b           # [B,H]
        x = tok_proj + edit_proj.unsqueeze(1)                      # broadcast add
        x = self.act(x)

        # the rest is unchanged
        x = self.act(self.layer2(x))
        x = self.layer3(x)
        return torch.sigmoid(x)                                              # [B, N, 1]

class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.is_pruning_enabled = False
        self.pruning_head = False
        self.prune_ratio = None # 0.0

    def _standard_attention(self, x):
        h_ = self.norm(x)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)
        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w).permute(0, 2, 1)
        k = k.reshape(b, c, h * w)
        w_ = torch.bmm(q, k) * (int(c) ** (-0.5))
        w_ = F.softmax(w_, dim=2)
        v = v.reshape(b, c, h * w)
        w_ = w_.permute(0, 2, 1)
        h_ = torch.bmm(v, w_)
        h_ = h_.reshape(b, c, h, w)
        h_ = self.proj_out(h_)
        return x + h_

    def _soft_pruning_for_training(self, x, clip_direction):
        unpruned_output = self._standard_attention(x)
        h_ = self.norm(x)
        b, c, h, w = h_.shape
        tokens = h_.view(b, c, h * w).permute(0, 2, 1)
        importance_scores = self.pruning_head(tokens.detach(), clip_direction.detach())
        attention_delta = unpruned_output - x
        gated_delta = attention_delta * importance_scores.permute(0, 2, 1).view(b, 1, h, w)
        gated_output = x + gated_delta
        return {
            "gated_output": gated_output, 
            "unpruned_output": unpruned_output, 
            "scores": importance_scores
            }

    def _hard_pruning_for_inference(self, x, clip_direction):
        if x==None:
            print("x is None inside _hard_pruning_for_inference")
            exit(0)
        if clip_direction == None:
            print("Clip direction is None inside _hard_pruning_for_inference")
            exit(0)

        b, c, h, w = x.shape
        n = h * w
        h_ = self.norm(x)
        tokens = h_.view(b, c, n).permute(0, 2, 1)
        importance_scores = self.pruning_head(tokens.detach(), clip_direction.detach())

        num_to_keep = int(n * (1.0 - self.prune_ratio))
        if num_to_keep < 1: num_to_keep = 1

        _, keep_indices = torch.topk(importance_scores.squeeze(-1), k=num_to_keep, dim=1)

        q_all = self.q(h_).view(b, c, n).permute(0, 2, 1)
        k_all = self.k(h_).view(b, c, n).permute(0, 2, 1)
        v_all = self.v(h_).view(b, c, n).permute(0, 2, 1)

        gather_indices = keep_indices.unsqueeze(-1).expand(-1, -1, c)
        q_kept = torch.gather(q_all, 1, gather_indices)
        k_kept = torch.gather(k_all, 1, gather_indices)
        v_kept = torch.gather(v_all, 1, gather_indices)

        attn_weights = torch.bmm(q_kept, k_kept.transpose(1, 2)) * (c ** (-0.5))
        attn_weights = F.softmax(attn_weights, dim=2)
        attended_v = torch.bmm(attn_weights, v_kept)

        attn_result = torch.zeros_like(tokens)
        attn_result.scatter_(1, gather_indices, attended_v)

        attn_result = attn_result.transpose(1, 2).reshape(b, c, h, w)
        h_out = self.proj_out(attn_result)

        return x + h_out

    def forward(self, x, clip_direction=None, prune_ratio=None):
        self.prune_ratio = prune_ratio

        ## FIX: Strict bypass
        if not self.is_pruning_enabled or self.prune_ratio <= 0.0 or self.pruning_head is None:
            # print("Running NO pruning")
            return self._standard_attention(x)

        if self.pruning_head.training:
            # print("Running soft pruning")
            return self._soft_pruning_for_training(x, clip_direction)
        
        if clip_direction is not None:
            # print("Running hard pruning")
            return self._hard_pruning_for_inference(x, clip_direction)

        return self._standard_attention(x)

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

        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList([
            torch.nn.Linear(self.ch, self.temb_ch),
            torch.nn.Linear(self.temb_ch, self.temb_ch),
        ])

        self.conv_in = torch.nn.Conv2d(in_channels, self.ch, kernel_size=3, stride=1, padding=1)
        
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
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out, temb_channels=self.temb_ch, dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in)) # This will now be our prunable version
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)
        
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in) # Prunable version
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout)
        
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            skip_in = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                if i_block == self.num_res_blocks:
                    skip_in = ch * in_ch_mult[i_level]
                block.append(ResnetBlock(in_channels=block_in + skip_in, out_channels=block_out, temb_channels=self.temb_ch, dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in)) # Prunable version
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)

        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1)

        # --- Pruning Specific Setup ---
        token_dim = config.model.ch * config.model.ch_mult[-1]
        edit_vector_dim = 512
        self.pruning_head = PruningHead(token_dim, edit_vector_dim)
        for module in self.modules():
            if isinstance(module, AttnBlock):
                module.pruning_head = self.pruning_head

    def set_pruning_status(self, is_enabled: bool, prune_ratio: float = 0.0):
        for module in self.modules():
            if isinstance(module, AttnBlock):
                module.is_pruning_enabled = is_enabled
                module.prune_ratio = prune_ratio

    def forward(self, x, t, index=None, t_edit=400, hs_coeff=(1.0, 1.0), delta_h=None, ignore_timestep=False, use_mask=False, clip_direction=None, prune_ratio=0.2):
        # prune_ratio = kwargs.get('prune_ratio', 0.0)
        
        temb = self.get_temb(t)
        hs = [self.conv_in(x)]
        attn_outputs_for_loss = []

        # Downsampling
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    output = self.down[i_level].attn[i_block](h, clip_direction=clip_direction, prune_ratio=prune_ratio)
                    if isinstance(output, dict):
                        attn_outputs_for_loss.append(output)
                        # if self.prune_ratio > 0.0:
                        if "gated_output" in output and (self.prune_ratio > 0.0 or self.pruning_head is not None):
                            h = output['gated_output']
                        else:
                            h = output['unpruned_output']
                    else:
                        h = output
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # Middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        output = self.mid.attn_1(h, clip_direction=clip_direction, prune_ratio=prune_ratio)
        if isinstance(output, dict):
            attn_outputs_for_loss.append(output)
            if "gated_output" in output and (self.prune_ratio > 0.0 or self.pruning_head is not None):
                h = output['gated_output']
            else:
                h = output['unpruned_output']
        else:
            h = output
        h = self.mid.block_2(h, temb)
        middle_h = h
        
        # Editing Logic (Single Path)
        if index is not None and t[0] >= t_edit:
            if delta_h is None:
                h2_calc = h * hs_coeff[0]
                for i in range(index + 1):
                    current_delta_h = getattr(self, f"layer_{i}")(h, None if ignore_timestep else temb, text_emb=clip_direction)
                    h2_calc += current_delta_h * hs_coeff[i + 1]
                h = h2_calc
        
        # Upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](torch.cat([h, hs.pop()], dim=1), temb)
                if len(self.up[i_level].attn) > 0:
                    output = self.up[i_level].attn[i_block](h, clip_direction=clip_direction, prune_ratio=prune_ratio)
                    if isinstance(output, dict):
                        attn_outputs_for_loss.append(output)
                        if "gated_output" in output and (self.prune_ratio > 0.0 or self.pruning_head is not None):
                            h = output['gated_output']
                        else:
                            h = output['unpruned_output']
                    else:
                        h = output
            if i_level != 0:
                h = self.up[i_level].upsample(h)
        
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        
        et = h
        et_modified = h if index is not None else None

        if self.pruning_head.training:
            return et, et_modified, delta_h, middle_h, attn_outputs_for_loss
        else:
            return et, et_modified, delta_h, middle_h

    def setattr_layers(self, nums):
        ch, ch_mult = self.config.model.ch, tuple(self.config.model.ch_mult)
        block_in = None
        for i_level in range(self.num_resolutions):
            block_in = ch * ch_mult[i_level]

        # FIX: Determine the device from an existing parameter
        device = self.conv_in.weight.device

        for i in range(nums):
            delta_block_layer = DeltaBlock(
                in_channels=block_in,
                out_channels=block_in,
                temb_channels=self.temb_ch,
                dropout=0.0,
                layer_type=self.db_layer_type,
                nheads=self.db_nheads,
                num_layers=self.db_num_layers,
                dim_feedforward=self.db_dim_feedforward,
                emb_type=self.db_emb_type,
            )

            setattr(self, f"layer_{i}", delta_block_layer.to(device))


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

    def forward_layer_check(self, x, t, index=None, t_edit=400, hs_coeff=(1.0, 1.0), delta_h=None, ignore_timestep=False, prune_ratio=0.0):
        assert x.shape[2] == x.shape[3] == self.resolution

        # timestep embedding
        temb = get_timestep_embedding(t, self.ch)
        temb = self.temb.dense[0](temb)
        temb = nonlinearity(temb)
        temb = self.temb.dense[1](temb)
        cnt = 0

        print(f"{cnt} <- x.shape:{x.shape}")
        cnt += 1

        # downsampling
        hs = [self.conv_in(x)]
        print(f"{cnt} <- h.shape:{hs[-1].shape}")
        cnt += 1
        for i_level in range(self.num_resolutions):
            if i_level > 0.1:
                print(f"{cnt} <- h.shape:{h.shape},i_level:{i_level}")
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
        print(f"{cnt} <- mid, h.shape:{h.shape}")
        cnt += 1
        h = self.mid.block_1(h, temb)
        print(f"{cnt} <- mid, h.shape:{h.shape}")
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

    def multiple_attr(self, x, t, index=None, maintain=400, rambda=(1.0, 1.0), prune_ratio=0.0):
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

    def interpolation2(self, x, t, index=None, maintain=400, alpha=None, prune_ratio=0.0):
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

    def forward_at(self, x, t, index=None, prune_ratio=0.0):
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

    def forward_global(self, x, t, index=None, maintain=400, direction=None, prune_ratio=0.0):
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


class RiemannianBlock(nn.Module):
    """
    A DeltaBlock variant that learns a Riemannian geodesic update:
    1) Projects features and time embedding via 1×1 conv + linear.
    2) Normalizes via GroupNorm.
    3) Applies the Riemannian exponential map to compute Δh.
    """
    def __init__(self, in_channels, out_channels, temb_channels, num_groups=32, layer_type="conv", fft_seqlen=1024, fft_dtype=torch.bfloat16):
        super().__init__()
        self.layer_type = layer_type
        if layer_type == "flashfft":
            self.flashfftconv = FlashFFTConv(fft_seqlen, dtype=torch.bfloat16)
            self.kernel = nn.Parameter(torch.randn(out_channels, fft_seqlen, dtype=torch.float32))
        else:
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
        if self.layer_type == "flashfft":
            # Collapse spatial dims for FFT conv: [B, C, H, W] -> [B, C, L]
            B, C, H, W = h.shape
            h_1d = h.view(B, C, H * W).to(dtype=torch.bfloat16)
            kernel = self.kernel[:C, :h_1d.shape[-1]].to(dtype=torch.float32, device=h_1d.device)
            h_proj = self.flashfftconv(h_1d, kernel)
            h_proj = h_proj.view(B, C, H, W)
        else:
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
        layer_type="flashfft", # "conv",
        nheads=1, 
        num_layers=1, 
        dim_feedforward=2048, 
        emb_type="add", 
        use_midblock=False,
        fft_seqlen=1024,
        fft_dtype=torch.float16
    ):

        super().__init__()
        self.emb_type = emb_type
        out_channels = out_channels or in_channels

        if layer_type == "flashfft":
            self.in_flashfft = FlashFFTConv(fft_seqlen, dtype=fft_dtype) # torch.float32)
            self.in_kernel = nn.Parameter(torch.randn(512, fft_seqlen, dtype=torch.float32))
            self.out_flashfft = FlashFFTConv(fft_seqlen, dtype=fft_dtype) #torch.float32)
            self.out_kernel = nn.Parameter(torch.randn(512, fft_seqlen, dtype=torch.float32))
        else:
            self.in_layer = nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0)
            self.out_layer = nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0)

        self.temb_proj = nn.Linear(temb_channels, out_channels)
        self.norm2 = Normalize(out_channels)
        self.final_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1)

        if self.emb_type == "adagn":
            self.adagn = AdaGroupNorm(embedding_dim=512, out_dim=512, num_groups=32)

        self.riemannian_block = RiemannianBlock(out_channels, out_channels, temb_channels, layer_type=layer_type, fft_seqlen=fft_seqlen, fft_dtype=fft_dtype)

    def forward(self, x, temb=None, text_emb=None):
        if hasattr(self, "in_flashfft"):
            B, C, H, W = x.shape
            x_1d = x.view(B, C, H * W).to(dtype=torch.bfloat16)
            kernel = self.in_kernel[:C, :x_1d.shape[-1]].to(device=x_1d.device, dtype=torch.float32)
            h = self.in_flashfft(x_1d, kernel)
            h = h.view(B, C, H, W)
        else:
            h = self.in_layer(x)

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

        if hasattr(self, "out_flashfft"):
            B, C, H, W = h.shape
            h_1d = h.view(B, C, H * W).to(dtype=torch.bfloat16)
            kernel = self.out_kernel[:C, :h_1d.shape[-1]].to(device=h_1d.device, dtype=torch.float32)
            h = self.out_flashfft(h_1d, kernel)
            h = h.view(B, C, H, W)
        elif hasattr(self, "out_layer"):
            h = self.out_layer(h)

        delta_h = self.riemannian_block(h, temb)
        # delta_h = self.riemannian_block(h, temb, text_emb=text_emb)
        h = h + delta_h

        h = self.final_conv(h)
        return h