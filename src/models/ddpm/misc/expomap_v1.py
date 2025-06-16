import math
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchdiffeq import odeint
from torchdiffeq import odeint_adjoint as odeint  # Use adjoint for stable backprop

def nonlinearity(x):
    # swish
    return x * torch.sigmoid(x)

def Normalize(in_channels):
    return torch.nn.GroupNorm(
        num_groups=32, num_channels=in_channels, eps=1e-6, affine=True
    )

######
## W/o text_emb w/o CLIP Loss
######
class ExponentialMapVanilla(nn.Module):
    def __init__(self, channels, temb_channels):
        super().__init__()
        self.channels = channels
        
        # Learn local tangent direction using linear layers
        self.linear_tangent = nn.Linear(channels, channels)
        self.temb_proj = nn.Linear(temb_channels, channels)

        # Small trainable scalar for controlling the magnitude of movement along geodesics
        self.scale = nn.Parameter(torch.tensor(0.6))

    def forward(self, h, temb):
        batch, channels, height, width = h.shape

        # Project temporal embedding and combine with latent
        temb_proj = self.temb_proj(temb)[:, :, None, None]
        combined_features = (h + temb_proj).permute(0, 2, 3, 1) # [batch, height, width, channels]
        tangent_vec = self.linear_tangent(combined_features) # [batch, height, width, channels]

        # Compute norm (magnitude)
        norm = torch.norm(tangent_vec, dim=-1, keepdim=True)  # [batch, height, width, 1]
        # Explicit normalization of tangent directions
        tangent_dir = F.normalize(tangent_vec, p=2, dim=-1)
        # Theoretically-driven exponential map approximation (with learned scaling)
        # exp_factor = torch.sin(norm * self.scale) / (norm + 1e-6)
        exp_factor = torch.tanh(norm * self.scale) / (norm + 1e-6)
        # Move along tangent direction explicitly (exponential map)
        manifold_point = combined_features + tangent_dir * exp_factor
        # Permute back to original shape [batch, channels, height, width]
        manifold_point = manifold_point.permute(0, 3, 1, 2)
        # Compute delta_h explicitly
        delta_h = manifold_point - h
        return delta_h


######
## With text_emb version1
######
class ExponentialMapwithPrompt(nn.Module):
    def __init__(self, channels, temb_channels):
        super().__init__()
        self.channels = channels
        
        # Learn local tangent direction using linear layers
        self.linear_tangent = nn.Linear(channels, channels)
        self.temb_proj = nn.Linear(temb_channels, channels)

        # Small trainable scalar for controlling the magnitude of movement along geodesics
        self.scale = nn.Parameter(torch.tensor(0.6))

    def forward(self, h, temb, g=None):
        batch, channels, height, width = h.shape

        # Project temporal embedding and combine with latent
        temb_proj = self.temb_proj(temb)[:, :, None, None]
        combined_features = (h + temb_proj).permute(0, 2, 3, 1)  # [B, H, W, C]

        # Learn tangent vector
        tangent_vec = self.linear_tangent(combined_features)  # [B, H, W, C]

        if g is not None:
            # Flatten spatial dims
            tangent_vec_flat = tangent_vec.view(batch, -1, channels)  # [B, HW, C]
            tangent_vec_flat = tangent_vec_flat.transpose(1, 2)        # [B, C, HW]

            # Mahalanobis norm: sqrt(v^T g v)
            gv = torch.bmm(g, tangent_vec_flat)                       # [B, C, HW]
            dot = (tangent_vec_flat * gv).sum(dim=1, keepdim=True)    # [B, 1, HW]
            norm = torch.sqrt(dot + 1e-6)                             # [B, 1, HW]
            norm = norm.view(batch, 1, height, width)                 # [B, 1, H, W]

            # Normalize direction and scale
            tangent_dir = F.normalize(tangent_vec.permute(0, 3, 1, 2), p=2, dim=1)  # [B, C, H, W]
            exp_factor = torch.tanh(norm * self.scale) / (norm + 1e-6)             # [B, 1, H, W]
            delta = exp_factor * tangent_dir
        else:
            # Original behavior
            norm = torch.norm(tangent_vec, dim=-1, keepdim=True)  # [B, H, W, 1]
            tangent_dir = F.normalize(tangent_vec, p=2, dim=-1)    # [B, H, W, C]
            exp_factor = torch.tanh(norm * self.scale) / (norm + 1e-6)
            delta = exp_factor * tangent_dir                      # [B, H, W, C]
            delta = delta.permute(0, 3, 1, 2)                     # [B, C, H, W]

        return h + delta


######
## With text_emb version 2 w/o CLIP Loss
######
class ExponentialMapwithPrompt2(nn.Module):
    def __init__(self, channels, temb_channels, text_dim=None):
        super().__init__()
        self.channels = channels

        # Stable, bounded tangent direction
        self.linear_tangent = nn.Sequential(
            nn.Linear(channels, channels),
            nn.Tanh()
        )
        self.temb_proj = nn.Linear(temb_channels, channels)
        self.text_proj = nn.Linear(text_dim, channels) if text_dim is not None else None

        self.scale = nn.Parameter(torch.tensor(0.6))

    def forward(self, h, temb, g=None, text_emb=None, use_geodesic=False):
        B, C, H, W = h.shape

        # Temporal embedding projection
        temb_proj = self.temb_proj(temb)[:, :, None, None]  # [B, C, 1, 1]
        combined = h + temb_proj

        # Optional text embedding projection
        if text_emb is not None and self.text_proj is not None:
            text_proj = self.text_proj(text_emb)[:, :, None, None]  # [B, C, 1, 1]
            text_proj = F.normalize(text_proj, p=2, dim=1)  # Normalize across channel
            combined = combined + 0.1 * text_proj  # Modest residual injection

        # Project to tangent space
        combined_features = combined.permute(0, 2, 3, 1)  # [B, H, W, C]
        tangent_vec = self.linear_tangent(combined_features)  # [B, H, W, C]

        if g is not None:
            # Mahalanobis-aware metric norm (optional branch)
            tangent_vec_flat = tangent_vec.view(B, -1, C).transpose(1, 2)  # [B, C, HW]
            gv = torch.bmm(g, tangent_vec_flat)  # [B, C, HW]
            dot = (tangent_vec_flat * gv).sum(dim=1, keepdim=True)  # [B, 1, HW]
            norm = torch.sqrt(dot + 1e-6).view(B, 1, H, W)  # [B, 1, H, W]

            tangent_dir = F.normalize(tangent_vec.permute(0, 3, 1, 2), p=2, dim=1)  # [B, C, H, W]
            exp_factor = torch.tanh(norm * self.scale) / (norm + 1e-6)
            delta = exp_factor * tangent_dir
        else:
            # Standard direction
            norm = torch.norm(tangent_vec, dim=-1, keepdim=True)  # [B, H, W, 1]
            tangent_dir = F.normalize(tangent_vec, p=2, dim=-1)  # [B, H, W, C]
            exp_factor = torch.tanh(norm * self.scale) / (norm + 1e-6)
            delta = exp_factor * tangent_dir  # [B, H, W, C]
            delta = delta.permute(0, 3, 1, 2)  # [B, C, H, W]

        # Final stability clamp
        delta_norm = torch.norm(delta.reshape(B, -1), dim=1, keepdim=True) + 1e-6
        scaling = torch.clamp(0.1 / delta_norm, max=1.0).view(B, 1, 1, 1)
        delta = delta * scaling

        return h + delta


######
## With text_emb with CLIP Loss
######
class ExponentialMapwithPromptandCLIP(nn.Module):
    def __init__(self, channels, temb_channels, text_dim=None, clip_model=None):
        super().__init__()
        self.channels = channels

        self.linear_tangent = nn.Sequential(
            nn.Linear(channels, channels),
            nn.Tanh()
        )
        self.temb_proj = nn.Linear(temb_channels, channels)
        self.text_proj = nn.Linear(text_dim, channels) if text_dim is not None else None

        self.scale = nn.Parameter(torch.tensor(0.6))
        self.clip_model = clip_model  # CLIPWrapper instance

    def forward(self, h, temb, g=None, text_emb=None, use_geodesic=False):
        B, C, H, W = h.shape

        temb_proj = self.temb_proj(temb)[:, :, None, None]
        combined = h + temb_proj

        if text_emb is not None and self.text_proj is not None:
            text_proj = self.text_proj(text_emb)[:, :, None, None]
            text_proj = F.normalize(text_proj, p=2, dim=1)
            combined = combined + 0.1 * text_proj

        combined_features = combined.permute(0, 2, 3, 1)
        tangent_vec = self.linear_tangent(combined_features)

        # if g is not None:
        #     tangent_vec_flat = tangent_vec.view(B, -1, C).transpose(1, 2)
        #     gv = torch.bmm(g, tangent_vec_flat)
        #     dot = (tangent_vec_flat * gv).sum(dim=1, keepdim=True)
        #     norm = torch.sqrt(dot + 1e-6).view(B, 1, H, W)

        #     tangent_dir = F.normalize(tangent_vec.permute(0, 3, 1, 2), p=2, dim=1)
        #     exp_factor = torch.tanh(norm * self.scale) / (norm + 1e-6)
        #     delta = exp_factor * tangent_dir
        # else:
        norm = torch.norm(tangent_vec, dim=-1, keepdim=True)
        tangent_dir = F.normalize(tangent_vec, p=2, dim=-1)
        exp_factor = torch.tanh(norm * self.scale) / (norm + 1e-6)
        delta = exp_factor * tangent_dir
        delta = delta.permute(0, 3, 1, 2)

        # Identity-preserving delta adjustment using CLIP
        delta_norm = torch.norm(delta.reshape(B, -1), dim=1, keepdim=True) + 1e-6
        scaling = torch.clamp(0.1 / delta_norm, max=1.0).view(B, 1, 1, 1)
        delta_h = delta * scaling

        # if self.clip_model is not None:
        #     with torch.no_grad():
        #         x0_clip = self.clip_model.encode_image(h)          # [B, D]
        #         x1_clip = self.clip_model.encode_image(h + delta_h)
        #         cosine_sim = F.cosine_similarity(x0_clip, x1_clip, dim=-1)  # [B]
        #         loss_id = 1 - cosine_sim.mean()
        #         delta_h = delta_h - 0.05 * loss_id  # identity-preserving penalty

        if self.clip_model is not None and h.shape[1] == 3:
            with torch.no_grad():
                x0_clip = self.clip_model.encode_image(h)
                x1_clip = self.clip_model.encode_image(h + delta_h)
                cosine_sim = F.cosine_similarity(x0_clip, x1_clip, dim=-1)
                loss_id = 1 - cosine_sim.mean()
                delta_h = delta_h - 0.05 * loss_id

        return h + delta_h

######
## With text_emb (modulated + gated) with CLIP Loss
######
# class ExponentialMapwithPromptandCLIP(nn.Module):
class ExponentialMap(nn.Module):
    def __init__(self, channels, temb_channels, text_dim=None, clip_model=None):
        super().__init__()
        self.channels = channels

        self.linear_tangent = nn.Sequential(
            nn.Linear(channels, channels),
            nn.Tanh()
        )
        self.temb_proj = nn.Linear(temb_channels, channels)
        self.text_proj = nn.Linear(text_dim, channels * 2) if text_dim is not None else None  # FiLM = scale + shift

        self.scale = nn.Parameter(torch.tensor(0.6))
        self.clip_model = clip_model  # CLIPWrapper instance

    def forward(self, h, temb, g=None, text_emb=None, use_geodesic=False):
        B, C, H, W = h.shape

        temb_proj = self.temb_proj(temb)[:, :, None, None]
        combined = h + temb_proj

        # FiLM-style modulation with gate
        if text_emb is not None and self.text_proj is not None:
            film_params = self.text_proj(text_emb)  # [B, 2C]
            scale, shift = film_params.chunk(2, dim=1)
            scale = torch.tanh(scale) * 0.05  # smooth small modulation
            shift = torch.tanh(shift) * 0.05
            scale = scale[:, :, None, None]
            shift = shift[:, :, None, None]

            gate = torch.sigmoid(scale.mean(dim=1, keepdim=True))  # [B, 1, 1, 1]
            combined = combined * (1 + gate * scale) + gate * shift

        combined_features = combined.permute(0, 2, 3, 1)
        tangent_vec = self.linear_tangent(combined_features)

        norm = torch.norm(tangent_vec, dim=-1, keepdim=True)
        tangent_dir = F.normalize(tangent_vec, p=2, dim=-1)
        exp_factor = torch.tanh(norm * self.scale) / (norm + 1e-6)
        delta = exp_factor * tangent_dir
        delta = delta.permute(0, 3, 1, 2)

        # Normalize final update
        delta_norm = torch.norm(delta.reshape(B, -1), dim=1, keepdim=True) + 1e-6
        scaling = torch.clamp(0.1 / delta_norm, max=1.0).view(B, 1, 1, 1)
        delta_h = delta * scaling

        # Optional: CLIP-based identity regularization (image input only)
        if self.clip_model is not None and h.shape[1] == 3:
            with torch.no_grad():
                x0_clip = self.clip_model.encode_image(h)
                x1_clip = self.clip_model.encode_image(h + delta_h)
                cosine_sim = F.cosine_similarity(x0_clip, x1_clip, dim=-1)
                loss_id = 1 - cosine_sim.mean()
                delta_h = delta_h - 0.05 * loss_id

        return h + delta_h

######
## With text_emb (modulated + gated) with CLIP Loss on vanilla simplest network
######
class ExponentialMapVanilla2(nn.Module):
    def __init__(self, channels, temb_channels, text_dim=None):
        super().__init__()
        self.channels = channels
        
        self.linear_tangent = nn.Linear(channels, channels)
        self.temb_proj = nn.Linear(temb_channels, channels)
        self.text_proj = nn.Linear(text_dim, channels * 2) if text_dim is not None else None

        self.scale = nn.Parameter(torch.tensor(0.25))
        self.global_step = 0  # Optional external hook for time gating

    def forward(self, h, temb, text_emb=None):
        B, C, H, W = h.shape

        temb_proj = self.temb_proj(temb)[:, :, None, None]
        combined = h + temb_proj

        if (
            self.training and 
            self.global_step > 2 and 
            text_emb is not None and 
            self.text_proj is not None
        ):
            film_params = self.text_proj(text_emb)  # [B, 2C]
            scale, shift = film_params.chunk(2, dim=1)  # [B, C] each
            scale = torch.tanh(scale) * 0.05
            shift = torch.tanh(shift) * 0.05
            scale = scale[:, :, None, None]
            shift = shift[:, :, None, None]

            gate = torch.sigmoid(scale.mean(dim=1, keepdim=True))  # [B, 1, 1, 1]
            combined = combined * (1 + gate * scale) + gate * shift

        combined_features = combined.permute(0, 2, 3, 1)  # [B, H, W, C]
        tangent_vec = self.linear_tangent(combined_features)  # [B, H, W, C]

        norm = torch.norm(tangent_vec, dim=-1, keepdim=True)
        tangent_dir = F.normalize(tangent_vec, p=2, dim=-1)
        exp_factor = torch.tanh(norm * self.scale) / (norm + 1e-6)
        manifold_point = combined_features + tangent_dir * exp_factor

        manifold_point = manifold_point.permute(0, 3, 1, 2)
        delta_h = manifold_point - h

        # if self.global_step > 2:
        #     delta_h = delta_h - 0.05 * loss_id  
        
        self.global_step += 1

        return delta_h
    

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint

class ExponentialMapTrue(nn.Module):
    def __init__(self, channels, temb_channels):
        super().__init__()
        self.channels = channels
        self.temb_proj = nn.Linear(temb_channels, channels)
        self.linear_tangent = nn.Linear(channels, channels)
        # small helper MLP to predict Christoffel symbols Γᵏᵢⱼ(y)
        self.gamma_net = nn.Sequential(
            nn.Linear(channels, channels * channels)
        )
        # fixed scaling for ODE step size
        self.scale = nn.Parameter(torch.tensor(0.6))

    def compute_christoffel(self, y_flat):
        # y_flat: [B, N, C]
        B, N, C = y_flat.shape
        gamma = self.gamma_net(y_flat)               # [B, N, C²]
        gamma = gamma.view(B, N, C, C)              # [B, N, C, C]
        # clamp for stability
        return gamma.clamp(-3.0, 3.0)

    def geodesic_rhs(self, t, state):
        # state: [B, 2*N*C] = [y_flat, v_flat]
        B, L = state.shape
        half = L // 2
        y_flat = state[:, :half].view(B, -1, self.channels)   # [B, N, C]
        v_flat = state[:, half:].view(B, -1, self.channels)  # [B, N, C]

        Γ = self.compute_christoffel(y_flat)                  # [B, N, C, C]
        # dv/dt = - Γᵢⱼᵏ vᵢ vⱼ
        dv = -torch.einsum('bnij,bni,bnj->bnj', Γ, v_flat, v_flat)
        dy = v_flat                                          # dy/dt = v
        return torch.cat([dy.reshape(B, -1), dv.reshape(B, -1)], dim=1)

    def forward(self, h, temb):
        B, C, H, W = h.shape
        # project time embedding and form initial manifold point
        temb_proj = self.temb_proj(temb)[:, :, None, None]  # [B, C, 1, 1]
        combined = h + temb_proj                            # [B, C, H, W]
        # --- Begin true exponential map via ODE integration ---
        # 1) Flatten initial point y0 and compute initial tangent v0
        y0_flat = combined.permute(0, 2, 3, 1).reshape(B, -1, C)  # [B, N, C]
        t_vec   = self.linear_tangent(y0_flat)                 # [B, N, C]
        norm    = torch.norm(t_vec, dim=-1, keepdim=True)
        dir     = F.normalize(t_vec, dim=-1)
        v0_flat = (dir * (torch.tanh(norm * self.scale) / (norm + 1e-6))).reshape(B, -1)

        # 2) initial tangent vector
        # (already computed above, so skip duplicate lines)

        # 3) solve the geodesic ODE from t=0→1
        state0 = torch.cat([y0_flat.reshape(B, -1), v0_flat], dim=1)
        t_span = torch.tensor([0.0, 1.0], device=h.device)
        sol = odeint(self.geodesic_rhs, state0, t_span, method='dopri5',
                     rtol=1e-4, atol=1e-6)

        # 4) read off the “landed” point γ(1)
        y1_flat = sol[1][:, : (y0_flat.numel() // B) ].view(B, H, W, C)
        y1 = y1_flat.permute(0, 3, 1, 2)             # [B,C,H,W]

        return y1 - h