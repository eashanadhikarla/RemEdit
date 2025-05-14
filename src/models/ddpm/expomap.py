import math
import torch
import torch.nn as nn
from torchdiffeq import odeint
import torch.nn.functional as F

def nonlinearity(x):
    # swish
    return x * torch.sigmoid(x)

def Normalize(in_channels):
    return torch.nn.GroupNorm(
        num_groups=32, num_channels=in_channels, eps=1e-6, affine=True
    )

class ExponentialMap(nn.Module):
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
        combined_features = (h + temb_proj).permute(0, 2, 3, 1)  # [batch, height, width, channels]

        tangent_vec = self.linear_tangent(combined_features)  # [batch, height, width, channels]

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


# class ODEExponentialMap(nn.Module):
#     def __init__(self, channels, temb_channels):
#         super().__init__()
#         self.channels = channels

#         # Linear layers to define tangent vector
#         self.linear_tangent = nn.Linear(channels, channels)
#         self.temb_proj = nn.Linear(temb_channels, channels)

#         # Learnable scale for initial velocity
#         self.scale = nn.Parameter(torch.tensor(0.6))

#     def geodesic_ode(self, t, y):
#         """
#         Simple straight-line ODE approximation of geodesic:
#         dy/dt = v (constant velocity model).
#         For actual manifolds, this should incorporate Christoffel symbols.
#         """
#         v = self.current_tangent_vector
#         return v

#     def forward(self, h, temb):
#         batch, channels, height, width = h.shape
        
#         # Project temporal embedding and latent combination
#         temb_proj = self.temb_proj(temb)[:, :, None, None]
#         combined_features = (h + temb_proj).permute(0, 2, 3, 1)  # [batch, H, W, C]

#         tangent_vec = self.linear_tangent(combined_features)  # [batch, H, W, C]

#         # Store scaled tangent vector for ODE (initial velocity)
#         self.current_tangent_vector = tangent_vec * self.scale

#         # Flatten to [batch, -1] for ODE solver
#         y0 = combined_features.view(batch, -1)

#         # Solve the ODE explicitly from t=0 to t=1
#         t = torch.tensor([0.0, 1.0], device=h.device)
#         solution = odeint(self.geodesic_ode, y0, t, method='rk4')

#         # Take the solution at t=1
#         manifold_point = solution[1].view(batch, height, width, channels)

#         # Permute back to original shape [batch, C, H, W]
#         manifold_point = manifold_point.permute(0, 3, 1, 2)

#         # Compute delta_h explicitly
#         delta_h = manifold_point - h

#         return delta_h


# ###################
# ## ODE with Linear
# ###################

# class ODEExponentialMap(nn.Module):
#     def __init__(self, channels, temb_channels):
#         super().__init__()
#         self.channels = channels
#         self.linear_tangent = nn.Linear(channels, channels)
#         self.temb_proj = nn.Linear(temb_channels, channels)
#         self.scale = nn.Parameter(torch.tensor(0.6))

#         # Memory-efficient learned Christoffel symbols
#         self.gamma_net = nn.Sequential(
#             nn.Linear(self.channels, self.channels * self.channels)
#         )

#     def compute_christoffel_symbols(self, y):
#         batch_size, num_points, channels = y.shape
#         gamma = self.gamma_net(y)  # [B, N, C²]
#         gamma = gamma.view(batch_size, num_points, self.channels, self.channels)  # [B, N, C, C]
#         gamma = torch.clamp(gamma, min=-3.0, max=3.0)  # prevent extreme curvature
#         return gamma

#     def geodesic_ode(self, t, state):
#         batch_size = state.shape[0]
#         y, v = torch.chunk(state, 2, dim=-1)

#         y = y.view(batch_size, -1, self.channels)  # [B, N, C]
#         v = v.view(batch_size, -1, self.channels)  # [B, N, C]

#         gamma = self.compute_christoffel_symbols(y)  # [B, N, C, C]

#         # Approximated second-order term: -sum_j gamma_ij * v_j^2
#         dv_dt = -torch.einsum('bnij,bnj,bnj->bni', gamma, v, v)
#         dy_dt = v

#         dy_dt = dy_dt.reshape(batch_size, -1)
#         dv_dt = dv_dt.reshape(batch_size, -1)

#         return torch.cat([dy_dt, dv_dt], dim=-1)

#     def forward(self, h, temb):
#         batch, channels, height, width = h.shape

#         temb_proj = self.temb_proj(temb)[:, :, None, None]
#         combined_features = (h + temb_proj).permute(0, 2, 3, 1)  # [B, H, W, C]

#         tangent_vec = self.linear_tangent(combined_features)  # [B, H, W, C]
#         tangent_vec = F.normalize(tangent_vec, p=2, dim=-1)    # unit direction
#         tangent_vec_scaled = tangent_vec * self.scale          # scaled step

#         y0 = combined_features.reshape(batch, -1)
#         v0 = tangent_vec_scaled.reshape(batch, -1)
#         state0 = torch.cat([y0, v0], dim=-1)  # [B, 2 * HWC]

#         t_span = torch.tensor([0.0, 1.0], device=h.device)

#         # Adaptive ODE solver (more stable for stiff problems)
#         solution = odeint(self.geodesic_ode, state0, t_span, method='dopri5', rtol=1e-4, atol=1e-6)

#         y1 = solution[1][:, :y0.shape[1]].view(batch, height, width, channels)
#         manifold_point = y1.permute(0, 3, 1, 2)  # [B, C, H, W]

#         delta_h = torch.clamp(manifold_point - h, min=-1.0, max=1.0)  # safe range
#         return delta_h

# #################
# ## ODE with Mamba
# #################
# from torchdiffeq import odeint

# class ODEExponentialMap(nn.Module):
#     def __init__(self, channels, temb_channels):
#         super().__init__()
#         self.channels = channels
#         # self.temb_proj = nn.Linear(temb_channels, channels)
#         # Use Mamba for temporal embedding projection
#         self.temb_mamba = Mamba(d_model=temb_channels, d_state=16, d_conv=3, expand=2)
#         self.linear_tangent = Mamba(d_model=channels, d_state=16, d_conv=3, expand=2)

#         self.scale = nn.Parameter(torch.tensor(0.6))

#         # Memory-efficient learned Christoffel symbols
#         self.gamma_net = nn.Sequential(
#             nn.Linear(self.channels, self.channels * self.channels)
#         )

#     def compute_christoffel_symbols(self, y):
#         batch_size, num_points, channels = y.shape
#         gamma = self.gamma_net(y)  # [B, N, C²]
#         gamma = gamma.view(batch_size, num_points, self.channels, self.channels)  # [B, N, C, C]
#         gamma = torch.clamp(gamma, min=-3.0, max=3.0)  # prevent extreme curvature
#         return gamma

#     def geodesic_ode(self, t, state):
#         batch_size = state.shape[0]
#         y, v = torch.chunk(state, 2, dim=-1)

#         y = y.view(batch_size, -1, self.channels)  # [B, N, C]
#         v = v.view(batch_size, -1, self.channels)  # [B, N, C]

#         gamma = self.compute_christoffel_symbols(y)  # [B, N, C, C]

#         # Approximated second-order term: -sum_j gamma_ij * v_j^2
#         dv_dt = -torch.einsum('bnij,bnj,bnj->bni', gamma, v, v)
#         dy_dt = v

#         dy_dt = dy_dt.reshape(batch_size, -1)
#         dv_dt = dv_dt.reshape(batch_size, -1)

#         return torch.cat([dy_dt, dv_dt], dim=-1)

#     def forward(self, h, temb):
#         batch, channels, height, width = h.shape

#         # Project temporal embedding via Mamba over a singleton time dimension
#         temb_seq = temb.unsqueeze(1)                              # [B, 1, temb_channels]
#         temb_out = self.temb_mamba(temb_seq)                      # [B, 1, channels]
#         temb_proj = temb_out.squeeze(1)[:, :, None, None]         # [B, channels, 1, 1]

#         combined_features = (h + temb_proj).permute(0, 2, 3, 1)  # [B, H, W, C]

#         batch_, height_, width_, channels_ = combined_features.shape
#         # Prepare sequence for Mamba: [batch, sequence, channels]
#         combined_features_flat = combined_features.view(batch_, height_ * width_, channels_)
#         tangent_vec = self.linear_tangent(combined_features_flat)  # [batch, sequence, channels]
#         tangent_vec = tangent_vec.view(batch_, height_, width_, channels_)

#         tangent_vec = F.normalize(tangent_vec, p=2, dim=-1)    # unit direction
#         tangent_vec_scaled = tangent_vec * self.scale          # scaled step

#         y0 = combined_features.reshape(batch_, -1)
#         v0 = tangent_vec_scaled.reshape(batch_, -1)
#         state0 = torch.cat([y0, v0], dim=-1)  # [B, 2 * HWC]

#         t_span = torch.tensor([0.0, 1.0], device=h.device)

#         # Adaptive ODE solver (more stable for stiff problems)
#         solution = odeint(self.geodesic_ode, state0, t_span, method='dopri5', rtol=1e-4, atol=1e-6)

#         y1 = solution[1][:, :y0.shape[1]].view(batch_, height_, width_, channels_)
#         manifold_point = y1.permute(0, 3, 1, 2)  # [B, C, H, W]

#         delta_h = torch.clamp(manifold_point - h, min=-1.0, max=1.0)  # safe range
#         return delta_h