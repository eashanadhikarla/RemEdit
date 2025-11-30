import math
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchdiffeq import odeint
from torchdiffeq import odeint_adjoint as odeint  # Use adjoint for stable backprop
from mamba_ssm import Mamba

def nonlinearity(x):
    # swish
    return x * torch.sigmoid(x)

def Normalize(in_channels):
    return torch.nn.GroupNorm(
        num_groups=32, num_channels=in_channels, eps=1e-6, affine=True
    )

from torchdiffeq import odeint

class ExponentialMapTrue(nn.Module):

    def __init__(self, channels, temb_channels, manifold_layer='ssm'):
        super().__init__()
        '''
        Number of feature channels per spatial location
        Linear projection of the timestep embedding into the feature space
        Linear layer to predict the raw tangent vector at each point y

        Small MLP that outputs flattened Christoffel symbols Γᵏᵢⱼ for each feature vector
        We reshape its C² outputs into a (C×C) matrix per spatial location
        '''
        
        self.channels = channels
        self.manifold_layer = manifold_layer
        print(f'Using layer type: {manifold_layer}')
        if self.manifold_layer=='linear':
            self.temb_proj = nn.Linear(temb_channels, channels)
        else:
            self.temb_proj = Mamba(
                d_model=temb_channels,    # input dimension
                d_state=32,               # hidden state size (tunable)
                d_conv=4,                 # conv width inside SSM
                expand=1                  # no expansion along channel
            )
        # Project from SSM output back into the feature channels
        self.temb_head = nn.Linear(temb_channels, channels)
        self.linear_tangent = nn.Linear(channels, channels)

        self.gamma_net = nn.Sequential(
            nn.Linear(channels, channels * channels)
        )
        # Learnable scaling factor for the initial geodesic retraction step
        # self.scale = nn.Parameter(torch.tensor(0.6))
        self.scale_net = nn.Sequential(
            nn.Linear(temb_channels, 1),
            nn.Sigmoid()     # outputs in (0,1), you can rescale if you like
        )
        # optionally keep a global bias
        self.scale_bias = nn.Parameter(torch.tensor(0.1))

    def compute_christoffel(self, y_flat):
        """
        Given flattened manifold points y_flat of shape [B, N, C],
        predict and return the Christoffel-symbol tensor Γ of shape [B, N, C, C].
        Clamps the values to [-3,3] to keep curvature estimates stable.
        """
        B, N, C = y_flat.shape
        # Predict C² values per point, then reshape
        gamma = self.gamma_net(y_flat)  # [B, N, C*C]
        gamma = gamma.view(B, N, C, C)  # [B, N, C, C]
        return gamma.clamp(-3.0, 3.0)   # clamp for stability

    def geodesic_rhs(self, t, state):
        """
        Right-hand side of the geodesic ODE system:
            dy/dt = v,
            dv/dt = -Γ(y)·(v, v).
        'state' concatenates y_flat and v_flat into a [B, 2*N*C] vector.
        Returns the time derivative of the state, same shape.
        """
        B, L = state.shape
        half = L // 2

        # Split state into position y_flat and velocity v_flat
        y_flat = state[:, :half].view(B, -1, self.channels)    # [B, N, C]
        v_flat = state[:, half:].view(B, -1, self.channels)    # [B, N, C]

        # Compute Christoffel symbols at each y_flat
        Γ = self.compute_christoffel(y_flat)                   # [B, N, C, C]
        # dv/dt = - sum_jk Γᵢⱼᵏ v_j v_k (Einstein summation)
        dv = -torch.einsum('bnij,bni,bnj->bnj', Γ, v_flat, v_flat)
        # dy/dt = v
        dy = v_flat

        # Concatenate derivatives back into a single vector
        return torch.cat([dy.reshape(B, -1), dv.reshape(B, -1)], dim=1)

    def forward(self, h, temb):
        """
        Execute the Riemannian exponential map on feature map h:
        1. Form initial point y0 = h + projected timestep embedding.
        2. Predict initial tangent v0 using a smooth tanh-based retraction.
        3. Integrate the geodesic ODE from t=0 to t=1.
        4. Return the change Δh = γ(1) - h.
        """
        B, C, H, W = h.shape

        if self.manifold_layer=='linear':
            # 1) Form the manifold point y0 by injecting timestep conditioning
            temb_proj = self.temb_proj(temb)[:, :, None, None]   # [B, C, 1, 1]
        elif self.manifold_layer=='ssm':
            #    temb: [B, temb_channels]
            temb_seq = temb.unsqueeze(1)              # [B, 1, temb_channels]
            temb_proj = self.temb_proj(temb_seq)      # [B, 1, temb_channels]
            temb_out = temb_proj[:, 0, :]             # [B, temb_channels]
            temb_proj = self.temb_head(temb_out)      # [B, channels]
            temb_proj = temb_proj[:,:,None,None]      # [B, channels, 1, 1]

        combined  = h + temb_proj                            # [B, C, H, W]

        # 2) Flatten y0 and predict initial tangent v0
        y0_flat = combined.permute(0, 2, 3, 1).reshape(B, -1, C)  # [B, N, C]
        t_vec   = self.linear_tangent(y0_flat)                    # raw V_raw
        norm    = torch.norm(t_vec, dim=-1, keepdim=True)         # magnitude N
        # dir     = F.normalize(t_vec, dim=-1)                      # unit direction Ĥ
        dir     = F.normalize(t_vec, p=2, dim=-1)               # unit direction Ĥ

        # Apply tanh-based retraction: v0 = Ĥ * tanh(N·scale) / (N + ε)
        # v0_flat = (dir * (torch.tanh(norm * self.scale) / (norm + 1e-6))).reshape(B, -1)
        alpha = self.scale_net(temb) + self.scale_bias
        alpha = alpha.view(B, 1, 1, 1)
        v0_flat = (dir * (torch.tanh(norm * alpha) / (norm + 1e-6))).reshape(B, -1)

        # 3) Solve the geodesic ODE system from t=0 → t=1
        state0 = torch.cat([y0_flat.reshape(B, -1), v0_flat], dim=1)
        t_span = torch.tensor([0.0, 1.0], device=h.device)
        sol    = odeint(self.geodesic_rhs, state0, t_span,
                        method='dopri5', rtol=1e-4, atol=1e-6)

        # 4) Extract γ(1) (the endpoint of the geodesic) and compute Δh
        K        = y0_flat.numel() // B               # points per instance
        y1_flat  = sol[1][:, :K].view(B, H, W, C)     # [B, H, W, C]
        y1       = y1_flat.permute(0, 3, 1, 2)        # [B, C, H, W]
        delta_h  = y1 - h

        return delta_h



'''
Reasons for clamping [-3,3]

We clamp our learned Christoffel symbols Γ to the range [−3, 3] for two main reasons:
1.	Numerical stability of the ODE solver.
    In the geodesic‐ODE
    \dot v(t) = \Gamma\bigl(y(t)\bigr)\bigl(v(t), v(t)\bigr),
    extremely large values in Γ lead to enormous accelerations \dot v, which can blow up during 
    integration (even with a high–precision solver). By restricting each component of Γ to lie in [−3, 3], 
    we ensure that no single Christoffel entry can induce a change in v larger than roughly 3\|v\|^2, 
    keeping each integration step well within the solver’s stable regime.
2.	Preventing over‐curvature (overfitting).
    Letting Γ take arbitrarily large magnitudes would allow the network to “warp” the manifold excessively, 
    effectively collapsing or stretching regions in uncontrolled ways.  Clamping to [−3, 3] provides a 
    soft regularizer on the manifold’s curvature: it ensures that our learned connection never deviates 
    more than a moderate amount from flat Euclidean space, while still permitting meaningful nonzero 
    curvature for semantic editing.

In practice, we found via ablation (sweeping clamp ranges from [−1, 1] up to [−10, 10]) that [−3, 3] strikes 
a good balance: it prevents numerical blow‐ups and over‐warping, yet still allows enough flexibility for 
rich geodesic trajectories.
'''