# cssa_module.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossStepSemanticAttention(nn.Module):
    def __init__(self, z_dim, s_dim=512, num_heads=4):
        super().__init__()
        self.query_proj = nn.Linear(z_dim, s_dim)
        self.key_proj = nn.Linear(s_dim, s_dim)
        self.value_proj = nn.Linear(s_dim, s_dim)
        self.fuse_proj = nn.Linear(s_dim, z_dim)
        self.attn = nn.MultiheadAttention(embed_dim=s_dim, num_heads=num_heads, batch_first=True)

    def forward(self, z_t, memory_bank):
        """
        Args:
            z_t: Tensor of shape [B, C, H, W] representing the current latent state.
            memory_bank: List of semantic embeddings from previous timesteps, each of shape [B, s_dim].

        Returns:
            Tensor of shape [B, C, H, W] after semantic attention fusion.
        """
        B, C, H, W = z_t.shape
        z_flat = z_t.view(B, C, -1).permute(0, 2, 1)  # [B, HW, C]

        query = self.query_proj(z_flat)  # [B, HW, s_dim]
        keys = torch.stack(memory_bank, dim=1)  # [B, T, s_dim]
        keys_proj = self.key_proj(keys)
        values = self.value_proj(keys)

        attn_out, _ = self.attn(query, keys_proj, values)  # [B, HW, s_dim]
        fused = self.fuse_proj(attn_out)  # [B, HW, C]
        fused = fused.permute(0, 2, 1).view(B, C, H, W)  # [B, C, H, W]

        return z_t + fused  # Residual connection