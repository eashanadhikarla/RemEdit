import torch
import torch.nn as nn
import torch.nn.functional as F
from models.ddpm.deltadiffusion import AttnBlock

class PruningHead(nn.Module):
    """
    The trainable MLP that learns to predict token importance based on
    the token's features and the semantic editing direction.
    """
    def __init__(self, token_dim, edit_vector_dim, hidden_dim=256):
        super().__init__()
        self.layer1 = nn.Linear(token_dim + edit_vector_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.layer3 = nn.Linear(hidden_dim // 2, 1)
        self.activation = nn.SiLU()

    def forward(self, token_features, edit_vector):
        B, N, _ = token_features.shape
        edit_vector_expanded = edit_vector.unsqueeze(1).expand(-1, N, -1)
        combined_input = torch.cat([token_features, edit_vector_expanded], dim=-1)
        x = self.activation(self.layer1(combined_input))
        x = self.activation(self.layer2(x))
        x = self.layer3(x)
        return torch.sigmoid(x)

class PrunableAttnBlock(nn.Module):
    """
    This block wraps a pretrained AttnBlock. It can operate in three modes:
    1. Standard attention (for sanity checks).
    2. Soft-pruning (for training the PruningHead).
    3. Hard-pruning (for accelerated inference).
    """
    def __init__(self, original_attn_block: AttnBlock, pruning_head: PruningHead):
        super().__init__()
        # Store the original block to use its pretrained Q, K, V layers
        self.original_attn = original_attn_block
        self.pruning_head = pruning_head
        self.is_pruning_enabled = False
        self.prune_ratio = 0.0

    def set_pruning_status(self, is_enabled: bool, prune_ratio: float = 0.0):
        self.is_pruning_enabled = is_enabled
        self.prune_ratio = prune_ratio

    def _soft_pruning_for_training(self, x, clip_direction):
        # Calculate the unpruned output using the original block's forward pass
        unpruned_output = self.original_attn(x)
        attention_delta = unpruned_output - x

        # Get importance scores from the head
        tokens = x.flatten(2).transpose(1, 2)
        scores = self.pruning_head(tokens, clip_direction)

        # Gate the attention delta using the scores
        b, _, h, w = x.shape
        scores_reshaped = scores.permute(0, 2, 1).view(b, 1, h, w)
        gated_delta = attention_delta * scores_reshaped
        gated_output = x + gated_delta

        return {
            "gated_output": gated_output,
            "unpruned_output": unpruned_output,
            "scores": scores
        }

    def _hard_pruning_for_inference(self, x, clip_direction):
        b, c, h, w = x.shape
        n = h * w

        # Use the original block's layers
        h_ = self.original_attn.norm(x)
        tokens = h_.view(b, c, n).transpose(1, 2)

        # Get scores from the trained head
        importance_scores = self.pruning_head(tokens.detach(), clip_direction.detach())

        num_to_keep = int(n * (1.0 - self.prune_ratio))
        if num_to_keep >= n: return self.original_attn(x)
        if num_to_keep < 1: num_to_keep = 1

        _, keep_indices = torch.topk(importance_scores.squeeze(-1), k=num_to_keep, dim=1)

        q_all = self.original_attn.q(h_).view(b, c, n).transpose(1, 2)
        k_all = self.original_attn.k(h_).view(b, c, n).transpose(1, 2)
        v_all = self.original_attn.v(h_).view(b, c, n).transpose(1, 2)

        gather_indices = keep_indices.unsqueeze(-1).expand(-1, -1, c)
        q_kept = torch.gather(q_all, 1, gather_indices)
        k_kept = torch.gather(k_all, 1, gather_indices)
        v_kept = torch.gather(v_all, 1, gather_indices)

        attn_weights = torch.bmm(q_kept, k_kept.transpose(1, 2)) * (c ** (-0.5))
        attn_weights = F.softmax(attn_weights, dim=2)
        attended_v = torch.bmm(attn_weights, v_kept)

        # Pass-through pruned tokens by starting with a clone of the original features
        output_h_flat = tokens.clone()
        output_h_flat.scatter_(1, gather_indices, attended_v)

        output_h = output_h_flat.transpose(1, 2).reshape(b, c, h, w)
        output_h = self.original_attn.proj_out(output_h)

        return x + output_h

    def forward(self, x, clip_direction=None, prune_ratio=0.0):
        self.prune_ratio = prune_ratio
        if self.pruning_head.training:
            return self._soft_pruning_for_training(x, clip_direction)
        elif self.is_pruning_enabled and self.prune_ratio > 0.0:
            return self._hard_pruning_for_inference(x, clip_direction)
        else:
            # Fallback to the original block for sanity check or prune_ratio=0
            return self.original_attn(x)