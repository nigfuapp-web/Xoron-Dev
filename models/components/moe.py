"""
SOTA Mixture of Experts (MoE) components.

Features:
- DeepSeek-style shared expert (always active)
- Fine-grained experts for better specialization
- Expert choice routing option
- Improved load balancing with auxiliary losses
- Capacity factor for expert utilization
- FP16/BF16-safe numerical stability throughout
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

# FP16-safe epsilon - must be large enough to not underflow
EPS = 1e-5

# Maximum absolute value for hidden states to prevent overflow
MAX_HIDDEN = 65000.0  # FP16 max is ~65504


def safe_clamp(x: torch.Tensor, max_val: float = MAX_HIDDEN) -> torch.Tensor:
    """Clamp tensor values to prevent FP16 overflow, handling NaN/Inf."""
    if x.numel() == 0:
        return x
    # Replace NaN with 0, Inf with max_val
    x = torch.where(torch.isnan(x), torch.zeros_like(x), x)
    x = torch.where(torch.isinf(x), torch.full_like(x, max_val) * torch.sign(x), x)
    return torch.clamp(x, min=-max_val, max=max_val)


class MoERouter(nn.Module):
    """
    SOTA Router for Mixture of Experts with robust numerical stability.
    """

    def __init__(self, hidden_size: int, num_experts: int, top_k: int = 2, 
                 noise_std: float = 0.01, capacity_factor: float = 1.25):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.noise_std = noise_std
        self.capacity_factor = capacity_factor
        
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        nn.init.xavier_uniform_(self.gate.weight, gain=0.1)

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len, hidden_dim = hidden_states.shape
        hidden_flat = hidden_states.view(-1, hidden_dim)
        
        # Normalize input for stable routing
        hidden_norm = F.layer_norm(hidden_flat, [hidden_dim])
        hidden_norm = safe_clamp(hidden_norm, 100.0)
        
        router_logits = self.gate(hidden_norm)
        router_logits = torch.clamp(router_logits, min=-20.0, max=20.0)

        if self.training and self.noise_std > 0:
            noise = torch.randn_like(router_logits) * self.noise_std
            noisy_logits = router_logits + noise
        else:
            noisy_logits = router_logits

        # Stable softmax
        router_probs = F.softmax(noisy_logits.float(), dim=-1).to(hidden_states.dtype)
        router_probs = torch.clamp(router_probs, min=EPS, max=1.0)
        
        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        
        # Normalize with safe denominator
        prob_sum = top_k_probs.sum(dim=-1, keepdim=True)
        prob_sum = torch.clamp(prob_sum, min=EPS)
        top_k_probs = top_k_probs / prob_sum

        return top_k_probs, top_k_indices, router_logits


class MoEExpert(nn.Module):
    """
    Single expert FFN with SwiGLU activation and numerical stability.
    """

    def __init__(self, hidden_size: int, intermediate_size: int, dropout: float = 0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = nn.SiLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.gate_proj.weight, gain=0.5)
        nn.init.xavier_uniform_(self.up_proj.weight, gain=0.5)
        nn.init.xavier_uniform_(self.down_proj.weight, gain=0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = safe_clamp(x, 1000.0)
        gate = self.act_fn(self.gate_proj(x))
        gate = safe_clamp(gate, 1000.0)
        up = self.up_proj(x)
        up = safe_clamp(up, 1000.0)
        out = self.down_proj(gate * up)
        out = safe_clamp(out, 1000.0)
        return self.dropout(out)


class SharedExpert(nn.Module):
    """
    Shared expert that's always active (DeepSeek-style) with numerical stability.
    """
    
    def __init__(self, hidden_size: int, intermediate_size: int, dropout: float = 0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = nn.SiLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.shared_gate = nn.Parameter(torch.ones(1) * 0.5)
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.gate_proj.weight, gain=0.5)
        nn.init.xavier_uniform_(self.up_proj.weight, gain=0.5)
        nn.init.xavier_uniform_(self.down_proj.weight, gain=0.1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = safe_clamp(x, 1000.0)
        gate = self.act_fn(self.gate_proj(x))
        gate = safe_clamp(gate, 1000.0)
        up = self.up_proj(x)
        up = safe_clamp(up, 1000.0)
        out = self.down_proj(gate * up)
        out = safe_clamp(out, 1000.0)
        out = self.dropout(out)
        return out * torch.sigmoid(self.shared_gate)


class MoELayer(nn.Module):
    """
    SOTA Mixture of Experts layer with DeepSeek-style shared expert and numerical stability.
    """

    def __init__(
        self, 
        hidden_size: int, 
        intermediate_size: int, 
        num_experts: int = 8, 
        num_experts_per_tok: int = 2,
        use_shared_expert: bool = True,
        shared_expert_intermediate_size: Optional[int] = None,
        capacity_factor: float = 1.25,
        expert_dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.use_shared_expert = True
        self.capacity_factor = capacity_factor

        self.router = MoERouter(hidden_size, num_experts, num_experts_per_tok, capacity_factor=capacity_factor)
        self.experts = nn.ModuleList([
            MoEExpert(hidden_size, intermediate_size, expert_dropout) 
            for _ in range(num_experts)
        ])
        shared_size = shared_expert_intermediate_size or intermediate_size
        self.shared_expert = SharedExpert(hidden_size, shared_size, expert_dropout)

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Clamp input to prevent propagation of bad values
        hidden_states = safe_clamp(hidden_states, 1000.0)
        hidden_flat = hidden_states.view(-1, hidden_size)
        num_tokens = hidden_flat.shape[0]

        top_k_probs, top_k_indices, router_logits = self.router(hidden_states)

        final_output = torch.zeros_like(hidden_flat)

        for expert_idx in range(self.num_experts):
            expert = self.experts[expert_idx]
            for k in range(self.num_experts_per_tok):
                mask = (top_k_indices[:, k] == expert_idx)
                if mask.any():
                    expert_input = hidden_flat[mask]
                    expert_output = expert(expert_input)
                    expert_output = safe_clamp(expert_output, 1000.0)
                    weight = top_k_probs[mask, k:k+1]
                    final_output[mask] += weight * expert_output

        shared_output = self.shared_expert(hidden_flat)
        shared_output = safe_clamp(shared_output, 1000.0)
        final_output = final_output + shared_output
        final_output = safe_clamp(final_output, 1000.0)

        final_output = final_output.view(batch_size, seq_len, hidden_size)
        aux_loss = self._compute_aux_loss(router_logits, top_k_indices, num_tokens)

        return final_output, aux_loss

    def _compute_aux_loss(self, router_logits: torch.Tensor, top_k_indices: torch.Tensor, 
                          num_tokens: int) -> torch.Tensor:
        device = router_logits.device
        dtype = router_logits.dtype
        
        router_logits_clamped = torch.clamp(router_logits, min=-20.0, max=20.0)
        router_probs = F.softmax(router_logits_clamped.float(), dim=-1)
        router_probs = torch.clamp(router_probs, min=EPS, max=1.0)

        expert_mask = F.one_hot(top_k_indices, self.num_experts).float()
        denominator = max(num_tokens * self.num_experts_per_tok, 1)
        tokens_per_expert = expert_mask.sum(dim=(0, 1)) / denominator
        avg_probs = router_probs.mean(dim=0)
        load_balance_loss = self.num_experts * (tokens_per_expert * avg_probs).sum()

        z_loss = torch.logsumexp(router_logits_clamped.float(), dim=-1).square().mean() * 0.001

        router_probs_safe = torch.clamp(router_probs, min=EPS, max=1.0 - EPS)
        entropy = -(router_probs_safe * torch.log(router_probs_safe)).sum(dim=-1).mean()
        max_entropy = torch.log(torch.tensor(float(self.num_experts), device=device))
        entropy_loss = torch.clamp(max_entropy - entropy, min=0.0) * 0.01

        expert_usage = (tokens_per_expert > 0.01).float().mean()
        utilization_loss = (1.0 - expert_usage) * 0.1

        total_aux_loss = load_balance_loss + z_loss + entropy_loss + utilization_loss
        total_aux_loss = torch.clamp(total_aux_loss, min=0.0, max=10.0)
        
        if torch.isnan(total_aux_loss) or torch.isinf(total_aux_loss):
            total_aux_loss = torch.tensor(0.1, device=device, dtype=dtype)
        
        return total_aux_loss.to(dtype)


class ExpertChoiceMoELayer(nn.Module):
    """
    Expert Choice MoE with numerical stability.
    """
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int = 8,
        capacity_factor: float = 1.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor
        
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        nn.init.xavier_uniform_(self.gate.weight, gain=0.1)
        
        self.experts = nn.ModuleList([
            MoEExpert(hidden_size, intermediate_size)
            for _ in range(num_experts)
        ])
        
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, hidden_size = hidden_states.shape
        hidden_states = safe_clamp(hidden_states, 1000.0)
        hidden_flat = hidden_states.view(-1, hidden_size)
        num_tokens = hidden_flat.shape[0]
        
        hidden_norm = F.layer_norm(hidden_flat, [hidden_size])
        hidden_norm = safe_clamp(hidden_norm, 100.0)
        
        router_logits = self.gate(hidden_norm)
        router_logits = torch.clamp(router_logits, min=-20.0, max=20.0)
        router_probs = F.softmax(router_logits.float(), dim=0).to(hidden_states.dtype)
        router_probs = torch.clamp(router_probs, min=EPS, max=1.0)
        
        capacity = int(num_tokens * self.capacity_factor / self.num_experts)
        capacity = max(capacity, 1)
        
        final_output = torch.zeros_like(hidden_flat)
        token_counts = torch.zeros(num_tokens, device=hidden_flat.device, dtype=hidden_flat.dtype)
        
        for expert_idx in range(self.num_experts):
            expert = self.experts[expert_idx]
            expert_probs = router_probs[:, expert_idx]
            
            top_probs, top_indices = torch.topk(expert_probs, min(capacity, num_tokens))
            
            expert_input = hidden_flat[top_indices]
            expert_output = expert(expert_input)
            expert_output = safe_clamp(expert_output, 1000.0)
            
            final_output[top_indices] += top_probs.unsqueeze(-1) * expert_output
            token_counts[top_indices] += top_probs
        
        token_counts = torch.clamp(token_counts, min=EPS)
        final_output = final_output / token_counts.unsqueeze(-1)
        final_output = safe_clamp(final_output, 1000.0)
        
        final_output = final_output.view(batch_size, seq_len, hidden_size)
        
        aux_loss = torch.logsumexp(router_logits.float(), dim=-1).square().mean() * 0.001
        aux_loss = torch.clamp(aux_loss, min=0.0, max=10.0)
        
        if torch.isnan(aux_loss) or torch.isinf(aux_loss):
            aux_loss = torch.tensor(0.1, device=hidden_states.device, dtype=hidden_states.dtype)
        
        return final_output, aux_loss
