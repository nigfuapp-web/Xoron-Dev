"""
SOTA Mixture of Experts (MoE) components v2.0.

Features:
- Aux-Lossless MoE (no auxiliary loss needed, load balance through architecture)
- Isolated Shared Expert (always active, separate from routed experts)
- Fine-grained experts for better specialization
- Expert choice routing option
- Optional auxiliary losses for legacy compatibility
- Capacity factor for expert utilization
- FP16-native numerical stability
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

# FP16 constants
EPS = 1e-5


class MoERouter(nn.Module):
    """
    SOTA Router for Mixture of Experts v2.0 - FP16 native.
    
    Supports both traditional aux-loss routing and aux-lossless routing.
    """

    def __init__(self, hidden_size: int, num_experts: int, top_k: int = 2, 
                 noise_std: float = 0.01, capacity_factor: float = 1.25,
                 aux_lossless: bool = True):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.noise_std = noise_std
        self.capacity_factor = capacity_factor
        self.hidden_size = hidden_size
        self.aux_lossless = aux_lossless
        
        # Layer norm for input stability
        self.input_norm = nn.LayerNorm(hidden_size, eps=1e-5)
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        # Small init prevents large logits
        nn.init.normal_(self.gate.weight, mean=0.0, std=0.01)
        
        if aux_lossless:
            # Bias term for Aux-Lossless load balancing
            self.expert_bias = nn.Parameter(torch.zeros(num_experts))

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len, hidden_dim = hidden_states.shape
        hidden_flat = hidden_states.view(-1, hidden_dim)
        
        # Normalize input for stability
        hidden_norm = self.input_norm(hidden_flat)
        
        router_logits = self.gate(hidden_norm)
        
        # Add expert bias for Aux-Lossless load balancing
        if self.aux_lossless:
            router_logits = router_logits + self.expert_bias

        if self.training and self.noise_std > 0:
            noise = torch.randn_like(router_logits) * self.noise_std
            noisy_logits = router_logits + noise
        else:
            noisy_logits = router_logits

        router_probs = F.softmax(noisy_logits, dim=-1, dtype=hidden_states.dtype)
        
        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        
        # Normalize top-k probs
        prob_sum = top_k_probs.sum(dim=-1, keepdim=True).clamp(min=EPS)
        top_k_probs = top_k_probs / prob_sum

        return top_k_probs, top_k_indices, router_logits


class MoEExpert(nn.Module):
    """
    Single expert FFN with SwiGLU activation - FP16 native.
    """

    def __init__(self, hidden_size: int, intermediate_size: int, dropout: float = 0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = nn.SiLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self._init_weights()
    
    def _init_weights(self):
        # Small init for FP16 stability
        std = 0.02
        nn.init.normal_(self.gate_proj.weight, mean=0.0, std=std)
        nn.init.normal_(self.up_proj.weight, mean=0.0, std=std)
        nn.init.normal_(self.down_proj.weight, mean=0.0, std=std * 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.act_fn(self.gate_proj(x))
        up = self.up_proj(x)
        out = self.down_proj(gate * up)
        return self.dropout(out)


class SharedExpert(nn.Module):
    """
    Isolated Shared Expert (v2.0) - FP16 native.
    
    Always active, separate from routed experts.
    The shared expert processes all tokens independently of routing decisions.
    """
    
    def __init__(self, hidden_size: int, intermediate_size: int, dropout: float = 0.0,
                 isolated: bool = True):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.isolated = isolated
        
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = nn.SiLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Learnable gate for isolated shared expert contribution
        self.shared_gate = nn.Parameter(torch.ones(1) * 0.5)
        
        if isolated:
            # Separate normalization for isolation
            self.pre_norm = nn.LayerNorm(hidden_size, eps=1e-5)
        
        self._init_weights()
    
    def _init_weights(self):
        std = 0.02
        nn.init.normal_(self.gate_proj.weight, mean=0.0, std=std)
        nn.init.normal_(self.up_proj.weight, mean=0.0, std=std)
        nn.init.normal_(self.down_proj.weight, mean=0.0, std=std * 0.5)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.isolated:
            x = self.pre_norm(x)
        
        gate = self.act_fn(self.gate_proj(x))
        up = self.up_proj(x)
        out = self.down_proj(gate * up)
        out = self.dropout(out)
        return out * torch.sigmoid(self.shared_gate)


class MoELayer(nn.Module):
    """
    SOTA Mixture of Experts layer v2.0 - FP16 native.
    
    Supports Aux-Lossless MoE with Isolated Shared Expert.
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
        aux_lossless: bool = True,
        isolated_shared: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.use_shared_expert = use_shared_expert
        self.capacity_factor = capacity_factor
        self.aux_lossless = aux_lossless

        self.router = MoERouter(
            hidden_size, num_experts, num_experts_per_tok, 
            capacity_factor=capacity_factor, aux_lossless=aux_lossless
        )
        self.experts = nn.ModuleList([
            MoEExpert(hidden_size, intermediate_size, expert_dropout) 
            for _ in range(num_experts)
        ])
        
        if use_shared_expert:
            shared_size = shared_expert_intermediate_size or intermediate_size
            self.shared_expert = SharedExpert(
                hidden_size, shared_size, expert_dropout, isolated=isolated_shared
            )
        else:
            self.shared_expert = None

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, hidden_size = hidden_states.shape
        
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
                    weight = top_k_probs[mask, k:k+1]
                    final_output[mask] = final_output[mask] + weight * expert_output

        if self.shared_expert is not None:
            shared_output = self.shared_expert(hidden_flat)
            final_output = final_output + shared_output

        final_output = final_output.view(batch_size, seq_len, hidden_size)
        
        # Compute aux loss (returns 0 for aux-lossless mode)
        aux_loss = self._compute_aux_loss(router_logits, top_k_indices, num_tokens)

        return final_output, aux_loss

    def _compute_aux_loss(self, router_logits: torch.Tensor, top_k_indices: torch.Tensor, 
                          num_tokens: int) -> torch.Tensor:
        device = router_logits.device
        dtype = router_logits.dtype
        
        # Aux-Lossless: minimal z-loss only for stability
        if self.aux_lossless:
            z_loss = torch.logsumexp(router_logits, dim=-1).square().mean() * 0.0001
            return z_loss
        
        # Traditional aux loss computation
        router_probs = F.softmax(router_logits, dim=-1, dtype=dtype)

        expert_mask = F.one_hot(top_k_indices, self.num_experts).to(dtype)
        denominator = max(num_tokens * self.num_experts_per_tok, 1)
        tokens_per_expert = expert_mask.sum(dim=(0, 1)) / denominator
        avg_probs = router_probs.mean(dim=0)
        load_balance_loss = self.num_experts * (tokens_per_expert * avg_probs).sum()

        # z_loss for router stability
        z_loss = torch.logsumexp(router_logits, dim=-1).square().mean() * 0.001

        # Entropy loss to encourage exploration
        router_probs_safe = router_probs.clamp(EPS, 1.0 - EPS)
        log_probs = torch.log(router_probs_safe)
        entropy = -(router_probs_safe * log_probs).sum(dim=-1).mean()
        max_entropy = torch.log(torch.tensor(float(self.num_experts), device=device, dtype=dtype))
        entropy_loss = (max_entropy - entropy).clamp(min=0.0) * 0.01

        expert_usage = (tokens_per_expert > 0.01).to(dtype).mean()
        utilization_loss = (1.0 - expert_usage) * 0.1

        total_aux_loss = load_balance_loss + z_loss + entropy_loss + utilization_loss
        
        return total_aux_loss


class ExpertChoiceMoELayer(nn.Module):
    """
    Expert Choice MoE - FP16 native.
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
        
        self.input_norm = nn.LayerNorm(hidden_size, eps=1e-5)
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        nn.init.normal_(self.gate.weight, mean=0.0, std=0.01)
        
        self.experts = nn.ModuleList([
            MoEExpert(hidden_size, intermediate_size)
            for _ in range(num_experts)
        ])
        
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, hidden_size = hidden_states.shape
        hidden_flat = hidden_states.view(-1, hidden_size)
        num_tokens = hidden_flat.shape[0]
        
        hidden_norm = self.input_norm(hidden_flat)
        
        router_logits = self.gate(hidden_norm)
        router_probs = F.softmax(router_logits, dim=0, dtype=hidden_states.dtype)
        
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
            
            final_output[top_indices] = final_output[top_indices] + top_probs.unsqueeze(-1) * expert_output
            token_counts[top_indices] = token_counts[top_indices] + top_probs
        
        token_counts = token_counts.clamp(min=EPS)
        final_output = final_output / token_counts.unsqueeze(-1)
        
        final_output = final_output.view(batch_size, seq_len, hidden_size)
        
        aux_loss = torch.logsumexp(router_logits, dim=-1).square().mean() * 0.001
        
        return final_output, aux_loss
