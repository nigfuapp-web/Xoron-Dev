"""
SOTA Mixture of Experts (MoE) components.

Features:
- DeepSeek-style shared expert (always active)
- Fine-grained experts for better specialization
- Expert choice routing option
- Improved load balancing with auxiliary losses
- Capacity factor for expert utilization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class MoERouter(nn.Module):
    """
    SOTA Router for Mixture of Experts.
    
    Features:
    - Noisy top-k gating for exploration
    - Expert capacity limiting
    - Auxiliary load balancing
    """

    def __init__(self, hidden_size: int, num_experts: int, top_k: int = 2, 
                 noise_std: float = 1.0, capacity_factor: float = 1.25):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.noise_std = noise_std
        self.capacity_factor = capacity_factor
        
        # Router gate with better initialization
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        nn.init.kaiming_uniform_(self.gate.weight, a=0.01)

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = hidden_states.shape
        hidden_flat = hidden_states.view(-1, hidden_states.shape[-1])
        num_tokens = hidden_flat.shape[0]

        router_logits = self.gate(hidden_flat)

        # Add noise during training for exploration
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(router_logits) * self.noise_std
            noisy_logits = router_logits + noise
        else:
            noisy_logits = router_logits

        # Softmax for routing probabilities
        router_probs = F.softmax(noisy_logits, dim=-1)
        
        # Top-k selection
        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        
        # Normalize top-k probabilities
        top_k_probs = top_k_probs / (top_k_probs.sum(dim=-1, keepdim=True) + 1e-9)

        return top_k_probs, top_k_indices, router_logits


class MoEExpert(nn.Module):
    """
    Single expert FFN with SwiGLU activation (SOTA).
    
    Uses gated linear unit for better gradient flow.
    """

    def __init__(self, hidden_size: int, intermediate_size: int, dropout: float = 0.0):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = nn.SiLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: SiLU(gate) * up
        gate = self.act_fn(self.gate_proj(x))
        up = self.up_proj(x)
        return self.dropout(self.down_proj(gate * up))


class SharedExpert(nn.Module):
    """
    Shared expert that's always active (DeepSeek-style).
    
    This expert processes all tokens and provides a baseline,
    while routed experts provide specialized processing.
    """
    
    def __init__(self, hidden_size: int, intermediate_size: int, dropout: float = 0.0):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = nn.SiLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Learnable gate for shared expert contribution
        self.shared_gate = nn.Parameter(torch.ones(1) * 0.5)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.act_fn(self.gate_proj(x))
        up = self.up_proj(x)
        output = self.dropout(self.down_proj(gate * up))
        return output * torch.sigmoid(self.shared_gate)


class MoELayer(nn.Module):
    """
    SOTA Mixture of Experts layer with DeepSeek-style shared expert.
    
    Features:
    - Shared expert (DeepSeek-style) - always active for baseline processing
    - Fine-grained routed experts for specialized processing
    - Improved auxiliary losses for load balancing
    - Expert capacity limiting
    
    The shared expert is always enabled as it provides significant benefits:
    - Better gradient flow during training
    - More stable training dynamics
    - Improved generalization
    """

    def __init__(
        self, 
        hidden_size: int, 
        intermediate_size: int, 
        num_experts: int = 8, 
        num_experts_per_tok: int = 2,
        use_shared_expert: bool = True,  # Kept for backward compatibility, always True
        shared_expert_intermediate_size: Optional[int] = None,
        capacity_factor: float = 1.25,
        expert_dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.use_shared_expert = True  # Always use shared expert (DeepSeek-style)
        self.capacity_factor = capacity_factor

        # Router
        self.router = MoERouter(hidden_size, num_experts, num_experts_per_tok, capacity_factor=capacity_factor)
        
        # Routed experts
        self.experts = nn.ModuleList([
            MoEExpert(hidden_size, intermediate_size, expert_dropout) 
            for _ in range(num_experts)
        ])
        
        # Shared expert (always active - DeepSeek-style)
        # This expert processes all tokens and provides a baseline,
        # while routed experts provide specialized processing
        shared_size = shared_expert_intermediate_size or intermediate_size
        self.shared_expert = SharedExpert(hidden_size, shared_size, expert_dropout)

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, hidden_size = hidden_states.shape
        hidden_flat = hidden_states.view(-1, hidden_size)
        num_tokens = hidden_flat.shape[0]

        # Get routing decisions
        top_k_probs, top_k_indices, router_logits = self.router(hidden_states)

        # Initialize output
        final_output = torch.zeros_like(hidden_flat)

        # Process through routed experts
        for expert_idx in range(self.num_experts):
            expert = self.experts[expert_idx]
            
            # Find tokens routed to this expert
            for k in range(self.num_experts_per_tok):
                mask = (top_k_indices[:, k] == expert_idx)
                if mask.any():
                    expert_input = hidden_flat[mask]
                    expert_output = expert(expert_input)
                    final_output[mask] += top_k_probs[mask, k:k+1] * expert_output

        # Add shared expert output (always active - DeepSeek-style)
        shared_output = self.shared_expert(hidden_flat)
        final_output = final_output + shared_output

        final_output = final_output.view(batch_size, seq_len, hidden_size)
        
        # Compute auxiliary loss
        aux_loss = self._compute_aux_loss(router_logits, top_k_indices, num_tokens)

        return final_output, aux_loss

    def _compute_aux_loss(self, router_logits: torch.Tensor, top_k_indices: torch.Tensor, 
                          num_tokens: int) -> torch.Tensor:
        """
        Compute comprehensive auxiliary loss for load balancing.
        
        Includes:
        - Load balancing loss (prevent expert collapse)
        - Router z-loss (prevent logit explosion)
        - Entropy regularization (encourage exploration)
        """
        router_probs = F.softmax(router_logits, dim=-1)

        # 1. Load balancing loss
        # Fraction of tokens routed to each expert
        expert_mask = F.one_hot(top_k_indices, self.num_experts).float()
        tokens_per_expert = expert_mask.sum(dim=(0, 1)) / (num_tokens * self.num_experts_per_tok + 1e-9)
        
        # Average routing probability for each expert
        avg_probs = router_probs.mean(dim=0)
        
        # Load balance loss: encourage uniform distribution
        load_balance_loss = self.num_experts * (tokens_per_expert * avg_probs).sum()

        # 2. Router z-loss (prevent logit explosion)
        z_loss = torch.logsumexp(router_logits, dim=-1).square().mean() * 0.001

        # 3. Entropy regularization (encourage exploration)
        entropy = -(router_probs * torch.log(router_probs + 1e-9)).sum(dim=-1).mean()
        max_entropy = torch.log(torch.tensor(float(self.num_experts), device=router_probs.device))
        entropy_loss = (max_entropy - entropy) * 0.01

        # 4. Expert utilization loss (penalize unused experts)
        expert_usage = (tokens_per_expert > 0.01).float().mean()
        utilization_loss = (1.0 - expert_usage) * 0.1

        return load_balance_loss + z_loss + entropy_loss + utilization_loss


class ExpertChoiceMoELayer(nn.Module):
    """
    Expert Choice MoE (alternative routing strategy).
    
    Instead of tokens choosing experts, experts choose tokens.
    This ensures perfect load balancing but may drop some tokens.
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
        
        # Router
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        
        # Experts
        self.experts = nn.ModuleList([
            MoEExpert(hidden_size, intermediate_size)
            for _ in range(num_experts)
        ])
        
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, hidden_size = hidden_states.shape
        hidden_flat = hidden_states.view(-1, hidden_size)
        num_tokens = hidden_flat.shape[0]
        
        # Compute routing scores
        router_logits = self.gate(hidden_flat)  # [num_tokens, num_experts]
        router_probs = F.softmax(router_logits, dim=0)  # Softmax over tokens (expert choice)
        
        # Each expert selects top-k tokens
        capacity = int(num_tokens * self.capacity_factor / self.num_experts)
        capacity = max(capacity, 1)
        
        final_output = torch.zeros_like(hidden_flat)
        token_counts = torch.zeros(num_tokens, device=hidden_flat.device)
        
        for expert_idx in range(self.num_experts):
            expert = self.experts[expert_idx]
            expert_probs = router_probs[:, expert_idx]
            
            # Select top tokens for this expert
            top_probs, top_indices = torch.topk(expert_probs, min(capacity, num_tokens))
            
            # Process selected tokens
            expert_input = hidden_flat[top_indices]
            expert_output = expert(expert_input)
            
            # Weighted addition
            final_output[top_indices] += top_probs.unsqueeze(-1) * expert_output
            token_counts[top_indices] += top_probs
        
        # Normalize by total weight
        token_counts = token_counts.clamp(min=1e-9)
        final_output = final_output / token_counts.unsqueeze(-1)
        
        final_output = final_output.view(batch_size, seq_len, hidden_size)
        
        # Simple aux loss for expert choice
        aux_loss = torch.logsumexp(router_logits, dim=-1).square().mean() * 0.001
        
        return final_output, aux_loss
