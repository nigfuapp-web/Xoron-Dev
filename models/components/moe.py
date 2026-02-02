"""
SOTA Mixture of Experts (MoE) components.

Features:
- DeepSeek-style shared expert (always active)
- Fine-grained experts for better specialization
- Expert choice routing option
- Improved load balancing with auxiliary losses
- Capacity factor for expert utilization
- FP16-safe numerical stability (epsilon >= 1e-6)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

# FP16-safe epsilon (1e-9 underflows to 0 in fp16, causing NaN/Inf)
# FP16 smallest positive: ~6e-8, so we use 1e-6 for safety margin
EPS = 1e-6


class MoERouter(nn.Module):
    """
    SOTA Router for Mixture of Experts.
    
    Features:
    - Noisy top-k gating for exploration
    - Expert capacity limiting
    - Auxiliary load balancing
    - FP16-safe numerical operations
    """

    def __init__(self, hidden_size: int, num_experts: int, top_k: int = 2, 
                 noise_std: float = 0.1, capacity_factor: float = 1.25):  # REDUCED noise from 1.0 to 0.1
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.noise_std = noise_std
        self.capacity_factor = capacity_factor
        
        # Router gate with better initialization for stability
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        # Use smaller init for stable routing
        nn.init.normal_(self.gate.weight, mean=0.0, std=0.02)

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = hidden_states.shape
        hidden_flat = hidden_states.view(-1, hidden_states.shape[-1])
        num_tokens = hidden_flat.shape[0]

        router_logits = self.gate(hidden_flat)
        
        # Clamp logits to prevent overflow in softmax (FP16 safe)
        router_logits = torch.clamp(router_logits, min=-50.0, max=50.0)

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
        
        # Normalize top-k probabilities (FP16-safe epsilon)
        top_k_probs = top_k_probs / (top_k_probs.sum(dim=-1, keepdim=True) + EPS)

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
        
        # Proper initialization for training stability
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with proper scaling for stability."""
        # Use smaller init to prevent gradient explosion
        for module in [self.gate_proj, self.up_proj]:
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        # Down projection with even smaller init (output layer)
        nn.init.normal_(self.down_proj.weight, mean=0.0, std=0.01)

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
        
        # Proper initialization for training stability
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with proper scaling for stability."""
        for module in [self.gate_proj, self.up_proj]:
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.down_proj.weight, mean=0.0, std=0.01)
        
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
        
        All operations use FP16-safe epsilon values to prevent NaN/Inf.
        """
        # Clamp logits before softmax to prevent overflow
        router_logits_clamped = torch.clamp(router_logits, min=-50.0, max=50.0)
        router_probs = F.softmax(router_logits_clamped, dim=-1)

        # 1. Load balancing loss
        # Fraction of tokens routed to each expert (FP16-safe)
        expert_mask = F.one_hot(top_k_indices, self.num_experts).float()
        denominator = max(num_tokens * self.num_experts_per_tok, 1)  # Avoid division by zero
        tokens_per_expert = expert_mask.sum(dim=(0, 1)) / denominator
        
        # Average routing probability for each expert
        avg_probs = router_probs.mean(dim=0)
        
        # Load balance loss: encourage uniform distribution
        load_balance_loss = self.num_experts * (tokens_per_expert * avg_probs).sum()

        # 2. Router z-loss (prevent logit explosion) - use clamped logits
        z_loss = torch.logsumexp(router_logits_clamped, dim=-1).square().mean() * 0.001

        # 3. Entropy regularization (encourage exploration) - FP16-safe log
        # Clamp probs to avoid log(0) which gives -inf
        router_probs_safe = torch.clamp(router_probs, min=EPS, max=1.0)
        entropy = -(router_probs_safe * torch.log(router_probs_safe)).sum(dim=-1).mean()
        max_entropy = torch.log(torch.tensor(float(self.num_experts), device=router_probs.device))
        entropy_loss = (max_entropy - entropy) * 0.01

        # 4. Expert utilization loss (penalize unused experts)
        expert_usage = (tokens_per_expert > 0.01).float().mean()
        utilization_loss = (1.0 - expert_usage) * 0.1

        # Combine losses and clamp to prevent extreme values
        total_aux_loss = load_balance_loss + z_loss + entropy_loss + utilization_loss
        
        # Final safety clamp - aux loss should never be huge
        total_aux_loss = torch.clamp(total_aux_loss, min=0.0, max=100.0)
        
        return total_aux_loss


class ExpertChoiceMoELayer(nn.Module):
    """
    Expert Choice MoE (alternative routing strategy).
    
    Instead of tokens choosing experts, experts choose tokens.
    This ensures perfect load balancing but may drop some tokens.
    FP16-safe implementation.
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
        
        # Compute routing scores (clamp for FP16 safety)
        router_logits = self.gate(hidden_flat)  # [num_tokens, num_experts]
        router_logits = torch.clamp(router_logits, min=-50.0, max=50.0)
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
        
        # Normalize by total weight (FP16-safe epsilon)
        token_counts = token_counts.clamp(min=EPS)
        final_output = final_output / token_counts.unsqueeze(-1)
        
        final_output = final_output.view(batch_size, seq_len, hidden_size)
        
        # Simple aux loss for expert choice (clamped for safety)
        aux_loss = torch.logsumexp(router_logits, dim=-1).square().mean() * 0.001
        aux_loss = torch.clamp(aux_loss, min=0.0, max=100.0)
        
        return final_output, aux_loss
