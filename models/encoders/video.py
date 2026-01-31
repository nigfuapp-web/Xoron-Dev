"""
Video encoder with temporal attention.

Uses the same vision encoder (SigLIP 2 or CLIP) for frame encoding,
with temporal attention to capture motion and temporal relationships.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.encoders.vision import VisionEncoder


class VideoEncoder(nn.Module):
    """
    Video encoder with temporal attention over frame features.
    
    Uses SigLIP 2 (or CLIP) for per-frame encoding, then applies
    temporal attention to capture motion and temporal relationships.
    """

    def __init__(self, vision_encoder: VisionEncoder, max_frames: int = 32):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.max_frames = max_frames
        self.hidden_size = vision_encoder.hidden_size
        
        # Get expected image size from vision encoder
        self.image_size = getattr(vision_encoder, 'image_size', 384)

        # Temporal attention for capturing motion
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=8,
            batch_first=True,
            dropout=0.1,
        )
        self.temporal_norm = nn.LayerNorm(self.hidden_size)
        
        # Temporal feed-forward
        self.temporal_ff = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size * 4, self.hidden_size),
            nn.Dropout(0.1),
        )
        self.temporal_ff_norm = nn.LayerNorm(self.hidden_size)

        # Learnable frame position embeddings
        self.frame_pos_embed = nn.Parameter(torch.randn(1, max_frames, self.hidden_size) * 0.02)
        
        # Learnable temporal CLS token for video-level representation
        self.temporal_cls = nn.Parameter(torch.randn(1, 1, self.hidden_size) * 0.02)

        print(f"   🎬 Video encoder: max {max_frames} frames @ {self.image_size}x{self.image_size}")

    def forward(self, frames: torch.Tensor, return_all_frames: bool = False) -> torch.Tensor:
        """
        Process video frames.
        
        Args:
            frames: [B, T, C, H, W] tensor of video frames
            return_all_frames: If True, return all frame features; else return pooled
            
        Returns:
            [B, T, hidden_size] if return_all_frames else [B, hidden_size]
        """
        batch_size, num_frames = frames.shape[:2]

        # Flatten frames for batch processing
        frames_flat = frames.view(-1, *frames.shape[2:])

        # Resize to expected size for vision encoder
        if frames_flat.shape[-1] != self.image_size or frames_flat.shape[-2] != self.image_size:
            frames_flat = F.interpolate(
                frames_flat, 
                size=(self.image_size, self.image_size), 
                mode='bilinear', 
                align_corners=False
            )

        # Extract frame features using vision encoder
        # Use no_grad if vision encoder is frozen
        if not any(p.requires_grad for p in self.vision_encoder.parameters()):
            with torch.no_grad():
                frame_features = self.vision_encoder(frames_flat)
        else:
            frame_features = self.vision_encoder(frames_flat)

        # Pool spatial features (mean over patches)
        frame_features = frame_features.mean(dim=1)  # [B*T, hidden_size]
        frame_features = frame_features.view(batch_size, num_frames, -1)  # [B, T, hidden_size]

        # Add frame position embeddings
        frame_features = frame_features + self.frame_pos_embed[:, :num_frames]
        
        # Add temporal CLS token
        temporal_cls = self.temporal_cls.expand(batch_size, -1, -1)
        frame_features = torch.cat([temporal_cls, frame_features], dim=1)  # [B, T+1, hidden_size]

        # Temporal self-attention
        attended, _ = self.temporal_attention(frame_features, frame_features, frame_features)
        frame_features = self.temporal_norm(frame_features + attended)
        
        # Temporal feed-forward
        frame_features = self.temporal_ff_norm(frame_features + self.temporal_ff(frame_features))

        if return_all_frames:
            # Return all frame features (excluding CLS)
            return frame_features[:, 1:]
        else:
            # Return temporal CLS token as video representation
            return frame_features[:, 0]
    
    def encode_frames_separately(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Encode frames without temporal attention (for generation conditioning).
        
        Args:
            frames: [B, T, C, H, W] tensor of video frames
            
        Returns:
            [B, T, hidden_size] tensor of frame features
        """
        batch_size, num_frames = frames.shape[:2]
        frames_flat = frames.view(-1, *frames.shape[2:])
        
        if frames_flat.shape[-1] != self.image_size or frames_flat.shape[-2] != self.image_size:
            frames_flat = F.interpolate(
                frames_flat,
                size=(self.image_size, self.image_size),
                mode='bilinear',
                align_corners=False
            )
        
        with torch.no_grad():
            frame_features = self.vision_encoder(frames_flat)
        
        frame_features = frame_features.mean(dim=1)
        return frame_features.view(batch_size, num_frames, -1)
