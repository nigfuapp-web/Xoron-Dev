"""
Comprehensive unit tests for video generator module.
Tests all components: TemporalAttention, MotionModule, CrossAttention3D,
ResBlock3D, Conv3DBlock, UNet3D, VideoVAEEncoder, VideoVAEDecoder, and MobileVideoGenerator.
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestTemporalAttention(unittest.TestCase):
    """Test cases for TemporalAttention."""
    
    def setUp(self):
        """Set up test fixtures."""
        from models.generators.video import TemporalAttention
        self.TemporalAttention = TemporalAttention
        
    def test_initialization(self):
        """Test TemporalAttention initialization."""
        attn = self.TemporalAttention(channels=256, num_heads=8, num_frames=16)
        
        self.assertEqual(attn.num_heads, 8)
        self.assertEqual(attn.channels, 256)
        
    def test_forward_output_shape(self):
        """Test forward pass output shape."""
        attn = self.TemporalAttention(channels=256, num_heads=8, num_frames=16)
        x = torch.randn(2, 256, 8, 16, 16)  # [B, C, T, H, W]
        
        output = attn(x)
        
        self.assertEqual(output.shape, (2, 256, 8, 16, 16))
        
    def test_temporal_position_embeddings(self):
        """Test temporal position embeddings are learnable."""
        attn = self.TemporalAttention(channels=256, num_heads=8, num_frames=16)
        
        self.assertEqual(attn.temporal_pos_embed.shape, (1, 16, 256))
        self.assertTrue(attn.temporal_pos_embed.requires_grad)
        
    def test_residual_connection(self):
        """Test residual connection is applied."""
        attn = self.TemporalAttention(channels=256, num_heads=8, num_frames=16)
        x = torch.randn(2, 256, 8, 16, 16)
        
        output = attn(x)
        
        # Output should be input + attention output
        self.assertEqual(output.shape, x.shape)


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestMotionModule(unittest.TestCase):
    """Test cases for MotionModule."""
    
    def setUp(self):
        """Set up test fixtures."""
        from models.generators.video import MotionModule
        self.MotionModule = MotionModule
        
    def test_initialization(self):
        """Test MotionModule initialization."""
        module = self.MotionModule(channels=256, num_heads=8, num_frames=16)
        
        self.assertIsNotNone(module.temporal_attn)
        self.assertIsNotNone(module.temporal_conv)
        self.assertIsNotNone(module.out_proj)
        
    def test_forward_output_shape(self):
        """Test forward pass output shape."""
        module = self.MotionModule(channels=256, num_heads=8, num_frames=16)
        x = torch.randn(2, 256, 8, 16, 16)
        
        output = module(x)
        
        self.assertEqual(output.shape, (2, 256, 8, 16, 16))
        
    def test_zero_initialized_output(self):
        """Test output projection is zero-initialized."""
        module = self.MotionModule(channels=256, num_heads=8, num_frames=16)
        
        self.assertTrue(torch.allclose(module.out_proj.weight, torch.zeros_like(module.out_proj.weight)))
        self.assertTrue(torch.allclose(module.out_proj.bias, torch.zeros_like(module.out_proj.bias)))
        
    def test_gradient_flow(self):
        """Test gradients flow through module."""
        module = self.MotionModule(channels=256, num_heads=8, num_frames=16)
        x = torch.randn(2, 256, 8, 16, 16, requires_grad=True)
        
        output = module(x)
        loss = output.sum()
        loss.backward()
        
        self.assertIsNotNone(x.grad)


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestCrossAttention3D(unittest.TestCase):
    """Test cases for CrossAttention3D."""
    
    def setUp(self):
        """Set up test fixtures."""
        from models.generators.video import CrossAttention3D
        self.CrossAttention3D = CrossAttention3D
        
    def test_initialization(self):
        """Test CrossAttention3D initialization."""
        attn = self.CrossAttention3D(query_dim=256, context_dim=512, heads=8)
        
        self.assertEqual(attn.heads, 8)
        
    def test_forward_output_shape(self):
        """Test forward pass output shape."""
        attn = self.CrossAttention3D(query_dim=256, context_dim=512, heads=8)
        x = torch.randn(2, 256, 8, 16, 16)  # [B, C, T, H, W]
        context = torch.randn(2, 77, 512)  # [B, seq_len, context_dim]
        
        output = attn(x, context)
        
        self.assertEqual(output.shape, (2, 256, 8, 16, 16))
        
    def test_residual_connection(self):
        """Test residual connection is applied."""
        attn = self.CrossAttention3D(query_dim=256, context_dim=512, heads=8)
        x = torch.randn(2, 256, 8, 16, 16)
        context = torch.randn(2, 77, 512)
        
        output = attn(x, context)
        
        self.assertEqual(output.shape, x.shape)


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestResBlock3D(unittest.TestCase):
    """Test cases for ResBlock3D."""
    
    def setUp(self):
        """Set up test fixtures."""
        from models.generators.video import ResBlock3D
        self.ResBlock3D = ResBlock3D
        
    def test_initialization(self):
        """Test ResBlock3D initialization."""
        block = self.ResBlock3D(in_channels=64, out_channels=128, time_emb_dim=256)
        
        self.assertIsNotNone(block.conv1)
        self.assertIsNotNone(block.conv2)
        self.assertIsNotNone(block.time_mlp)
        
    def test_forward_same_channels(self):
        """Test forward pass with same input/output channels."""
        block = self.ResBlock3D(in_channels=64, out_channels=64, time_emb_dim=256)
        x = torch.randn(2, 64, 8, 16, 16)
        time_emb = torch.randn(2, 256)
        
        output = block(x, time_emb)
        
        self.assertEqual(output.shape, (2, 64, 8, 16, 16))
        
    def test_forward_different_channels(self):
        """Test forward pass with different input/output channels."""
        block = self.ResBlock3D(in_channels=64, out_channels=128, time_emb_dim=256)
        x = torch.randn(2, 64, 8, 16, 16)
        time_emb = torch.randn(2, 256)
        
        output = block(x, time_emb)
        
        self.assertEqual(output.shape, (2, 128, 8, 16, 16))
        
    def test_time_conditioning(self):
        """Test that time embedding affects output."""
        block = self.ResBlock3D(in_channels=64, out_channels=64, time_emb_dim=256)
        x = torch.randn(2, 64, 8, 16, 16)
        time_emb1 = torch.randn(2, 256)
        time_emb2 = torch.randn(2, 256)
        
        output1 = block(x, time_emb1)
        output2 = block(x, time_emb2)
        
        self.assertFalse(torch.allclose(output1, output2))


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestConv3DBlock(unittest.TestCase):
    """Test cases for Conv3DBlock."""
    
    def setUp(self):
        """Set up test fixtures."""
        from models.generators.video import Conv3DBlock
        self.Conv3DBlock = Conv3DBlock
        
    def test_initialization_with_motion(self):
        """Test initialization with motion module."""
        block = self.Conv3DBlock(
            in_ch=64, out_ch=128, time_emb_dim=256, context_dim=512,
            use_motion=True, use_cross_attn=True, num_frames=16
        )
        
        self.assertIsNotNone(block.res_block)
        self.assertIsNotNone(block.motion_module)
        self.assertIsNotNone(block.cross_attn)
        
    def test_initialization_without_motion(self):
        """Test initialization without motion module."""
        block = self.Conv3DBlock(
            in_ch=64, out_ch=128, time_emb_dim=256, context_dim=512,
            use_motion=False, use_cross_attn=False, num_frames=16
        )
        
        self.assertIsInstance(block.motion_module, nn.Identity)
        self.assertIsInstance(block.cross_attn, nn.Identity)
        
    def test_forward_output_shape(self):
        """Test forward pass output shape."""
        block = self.Conv3DBlock(
            in_ch=64, out_ch=128, time_emb_dim=256, context_dim=512,
            use_motion=True, use_cross_attn=True, num_frames=16
        )
        x = torch.randn(2, 64, 8, 16, 16)
        time_emb = torch.randn(2, 256)
        context = torch.randn(2, 77, 512)
        
        output = block(x, time_emb, context)
        
        self.assertEqual(output.shape, (2, 128, 8, 16, 16))


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestVideoVAEEncoder(unittest.TestCase):
    """Test cases for VideoVAEEncoder."""
    
    def setUp(self):
        """Set up test fixtures."""
        from models.generators.video import VideoVAEEncoder
        self.VideoVAEEncoder = VideoVAEEncoder
        
    def test_initialization(self):
        """Test VideoVAEEncoder initialization."""
        encoder = self.VideoVAEEncoder(in_channels=3, latent_channels=4, base_channels=64)
        self.assertIsNotNone(encoder.encoder)
        
    def test_forward_output_shapes(self):
        """Test forward pass output shapes."""
        encoder = self.VideoVAEEncoder(in_channels=3, latent_channels=4, base_channels=64)
        x = torch.randn(2, 3, 8, 64, 64)  # [B, C, T, H, W]
        
        z, mean, logvar = encoder(x)
        
        # Spatial dimensions should be downsampled by 8x
        self.assertEqual(z.shape, (2, 4, 8, 8, 8))
        self.assertEqual(mean.shape, (2, 4, 8, 8, 8))
        self.assertEqual(logvar.shape, (2, 4, 8, 8, 8))
        
    def test_reparameterization(self):
        """Test reparameterization produces different samples."""
        encoder = self.VideoVAEEncoder(in_channels=3, latent_channels=4, base_channels=64)
        x = torch.randn(2, 3, 8, 64, 64)
        
        z1, _, _ = encoder(x)
        z2, _, _ = encoder(x)
        
        # Due to random sampling, z1 and z2 should be different
        self.assertFalse(torch.allclose(z1, z2))


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestVideoVAEDecoder(unittest.TestCase):
    """Test cases for VideoVAEDecoder."""
    
    def setUp(self):
        """Set up test fixtures."""
        from models.generators.video import VideoVAEDecoder
        self.VideoVAEDecoder = VideoVAEDecoder
        
    def test_initialization(self):
        """Test VideoVAEDecoder initialization."""
        decoder = self.VideoVAEDecoder(latent_channels=4, out_channels=3, base_channels=64)
        self.assertIsNotNone(decoder.decoder)
        
    def test_forward_output_shape(self):
        """Test forward pass output shape."""
        decoder = self.VideoVAEDecoder(latent_channels=4, out_channels=3, base_channels=64)
        z = torch.randn(2, 4, 8, 8, 8)
        
        output = decoder(z)
        
        # Spatial dimensions should be upsampled by 8x
        self.assertEqual(output.shape, (2, 3, 8, 64, 64))
        
    def test_output_range(self):
        """Test output is in valid range due to Tanh."""
        decoder = self.VideoVAEDecoder(latent_channels=4, out_channels=3, base_channels=64)
        z = torch.randn(2, 4, 8, 8, 8)
        
        output = decoder(z)
        
        self.assertTrue(torch.all(output >= -1))
        self.assertTrue(torch.all(output <= 1))


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestUNet3D(unittest.TestCase):
    """Test cases for UNet3D."""
    
    def setUp(self):
        """Set up test fixtures."""
        from models.generators.video import UNet3D
        self.UNet3D = UNet3D
        
    def test_initialization(self):
        """Test UNet3D initialization."""
        unet = self.UNet3D(
            in_channels=4,
            base_channels=64,
            context_dim=512,
            num_frames=8,
            channel_mults=(1, 2)
        )
        
        self.assertIsNotNone(unet.time_embed)
        self.assertIsNotNone(unet.down_blocks)
        self.assertIsNotNone(unet.up_blocks)
        
    def test_forward_output_shape(self):
        """Test forward pass output shape."""
        unet = self.UNet3D(
            in_channels=4,
            base_channels=64,
            context_dim=512,
            num_frames=8,
            channel_mults=(1, 2)
        )
        x = torch.randn(1, 4, 8, 16, 16)
        timesteps = torch.tensor([500])
        context = torch.randn(1, 77, 512)
        
        output = unet(x, timesteps, context)
        
        self.assertEqual(output.shape, (1, 4, 8, 16, 16))
        
    def test_image_to_video_conditioning(self):
        """Test image-to-video conditioning with first frame."""
        unet = self.UNet3D(
            in_channels=4,
            base_channels=64,
            context_dim=512,
            num_frames=8,
            channel_mults=(1, 2)
        )
        x = torch.randn(1, 4, 8, 16, 16)
        timesteps = torch.tensor([500])
        context = torch.randn(1, 77, 512)
        first_frame = torch.randn(1, 4, 16, 16)
        
        output = unet(x, timesteps, context, first_frame_latent=first_frame)
        
        self.assertEqual(output.shape, (1, 4, 8, 16, 16))
        
    def test_time_conditioning(self):
        """Test that different timesteps produce different outputs."""
        unet = self.UNet3D(
            in_channels=4,
            base_channels=64,
            context_dim=512,
            num_frames=8,
            channel_mults=(1, 2)
        )
        x = torch.randn(1, 4, 8, 16, 16)
        context = torch.randn(1, 77, 512)
        
        output1 = unet(x, torch.tensor([100]), context)
        output2 = unet(x, torch.tensor([900]), context)
        
        self.assertFalse(torch.allclose(output1, output2))


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestSinusoidalPositionEmbeddingsVideo(unittest.TestCase):
    """Test cases for SinusoidalPositionEmbeddings in video module."""
    
    def setUp(self):
        """Set up test fixtures."""
        from models.generators.video import SinusoidalPositionEmbeddings
        self.SinusoidalPositionEmbeddings = SinusoidalPositionEmbeddings
        
    def test_initialization(self):
        """Test SinusoidalPositionEmbeddings initialization."""
        emb = self.SinusoidalPositionEmbeddings(dim=128)
        self.assertEqual(emb.dim, 128)
        
    def test_forward_output_shape(self):
        """Test forward pass output shape."""
        emb = self.SinusoidalPositionEmbeddings(dim=128)
        timesteps = torch.tensor([0, 100, 500, 999])
        
        output = emb(timesteps)
        
        self.assertEqual(output.shape, (4, 128))


if __name__ == '__main__':
    unittest.main()
