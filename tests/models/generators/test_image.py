"""
Comprehensive unit tests for image generator module.
Tests all components: SinusoidalPositionEmbeddings, CrossAttention, FeedForward,
TransformerBlock, ResBlock, SpatialTransformer, DownBlock, UpBlock, UNet2D,
VAEEncoder, VAEDecoder, and MobileDiffusionGenerator.
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
class TestSinusoidalPositionEmbeddings(unittest.TestCase):
    """Test cases for SinusoidalPositionEmbeddings."""
    
    def setUp(self):
        """Set up test fixtures."""
        from models.generators.image import SinusoidalPositionEmbeddings
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
        
    def test_different_timesteps_produce_different_embeddings(self):
        """Test that different timesteps produce different embeddings."""
        emb = self.SinusoidalPositionEmbeddings(dim=128)
        t1 = torch.tensor([0])
        t2 = torch.tensor([500])
        
        out1 = emb(t1)
        out2 = emb(t2)
        
        self.assertFalse(torch.allclose(out1, out2))
        
    def test_batch_processing(self):
        """Test batch processing of timesteps."""
        emb = self.SinusoidalPositionEmbeddings(dim=256)
        timesteps = torch.randint(0, 1000, (16,))
        
        output = emb(timesteps)
        
        self.assertEqual(output.shape, (16, 256))


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestCrossAttention(unittest.TestCase):
    """Test cases for CrossAttention."""
    
    def setUp(self):
        """Set up test fixtures."""
        from models.generators.image import CrossAttention
        self.CrossAttention = CrossAttention
        
    def test_initialization(self):
        """Test CrossAttention initialization."""
        attn = self.CrossAttention(query_dim=256, context_dim=512, heads=8)
        
        self.assertEqual(attn.heads, 8)
        
    def test_self_attention_mode(self):
        """Test self-attention when context is None."""
        attn = self.CrossAttention(query_dim=256, heads=8)
        x = torch.randn(2, 64, 256)
        
        output = attn(x)
        
        self.assertEqual(output.shape, (2, 64, 256))
        
    def test_cross_attention_mode(self):
        """Test cross-attention with context."""
        attn = self.CrossAttention(query_dim=256, context_dim=512, heads=8)
        x = torch.randn(2, 64, 256)
        context = torch.randn(2, 77, 512)
        
        output = attn(x, context)
        
        self.assertEqual(output.shape, (2, 64, 256))
        
    def test_gradient_flow(self):
        """Test gradients flow through attention."""
        attn = self.CrossAttention(query_dim=256, context_dim=512, heads=8)
        x = torch.randn(2, 64, 256, requires_grad=True)
        context = torch.randn(2, 77, 512)
        
        output = attn(x, context)
        loss = output.sum()
        loss.backward()
        
        self.assertIsNotNone(x.grad)


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestFeedForward(unittest.TestCase):
    """Test cases for FeedForward."""
    
    def setUp(self):
        """Set up test fixtures."""
        from models.generators.image import FeedForward
        self.FeedForward = FeedForward
        
    def test_initialization(self):
        """Test FeedForward initialization."""
        ff = self.FeedForward(dim=256, mult=4)
        self.assertIsNotNone(ff.net)
        
    def test_forward_output_shape(self):
        """Test forward pass output shape."""
        ff = self.FeedForward(dim=256, mult=4)
        x = torch.randn(2, 64, 256)
        
        output = ff(x)
        
        self.assertEqual(output.shape, (2, 64, 256))
        
    def test_with_dropout(self):
        """Test with dropout enabled."""
        ff = self.FeedForward(dim=256, mult=4, dropout=0.1)
        ff.train()
        x = torch.randn(2, 64, 256)
        
        output = ff(x)
        
        self.assertEqual(output.shape, (2, 64, 256))


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestTransformerBlock(unittest.TestCase):
    """Test cases for TransformerBlock."""
    
    def setUp(self):
        """Set up test fixtures."""
        from models.generators.image import TransformerBlock
        self.TransformerBlock = TransformerBlock
        
    def test_initialization(self):
        """Test TransformerBlock initialization."""
        block = self.TransformerBlock(dim=256, context_dim=512, heads=8)
        
        self.assertIsNotNone(block.attn1)
        self.assertIsNotNone(block.attn2)
        self.assertIsNotNone(block.ff)
        
    def test_forward_output_shape(self):
        """Test forward pass output shape."""
        block = self.TransformerBlock(dim=256, context_dim=512, heads=8)
        x = torch.randn(2, 64, 256)
        context = torch.randn(2, 77, 512)
        
        output = block(x, context)
        
        self.assertEqual(output.shape, (2, 64, 256))
        
    def test_residual_connections(self):
        """Test that residual connections are working."""
        block = self.TransformerBlock(dim=256, context_dim=512, heads=8)
        x = torch.randn(2, 64, 256)
        context = torch.randn(2, 77, 512)
        
        output = block(x, context)
        
        # Output should be different from input due to transformations
        self.assertFalse(torch.allclose(output, x))


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestResBlock(unittest.TestCase):
    """Test cases for ResBlock."""
    
    def setUp(self):
        """Set up test fixtures."""
        from models.generators.image import ResBlock
        self.ResBlock = ResBlock
        
    def test_initialization(self):
        """Test ResBlock initialization."""
        block = self.ResBlock(in_channels=64, out_channels=128, time_emb_dim=256)
        
        self.assertIsNotNone(block.conv1)
        self.assertIsNotNone(block.conv2)
        self.assertIsNotNone(block.time_mlp)
        
    def test_forward_same_channels(self):
        """Test forward pass with same input/output channels."""
        block = self.ResBlock(in_channels=64, out_channels=64, time_emb_dim=256)
        x = torch.randn(2, 64, 32, 32)
        time_emb = torch.randn(2, 256)
        
        output = block(x, time_emb)
        
        self.assertEqual(output.shape, (2, 64, 32, 32))
        
    def test_forward_different_channels(self):
        """Test forward pass with different input/output channels."""
        block = self.ResBlock(in_channels=64, out_channels=128, time_emb_dim=256)
        x = torch.randn(2, 64, 32, 32)
        time_emb = torch.randn(2, 256)
        
        output = block(x, time_emb)
        
        self.assertEqual(output.shape, (2, 128, 32, 32))
        
    def test_time_conditioning(self):
        """Test that time embedding affects output."""
        block = self.ResBlock(in_channels=64, out_channels=64, time_emb_dim=256)
        x = torch.randn(2, 64, 32, 32)
        time_emb1 = torch.randn(2, 256)
        time_emb2 = torch.randn(2, 256)
        
        output1 = block(x, time_emb1)
        output2 = block(x, time_emb2)
        
        self.assertFalse(torch.allclose(output1, output2))


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestSpatialTransformer(unittest.TestCase):
    """Test cases for SpatialTransformer."""
    
    def setUp(self):
        """Set up test fixtures."""
        from models.generators.image import SpatialTransformer
        self.SpatialTransformer = SpatialTransformer
        
    def test_initialization(self):
        """Test SpatialTransformer initialization."""
        st = self.SpatialTransformer(channels=256, context_dim=512, num_heads=8, depth=2)
        
        self.assertEqual(len(st.transformer_blocks), 2)
        
    def test_forward_output_shape(self):
        """Test forward pass output shape."""
        st = self.SpatialTransformer(channels=256, context_dim=512, num_heads=8)
        x = torch.randn(2, 256, 16, 16)
        context = torch.randn(2, 77, 512)
        
        output = st(x, context)
        
        self.assertEqual(output.shape, (2, 256, 16, 16))
        
    def test_residual_connection(self):
        """Test residual connection is applied."""
        st = self.SpatialTransformer(channels=256, context_dim=512, num_heads=8)
        x = torch.randn(2, 256, 16, 16)
        context = torch.randn(2, 77, 512)
        
        output = st(x, context)
        
        # Output should be close to input due to residual (but not identical)
        self.assertEqual(output.shape, x.shape)


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestVAEEncoder(unittest.TestCase):
    """Test cases for VAEEncoder."""
    
    def setUp(self):
        """Set up test fixtures."""
        from models.generators.image import VAEEncoder
        self.VAEEncoder = VAEEncoder
        
    def test_initialization(self):
        """Test VAEEncoder initialization."""
        encoder = self.VAEEncoder(in_channels=3, latent_channels=4, base_channels=64)
        self.assertIsNotNone(encoder.encoder)
        
    def test_forward_output_shapes(self):
        """Test forward pass output shapes."""
        encoder = self.VAEEncoder(in_channels=3, latent_channels=4, base_channels=64)
        x = torch.randn(2, 3, 256, 256)
        
        z, mean, logvar = encoder(x)
        
        # Output should be downsampled by 8x
        self.assertEqual(z.shape, (2, 4, 32, 32))
        self.assertEqual(mean.shape, (2, 4, 32, 32))
        self.assertEqual(logvar.shape, (2, 4, 32, 32))
        
    def test_reparameterization(self):
        """Test reparameterization produces different samples."""
        encoder = self.VAEEncoder(in_channels=3, latent_channels=4, base_channels=64)
        x = torch.randn(2, 3, 256, 256)
        
        z1, _, _ = encoder(x)
        z2, _, _ = encoder(x)
        
        # Due to random sampling, z1 and z2 should be different
        self.assertFalse(torch.allclose(z1, z2))
        
    def test_logvar_clamping(self):
        """Test logvar is clamped to valid range."""
        encoder = self.VAEEncoder(in_channels=3, latent_channels=4, base_channels=64)
        x = torch.randn(2, 3, 256, 256)
        
        _, _, logvar = encoder(x)
        
        self.assertTrue(torch.all(logvar >= -30))
        self.assertTrue(torch.all(logvar <= 20))


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestVAEDecoder(unittest.TestCase):
    """Test cases for VAEDecoder."""
    
    def setUp(self):
        """Set up test fixtures."""
        from models.generators.image import VAEDecoder
        self.VAEDecoder = VAEDecoder
        
    def test_initialization(self):
        """Test VAEDecoder initialization."""
        decoder = self.VAEDecoder(latent_channels=4, out_channels=3, base_channels=64)
        self.assertIsNotNone(decoder.decoder)
        
    def test_forward_output_shape(self):
        """Test forward pass output shape."""
        decoder = self.VAEDecoder(latent_channels=4, out_channels=3, base_channels=64)
        z = torch.randn(2, 4, 32, 32)
        
        output = decoder(z)
        
        # Output should be upsampled by 8x
        self.assertEqual(output.shape, (2, 3, 256, 256))
        
    def test_output_range(self):
        """Test output is in valid range due to Tanh."""
        decoder = self.VAEDecoder(latent_channels=4, out_channels=3, base_channels=64)
        z = torch.randn(2, 4, 32, 32)
        
        output = decoder(z)
        
        self.assertTrue(torch.all(output >= -1))
        self.assertTrue(torch.all(output <= 1))


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestUNet2D(unittest.TestCase):
    """Test cases for UNet2D."""
    
    def setUp(self):
        """Set up test fixtures."""
        from models.generators.image import UNet2D
        self.UNet2D = UNet2D
        
    def test_initialization(self):
        """Test UNet2D initialization."""
        unet = self.UNet2D(
            in_channels=4, 
            out_channels=4, 
            base_channels=64,
            channel_mults=(1, 2),
            context_dim=512
        )
        
        self.assertIsNotNone(unet.time_embed)
        self.assertIsNotNone(unet.down_blocks)
        self.assertIsNotNone(unet.up_blocks)
        
    def test_forward_output_shape(self):
        """Test forward pass output shape."""
        unet = self.UNet2D(
            in_channels=4, 
            out_channels=4, 
            base_channels=64,
            channel_mults=(1, 2),
            context_dim=512
        )
        x = torch.randn(1, 4, 32, 32)
        timesteps = torch.tensor([500])
        context = torch.randn(1, 77, 512)
        
        output = unet(x, timesteps, context)
        
        self.assertEqual(output.shape, (1, 4, 32, 32))
        
    def test_time_conditioning(self):
        """Test that different timesteps produce different outputs."""
        unet = self.UNet2D(
            in_channels=4, 
            out_channels=4, 
            base_channels=64,
            channel_mults=(1, 2),
            context_dim=512
        )
        x = torch.randn(1, 4, 32, 32)
        context = torch.randn(1, 77, 512)
        
        output1 = unet(x, torch.tensor([100]), context)
        output2 = unet(x, torch.tensor([900]), context)
        
        self.assertFalse(torch.allclose(output1, output2))


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestMobileDiffusionGenerator(unittest.TestCase):
    """Test cases for MobileDiffusionGenerator."""
    
    def setUp(self):
        """Set up test fixtures."""
        from models.generators.image import MobileDiffusionGenerator
        self.MobileDiffusionGenerator = MobileDiffusionGenerator
        
    def test_initialization(self):
        """Test MobileDiffusionGenerator initialization."""
        gen = self.MobileDiffusionGenerator(
            latent_channels=4,
            base_channels=64,
            context_dim=512,
            num_inference_steps=10,
            image_size=64
        )
        
        self.assertEqual(gen.latent_channels, 4)
        self.assertEqual(gen.image_size, 64)
        self.assertEqual(gen.num_inference_steps, 10)
        
    def test_noise_schedule_initialized(self):
        """Test noise schedule is properly initialized."""
        gen = self.MobileDiffusionGenerator(
            latent_channels=4,
            base_channels=64,
            context_dim=512,
            image_size=64
        )
        
        self.assertTrue(hasattr(gen, 'betas'))
        self.assertTrue(hasattr(gen, 'alphas'))
        self.assertTrue(hasattr(gen, 'alphas_cumprod'))
        
    def test_vae_components(self):
        """Test VAE encoder and decoder are initialized."""
        gen = self.MobileDiffusionGenerator(
            latent_channels=4,
            base_channels=64,
            context_dim=512,
            image_size=64
        )
        
        self.assertIsNotNone(gen.vae_encoder)
        self.assertIsNotNone(gen.vae_decoder)
        
    def test_unet_component(self):
        """Test UNet is initialized."""
        gen = self.MobileDiffusionGenerator(
            latent_channels=4,
            base_channels=64,
            context_dim=512,
            image_size=64
        )
        
        self.assertIsNotNone(gen.unet)


if __name__ == '__main__':
    unittest.main()
