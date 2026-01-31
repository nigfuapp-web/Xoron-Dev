#!/usr/bin/env python3
"""
Generate a high-quality SVG architecture diagram for Xoron-Dev.

This script creates a professional, sharp, and visually appealing
architecture diagram that accurately represents the model structure.
"""

import os
import math

# SVG dimensions - 4K quality
WIDTH = 1920
HEIGHT = 1200

# Color palette - Modern dark theme
COLORS = {
    'bg': '#0a0a0f',
    'bg_secondary': '#12121a',
    'text_primary': '#ffffff',
    'text_secondary': '#a0a0b0',
    'text_muted': '#606070',
    'accent_blue': '#3b82f6',
    'accent_purple': '#8b5cf6',
    'accent_green': '#10b981',
    'accent_orange': '#f59e0b',
    'accent_pink': '#ec4899',
    'accent_cyan': '#06b6d4',
    'accent_red': '#ef4444',
    'border': '#2a2a3a',
    'glow_blue': '#3b82f6',
    'glow_purple': '#8b5cf6',
}


def create_gradient(id_name: str, color1: str, color2: str, angle: int = 135) -> str:
    """Create a linear gradient definition."""
    rad = math.radians(angle)
    x1 = 50 - 50 * math.cos(rad)
    y1 = 50 - 50 * math.sin(rad)
    x2 = 50 + 50 * math.cos(rad)
    y2 = 50 + 50 * math.sin(rad)
    
    return f'''    <linearGradient id="{id_name}" x1="{x1:.1f}%" y1="{y1:.1f}%" x2="{x2:.1f}%" y2="{y2:.1f}%">
      <stop offset="0%" stop-color="{color1}"/>
      <stop offset="100%" stop-color="{color2}"/>
    </linearGradient>'''


def create_glow_filter(id_name: str, color: str, intensity: float = 0.5) -> str:
    """Create a glow filter effect."""
    return f'''    <filter id="{id_name}" x="-50%" y="-50%" width="200%" height="200%">
      <feGaussianBlur in="SourceAlpha" stdDeviation="4" result="blur"/>
      <feFlood flood-color="{color}" flood-opacity="{intensity}"/>
      <feComposite in2="blur" operator="in"/>
      <feMerge>
        <feMergeNode/>
        <feMergeNode in="SourceGraphic"/>
      </feMerge>
    </filter>'''


def create_shadow_filter() -> str:
    """Create a drop shadow filter."""
    return '''    <filter id="shadow" x="-20%" y="-20%" width="140%" height="140%">
      <feDropShadow dx="0" dy="4" stdDeviation="8" flood-color="#000000" flood-opacity="0.5"/>
    </filter>'''


def rounded_rect(x: int, y: int, w: int, h: int, rx: int, fill: str, 
                 stroke: str = None, stroke_width: int = 0, filter_id: str = None,
                 opacity: float = 1.0) -> str:
    """Create a rounded rectangle."""
    attrs = f'x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}"'
    if stroke:
        attrs += f' stroke="{stroke}" stroke-width="{stroke_width}"'
    if filter_id:
        attrs += f' filter="url(#{filter_id})"'
    if opacity < 1.0:
        attrs += f' opacity="{opacity}"'
    return f'    <rect {attrs}/>'


def text_element(x: int, y: int, content: str, size: int = 14, 
                 fill: str = '#ffffff', weight: str = 'normal',
                 anchor: str = 'middle', font: str = 'Inter, -apple-system, sans-serif') -> str:
    """Create a text element."""
    return f'    <text x="{x}" y="{y}" text-anchor="{anchor}" fill="{fill}" font-family="{font}" font-size="{size}" font-weight="{weight}">{content}</text>'


def create_component_box(x: int, y: int, w: int, h: int, 
                         title: str, subtitle: str, detail: str,
                         gradient_id: str, icon: str = '') -> str:
    """Create a styled component box."""
    lines = []
    # Outer glow/border
    lines.append(rounded_rect(x-2, y-2, w+4, h+4, 14, 'none', f'url(#{gradient_id})', 2, opacity=0.3))
    # Main box
    lines.append(rounded_rect(x, y, w, h, 12, COLORS['bg_secondary'], f'url(#{gradient_id})', 1, 'shadow'))
    # Inner highlight
    lines.append(rounded_rect(x+2, y+2, w-4, 30, 10, f'url(#{gradient_id})', opacity=0.15))
    # Text
    title_text = f'{icon} {title}' if icon else title
    lines.append(text_element(x + w//2, y + 24, title_text, 16, COLORS['text_primary'], 'bold'))
    lines.append(text_element(x + w//2, y + 46, subtitle, 13, COLORS['text_secondary']))
    lines.append(text_element(x + w//2, y + 66, detail, 11, COLORS['text_muted']))
    return '\n'.join(lines)


def create_large_box(x: int, y: int, w: int, h: int,
                     title: str, subtitle: str, gradient_id: str) -> str:
    """Create a large container box."""
    lines = []
    lines.append(rounded_rect(x, y, w, h, 16, COLORS['bg_secondary'], f'url(#{gradient_id})', 2, 'shadow'))
    lines.append(text_element(x + w//2, y + 30, title, 20, COLORS['text_primary'], 'bold'))
    lines.append(text_element(x + w//2, y + 52, subtitle, 12, COLORS['text_secondary']))
    return '\n'.join(lines)


def create_arrow(x1: int, y1: int, x2: int, y2: int, color: str = '#404050') -> str:
    """Create an arrow line."""
    return f'    <line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="2" marker-end="url(#arrowhead)"/>'


def create_curved_arrow(x1: int, y1: int, x2: int, y2: int, curve: int = 30, color: str = '#404050') -> str:
    """Create a curved arrow."""
    mx = (x1 + x2) // 2
    my = min(y1, y2) - curve
    return f'    <path d="M{x1},{y1} Q{mx},{my} {x2},{y2}" fill="none" stroke="{color}" stroke-width="2" marker-end="url(#arrowhead)"/>'


def generate_svg() -> str:
    """Generate the complete SVG."""
    
    # Start SVG
    svg_parts = [f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {HEIGHT}" width="{WIDTH}" height="{HEIGHT}">
  <defs>
    <!-- Gradients -->
{create_gradient('grad_blue', '#4f8ff7', '#2563eb')}
{create_gradient('grad_purple', '#a78bfa', '#7c3aed')}
{create_gradient('grad_green', '#34d399', '#059669')}
{create_gradient('grad_orange', '#fbbf24', '#d97706')}
{create_gradient('grad_pink', '#f472b6', '#db2777')}
{create_gradient('grad_cyan', '#22d3ee', '#0891b2')}
{create_gradient('grad_red', '#f87171', '#dc2626')}
    
    <!-- Filters -->
{create_shadow_filter()}
{create_glow_filter('glow_blue', COLORS['glow_blue'], 0.4)}
{create_glow_filter('glow_purple', COLORS['glow_purple'], 0.4)}
    
    <!-- Arrow marker -->
    <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#505060"/>
    </marker>
  </defs>
  
  <!-- Background -->
  <rect width="{WIDTH}" height="{HEIGHT}" fill="{COLORS['bg']}"/>
  
  <!-- Grid pattern for depth -->
  <defs>
    <pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse">
      <path d="M 40 0 L 0 0 0 40" fill="none" stroke="#151520" stroke-width="1"/>
    </pattern>
  </defs>
  <rect width="{WIDTH}" height="{HEIGHT}" fill="url(#grid)" opacity="0.5"/>
''']

    # Title
    svg_parts.append(f'''
  <!-- Title -->
  <text x="{WIDTH//2}" y="50" text-anchor="middle" fill="{COLORS['text_primary']}" font-family="Inter, -apple-system, sans-serif" font-size="36" font-weight="bold">
    <tspan fill="{COLORS['accent_blue']}">XORON</tspan><tspan fill="{COLORS['text_primary']}">-DEV</tspan>
  </text>
  <text x="{WIDTH//2}" y="80" text-anchor="middle" fill="{COLORS['text_secondary']}" font-family="Inter, -apple-system, sans-serif" font-size="16">
    Unified Multimodal AI Architecture
  </text>
''')

    # Section: Input Encoders
    svg_parts.append(f'''
  <!-- Section Label: Encoders -->
  <text x="100" y="130" fill="{COLORS['text_muted']}" font-family="Inter, -apple-system, sans-serif" font-size="12" font-weight="bold" letter-spacing="2">INPUT ENCODERS</text>
  <line x1="100" y1="140" x2="350" y2="140" stroke="{COLORS['border']}" stroke-width="1"/>
''')

    # Encoder boxes
    encoder_y = 160
    encoder_h = 85
    encoder_w = 220
    gap = 30
    
    # Vision Encoder
    svg_parts.append(create_component_box(
        100, encoder_y, encoder_w, encoder_h,
        'Vision Encoder', 'SigLIP-SO400M', '384×384 → 64 tokens',
        'grad_blue', '👁️'
    ))
    
    # Video Encoder
    svg_parts.append(create_component_box(
        100 + encoder_w + gap, encoder_y, encoder_w, encoder_h,
        'Video Encoder', 'Temporal Processing', 'Up to 32 frames',
        'grad_purple', '🎬'
    ))
    
    # Audio Encoder
    svg_parts.append(create_component_box(
        100 + 2*(encoder_w + gap), encoder_y, encoder_w, encoder_h,
        'Audio Encoder', 'Conformer ASR', '16kHz • 80 mel bins',
        'grad_green', '🎤'
    ))
    
    # Text Tokenizer
    svg_parts.append(create_component_box(
        100 + 3*(encoder_w + gap), encoder_y, encoder_w, encoder_h,
        'Text Tokenizer', 'Qwen2.5 Vocabulary', '151,643 tokens',
        'grad_orange', '📝'
    ))
    
    # Special Tokens
    svg_parts.append(create_component_box(
        100 + 4*(encoder_w + gap), encoder_y, encoder_w, encoder_h,
        'Special Tokens', '400+ Structured Tokens', 'CoT • Tools • Code',
        'grad_pink', '🏷️'
    ))

    # Arrows from encoders to projector
    proj_y = encoder_y + encoder_h + 60
    for i in range(3):
        x = 100 + i*(encoder_w + gap) + encoder_w//2
        svg_parts.append(create_arrow(x, encoder_y + encoder_h, x, proj_y - 10))

    # Multimodal Projector
    proj_w = 3 * encoder_w + 2 * gap
    svg_parts.append(f'''
  <!-- Multimodal Projector -->
{create_large_box(100, proj_y, proj_w, 70, '🔗 Multimodal Projectors', 'Perceiver Resampler • Spatial Pooling • C-Abstractor', 'grad_cyan')}
''')

    # Arrows from text/special tokens to cross-attention
    cross_y = proj_y + 70 + 50
    svg_parts.append(create_arrow(100 + 3*(encoder_w + gap) + encoder_w//2, encoder_y + encoder_h, 
                                   100 + 3*(encoder_w + gap) + encoder_w//2, cross_y - 10))
    svg_parts.append(create_arrow(100 + 4*(encoder_w + gap) + encoder_w//2, encoder_y + encoder_h,
                                   100 + 4*(encoder_w + gap) + encoder_w//2, cross_y - 10))
    
    # Arrow from projector to cross-attention
    svg_parts.append(create_arrow(100 + proj_w//2, proj_y + 70, 100 + proj_w//2, cross_y - 10))

    # Cross-Attention Fusion
    cross_w = WIDTH - 200
    svg_parts.append(f'''
  <!-- Cross-Attention Fusion -->
{create_large_box(100, cross_y, cross_w, 70, '🔀 Cross-Attention Fusion', '4 Layers • 8 Heads • Flash Attention Enabled', 'grad_red')}
''')

    # Arrow to MoE LLM
    moe_y = cross_y + 70 + 40
    svg_parts.append(create_arrow(WIDTH//2, cross_y + 70, WIDTH//2, moe_y - 10))

    # MoE LLM Backbone - Large container
    moe_h = 280
    svg_parts.append(f'''
  <!-- MoE LLM Backbone Container -->
{rounded_rect(100, moe_y, cross_w, moe_h, 20, COLORS['bg_secondary'], COLORS['accent_purple'], 2, 'shadow')}
{text_element(WIDTH//2, moe_y + 35, '🧠 MoE LLM Backbone', 24, COLORS['text_primary'], 'bold')}
{text_element(WIDTH//2, moe_y + 60, '12 Transformer Layers • 1024 Hidden Dim • 16 Attention Heads • 128K Context Length', 13, COLORS['text_secondary'])}
''')

    # Inner components of MoE
    inner_y = moe_y + 80
    inner_h = 180
    inner_w = 380
    inner_gap = 40
    
    # Transformer Layer
    svg_parts.append(f'''
  <!-- Transformer Layer -->
{rounded_rect(130, inner_y, inner_w, inner_h, 12, '#1a1a25', COLORS['accent_blue'], 1)}
{text_element(130 + inner_w//2, inner_y + 25, 'Transformer Layer', 16, COLORS['text_primary'], 'bold')}
  
  <!-- Self-Attention -->
{rounded_rect(145, inner_y + 40, 170, 55, 8, '#0f0f15')}
{text_element(230, inner_y + 62, 'Self-Attention', 13, COLORS['text_secondary'])}
{text_element(230, inner_y + 80, 'Sliding Window: 4096', 10, COLORS['text_muted'])}
  
  <!-- FFN -->
{rounded_rect(325, inner_y + 40, 170, 55, 8, '#0f0f15')}
{text_element(410, inner_y + 62, 'Feed-Forward', 13, COLORS['text_secondary'])}
{text_element(410, inner_y + 80, 'MoE every 2nd layer', 10, COLORS['text_muted'])}
  
  <!-- RMSNorm -->
{rounded_rect(145, inner_y + 105, 350, 35, 8, '#0f0f15')}
{text_element(320, inner_y + 128, 'RMSNorm + Residual Connections', 12, COLORS['text_secondary'])}
  
  <!-- Flash Attention Badge -->
{rounded_rect(145, inner_y + 150, 130, 22, 6, COLORS['accent_cyan'])}
{text_element(210, inner_y + 165, '⚡ Flash Attention', 10, COLORS['text_primary'], 'bold')}
''')

    # MoE Layer
    moe_inner_x = 130 + inner_w + inner_gap
    svg_parts.append(f'''
  <!-- MoE Layer -->
{rounded_rect(moe_inner_x, inner_y, inner_w, inner_h, 12, '#1a1a25', COLORS['accent_purple'], 1)}
{text_element(moe_inner_x + inner_w//2, inner_y + 25, '🎯 Mixture of Experts', 16, COLORS['text_primary'], 'bold')}
  
  <!-- Router -->
{rounded_rect(moe_inner_x + 15, inner_y + 40, 100, 85, 8, '#0f0f15')}
{text_element(moe_inner_x + 65, inner_y + 62, 'Router', 13, COLORS['text_secondary'])}
{text_element(moe_inner_x + 65, inner_y + 80, 'Top-2', 11, COLORS['accent_orange'])}
{text_element(moe_inner_x + 65, inner_y + 96, 'Routing', 11, COLORS['text_muted'])}
{text_element(moe_inner_x + 65, inner_y + 112, 'Aux Loss', 10, COLORS['accent_green'])}
  
  <!-- Expert Grid -->
''')
    
    # Draw 8 experts in a 4x2 grid - smaller to fit above shared expert
    expert_size = 45
    expert_gap = 6
    expert_start_x = moe_inner_x + 130
    expert_start_y = inner_y + 40
    for row in range(2):
        for col in range(4):
            ex = expert_start_x + col * (expert_size + expert_gap)
            ey = expert_start_y + row * (expert_size + expert_gap)
            svg_parts.append(rounded_rect(ex, ey, expert_size, expert_size, 6, COLORS['accent_blue']))
            svg_parts.append(text_element(ex + expert_size//2, ey + expert_size//2 + 5, f'E{row*4 + col + 1}', 12, COLORS['text_primary'], 'bold'))
    
    # Shared Expert - positioned below the expert grid with proper spacing
    shared_expert_y = expert_start_y + 2 * (expert_size + expert_gap) + 5
    shared_expert_width = 4 * (expert_size + expert_gap) - expert_gap
    svg_parts.append(f'''
  <!-- Shared Expert -->
{rounded_rect(expert_start_x, shared_expert_y, shared_expert_width, 28, 6, COLORS['accent_green'])}
{text_element(expert_start_x + shared_expert_width//2, shared_expert_y + 18, 'Shared Expert (DeepSeek)', 10, COLORS['text_primary'], 'bold')}
''')

    # LoRA Section
    lora_x = moe_inner_x + inner_w + inner_gap
    svg_parts.append(f'''
  <!-- LoRA Section -->
{rounded_rect(lora_x, inner_y, inner_w, inner_h, 12, '#1a1a25', COLORS['accent_green'], 1)}
{text_element(lora_x + inner_w//2, inner_y + 25, '🔧 LoRA+ Fine-tuning', 16, COLORS['text_primary'], 'bold')}
  
  <!-- rsLoRA -->
{rounded_rect(lora_x + 15, inner_y + 40, 170, 100, 8, '#0f0f15')}
{text_element(lora_x + 100, inner_y + 62, 'rsLoRA', 14, COLORS['text_secondary'], 'bold')}
{text_element(lora_x + 100, inner_y + 82, 'r=32, α=64', 11, COLORS['text_muted'])}
{text_element(lora_x + 100, inner_y + 100, 'Rank-Stabilized', 10, COLORS['text_muted'])}
{text_element(lora_x + 100, inner_y + 120, 'B learns 16× faster', 10, COLORS['accent_orange'])}
  
  <!-- Target Modules -->
{rounded_rect(lora_x + 195, inner_y + 40, 170, 100, 8, '#0f0f15')}
{text_element(lora_x + 280, inner_y + 62, 'Target Modules', 14, COLORS['text_secondary'], 'bold')}
{text_element(lora_x + 280, inner_y + 82, 'q, k, v, o_proj', 11, COLORS['text_muted'])}
{text_element(lora_x + 280, inner_y + 100, 'gate, up, down', 11, COLORS['text_muted'])}
{text_element(lora_x + 280, inner_y + 120, '7 modules total', 10, COLORS['accent_green'])}
''')

    # Output Section
    output_y = moe_y + moe_h + 50
    svg_parts.append(f'''
  <!-- Section Label: Outputs -->
  <text x="100" y="{output_y - 15}" fill="{COLORS['text_muted']}" font-family="Inter, -apple-system, sans-serif" font-size="12" font-weight="bold" letter-spacing="2">OUTPUT GENERATORS</text>
  <line x1="100" y1="{output_y - 5}" x2="350" y2="{output_y - 5}" stroke="{COLORS['border']}" stroke-width="1"/>
''')

    # Arrow from MoE to outputs
    svg_parts.append(create_arrow(WIDTH//2, moe_y + moe_h, WIDTH//2, output_y - 25))

    # Output generators
    out_w = 280
    out_h = 90
    out_gap = 40
    out_start_x = (WIDTH - 4*out_w - 3*out_gap) // 2
    
    # Text Generation
    svg_parts.append(create_component_box(
        out_start_x, output_y, out_w, out_h,
        'Text Generation', 'LM Head + Sampling', 'Chain-of-Thought • Tools • Code',
        'grad_orange', '📝'
    ))
    
    # Image Generator
    svg_parts.append(create_component_box(
        out_start_x + out_w + out_gap, output_y, out_w, out_h,
        'Image Generator', 'MobileDiffusion', '256×256 • CFG=7.5 • 20 steps',
        'grad_blue', '🎨'
    ))
    
    # Video Generator
    svg_parts.append(create_component_box(
        out_start_x + 2*(out_w + out_gap), output_y, out_w, out_h,
        'Video Generator', 'Temporal Diffusion', '16 frames • 256×256',
        'grad_purple', '🎬'
    ))
    
    # Audio Decoder
    svg_parts.append(create_component_box(
        out_start_x + 3*(out_w + out_gap), output_y, out_w, out_h,
        'Audio Decoder', 'Neural TTS', '13 emotions • 256 speakers',
        'grad_green', '🔊'
    ))

    # Arrows to outputs
    for i in range(4):
        target_x = out_start_x + i*(out_w + out_gap) + out_w//2
        svg_parts.append(create_curved_arrow(WIDTH//2, moe_y + moe_h + 10, target_x, output_y - 5, 20))

    # Stats box
    stats_x = WIDTH - 220
    stats_y = output_y
    svg_parts.append(f'''
  <!-- Stats Box -->
{rounded_rect(stats_x, stats_y, 180, out_h, 12, COLORS['bg_secondary'], COLORS['border'], 1, 'shadow')}
{text_element(stats_x + 90, stats_y + 22, '📊 Model Stats', 14, COLORS['text_primary'], 'bold')}
{text_element(stats_x + 90, stats_y + 44, '~2B Parameters', 11, COLORS['text_muted'])}
{text_element(stats_x + 90, stats_y + 60, '128K Context', 11, COLORS['text_muted'])}
{text_element(stats_x + 90, stats_y + 76, '6 MoE Layers', 11, COLORS['text_muted'])}
''')

    # Footer
    svg_parts.append(f'''
  <!-- Footer -->
  <text x="{WIDTH//2}" y="{HEIGHT - 30}" text-anchor="middle" fill="{COLORS['text_muted']}" font-family="Inter, -apple-system, sans-serif" font-size="12">
    Xoron-Dev • State-of-the-Art Multimodal AI • github.com/nigfuapp-web/Xoron-Dev
  </text>
''')

    # Close SVG
    svg_parts.append('</svg>')
    
    return '\n'.join(svg_parts)


def main():
    """Generate and save the SVG."""
    svg_content = generate_svg()
    
    # Ensure assets directory exists
    script_dir = os.path.dirname(os.path.abspath(__file__))
    assets_dir = os.path.join(os.path.dirname(script_dir), 'assets')
    os.makedirs(assets_dir, exist_ok=True)
    
    # Save SVG
    output_path = os.path.join(assets_dir, 'xoron_architecture.svg')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(svg_content)
    
    print(f"✅ Generated high-quality SVG: {output_path}")
    print(f"   Dimensions: {WIDTH}x{HEIGHT} (4K quality)")
    

if __name__ == '__main__':
    main()
