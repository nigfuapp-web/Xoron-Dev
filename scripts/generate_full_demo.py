#!/usr/bin/env python3
"""
Generate a comprehensive 60-second animated SVG demo of Xoron-Dev.

Scenes (60 seconds total):
1. (0-10s)  Text Generation - User asks for code, model thinks and generates
2. (10-20s) Text to Image - User requests image, diffusion process shown
3. (20-30s) Image to Text - User uploads image, vision encoder analyzes
4. (30-40s) Text to Audio - TTS with emotion control
5. (40-50s) Tool Calling - Agentic workflow with code execution
6. (50-60s) Video Generation - Text/Image to video frames

Each scene shows:
- Left: Chat interface with user/assistant messages
- Right: Backend processing visualization

Architecture specs (from config/model_config.py):
- LLM: 1024 hidden, 12 layers, 16 heads, 2048 intermediate
- MoE: 8 experts, top-2, every 2nd layer (6 MoE layers total)
- Vision: SigLIP-so400m-patch14-384 + TiTok (256 tokens) + 2D-RoPE
- Video: 3D-RoPE + Temporal MoE (4 experts) + 3D Causal (4 layers)
- Audio: Raw Waveform Tokenizer + Conformer + RMLA + MAS
- Generation: MoE-DiT + Flow Matching + Dual-Stream
- Context: 128K with Ring Attention (4096 chunk)
"""

import os

WIDTH = 1600
HEIGHT = 900

C = {
    'bg': '#0a0a12',
    'bg_card': '#13131f',
    'bg_dark': '#08080d',
    'border': '#2a2a3a',
    'text': '#e8e8f0',
    'text_dim': '#8888a0',
    'text_muted': '#555566',
    'blue': '#4a9eff',
    'purple': '#a855f7',
    'green': '#22c55e',
    'orange': '#f59e0b',
    'pink': '#ec4899',
    'cyan': '#06b6d4',
    'red': '#ef4444',
    'yellow': '#eab308',
    'user': '#2563eb',
    'assistant': '#166534',
}

def e(text):
    """Escape XML special characters."""
    return text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;').replace("'", '&apos;')

def generate_svg():
    # Animation timing (in seconds)
    SCENE_DUR = 10
    TOTAL_DUR = 60
    
    svg = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {HEIGHT}" width="{WIDTH}" height="{HEIGHT}">
  <defs>
    <!-- Gradients -->
    <linearGradient id="headerGrad" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="{C['blue']}"/>
      <stop offset="50%" stop-color="{C['purple']}"/>
      <stop offset="100%" stop-color="{C['pink']}"/>
    </linearGradient>
    
    <linearGradient id="sunsetGrad" x1="0%" y1="100%" x2="0%" y2="0%">
      <stop offset="0%" stop-color="#1e1b4b"/>
      <stop offset="30%" stop-color="#7c3aed"/>
      <stop offset="50%" stop-color="#f97316"/>
      <stop offset="70%" stop-color="#fbbf24"/>
      <stop offset="100%" stop-color="#fef3c7"/>
    </linearGradient>
    
    <linearGradient id="codeGrad" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="{C['bg_dark']}"/>
      <stop offset="100%" stop-color="#0f0f1a"/>
    </linearGradient>
    
    <!-- Filters -->
    <filter id="glow" x="-50%" y="-50%" width="200%" height="200%">
      <feGaussianBlur stdDeviation="4" result="blur"/>
      <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>
    
    <filter id="shadow" x="-20%" y="-20%" width="140%" height="140%">
      <feDropShadow dx="0" dy="4" stdDeviation="8" flood-color="#000" flood-opacity="0.5"/>
    </filter>
    
    <!-- Clip paths -->
    <clipPath id="chatClip"><rect x="20" y="75" width="540" height="720"/></clipPath>
    
    <!-- Arrow marker -->
    <marker id="arrow" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="{C['cyan']}"/>
    </marker>
  </defs>
  
  <!-- Background -->
  <rect width="{WIDTH}" height="{HEIGHT}" fill="{C['bg']}"/>
  
  <!-- Subtle grid -->
  <defs>
    <pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse">
      <path d="M 40 0 L 0 0 0 40" fill="none" stroke="{C['border']}" stroke-width="0.3" opacity="0.3"/>
    </pattern>
  </defs>
  <rect width="{WIDTH}" height="{HEIGHT}" fill="url(#grid)"/>
  
  <!-- Header -->
  <rect x="0" y="0" width="{WIDTH}" height="55" fill="{C['bg_card']}"/>
  <rect x="0" y="53" width="{WIDTH}" height="2" fill="url(#headerGrad)"/>
  <text x="25" y="35" fill="{C['text']}" font-family="SF Pro Display, -apple-system, sans-serif" font-size="22" font-weight="700">
    <tspan fill="{C['blue']}">XORON</tspan><tspan fill="{C['text']}">-DEV</tspan>
  </text>
  <text x="180" y="35" fill="{C['text_dim']}" font-family="SF Pro Display, -apple-system, sans-serif" font-size="14">
    Multimodal AI Demo
  </text>
  
  <!-- Scene indicator -->
  <g id="sceneIndicator" transform="translate({WIDTH - 300}, 20)">
    <rect width="280" height="30" rx="15" fill="{C['bg_dark']}" stroke="{C['border']}" stroke-width="1"/>
    
    <!-- Scene dots -->'''
    
    # Add scene indicator dots
    scenes = ['Code', 'Image', 'Vision', 'Audio', 'Tools', 'Video']
    for i, scene in enumerate(scenes):
        x = 25 + i * 45
        # Each scene lights up during its time
        svg += f'''
    <circle cx="{x}" cy="15" r="6" fill="{C['bg']}" stroke="{C['border']}" stroke-width="1">
      <animate attributeName="fill" values="{C['bg']};{C['cyan']};{C['bg']}" dur="{TOTAL_DUR}s" 
               keyTimes="0;{(i*SCENE_DUR + 1)/TOTAL_DUR:.3f};{((i+1)*SCENE_DUR)/TOTAL_DUR:.3f}" repeatCount="indefinite"/>
    </circle>'''
    
    svg += f'''
  </g>
  
  <!-- Timer -->
  <g transform="translate({WIDTH - 80}, 25)">
    <text fill="{C['text_dim']}" font-family="SF Mono, monospace" font-size="12">
      <tspan id="timer">60s</tspan>
    </text>
  </g>
  
  <!-- ==================== LEFT PANEL: CHAT INTERFACE ==================== -->
  
  <g id="chatPanel">
    <rect x="15" y="65" width="550" height="820" rx="16" fill="{C['bg_card']}" stroke="{C['border']}" stroke-width="1" filter="url(#shadow)"/>
    
    <!-- Chat header -->
    <rect x="15" y="65" width="550" height="50" rx="16" fill="{C['bg_dark']}"/>
    <rect x="15" y="99" width="550" height="16" fill="{C['bg_dark']}"/>
    <circle cx="45" cy="90" r="8" fill="{C['green']}"/>
    <text x="65" y="95" fill="{C['text']}" font-family="SF Pro Display, sans-serif" font-size="15" font-weight="600">Xoron Assistant</text>
    <text x="200" y="95" fill="{C['green']}" font-family="SF Pro Display, sans-serif" font-size="11">● Online</text>
    
    <!-- Chat content area with clipping -->
    <g clip-path="url(#chatClip)">
'''
    
    # ==================== SCENE 1: CODE GENERATION (0-10s) ====================
    svg += f'''
      <!-- SCENE 1: Code Generation (0-10s) -->
      <g id="scene1Chat">
        <animate attributeName="opacity" dur="{TOTAL_DUR}s" repeatCount="indefinite"
                 values="1;1;0;0;0;0;0;0;0;0;0;0" keyTimes="0;0.15;0.167;0.33;0.5;0.67;0.833;1;1;1;1;1"/>
        
        <!-- User message -->
        <g>
          <animate attributeName="opacity" values="0;1" dur="0.5s" begin="0.5s" fill="freeze"/>
          <rect x="180" y="130" width="360" height="55" rx="16" fill="{C['user']}"/>
          <text x="200" y="155" fill="{C['text']}" font-family="SF Pro Text, sans-serif" font-size="14">Write a Python function to calculate</text>
          <text x="200" y="173" fill="{C['text']}" font-family="SF Pro Text, sans-serif" font-size="14">the nth Fibonacci number efficiently</text>
        </g>
        
        <!-- Typing indicator - centered dots in box -->
        <g>
          <animate attributeName="opacity" values="0;0;1;1;0" dur="10s" repeatCount="indefinite" keyTimes="0;0.15;0.2;0.35;0.4"/>
          <rect x="35" y="200" width="70" height="35" rx="12" fill="{C['bg_dark']}" stroke="{C['border']}" stroke-width="1"/>
          <circle cx="55" cy="217.5" r="4" fill="{C['text_dim']}"><animate attributeName="opacity" values="0.3;1;0.3" dur="0.6s" repeatCount="indefinite"/></circle>
          <circle cx="70" cy="217.5" r="4" fill="{C['text_dim']}"><animate attributeName="opacity" values="0.3;1;0.3" dur="0.6s" begin="0.2s" repeatCount="indefinite"/></circle>
          <circle cx="85" cy="217.5" r="4" fill="{C['text_dim']}"><animate attributeName="opacity" values="0.3;1;0.3" dur="0.6s" begin="0.4s" repeatCount="indefinite"/></circle>
        </g>
        
        <!-- Assistant response -->
        <g>
          <animate attributeName="opacity" values="0;0;0;0;1" dur="10s" repeatCount="indefinite" keyTimes="0;0.3;0.35;0.4;0.45"/>
          <rect x="35" y="200" width="400" height="220" rx="16" fill="{C['assistant']}" opacity="0.15" stroke="{C['green']}" stroke-width="1"/>
          <text x="50" y="225" fill="{C['text']}" font-family="SF Pro Text, sans-serif" font-size="13">Here's an efficient fibonacci function using memoization:</text>
          
          <!-- Code block -->
          <rect x="45" y="240" width="380" height="160" rx="10" fill="url(#codeGrad)" stroke="{C['border']}" stroke-width="1"/>
          <text x="55" y="265" fill="{C['purple']}" font-family="SF Mono, monospace" font-size="12">def</text>
          <text x="85" y="265" fill="{C['cyan']}" font-family="SF Mono, monospace" font-size="12">fibonacci</text>
          <text x="160" y="265" fill="{C['text']}" font-family="SF Mono, monospace" font-size="12">(n, memo={{}}):</text>
          <text x="65" y="285" fill="{C['purple']}" font-family="SF Mono, monospace" font-size="12">if</text>
          <text x="85" y="285" fill="{C['text']}" font-family="SF Mono, monospace" font-size="12">n in memo:</text>
          <text x="75" y="305" fill="{C['purple']}" font-family="SF Mono, monospace" font-size="12">return</text>
          <text x="125" y="305" fill="{C['text']}" font-family="SF Mono, monospace" font-size="12">memo[n]</text>
          <text x="65" y="325" fill="{C['purple']}" font-family="SF Mono, monospace" font-size="12">if</text>
          <text x="85" y="325" fill="{C['text']}" font-family="SF Mono, monospace" font-size="12">n &lt;= 1:</text>
          <text x="75" y="345" fill="{C['purple']}" font-family="SF Mono, monospace" font-size="12">return</text>
          <text x="125" y="345" fill="{C['orange']}" font-family="SF Mono, monospace" font-size="12">n</text>
          <text x="65" y="365" fill="{C['text']}" font-family="SF Mono, monospace" font-size="12">memo[n] = fibonacci(n-1) + fibonacci(n-2)</text>
          <text x="65" y="385" fill="{C['purple']}" font-family="SF Mono, monospace" font-size="12">return</text>
          <text x="115" y="385" fill="{C['text']}" font-family="SF Mono, monospace" font-size="12">memo[n]</text>
        </g>
      </g>
'''
    
    # ==================== SCENE 2: TEXT TO IMAGE (10-20s) ====================
    svg += f'''
      <!-- SCENE 2: Text to Image (10-20s) -->
      <g id="scene2Chat">
        <animate attributeName="opacity" dur="{TOTAL_DUR}s" repeatCount="indefinite"
                 values="0;0;1;1;0;0;0;0;0;0;0;0" keyTimes="0;0.15;0.167;0.32;0.333;0.5;0.67;0.833;1;1;1;1"/>
        
        <!-- User message -->
        <g>
          <rect x="120" y="130" width="420" height="40" rx="16" fill="{C['user']}"/>
          <text x="140" y="155" fill="{C['text']}" font-family="SF Pro Text, sans-serif" font-size="14">Generate an image of a magical sunset over mountains</text>
        </g>
        
        <!-- Assistant response with generated image -->
        <g>
          <animate attributeName="opacity" values="0;0;1" dur="10s" repeatCount="indefinite" keyTimes="0;0.4;0.5"/>
          <rect x="35" y="190" width="320" height="280" rx="16" fill="{C['assistant']}" opacity="0.15" stroke="{C['green']}" stroke-width="1"/>
          <text x="50" y="215" fill="{C['text']}" font-family="SF Pro Text, sans-serif" font-size="13">Here's your generated image:</text>
          
          <!-- Generated image -->
          <rect x="50" y="230" width="280" height="180" rx="12" fill="url(#sunsetGrad)"/>
          <!-- Mountain silhouettes -->
          <polygon points="50,410 120,340 180,380 240,320 330,410" fill="#1e1b4b" opacity="0.9"/>
          <polygon points="50,410 100,360 150,390 200,350 250,380 330,410" fill="#312e81" opacity="0.7"/>
          
          <!-- Sun -->
          <circle cx="190" cy="300" r="30" fill="#fbbf24" filter="url(#glow)">
            <animate attributeName="cy" values="310;300;310" dur="3s" repeatCount="indefinite"/>
          </circle>
          
          <text x="50" y="435" fill="{C['green']}" font-family="SF Pro Text, sans-serif" font-size="11">✓ Generated with MobileDiffusion (384×384)</text>
          <text x="50" y="455" fill="{C['text_dim']}" font-family="SF Pro Text, sans-serif" font-size="10">20 diffusion steps • CFG 7.5</text>
        </g>
      </g>
'''
    
    # ==================== SCENE 3: IMAGE TO TEXT (20-30s) ====================
    svg += f'''
      <!-- SCENE 3: Image to Text (20-30s) -->
      <g id="scene3Chat">
        <animate attributeName="opacity" dur="{TOTAL_DUR}s" repeatCount="indefinite"
                 values="0;0;0;0;1;1;0;0;0;0;0;0" keyTimes="0;0.167;0.32;0.333;0.35;0.49;0.5;0.67;0.833;1;1;1"/>
        
        <!-- User message with image -->
        <g>
          <rect x="150" y="130" width="390" height="120" rx="16" fill="{C['user']}"/>
          <text x="170" y="155" fill="{C['text']}" font-family="SF Pro Text, sans-serif" font-size="14">What can you see in this image?</text>
          <rect x="170" y="170" width="100" height="65" rx="8" fill="{C['bg_dark']}"/>
          <text x="220" y="210" text-anchor="middle" font-size="28">🏔️</text>
        </g>
        
        <!-- Assistant analysis -->
        <g>
          <animate attributeName="opacity" values="0;0;1" dur="10s" repeatCount="indefinite" keyTimes="0;0.4;0.5"/>
          <rect x="35" y="270" width="450" height="180" rx="16" fill="{C['assistant']}" opacity="0.15" stroke="{C['green']}" stroke-width="1"/>
          <text x="50" y="295" fill="{C['cyan']}" font-family="SF Pro Text, sans-serif" font-size="12" font-weight="600">🔍 Image Analysis</text>
          
          <text x="50" y="325" fill="{C['text']}" font-family="SF Pro Text, sans-serif" font-size="13">I can see a stunning mountain landscape featuring:</text>
          <text x="60" y="350" fill="{C['text']}" font-family="SF Pro Text, sans-serif" font-size="12">• Snow-capped mountain peaks</text>
          <text x="60" y="370" fill="{C['text']}" font-family="SF Pro Text, sans-serif" font-size="12">• A clear blue sky with scattered clouds</text>
          <text x="60" y="390" fill="{C['text']}" font-family="SF Pro Text, sans-serif" font-size="12">• Pine forest in the foreground</text>
          <text x="60" y="410" fill="{C['text']}" font-family="SF Pro Text, sans-serif" font-size="12">• A serene alpine lake reflecting the mountains</text>
          
          <text x="50" y="440" fill="{C['green']}" font-family="SF Pro Text, sans-serif" font-size="10">Confidence: 96% • SigLIP Vision Encoder</text>
        </g>
      </g>
'''
    
    # ==================== SCENE 4: TEXT TO AUDIO (30-40s) ====================
    svg += f'''
      <!-- SCENE 4: Text to Audio (30-40s) -->
      <g id="scene4Chat">
        <animate attributeName="opacity" dur="{TOTAL_DUR}s" repeatCount="indefinite"
                 values="0;0;0;0;0;0;1;1;0;0;0;0" keyTimes="0;0.167;0.333;0.49;0.5;0.52;0.54;0.65;0.667;0.833;1;1"/>
        
        <!-- User message -->
        <g>
          <rect x="100" y="130" width="440" height="55" rx="16" fill="{C['user']}"/>
          <text x="120" y="152" fill="{C['text']}" font-family="SF Pro Text, sans-serif" font-size="14">Read this text aloud with a cheerful tone:</text>
          <text x="120" y="172" fill="{C['text']}" font-family="SF Pro Text, sans-serif" font-size="14">"Welcome to Xoron, your AI assistant!"</text>
        </g>
        
        <!-- Assistant audio response -->
        <g>
          <animate attributeName="opacity" values="0;0;1" dur="10s" repeatCount="indefinite" keyTimes="0;0.4;0.5"/>
          <rect x="35" y="205" width="380" height="200" rx="16" fill="{C['assistant']}" opacity="0.15" stroke="{C['green']}" stroke-width="1"/>
          <text x="50" y="230" fill="{C['text']}" font-family="SF Pro Text, sans-serif" font-size="13">🔊 Generated Audio:</text>
          
          <!-- Audio player -->
          <rect x="50" y="250" width="340" height="60" rx="10" fill="{C['bg_dark']}" stroke="{C['border']}" stroke-width="1"/>
          <circle cx="85" cy="280" r="18" fill="{C['green']}"/>
          <polygon points="80,272 80,288 93,280" fill="{C['text']}"/>
          
          <!-- Waveform -->
          <g transform="translate(115, 260)">'''
    
    # Add animated waveform bars
    for i in range(25):
        h = 8 + (i % 5) * 6 + ((i * 3) % 7) * 2
        svg += f'''
            <rect x="{i * 9}" y="{20 - h//2}" width="5" height="{h}" rx="2" fill="{C['cyan']}">
              <animate attributeName="height" values="{h};{h+8};{h}" dur="0.4s" repeatCount="indefinite" begin="{i*0.03}s"/>
              <animate attributeName="y" values="{20-h//2};{16-h//2};{20-h//2}" dur="0.4s" repeatCount="indefinite" begin="{i*0.03}s"/>
            </rect>'''
    
    svg += f'''
          </g>
          
          <text x="50" y="335" fill="{C['text_dim']}" font-family="SF Pro Text, sans-serif" font-size="11">Duration: 3.2s • 16kHz</text>
          
          <!-- Emotion tag -->
          <rect x="50" y="355" width="120" height="28" rx="8" fill="{C['orange']}" opacity="0.2" stroke="{C['orange']}" stroke-width="1"/>
          <text x="110" y="374" text-anchor="middle" fill="{C['orange']}" font-family="SF Pro Text, sans-serif" font-size="11">🎭 Cheerful</text>
          
          <text x="180" y="374" fill="{C['text_dim']}" font-family="SF Pro Text, sans-serif" font-size="10">Neural TTS with emotion control</text>
        </g>
      </g>
'''
    
    # ==================== SCENE 5: TOOL CALLING (40-50s) ====================
    svg += f'''
      <!-- SCENE 5: Tool Calling (40-50s) -->
      <g id="scene5Chat">
        <animate attributeName="opacity" dur="{TOTAL_DUR}s" repeatCount="indefinite"
                 values="0;0;0;0;0;0;0;0;1;1;0;0" keyTimes="0;0.167;0.333;0.5;0.65;0.667;0.69;0.71;0.72;0.82;0.833;1"/>
        
        <!-- User message -->
        <g>
          <rect x="80" y="130" width="460" height="55" rx="16" fill="{C['user']}"/>
          <text x="100" y="152" fill="{C['text']}" font-family="SF Pro Text, sans-serif" font-size="14">Search the web for the latest Python 3.12 features</text>
          <text x="100" y="172" fill="{C['text']}" font-family="SF Pro Text, sans-serif" font-size="14">and summarize them for me</text>
        </g>
        
        <!-- Tool execution -->
        <g>
          <animate attributeName="opacity" values="0;0;1" dur="10s" repeatCount="indefinite" keyTimes="0;0.3;0.4"/>
          <rect x="35" y="205" width="480" height="280" rx="16" fill="{C['assistant']}" opacity="0.15" stroke="{C['green']}" stroke-width="1"/>
          
          <!-- Tool call indicator -->
          <rect x="50" y="220" width="140" height="28" rx="8" fill="{C['purple']}" opacity="0.2" stroke="{C['purple']}" stroke-width="1"/>
          <text x="120" y="239" text-anchor="middle" fill="{C['purple']}" font-family="SF Pro Text, sans-serif" font-size="11">🔧 Using Tool</text>
          
          <!-- Tool call code -->
          <rect x="50" y="260" width="440" height="80" rx="8" fill="{C['bg_dark']}" stroke="{C['border']}" stroke-width="1"/>
          <text x="60" y="280" fill="{C['cyan']}" font-family="SF Mono, monospace" font-size="10">&lt;|tool_call|&gt;</text>
          <text x="70" y="298" fill="{C['text_dim']}" font-family="SF Mono, monospace" font-size="10">web_search("Python 3.12 new features")</text>
          <text x="60" y="316" fill="{C['cyan']}" font-family="SF Mono, monospace" font-size="10">&lt;|/tool_call|&gt;</text>
          <text x="60" y="334" fill="{C['green']}" font-family="SF Mono, monospace" font-size="10">&lt;|tool_result|&gt; Found 5 results...</text>
          
          <!-- Summary -->
          <text x="50" y="365" fill="{C['text']}" font-family="SF Pro Text, sans-serif" font-size="13" font-weight="600">Python 3.12 Key Features:</text>
          <text x="60" y="390" fill="{C['text']}" font-family="SF Pro Text, sans-serif" font-size="12">• Improved error messages with suggestions</text>
          <text x="60" y="410" fill="{C['text']}" font-family="SF Pro Text, sans-serif" font-size="12">• Per-interpreter GIL (PEP 684)</text>
          <text x="60" y="430" fill="{C['text']}" font-family="SF Pro Text, sans-serif" font-size="12">• Type parameter syntax (PEP 695)</text>
          <text x="60" y="450" fill="{C['text']}" font-family="SF Pro Text, sans-serif" font-size="12">• F-string improvements</text>
          
          <text x="50" y="475" fill="{C['green']}" font-family="SF Pro Text, sans-serif" font-size="10">✓ Search completed • 3 sources cited</text>
        </g>
      </g>
'''
    
    # ==================== SCENE 6: VIDEO GENERATION (50-60s) ====================
    svg += f'''
      <!-- SCENE 6: Video Generation (50-60s) -->
      <g id="scene6Chat">
        <animate attributeName="opacity" dur="{TOTAL_DUR}s" repeatCount="indefinite"
                 values="0;0;0;0;0;0;0;0;0;0;1;1" keyTimes="0;0.167;0.333;0.5;0.667;0.82;0.833;0.85;0.87;0.89;0.9;1"/>
        
        <!-- User message -->
        <g>
          <rect x="100" y="130" width="440" height="55" rx="16" fill="{C['user']}"/>
          <text x="120" y="152" fill="{C['text']}" font-family="SF Pro Text, sans-serif" font-size="14">Create a short video of a butterfly landing</text>
          <text x="120" y="172" fill="{C['text']}" font-family="SF Pro Text, sans-serif" font-size="14">on a flower in slow motion</text>
        </g>
        
        <!-- Video generation response -->
        <g>
          <animate attributeName="opacity" values="0;0;1" dur="10s" repeatCount="indefinite" keyTimes="0;0.4;0.5"/>
          <rect x="35" y="205" width="420" height="300" rx="16" fill="{C['assistant']}" opacity="0.15" stroke="{C['green']}" stroke-width="1"/>
          <text x="50" y="230" fill="{C['text']}" font-family="SF Pro Text, sans-serif" font-size="13">🎬 Generated Video:</text>
          
          <!-- Video frames preview -->
          <g transform="translate(50, 245)">'''
    
    # Add video frame previews
    for i in range(4):
        x = i * 95
        svg += f'''
            <rect x="{x}" y="0" width="85" height="60" rx="6" fill="{C['bg_dark']}" stroke="{C['border']}" stroke-width="1">
              <animate attributeName="stroke" values="{C['border']};{C['cyan']};{C['border']}" dur="2s" repeatCount="indefinite" begin="{i*0.5}s"/>
            </rect>
            <text x="{x + 42}" y="35" text-anchor="middle" font-size="20">🦋</text>
            <text x="{x + 42}" y="55" text-anchor="middle" fill="{C['text_muted']}" font-size="8">Frame {i*4 + 1}</text>'''
    
    svg += f'''
          </g>
          
          <!-- Progress bar -->
          <rect x="50" y="320" width="380" height="8" rx="4" fill="{C['bg_dark']}"/>
          <rect x="50" y="320" width="0" height="8" rx="4" fill="{C['cyan']}">
            <animate attributeName="width" values="0;380" dur="3s" repeatCount="indefinite"/>
          </rect>
          
          <text x="50" y="350" fill="{C['text_dim']}" font-family="SF Pro Text, sans-serif" font-size="11">16 frames @ 384×384 • 4 fps</text>
          
          <!-- Video specs -->
          <rect x="50" y="370" width="380" height="60" rx="8" fill="{C['bg_dark']}" stroke="{C['border']}" stroke-width="1"/>
          <text x="60" y="390" fill="{C['text']}" font-family="SF Pro Text, sans-serif" font-size="11">Temporal Diffusion Model</text>
          <text x="60" y="410" fill="{C['text_dim']}" font-family="SF Pro Text, sans-serif" font-size="10">Motion consistency: 94% • Style coherence: 97%</text>
          <text x="60" y="425" fill="{C['green']}" font-family="SF Pro Text, sans-serif" font-size="10">✓ Video generated successfully</text>
        </g>
      </g>
'''
    
    svg += f'''
    </g>
    
    <!-- Input box -->
    <rect x="25" y="805" width="530" height="45" rx="22" fill="{C['bg_dark']}" stroke="{C['border']}" stroke-width="1"/>
    <text x="50" y="833" fill="{C['text_muted']}" font-family="SF Pro Text, sans-serif" font-size="14">Type a message...</text>
    <circle cx="520" cy="827" r="16" fill="{C['blue']}"/>
    <text x="520" y="833" text-anchor="middle" fill="{C['text']}" font-size="14">↑</text>
  </g>
  
  <!-- ==================== RIGHT PANEL: BACKEND VISUALIZATION ==================== -->
  
  <g id="backendPanel">
    <rect x="580" y="65" width="1000" height="820" rx="16" fill="{C['bg_card']}" stroke="{C['border']}" stroke-width="1" filter="url(#shadow)"/>
    
    <!-- Backend header -->
    <rect x="580" y="65" width="1000" height="50" rx="16" fill="{C['bg_dark']}"/>
    <rect x="580" y="99" width="1000" height="16" fill="{C['bg_dark']}"/>
    <text x="610" y="95" fill="{C['text']}" font-family="SF Pro Display, sans-serif" font-size="15" font-weight="600">⚙️ Backend Processing Pipeline</text>
'''
    
    # ==================== BACKEND SCENE 1: CODE GENERATION ====================
    svg += f'''
    <!-- Backend Scene 1: Code Generation -->
    <g id="backend1">
      <animate attributeName="opacity" dur="{TOTAL_DUR}s" repeatCount="indefinite"
               values="1;1;0;0;0;0;0;0;0;0;0;0" keyTimes="0;0.15;0.167;0.33;0.5;0.67;0.833;1;1;1;1;1"/>
      
      <!-- Step 1: Tokenization -->
      <g transform="translate(600, 130)">
        <rect width="300" height="90" rx="10" fill="{C['bg_dark']}" stroke="{C['blue']}" stroke-width="2">
          <animate attributeName="stroke-opacity" values="0.5;1;0.5" dur="2s" repeatCount="indefinite"/>
        </rect>
        <text x="150" y="25" text-anchor="middle" fill="{C['blue']}" font-family="SF Pro Display, sans-serif" font-size="13" font-weight="600">1. Tokenization</text>
        <text x="15" y="50" fill="{C['cyan']}" font-family="SF Mono, monospace" font-size="10">&lt;|user|&gt;</text>
        <text x="75" y="50" fill="{C['text_dim']}" font-family="SF Mono, monospace" font-size="10">Write a Python function...</text>
        <text x="15" y="70" fill="{C['cyan']}" font-family="SF Mono, monospace" font-size="10">&lt;|/user|&gt;</text>
        <text x="15" y="85" fill="{C['text_muted']}" font-family="SF Pro Text, sans-serif" font-size="9">Qwen2.5 Tokenizer • 151,643 vocab</text>
      </g>
      
      <!-- Arrow -->
      <path d="M 910,175 L 940,175" stroke="{C['cyan']}" stroke-width="2" marker-end="url(#arrow)">
        <animate attributeName="opacity" values="0.3;1;0.3" dur="1s" repeatCount="indefinite"/>
      </path>
      
      <!-- Step 2: Embedding -->
      <g transform="translate(960, 130)">
        <rect width="280" height="90" rx="10" fill="{C['bg_dark']}" stroke="{C['purple']}" stroke-width="2">
          <animate attributeName="stroke-opacity" values="0.5;1;0.5" dur="2s" repeatCount="indefinite" begin="0.3s"/>
        </rect>
        <text x="140" y="25" text-anchor="middle" fill="{C['purple']}" font-family="SF Pro Display, sans-serif" font-size="13" font-weight="600">2. Embedding Layer</text>
        <text x="15" y="50" fill="{C['text_dim']}" font-family="SF Pro Text, sans-serif" font-size="10">Token IDs → Dense Vectors</text>
        <text x="15" y="70" fill="{C['text_dim']}" font-family="SF Pro Text, sans-serif" font-size="10">Dimension: 1024</text>
        <text x="15" y="85" fill="{C['text_muted']}" font-family="SF Pro Text, sans-serif" font-size="9">+ Rotary Position Embeddings (RoPE)</text>
      </g>
      
      <!-- Step 3: MoE Router -->
      <g transform="translate(600, 240)">
        <rect width="640" height="160" rx="10" fill="{C['bg_dark']}" stroke="{C['orange']}" stroke-width="2">
          <animate attributeName="stroke-opacity" values="0.5;1;0.5" dur="2s" repeatCount="indefinite" begin="0.6s"/>
        </rect>
        <text x="320" y="25" text-anchor="middle" fill="{C['orange']}" font-family="SF Pro Display, sans-serif" font-size="13" font-weight="600">3. MoE Router - Expert Selection</text>
        
        <!-- Expert grid -->'''
    
    expert_names = ['Code', 'Math', 'Language', 'Logic', 'Vision', 'Audio', 'Tools', 'General']
    for i in range(8):
        col = i % 4
        row = i // 4
        x = 20 + col * 155
        y = 45 + row * 50
        is_selected = i in [0, 3]  # Code and Logic for code generation
        stroke = C['green'] if is_selected else C['border']
        fill_opacity = "0.3" if is_selected else "0.1"
        
        svg += f'''
        <rect x="{x}" y="{y}" width="145" height="40" rx="8" fill="{C['green'] if is_selected else C['bg']}" fill-opacity="{fill_opacity}" stroke="{stroke}" stroke-width="{'2' if is_selected else '1'}">
          {'<animate attributeName="fill-opacity" values="0.15;0.4;0.15" dur="1.5s" repeatCount="indefinite"/>' if is_selected else ''}
        </rect>
        <text x="{x + 72}" y="{y + 25}" text-anchor="middle" fill="{C['green'] if is_selected else C['text_dim']}" font-family="SF Pro Text, sans-serif" font-size="11" font-weight="{'600' if is_selected else '400'}">E{i+1}: {expert_names[i]}{'  ✓' if is_selected else ''}</text>'''
    
    svg += f'''
        <text x="320" y="150" text-anchor="middle" fill="{C['green']}" font-family="SF Pro Text, sans-serif" font-size="10">Top-2 Selected: E1 (Code) + E4 (Logic) + Isolated Shared Expert (Aux-Lossless)</text>
      </g>
      
      <!-- Step 4: Chain of Thought -->
      <g transform="translate(600, 420)">
        <rect width="640" height="120" rx="10" fill="{C['bg_dark']}" stroke="{C['purple']}" stroke-width="2">
          <animate attributeName="stroke-opacity" values="0.5;1;0.5" dur="2s" repeatCount="indefinite" begin="0.9s"/>
        </rect>
        <text x="320" y="25" text-anchor="middle" fill="{C['purple']}" font-family="SF Pro Display, sans-serif" font-size="13" font-weight="600">4. Chain-of-Thought Reasoning</text>
        
        <text x="20" y="50" fill="{C['purple']}" font-family="SF Mono, monospace" font-size="10">&lt;|think|&gt;</text>
        <text x="20" y="70" fill="{C['text_dim']}" font-family="SF Mono, monospace" font-size="10">User wants efficient fibonacci. Options: recursive, iterative, memoization.</text>
        <text x="20" y="90" fill="{C['text_dim']}" font-family="SF Mono, monospace" font-size="10">Memoization gives O(n) time complexity. Best approach for efficiency.</text>
        <text x="20" y="110" fill="{C['purple']}" font-family="SF Mono, monospace" font-size="10">&lt;|/think|&gt;</text>
      </g>
      
      <!-- Step 5: Code Generation -->
      <g transform="translate(600, 560)">
        <rect width="640" height="140" rx="10" fill="{C['bg_dark']}" stroke="{C['green']}" stroke-width="2">
          <animate attributeName="stroke-opacity" values="0.5;1;0.5" dur="2s" repeatCount="indefinite" begin="1.2s"/>
        </rect>
        <text x="320" y="25" text-anchor="middle" fill="{C['green']}" font-family="SF Pro Display, sans-serif" font-size="13" font-weight="600">5. Code Generation</text>
        
        <text x="20" y="50" fill="{C['cyan']}" font-family="SF Mono, monospace" font-size="10">&lt;|code|&gt;&lt;|lang:python|&gt;</text>
        <text x="20" y="70" fill="{C['purple']}" font-family="SF Mono, monospace" font-size="10">def <tspan fill="{C['cyan']}">fibonacci</tspan><tspan fill="{C['text']}">(n, memo={{}}):</tspan></text>
        <text x="30" y="90" fill="{C['purple']}" font-family="SF Mono, monospace" font-size="10">if <tspan fill="{C['text']}">n in memo:</tspan> <tspan fill="{C['purple']}">return</tspan> <tspan fill="{C['text']}">memo[n]</tspan></text>
        <text x="30" y="110" fill="{C['text']}" font-family="SF Mono, monospace" font-size="10">...</text>
        <text x="20" y="130" fill="{C['cyan']}" font-family="SF Mono, monospace" font-size="10">&lt;|/code|&gt;</text>
      </g>
      
      <!-- Step 6: Output Processing -->
      <g transform="translate(600, 720)">
        <rect width="640" height="80" rx="10" fill="{C['bg_dark']}" stroke="{C['cyan']}" stroke-width="2">
          <animate attributeName="stroke-opacity" values="0.5;1;0.5" dur="2s" repeatCount="indefinite" begin="1.5s"/>
        </rect>
        <text x="320" y="25" text-anchor="middle" fill="{C['cyan']}" font-family="SF Pro Display, sans-serif" font-size="13" font-weight="600">6. Output Processing</text>
        
        <text x="20" y="50" fill="{C['text_dim']}" font-family="SF Pro Text, sans-serif" font-size="10">Strip hidden tokens: <tspan fill="{C['red']}" text-decoration="line-through">&lt;|think|&gt;...&lt;|/think|&gt;</tspan></text>
        <text x="20" y="70" fill="{C['green']}" font-family="SF Pro Text, sans-serif" font-size="10">✓ Keep: code block, explanation • Final output ready</text>
      </g>
      
      <!-- Data flow animation -->
      <circle r="6" fill="{C['cyan']}" filter="url(#glow)">
        <animateMotion dur="5s" repeatCount="indefinite" path="M 750,220 L 750,240 L 920,400 L 920,540 L 920,700 L 920,800"/>
        <animate attributeName="opacity" values="0;1;1;1;1;0" dur="5s" repeatCount="indefinite"/>
      </circle>
    </g>
'''
    
    # ==================== BACKEND SCENE 2: IMAGE GENERATION ====================
    svg += f'''
    <!-- Backend Scene 2: Image Generation -->
    <g id="backend2">
      <animate attributeName="opacity" dur="{TOTAL_DUR}s" repeatCount="indefinite"
               values="0;0;1;1;0;0;0;0;0;0;0;0" keyTimes="0;0.15;0.167;0.32;0.333;0.5;0.67;0.833;1;1;1;1"/>
      
      <!-- Text Encoding -->
      <g transform="translate(600, 130)">
        <rect width="300" height="80" rx="10" fill="{C['bg_dark']}" stroke="{C['blue']}" stroke-width="2"/>
        <text x="150" y="25" text-anchor="middle" fill="{C['blue']}" font-family="SF Pro Display, sans-serif" font-size="13" font-weight="600">1. Text Encoding</text>
        <text x="15" y="50" fill="{C['text_dim']}" font-family="SF Pro Text, sans-serif" font-size="10">"magical sunset over mountains"</text>
        <text x="15" y="70" fill="{C['text_muted']}" font-family="SF Pro Text, sans-serif" font-size="9">CLIP Text Encoder → 768-dim embeddings</text>
      </g>
      
      <!-- Diffusion Process -->
      <g transform="translate(600, 230)">
        <rect width="640" height="200" rx="10" fill="{C['bg_dark']}" stroke="{C['pink']}" stroke-width="2"/>
        <text x="320" y="25" text-anchor="middle" fill="{C['pink']}" font-family="SF Pro Display, sans-serif" font-size="13" font-weight="600">2. Diffusion Process (MobileDiffusion)</text>
        
        <!-- Denoising steps visualization -->
        <text x="20" y="50" fill="{C['text_dim']}" font-family="SF Pro Text, sans-serif" font-size="10">Denoising Steps:</text>'''
    
    # Add diffusion step boxes
    for i in range(5):
        x = 20 + i * 125
        noise = 100 - i * 25
        svg += f'''
        <rect x="{x}" y="60" width="115" height="70" rx="6" fill="{C['bg']}" stroke="{C['border']}" stroke-width="1"/>
        <rect x="{x + 5}" y="65" width="105" height="50" rx="4" fill="url(#sunsetGrad)" opacity="{0.2 + i * 0.2}"/>
        <text x="{x + 57}" y="130" text-anchor="middle" fill="{C['text_muted']}" font-family="SF Pro Text, sans-serif" font-size="9">Step {i * 5}</text>
        <text x="{x + 57}" y="145" text-anchor="middle" fill="{C['text_muted']}" font-family="SF Pro Text, sans-serif" font-size="8">Noise: {noise}%</text>'''
        if i < 4:
            svg += f'''
        <text x="{x + 120}" y="95" fill="{C['pink']}" font-size="14">→</text>'''
    
    svg += f'''
        <text x="320" y="175" text-anchor="middle" fill="{C['text_dim']}" font-family="SF Pro Text, sans-serif" font-size="10">20 steps • CFG Scale: 7.5 • Scheduler: DDIM</text>
        <text x="320" y="195" text-anchor="middle" fill="{C['green']}" font-family="SF Pro Text, sans-serif" font-size="10">✓ Latent space denoising complete</text>
      </g>
      
      <!-- VAE Decode -->
      <g transform="translate(600, 450)">
        <rect width="640" height="100" rx="10" fill="{C['bg_dark']}" stroke="{C['green']}" stroke-width="2"/>
        <text x="320" y="25" text-anchor="middle" fill="{C['green']}" font-family="SF Pro Display, sans-serif" font-size="13" font-weight="600">3. VAE Decoder</text>
        <text x="20" y="55" fill="{C['text_dim']}" font-family="SF Pro Text, sans-serif" font-size="10">Latent (96×96×4) → Image (384×384×3)</text>
        <text x="20" y="75" fill="{C['text_dim']}" font-family="SF Pro Text, sans-serif" font-size="10">Upscaling factor: 8x</text>
        <text x="20" y="95" fill="{C['green']}" font-family="SF Pro Text, sans-serif" font-size="10">✓ Final image generated</text>
      </g>
    </g>
'''
    
    # ==================== BACKEND SCENE 3: VISION ====================
    svg += f'''
    <!-- Backend Scene 3: Vision Understanding -->
    <g id="backend3">
      <animate attributeName="opacity" dur="{TOTAL_DUR}s" repeatCount="indefinite"
               values="0;0;0;0;1;1;0;0;0;0;0;0" keyTimes="0;0.167;0.32;0.333;0.35;0.49;0.5;0.67;0.833;1;1;1"/>
      
      <!-- Vision Encoder -->
      <g transform="translate(600, 130)">
        <rect width="300" height="100" rx="10" fill="{C['bg_dark']}" stroke="{C['blue']}" stroke-width="2"/>
        <text x="150" y="25" text-anchor="middle" fill="{C['blue']}" font-family="SF Pro Display, sans-serif" font-size="13" font-weight="600">1. Vision Encoder (SigLIP-2)</text>
        <text x="15" y="50" fill="{C['text_dim']}" font-family="SF Pro Text, sans-serif" font-size="10">Input: 384×384 image + 2D-RoPE</text>
        <text x="15" y="70" fill="{C['text_dim']}" font-family="SF Pro Text, sans-serif" font-size="10">TiTok 1D tokenization → 256 tokens</text>
        <text x="15" y="90" fill="{C['text_muted']}" font-family="SF Pro Text, sans-serif" font-size="9">Dual-stream attention • 1152-dim</text>
      </g>
      
      <!-- Cross Attention -->
      <g transform="translate(920, 130)">
        <rect width="320" height="100" rx="10" fill="{C['bg_dark']}" stroke="{C['cyan']}" stroke-width="2"/>
        <text x="160" y="25" text-anchor="middle" fill="{C['cyan']}" font-family="SF Pro Display, sans-serif" font-size="13" font-weight="600">2. Cross-Attention Fusion</text>
        <text x="15" y="50" fill="{C['text_dim']}" font-family="SF Pro Text, sans-serif" font-size="10">Vision tokens + Text query</text>
        <text x="15" y="70" fill="{C['text_dim']}" font-family="SF Pro Text, sans-serif" font-size="10">4 cross-attention layers</text>
        <text x="15" y="90" fill="{C['text_muted']}" font-family="SF Pro Text, sans-serif" font-size="9">Multimodal understanding</text>
      </g>
      
      <!-- LLM Processing -->
      <g transform="translate(600, 250)">
        <rect width="640" height="120" rx="10" fill="{C['bg_dark']}" stroke="{C['purple']}" stroke-width="2"/>
        <text x="320" y="25" text-anchor="middle" fill="{C['purple']}" font-family="SF Pro Display, sans-serif" font-size="13" font-weight="600">3. MoE LLM - Vision Expert Active</text>
        
        <!-- Show vision expert selected -->
        <rect x="20" y="45" width="140" height="35" rx="6" fill="{C['green']}" fill-opacity="0.3" stroke="{C['green']}" stroke-width="2"/>
        <text x="90" y="68" text-anchor="middle" fill="{C['green']}" font-family="SF Pro Text, sans-serif" font-size="11" font-weight="600">E5: Vision ✓</text>
        
        <rect x="170" y="45" width="140" height="35" rx="6" fill="{C['green']}" fill-opacity="0.3" stroke="{C['green']}" stroke-width="2"/>
        <text x="240" y="68" text-anchor="middle" fill="{C['green']}" font-family="SF Pro Text, sans-serif" font-size="11" font-weight="600">E3: Language ✓</text>
        
        <text x="20" y="105" fill="{C['text_dim']}" font-family="SF Pro Text, sans-serif" font-size="10">Generating detailed scene description with object detection and spatial reasoning</text>
      </g>
      
      <!-- Output -->
      <g transform="translate(600, 390)">
        <rect width="640" height="100" rx="10" fill="{C['bg_dark']}" stroke="{C['green']}" stroke-width="2"/>
        <text x="320" y="25" text-anchor="middle" fill="{C['green']}" font-family="SF Pro Display, sans-serif" font-size="13" font-weight="600">4. Generated Description</text>
        <text x="20" y="55" fill="{C['text']}" font-family="SF Pro Text, sans-serif" font-size="11">"I can see a stunning mountain landscape featuring snow-capped peaks,</text>
        <text x="20" y="75" fill="{C['text']}" font-family="SF Pro Text, sans-serif" font-size="11">a clear blue sky, pine forests, and a serene alpine lake..."</text>
        <text x="20" y="95" fill="{C['green']}" font-family="SF Pro Text, sans-serif" font-size="10">Confidence: 96% • Objects detected: 5</text>
      </g>
    </g>
'''
    
    # ==================== BACKEND SCENE 4: AUDIO ====================
    svg += f'''
    <!-- Backend Scene 4: Text to Audio -->
    <g id="backend4">
      <animate attributeName="opacity" dur="{TOTAL_DUR}s" repeatCount="indefinite"
               values="0;0;0;0;0;0;1;1;0;0;0;0" keyTimes="0;0.167;0.333;0.49;0.5;0.52;0.54;0.65;0.667;0.833;1;1"/>
      
      <!-- Text Processing -->
      <g transform="translate(600, 130)">
        <rect width="300" height="90" rx="10" fill="{C['bg_dark']}" stroke="{C['blue']}" stroke-width="2"/>
        <text x="150" y="25" text-anchor="middle" fill="{C['blue']}" font-family="SF Pro Display, sans-serif" font-size="13" font-weight="600">1. Text Processing</text>
        <text x="15" y="50" fill="{C['text_dim']}" font-family="SF Pro Text, sans-serif" font-size="10">Input: "Welcome to Xoron..."</text>
        <text x="15" y="70" fill="{C['text_dim']}" font-family="SF Pro Text, sans-serif" font-size="10">MAS alignment + Phoneme G2P</text>
        <text x="15" y="85" fill="{C['orange']}" font-family="SF Pro Text, sans-serif" font-size="9">&lt;|emotion:cheerful|&gt; tag detected</text>
      </g>
      
      <!-- TTS Model -->
      <g transform="translate(920, 130)">
        <rect width="320" height="90" rx="10" fill="{C['bg_dark']}" stroke="{C['orange']}" stroke-width="2"/>
        <text x="160" y="25" text-anchor="middle" fill="{C['orange']}" font-family="SF Pro Display, sans-serif" font-size="13" font-weight="600">2. Neural TTS (RMLA)</text>
        <text x="15" y="50" fill="{C['text_dim']}" font-family="SF Pro Text, sans-serif" font-size="10">Emotion-aware synthesis</text>
        <text x="15" y="70" fill="{C['text_dim']}" font-family="SF Pro Text, sans-serif" font-size="10">Zero-shot voice cloning</text>
        <text x="15" y="85" fill="{C['text_muted']}" font-family="SF Pro Text, sans-serif" font-size="9">Raw waveform generation</text>
      </g>
      
      <!-- Mel Spectrogram Visualization -->
      <g transform="translate(600, 240)">
        <rect width="640" height="150" rx="10" fill="{C['bg_dark']}" stroke="{C['pink']}" stroke-width="2"/>
        <text x="320" y="25" text-anchor="middle" fill="{C['pink']}" font-family="SF Pro Display, sans-serif" font-size="13" font-weight="600">3. Mel Spectrogram</text>
        
        <!-- Spectrogram bars -->
        <g transform="translate(20, 45)">'''
    
    import random
    random.seed(42)
    for i in range(60):
        h = random.randint(15, 80)
        svg += f'''
          <rect x="{i * 10}" y="{90 - h}" width="7" height="{h}" rx="2" fill="{C['orange']}" opacity="0.7">
            <animate attributeName="height" values="{h};{h + 10};{h}" dur="0.3s" repeatCount="indefinite" begin="{i * 0.02}s"/>
          </rect>'''
    
    svg += f'''
        </g>
        <text x="320" y="145" text-anchor="middle" fill="{C['text_muted']}" font-family="SF Pro Text, sans-serif" font-size="9">80 mel bins × 320 frames</text>
      </g>
      
      <!-- Waveform Decoder -->
      <g transform="translate(600, 410)">
        <rect width="640" height="80" rx="10" fill="{C['bg_dark']}" stroke="{C['green']}" stroke-width="2"/>
        <text x="320" y="25" text-anchor="middle" fill="{C['green']}" font-family="SF Pro Display, sans-serif" font-size="13" font-weight="600">4. Raw Waveform Decoder</text>
        <text x="20" y="50" fill="{C['text_dim']}" font-family="SF Pro Text, sans-serif" font-size="10">Direct audio output • No vocoder needed • 16kHz</text>
        <text x="20" y="70" fill="{C['green']}" font-family="SF Pro Text, sans-serif" font-size="10">✓ Audio generated: 3.2 seconds • Speech-to-Speech ready</text>
      </g>
    </g>
'''
    
    # ==================== BACKEND SCENE 5: TOOL CALLING ====================
    svg += f'''
    <!-- Backend Scene 5: Tool Calling -->
    <g id="backend5">
      <animate attributeName="opacity" dur="{TOTAL_DUR}s" repeatCount="indefinite"
               values="0;0;0;0;0;0;0;0;1;1;0;0" keyTimes="0;0.167;0.333;0.5;0.65;0.667;0.69;0.71;0.72;0.82;0.833;1"/>
      
      <!-- Intent Detection -->
      <g transform="translate(600, 130)">
        <rect width="300" height="80" rx="10" fill="{C['bg_dark']}" stroke="{C['blue']}" stroke-width="2"/>
        <text x="150" y="25" text-anchor="middle" fill="{C['blue']}" font-family="SF Pro Display, sans-serif" font-size="13" font-weight="600">1. Intent Detection</text>
        <text x="15" y="50" fill="{C['text_dim']}" font-family="SF Pro Text, sans-serif" font-size="10">Query requires external data</text>
        <text x="15" y="70" fill="{C['green']}" font-family="SF Pro Text, sans-serif" font-size="10">→ Tool calling required</text>
      </g>
      
      <!-- Available Tools -->
      <g transform="translate(920, 130)">
        <rect width="320" height="80" rx="10" fill="{C['bg_dark']}" stroke="{C['purple']}" stroke-width="2"/>
        <text x="160" y="25" text-anchor="middle" fill="{C['purple']}" font-family="SF Pro Display, sans-serif" font-size="13" font-weight="600">2. Available Tools</text>
        <text x="15" y="50" fill="{C['cyan']}" font-family="SF Mono, monospace" font-size="9">&lt;|available_tools|&gt;</text>
        <text x="15" y="70" fill="{C['text_dim']}" font-family="SF Mono, monospace" font-size="9">web_search, execute_python, read_file...</text>
      </g>
      
      <!-- Tool Selection -->
      <g transform="translate(600, 230)">
        <rect width="640" height="100" rx="10" fill="{C['bg_dark']}" stroke="{C['orange']}" stroke-width="2"/>
        <text x="320" y="25" text-anchor="middle" fill="{C['orange']}" font-family="SF Pro Display, sans-serif" font-size="13" font-weight="600">3. Tool Selection &amp; Execution</text>
        
        <text x="20" y="50" fill="{C['cyan']}" font-family="SF Mono, monospace" font-size="10">&lt;|tool_call|&gt;</text>
        <text x="30" y="70" fill="{C['text_dim']}" font-family="SF Mono, monospace" font-size="10">&lt;|function_name|&gt;web_search&lt;|/function_name|&gt;</text>
        <text x="30" y="90" fill="{C['text_dim']}" font-family="SF Mono, monospace" font-size="10">&lt;|function_args|&gt;{{"query": "Python 3.12 features"}}&lt;|/function_args|&gt;</text>
      </g>
      
      <!-- Tool Result -->
      <g transform="translate(600, 350)">
        <rect width="640" height="100" rx="10" fill="{C['bg_dark']}" stroke="{C['green']}" stroke-width="2"/>
        <text x="320" y="25" text-anchor="middle" fill="{C['green']}" font-family="SF Pro Display, sans-serif" font-size="13" font-weight="600">4. Tool Result Processing</text>
        
        <text x="20" y="50" fill="{C['green']}" font-family="SF Mono, monospace" font-size="10">&lt;|tool_result|&gt;</text>
        <text x="30" y="70" fill="{C['text_dim']}" font-family="SF Mono, monospace" font-size="10">Found 5 results from python.org, realpython.com...</text>
        <text x="20" y="90" fill="{C['green']}" font-family="SF Mono, monospace" font-size="10">&lt;|/tool_result|&gt;</text>
      </g>
      
      <!-- Response Generation -->
      <g transform="translate(600, 470)">
        <rect width="640" height="80" rx="10" fill="{C['bg_dark']}" stroke="{C['cyan']}" stroke-width="2"/>
        <text x="320" y="25" text-anchor="middle" fill="{C['cyan']}" font-family="SF Pro Display, sans-serif" font-size="13" font-weight="600">5. Response Synthesis</text>
        <text x="20" y="50" fill="{C['text_dim']}" font-family="SF Pro Text, sans-serif" font-size="10">Combining tool results with LLM knowledge</text>
        <text x="20" y="70" fill="{C['green']}" font-family="SF Pro Text, sans-serif" font-size="10">✓ Cited sources: 3 • Confidence: High</text>
      </g>
    </g>
'''
    
    # ==================== BACKEND SCENE 6: VIDEO ====================
    svg += f'''
    <!-- Backend Scene 6: Video Generation -->
    <g id="backend6">
      <animate attributeName="opacity" dur="{TOTAL_DUR}s" repeatCount="indefinite"
               values="0;0;0;0;0;0;0;0;0;0;1;1" keyTimes="0;0.167;0.333;0.5;0.667;0.82;0.833;0.85;0.87;0.89;0.9;1"/>
      
      <!-- Text/Image Encoding -->
      <g transform="translate(600, 130)">
        <rect width="300" height="80" rx="10" fill="{C['bg_dark']}" stroke="{C['blue']}" stroke-width="2"/>
        <text x="150" y="25" text-anchor="middle" fill="{C['blue']}" font-family="SF Pro Display, sans-serif" font-size="13" font-weight="600">1. Prompt Encoding</text>
        <text x="15" y="50" fill="{C['text_dim']}" font-family="SF Pro Text, sans-serif" font-size="10">"butterfly landing on flower"</text>
        <text x="15" y="70" fill="{C['text_muted']}" font-family="SF Pro Text, sans-serif" font-size="9">CLIP + Motion description encoding</text>
      </g>
      
      <!-- Temporal Diffusion -->
      <g transform="translate(600, 230)">
        <rect width="640" height="180" rx="10" fill="{C['bg_dark']}" stroke="{C['purple']}" stroke-width="2"/>
        <text x="320" y="25" text-anchor="middle" fill="{C['purple']}" font-family="SF Pro Display, sans-serif" font-size="13" font-weight="600">2. 3D Causal Transformer + Flow Matching</text>
        
        <!-- Frame generation visualization -->
        <text x="20" y="50" fill="{C['text_dim']}" font-family="SF Pro Text, sans-serif" font-size="10">Generating 16 frames with 3D-RoPE + Temporal MoE:</text>
        
        <g transform="translate(20, 60)">'''
    
    # Add video frame boxes
    for i in range(8):
        x = i * 75
        svg += f'''
          <rect x="{x}" y="0" width="65" height="45" rx="4" fill="{C['bg']}" stroke="{C['border']}" stroke-width="1">
            <animate attributeName="stroke" values="{C['border']};{C['purple']};{C['border']}" dur="2s" repeatCount="indefinite" begin="{i * 0.2}s"/>
          </rect>
          <text x="{x + 32}" y="28" text-anchor="middle" font-size="16">🦋</text>
          <text x="{x + 32}" y="60" text-anchor="middle" fill="{C['text_muted']}" font-size="8">F{i + 1}</text>'''
    
    svg += f'''
        </g>
        
        <text x="20" y="140" fill="{C['text_dim']}" font-family="SF Pro Text, sans-serif" font-size="10">Motion vectors: Smooth landing trajectory</text>
        <text x="20" y="160" fill="{C['text_dim']}" font-family="SF Pro Text, sans-serif" font-size="10">Temporal attention: Cross-frame consistency</text>
        <text x="20" y="175" fill="{C['green']}" font-family="SF Pro Text, sans-serif" font-size="10">✓ Motion coherence: 94%</text>
      </g>
      
      <!-- Frame Interpolation -->
      <g transform="translate(600, 430)">
        <rect width="640" height="80" rx="10" fill="{C['bg_dark']}" stroke="{C['green']}" stroke-width="2"/>
        <text x="320" y="25" text-anchor="middle" fill="{C['green']}" font-family="SF Pro Display, sans-serif" font-size="13" font-weight="600">3. Output Processing</text>
        <text x="20" y="50" fill="{C['text_dim']}" font-family="SF Pro Text, sans-serif" font-size="10">16 frames @ 384×384 • 4 fps • Duration: 4 seconds</text>
        <text x="20" y="70" fill="{C['green']}" font-family="SF Pro Text, sans-serif" font-size="10">✓ Video generated successfully • Style coherence: 97%</text>
      </g>
    </g>
'''
    
    svg += f'''
  </g>
  
  <!-- Footer -->
  <rect x="0" y="{HEIGHT - 35}" width="{WIDTH}" height="35" fill="{C['bg_card']}"/>
  <text x="{WIDTH // 2}" y="{HEIGHT - 12}" text-anchor="middle" fill="{C['text_muted']}" font-family="SF Pro Text, sans-serif" font-size="11">
    Xoron-Dev • Unified Multimodal AI • 400+ Special Tokens • MoE Architecture • github.com/nigfuapp-web/Xoron-Dev
  </text>

</svg>'''
    
    return svg


def main():
    svg_content = generate_svg()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    assets_dir = os.path.join(os.path.dirname(script_dir), 'assets')
    os.makedirs(assets_dir, exist_ok=True)
    
    output_path = os.path.join(assets_dir, 'xoron_demo_animated.svg')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(svg_content)
    
    print(f"✅ Generated comprehensive 60-second demo SVG: {output_path}")
    print(f"   Dimensions: {WIDTH}x{HEIGHT}")
    print(f"   Duration: 60 seconds (6 scenes × 10s each)")
    print(f"   Scenes:")
    print(f"     1. (0-10s)  Code Generation - Fibonacci with memoization")
    print(f"     2. (10-20s) Text to Image - Sunset generation with diffusion")
    print(f"     3. (20-30s) Image to Text - Vision understanding with SigLIP")
    print(f"     4. (30-40s) Text to Audio - TTS with emotion control")
    print(f"     5. (40-50s) Tool Calling - Web search agentic workflow")
    print(f"     6. (50-60s) Video Generation - Temporal diffusion")


if __name__ == '__main__':
    main()
