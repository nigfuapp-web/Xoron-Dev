#!/usr/bin/env python3
"""
Generate a high-quality SVG visualization of Xoron-Dev synthetic datasets.

Creates a beautiful 4K infographic showing all dataset types and their purposes.
"""

import os
import json

# SVG dimensions - 4K quality
WIDTH = 1920
HEIGHT = 1600

# Color palette
COLORS = {
    'bg': '#0a0a0f',
    'bg_card': '#12121a',
    'bg_inner': '#1a1a25',
    'text_primary': '#ffffff',
    'text_secondary': '#a0a0b0',
    'text_muted': '#606070',
    'text_count': '#22d3ee',
    'border': '#2a2a3a',
    # Category colors
    'cat_shell': '#3b82f6',          # Blue - Shell/Execution
    'cat_code': '#10b981',           # Green - Code/Programming
    'cat_git': '#8b5cf6',            # Purple - Git/Version Control
    'cat_sysadmin': '#f59e0b',       # Orange - System Admin
    'cat_antihallucination': '#ef4444',  # Red - Anti-Hallucination
    'cat_document': '#06b6d4',       # Cyan - Documents
}

# Dataset categories
DATASET_CATEGORIES = {
    'Shell & Execution': {
        'color': COLORS['cat_shell'],
        'icon': '⚡',
        'datasets': [
            {'name': 'shell_execution', 'desc': 'Basic shell commands', 'samples': 2000},
            {'name': 'shell_error', 'desc': 'Error handling patterns', 'samples': 2000},
            {'name': 'shell_timeout', 'desc': 'Timeout scenarios', 'samples': 2000},
            {'name': 'multi_step_execution', 'desc': 'Multi-command workflows', 'samples': 2000},
            {'name': 'execution', 'desc': 'Code execution results', 'samples': 2000},
        ]
    },
    'Code & Programming': {
        'color': COLORS['cat_code'],
        'icon': '💻',
        'datasets': [
            {'name': 'python_script', 'desc': 'Python script execution', 'samples': 2000},
            {'name': 'jupyter', 'desc': 'Jupyter notebook cells', 'samples': 2000},
            {'name': 'debugging', 'desc': 'Debug & fix scenarios', 'samples': 2000},
            {'name': 'file_ops', 'desc': 'File operations', 'samples': 2000},
            {'name': 'edit_lines', 'desc': 'Line-level edits', 'samples': 2000},
            {'name': 'fim', 'desc': 'Fill-in-the-middle', 'samples': 2000},
        ]
    },
    'Git & Version Control': {
        'color': COLORS['cat_git'],
        'icon': '📦',
        'datasets': [
            {'name': 'commit', 'desc': 'Commit messages', 'samples': 2000},
            {'name': 'diff', 'desc': 'Code diffs', 'samples': 2000},
            {'name': 'issue', 'desc': 'GitHub issues', 'samples': 2000},
            {'name': 'repo_context', 'desc': 'Repository context', 'samples': 2000},
        ]
    },
    'System Administration': {
        'color': COLORS['cat_sysadmin'],
        'icon': '🔧',
        'datasets': [
            {'name': 'apt_install', 'desc': 'Package installation', 'samples': 2000},
            {'name': 'docker', 'desc': 'Docker operations', 'samples': 2000},
            {'name': 'database_setup', 'desc': 'Database configuration', 'samples': 2000},
            {'name': 'webserver_setup', 'desc': 'Web server setup', 'samples': 2000},
            {'name': 'ssh_setup', 'desc': 'SSH configuration', 'samples': 2000},
            {'name': 'monitoring', 'desc': 'System monitoring', 'samples': 2000},
            {'name': 'download', 'desc': 'File downloads', 'samples': 2000},
            {'name': 'language_setup', 'desc': 'Language environments', 'samples': 2000},
            {'name': 'desktop_setup', 'desc': 'Desktop configuration', 'samples': 2000},
        ]
    },
    'Anti-Hallucination': {
        'color': COLORS['cat_antihallucination'],
        'icon': '🛡️',
        'datasets': [
            {'name': 'uncertainty', 'desc': 'Uncertainty expression', 'samples': 2000},
            {'name': 'idk', 'desc': '"I don\'t know" responses', 'samples': 2000},
            {'name': 'fact_check', 'desc': 'Fact verification', 'samples': 2000},
            {'name': 'grounded_response', 'desc': 'Grounded answers', 'samples': 2000},
            {'name': 'self_correction', 'desc': 'Self-correction', 'samples': 2000},
            {'name': 'confidence_level', 'desc': 'Confidence scoring', 'samples': 2000},
            {'name': 'citation', 'desc': 'Source citations', 'samples': 2000},
            {'name': 'retrieval_grounded', 'desc': 'RAG grounding', 'samples': 2000},
            {'name': 'knowledge_cutoff', 'desc': 'Knowledge boundaries', 'samples': 2000},
        ]
    },
    'Documents & Reasoning': {
        'color': COLORS['cat_document'],
        'icon': '📄',
        'datasets': [
            {'name': 'document', 'desc': 'Document processing', 'samples': 2000},
            {'name': 'cot', 'desc': 'Chain-of-thought', 'samples': 2000},
        ]
    },
}


def escape_xml(text: str) -> str:
    """Escape special XML characters."""
    return (text
            .replace('&', '&amp;')
            .replace('<', '&lt;')
            .replace('>', '&gt;')
            .replace('"', '&quot;')
            .replace("'", '&apos;'))


def generate_svg() -> str:
    """Generate the complete SVG."""
    
    margin = 60
    
    # Calculate total samples
    total_samples = sum(
        sum(d['samples'] for d in cat['datasets'])
        for cat in DATASET_CATEGORIES.values()
    )
    total_datasets = sum(len(cat['datasets']) for cat in DATASET_CATEGORIES.values())
    
    svg_parts = [f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {HEIGHT}" width="{WIDTH}" height="{HEIGHT}">
  <defs>
    <filter id="shadow" x="-20%" y="-20%" width="140%" height="140%">
      <feDropShadow dx="0" dy="4" stdDeviation="8" flood-color="#000000" flood-opacity="0.5"/>
    </filter>
    <linearGradient id="titleGrad" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#10b981"/>
      <stop offset="50%" stop-color="#3b82f6"/>
      <stop offset="100%" stop-color="#8b5cf6"/>
    </linearGradient>
    <linearGradient id="barGrad" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#3b82f6"/>
      <stop offset="100%" stop-color="#8b5cf6"/>
    </linearGradient>
  </defs>
  
  <!-- Background -->
  <rect width="{WIDTH}" height="{HEIGHT}" fill="{COLORS['bg']}"/>
  
  <!-- Grid pattern -->
  <defs>
    <pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse">
      <path d="M 40 0 L 0 0 0 40" fill="none" stroke="#151520" stroke-width="1"/>
    </pattern>
  </defs>
  <rect width="{WIDTH}" height="{HEIGHT}" fill="url(#grid)" opacity="0.5"/>
  
  <!-- Title -->
  <text x="{WIDTH//2}" y="55" text-anchor="middle" fill="url(#titleGrad)" font-family="Inter, -apple-system, sans-serif" font-size="42" font-weight="bold">
    📊 XORON-DEV SYNTHETIC DATASETS
  </text>
  <text x="{WIDTH//2}" y="90" text-anchor="middle" fill="{COLORS['text_secondary']}" font-family="Inter, -apple-system, sans-serif" font-size="18">
    High-Quality Training Data for Agentic AI
  </text>
  
  <!-- Stats bar -->
  <rect x="{margin}" y="115" width="{WIDTH - 2*margin}" height="60" rx="12" fill="{COLORS['bg_card']}" stroke="{COLORS['border']}" stroke-width="1" filter="url(#shadow)"/>
  
  <!-- Stats content -->
  <text x="{margin + 40}" y="152" fill="{COLORS['text_primary']}" font-family="Inter, -apple-system, sans-serif" font-size="16" font-weight="bold">
    📁 {total_datasets} Datasets
  </text>
  <text x="{margin + 220}" y="152" fill="{COLORS['text_secondary']}" font-family="Inter, -apple-system, sans-serif" font-size="16">
    |  📝 {total_samples:,} Total Samples
  </text>
  <text x="{margin + 500}" y="152" fill="{COLORS['text_secondary']}" font-family="Inter, -apple-system, sans-serif" font-size="16">
    |  🗂️ {len(DATASET_CATEGORIES)} Categories
  </text>
  <text x="{margin + 730}" y="152" fill="{COLORS['text_secondary']}" font-family="Inter, -apple-system, sans-serif" font-size="16">
    |  🔄 100% Unique Samples
  </text>
  <text x="{margin + 1000}" y="152" fill="{COLORS['text_secondary']}" font-family="Inter, -apple-system, sans-serif" font-size="16">
    |  🏷️ Available Tools Tokens Included
  </text>
''']

    # Layout: 3 columns
    col_width = (WIDTH - 2*margin - 40) // 3
    col_gap = 20
    y_start = 200
    
    # Distribute categories across columns
    categories = list(DATASET_CATEGORIES.items())
    cols = [[], [], []]
    col_heights = [0, 0, 0]
    
    for cat_name, cat_info in categories:
        # Find shortest column
        min_col = col_heights.index(min(col_heights))
        cols[min_col].append((cat_name, cat_info))
        # Estimate height: header (60) + datasets (40 each) + padding (20)
        col_heights[min_col] += 80 + len(cat_info['datasets']) * 45
    
    # Draw columns
    for col_idx, col_categories in enumerate(cols):
        x = margin + col_idx * (col_width + col_gap)
        y = y_start
        
        for cat_name, cat_info in col_categories:
            num_datasets = len(cat_info['datasets'])
            card_height = 75 + num_datasets * 45
            
            # Category card - escape text
            cat_name_escaped = escape_xml(cat_name)
            
            svg_parts.append(f'''
  <!-- {cat_name_escaped} -->
  <rect x="{x}" y="{y}" width="{col_width}" height="{card_height}" rx="12" fill="{COLORS['bg_card']}" stroke="{cat_info['color']}" stroke-width="2" filter="url(#shadow)"/>
  
  <!-- Header gradient overlay -->
  <rect x="{x + 1}" y="{y + 1}" width="{col_width - 2}" height="48" rx="11" fill="{cat_info['color']}" opacity="0.15"/>
  <text x="{x + 20}" y="{y + 32}" fill="{COLORS['text_primary']}" font-family="Inter, -apple-system, sans-serif" font-size="18" font-weight="bold">{cat_info['icon']} {cat_name_escaped}</text>
  <text x="{x + col_width - 20}" y="{y + 32}" text-anchor="end" fill="{cat_info['color']}" font-family="Inter, -apple-system, sans-serif" font-size="14" font-weight="bold">{num_datasets} datasets</text>
''')
            
            # Draw datasets
            dataset_y = y + 60
            for dataset in cat_info['datasets']:
                # Dataset row - escape text
                ds_name = escape_xml(dataset['name'].replace('_', ' ').title())
                ds_desc = escape_xml(dataset['desc'])
                
                svg_parts.append(f'''
  <rect x="{x + 12}" y="{dataset_y}" width="{col_width - 24}" height="38" rx="8" fill="{COLORS['bg_inner']}"/>
  <text x="{x + 22}" y="{dataset_y + 16}" fill="{COLORS['text_primary']}" font-family="Inter, -apple-system, sans-serif" font-size="13" font-weight="bold">{ds_name}</text>
  <text x="{x + 22}" y="{dataset_y + 31}" fill="{COLORS['text_muted']}" font-family="Inter, -apple-system, sans-serif" font-size="10">{ds_desc}</text>
  <text x="{x + col_width - 32}" y="{dataset_y + 24}" text-anchor="end" fill="{COLORS['text_count']}" font-family="monospace" font-size="12" font-weight="bold">{dataset['samples']:,}</text>
''')
                dataset_y += 45
            
            y += card_height + 20
    
    # Features section
    features_y = HEIGHT - 180
    svg_parts.append(f'''
  <!-- Features Section -->
  <rect x="{margin}" y="{features_y}" width="{WIDTH - 2*margin}" height="120" rx="12" fill="{COLORS['bg_card']}" stroke="{COLORS['border']}" stroke-width="1" filter="url(#shadow)"/>
  <text x="{WIDTH//2}" y="{features_y + 30}" text-anchor="middle" fill="{COLORS['text_primary']}" font-family="Inter, -apple-system, sans-serif" font-size="18" font-weight="bold">
    ✨ Dataset Features
  </text>
  
  <!-- Feature items -->
  <g transform="translate({margin + 60}, {features_y + 55})">
    <rect width="200" height="50" rx="8" fill="{COLORS['bg_inner']}"/>
    <text x="100" y="22" text-anchor="middle" fill="{COLORS['text_primary']}" font-family="Inter, -apple-system, sans-serif" font-size="13" font-weight="bold">🔄 Parameterized</text>
    <text x="100" y="40" text-anchor="middle" fill="{COLORS['text_muted']}" font-family="Inter, -apple-system, sans-serif" font-size="10">Unique samples via generation</text>
  </g>
  
  <g transform="translate({margin + 290}, {features_y + 55})">
    <rect width="200" height="50" rx="8" fill="{COLORS['bg_inner']}"/>
    <text x="100" y="22" text-anchor="middle" fill="{COLORS['text_primary']}" font-family="Inter, -apple-system, sans-serif" font-size="13" font-weight="bold">🏷️ Special Tokens</text>
    <text x="100" y="40" text-anchor="middle" fill="{COLORS['text_muted']}" font-family="Inter, -apple-system, sans-serif" font-size="10">Structured output format</text>
  </g>
  
  <g transform="translate({margin + 520}, {features_y + 55})">
    <rect width="200" height="50" rx="8" fill="{COLORS['bg_inner']}"/>
    <text x="100" y="22" text-anchor="middle" fill="{COLORS['text_primary']}" font-family="Inter, -apple-system, sans-serif" font-size="13" font-weight="bold">🔧 Available Tools</text>
    <text x="100" y="40" text-anchor="middle" fill="{COLORS['text_muted']}" font-family="Inter, -apple-system, sans-serif" font-size="10">Tool definitions in 50% samples</text>
  </g>
  
  <g transform="translate({margin + 750}, {features_y + 55})">
    <rect width="200" height="50" rx="8" fill="{COLORS['bg_inner']}"/>
    <text x="100" y="22" text-anchor="middle" fill="{COLORS['text_primary']}" font-family="Inter, -apple-system, sans-serif" font-size="13" font-weight="bold">⚖️ Weighted Loss</text>
    <text x="100" y="40" text-anchor="middle" fill="{COLORS['text_muted']}" font-family="Inter, -apple-system, sans-serif" font-size="10">Higher weight for key tokens</text>
  </g>
  
  <g transform="translate({margin + 980}, {features_y + 55})">
    <rect width="200" height="50" rx="8" fill="{COLORS['bg_inner']}"/>
    <text x="100" y="22" text-anchor="middle" fill="{COLORS['text_primary']}" font-family="Inter, -apple-system, sans-serif" font-size="13" font-weight="bold">📦 JSONL Format</text>
    <text x="100" y="40" text-anchor="middle" fill="{COLORS['text_muted']}" font-family="Inter, -apple-system, sans-serif" font-size="10">Easy to load and process</text>
  </g>
  
  <g transform="translate({margin + 1210}, {features_y + 55})">
    <rect width="200" height="50" rx="8" fill="{COLORS['bg_inner']}"/>
    <text x="100" y="22" text-anchor="middle" fill="{COLORS['text_primary']}" font-family="Inter, -apple-system, sans-serif" font-size="13" font-weight="bold">🎯 Task-Specific</text>
    <text x="100" y="40" text-anchor="middle" fill="{COLORS['text_muted']}" font-family="Inter, -apple-system, sans-serif" font-size="10">Targeted training data</text>
  </g>
''')

    # Footer
    svg_parts.append(f'''
  <!-- Footer -->
  <text x="{WIDTH//2}" y="{HEIGHT - 25}" text-anchor="middle" fill="{COLORS['text_muted']}" font-family="Inter, -apple-system, sans-serif" font-size="12">
    Xoron-Dev Synthetic Datasets • Generated with unique_generator.py • github.com/nigfuapp-web/Xoron-Dev
  </text>
</svg>''')
    
    return '\n'.join(svg_parts)


def main():
    """Generate and save the SVG."""
    svg_content = generate_svg()
    
    # Ensure assets directory exists
    script_dir = os.path.dirname(os.path.abspath(__file__))
    assets_dir = os.path.join(os.path.dirname(script_dir), 'assets')
    os.makedirs(assets_dir, exist_ok=True)
    
    # Save SVG
    output_path = os.path.join(assets_dir, 'datasets_overview.svg')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(svg_content)
    
    print(f"✅ Generated datasets overview SVG: {output_path}")
    print(f"   Dimensions: {WIDTH}x{HEIGHT}")
    print(f"   Categories: {len(DATASET_CATEGORIES)}")
    
    total_datasets = sum(len(cat['datasets']) for cat in DATASET_CATEGORIES.values())
    total_samples = sum(
        sum(d['samples'] for d in cat['datasets'])
        for cat in DATASET_CATEGORIES.values()
    )
    print(f"   Total datasets: {total_datasets}")
    print(f"   Total samples: {total_samples:,}")


if __name__ == '__main__':
    main()
