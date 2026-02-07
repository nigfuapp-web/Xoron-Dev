#!/usr/bin/env python3
"""
Generate a high-quality SVG visualization of Xoron-Dev special tokens.

Creates a beautiful 4K infographic showing all token categories and their purposes.
"""

import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.special_tokens import SPECIAL_TOKENS

# SVG dimensions - 4K quality
WIDTH = 1920
HEIGHT = 3200  # Taller to fit all tokens with 2 per row

# Color palette
COLORS = {
    'bg': '#0a0a0f',
    'bg_card': '#12121a',
    'bg_token': '#1a1a25',
    'text_primary': '#ffffff',
    'text_secondary': '#a0a0b0',
    'text_muted': '#606070',
    'text_token': '#22d3ee',
    'border': '#2a2a3a',
    # Category colors
    'cat_sequence': '#3b82f6',      # Blue
    'cat_conversation': '#8b5cf6',   # Purple
    'cat_memory': '#06b6d4',         # Cyan
    'cat_fim': '#f59e0b',            # Orange
    'cat_git': '#10b981',            # Green
    'cat_jupyter': '#ec4899',        # Pink
    'cat_file': '#f97316',           # Orange-red
    'cat_document': '#6366f1',       # Indigo
    'cat_multimodal': '#14b8a6',     # Teal
    'cat_tool': '#ef4444',           # Red
    'cat_code': '#84cc16',           # Lime
    'cat_thinking': '#a855f7',       # Violet
    'cat_antihallucination': '#f43f5e',  # Rose
    'cat_audio': '#22c55e',          # Green
    'cat_data': '#0ea5e9',           # Sky
}

# Token categories with their tokens
TOKEN_CATEGORIES = {
    'Sequence Control': {
        'color': COLORS['cat_sequence'],
        'icon': '🔄',
        'tokens': ['bos', 'eos', 'pad', 'prompt_start', 'prompt_end', 'text_start', 'text_end', 'response_start', 'response_end'],
        'description': 'Control sequence boundaries and text flow'
    },
    'Conversation': {
        'color': COLORS['cat_conversation'],
        'icon': '💬',
        'tokens': ['system_start', 'system_end', 'user_start', 'user_end', 'assistant_start', 'assistant_end'],
        'description': 'Multi-turn conversation structure'
    },
    'Memory & Context': {
        'color': COLORS['cat_memory'],
        'icon': '🧠',
        'tokens': ['memory_start', 'memory_end', 'working_memory_start', 'working_memory_end', 'summary_start', 'summary_end', 'user_profile_start', 'user_profile_end', 'session_start', 'session_end'],
        'description': 'Long-term memory and context management'
    },
    'Fill-in-Middle (FIM)': {
        'color': COLORS['cat_fim'],
        'icon': '✏️',
        'tokens': ['fim_prefix', 'fim_middle', 'fim_suffix', 'fim_pad'],
        'description': 'Code completion and ghost text'
    },
    'Git & Version Control': {
        'color': COLORS['cat_git'],
        'icon': '📦',
        'tokens': ['commit_before', 'commit_after', 'commit_msg', 'diff_start', 'diff_end', 'diff_add', 'diff_del', 'reponame', 'branch', 'issue_start', 'issue_end', 'pr_start', 'pr_end'],
        'description': 'Git operations and repository context'
    },
    'Code Execution': {
        'color': COLORS['cat_jupyter'],
        'icon': '⚡',
        'tokens': ['jupyter_start', 'jupyter_end', 'jupyter_code', 'jupyter_output', 'jupyter_error', 'exec_start', 'exec_end', 'exec_result', 'exec_error', 'exec_timeout', 'empty_output'],
        'description': 'Jupyter notebooks and shell execution'
    },
    'File Operations': {
        'color': COLORS['cat_file'],
        'icon': '📁',
        'tokens': ['add_file', 'delete_file', 'rename_file', 'edit_file', 'read_file', 'file_content', 'filepath_start', 'filepath_end', 'edit_range', 'line_num', 'replace', 'insert_before', 'insert_after'],
        'description': 'Agentic file system operations'
    },
    'Document Types': {
        'color': COLORS['cat_document'],
        'icon': '📄',
        'tokens': ['doc_start', 'doc_end', 'file_txt', 'file_md', 'file_json', 'file_yaml', 'file_html', 'file_css', 'file_csv', 'filename_start', 'filename_end'],
        'description': 'Document and file type markers'
    },
    'Multimodal': {
        'color': COLORS['cat_multimodal'],
        'icon': '🎨',
        'tokens': ['image_start', 'image_end', 'video_start', 'video_end', 'gen_image_start', 'gen_image_end', 'gen_video_start', 'gen_video_end', 'timestamp_start', 'keyframe', 'scene_change', 'bbox_start', 'region_start'],
        'description': 'Image, video, and spatial markers'
    },
    'Tool Calling': {
        'color': COLORS['cat_tool'],
        'icon': '🔧',
        'tokens': ['tool_call_start', 'tool_call_end', 'tool_result_start', 'tool_result_end', 'function_name_start', 'function_args_start', 'available_tools_start', 'available_tools_end', 'tool_def_start', 'tool_error_start', 'tool_success'],
        'description': 'Function calling and tool use'
    },
    'Code Languages': {
        'color': COLORS['cat_code'],
        'icon': '💻',
        'tokens': ['code_start', 'code_end', 'lang_python', 'lang_javascript', 'lang_typescript', 'lang_rust', 'lang_go', 'lang_java', 'lang_cpp', 'lang_shell', 'lang_sql'],
        'description': 'Programming language markers'
    },
    'Chain-of-Thought': {
        'color': COLORS['cat_thinking'],
        'icon': '💭',
        'tokens': ['think_start', 'think_end', 'observation_start', 'step_start', 'reflection_start', 'hypothesis_start', 'conclusion_start', 'plan_start', 'plan_end', 'critique_start', 'analysis_start', 'decision_start', 'because', 'therefore'],
        'description': 'Reasoning and inner monologue'
    },
    'Anti-Hallucination': {
        'color': COLORS['cat_antihallucination'],
        'icon': '🛡️',
        'tokens': ['confidence_high', 'confidence_medium', 'confidence_low', 'uncertain_start', 'unknown', 'need_verification', 'speculative', 'verify_start', 'fact_check', 'self_correct', 'cite_start', 'source_start', 'grounded', 'knowledge_cutoff'],
        'description': 'Uncertainty and citation tokens'
    },
    'Audio & Speech': {
        'color': COLORS['cat_audio'],
        'icon': '🎤',
        'tokens': ['listen_start', 'listen_end', 'speak_start', 'speak_end', 'audio_start', 'audio_end', 'audio_prompt_start', 'audio_prompt_end', 'speaker_ref_start', 'speaker_ref_end'],
        'description': 'Voice I/O and audio prompting for zero-shot cloning'
    },
    'Structured Data': {
        'color': COLORS['cat_data'],
        'icon': '📊',
        'tokens': ['table_start', 'table_end', 'table_row_start', 'table_cell_start', 'schema_start', 'json_start', 'yaml_start', 'list_start', 'list_item', 'kv_start', 'key_start', 'value_start'],
        'description': 'Tables, schemas, and data formats'
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
    
    # Calculate layout
    margin = 60
    card_width = (WIDTH - 3 * margin) // 2
    card_gap = 30
    
    svg_parts = [f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {HEIGHT}" width="{WIDTH}" height="{HEIGHT}">
  <defs>
    <filter id="shadow" x="-20%" y="-20%" width="140%" height="140%">
      <feDropShadow dx="0" dy="4" stdDeviation="8" flood-color="#000000" flood-opacity="0.5"/>
    </filter>
    <linearGradient id="titleGrad" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#3b82f6"/>
      <stop offset="50%" stop-color="#8b5cf6"/>
      <stop offset="100%" stop-color="#ec4899"/>
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
  <text x="{WIDTH//2}" y="60" text-anchor="middle" fill="url(#titleGrad)" font-family="Inter, -apple-system, sans-serif" font-size="42" font-weight="bold">
    🏷️ XORON-DEV SPECIAL TOKENS
  </text>
  <text x="{WIDTH//2}" y="95" text-anchor="middle" fill="{COLORS['text_secondary']}" font-family="Inter, -apple-system, sans-serif" font-size="18">
    400+ Structured Tokens for Multimodal AI
  </text>
  
  <!-- Stats bar -->
  <rect x="{margin}" y="120" width="{WIDTH - 2*margin}" height="50" rx="10" fill="{COLORS['bg_card']}" stroke="{COLORS['border']}" stroke-width="1"/>
  <text x="{margin + 30}" y="152" fill="{COLORS['text_primary']}" font-family="Inter, -apple-system, sans-serif" font-size="14" font-weight="bold">
    📊 Total Tokens: {len(SPECIAL_TOKENS)}
  </text>
  <text x="{margin + 250}" y="152" fill="{COLORS['text_secondary']}" font-family="Inter, -apple-system, sans-serif" font-size="14">
    |  🗂️ Categories: {len(TOKEN_CATEGORIES)}
  </text>
  <text x="{margin + 480}" y="152" fill="{COLORS['text_secondary']}" font-family="Inter, -apple-system, sans-serif" font-size="14">
    |  🎯 Weighted Loss: CoT 1.5x, Tools 1.3x, Anti-Hallucination 1.2x
  </text>
''']

    # Draw category cards - track y position for each column separately
    col_y = [190, 190]  # Starting y for each column
    col = 0
    
    for cat_name, cat_info in TOKEN_CATEGORIES.items():
        x = margin + col * (card_width + card_gap)
        y_offset = col_y[col]  # Use this column's current y position
        
        # Calculate card height based on tokens (2 per row for wider display)
        num_tokens = len(cat_info['tokens'])
        tokens_per_row = 2
        token_rows = (num_tokens + tokens_per_row - 1) // tokens_per_row
        card_height = 100 + token_rows * 35
        
        # Card background - escape all text content
        cat_name_escaped = escape_xml(cat_name)
        desc_escaped = escape_xml(cat_info['description'])
        
        svg_parts.append(f'''
  <!-- {cat_name_escaped} Card -->
  <rect x="{x}" y="{y_offset}" width="{card_width}" height="{card_height}" rx="12" fill="{COLORS['bg_card']}" stroke="{cat_info['color']}" stroke-width="2" filter="url(#shadow)"/>
  
  <!-- Header gradient overlay -->
  <rect x="{x + 1}" y="{y_offset + 1}" width="{card_width - 2}" height="44" rx="11" fill="{cat_info['color']}" opacity="0.15"/>
  <text x="{x + 20}" y="{y_offset + 30}" fill="{COLORS['text_primary']}" font-family="Inter, -apple-system, sans-serif" font-size="16" font-weight="bold">{cat_info['icon']} {cat_name_escaped}</text>
  
  <!-- Description -->
  <text x="{x + 20}" y="{y_offset + 62}" fill="{COLORS['text_muted']}" font-family="Inter, -apple-system, sans-serif" font-size="11">{desc_escaped}</text>
''')
        
        # Draw tokens - use 2 columns for wider token boxes
        token_y = y_offset + 80
        token_x = x + 15
        tokens_per_row = 2  # Reduced from 3 to 2 for wider boxes
        token_width = (card_width - 50) // tokens_per_row
        
        for i, token_key in enumerate(cat_info['tokens']):
            if token_key in SPECIAL_TOKENS:
                token_val = SPECIAL_TOKENS[token_key]
                # No truncation - show full token name
                display_token = escape_xml(token_val)
                
                tx = token_x + (i % tokens_per_row) * token_width
                ty = token_y + (i // tokens_per_row) * 35
                
                svg_parts.append(f'''
  <rect x="{tx}" y="{ty}" width="{token_width - 10}" height="28" rx="6" fill="{COLORS['bg_token']}" stroke="{cat_info['color']}" stroke-width="1" opacity="0.8"/>
  <text x="{tx + (token_width - 10)//2}" y="{ty + 18}" text-anchor="middle" fill="{COLORS['text_token']}" font-family="monospace" font-size="9">{display_token}</text>''')
        
        # Update this column's y position
        col_y[col] += card_height + card_gap
        
        # Move to next column
        col = (col + 1) % 2
    
    # Footer
    svg_parts.append(f'''
  <!-- Footer -->
  <text x="{WIDTH//2}" y="{HEIGHT - 30}" text-anchor="middle" fill="{COLORS['text_muted']}" font-family="Inter, -apple-system, sans-serif" font-size="12">
    Xoron-Dev Special Tokens • Structured Output for Multimodal AI • github.com/nigfuapp-web/Xoron-Dev
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
    output_path = os.path.join(assets_dir, 'special_tokens.svg')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(svg_content)
    
    print(f"✅ Generated special tokens SVG: {output_path}")
    print(f"   Dimensions: {WIDTH}x{HEIGHT}")
    print(f"   Categories: {len(TOKEN_CATEGORIES)}")
    print(f"   Total tokens shown: {sum(len(c['tokens']) for c in TOKEN_CATEGORIES.values())}")


if __name__ == '__main__':
    main()
