"""
Quality utilities for synthetic data generation.

Provides shared functions for generating high-quality, diverse training data
with proper use of all special tokens.
"""

import random
from typing import Dict, List, Any, Optional, Tuple
from config.special_tokens import SPECIAL_TOKENS


class QualityGenerator:
    """Base class with quality generation utilities."""
    
    def __init__(self, tokens: Dict[str, str] = None, seed: int = None):
        self.t = tokens or SPECIAL_TOKENS
        self.rng = random.Random(seed)
        
    def _wrap_sequence(self, text: str) -> str:
        """Wrap with BOS/EOS."""
        return f"{self.t['bos']}{text}{self.t['eos']}"
    
    def _format_qa(self, user: str, assistant: str, system: str = None) -> str:
        """Format as conversation."""
        parts = []
        if system:
            parts.append(f"{self.t['system_start']}\n{system}\n{self.t['system_end']}")
        parts.append(f"{self.t['user_start']}\n{user}\n{self.t['user_end']}")
        parts.append(f"{self.t['assistant_start']}\n{assistant}\n{self.t['assistant_end']}")
        return "\n".join(parts)
    
    # === REASONING TOKENS ===
    
    def wrap_think(self, content: str) -> str:
        return f"{self.t['think_start']}\n{content}\n{self.t['think_end']}"
    
    def wrap_plan(self, steps: List[str]) -> str:
        """Create a planning section with steps."""
        step_text = "\n".join([
            f"{self.t.get('plan_step', '')}{s}{self.t.get('plan_step_end', '')}" 
            for s in steps
        ])
        return f"{self.t.get('plan_start', '')}\n{step_text}\n{self.t.get('plan_end', '')}"
    
    def wrap_analysis(self, content: str) -> str:
        return f"{self.t.get('analysis_start', '')}{content}{self.t.get('analysis_end', '')}"
    
    def wrap_observation(self, content: str) -> str:
        return f"{self.t['observation_start']}{content}{self.t['observation_end']}"
    
    def wrap_step(self, content: str) -> str:
        return f"{self.t['step_start']}{content}{self.t['step_end']}"
    
    def wrap_note(self, content: str) -> str:
        return f"{self.t['note_start']}{content}{self.t['note_end']}"
    
    def wrap_reflection(self, content: str) -> str:
        return f"{self.t['reflection_start']}{content}{self.t['reflection_end']}"
    
    def wrap_conclusion(self, content: str) -> str:
        return f"{self.t['conclusion_start']}{content}{self.t['conclusion_end']}"
    
    def wrap_critique(self, content: str, has_error: bool = False) -> str:
        """Self-critique section."""
        marker = self.t.get('error_found', '') if has_error else self.t.get('no_error', '')
        return f"{self.t.get('critique_start', '')}\n{marker}{content}\n{self.t.get('critique_end', '')}"
    
    def wrap_decision(self, options: List[str], chosen_idx: int, reason: str) -> str:
        """Decision with options."""
        parts = [self.t.get('decision_start', '')]
        for i, opt in enumerate(options):
            marker = self.t.get('chosen', '') if i == chosen_idx else self.t.get('rejected', '')
            parts.append(f"{self.t.get('option_start', '')}{marker}{opt}{self.t.get('option_end', '')}")
        parts.append(f"{self.t.get('because', '')}{reason}")
        parts.append(self.t.get('decision_end', ''))
        return "\n".join(parts)
    
    # === UNCERTAINTY TOKENS ===
    
    def add_uncertainty_score(self, score: int) -> str:
        """Add uncertainty score (0-100)."""
        score = max(0, min(100, score))
        return f"{self.t.get('uncertainty_score', '')}{score}{self.t.get('uncertainty_score_end', '')}"
    
    def add_confidence(self, level: str = 'high') -> str:
        """Add confidence level."""
        key = f"confidence_{level}"
        return self.t.get(key, '')
    
    def wrap_uncertain(self, content: str) -> str:
        return f"{self.t['uncertain_start']}{content}{self.t['uncertain_end']}"
    
    # === MEMORY TOKENS ===
    
    def wrap_memory(self, content: str, memory_type: str = 'memory') -> str:
        """Wrap with memory tokens."""
        start = self.t.get(f'{memory_type}_start', '')
        end = self.t.get(f'{memory_type}_end', '')
        return f"{start}\n{content}\n{end}"
    
    def wrap_summary(self, content: str) -> str:
        return f"{self.t.get('summary_start', '')}\n{content}\n{self.t.get('summary_end', '')}"
    
    def wrap_user_profile(self, prefs: List[str], rules: List[str] = None) -> str:
        """Create user profile section."""
        parts = []
        for p in prefs:
            parts.append(f"{self.t.get('user_preference_start', '')}{p}{self.t.get('user_preference_end', '')}")
        if rules:
            for r in rules:
                parts.append(f"{self.t.get('hard_rule_start', '')}{r}{self.t.get('hard_rule_end', '')}")
        content = "\n".join(parts)
        return f"{self.t.get('user_profile_start', '')}\n{content}\n{self.t.get('user_profile_end', '')}"
    
    # === TEMPORAL TOKENS ===
    
    def wrap_timestamp(self, timestamp: str, content: str = "") -> str:
        ts = f"{self.t.get('timestamp_start', '')}{timestamp}{self.t.get('timestamp_end', '')}"
        return f"{ts}{content}" if content else ts
    
    def mark_keyframe(self, desc: str, frame_num: int = None) -> str:
        frame = ""
        if frame_num is not None:
            frame = f"{self.t.get('frame_num', '')}{frame_num}{self.t.get('frame_num_end', '')}"
        return f"{self.t.get('keyframe', '')}{frame}{desc}{self.t.get('keyframe_end', '')}"
    
    def mark_scene_change(self, from_scene: str = None, to_scene: str = None) -> str:
        parts = [self.t.get('scene_change', '')]
        if from_scene:
            parts.append(f"From: {from_scene}")
        if to_scene:
            parts.append(f"To: {to_scene}")
        return " ".join(parts)
    
    # === STRUCTURED DATA TOKENS ===
    
    def wrap_table(self, headers: List[str], rows: List[List[str]]) -> str:
        """Create a table with proper tokens."""
        parts = [self.t.get('table_start', '')]
        
        # Header
        hcells = "".join([f"{self.t.get('table_cell_start', '')}{h}{self.t.get('table_cell_end', '')}" for h in headers])
        parts.append(f"{self.t.get('table_header_start', '')}{self.t.get('table_row_start', '')}{hcells}{self.t.get('table_row_end', '')}{self.t.get('table_header_end', '')}")
        
        # Body
        parts.append(self.t.get('table_body_start', ''))
        for row in rows:
            cells = "".join([f"{self.t.get('table_cell_start', '')}{c}{self.t.get('table_cell_end', '')}" for c in row])
            parts.append(f"{self.t.get('table_row_start', '')}{cells}{self.t.get('table_row_end', '')}")
        parts.append(self.t.get('table_body_end', ''))
        parts.append(self.t.get('table_end', ''))
        
        return "\n".join(parts)
    
    def wrap_schema(self, schema: str, schema_type: str = None) -> str:
        type_marker = ""
        if schema_type:
            type_marker = f"{self.t.get('schema_type', '')}{schema_type}{self.t.get('schema_type_end', '')}"
        return f"{self.t.get('schema_start', '')}{type_marker}\n{schema}\n{self.t.get('schema_end', '')}"
    
    def wrap_json(self, content: str) -> str:
        return f"{self.t.get('json_start', '')}\n{content}\n{self.t.get('json_end', '')}"
    
    def add_version(self, version: str) -> str:
        return f"{self.t.get('version', '')}{version}{self.t.get('version_end', '')}"
    
    # === CITATION TOKENS ===
    
    def wrap_citation(self, source: str, quote: str = None) -> str:
        parts = [f"{self.t.get('source_start', '')}{source}{self.t.get('source_end', '')}"]
        if quote:
            parts.append(f"{self.t.get('quote_start', '')}{quote}{self.t.get('quote_end', '')}")
        return f"{self.t.get('cite_start', '')}\n" + "\n".join(parts) + f"\n{self.t.get('cite_end', '')}"
    
    def wrap_retrieved(self, context: str, grounded: bool = True) -> str:
        marker = self.t.get('grounded', '') if grounded else self.t.get('ungrounded', '')
        return f"{self.t.get('retrieved_start', '')}\n{context}\n{self.t.get('retrieved_end', '')}{marker}"
    
    # === CODE TOKENS ===
    
    def wrap_code(self, code: str, lang: str = None) -> str:
        lang_token = self.t.get(f'lang_{lang}', '') if lang else ''
        return f"{self.t['code_start']}{lang_token}\n{code}\n{self.t['code_end']}"
    
    # === TASK TOKENS ===
    
    def wrap_task(self, task_type: str, instruction: str, 
                  constraints: List[str] = None, examples: List[Dict] = None) -> str:
        """Format a task with type, instruction, constraints, examples."""
        parts = [self.t.get('task_start', '')]
        parts.append(f"{self.t.get('task_type', '')}{task_type}{self.t.get('task_type_end', '')}")
        parts.append(f"{self.t.get('instruction_start', '')}\n{instruction}\n{self.t.get('instruction_end', '')}")
        
        if constraints:
            for c in constraints:
                parts.append(f"{self.t.get('constraint_start', '')}{c}{self.t.get('constraint_end', '')}")
        
        if examples:
            for ex in examples:
                ex_parts = [self.t.get('example_start', '')]
                if 'input' in ex:
                    ex_parts.append(f"{self.t.get('input_start', '')}{ex['input']}{self.t.get('input_end', '')}")
                if 'output' in ex:
                    ex_parts.append(f"{self.t.get('output_start', '')}{ex['output']}{self.t.get('output_end', '')}")
                ex_parts.append(self.t.get('example_end', ''))
                parts.append("\n".join(ex_parts))
        
        parts.append(self.t.get('task_end', ''))
        return "\n".join(parts)
    
    # === DIVERSITY HELPERS ===
    
    def vary_phrasing(self, templates: List[str], **kwargs) -> str:
        """Select random template and fill in values."""
        template = self.rng.choice(templates)
        return template.format(**kwargs)
    
    def shuffle_and_pick(self, items: List[Any], count: int = 1) -> List[Any]:
        """Randomly pick items without replacement."""
        return self.rng.sample(items, min(count, len(items)))


# === DIVERSE DATA POOLS ===

NAMES = [
    "Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Henry",
    "Iris", "Jack", "Kate", "Leo", "Maya", "Noah", "Olivia", "Peter",
    "Quinn", "Rachel", "Sam", "Tara", "Uma", "Victor", "Wendy", "Xavier",
    "Yuki", "Zara", "Ahmed", "Bella", "Carlos", "Diya", "Elena", "Fatima",
    "George", "Hannah", "Ivan", "Julia", "Kevin", "Luna", "Marcus", "Nina",
    "Oscar", "Priya", "Raj", "Sofia", "Thomas", "Ursula", "Wei", "Xena",
]

COMPANIES = [
    "TechCorp", "DataSoft", "CloudNine", "ByteWorks", "CodeCraft",
    "NetSphere", "PixelPerfect", "QuantumLeap", "StreamLine", "SynergyTech",
    "InnovateLabs", "FutureTech", "SmartSolutions", "DigitalDynamics", "CyberCore",
]

PRODUCTS = [
    "laptop", "smartphone", "tablet", "headphones", "smartwatch",
    "camera", "speaker", "monitor", "keyboard", "mouse",
    "printer", "router", "hard drive", "USB drive", "charger",
]

PROGRAMMING_LANGUAGES = [
    ("python", "Python"), ("javascript", "JavaScript"), ("typescript", "TypeScript"),
    ("java", "Java"), ("cpp", "C++"), ("go", "Go"), ("rust", "Rust"),
    ("ruby", "Ruby"), ("php", "PHP"), ("swift", "Swift"), ("kotlin", "Kotlin"),
]

FRAMEWORKS = [
    ("React", "javascript"), ("Vue", "javascript"), ("Angular", "typescript"),
    ("Django", "python"), ("Flask", "python"), ("FastAPI", "python"),
    ("Express", "javascript"), ("Spring", "java"), ("Rails", "ruby"),
    ("Gin", "go"), ("Actix", "rust"), ("Laravel", "php"),
]

ERROR_TYPES = [
    ("TypeError", "type mismatch or invalid operation on a type"),
    ("ValueError", "invalid value passed to a function"),
    ("IndexError", "list index out of range"),
    ("KeyError", "dictionary key not found"),
    ("AttributeError", "object has no such attribute"),
    ("NameError", "variable not defined"),
    ("ZeroDivisionError", "division by zero"),
    ("FileNotFoundError", "file does not exist"),
    ("ImportError", "module not found"),
    ("SyntaxError", "invalid syntax"),
]

TOPICS = [
    "machine learning", "web development", "database design", "API development",
    "cloud computing", "cybersecurity", "data analysis", "mobile development",
    "DevOps", "microservices", "testing", "performance optimization",
    "system design", "algorithms", "data structures", "networking",
]
