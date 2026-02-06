"""Dataset formatters for multimodal training."""

import json
import re
import random
from typing import Dict, Optional, Any, List, Union


class MultimodalFormatter:
    """
    Comprehensive formatter for all dataset types in multimodal training.
    
    Handles various dataset formats and normalizes them to a consistent
    format for training. Designed to be extensible for future datasets.
    
    Token structure (ALWAYS used):
    - <|bos|> ... <|eos|> wraps entire sequence
    - <|system|>, <|user|>, <|assistant|> for conversation roles
    - Audio prompting tokens for zero-shot voice cloning
    """

    def __init__(self, tokens: Dict[str, str], image_processor=None):
        self.t = tokens
        self.image_processor = image_processor
        
        # Emotion list for TTS/voice synthesis
        self.emotions = [
            'neutral', 'happy', 'sad', 'angry', 'surprised',
            'fearful', 'disgusted', 'excited', 'calm', 'whisper',
            'amused', 'confused', 'curious'
        ]
        
        # Prosody options for TTS
        self.prosody_options = {
            'speed': ['fast', 'slow', 'normal_speed'],
            'volume': ['loud', 'soft', 'normal_volume'],
            'pitch': ['high_pitch', 'low_pitch', 'normal_pitch'],
        }
    
    def _wrap_sequence(self, text: str) -> str:
        """Wrap text with BOS/EOS tokens. ALWAYS applied."""
        return f"{self.t['bos']}{text}{self.t['eos']}"
    
    def _format_simple_qa(self, user_content: str, assistant_content: str, 
                          system_content: str = None, wrap: bool = True) -> str:
        """Format a simple Q&A pair with proper tokens."""
        parts = []
        if system_content:
            parts.append(f"{self.t['system_start']}\n{system_content}\n{self.t['system_end']}")
        parts.append(f"{self.t['user_start']}\n{user_content}\n{self.t['user_end']}")
        parts.append(f"{self.t['assistant_start']}\n{assistant_content}\n{self.t['assistant_end']}")
        text = "\n".join(parts)
        return self._wrap_sequence(text) if wrap else text
    
    def _wrap_with_audio_prompt(self, text: str, has_speaker_ref: bool = False) -> str:
        """Wrap text with audio prompting tokens for zero-shot voice cloning."""
        if has_speaker_ref:
            return f"{self.t.get('speaker_ref_start', '')}{text}{self.t.get('speaker_ref_end', '')}"
        return f"{self.t.get('audio_prompt_start', '')}{text}{self.t.get('audio_prompt_end', '')}"

    # === TOKEN HELPER METHODS ===
    
    def _wrap_with_planning(self, plan_steps: List[str], content: str) -> str:
        """Wrap content with planning tokens showing the steps before execution."""
        plan_text = ""
        if plan_steps:
            steps = "\n".join([f"{self.t.get('plan_step', '')}{step}{self.t.get('plan_step_end', '')}" 
                              for step in plan_steps])
            plan_text = f"{self.t.get('plan_start', '')}\n{steps}\n{self.t.get('plan_end', '')}\n"
        return f"{plan_text}{content}"
    
    def _wrap_with_thinking(self, thinking: str, conclusion: str = None) -> str:
        """Wrap content with thinking/reasoning tokens."""
        parts = [f"{self.t['think_start']}\n{thinking}\n{self.t['think_end']}"]
        if conclusion:
            parts.append(f"{self.t.get('conclusion_start', '')}\n{conclusion}\n{self.t.get('conclusion_end', '')}")
        return "\n".join(parts)
    
    def _wrap_with_critique(self, content: str, critique: str, has_error: bool = False) -> str:
        """Add self-critique/correction to content."""
        error_marker = self.t.get('error_found', '') if has_error else self.t.get('no_error', '')
        critique_text = f"{self.t.get('critique_start', '')}\n{error_marker}{critique}\n{self.t.get('critique_end', '')}"
        return f"{content}\n{critique_text}"
    
    def _add_uncertainty_score(self, content: str, score: int) -> str:
        """Add uncertainty score (0-100) to content."""
        score = max(0, min(100, score))  # Clamp to 0-100
        return f"{self.t.get('uncertainty_score', '')}{score}{self.t.get('uncertainty_score_end', '')}{content}"
    
    def _add_confidence_level(self, content: str, level: str = 'medium') -> str:
        """Add confidence level token to content."""
        level_key = f"confidence_{level}"
        if level_key in self.t:
            return f"{self.t[level_key]}{content}"
        return content
    
    def _wrap_with_memory(self, content: str, memory_type: str = 'memory') -> str:
        """Wrap content with memory tokens."""
        start_key = f"{memory_type}_start"
        end_key = f"{memory_type}_end"
        if start_key in self.t and end_key in self.t:
            return f"{self.t[start_key]}\n{content}\n{self.t[end_key]}"
        return content
    
    def _wrap_with_summary(self, content: str) -> str:
        """Wrap content with summary tokens."""
        return f"{self.t.get('summary_start', '')}\n{content}\n{self.t.get('summary_end', '')}"
    
    def _wrap_with_user_profile(self, preferences: List[str], hard_rules: List[str] = None) -> str:
        """Create user profile section with preferences and hard rules."""
        parts = []
        if preferences:
            prefs = "\n".join([f"{self.t.get('user_preference_start', '')}{p}{self.t.get('user_preference_end', '')}" 
                             for p in preferences])
            parts.append(prefs)
        if hard_rules:
            rules = "\n".join([f"{self.t.get('hard_rule_start', '')}{r}{self.t.get('hard_rule_end', '')}" 
                             for r in hard_rules])
            parts.append(rules)
        content = "\n".join(parts)
        return f"{self.t.get('user_profile_start', '')}\n{content}\n{self.t.get('user_profile_end', '')}"
    
    def _wrap_with_timestamp(self, content: str, timestamp: str) -> str:
        """Wrap content with timestamp for temporal grounding."""
        return f"{self.t.get('timestamp_start', '')}{timestamp}{self.t.get('timestamp_end', '')}{content}"
    
    def _mark_keyframe(self, description: str, frame_num: int = None) -> str:
        """Mark a keyframe in video content."""
        frame_marker = ""
        if frame_num is not None:
            frame_marker = f"{self.t.get('frame_num', '')}{frame_num}{self.t.get('frame_num_end', '')}"
        return f"{self.t.get('keyframe', '')}{frame_marker}{description}{self.t.get('keyframe_end', '')}"
    
    def _mark_scene_change(self, from_scene: str = None, to_scene: str = None) -> str:
        """Mark a scene change in video content."""
        parts = [self.t.get('scene_change', '')]
        if from_scene:
            parts.append(f"From: {from_scene}")
        if to_scene:
            parts.append(f"To: {to_scene}")
        return " ".join(parts)
    
    # === EMOTION AND PROSODY METHODS ===
    
    def _add_emotion_token(self, text: str, emotion: Optional[str]) -> str:
        """Add emotion token prefix to text for TTS/voice synthesis."""
        if emotion is None:
            return text
        token_key = f"emotion_{emotion}"
        if token_key in self.t:
            return f"{self.t[token_key]}{text}"
        return text
    
    def _get_random_emotion(self) -> str:
        """Get a random emotion from the emotions list."""
        return random.choice(self.emotions)
    
    def _add_prosody_tokens(self, text: str, speed: str = None, 
                           volume: str = None, pitch: str = None) -> str:
        """Add prosody tokens (speed, volume, pitch) to text for TTS."""
        prefixes = []
        if speed and f"prosody_{speed}" in self.t:
            prefixes.append(self.t[f"prosody_{speed}"])
        if volume and f"prosody_{volume}" in self.t:
            prefixes.append(self.t[f"prosody_{volume}"])
        if pitch and f"prosody_{pitch}" in self.t:
            prefixes.append(self.t[f"prosody_{pitch}"])
        prefix = "".join(prefixes)
        return f"{prefix}{text}" if prefix else text
    
    def _get_random_prosody(self) -> Dict[str, str]:
        """Get random prosody settings for TTS."""
        return {
            'speed': random.choice(self.prosody_options['speed']),
            'volume': random.choice(self.prosody_options['volume']),
            'pitch': random.choice(self.prosody_options['pitch']),
        }
    
    def _wrap_with_table(self, headers: List[str], rows: List[List[str]]) -> str:
        """Format data as a table with proper tokens."""
        parts = [self.t.get('table_start', '')]
        
        # Header
        header_cells = "".join([f"{self.t.get('table_cell_start', '')}{h}{self.t.get('table_cell_end', '')}" 
                               for h in headers])
        parts.append(f"{self.t.get('table_header_start', '')}{self.t.get('table_row_start', '')}{header_cells}{self.t.get('table_row_end', '')}{self.t.get('table_header_end', '')}")
        
        # Body
        parts.append(self.t.get('table_body_start', ''))
        for row in rows:
            cells = "".join([f"{self.t.get('table_cell_start', '')}{c}{self.t.get('table_cell_end', '')}" 
                           for c in row])
            parts.append(f"{self.t.get('table_row_start', '')}{cells}{self.t.get('table_row_end', '')}")
        parts.append(self.t.get('table_body_end', ''))
        
        parts.append(self.t.get('table_end', ''))
        return "\n".join(parts)
    
    def _wrap_with_schema(self, schema_def: str, schema_type: str = None) -> str:
        """Wrap schema definition with proper tokens."""
        type_marker = ""
        if schema_type:
            type_marker = f"{self.t.get('schema_type', '')}{schema_type}{self.t.get('schema_type_end', '')}"
        return f"{self.t.get('schema_start', '')}{type_marker}\n{schema_def}\n{self.t.get('schema_end', '')}"
    
    def _add_version(self, content: str, version: str) -> str:
        """Add version marker to content."""
        return f"{self.t.get('version', '')}{version}{self.t.get('version_end', '')}{content}"
    
    def _wrap_json(self, json_content: str) -> str:
        """Wrap JSON content with proper tokens."""
        return f"{self.t.get('json_start', '')}\n{json_content}\n{self.t.get('json_end', '')}"
    
    def _wrap_with_task(self, task_type: str, instruction: str, constraints: List[str] = None,
                        examples: List[Dict] = None) -> str:
        """Format a task with type, instruction, constraints, and examples."""
        parts = [self.t.get('task_start', '')]
        
        # Task type
        parts.append(f"{self.t.get('task_type', '')}{task_type}{self.t.get('task_type_end', '')}")
        
        # Instruction
        parts.append(f"{self.t.get('instruction_start', '')}\n{instruction}\n{self.t.get('instruction_end', '')}")
        
        # Constraints
        if constraints:
            for c in constraints:
                parts.append(f"{self.t.get('constraint_start', '')}{c}{self.t.get('constraint_end', '')}")
        
        # Examples
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
    
    def _wrap_with_citation(self, content: str, source: str, quote: str = None) -> str:
        """Add citation/source to content."""
        cite_parts = [f"{self.t.get('source_start', '')}{source}{self.t.get('source_end', '')}"]
        if quote:
            cite_parts.append(f"{self.t.get('quote_start', '')}{quote}{self.t.get('quote_end', '')}")
        citation = f"{self.t.get('cite_start', '')}\n" + "\n".join(cite_parts) + f"\n{self.t.get('cite_end', '')}"
        return f"{content}\n{citation}"
    
    def _wrap_with_retrieved_context(self, context: str, is_grounded: bool = True) -> str:
        """Wrap retrieved context for RAG with grounding marker."""
        grounding = self.t.get('grounded', '') if is_grounded else self.t.get('ungrounded', '')
        return f"{self.t.get('retrieved_start', '')}\n{context}\n{self.t.get('retrieved_end', '')}{grounding}"

    def _format_available_tools(self, tools: List[Dict]) -> str:
        """
        Format a list of available tools for the model to see.
        
        Args:
            tools: List of tool definitions, each with:
                - name: Tool name
                - description: Tool description
                - parameters: List of parameter dicts with name, type, required
        
        Returns:
            Formatted string with available tools tokens
        """
        if not tools:
            return ""
        
        parts = [self.t.get('available_tools_start', '<|available_tools|>')]
        
        for tool in tools:
            tool_parts = [self.t.get('tool_def_start', '<|tool_def|>')]
            
            # Tool name
            name = tool.get('name', tool.get('function', {}).get('name', ''))
            if name:
                tool_parts.append(f"{self.t.get('tool_name', '<|tool_name|>')}{name}{self.t.get('tool_name_end', '<|/tool_name|>')}")
            
            # Tool description
            desc = tool.get('description', tool.get('function', {}).get('description', ''))
            if desc:
                tool_parts.append(f"{self.t.get('tool_description', '<|tool_desc|>')}{desc}{self.t.get('tool_description_end', '<|/tool_desc|>')}")
            
            # Parameters
            params = tool.get('parameters', tool.get('function', {}).get('parameters', {}))
            if params:
                param_parts = [self.t.get('tool_params_start', '<|tool_params|>')]
                
                # Handle OpenAI-style parameters (properties dict)
                properties = params.get('properties', params)
                required = params.get('required', [])
                
                for param_name, param_info in properties.items():
                    if param_name in ['type', 'required']:
                        continue
                    param_type = param_info.get('type', 'string') if isinstance(param_info, dict) else 'string'
                    is_required = param_name in required
                    req_marker = self.t.get('param_required', '<|param_required|>') if is_required else self.t.get('param_optional', '<|param_optional|>')
                    
                    param_parts.append(
                        f"{self.t.get('param_name', '<|param_name|>')}{param_name}{self.t.get('param_name_end', '<|/param_name|>')}"
                        f"{self.t.get('param_type', '<|param_type|>')}{param_type}{self.t.get('param_type_end', '<|/param_type|>')}"
                        f"{req_marker}"
                    )
                
                param_parts.append(self.t.get('tool_params_end', '<|/tool_params|>'))
                tool_parts.append('\n'.join(param_parts))
            
            tool_parts.append(self.t.get('tool_def_end', '<|/tool_def|>'))
            parts.append('\n'.join(tool_parts))
        
        parts.append(self.t.get('available_tools_end', '<|/available_tools|>'))
        return '\n'.join(parts)

    def _safe_get(self, sample: Dict, keys: List[str], default: Any = None) -> Any:
        """Safely get a value from sample using multiple possible keys."""
        for key in keys:
            if key in sample and sample[key]:
                return sample[key]
        return default
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        if not text:
            return ""
        text = str(text).strip()
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        return text
    
    def _detect_language(self, code: str, filename: str = None, language_hint: str = None) -> str:
        """
        Detect programming language from code content, filename, or hint.
        Returns the language key (e.g., 'python', 'javascript') or 'other'.
        """
        # If language hint is provided, use it
        if language_hint:
            lang = language_hint.lower().strip()
            lang_map = {
                'py': 'python', 'python': 'python', 'python3': 'python',
                'js': 'javascript', 'javascript': 'javascript', 'node': 'javascript',
                'ts': 'typescript', 'typescript': 'typescript',
                'java': 'java',
                'cpp': 'cpp', 'c++': 'cpp', 'cxx': 'cpp',
                'c': 'c',
                'cs': 'csharp', 'csharp': 'csharp', 'c#': 'csharp',
                'go': 'go', 'golang': 'go',
                'rs': 'rust', 'rust': 'rust',
                'rb': 'ruby', 'ruby': 'ruby',
                'php': 'php',
                'swift': 'swift',
                'kt': 'kotlin', 'kotlin': 'kotlin',
                'scala': 'scala',
                'sh': 'shell', 'shell': 'shell', 'bash': 'bash', 'zsh': 'shell',
                'sql': 'sql',
                'r': 'r',
                'matlab': 'matlab', 'm': 'matlab',
                'lua': 'lua',
                'pl': 'perl', 'perl': 'perl',
                'hs': 'haskell', 'haskell': 'haskell',
            }
            if lang in lang_map:
                return lang_map[lang]
        
        # Try to detect from filename extension
        if filename:
            ext_map = {
                '.py': 'python', '.pyw': 'python',
                '.js': 'javascript', '.mjs': 'javascript', '.cjs': 'javascript',
                '.ts': 'typescript', '.tsx': 'typescript',
                '.java': 'java',
                '.cpp': 'cpp', '.cc': 'cpp', '.cxx': 'cpp', '.hpp': 'cpp', '.h': 'c',
                '.c': 'c',
                '.cs': 'csharp',
                '.go': 'go',
                '.rs': 'rust',
                '.rb': 'ruby',
                '.php': 'php',
                '.swift': 'swift',
                '.kt': 'kotlin', '.kts': 'kotlin',
                '.scala': 'scala',
                '.sh': 'shell', '.bash': 'bash', '.zsh': 'shell',
                '.sql': 'sql',
                '.r': 'r', '.R': 'r',
                '.m': 'matlab',
                '.lua': 'lua',
                '.pl': 'perl', '.pm': 'perl',
                '.hs': 'haskell',
            }
            for ext, lang in ext_map.items():
                if filename.endswith(ext):
                    return lang
        
        # Try to detect from code content patterns
        if code:
            code_lower = code.lower()[:500]  # Check first 500 chars
            patterns = [
                (r'def\s+\w+\s*\(|import\s+\w+|from\s+\w+\s+import', 'python'),
                (r'function\s+\w+\s*\(|const\s+\w+\s*=|let\s+\w+\s*=|var\s+\w+\s*=', 'javascript'),
                (r'interface\s+\w+|type\s+\w+\s*=', 'typescript'),
                (r'public\s+class|private\s+void|System\.out\.println', 'java'),
                (r'#include\s*<|std::|cout\s*<<|cin\s*>>', 'cpp'),
                (r'func\s+\w+\s*\(|package\s+main|fmt\.Println', 'go'),
                (r'fn\s+\w+\s*\(|let\s+mut|impl\s+\w+', 'rust'),
                (r'<?php|echo\s+|function\s+\w+\s*\(.*\)\s*{', 'php'),
                (r'SELECT\s+|INSERT\s+INTO|CREATE\s+TABLE', 'sql'),
            ]
            for pattern, lang in patterns:
                if re.search(pattern, code, re.IGNORECASE):
                    return lang
        
        return 'other'
    
    def _get_language_token(self, language: str) -> str:
        """Get the language token for a detected language."""
        lang_key = f"lang_{language}"
        if lang_key in self.t:
            return self.t[lang_key]
        return self.t.get('lang_other', '')
    
    def _get_file_type_tokens(self, file_type: str) -> tuple:
        """Get start and end tokens for a file type."""
        file_type = file_type.lower().strip().lstrip('.')
        start_key = f"file_{file_type}"
        end_key = f"file_{file_type}_end"
        
        if start_key in self.t and end_key in self.t:
            return self.t[start_key], self.t[end_key]
        
        # Fallback to generic document tokens
        return self.t.get('doc_start', ''), self.t.get('doc_end', '')
    
    def _wrap_document(self, content: str, file_type: str = None, filename: str = None) -> str:
        """
        Wrap document content with appropriate file type tokens.
        
        Args:
            content: The document content
            file_type: File type (e.g., 'txt', 'md', 'json')
            filename: Optional filename for context
        
        Returns:
            Wrapped content with file type tokens
        """
        start_token, end_token = self._get_file_type_tokens(file_type) if file_type else ('', '')
        
        parts = []
        if filename and 'filename_start' in self.t:
            parts.append(f"{self.t['filename_start']}{filename}{self.t['filename_end']}")
        
        if start_token:
            parts.append(f"{start_token}\n{content}\n{end_token}")
        else:
            # Use generic document tokens
            parts.append(f"{self.t.get('doc_start', '')}\n{content}\n{self.t.get('doc_end', '')}")
        
        return "\n".join(parts)
    
    def _format_conversation(self, messages: List[Dict], is_code: bool = False, 
                            add_emotion: bool = False, add_prosody: bool = False,
                            language_hint: str = None) -> Optional[str]:
        """Format a list of messages into conversation format with proper tokens."""
        if not messages or not isinstance(messages, list):
            return None
        
        parts = []
        for msg in messages:
            role = msg.get("role", msg.get("from", "")).lower()
            content = msg.get("content", msg.get("value", msg.get("text", "")))
            
            if not content:
                continue
            
            if role in ["user", "human", "prompter"]:
                parts.append(f"{self.t['user_start']}\n{content}\n{self.t['user_end']}")
            elif role in ["assistant", "gpt", "bot", "model"]:
                # Optionally add emotion/prosody for TTS training
                assistant_content = content
                if add_emotion:
                    assistant_content = self._add_emotion_token(assistant_content, self._get_random_emotion())
                if add_prosody:
                    prosody = self._get_random_prosody()
                    assistant_content = self._add_prosody_tokens(assistant_content, **prosody)
                
                if is_code:
                    # Detect language and use language token
                    lang = language_hint or self._detect_language(assistant_content)
                    lang_token = self._get_language_token(lang)
                    if lang_token:
                        parts.append(f"{self.t['assistant_start']}\n{self.t['code_start']}{lang_token}\n{assistant_content}\n{self.t['code_end']}\n{self.t['assistant_end']}")
                    else:
                        parts.append(f"{self.t['assistant_start']}\n{self.t['code_start']}\n{assistant_content}\n{self.t['code_end']}\n{self.t['assistant_end']}")
                else:
                    parts.append(f"{self.t['assistant_start']}\n{assistant_content}\n{self.t['assistant_end']}")
            elif role in ["system"]:
                parts.append(f"{self.t['system_start']}\n{content}\n{self.t['system_end']}")
            elif role in ["function", "tool"]:
                parts.append(f"{self.t['tool_result_start']}\n{content}\n{self.t['tool_result_end']}")
        
        if parts:
            # Always wrap with BOS/EOS
            return self._wrap_sequence("\n".join(parts))
        return None

    def format_generic(self, sample: Dict, sample_type: str = "text") -> Optional[Dict]:
        """
        Generic formatter that tries to handle any dataset format.
        Use this as a fallback for unknown dataset formats.
        """
        try:
            # Try conversation format first
            messages = self._safe_get(sample, ['messages', 'conversations', 'conversation', 'dialog', 'dialogue'])
            if messages and isinstance(messages, list):
                text = self._format_conversation(messages)
                if text:
                    return {"text": text, "type": sample_type}
            
            # Try instruction/output format
            instruction = self._safe_get(sample, ['instruction', 'prompt', 'query', 'question', 'input_text'])
            output = self._safe_get(sample, ['output', 'response', 'answer', 'completion', 'target', 'output_text'])
            
            if instruction and output:
                text = self._format_simple_qa(instruction, output)
                return {"text": text, "type": sample_type}
            
            # Try text-only format - wrap with BOS/EOS
            text_content = self._safe_get(sample, ['text', 'content', 'document', 'passage'])
            if text_content and len(str(text_content)) > 50:
                return {"text": self._wrap_sequence(str(text_content)), "type": sample_type}
            
        except Exception:
            pass
        return None

    def format_passthrough_sample(self, sample: Dict) -> Optional[Dict]:
        """
        Passthrough formatter for pre-formatted samples.
        
        Used for synthetic datasets that are already formatted with special tokens
        (e.g., FIM, git operations, code execution, file operations).
        """
        try:
            text = sample.get("text", "")
            if text and len(text) > 10:
                sample_type = sample.get("type", "passthrough")
                return {"text": text, "type": sample_type}
        except Exception:
            pass
        return None

    def _format_code_with_language(self, code: str, language: str = None, filename: str = None) -> str:
        """Format code with language token and code block markers."""
        # Detect language if not provided
        if not language:
            language = self._detect_language(code, filename)
        
        lang_token = self._get_language_token(language)
        
        # Format: <|code|><|lang:python|>\ncode\n<|/code|>
        if lang_token:
            return f"{self.t['code_start']}{lang_token}\n{code}\n{self.t['code_end']}"
        else:
            return f"{self.t['code_start']}\n{code}\n{self.t['code_end']}"

    def format_code_sample(self, sample: Dict) -> Optional[Dict]:
        """Handle multiple code dataset formats with language detection."""
        try:
            # Get language hint if available - check target_language for Swift-Code-Edit style datasets
            language_hint = self._safe_get(sample, ['language', 'lang', 'programming_language', 'target_language'])
            filename = self._safe_get(sample, ['filename', 'file_name', 'path'])
            
            # Swift-Code-Edit format: translated_problem + translated_solution + messages
            # This dataset has problem description in translated_problem and solution in translated_solution
            if "translated_problem" in sample and "translated_solution" in sample:
                problem = sample["translated_problem"]
                solution = sample["translated_solution"]
                target_lang = sample.get("target_language", "swift").lower()
                
                # Clean up the solution - remove markdown code blocks if present
                solution_clean = solution
                if solution_clean.startswith("```"):
                    lines = solution_clean.split("\n")
                    # Skip first line (```lang) and last line (```)
                    if len(lines) > 2:
                        solution_clean = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
                
                user_content = problem
                assistant = self._format_code_with_language(solution_clean, target_lang, filename)
                text = self._format_simple_qa(user_content, assistant)
                return {"text": text, "type": "code", "language": target_lang}
            
            # Try messages/conversations format first
            messages = self._safe_get(sample, ['messages', 'conversations'])
            if messages and isinstance(messages, list):
                text = self._format_conversation(messages, is_code=True)
                if text:
                    return {"text": text, "type": "code", "language": language_hint}

            # Instruction + output format
            instruction = self._safe_get(sample, ['instruction', 'prompt', 'query'])
            output = self._safe_get(sample, ['output', 'response', 'answer', 'completion'])
            
            if instruction and output:
                inp = sample.get("input", "")
                user = f"{instruction}\n\nInput:\n{inp}" if inp else instruction
                # Use language-aware code formatting
                assistant = self._format_code_with_language(output, language_hint, filename)
                text = self._format_simple_qa(user, assistant)
                return {"text": text, "type": "code", "language": language_hint or self._detect_language(output, filename)}

            # HumanEval format - detect language from task_id or use language_hint
            if "instruction" in sample and "canonical_solution" in sample:
                inst = sample["instruction"]
                solution = sample["canonical_solution"]
                docstring = sample.get("docstring", "")
                prompt = sample.get("prompt", "")
                declaration = sample.get("declaration", "")
                
                # Detect language from task_id (e.g., "Python/0", "CPP/1", "JavaScript/2", "Rust/3", "Go/4", "Java/5")
                task_id = sample.get("task_id", "")
                detected_lang = language_hint
                if not detected_lang and task_id:
                    lang_prefix = task_id.split("/")[0].lower() if "/" in task_id else ""
                    lang_map = {
                        "python": "python",
                        "cpp": "cpp",
                        "javascript": "javascript", 
                        "js": "javascript",
                        "rust": "rust",
                        "go": "go",
                        "java": "java",
                    }
                    detected_lang = lang_map.get(lang_prefix, "python")
                
                user_content = f"{inst}\n\n{docstring}" if docstring else inst
                # Include declaration for non-Python languages
                if declaration and detected_lang != "python":
                    code_content = f"{declaration}{solution}"
                else:
                    code_content = f"{prompt}{solution}" if prompt else solution
                    
                assistant = self._format_code_with_language(code_content, detected_lang or 'python', filename)
                text = self._format_simple_qa(user_content, assistant)
                return {"text": text, "type": "code", "language": detected_lang or "python"}

            # Raw code content
            content = self._safe_get(sample, ['content', 'code', 'source_code', 'text'])
            if content and len(str(content)) > 100:
                code_str = str(content)[:2000]
                detected_lang = self._detect_language(code_str, filename, language_hint)
                user = "Analyze and explain this code:"
                assistant = self._format_code_with_language(code_str, detected_lang, filename)
                text = self._format_simple_qa(user, assistant)
                return {"text": text, "type": "code", "language": detected_lang}

            # Question/answer format
            question = self._safe_get(sample, ['question', 'query'])
            answer = self._safe_get(sample, ['answer', 'response'])
            if question and answer:
                text = self._format_simple_qa(question, answer)
                return {"text": text, "type": "code"}

        except Exception:
            pass
        return None

    def format_conversation_sample(self, sample: Dict) -> Optional[Dict]:
        """Handle multiple conversation dataset formats."""
        try:
            # Try messages/conversations format first
            messages = self._safe_get(sample, ['messages', 'conversations', 'conversation', 'dialog', 'turns'])
            if messages and isinstance(messages, list):
                text = self._format_conversation(messages)
                if text:
                    return {"text": text, "type": "conversation"}

            # Dialogue + Summary format (conversation summarization datasets)
            dialogue = sample.get("dialogue", "")
            summary = sample.get("summary", "")
            if dialogue and summary:
                text = (
                    f"{self.t['user_start']}\nSummarize this conversation:\n\n{dialogue}\n{self.t['user_end']}\n"
                    f"{self.t['assistant_start']}\n{summary}\n{self.t['assistant_end']}"
                )
                return {"text": self._wrap_sequence(text), "type": "conversation"}

            # Instruction + response format
            instruction = self._safe_get(sample, ['instruction', 'prompt', 'query', 'question', 'input'])
            response = self._safe_get(sample, ['response', 'output', 'answer', 'completion', 'reply'])
            
            if instruction and response:
                ctx = sample.get("context", "")
                system = self._safe_get(sample, ['system_prompt', 'system', 'system_message'])
                
                parts = []
                if system:
                    parts.append(f"{self.t['system_start']}\n{system}\n{self.t['system_end']}")
                
                user = f"{instruction}\n\nContext:\n{ctx}" if ctx else instruction
                parts.append(f"{self.t['user_start']}\n{user}\n{self.t['user_end']}")
                parts.append(f"{self.t['assistant_start']}\n{response}\n{self.t['assistant_end']}")
                return {"text": "\n".join(parts), "type": "conversation"}

            # OpenAssistant format (text + role)
            if "text" in sample and "role" in sample:
                role = sample["role"].lower()
                content = sample["text"]
                if role in ["prompter", "user", "human"]:
                    text = f"{self.t['user_start']}\n{content}\n{self.t['user_end']}"
                else:
                    text = f"{self.t['assistant_start']}\n{content}\n{self.t['assistant_end']}"
                return {"text": self._wrap_sequence(text), "type": "conversation"}

            # ShareGPT format
            if "data" in sample and isinstance(sample["data"], list):
                parts = []
                for i, msg in enumerate(sample["data"]):
                    if isinstance(msg, str):
                        if i % 2 == 0:
                            parts.append(f"{self.t['user_start']}\n{msg}\n{self.t['user_end']}")
                        else:
                            parts.append(f"{self.t['assistant_start']}\n{msg}\n{self.t['assistant_end']}")
                if parts:
                    return {"text": "\n".join(parts), "type": "conversation"}

        except Exception:
            pass
        return None

    def _format_tool_call(self, tool_data: Dict) -> str:
        """
        Format a tool call with structured tokens.
        
        Supports both JSON format and structured token format.
        """
        # Extract function name and arguments
        func_name = tool_data.get("name", tool_data.get("function", {}).get("name", ""))
        func_args = tool_data.get("arguments", tool_data.get("function", {}).get("arguments", {}))
        tool_id = tool_data.get("id", "")
        
        if not func_name:
            # Fallback: try to parse from string content
            content = str(tool_data)
            return f"{self.t['tool_call_start']}\n{content}\n{self.t['tool_call_end']}"
        
        # Build structured tool call
        parts = [self.t['tool_call_start']]
        
        # Add tool ID if present
        if tool_id:
            parts.append(f"{self.t['tool_id_start']}{tool_id}{self.t['tool_id_end']}")
        
        # Add function name
        parts.append(f"{self.t['function_name_start']}{func_name}{self.t['function_name_end']}")
        
        # Add arguments
        if func_args:
            if isinstance(func_args, str):
                # Arguments as JSON string
                parts.append(f"{self.t['function_args_start']}{func_args}{self.t['function_args_end']}")
            elif isinstance(func_args, dict):
                # Arguments as dict - format with structured tokens
                args_parts = []
                for arg_name, arg_value in func_args.items():
                    if isinstance(arg_value, (dict, list)):
                        arg_value_str = json.dumps(arg_value)
                    else:
                        arg_value_str = str(arg_value)
                    args_parts.append(
                        f"{self.t['arg_name_start']}{arg_name}{self.t['arg_name_end']}"
                        f"{self.t['arg_value_start']}{arg_value_str}{self.t['arg_value_end']}"
                    )
                parts.append(f"{self.t['function_args_start']}\n{''.join(args_parts)}\n{self.t['function_args_end']}")
        
        parts.append(self.t['tool_call_end'])
        return "\n".join(parts)

    def _format_tool_result(self, result_data, tool_id: str = "", is_error: bool = False) -> str:
        """Format a tool result with structured tokens."""
        parts = [self.t['tool_result_start']]
        
        if tool_id:
            parts.append(f"{self.t['tool_id_start']}{tool_id}{self.t['tool_id_end']}")
        
        if is_error:
            parts.append(f"{self.t['tool_error_start']}{result_data}{self.t['tool_error_end']}")
        else:
            if isinstance(result_data, (dict, list)):
                result_str = json.dumps(result_data)
            else:
                result_str = str(result_data)
            parts.append(result_str)
        
        parts.append(self.t['tool_result_end'])
        return "\n".join(parts)

    def _format_tools_schema(self, tools: List) -> str:
        """Format available tools/functions schema."""
        parts = [self.t['tools_start']]
        
        for tool in tools:
            # Handle string tools (JSON strings)
            if isinstance(tool, str):
                try:
                    tool = json.loads(tool)
                except:
                    # If it's not valid JSON, skip it
                    continue
            
            # Handle OpenAI-style {"type": "function", "function": {...}}
            if isinstance(tool, dict):
                func_def = tool.get("function", tool)
                parts.append(self.t['function_def_start'])
                parts.append(json.dumps(func_def, indent=2))
                parts.append(self.t['function_def_end'])
        
        parts.append(self.t['tools_end'])
        return "\n".join(parts)

    def format_tool_use_sample(self, sample: Dict) -> Optional[Dict]:
        """
        Handle multiple tool-use dataset formats with structured tokens.
        
        Supports:
        - OpenAI-style function calling
        - Anthropic-style tool use
        - Generic conversation with tool calls
        - Query/answer format
        - Pythonic function calling (driaforall format)
        """
        try:
            parts = []
            
            # Check for tools/functions schema in the sample
            tools = sample.get("tools", sample.get("functions", []))
            # Handle function_description field (Locutusque format)
            if not tools and "function_description" in sample:
                func_desc = sample["function_description"]
                if isinstance(func_desc, str):
                    try:
                        tools = [json.loads(func_desc)]
                    except:
                        tools = []
            
            if tools and isinstance(tools, list) and len(tools) > 0:
                # Check if tools are already formatted as strings (pythonic format)
                if isinstance(tools[0], str) and "def " in tools[0]:
                    # Pythonic function format - include as-is in system message
                    pass  # Will be handled by system message in conversations
                else:
                    parts.append(f"{self.t['system_start']}\nYou have access to the following tools:\n{self._format_tools_schema(tools)}\n{self.t['system_end']}")
            
            # Handle conversation format
            convs = sample.get("conversations", sample.get("messages", []))
            if convs and isinstance(convs, list):
                for c in convs:
                    # Support both from/value and role/content formats
                    role = c.get("from", c.get("role", "")).lower()
                    content = c.get("value", c.get("content", ""))
                    
                    # Check for tool_calls in the message
                    tool_calls = c.get("tool_calls", c.get("function_call", []))
                    if tool_calls and not isinstance(tool_calls, list):
                        tool_calls = [tool_calls]
                    
                    if role in ["human", "user"]:
                        parts.append(f"{self.t['user_start']}\n{content}\n{self.t['user_end']}")
                    
                    elif role in ["gpt", "assistant"]:
                        if tool_calls:
                            # Format with structured tool calls
                            assistant_parts = [self.t['assistant_start']]
                            if content:
                                assistant_parts.append(content)
                            
                            if len(tool_calls) > 1:
                                assistant_parts.append(self.t['tool_calls_start'])
                            
                            for tc in tool_calls:
                                assistant_parts.append(self._format_tool_call(tc))
                            
                            if len(tool_calls) > 1:
                                assistant_parts.append(self.t['tool_calls_end'])
                            
                            assistant_parts.append(self.t['assistant_end'])
                            parts.append("\n".join(assistant_parts))
                        elif "<tool_call>" in content or "```python" in content:
                            # XML-style tool calls (interstellarninja format) or Python code blocks
                            # Keep the content as-is since it's already formatted
                            parts.append(f"{self.t['assistant_start']}\n{content}\n{self.t['assistant_end']}")
                        elif "function" in str(content).lower() or "{" in content:
                            # Legacy format - try to parse as tool call
                            try:
                                parsed = json.loads(content) if isinstance(content, str) else content
                                if isinstance(parsed, dict) and ("name" in parsed or "function" in parsed):
                                    parts.append(f"{self.t['assistant_start']}\n{self._format_tool_call(parsed)}\n{self.t['assistant_end']}")
                                else:
                                    parts.append(f"{self.t['assistant_start']}\n{self.t['tool_call_start']}\n{content}\n{self.t['tool_call_end']}\n{self.t['assistant_end']}")
                            except:
                                parts.append(f"{self.t['assistant_start']}\n{self.t['tool_call_start']}\n{content}\n{self.t['tool_call_end']}\n{self.t['assistant_end']}")
                        else:
                            parts.append(f"{self.t['assistant_start']}\n{content}\n{self.t['assistant_end']}")
                    
                    elif role in ["function", "tool"]:
                        tool_id = c.get("tool_call_id", c.get("id", ""))
                        is_error = "error" in str(content).lower() if isinstance(content, str) else False
                        parts.append(self._format_tool_result(content, tool_id, is_error))
                    
                    elif role == "system":
                        parts.append(f"{self.t['system_start']}\n{content}\n{self.t['system_end']}")
                
                if parts:
                    return {"text": self._wrap_sequence("\n".join(parts)), "type": "tool_use"}

            # Handle query/answers format
            if "query" in sample and "answers" in sample:
                answers = sample["answers"]
                if isinstance(answers, list) and answers:
                    answer = answers[0] if isinstance(answers[0], str) else answers[0]
                else:
                    answer = answers
                
                # Try to parse as structured tool call
                if isinstance(answer, dict):
                    tool_call_str = self._format_tool_call(answer)
                elif isinstance(answer, str):
                    try:
                        parsed = json.loads(answer)
                        if isinstance(parsed, dict):
                            tool_call_str = self._format_tool_call(parsed)
                        else:
                            tool_call_str = f"{self.t['tool_call_start']}\n{answer}\n{self.t['tool_call_end']}"
                    except:
                        tool_call_str = f"{self.t['tool_call_start']}\n{answer}\n{self.t['tool_call_end']}"
                else:
                    tool_call_str = f"{self.t['tool_call_start']}\n{json.dumps(answer)}\n{self.t['tool_call_end']}"
                
                text = (
                    f"{self.t['user_start']}\n{sample['query']}\n{self.t['user_end']}\n"
                    f"{self.t['assistant_start']}\n{tool_call_str}\n{self.t['assistant_end']}"
                )
                return {"text": self._wrap_sequence(text), "type": "tool_use"}

            # Handle instruction/output format with function calls
            if "instruction" in sample and "output" in sample:
                output = sample["output"]
                if isinstance(output, dict) or (isinstance(output, str) and "{" in output):
                    try:
                        parsed = json.loads(output) if isinstance(output, str) else output
                        if isinstance(parsed, dict) and ("name" in parsed or "function" in parsed):
                            tool_call_str = self._format_tool_call(parsed)
                        else:
                            tool_call_str = f"{self.t['tool_call_start']}\n{output}\n{self.t['tool_call_end']}"
                    except:
                        tool_call_str = f"{self.t['tool_call_start']}\n{output}\n{self.t['tool_call_end']}"
                    
                    text = (
                        f"{self.t['user_start']}\n{sample['instruction']}\n{self.t['user_end']}\n"
                        f"{self.t['assistant_start']}\n{tool_call_str}\n{self.t['assistant_end']}"
                    )
                    return {"text": self._wrap_sequence(text), "type": "tool_use"}

        except:
            pass
        return None

    def format_agentic_sample(self, sample: Dict) -> Optional[Dict]:
        """Handle multiple agentic dataset formats."""
        try:
            if "conversation" in sample and isinstance(sample["conversation"], list):
                parts = []
                for c in sample["conversation"]:
                    role = c.get("role", "").lower()
                    content = c.get("content", "")
                    if role == "user":
                        parts.append(f"{self.t['user_start']}\n{content}\n{self.t['user_end']}")
                    elif role == "assistant":
                        parts.append(f"{self.t['assistant_start']}\n{content}\n{self.t['assistant_end']}")
                if parts:
                    return {"text": "\n".join(parts), "type": "agentic"}

            convs = sample.get("conversations", [])
            if convs and isinstance(convs, list):
                parts = []
                for c in convs:
                    role = c.get("from", c.get("role", "")).lower()
                    content = c.get("value", c.get("content", ""))
                    if role in ["human", "user"]:
                        parts.append(f"{self.t['user_start']}\n{content}\n{self.t['user_end']}")
                    elif role in ["gpt", "assistant"]:
                        parts.append(f"{self.t['assistant_start']}\n{content}\n{self.t['assistant_end']}")
                if parts:
                    return {"text": "\n".join(parts), "type": "agentic"}

            if "data" in sample and isinstance(sample["data"], list):
                parts = []
                for i, msg in enumerate(sample["data"]):
                    if isinstance(msg, str):
                        if i % 2 == 0:
                            parts.append(f"{self.t['user_start']}\n{msg}\n{self.t['user_end']}")
                        else:
                            parts.append(f"{self.t['assistant_start']}\n{msg}\n{self.t['assistant_end']}")
                if parts:
                    return {"text": "\n".join(parts), "type": "agentic"}

            if "question" in sample and "answer" in sample:
                text = (
                    f"{self.t['user_start']}\n{sample['question']}\n{self.t['user_end']}\n"
                    f"{self.t['assistant_start']}\n{sample['answer']}\n{self.t['assistant_end']}"
                )
                return {"text": self._wrap_sequence(text), "type": "agentic"}

            if "instruction" in sample and "output" in sample:
                text = (
                    f"{self.t['user_start']}\n{sample['instruction']}\n{self.t['user_end']}\n"
                    f"{self.t['assistant_start']}\n{sample['output']}\n{self.t['assistant_end']}"
                )
                return {"text": self._wrap_sequence(text), "type": "agentic"}

        except:
            pass
        return None

    def format_image_caption_sample(self, sample: Dict) -> Optional[Dict]:
        try:
            caption = sample.get("caption", sample.get("text", sample.get("caption_1", "")))
            if not caption:
                caption = sample.get("image_description", sample.get("image_uncanny_description", ""))
            if not caption:
                return None
            text = (
                f"{self.t['user_start']}\n{self.t['image_start']}[IMAGE]{self.t['image_end']}\n"
                f"Describe this image in detail.\n{self.t['user_end']}\n"
                f"{self.t['assistant_start']}\n{caption}\n{self.t['assistant_end']}"
            )
            return {"text": self._wrap_sequence(text), "type": "image_caption", "has_image": True}
        except:
            pass
        return None

    def format_image_vqa_sample(self, sample: Dict) -> Optional[Dict]:
        try:
            question = sample.get("question", "")
            if not question:
                return None
            answer = sample.get("answer", "")
            choices = sample.get("choices", [])
            if choices and isinstance(answer, int) and answer < len(choices):
                answer_text = choices[answer]
            else:
                answer_text = str(answer)
            text = (
                f"{self.t['user_start']}\n{self.t['image_start']}[IMAGE]{self.t['image_end']}\n"
                f"{question}\n{self.t['user_end']}\n"
                f"{self.t['assistant_start']}\n{answer_text}\n{self.t['assistant_end']}"
            )
            return {"text": self._wrap_sequence(text), "type": "image_vqa", "has_image": True}
        except:
            pass
        return None

    def format_video_caption_sample(self, sample: Dict) -> Optional[Dict]:
        try:
            # Video-MME format (has url, videoID, task_type, question, options, answer)
            # Dataset: lmms-lab/Video-MME
            # Columns: category, url, videoID, task_type, question, options, answer
            # - url: Full YouTube URL (https://www.youtube.com/watch?v=...)
            # - videoID: YouTube video ID (11 chars)
            # - options: List like ["A. 1.", "B. 4.", "C. 5.", "D. 6."]
            # - answer: Correct answer letter ("A", "B", "C", or "D")
            if "question" in sample and "options" in sample:
                question = sample.get("question", "")
                options = sample.get("options", [])
                # Get the answer - Video-MME uses 'answer' field (or 'a' in some versions)
                answer_letter = sample.get("answer") or sample.get("a", "")
                
                if question and options:
                    # Format options as part of the question
                    options_text = "\n".join(options) if isinstance(options, list) else str(options)
                    
                    # Find the full answer text from options based on the answer letter
                    answer = answer_letter  # Default to just the letter
                    if isinstance(options, list) and answer_letter:
                        # Options format: ["A. text", "B. text", ...] - find matching option
                        for opt in options:
                            if isinstance(opt, str) and opt.startswith(f"{answer_letter}."):
                                answer = opt
                                break
                    
                    text = (
                        f"{self.t['user_start']}\n{self.t['video_start']}[VIDEO]{self.t['video_end']}\n"
                        f"{question}\n\nOptions:\n{options_text}\n{self.t['user_end']}\n"
                        f"{self.t['assistant_start']}\n{answer}\n{self.t['assistant_end']}"
                    )
                    return {"text": self._wrap_sequence(text), "type": "video_caption", "has_video": True}
            
            # Fallback: domain/sub_category format
            if "domain" in sample and "sub_category" in sample:
                # If it has question/answer directly, treat as QA
                if "question" in sample and ("answer" in sample or "a" in sample):
                    question = sample.get("question", "")
                    answer = sample.get("answer", sample.get("a", ""))
                    if question and answer:
                        text = (
                            f"{self.t['user_start']}\n{self.t['video_start']}[VIDEO]{self.t['video_end']}\n"
                            f"{question}\n{self.t['user_end']}\n"
                            f"{self.t['assistant_start']}\n{answer}\n{self.t['assistant_end']}"
                        )
                        return {"text": self._wrap_sequence(text), "type": "video_caption", "has_video": True}
                # Otherwise use domain/category as caption
                domain = sample.get("domain", "")
                sub_cat = sample.get("sub_category", "")
                duration = sample.get("duration", "")
                caption = f"A {duration} video about {domain} - {sub_cat}"
                text = (
                    f"{self.t['user_start']}\n{self.t['video_start']}[VIDEO]{self.t['video_end']}\n"
                    f"Describe what happens in this video.\n{self.t['user_end']}\n"
                    f"{self.t['assistant_start']}\n{caption}\n{self.t['assistant_end']}"
                )
                return {"text": self._wrap_sequence(text), "type": "video_caption", "has_video": True}
            
            # Vript format (caption is a dict with shot_type, camera_movement, etc.)
            caption = sample.get("caption", "")
            if isinstance(caption, dict):
                parts = []
                if caption.get("shot_type"):
                    parts.append(f"Shot type: {caption['shot_type']}")
                if caption.get("camera_movement"):
                    parts.append(f"Camera: {caption['camera_movement']}")
                if caption.get("content"):
                    parts.append(caption["content"])
                caption = " ".join(parts) if parts else ""
            
            # Standard caption fields
            if not caption:
                caption = sample.get("description", sample.get("text", sample.get("short_caption", sample.get("dense_caption", ""))))
            
            if caption:
                if isinstance(caption, list):
                    caption = caption[0] if caption else ""
                text = (
                    f"{self.t['user_start']}\n{self.t['video_start']}[VIDEO]{self.t['video_end']}\n"
                    f"Describe what happens in this video.\n{self.t['user_end']}\n"
                    f"{self.t['assistant_start']}\n{caption}\n{self.t['assistant_end']}"
                )
                return {"text": self._wrap_sequence(text), "type": "video_caption", "has_video": True}
        except:
            pass
        return None

    def format_video_qa_sample(self, sample: Dict) -> Optional[Dict]:
        try:
            question = sample.get("q", sample.get("question", ""))
            answer = sample.get("a", sample.get("answer", ""))
            if question and answer:
                text = (
                    f"{self.t['user_start']}\n{self.t['video_start']}[VIDEO]{self.t['video_end']}\n"
                    f"{question}\n{self.t['user_end']}\n"
                    f"{self.t['assistant_start']}\n{answer}\n{self.t['assistant_end']}"
                )
                return {"text": self._wrap_sequence(text), "type": "video_qa", "has_video": True}
        except:
            pass
        return None

    def format_image_generation_sample(self, sample: Dict) -> Optional[Dict]:
        try:
            prompt = sample.get("prompt", sample.get("Prompt", sample.get("text", "")))
            if not prompt and "json" in sample:
                json_data = sample["json"]
                if isinstance(json_data, dict):
                    prompt = json_data.get("prompt", "")
            if not prompt:
                return None
            text = (
                f"{self.t['user_start']}\nGenerate an image: {prompt}\n{self.t['user_end']}\n"
                f"{self.t['assistant_start']}\n{self.t['gen_image_start']}\n{prompt}\n{self.t['gen_image_end']}\n{self.t['assistant_end']}"
            )
            return {"text": self._wrap_sequence(text), "type": "image_generation"}
        except:
            pass
        return None

    def format_video_generation_sample(self, sample: Dict) -> Optional[Dict]:
        """Format video generation samples including Rapidata likert/preference datasets."""
        try:
            # Try various prompt field names
            prompt = sample.get("caption", sample.get("text", sample.get("description", "")))
            if isinstance(prompt, list) and prompt:
                prompt = prompt[0]
            if not prompt:
                # Rapidata format uses "Prompt" (capitalized) or "prompt"
                prompt = sample.get("Prompt", sample.get("prompt", ""))
            if not prompt:
                prompt = sample.get("name", sample.get("page_dir", ""))
            if not prompt:
                return None
            prompt = str(prompt).strip()
            if len(prompt) < 5:
                return None
            text = (
                f"{self.t['user_start']}\nGenerate a video: {prompt}\n{self.t['user_end']}\n"
                f"{self.t['assistant_start']}\n{self.t['gen_video_start']}\n{prompt}\n{self.t['gen_video_end']}\n{self.t['assistant_end']}"
            )
            return {"text": self._wrap_sequence(text), "type": "video_generation"}
        except:
            pass
        return None

    def format_image_to_video_sample(self, sample: Dict) -> Optional[Dict]:
        """Format image-to-video samples."""
        try:
            prompt = sample.get("Text_Prompt", "")
            
            # Rapidata image-to-video preference format (has prompt_asset as source image)
            if not prompt:
                prompt = sample.get("prompt", "")
            
            # MiraData format
            if not prompt:
                prompt = sample.get("short_caption", sample.get("dense_caption", ""))
            
            # UltraVideo format
            if not prompt:
                prompt = sample.get("Brief Description", sample.get("Detailed Description", sample.get("Summarized Description", "")))
            
            # Vript format (caption is a dict)
            if not prompt:
                caption = sample.get("caption", "")
                if isinstance(caption, dict):
                    prompt = caption.get("content", caption.get("shot_type", ""))
                elif isinstance(caption, str):
                    prompt = caption
            
            # Standard fields
            if not prompt:
                prompt = sample.get("text", sample.get("description", ""))
            
            # Pexels format (column0 = image URL, column1 = video URL)
            if not prompt and sample.get("column0") and sample.get("column1"):
                col0 = str(sample.get("column0", ""))
                if col0 == "thumbnail_loc":
                    return None
                if 'pexels.com' in col0 or 'images.pexels' in col0:
                    filename = col0.split('/')[-1].replace('.jpeg', '').replace('.jpg', '').replace('.png', '')
                    parts = filename.split('-')[:-1]
                    if parts:
                        prompt = ' '.join(parts)
            
            if not prompt:
                return None
            
            # Handle list prompts
            if isinstance(prompt, list):
                prompt = prompt[0] if prompt else ""
            
            prompt = str(prompt).strip()
            if len(prompt) < 3:
                return None
            
            text = (
                f"{self.t['user_start']}\n{self.t['image_start']}[IMAGE]{self.t['image_end']}\n"
                f"Animate this image: {prompt}\n{self.t['user_end']}\n"
                f"{self.t['assistant_start']}\n{self.t['gen_video_start']}[ANIMATED_VIDEO]{self.t['gen_video_end']}\n{self.t['assistant_end']}"
            )
            return {"text": self._wrap_sequence(text), "type": "image_to_video", "has_image": True}
        except:
            pass
        return None

    def format_image_editing_sample(self, sample: Dict) -> Optional[Dict]:
        try:
            instruction = sample.get("instruction", sample.get("edit_prompt", sample.get("edit", "")))
            if not instruction:
                return None
            text = (
                f"{self.t['user_start']}\n{self.t['image_start']}[IMAGE]{self.t['image_end']}\n"
                f"Edit this image: {instruction}\n{self.t['user_end']}\n"
                f"{self.t['assistant_start']}\n{self.t['gen_image_start']}[EDITED_IMAGE]{self.t['gen_image_end']}\n{self.t['assistant_end']}"
            )
            return {"text": self._wrap_sequence(text), "type": "image_editing", "has_image": True}
        except:
            pass
        return None

    def format_ui_to_code_sample(self, sample: Dict) -> Optional[Dict]:
        try:
            if "text" in sample and "<html" in str(sample.get("text", "")).lower():
                code = sample["text"]
                text = (
                    f"{self.t['user_start']}\n{self.t['image_start']}[IMAGE]{self.t['image_end']}\n"
                    f"Look at this website screenshot. Generate the HTML/CSS code to recreate it.\n{self.t['user_end']}\n"
                    f"{self.t['assistant_start']}\n{self.t['code_start']}\n{code[:3000]}\n{self.t['code_end']}\n{self.t['assistant_end']}"
                )
                return {"text": self._wrap_sequence(text), "type": "ui_to_code", "has_image": True}
        except:
            pass
        return None

    def format_voice_asr_sample(self, sample: Dict) -> Optional[Dict]:
        """Format ASR samples for speech-to-text training."""
        try:
            # LibriSpeech uses 'text', MLS uses 'transcript'
            text = sample.get("text", "")
            if not text:
                text = sample.get("transcript", sample.get("transcription", sample.get("sentence", "")))
            if not text:
                return None
            text = str(text).strip()
            if len(text) < 2:
                return None
            
            # Format: [AUDIO] -> transcribed text
            formatted_text = (
                f"{self.t['user_start']}\n"
                f"{self.t['listen_start']}[AUDIO]{self.t['listen_end']}\n"
                f"Transcribe this audio.\n"
                f"{self.t['user_end']}\n"
                f"{self.t['assistant_start']}\n{text}\n{self.t['assistant_end']}"
            )
            
            # Wrap with sequence tokens
            formatted_text = self._wrap_sequence(formatted_text)
            
            return {"text": formatted_text, "type": "voice_asr", "has_audio": True}
        except:
            pass
        return None

    def format_voice_tts_sample(self, sample: Dict, add_emotion: bool = True, add_prosody: bool = True) -> Optional[Dict]:
        """
        Format TTS samples for text-to-speech training.
        
        Includes hidden emotion and prosody tokens that control speech synthesis
        but are stripped from user-visible text output.
        """
        try:
            # LibriTTS-R uses 'text_normalized', MLS uses 'transcript'
            text = sample.get("text_normalized", "")
            if not text:
                text = sample.get("text", sample.get("transcript", sample.get("transcription", "")))
            if not text:
                text = sample.get("normalized_text", sample.get("sentence", ""))
            if not text:
                return None
            text = str(text).strip()
            if len(text) < 2:
                return None
            
            # Build the speech output with emotion/prosody tokens
            speech_content = "[SPEECH]"
            
            # Add emotion token (hidden from user, used by TTS decoder)
            if add_emotion:
                emotion = self._get_random_emotion()
                speech_content = self._add_emotion_token(speech_content, emotion)
            
            # Add prosody tokens (hidden from user, used by TTS decoder)
            if add_prosody:
                prosody = self._get_random_prosody()
                speech_content = self._add_prosody_tokens(speech_content, **prosody)
            
            formatted_text = (
                f"{self.t['user_start']}\nSay: {text}\n{self.t['user_end']}\n"
                f"{self.t['assistant_start']}\n"
                f"{self.t['speak_start']}{speech_content}{self.t['speak_end']}\n"
                f"{self.t['assistant_end']}"
            )
            
            # Wrap with sequence tokens
            formatted_text = self._wrap_sequence(formatted_text)
            
            return {"text": formatted_text, "type": "voice_tts", "has_audio": True}
        except:
            pass
        return None
    
    def format_expressive_tts_sample(self, sample: Dict, emotion: str = None, 
                                      speed: str = None, volume: str = None, pitch: str = None) -> Optional[Dict]:
        """
        Format TTS samples with specific emotion and prosody settings.
        
        This allows training on samples with explicit emotion/prosody labels.
        
        Args:
            sample: Input sample with text
            emotion: Specific emotion (happy, sad, angry, etc.)
            speed: Speech speed (fast, slow, normal_speed)
            volume: Speech volume (loud, soft, whisper, normal_volume)
            pitch: Speech pitch (high_pitch, low_pitch, normal_pitch)
        """
        try:
            text = sample.get("text_normalized", sample.get("text", sample.get("transcript", "")))
            if not text:
                return None
            text = str(text).strip()
            if len(text) < 2:
                return None
            
            # Build speech content with specified emotion/prosody
            speech_content = "[SPEECH]"
            
            if emotion:
                speech_content = self._add_emotion_token(speech_content, emotion)
            
            speech_content = self._add_prosody_tokens(speech_content, speed, volume, pitch)
            
            # Build user prompt that specifies the desired style
            style_parts = []
            if emotion:
                style_parts.append(f"in a {emotion} tone")
            if speed and speed != 'normal_speed':
                style_parts.append(f"speaking {speed.replace('_', ' ')}")
            if volume and volume != 'normal_volume':
                style_parts.append(f"{volume.replace('_', ' ')}")
            
            style_instruction = f" ({', '.join(style_parts)})" if style_parts else ""
            
            formatted_text = (
                f"{self.t['user_start']}\nSay{style_instruction}: {text}\n{self.t['user_end']}\n"
                f"{self.t['assistant_start']}\n"
                f"{self.t['speak_start']}{speech_content}{self.t['speak_end']}\n"
                f"{self.t['assistant_end']}"
            )
            
            formatted_text = self._wrap_sequence(formatted_text)
            
            return {
                "text": formatted_text, 
                "type": "voice_tts_expressive", 
                "has_audio": True,
                "emotion": emotion,
                "prosody": {"speed": speed, "volume": volume, "pitch": pitch}
            }
        except:
            pass
        return None

    def format_chain_of_thought_sample(self, sample: Dict) -> Optional[Dict]:
        """
        Format chain-of-thought reasoning samples.
        
        Expected format from synthetic dataset:
        {
            "text": "<|user|>...<|/user|><|assistant|><|think|>...<|/think|>...<|/assistant|>",
            "type": "chain_of_thought",
            "category": "math_arithmetic",
            "difficulty": "medium"
        }
        
        Or raw format:
        {
            "question": "...",
            "thinking": "...",
            "answer": "..."
        }
        """
        try:
            # If already formatted (from synthetic generator)
            if "text" in sample and sample.get("type") == "chain_of_thought":
                text = sample["text"]
                # Ensure BOS/EOS are present
                if not text.startswith(self.t['bos']):
                    text = self._wrap_sequence(text)
                return {
                    "text": text,
                    "type": "chain_of_thought",
                    "category": sample.get("category", "reasoning"),
                    "difficulty": sample.get("difficulty", "medium")
                }
            
            # Raw format with question/thinking/answer
            question = sample.get("question", sample.get("prompt", sample.get("input", "")))
            thinking = sample.get("thinking", sample.get("reasoning", sample.get("thought", "")))
            answer = sample.get("answer", sample.get("response", sample.get("output", "")))
            
            if not question or not answer:
                return None
            
            # Build the formatted text
            if thinking:
                # Full chain-of-thought format
                formatted_text = (
                    f"{self.t['user_start']}\n{question}\n{self.t['user_end']}\n"
                    f"{self.t['assistant_start']}\n"
                    f"{self.t['think_start']}\n{thinking}\n{self.t['think_end']}\n"
                    f"{answer}\n"
                    f"{self.t['assistant_end']}"
                )
            else:
                # Simple Q&A format (no thinking)
                formatted_text = (
                    f"{self.t['user_start']}\n{question}\n{self.t['user_end']}\n"
                    f"{self.t['assistant_start']}\n{answer}\n{self.t['assistant_end']}"
                )
            
            return {
                "text": self._wrap_sequence(formatted_text),
                "type": "chain_of_thought",
                "category": sample.get("category", "reasoning"),
                "difficulty": sample.get("difficulty", "medium")
            }
        except Exception:
            pass
        return None

    def format_reasoning_with_steps(self, sample: Dict) -> Optional[Dict]:
        """
        Format reasoning samples with explicit step-by-step structure.
        
        Uses the full set of reasoning tokens:
        - <|think|> ... <|/think|> - Main thinking block
        - <|plan|> ... <|/plan|> - Planning phase with steps
        - <|observation|> ... <|/observation|> - Observations about the problem
        - <|analysis|> ... <|/analysis|> - Problem analysis
        - <|note|> ... <|/note|> - Important notes
        - <|step|> ... <|/step|> - Individual reasoning steps
        - <|reflection|> ... <|/reflection|> - Self-reflection
        - <|hypothesis|> ... <|/hypothesis|> - Hypotheses
        - <|critique|> ... <|/critique|> - Self-critique with error checking
        - <|conclusion|> ... <|/conclusion|> - Final conclusion
        - <|uncertainty_score|> - Confidence/uncertainty level
        """
        try:
            question = sample.get("question", sample.get("prompt", ""))
            steps = sample.get("steps", [])
            observations = sample.get("observations", [])
            notes = sample.get("notes", [])
            hypothesis = sample.get("hypothesis", "")
            reflection = sample.get("reflection", "")
            conclusion = sample.get("conclusion", "")
            answer = sample.get("answer", sample.get("response", ""))
            
            # New fields for enhanced reasoning
            plan_steps = sample.get("plan", sample.get("plan_steps", []))
            analysis = sample.get("analysis", "")
            critique = sample.get("critique", "")
            has_error = sample.get("has_error", False)
            uncertainty = sample.get("uncertainty", sample.get("uncertainty_score", None))
            
            if not question or not answer:
                return None
            
            # Build thinking content
            thinking_parts = []
            
            # Add planning phase if present
            if plan_steps:
                plan_content = "\n".join([
                    f"{self.t.get('plan_step', '')}{step}{self.t.get('plan_step_end', '')}" 
                    for step in plan_steps
                ])
                thinking_parts.append(f"{self.t.get('plan_start', '')}\n{plan_content}\n{self.t.get('plan_end', '')}")
            
            # Add observations
            for obs in observations:
                thinking_parts.append(f"{self.t['observation_start']}{obs}{self.t['observation_end']}")
            
            # Add analysis if present
            if analysis:
                thinking_parts.append(f"{self.t.get('analysis_start', '')}{analysis}{self.t.get('analysis_end', '')}")
            
            # Add notes
            for note in notes:
                thinking_parts.append(f"{self.t['note_start']}{note}{self.t['note_end']}")
            
            # Add hypothesis if present
            if hypothesis:
                thinking_parts.append(f"{self.t['hypothesis_start']}{hypothesis}{self.t['hypothesis_end']}")
            
            # Add steps
            for step in steps:
                thinking_parts.append(f"{self.t['step_start']}{step}{self.t['step_end']}")
            
            # Add reflection if present
            if reflection:
                thinking_parts.append(f"{self.t['reflection_start']}{reflection}{self.t['reflection_end']}")
            
            # Add critique/self-check if present
            if critique:
                error_marker = self.t.get('error_found', '') if has_error else self.t.get('no_error', '')
                thinking_parts.append(f"{self.t.get('critique_start', '')}\n{error_marker}{critique}\n{self.t.get('critique_end', '')}")
            
            # Add conclusion
            if conclusion:
                thinking_parts.append(f"{self.t['conclusion_start']}{conclusion}{self.t['conclusion_end']}")
            
            thinking_content = "\n".join(thinking_parts) if thinking_parts else ""
            
            # Build uncertainty marker if present
            uncertainty_marker = ""
            if uncertainty is not None:
                uncertainty_marker = f"{self.t.get('uncertainty_score', '')}{uncertainty}{self.t.get('uncertainty_score_end', '')}\n"
            
            if thinking_content:
                formatted_text = (
                    f"{self.t['user_start']}\n{question}\n{self.t['user_end']}\n"
                    f"{self.t['assistant_start']}\n"
                    f"{self.t['think_start']}\n{thinking_content}\n{self.t['think_end']}\n"
                    f"{uncertainty_marker}"
                    f"{answer}\n"
                    f"{self.t['assistant_end']}"
                )
            else:
                formatted_text = (
                    f"{self.t['user_start']}\n{question}\n{self.t['user_end']}\n"
                    f"{self.t['assistant_start']}\n{uncertainty_marker}{answer}\n{self.t['assistant_end']}"
                )
            
            return {
                "text": formatted_text,
                "type": "chain_of_thought",
                "category": sample.get("category", "reasoning"),
                "difficulty": sample.get("difficulty", "medium")
            }
        except Exception:
            pass
        return None

    def format_document_sample(self, sample: Dict) -> Optional[Dict]:
        """
        Format document samples with file type markers.
        
        Handles various document formats:
        - Plain text (.txt)
        - Markdown (.md)
        - JSON (.json)
        - XML (.xml)
        - YAML (.yaml)
        - HTML (.html)
        - CSV (.csv)
        - Config files (.toml, .ini)
        - Log files (.log)
        
        Expected sample format:
        {
            "content": "document content...",
            "file_type": "md",  # or detected from filename
            "filename": "readme.md",  # optional
            "filepath": "/path/to/file",  # optional
        }
        
        Or conversation format:
        {
            "instruction": "Summarize this document",
            "document": "document content...",
            "response": "summary...",
            "file_type": "txt"
        }
        
        Or pre-formatted (from synthetic generator):
        {
            "text": "<|bos|>...<|eos|>",
            "type": "document",
            "file_type": "json",
            "task": "parse"
        }
        """
        try:
            # If already formatted (from synthetic generator), return as-is
            if "text" in sample and sample.get("type") == "document":
                text = sample["text"]
                # Ensure BOS/EOS are present
                if not text.startswith(self.t['bos']):
                    text = self._wrap_sequence(text)
                return {
                    "text": text,
                    "type": "document",
                    "file_type": sample.get("file_type", "txt"),
                    "task": sample.get("task", "unknown")
                }
            
            # Get document content for raw samples
            content = self._safe_get(sample, ['content', 'document', 'body', 'data'])
            if not content:
                return None
            
            content = str(content)
            
            # Get file metadata
            file_type = self._safe_get(sample, ['file_type', 'type', 'format', 'extension'])
            filename = self._safe_get(sample, ['filename', 'file_name', 'name'])
            filepath = self._safe_get(sample, ['filepath', 'file_path', 'path'])
            
            # Try to detect file type from filename if not provided
            if not file_type and filename:
                ext_map = {
                    '.txt': 'txt', '.text': 'txt',
                    '.md': 'md', '.markdown': 'md',
                    '.json': 'json',
                    '.xml': 'xml',
                    '.yaml': 'yaml', '.yml': 'yaml',
                    '.html': 'html', '.htm': 'html',
                    '.css': 'css',
                    '.csv': 'csv',
                    '.toml': 'toml',
                    '.ini': 'ini', '.cfg': 'ini', '.conf': 'ini',
                    '.log': 'log',
                }
                for ext, ftype in ext_map.items():
                    if filename.lower().endswith(ext):
                        file_type = ftype
                        break
            
            # Default to txt if no type detected
            if not file_type:
                file_type = 'txt'
            
            # Check if this is a Q&A format about a document
            instruction = self._safe_get(sample, ['instruction', 'prompt', 'query', 'question'])
            response = self._safe_get(sample, ['response', 'output', 'answer', 'completion'])
            
            if instruction and response:
                # Q&A about document format
                doc_wrapped = self._wrap_document(content, file_type, filename)
                user_content = f"{instruction}\n\n{doc_wrapped}"
                text = self._format_simple_qa(user_content, response)
                return {
                    "text": text,
                    "type": "document",
                    "file_type": file_type,
                    "filename": filename
                }
            else:
                # Raw document format - wrap with file type tokens
                doc_wrapped = self._wrap_document(content, file_type, filename)
                text = self._wrap_sequence(doc_wrapped)
                return {
                    "text": text,
                    "type": "document",
                    "file_type": file_type,
                    "filename": filename
                }
        except Exception:
            pass
        return None

    def format_multi_document_sample(self, sample: Dict) -> Optional[Dict]:
        """
        Format samples with multiple documents.
        
        Expected sample format:
        {
            "documents": [
                {"content": "...", "filename": "file1.txt", "file_type": "txt"},
                {"content": "...", "filename": "file2.json", "file_type": "json"},
            ],
            "instruction": "Compare these documents",
            "response": "comparison..."
        }
        
        Or pre-formatted (from synthetic generator):
        {
            "text": "<|bos|>...<|eos|>",
            "type": "multi_document",
            "num_documents": 2
        }
        """
        try:
            # If already formatted (from synthetic generator), return as-is
            if "text" in sample and sample.get("type") == "multi_document":
                text = sample["text"]
                # Ensure BOS/EOS are present
                if not text.startswith(self.t['bos']):
                    text = self._wrap_sequence(text)
                return {
                    "text": text,
                    "type": "multi_document",
                    "num_documents": sample.get("num_documents", 0)
                }
            
            documents = sample.get("documents", sample.get("files", []))
            if not documents or not isinstance(documents, list):
                return None
            
            # Format each document
            doc_parts = []
            for doc in documents:
                content = doc.get("content", doc.get("text", ""))
                file_type = doc.get("file_type", doc.get("type", "txt"))
                filename = doc.get("filename", doc.get("name", ""))
                
                if content:
                    doc_wrapped = self._wrap_document(content, file_type, filename)
                    doc_parts.append(doc_wrapped)
            
            if not doc_parts:
                return None
            
            # Combine documents with separator
            combined_docs = f"\n{self.t.get('separator', '')}\n".join(doc_parts)
            
            # Check for Q&A format
            instruction = self._safe_get(sample, ['instruction', 'prompt', 'query', 'question'])
            response = self._safe_get(sample, ['response', 'output', 'answer', 'completion'])
            
            if instruction and response:
                user_content = f"{instruction}\n\n{combined_docs}"
                text = self._format_simple_qa(user_content, response)
            else:
                text = self._wrap_sequence(combined_docs)
            
            return {
                "text": text,
                "type": "multi_document",
                "num_documents": len(doc_parts)
            }
        except Exception:
            pass
        return None
