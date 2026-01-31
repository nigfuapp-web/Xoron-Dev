"""Special tokens for Xoron-Dev multimodal model."""

SPECIAL_TOKENS = {
    # === SEQUENCE CONTROL ===
    # Beginning/End of sequence
    "bos": "<|bos|>",
    "eos": "<|eos|>",
    "pad": "<|pad|>",
    
    # Prompt/Text boundaries
    "prompt_start": "<|prompt|>",
    "prompt_end": "<|/prompt|>",
    "text_start": "<|text|>",
    "text_end": "<|/text|>",
    "response_start": "<|response|>",
    "response_end": "<|/response|>",

    # === CONVERSATION ===
    "system_start": "<|system|>",
    "system_end": "<|/system|>",
    "user_start": "<|user|>",
    "user_end": "<|/user|>",
    "assistant_start": "<|assistant|>",
    "assistant_end": "<|/assistant|>",
    
    # === MEMORY & CONTEXT MANAGEMENT ===
    # Working memory / Long-term retrieval (RAG)
    "memory_start": "<|memory|>",
    "memory_end": "<|/memory|>",
    "working_memory_start": "<|working_memory|>",
    "working_memory_end": "<|/working_memory|>",
    "long_term_memory_start": "<|long_term_memory|>",
    "long_term_memory_end": "<|/long_term_memory|>",
    
    # Summary/condensation
    "summary_start": "<|summary|>",
    "summary_end": "<|/summary|>",
    "condensed_start": "<|condensed|>",
    "condensed_end": "<|/condensed|>",
    
    # User profile / preferences (hard rules)
    "user_profile_start": "<|user_profile|>",
    "user_profile_end": "<|/user_profile|>",
    "user_preference_start": "<|user_pref|>",
    "user_preference_end": "<|/user_pref|>",
    "hard_rule_start": "<|hard_rule|>",
    "hard_rule_end": "<|/hard_rule|>",
    
    # Session/conversation context
    "session_start": "<|session|>",
    "session_end": "<|/session|>",
    "conversation_history_start": "<|conv_history|>",
    "conversation_history_end": "<|/conv_history|>",

    # === FILL-IN-THE-MIDDLE (FIM) - Code Completion ===
    # For ghost text / autocomplete functionality
    "fim_prefix": "<|fim_prefix|>",
    "fim_middle": "<|fim_middle|>",
    "fim_suffix": "<|fim_suffix|>",
    "fim_pad": "<|fim_pad|>",

    # === VERSION CONTROL & GIT INTEGRATION ===
    # Commit operations
    "commit_before": "<|commit_before|>",
    "commit_before_end": "<|/commit_before|>",
    "commit_after": "<|commit_after|>",
    "commit_after_end": "<|/commit_after|>",
    "commit_msg": "<|commit_msg|>",
    "commit_msg_end": "<|/commit_msg|>",
    "diff_start": "<|diff|>",
    "diff_end": "<|/diff|>",
    "diff_add": "<|diff_add|>",  # Added lines
    "diff_del": "<|diff_del|>",  # Deleted lines
    "diff_context": "<|diff_ctx|>",  # Context lines
    
    # Repository metadata
    "reponame": "<|reponame|>",
    "reponame_end": "<|/reponame|>",
    "gh_stars": "<|gh_stars|>",
    "gh_stars_end": "<|/gh_stars|>",
    "branch": "<|branch|>",
    "branch_end": "<|/branch|>",
    
    # Issue tracking
    "issue_start": "<|issue|>",
    "issue_end": "<|/issue|>",
    "issue_title": "<|issue_title|>",
    "issue_title_end": "<|/issue_title|>",
    "issue_body": "<|issue_body|>",
    "issue_body_end": "<|/issue_body|>",
    "issue_closed": "<|issue_closed|>",
    "issue_open": "<|issue_open|>",
    
    # Pull request
    "pr_start": "<|pr|>",
    "pr_end": "<|/pr|>",
    "pr_title": "<|pr_title|>",
    "pr_title_end": "<|/pr_title|>",
    "pr_body": "<|pr_body|>",
    "pr_body_end": "<|/pr_body|>",
    "pr_merged": "<|pr_merged|>",
    "pr_closed": "<|pr_closed|>",

    # === INTERACTIVE EXECUTION (Jupyter/Code Interpreter) ===
    "jupyter_start": "<|jupyter|>",
    "jupyter_end": "<|/jupyter|>",
    "jupyter_code": "<|jupyter_code|>",
    "jupyter_code_end": "<|/jupyter_code|>",
    "jupyter_output": "<|jupyter_output|>",
    "jupyter_output_end": "<|/jupyter_output|>",
    "jupyter_text": "<|jupyter_text|>",
    "jupyter_text_end": "<|/jupyter_text|>",
    "jupyter_error": "<|jupyter_error|>",
    "jupyter_error_end": "<|/jupyter_error|>",
    "empty_output": "<|empty_output|>",
    
    # Execution control
    "exec_start": "<|exec|>",
    "exec_end": "<|/exec|>",
    "exec_result": "<|exec_result|>",
    "exec_result_end": "<|/exec_result|>",
    "exec_error": "<|exec_error|>",
    "exec_error_end": "<|/exec_error|>",
    "exec_timeout": "<|exec_timeout|>",

    # === FILE SYSTEM OPERATIONS ===
    # File actions (agentic operations)
    "add_file": "<|add_file|>",
    "add_file_end": "<|/add_file|>",
    "delete_file": "<|delete_file|>",
    "delete_file_end": "<|/delete_file|>",
    "rename_file": "<|rename_file|>",
    "rename_file_end": "<|/rename_file|>",
    "edit_file": "<|edit_file|>",
    "edit_file_end": "<|/edit_file|>",
    "read_file": "<|read_file|>",
    "read_file_end": "<|/read_file|>",
    "file_sep": "<|file_sep|>",
    "file_content": "<|file_content|>",
    "file_content_end": "<|/file_content|>",
    
    # Edit operations (for precise code modifications)
    "edit_range": "<|edit_range|>",  # Line range to edit
    "edit_range_end": "<|/edit_range|>",
    "line_num": "<|line|>",
    "line_num_end": "<|/line|>",
    "insert_before": "<|insert_before|>",
    "insert_after": "<|insert_after|>",
    "replace": "<|replace|>",
    "replace_end": "<|/replace|>",

    # === INTERNAL CONTROL SIGNALS (Embedding Markers) ===
    # These are used for internal state management and multimodal handoffs
    "encoder_start": "[e~[",
    "encoder_end": "]~e]",
    "decoder_start": "[d~[",
    "decoder_end": "]~d]",
    "projection_start": "[p~[",
    "projection_end": "]~p]",
    "state_begin": "]~b]",
    "state_end": "]~!b[",
    "modal_switch": "[m~[",
    "modal_switch_end": "]~m]",

    # === DOCUMENT/FILE HANDLING ===
    # Generic document boundaries (for multi-document contexts)
    "doc_start": "<|doc|>",
    "doc_end": "<|/doc|>",
    "eod": "<|eod|>",  # End of document (alternative to doc_end for single docs)
    
    # File type markers - wrap content to indicate format
    "file_txt": "<|file:txt|>",
    "file_txt_end": "<|/file:txt|>",
    "file_md": "<|file:md|>",
    "file_md_end": "<|/file:md|>",
    "file_json": "<|file:json|>",
    "file_json_end": "<|/file:json|>",
    "file_xml": "<|file:xml|>",
    "file_xml_end": "<|/file:xml|>",
    "file_yaml": "<|file:yaml|>",
    "file_yaml_end": "<|/file:yaml|>",
    "file_html": "<|file:html|>",
    "file_html_end": "<|/file:html|>",
    "file_css": "<|file:css|>",
    "file_css_end": "<|/file:css|>",
    "file_csv": "<|file:csv|>",
    "file_csv_end": "<|/file:csv|>",
    "file_toml": "<|file:toml|>",
    "file_toml_end": "<|/file:toml|>",
    "file_ini": "<|file:ini|>",
    "file_ini_end": "<|/file:ini|>",
    "file_log": "<|file:log|>",
    "file_log_end": "<|/file:log|>",
    
    # Document metadata (optional, for context)
    "filename_start": "<|filename|>",
    "filename_end": "<|/filename|>",
    "filepath_start": "<|filepath|>",
    "filepath_end": "<|/filepath|>",

    # === MULTIMODAL INPUT ===
    "image_start": "<|image|>",
    "image_end": "<|/image|>",
    "video_start": "<|video|>",
    "video_end": "<|/video|>",
    
    # === TEMPORAL / VIDEO NAVIGATION ===
    # Timestamp markers for temporal grounding
    "timestamp_start": "<|timestamp|>",
    "timestamp_end": "<|/timestamp|>",
    "time_range_start": "<|time_range|>",
    "time_range_end": "<|/time_range|>",
    
    # Keyframe markers for important frames
    "keyframe": "<|keyframe|>",
    "keyframe_end": "<|/keyframe|>",
    
    # Scene change detection
    "scene_change": "<|scene_change|>",
    "scene_start": "<|scene|>",
    "scene_end": "<|/scene|>",
    
    # Video segment markers
    "segment_start": "<|segment|>",
    "segment_end": "<|/segment|>",
    "frame_start": "<|frame|>",
    "frame_end": "<|/frame|>",
    "frame_num": "<|frame_num|>",
    "frame_num_end": "<|/frame_num|>",
    
    # Action/event markers in video
    "action_start": "<|action|>",
    "action_end": "<|/action|>",
    "event_start": "<|event|>",
    "event_end": "<|/event|>",
    
    # Spatial markers for image/video
    "region_start": "<|region|>",
    "region_end": "<|/region|>",
    "bbox_start": "<|bbox|>",
    "bbox_end": "<|/bbox|>",
    "object_start": "<|object|>",
    "object_end": "<|/object|>",

    # === GENERATION OUTPUT ===
    "gen_image_start": "<|gen_image|>",
    "gen_image_end": "<|/gen_image|>",
    "gen_video_start": "<|gen_video|>",
    "gen_video_end": "<|/gen_video|>",

    # === TOOL CALLING / FUNCTION CALLING ===
    # Main tool call wrapper
    "tool_call_start": "<|tool_call|>",
    "tool_call_end": "<|/tool_call|>",
    "tool_result_start": "<|tool_result|>",
    "tool_result_end": "<|/tool_result|>",
    
    # Function/tool name
    "function_name_start": "<|function_name|>",
    "function_name_end": "<|/function_name|>",
    
    # Function arguments (can contain JSON or structured args)
    "function_args_start": "<|function_args|>",
    "function_args_end": "<|/function_args|>",
    
    # Individual argument name/value pairs (for structured parsing)
    "arg_name_start": "<|arg_name|>",
    "arg_name_end": "<|/arg_name|>",
    "arg_value_start": "<|arg_value|>",
    "arg_value_end": "<|/arg_value|>",
    
    # Tool/function definitions (schema)
    "tools_start": "<|tools|>",
    "tools_end": "<|/tools|>",
    "function_def_start": "<|function_def|>",
    "function_def_end": "<|/function_def|>",
    
    # Available tools listing (for model to see all available tools)
    "available_tools_start": "<|available_tools|>",
    "available_tools_end": "<|/available_tools|>",
    "tool_def_start": "<|tool_def|>",
    "tool_def_end": "<|/tool_def|>",
    "tool_name": "<|tool_name|>",
    "tool_name_end": "<|/tool_name|>",
    "tool_description": "<|tool_desc|>",
    "tool_description_end": "<|/tool_desc|>",
    "tool_params_start": "<|tool_params|>",
    "tool_params_end": "<|/tool_params|>",
    "param_name": "<|param_name|>",
    "param_name_end": "<|/param_name|>",
    "param_type": "<|param_type|>",
    "param_type_end": "<|/param_type|>",
    "param_required": "<|param_required|>",
    "param_optional": "<|param_optional|>",
    
    # Tool execution status
    "tool_error_start": "<|tool_error|>",
    "tool_error_end": "<|/tool_error|>",
    "tool_success": "<|tool_success|>",
    
    # Multiple tool calls in sequence
    "tool_calls_start": "<|tool_calls|>",
    "tool_calls_end": "<|/tool_calls|>",
    
    # Tool call ID (for matching calls to results)
    "tool_id_start": "<|tool_id|>",
    "tool_id_end": "<|/tool_id|>",

    # === CODE ===
    "code_start": "<|code|>",
    "code_end": "<|/code|>",
    # Programming language markers for code blocks
    "lang_python": "<|lang:python|>",
    "lang_javascript": "<|lang:javascript|>",
    "lang_typescript": "<|lang:typescript|>",
    "lang_java": "<|lang:java|>",
    "lang_cpp": "<|lang:cpp|>",
    "lang_c": "<|lang:c|>",
    "lang_csharp": "<|lang:csharp|>",
    "lang_go": "<|lang:go|>",
    "lang_rust": "<|lang:rust|>",
    "lang_ruby": "<|lang:ruby|>",
    "lang_php": "<|lang:php|>",
    "lang_swift": "<|lang:swift|>",
    "lang_kotlin": "<|lang:kotlin|>",
    "lang_scala": "<|lang:scala|>",
    "lang_shell": "<|lang:shell|>",
    "lang_bash": "<|lang:bash|>",
    "lang_sql": "<|lang:sql|>",
    "lang_r": "<|lang:r|>",
    "lang_matlab": "<|lang:matlab|>",
    "lang_lua": "<|lang:lua|>",
    "lang_perl": "<|lang:perl|>",
    "lang_haskell": "<|lang:haskell|>",
    "lang_other": "<|lang:other|>",

    # === THINKING/REASONING (Chain-of-Thought) ===
    "think_start": "<|think|>",
    "think_end": "<|/think|>",
    
    # Sub-tokens for structured reasoning inside <think>
    "observation_start": "<|observation|>",
    "observation_end": "<|/observation|>",
    "note_start": "<|note|>",
    "note_end": "<|/note|>",
    "step_start": "<|step|>",
    "step_end": "<|/step|>",
    "reflection_start": "<|reflection|>",
    "reflection_end": "<|/reflection|>",
    "hypothesis_start": "<|hypothesis|>",
    "hypothesis_end": "<|/hypothesis|>",
    "conclusion_start": "<|conclusion|>",
    "conclusion_end": "<|/conclusion|>",
    
    # === ENHANCED REASONING / INNER MONOLOGUE ===
    # Planning - model plans steps before thinking
    "plan_start": "<|plan|>",
    "plan_end": "<|/plan|>",
    "plan_step": "<|plan_step|>",
    "plan_step_end": "<|/plan_step|>",
    
    # Critique / Self-correction - model checks its own logic
    "critique_start": "<|critique|>",
    "critique_end": "<|/critique|>",
    "error_found": "<|error_found|>",
    "no_error": "<|no_error|>",
    
    # Analysis / Breakdown
    "analysis_start": "<|analysis|>",
    "analysis_end": "<|/analysis|>",
    "breakdown_start": "<|breakdown|>",
    "breakdown_end": "<|/breakdown|>",
    
    # Decision making
    "decision_start": "<|decision|>",
    "decision_end": "<|/decision|>",
    "option_start": "<|option|>",
    "option_end": "<|/option|>",
    "chosen": "<|chosen|>",
    "rejected": "<|rejected|>",
    
    # Reasoning chain markers
    "because": "<|because|>",
    "therefore": "<|therefore|>",
    "however": "<|however|>",
    "alternatively": "<|alternatively|>",

    # === ANTI-HALLUCINATION / UNCERTAINTY TOKENS ===
    # Confidence levels - model expresses certainty about its response
    "confidence_high": "<|confidence:high|>",
    "confidence_medium": "<|confidence:medium|>",
    "confidence_low": "<|confidence:low|>",
    "confidence_uncertain": "<|confidence:uncertain|>",
    
    # Uncertainty score - numerical value 0-100
    "uncertainty_score": "<|uncertainty_score|>",
    "uncertainty_score_end": "<|/uncertainty_score|>",
    "certainty_score": "<|certainty_score|>",
    "certainty_score_end": "<|/certainty_score|>",
    
    # Explicit uncertainty markers
    "uncertain_start": "<|uncertain|>",
    "uncertain_end": "<|/uncertain|>",
    "unknown": "<|unknown|>",  # "I don't know"
    "need_verification": "<|need_verify|>",
    "may_be_outdated": "<|may_outdated|>",
    "speculative": "<|speculative|>",
    "approximate": "<|approximate|>",
    
    # Self-verification tokens
    "verify_start": "<|verify|>",
    "verify_end": "<|/verify|>",
    "fact_check": "<|fact_check|>",
    "self_correct": "<|self_correct|>",
    "correction_start": "<|correction|>",
    "correction_end": "<|/correction|>",
    "retract": "<|retract|>",
    "clarify": "<|clarify|>",
    
    # Citation/Source grounding tokens
    "cite_start": "<|cite|>",
    "cite_end": "<|/cite|>",
    "source_start": "<|source|>",
    "source_end": "<|/source|>",
    "reference_start": "<|ref|>",
    "reference_end": "<|/ref|>",
    "quote_start": "<|quote|>",
    "quote_end": "<|/quote|>",
    "attribution": "<|attribution|>",
    "attribution_end": "<|/attribution|>",
    
    # Grounding tokens (for RAG/retrieval)
    "context_start": "<|context|>",
    "context_end": "<|/context|>",
    "retrieved_start": "<|retrieved|>",
    "retrieved_end": "<|/retrieved|>",
    "grounded": "<|grounded|>",  # Response is grounded in provided context
    "ungrounded": "<|ungrounded|>",  # Response may not be grounded
    "partially_grounded": "<|partially_grounded|>",
    
    # Knowledge boundary markers
    "knowledge_cutoff": "<|knowledge_cutoff|>",
    "beyond_knowledge": "<|beyond_knowledge|>",
    "within_knowledge": "<|within_knowledge|>",
    "knowledge_date": "<|knowledge_date|>",
    "knowledge_date_end": "<|/knowledge_date|>",

    # === VOICE/AUDIO ===
    "listen_start": "<|listen|>",
    "listen_end": "<|/listen|>",
    "speak_start": "<|speak|>",
    "speak_end": "<|/speak|>",
    "audio_start": "<|audio|>",
    "audio_end": "<|/audio|>",

    # === HIDDEN EMOTION TAGS (for TTS - stripped from text output) ===
    # These control the emotional tone of speech synthesis
    # Format: <|emotion:TYPE|> - hidden from user, used by TTS decoder
    "emotion_start": "<|emotion|>",
    "emotion_end": "<|/emotion|>",
    "emotion_neutral": "<|emotion:neutral|>",
    "emotion_happy": "<|emotion:happy|>",
    "emotion_sad": "<|emotion:sad|>",
    "emotion_angry": "<|emotion:angry|>",
    "emotion_surprised": "<|emotion:surprised|>",
    "emotion_fearful": "<|emotion:fearful|>",
    "emotion_disgusted": "<|emotion:disgusted|>",
    "emotion_calm": "<|emotion:calm|>",
    "emotion_excited": "<|emotion:excited|>",
    "emotion_curious": "<|emotion:curious|>",
    "emotion_sympathetic": "<|emotion:sympathetic|>",
    "emotion_confident": "<|emotion:confident|>",

    # === HIDDEN PROSODY MARKERS (for TTS - stripped from text output) ===
    # These control speech characteristics like speed, pitch, volume
    # Format: <|prosody:TYPE|> - hidden from user, used by TTS decoder
    "prosody_start": "<|prosody|>",
    "prosody_end": "<|/prosody|>",
    "prosody_fast": "<|prosody:fast|>",
    "prosody_slow": "<|prosody:slow|>",
    "prosody_normal_speed": "<|prosody:normal_speed|>",
    "prosody_loud": "<|prosody:loud|>",
    "prosody_soft": "<|prosody:soft|>",
    "prosody_whisper": "<|prosody:whisper|>",
    "prosody_normal_volume": "<|prosody:normal_volume|>",
    "prosody_high_pitch": "<|prosody:high_pitch|>",
    "prosody_low_pitch": "<|prosody:low_pitch|>",
    "prosody_normal_pitch": "<|prosody:normal_pitch|>",
    "prosody_emphasis": "<|prosody:emphasis|>",
    "prosody_pause_short": "<|prosody:pause_short|>",
    "prosody_pause_long": "<|prosody:pause_long|>",
    
    # === STRUCTURED DATA TOKENS ===
    # Tables (markdown/CSV)
    "table_start": "<|table|>",
    "table_end": "<|/table|>",
    "table_row_start": "<|row|>",
    "table_row_end": "<|/row|>",
    "table_cell_start": "<|cell|>",
    "table_cell_end": "<|/cell|>",
    "table_header_start": "<|thead|>",
    "table_header_end": "<|/thead|>",
    "table_body_start": "<|tbody|>",
    "table_body_end": "<|/tbody|>",
    
    # Schema definitions (JSON/Code structure)
    "schema_start": "<|schema|>",
    "schema_end": "<|/schema|>",
    "schema_field": "<|field|>",
    "schema_field_end": "<|/field|>",
    "schema_type": "<|type|>",
    "schema_type_end": "<|/type|>",
    "schema_required": "<|required|>",
    "schema_optional": "<|optional|>",
    
    # Version markers (for code/documentation versioning)
    "version": "<|version|>",
    "version_end": "<|/version|>",
    "deprecated": "<|deprecated|>",
    "since": "<|since|>",
    "since_end": "<|/since|>",
    "changelog_start": "<|changelog|>",
    "changelog_end": "<|/changelog|>",
    
    # Data format markers
    "json_start": "<|json|>",
    "json_end": "<|/json|>",
    "xml_start": "<|xml|>",
    "xml_end": "<|/xml|>",
    "yaml_start": "<|yaml|>",
    "yaml_end": "<|/yaml|>",
    "csv_start": "<|csv|>",
    "csv_end": "<|/csv|>",
    
    # List/Array markers
    "list_start": "<|list|>",
    "list_end": "<|/list|>",
    "list_item": "<|item|>",
    "list_item_end": "<|/item|>",
    "ordered_list": "<|ol|>",
    "ordered_list_end": "<|/ol|>",
    "unordered_list": "<|ul|>",
    "unordered_list_end": "<|/ul|>",
    
    # Key-value pairs
    "kv_start": "<|kv|>",
    "kv_end": "<|/kv|>",
    "key_start": "<|key|>",
    "key_end": "<|/key|>",
    "value_start": "<|value|>",
    "value_end": "<|/value|>",
    
    # === TASK / INSTRUCTION MARKERS ===
    # Task type indicators
    "task_start": "<|task|>",
    "task_end": "<|/task|>",
    "task_type": "<|task_type|>",
    "task_type_end": "<|/task_type|>",
    "instruction_start": "<|instruction|>",
    "instruction_end": "<|/instruction|>",
    "constraint_start": "<|constraint|>",
    "constraint_end": "<|/constraint|>",
    "example_start": "<|example|>",
    "example_end": "<|/example|>",
    "input_start": "<|input|>",
    "input_end": "<|/input|>",
    "output_start": "<|output|>",
    "output_end": "<|/output|>",
    "expected_start": "<|expected|>",
    "expected_end": "<|/expected|>",
    
    # === SPECIAL MARKERS ===
    "separator": "<|sep|>",
    "newline": "<|nl|>",
    "mask": "<|mask|>",
    "ellipsis": "<|...|>",
    "continuation": "<|cont|>",
    "truncated": "<|truncated|>",
    "redacted": "<|redacted|>",
}

# Reasoning token groups for easy access
REASONING_TOKENS = {
    "think": ("think_start", "think_end"),
    "observation": ("observation_start", "observation_end"),
    "note": ("note_start", "note_end"),
    "step": ("step_start", "step_end"),
    "reflection": ("reflection_start", "reflection_end"),
    "hypothesis": ("hypothesis_start", "hypothesis_end"),
    "conclusion": ("conclusion_start", "conclusion_end"),
    "plan": ("plan_start", "plan_end"),
    "critique": ("critique_start", "critique_end"),
    "analysis": ("analysis_start", "analysis_end"),
    "decision": ("decision_start", "decision_end"),
}

# Memory and context tokens
MEMORY_TOKENS = {
    "memory": ("memory_start", "memory_end"),
    "working_memory": ("working_memory_start", "working_memory_end"),
    "long_term_memory": ("long_term_memory_start", "long_term_memory_end"),
    "summary": ("summary_start", "summary_end"),
    "user_profile": ("user_profile_start", "user_profile_end"),
    "hard_rule": ("hard_rule_start", "hard_rule_end"),
    "session": ("session_start", "session_end"),
}

# Temporal/Video tokens
TEMPORAL_TOKENS = {
    "timestamp": ("timestamp_start", "timestamp_end"),
    "time_range": ("time_range_start", "time_range_end"),
    "keyframe": ("keyframe_start", "keyframe_end"),
    "scene": ("scene_start", "scene_end"),
    "segment": ("segment_start", "segment_end"),
    "frame": ("frame_start", "frame_end"),
    "action": ("action_start", "action_end"),
    "event": ("event_start", "event_end"),
}

# Structured data tokens
STRUCTURED_DATA_TOKENS = {
    "table": ("table_start", "table_end"),
    "schema": ("schema_start", "schema_end"),
    "json": ("json_start", "json_end"),
    "xml": ("xml_start", "xml_end"),
    "yaml": ("yaml_start", "yaml_end"),
    "csv": ("csv_start", "csv_end"),
    "list": ("list_start", "list_end"),
    "kv": ("kv_start", "kv_end"),
}

# Uncertainty/Anti-hallucination tokens
UNCERTAINTY_TOKENS = {
    "uncertain": ("uncertain_start", "uncertain_end"),
    "verify": ("verify_start", "verify_end"),
    "correction": ("correction_start", "correction_end"),
    "cite": ("cite_start", "cite_end"),
    "source": ("source_start", "source_end"),
    "context": ("context_start", "context_end"),
    "retrieved": ("retrieved_start", "retrieved_end"),
    "uncertainty_score": ("uncertainty_score", "uncertainty_score_end"),
}

# Hidden tokens that should be stripped from user-visible output
# These are used internally by the model (e.g., for TTS) but not shown to users
HIDDEN_TOKENS = [
    # Emotion tokens - control TTS emotional tone
    "emotion_start", "emotion_end",
    "emotion_neutral", "emotion_happy", "emotion_sad", "emotion_angry",
    "emotion_surprised", "emotion_fearful", "emotion_disgusted", "emotion_calm",
    "emotion_excited", "emotion_curious", "emotion_sympathetic", "emotion_confident",
    # Prosody tokens - control TTS speech characteristics
    "prosody_start", "prosody_end",
    "prosody_fast", "prosody_slow", "prosody_normal_speed",
    "prosody_loud", "prosody_soft", "prosody_whisper", "prosody_normal_volume",
    "prosody_high_pitch", "prosody_low_pitch", "prosody_normal_pitch",
    "prosody_emphasis", "prosody_pause_short", "prosody_pause_long",
    # Internal verification tokens - used for self-checking but not shown to users
    "verify_start", "verify_end",
    "fact_check", "self_correct",
    "correction_start", "correction_end",
    # Internal grounding markers
    "grounded", "ungrounded",
    "within_knowledge", "beyond_knowledge", "knowledge_cutoff",
    # Internal control signals
    "encoder_start", "encoder_end",
    "decoder_start", "decoder_end",
    "projection_start", "projection_end",
    "state_begin", "state_end",
    "modal_switch", "modal_switch_end",
]

# Sequence control tokens
SEQUENCE_TOKENS = {
    "bos": "bos",
    "eos": "eos", 
    "pad": "pad",
    "prompt_start": "prompt_start",
    "prompt_end": "prompt_end",
    "text_start": "text_start",
    "text_end": "text_end",
    "response_start": "response_start",
    "response_end": "response_end",
}


def get_special_tokens_list():
    """Get list of all special token values."""
    return list(SPECIAL_TOKENS.values())


def get_reasoning_tokens():
    """Get all reasoning-related tokens for chain-of-thought training."""
    reasoning_keys = [
        'think_start', 'think_end',
        'observation_start', 'observation_end',
        'note_start', 'note_end',
        'step_start', 'step_end',
        'reflection_start', 'reflection_end',
        'hypothesis_start', 'hypothesis_end',
        'conclusion_start', 'conclusion_end',
        # Enhanced reasoning tokens
        'plan_start', 'plan_end',
        'plan_step', 'plan_step_end',
        'critique_start', 'critique_end',
        'error_found', 'no_error',
        'analysis_start', 'analysis_end',
        'breakdown_start', 'breakdown_end',
        'decision_start', 'decision_end',
        'option_start', 'option_end',
        'chosen', 'rejected',
        'because', 'therefore', 'however', 'alternatively',
    ]
    return {k: SPECIAL_TOKENS[k] for k in reasoning_keys if k in SPECIAL_TOKENS}


def get_all_reasoning_block_tokens():
    """
    Get all reasoning block start/end token pairs for weighted loss computation.
    
    Returns a list of (start_key, end_key) tuples for all reasoning blocks
    that should receive higher loss weight during training.
    """
    return [
        # Primary reasoning blocks
        ('think_start', 'think_end'),
        ('plan_start', 'plan_end'),
        ('critique_start', 'critique_end'),
        ('analysis_start', 'analysis_end'),
        ('breakdown_start', 'breakdown_end'),
        ('decision_start', 'decision_end'),
        # Secondary reasoning elements
        ('observation_start', 'observation_end'),
        ('reflection_start', 'reflection_end'),
        ('hypothesis_start', 'hypothesis_end'),
        ('conclusion_start', 'conclusion_end'),
        # Verification blocks
        ('verify_start', 'verify_end'),
    ]


def get_tool_block_tokens():
    """
    Get tool/function calling block token pairs for weighted loss computation.
    
    Tool calling is critical for agentic behavior, so these tokens
    should receive higher loss weight during training.
    """
    return [
        # Tool calling blocks
        ('tool_call_start', 'tool_call_end'),
        ('tool_result_start', 'tool_result_end'),
        ('function_name_start', 'function_name_end'),
        ('function_args_start', 'function_args_end'),
        ('tool_calls_start', 'tool_calls_end'),
        # Available tools blocks
        ('available_tools_start', 'available_tools_end'),
        ('tool_def_start', 'tool_def_end'),
        ('tool_params_start', 'tool_params_end'),
    ]


def get_anti_hallucination_block_tokens():
    """
    Get anti-hallucination block token pairs for weighted loss computation.
    
    These tokens help the model express uncertainty and cite sources,
    which is critical for reducing hallucinations.
    """
    return [
        # Uncertainty blocks
        ('uncertain_start', 'uncertain_end'),
        ('verify_start', 'verify_end'),
        ('correction_start', 'correction_end'),
        # Citation blocks
        ('cite_start', 'cite_end'),
        ('source_start', 'source_end'),
        ('reference_start', 'reference_end'),
        ('quote_start', 'quote_end'),
        # Grounding blocks
        ('context_start', 'context_end'),
        ('retrieved_start', 'retrieved_end'),
    ]


def get_code_execution_block_tokens():
    """
    Get code execution block token pairs for weighted loss computation.
    
    Code execution is critical for agentic coding tasks.
    """
    return [
        # Execution blocks
        ('exec_start', 'exec_end'),
        ('exec_result', 'exec_result_end'),
        ('exec_error', 'exec_error_end'),
        # Jupyter blocks
        ('jupyter_start', 'jupyter_end'),
        ('jupyter_code', 'jupyter_code_end'),
        ('jupyter_output', 'jupyter_output_end'),
        ('jupyter_error', 'jupyter_error_end'),
        # Code blocks
        ('code_start', 'code_end'),
    ]


def get_all_weighted_block_tokens():
    """
    Get ALL block token pairs that should receive higher loss weight.
    
    This combines reasoning, tool calling, anti-hallucination, and code execution
    blocks into a single list for comprehensive weighted loss computation.
    
    Returns:
        Dict mapping block category to list of (start_key, end_key) tuples
    """
    return {
        'reasoning': get_all_reasoning_block_tokens(),
        'tool_calling': get_tool_block_tokens(),
        'anti_hallucination': get_anti_hallucination_block_tokens(),
        'code_execution': get_code_execution_block_tokens(),
    }


def get_flat_weighted_block_tokens():
    """
    Get a flat list of all block token pairs that should receive higher loss weight.
    
    Returns:
        List of (start_key, end_key) tuples
    """
    all_blocks = get_all_weighted_block_tokens()
    flat_list = []
    for blocks in all_blocks.values():
        flat_list.extend(blocks)
    return flat_list


def get_memory_tokens():
    """Get all memory and context management tokens."""
    memory_keys = [
        'memory_start', 'memory_end',
        'working_memory_start', 'working_memory_end',
        'long_term_memory_start', 'long_term_memory_end',
        'summary_start', 'summary_end',
        'condensed_start', 'condensed_end',
        'user_profile_start', 'user_profile_end',
        'user_preference_start', 'user_preference_end',
        'hard_rule_start', 'hard_rule_end',
        'session_start', 'session_end',
        'conversation_history_start', 'conversation_history_end',
    ]
    return {k: SPECIAL_TOKENS[k] for k in memory_keys if k in SPECIAL_TOKENS}


def get_temporal_tokens():
    """Get all temporal/video navigation tokens."""
    temporal_keys = [
        'timestamp_start', 'timestamp_end',
        'time_range_start', 'time_range_end',
        'keyframe', 'keyframe_start', 'keyframe_end',
        'scene_change', 'scene_start', 'scene_end',
        'segment_start', 'segment_end',
        'frame_start', 'frame_end',
        'frame_num', 'frame_num_end',
        'action_start', 'action_end',
        'event_start', 'event_end',
        'region_start', 'region_end',
        'bbox_start', 'bbox_end',
        'object_start', 'object_end',
    ]
    return {k: SPECIAL_TOKENS[k] for k in temporal_keys if k in SPECIAL_TOKENS}


def get_structured_data_tokens():
    """Get all structured data tokens (tables, schema, etc.)."""
    structured_keys = [
        'table_start', 'table_end',
        'table_row_start', 'table_row_end',
        'table_cell_start', 'table_cell_end',
        'table_header_start', 'table_header_end',
        'table_body_start', 'table_body_end',
        'schema_start', 'schema_end',
        'schema_field', 'schema_field_end',
        'schema_type', 'schema_type_end',
        'schema_required', 'schema_optional',
        'version', 'version_end',
        'deprecated', 'since', 'since_end',
        'json_start', 'json_end',
        'xml_start', 'xml_end',
        'yaml_start', 'yaml_end',
        'csv_start', 'csv_end',
        'list_start', 'list_end',
        'list_item', 'list_item_end',
        'kv_start', 'kv_end',
        'key_start', 'key_end',
        'value_start', 'value_end',
    ]
    return {k: SPECIAL_TOKENS[k] for k in structured_keys if k in SPECIAL_TOKENS}


def get_uncertainty_tokens():
    """Get all uncertainty and anti-hallucination tokens."""
    uncertainty_keys = [
        'confidence_high', 'confidence_medium', 'confidence_low', 'confidence_uncertain',
        'uncertainty_score', 'uncertainty_score_end',
        'certainty_score', 'certainty_score_end',
        'uncertain_start', 'uncertain_end',
        'unknown', 'need_verification', 'may_be_outdated',
        'speculative', 'approximate',
        'verify_start', 'verify_end',
        'fact_check', 'self_correct',
        'correction_start', 'correction_end',
        'retract', 'clarify',
        'cite_start', 'cite_end',
        'source_start', 'source_end',
        'reference_start', 'reference_end',
        'quote_start', 'quote_end',
        'attribution', 'attribution_end',
        'context_start', 'context_end',
        'retrieved_start', 'retrieved_end',
        'grounded', 'ungrounded', 'partially_grounded',
        'knowledge_cutoff', 'beyond_knowledge', 'within_knowledge',
        'knowledge_date', 'knowledge_date_end',
    ]
    return {k: SPECIAL_TOKENS[k] for k in uncertainty_keys if k in SPECIAL_TOKENS}


def get_hidden_tokens():
    """Get all hidden tokens that should be stripped from user-visible output.
    
    Includes:
    - Emotion tokens (TTS emotional tone control)
    - Prosody tokens (TTS speech characteristics)
    - Internal verification tokens (fact_check, self_correct, etc.)
    - Internal grounding markers (grounded, ungrounded, knowledge boundaries)
    - Internal control signals (encoder, decoder, projection, modal_switch)
    """
    return {k: SPECIAL_TOKENS[k] for k in HIDDEN_TOKENS if k in SPECIAL_TOKENS}


def get_emotion_tokens():
    """Get all emotion tokens for TTS."""
    emotion_keys = [k for k in SPECIAL_TOKENS.keys() if k.startswith('emotion_')]
    return {k: SPECIAL_TOKENS[k] for k in emotion_keys}


def get_prosody_tokens():
    """Get all prosody tokens for TTS."""
    prosody_keys = [k for k in SPECIAL_TOKENS.keys() if k.startswith('prosody_')]
    return {k: SPECIAL_TOKENS[k] for k in prosody_keys}


def get_sequence_tokens():
    """Get sequence control tokens (BOS, EOS, etc.)."""
    return {k: SPECIAL_TOKENS[v] for k, v in SEQUENCE_TOKENS.items()}


def strip_hidden_tokens(text: str) -> str:
    """
    Remove hidden tokens from text for user display.
    
    Strips all tokens in HIDDEN_TOKENS including:
    - Emotion/prosody tokens (TTS control)
    - Internal verification tokens (fact_check, self_correct, etc.)
    - Internal grounding markers (grounded, ungrounded, knowledge boundaries)
    - Internal control signals (encoder, decoder, projection, modal_switch)
    """
    import re
    hidden_values = get_hidden_tokens().values()
    for token in hidden_values:
        text = text.replace(token, '')
    # Clean up any double spaces left behind
    text = re.sub(r' +', ' ', text)
    return text.strip()


def get_document_tokens():
    """Get all document/file handling tokens."""
    doc_keys = [k for k in SPECIAL_TOKENS.keys() if k.startswith(('doc_', 'file_', 'filename_', 'filepath_', 'eod'))]
    return {k: SPECIAL_TOKENS[k] for k in doc_keys}


def get_code_language_tokens():
    """Get all programming language tokens for code blocks."""
    lang_keys = [k for k in SPECIAL_TOKENS.keys() if k.startswith('lang_')]
    return {k: SPECIAL_TOKENS[k] for k in lang_keys}


def get_tool_tokens():
    """Get all tool/function calling related tokens."""
    tool_keys = [
        'tool_call_start', 'tool_call_end',
        'tool_result_start', 'tool_result_end',
        'function_name_start', 'function_name_end',
        'function_args_start', 'function_args_end',
        'arg_name_start', 'arg_name_end',
        'arg_value_start', 'arg_value_end',
        'tools_start', 'tools_end',
        'function_def_start', 'function_def_end',
        'tool_error_start', 'tool_error_end',
        'tool_success',
        'tool_calls_start', 'tool_calls_end',
        'tool_id_start', 'tool_id_end',
        # Available tools tokens
        'available_tools_start', 'available_tools_end',
        'tool_def_start', 'tool_def_end',
        'tool_name', 'tool_name_end',
        'tool_description', 'tool_description_end',
        'tool_params_start', 'tool_params_end',
        'param_name', 'param_name_end',
        'param_type', 'param_type_end',
        'param_required', 'param_optional',
    ]
    return {k: SPECIAL_TOKENS[k] for k in tool_keys if k in SPECIAL_TOKENS}


def get_available_tools_tokens():
    """Get tokens specifically for available tools listing."""
    keys = [
        'available_tools_start', 'available_tools_end',
        'tool_def_start', 'tool_def_end',
        'tool_name', 'tool_name_end',
        'tool_description', 'tool_description_end',
        'tool_params_start', 'tool_params_end',
        'param_name', 'param_name_end',
        'param_type', 'param_type_end',
        'param_required', 'param_optional',
    ]
    return {k: SPECIAL_TOKENS[k] for k in keys if k in SPECIAL_TOKENS}


def get_fim_tokens():
    """Get Fill-In-The-Middle tokens for code completion."""
    fim_keys = ['fim_prefix', 'fim_middle', 'fim_suffix', 'fim_pad']
    return {k: SPECIAL_TOKENS[k] for k in fim_keys if k in SPECIAL_TOKENS}


def get_git_tokens():
    """Get version control / Git integration tokens."""
    git_keys = [
        'commit_before', 'commit_before_end',
        'commit_after', 'commit_after_end',
        'commit_msg', 'commit_msg_end',
        'diff_start', 'diff_end',
        'diff_add', 'diff_del', 'diff_context',
        'reponame', 'reponame_end',
        'gh_stars', 'gh_stars_end',
        'branch', 'branch_end',
        'issue_start', 'issue_end',
        'issue_title', 'issue_title_end',
        'issue_body', 'issue_body_end',
        'issue_closed', 'issue_open',
        'pr_start', 'pr_end',
        'pr_title', 'pr_title_end',
        'pr_body', 'pr_body_end',
        'pr_merged', 'pr_closed',
    ]
    return {k: SPECIAL_TOKENS[k] for k in git_keys if k in SPECIAL_TOKENS}


def get_jupyter_tokens():
    """Get Jupyter/code execution tokens."""
    jupyter_keys = [
        'jupyter_start', 'jupyter_end',
        'jupyter_code', 'jupyter_code_end',
        'jupyter_output', 'jupyter_output_end',
        'jupyter_text', 'jupyter_text_end',
        'jupyter_error', 'jupyter_error_end',
        'empty_output',
        'exec_start', 'exec_end',
        'exec_result', 'exec_result_end',
        'exec_error', 'exec_error_end',
        'exec_timeout',
    ]
    return {k: SPECIAL_TOKENS[k] for k in jupyter_keys if k in SPECIAL_TOKENS}


def get_file_operation_tokens():
    """Get file system operation tokens for agentic coding."""
    file_keys = [
        'add_file', 'add_file_end',
        'delete_file', 'delete_file_end',
        'rename_file', 'rename_file_end',
        'edit_file', 'edit_file_end',
        'read_file', 'read_file_end',
        'file_sep', 'file_content', 'file_content_end',
        'edit_range', 'edit_range_end',
        'line_num', 'line_num_end',
        'insert_before', 'insert_after',
        'replace', 'replace_end',
    ]
    return {k: SPECIAL_TOKENS[k] for k in file_keys if k in SPECIAL_TOKENS}


def get_internal_control_tokens():
    """Get internal control/embedding marker tokens."""
    control_keys = [
        'encoder_start', 'encoder_end',
        'decoder_start', 'decoder_end',
        'projection_start', 'projection_end',
        'state_begin', 'state_end',
        'modal_switch', 'modal_switch_end',
    ]
    return {k: SPECIAL_TOKENS[k] for k in control_keys if k in SPECIAL_TOKENS}


def get_anti_hallucination_tokens():
    """Get all anti-hallucination and uncertainty tokens."""
    ah_keys = [
        # Confidence levels
        'confidence_high', 'confidence_medium', 'confidence_low', 'confidence_uncertain',
        # Uncertainty markers
        'uncertain_start', 'uncertain_end', 'unknown', 'need_verification', 'may_be_outdated',
        # Self-verification
        'verify_start', 'verify_end', 'fact_check', 'self_correct',
        'correction_start', 'correction_end',
        # Citation/Source
        'cite_start', 'cite_end', 'source_start', 'source_end',
        'reference_start', 'reference_end', 'quote_start', 'quote_end',
        # Grounding
        'context_start', 'context_end', 'retrieved_start', 'retrieved_end',
        'grounded', 'ungrounded',
        # Knowledge boundaries
        'knowledge_cutoff', 'beyond_knowledge', 'within_knowledge',
    ]
    return {k: SPECIAL_TOKENS[k] for k in ah_keys if k in SPECIAL_TOKENS}


def get_confidence_tokens():
    """Get confidence level tokens."""
    conf_keys = ['confidence_high', 'confidence_medium', 'confidence_low', 'confidence_uncertain']
    return {k: SPECIAL_TOKENS[k] for k in conf_keys if k in SPECIAL_TOKENS}


def get_citation_tokens():
    """Get citation and source grounding tokens."""
    cite_keys = [
        'cite_start', 'cite_end', 'source_start', 'source_end',
        'reference_start', 'reference_end', 'quote_start', 'quote_end',
        'context_start', 'context_end', 'retrieved_start', 'retrieved_end',
    ]
    return {k: SPECIAL_TOKENS[k] for k in cite_keys if k in SPECIAL_TOKENS}


def print_special_tokens():
    """Print all special tokens."""
    print(f"✅ {len(SPECIAL_TOKENS)} special tokens defined:")
    categories = {
        'Sequence Control': ['bos', 'eos', 'pad', 'prompt_start', 'prompt_end', 
                            'text_start', 'text_end', 'response_start', 'response_end'],
        'Conversation': ['system_start', 'system_end', 'user_start', 'user_end', 
                        'assistant_start', 'assistant_end'],
        'FIM (Code Completion)': ['fim_prefix', 'fim_middle', 'fim_suffix', 'fim_pad'],
        'Git/Version Control': ['commit_before', 'commit_after', 'commit_msg', 
                               'diff_start', 'diff_add', 'diff_del', 'reponame', 'gh_stars'],
        'Issues/PRs': ['issue_start', 'issue_title', 'issue_closed', 'issue_open',
                      'pr_start', 'pr_title', 'pr_merged', 'pr_closed'],
        'Jupyter/Execution': ['jupyter_start', 'jupyter_code', 'jupyter_output', 
                             'jupyter_error', 'empty_output', 'exec_start', 'exec_result'],
        'File Operations': ['add_file', 'delete_file', 'rename_file', 'edit_file', 
                           'read_file', 'file_sep', 'file_content'],
        'Edit Operations': ['edit_range', 'line_num', 'insert_before', 'insert_after', 'replace'],
        'Internal Control': ['encoder_start', 'decoder_start', 'projection_start', 
                            'state_begin', 'state_end', 'modal_switch'],
        'Document/File': ['doc_start', 'doc_end', 'eod', 'filename_start', 'filename_end',
                         'filepath_start', 'filepath_end'],
        'File Types': ['file_txt', 'file_md', 'file_json', 'file_xml', 'file_yaml', 
                      'file_html', 'file_css', 'file_csv', 'file_toml', 'file_ini', 'file_log'],
        'Multimodal': ['image_start', 'image_end', 'video_start', 'video_end'],
        'Generation': ['gen_image_start', 'gen_image_end', 'gen_video_start', 'gen_video_end'],
        'Tools (Basic)': ['tool_call_start', 'tool_call_end', 'tool_result_start', 'tool_result_end'],
        'Tools (Structured)': ['function_name_start', 'function_name_end', 
                               'function_args_start', 'function_args_end',
                               'arg_name_start', 'arg_name_end', 'arg_value_start', 'arg_value_end'],
        'Tools (Schema)': ['tools_start', 'tools_end', 'function_def_start', 'function_def_end'],
        'Tools (Status)': ['tool_error_start', 'tool_error_end', 'tool_success',
                          'tool_calls_start', 'tool_calls_end', 'tool_id_start', 'tool_id_end'],
        'Code': ['code_start', 'code_end'],
        'Languages': ['lang_python', 'lang_javascript', 'lang_typescript', 'lang_java', 
                     'lang_cpp', 'lang_c', 'lang_go', 'lang_rust', 'lang_shell', 'lang_sql'],
        'Reasoning': ['think_start', 'think_end', 'observation_start', 'observation_end', 
                      'note_start', 'note_end', 'step_start', 'step_end',
                      'reflection_start', 'reflection_end', 'hypothesis_start', 'hypothesis_end',
                      'conclusion_start', 'conclusion_end'],
        'Voice': ['listen_start', 'listen_end', 'speak_start', 'speak_end', 'audio_start', 'audio_end'],
        'Emotion (Hidden)': ['emotion_neutral', 'emotion_happy', 'emotion_sad', 'emotion_angry',
                            'emotion_surprised', 'emotion_calm', 'emotion_excited', 'emotion_curious'],
        'Prosody (Hidden)': ['prosody_fast', 'prosody_slow', 'prosody_loud', 'prosody_soft',
                            'prosody_high_pitch', 'prosody_low_pitch', 'prosody_emphasis'],
    }
    for cat, keys in categories.items():
        tokens = [SPECIAL_TOKENS[k] for k in keys if k in SPECIAL_TOKENS]
        if tokens:
            print(f"   {cat}: {', '.join(tokens)}")
