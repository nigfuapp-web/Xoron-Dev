"""Dataset configurations for Xoron-Dev multimodal training."""

from typing import Dict, List, Callable, Any, Optional
import copy
import os

# Dataset configurations organized by category
DATASET_CONFIGS: Dict[str, List[Dict[str, Any]]] = {
    # === TEXT CAPABILITIES ===
    "code": [
        {"name": "Code-Feedback", "path": "m-a-p/Code-Feedback", "split": "train", "streaming": True},
        {"name": "Python-Code-18k", "path": "iamtarun/python_code_instructions_18k_alpaca", "split": "train", "streaming": True},
        {"name": "CodeParrot-Clean", "path": "codeparrot/codeparrot-clean", "split": "train", "streaming": True},
        {"name": "HumanEval-Python", "path": "bigcode/humanevalpack", "config": "python", "split": "test", "streaming": True},
        {"name": "HumanEval-JavaScript", "path": "bigcode/humanevalpack", "config": "js", "split": "test", "streaming": True},
        {"name": "HumanEval-Java", "path": "bigcode/humanevalpack", "config": "java", "split": "test", "streaming": True},
        {"name": "HumanEval-CPP", "path": "bigcode/humanevalpack", "config": "cpp", "split": "test", "streaming": True},
        {"name": "HumanEval-Rust", "path": "bigcode/humanevalpack", "config": "rust", "split": "test", "streaming": True},
        {"name": "HumanEval-Go", "path": "bigcode/humanevalpack", "config": "go", "split": "test", "streaming": True},
        {"name": "Jupyter-Code", "path": "loubnabnl/github-jupyter-code-to-text", "split": "train", "streaming": True},
    ],
    "conversation": [
        {"name": "Dolly-15k", "path": "databricks/databricks-dolly-15k", "split": "train", "streaming": True},
        {"name": "OpenAssistant", "path": "OpenAssistant/oasst1", "split": "train", "streaming": True},
        {"name": "NoRobots", "path": "HuggingFaceH4/no_robots", "split": "train", "streaming": True},
        {"name": "OpenOrca", "path": "Open-Orca/OpenOrca", "split": "train", "streaming": True},
    ],
    "tool_use": [
        {"name": "Function-Calling-ChatML", "path": "Locutusque/function-calling-chatml", "split": "train", "streaming": True},
        {"name": "Pythonic-Function-Calling", "path": "driaforall/pythonic-function-calling", "split": "train", "streaming": True},
        {"name": "Synth-APIGen", "path": "argilla/Synth-APIGen-v0.1", "split": "train", "streaming": True},
        {"name": "Tool-Calls-SingleTurn", "path": "interstellarninja/tool-calls-singleturn", "split": "train", "streaming": True},
    ],
    "agentic": [
        {"name": "WildChat", "path": "allenai/WildChat-1M", "split": "train", "streaming": True},
        {"name": "AgentInstruct", "path": "THUDM/AgentInstruct", "split": "os", "streaming": True},
        {"name": "Glaive-Code-Assistant", "path": "glaiveai/glaive-code-assistant-v2", "split": "train", "streaming": True},
        {"name": "UltraChat", "path": "stingning/ultrachat", "split": "train", "streaming": True},
        {"name": "ShareGPT-Clean", "path": "RyokoAI/ShareGPT52K", "split": "train", "streaming": True},
    ],
    
    # === CHAIN-OF-THOUGHT REASONING ===
    "chain_of_thought": [
        # Synthetic CoT dataset (generated locally)
        {
            "name": "Synth-CoT",
            "path": "synth/data/cot_dataset.jsonl",
            "split": "train",
            "streaming": False,
            "local": True,
            "format": "jsonl"
        },
    ],
    
    # === ANTI-HALLUCINATION TRAINING ===
    "anti_hallucination": [
        {
            "name": "Synth-IDK",
            "path": "synth/data/idk_dataset.jsonl",
            "split": "train",
            "streaming": False,
            "local": True,
            "format": "jsonl"
        },
        {
            "name": "Synth-KnowledgeCutoff",
            "path": "synth/data/knowledge_cutoff_dataset.jsonl",
            "split": "train",
            "streaming": False,
            "local": True,
            "format": "jsonl"
        },
        {
            "name": "Synth-FactCheck",
            "path": "synth/data/fact_check_dataset.jsonl",
            "split": "train",
            "streaming": False,
            "local": True,
            "format": "jsonl"
        },
        {
            "name": "Synth-GroundedResponse",
            "path": "synth/data/grounded_response_dataset.jsonl",
            "split": "train",
            "streaming": False,
            "local": True,
            "format": "jsonl"
        },
        {
            "name": "Synth-SelfCorrection",
            "path": "synth/data/self_correction_dataset.jsonl",
            "split": "train",
            "streaming": False,
            "local": True,
            "format": "jsonl"
        },
        {
            "name": "Synth-ConfidenceLevel",
            "path": "synth/data/confidence_level_dataset.jsonl",
            "split": "train",
            "streaming": False,
            "local": True,
            "format": "jsonl"
        },
        {
            "name": "Synth-Uncertainty",
            "path": "synth/data/uncertainty_dataset.jsonl",
            "split": "train",
            "streaming": False,
            "local": True,
            "format": "jsonl"
        },
        {
            "name": "Synth-Citation",
            "path": "synth/data/citation_dataset.jsonl",
            "split": "train",
            "streaming": False,
            "local": True,
            "format": "jsonl"
        },
        {
            "name": "Synth-RetrievalGrounded",
            "path": "synth/data/retrieval_grounded_dataset.jsonl",
            "split": "train",
            "streaming": False,
            "local": True,
            "format": "jsonl"
        },
    ],
    
    # === DOCUMENT HANDLING ===
    "document": [
        # Synthetic document handling dataset (generated locally)
        {
            "name": "Synth-Documents",
            "path": "synth/data/document_dataset.jsonl",
            "split": "train",
            "streaming": False,
            "local": True,
            "format": "jsonl"
        },
    ],

    # === AGENTIC CODING CAPABILITIES ===
    # Fill-In-The-Middle (code completion)
    "fim": [
        {
            "name": "Synth-FIM",
            "path": "synth/data/fim_dataset.jsonl",
            "split": "train",
            "streaming": False,
            "local": True,
            "format": "jsonl"
        },
    ],
    
    # Git/Version Control
    "git_operations": [
        {
            "name": "Synth-Commits",
            "path": "synth/data/commit_dataset.jsonl",
            "split": "train",
            "streaming": False,
            "local": True,
            "format": "jsonl"
        },
        {
            "name": "Synth-Diffs",
            "path": "synth/data/diff_dataset.jsonl",
            "split": "train",
            "streaming": False,
            "local": True,
            "format": "jsonl"
        },
        {
            "name": "Synth-Issues",
            "path": "synth/data/issue_dataset.jsonl",
            "split": "train",
            "streaming": False,
            "local": True,
            "format": "jsonl"
        },
        {
            "name": "Synth-RepoContext",
            "path": "synth/data/repo_context_dataset.jsonl",
            "split": "train",
            "streaming": False,
            "local": True,
            "format": "jsonl"
        },
    ],
    
    # Code Execution (Jupyter/Interpreter)
    "code_execution": [
        {
            "name": "Synth-Jupyter",
            "path": "synth/data/jupyter_dataset.jsonl",
            "split": "train",
            "streaming": False,
            "local": True,
            "format": "jsonl"
        },
        {
            "name": "Synth-Execution",
            "path": "synth/data/execution_dataset.jsonl",
            "split": "train",
            "streaming": False,
            "local": True,
            "format": "jsonl"
        },
        {
            "name": "Synth-ShellExecution",
            "path": "synth/data/shell_execution_dataset.jsonl",
            "split": "train",
            "streaming": False,
            "local": True,
            "format": "jsonl"
        },
        {
            "name": "Synth-ShellErrors",
            "path": "synth/data/shell_error_dataset.jsonl",
            "split": "train",
            "streaming": False,
            "local": True,
            "format": "jsonl"
        },
        {
            "name": "Synth-ShellTimeout",
            "path": "synth/data/shell_timeout_dataset.jsonl",
            "split": "train",
            "streaming": False,
            "local": True,
            "format": "jsonl"
        },
        {
            "name": "Synth-MultiStepExecution",
            "path": "synth/data/multi_step_execution_dataset.jsonl",
            "split": "train",
            "streaming": False,
            "local": True,
            "format": "jsonl"
        },
        {
            "name": "Synth-PythonScripts",
            "path": "synth/data/python_script_dataset.jsonl",
            "split": "train",
            "streaming": False,
            "local": True,
            "format": "jsonl"
        },
        {
            "name": "Synth-Debugging",
            "path": "synth/data/debugging_dataset.jsonl",
            "split": "train",
            "streaming": False,
            "local": True,
            "format": "jsonl"
        },
    ],
    
    # File System Operations
    "file_operations": [
        {
            "name": "Synth-FileOps",
            "path": "synth/data/file_ops_dataset.jsonl",
            "split": "train",
            "streaming": False,
            "local": True,
            "format": "jsonl"
        },
        {
            "name": "Synth-EditLines",
            "path": "synth/data/edit_lines_dataset.jsonl",
            "split": "train",
            "streaming": False,
            "local": True,
            "format": "jsonl"
        },
    ],

    # === IMAGE CAPABILITIES ===
    "image_caption": [
        {"name": "Flickr8k", "path": "Naveengo/flickr8k", "split": "train", "streaming": True},
        {"name": "Football", "path": "ybelkada/football-dataset", "split": "train", "streaming": True},
        {"name": "NewYorker", "path": "jmhessel/newyorker_caption_contest", "config": "explanation", "split": "train", "streaming": True},
    ],
    "image_vqa": [
        {"name": "ScienceQA", "path": "derek-thomas/ScienceQA", "split": "train", "streaming": True, "filter_images": True},
    ],
    # image_generation: datasets with ACTUAL IMAGES for diffusion training
    # Note: Text-only prompt datasets moved to 'image_prompts' category (for LLM training)
    "image_generation": [
        # MagicBrush has source images + prompts (good for diffusion training)
        # For now, keep empty - diffusion trains on image_editing datasets which have images
    ],
    # image_prompts: Text-only datasets for teaching LLM to generate image prompts
    # These are processed as TEXT samples, not image samples (no diffusion training)
    "image_prompts": [
        {"name": "SD-Prompts", "path": "Gustavosta/Stable-Diffusion-Prompts", "split": "train", "streaming": True},
        {"name": "SD-Prompts-2M", "path": "FredZhang7/stable-diffusion-prompts-2.47M", "split": "train", "streaming": True},
        {"name": "Midjourney-Prompts", "path": "succinctly/midjourney-prompts", "split": "train", "streaming": True},
    ],
    "image_editing": [
        {"name": "MagicBrush", "path": "osunlp/MagicBrush", "split": "train", "streaming": True},
        {"name": "InstructPix2Pix", "path": "timbrooks/instructpix2pix-clip-filtered", "split": "train", "streaming": True},
    ],
    "ui_to_code": [
        {"name": "WebSight", "path": "HuggingFaceM4/WebSight", "split": "train", "streaming": True},
    ],

    # === VIDEO CAPABILITIES ===
    "video_caption": [
        {"name": "Video-MME", "path": "lmms-lab/Video-MME", "split": "test", "streaming": True},
    ],
    "video_qa": [
        {"name": "VideoInstruct-100K", "path": "MBZUAI/VideoInstruct-100K", "split": "train", "streaming": True},
    ],
    "video_generation": [
        # Rapidata Sora datasets - high quality with reliable video URLs
        {"name": "Sora-Physics-Likert", "path": "Rapidata/sora-video-generation-physics-likert-scoring", "split": "train", "streaming": True},
        {"name": "Sora-Style-Likert", "path": "Rapidata/sora-video-generation-style-likert-scoring", "split": "train", "streaming": True},
        {"name": "Sora-Alignment-Likert", "path": "Rapidata/sora-video-generation-alignment-likert-scoring", "split": "train", "streaming": True},
        # Original datasets (URLs may be less reliable)
        {"name": "WebVid-10M", "path": "TempoFunk/webvid-10M", "split": "train", "streaming": True},
        {"name": "Panda-70M", "path": "multimodalart/panda-70m", "split": "train", "streaming": True},
        {"name": "OpenVid-1M", "path": "nkp37/OpenVid-1M", "split": "train", "streaming": True},
        {"name": "VidProM", "path": "WenhaoWang/VidProM", "split": "train", "streaming": True},
    ],
    # Video preference datasets - comparing two videos for the same prompt
    "video_preference": [
        {"name": "T2V-Human-Preferences", "path": "Rapidata/text-2-video-human-preferences", "split": "train", "streaming": True},
        {"name": "T2V-Sora-Preferences-2", "path": "Rapidata/text-2-video-human-preferences-sora-2", "split": "train", "streaming": True},
    ],
    # Video quality scoring datasets
    "video_likert": [
        {"name": "Sora-Physics-Likert", "path": "Rapidata/sora-video-generation-physics-likert-scoring", "split": "train", "streaming": True},
        {"name": "Sora-Style-Likert", "path": "Rapidata/sora-video-generation-style-likert-scoring", "split": "train", "streaming": True},
        {"name": "Sora-Alignment-Likert", "path": "Rapidata/sora-video-generation-alignment-likert-scoring", "split": "train", "streaming": True},
    ],
    "image_to_video": [
        {"name": "TIP-I2V", "path": "WenhaoWang/TIP-I2V", "split": "Full", "streaming": True},
        {"name": "Pexels-I2V-350k", "path": "jovianzm/img2vid-pexels-350k", "split": "train", "streaming": True},
        {"name": "MiraData", "path": "TencentARC/MiraData", "split": "train", "streaming": True},
        {"name": "UltraVideo-Short", "path": "APRIL-AIGC/UltraVideo", "split": "short", "streaming": True},
        {"name": "Vript-Short", "path": "Mutonix/Vript", "config": "vript-short", "split": "train", "streaming": True},
    ],

    # === VOICE/SPEECH CAPABILITIES ===
    "voice_asr": [
        {"name": "LibriSpeech-Clean", "path": "openslr/librispeech_asr", "config": "clean", "split": "train.100", "streaming": True},
    ],
    "voice_tts": [
        {"name": "LibriTTS-R-Clean", "path": "blabble-io/libritts_r", "config": "clean", "split": "train.clean.100", "streaming": True},
        {"name": "MLS-Eng-10k", "path": "parler-tts/mls_eng_10k", "split": "train", "streaming": True},
    ],
}


# Modality groups for fine-tuning - maps modality to dataset categories
MODALITY_GROUPS: Dict[str, List[str]] = {
    # Text mode includes ALL text-based datasets including synth
    'text': [
        'code', 'conversation', 'tool_use', 'agentic',
        # Synth datasets for text training
        'chain_of_thought',      # CoT reasoning
        'anti_hallucination',    # IDK, fact-check, citations, etc.
        'document',              # Document handling
        'fim',                   # Fill-in-the-middle code completion
        'git_operations',        # Commits, diffs, issues, repo context
        'code_execution',        # Jupyter, execution traces
        'file_operations',       # File system operations
        'system_admin',          # Apt, Docker, SSH, databases, etc.
        'image_prompts',         # Text-only image generation prompts (trains LLM, not diffusion)
    ],
    'reasoning': ['chain_of_thought'],
    'anti_hallucination': ['anti_hallucination'],  # Critical for reducing hallucinations
    'agentic_coding': ['fim', 'git_operations', 'code_execution', 'file_operations', 'system_admin'],
    'image': ['image_caption', 'image_vqa', 'image_generation', 'image_editing', 'ui_to_code'],
    'video': ['video_caption', 'video_qa', 'video_generation', 'image_to_video', 'video_preference', 'video_likert'],
    'audio': ['voice_asr', 'voice_tts'],
}

# Reverse mapping: category to modality
CATEGORY_TO_MODALITY: Dict[str, str] = {}
for modality, categories in MODALITY_GROUPS.items():
    for cat in categories:
        CATEGORY_TO_MODALITY[cat] = modality


def filter_datasets_by_modalities(
    modalities: Optional[List[str]] = None,
    categories: Optional[List[str]] = None,
    exclude_categories: Optional[List[str]] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Filter dataset configurations by modality or category.
    
    Args:
        modalities: List of modalities to include ('text', 'image', 'video', 'audio').
                   If None, all modalities are included.
        categories: List of specific categories to include.
                   If None, all categories for the selected modalities are included.
        exclude_categories: List of categories to exclude.
        
    Returns:
        Filtered dataset configuration dictionary.
    """
    filtered = copy.deepcopy(DATASET_CONFIGS)
    
    # Determine which categories to include
    if modalities is not None:
        allowed_categories = set()
        for mod in modalities:
            if mod in MODALITY_GROUPS:
                allowed_categories.update(MODALITY_GROUPS[mod])
        
        # Remove categories not in allowed modalities
        for cat in list(filtered.keys()):
            if cat not in allowed_categories:
                filtered[cat] = []
    
    # Further filter by specific categories if provided
    if categories is not None:
        for cat in list(filtered.keys()):
            if cat not in categories:
                filtered[cat] = []
    
    # Exclude specific categories
    if exclude_categories is not None:
        for cat in exclude_categories:
            if cat in filtered:
                filtered[cat] = []
    
    return filtered


def get_finetune_datasets(finetune_mode: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Get dataset configuration for a specific fine-tuning mode.
    
    Args:
        finetune_mode: One of 'audio', 'video', 'image', 'text', 'vision' (image+video),
                      'generation' (image_gen + video_gen), or 'all'
                      
    Returns:
        Filtered dataset configuration.
    """
    if finetune_mode == 'all':
        return copy.deepcopy(DATASET_CONFIGS)
    
    if finetune_mode == 'audio':
        return filter_datasets_by_modalities(modalities=['audio'])
    
    if finetune_mode == 'video':
        return filter_datasets_by_modalities(modalities=['video'])
    
    if finetune_mode == 'image':
        return filter_datasets_by_modalities(modalities=['image'])
    
    if finetune_mode == 'text':
        return filter_datasets_by_modalities(modalities=['text'])
    
    if finetune_mode == 'vision':
        return filter_datasets_by_modalities(modalities=['image', 'video'])
    
    if finetune_mode == 'generation':
        return filter_datasets_by_categories(
            categories=['image_generation', 'video_generation', 'image_to_video']
        )
    
    # Default: return all
    return copy.deepcopy(DATASET_CONFIGS)


def filter_datasets_by_categories(categories: List[str]) -> Dict[str, List[Dict[str, Any]]]:
    """Filter to only include specific categories."""
    return filter_datasets_by_modalities(categories=categories)


def get_format_functions(formatter) -> Dict[str, Callable]:
    """Get format functions mapping from a MultimodalFormatter instance."""
    return {
        "code": formatter.format_code_sample,
        "conversation": formatter.format_conversation_sample,
        "tool_use": formatter.format_tool_use_sample,
        "agentic": formatter.format_agentic_sample,
        "chain_of_thought": formatter.format_chain_of_thought_sample,
        "image_caption": formatter.format_image_caption_sample,
        "image_vqa": formatter.format_image_vqa_sample,
        "video_caption": formatter.format_video_caption_sample,
        "video_qa": formatter.format_video_qa_sample,
        "video_generation": formatter.format_video_generation_sample,
        "image_to_video": formatter.format_image_to_video_sample,
        "image_generation": formatter.format_image_generation_sample,
        "image_prompts": formatter.format_image_generation_sample,  # Same format, processed as text
        "image_editing": formatter.format_image_editing_sample,
        "ui_to_code": formatter.format_ui_to_code_sample,
        "voice_asr": formatter.format_voice_asr_sample,
        "voice_tts": formatter.format_voice_tts_sample,
        # Document handling
        "document": formatter.format_document_sample,
        "multi_document": formatter.format_multi_document_sample,
        # Agentic coding capabilities - use agentic formatter for conversations
        "fim": formatter.format_passthrough_sample,
        "git_operations": formatter.format_passthrough_sample,
        "code_execution": formatter.format_passthrough_sample,
        "file_operations": formatter.format_agentic_sample,  # Has conversations format
        # Anti-hallucination training
        "anti_hallucination": formatter.format_passthrough_sample,
        # Video preference/quality datasets - use video_generation formatter
        "video_preference": formatter.format_video_generation_sample,
        "video_likert": formatter.format_video_generation_sample,
        # System admin tasks (apt, docker, ssh, databases, etc.)
        "system_admin": formatter.format_passthrough_sample,
        # Voice enhancement datasets
        "voice_emotion": formatter.format_voice_emotion_sample,
        "voice_singing": formatter.format_voice_singing_sample,
        "voice_beatbox": formatter.format_voice_beatbox_sample,
        "voice_interaction": formatter.format_voice_interaction_sample,
        "voice_expressive": formatter.format_voice_expressive_sample,
    }


def get_total_datasets() -> int:
    """Get total number of datasets."""
    return sum(len(v) for v in DATASET_CONFIGS.values())


def print_dataset_config():
    """Print dataset configuration summary."""
    total = get_total_datasets()
    print(f"✅ {len(DATASET_CONFIGS)} dataset categories configured")
    print(f"   Total datasets: {total}")
    print(f"   Code: {len(DATASET_CONFIGS['code'])} datasets")
    print(f"   Conversation: {len(DATASET_CONFIGS['conversation'])} datasets")
    print(f"   Tool-use: {len(DATASET_CONFIGS['tool_use'])} datasets")
    print(f"   Agentic: {len(DATASET_CONFIGS['agentic'])} datasets")
    print(f"   Chain-of-Thought: {len(DATASET_CONFIGS.get('chain_of_thought', []))} datasets")
    print(f"   Image Caption: {len(DATASET_CONFIGS['image_caption'])} datasets")
    print(f"   Image VQA: {len(DATASET_CONFIGS['image_vqa'])} datasets")
    print(f"   Image Generation: {len(DATASET_CONFIGS['image_generation'])} datasets")
    print(f"   Image Editing: {len(DATASET_CONFIGS['image_editing'])} datasets")
    print(f"   Video Caption: {len(DATASET_CONFIGS['video_caption'])} datasets")
    print(f"   Video QA: {len(DATASET_CONFIGS['video_qa'])} datasets")
    print(f"   Video Generation: {len(DATASET_CONFIGS['video_generation'])} datasets")
    print(f"   Image-to-Video: {len(DATASET_CONFIGS['image_to_video'])} datasets")
    print(f"   UI to Code: {len(DATASET_CONFIGS['ui_to_code'])} datasets")
    print(f"   Voice ASR: {len(DATASET_CONFIGS.get('voice_asr', []))} datasets")
    print(f"   Voice TTS: {len(DATASET_CONFIGS.get('voice_tts', []))} datasets")


def get_local_dataset_path(config: Dict[str, Any]) -> Optional[str]:
    """Get the full path for a local dataset."""
    if not config.get("local", False):
        return None
    
    path = config.get("path", "")
    if os.path.isabs(path):
        return path
    
    # Relative to project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(project_root, path)


def is_local_dataset_available(config: Dict[str, Any]) -> bool:
    """Check if a local dataset file exists."""
    if not config.get("local", False):
        return True  # Remote datasets are assumed available
    
    path = get_local_dataset_path(config)
    return path is not None and os.path.exists(path)


# === SYSTEM ADMINISTRATION DATASETS ===
DATASET_CONFIGS["system_admin"] = [
    {
        "name": "Synth-AptInstall",
        "path": "synth/data/apt_install_dataset.jsonl",
        "split": "train",
        "streaming": False,
        "local": True,
        "format": "jsonl"
    },
    {
        "name": "Synth-LanguageSetup",
        "path": "synth/data/language_setup_dataset.jsonl",
        "split": "train",
        "streaming": False,
        "local": True,
        "format": "jsonl"
    },
    {
        "name": "Synth-DesktopSetup",
        "path": "synth/data/desktop_setup_dataset.jsonl",
        "split": "train",
        "streaming": False,
        "local": True,
        "format": "jsonl"
    },
    {
        "name": "Synth-SSHSetup",
        "path": "synth/data/ssh_setup_dataset.jsonl",
        "split": "train",
        "streaming": False,
        "local": True,
        "format": "jsonl"
    },
    {
        "name": "Synth-Docker",
        "path": "synth/data/docker_dataset.jsonl",
        "split": "train",
        "streaming": False,
        "local": True,
        "format": "jsonl"
    },
    {
        "name": "Synth-DatabaseSetup",
        "path": "synth/data/database_setup_dataset.jsonl",
        "split": "train",
        "streaming": False,
        "local": True,
        "format": "jsonl"
    },
    {
        "name": "Synth-WebserverSetup",
        "path": "synth/data/webserver_setup_dataset.jsonl",
        "split": "train",
        "streaming": False,
        "local": True,
        "format": "jsonl"
    },
    {
        "name": "Synth-Downloads",
        "path": "synth/data/download_dataset.jsonl",
        "split": "train",
        "streaming": False,
        "local": True,
        "format": "jsonl"
    },
    {
        "name": "Synth-Monitoring",
        "path": "synth/data/monitoring_dataset.jsonl",
        "split": "train",
        "streaming": False,
        "local": True,
        "format": "jsonl"
    },
]

# === NEW DATASETS (Added) ===

# High-quality TTS dataset
DATASET_CONFIGS["voice_tts"].append({
    "name": "HiFi-TTS-Clean",
    "path": "MikhailT/hifi-tts",
    "config": "clean",
    "split": "train",
    "streaming": True
})

# Tool calling multi-turn conversations
DATASET_CONFIGS["tool_use"].append({
    "name": "Tool-Calls-Multiturn",
    "path": "interstellarninja/tool-calls-multiturn",
    "split": "train",
    "streaming": True
})

# Agentic Chain-of-Thought coding
DATASET_CONFIGS["agentic"].append({
    "name": "Agentic-CoT-Coding",
    "path": "AlicanKiraz0/Agentic-Chain-of-Thought-Coding-SFT-Dataset",
    "split": "train",
    "streaming": True
})

# File operations dataset
DATASET_CONFIGS["file_operations"].append({
    "name": "File-Operations-Medium",
    "path": "renjiepi/medium_20000-file_operations_n100k1",
    "split": "train",
    "streaming": True
})

# Swift code datasets
DATASET_CONFIGS["code"].extend([
    {
        "name": "Swift-Code-RLVR",
        "path": "saurabh5/rlvr-code-data-Swift",
        "split": "train",
        "streaming": True
    },
    {
        "name": "Swift-Code-Edit",
        "path": "finbarr/rlvr-code-data-swift-code-edit",
        "split": "train",
        "streaming": True
    },
])

# Go/Golang code datasets
DATASET_CONFIGS["code"].extend([
    {
        "name": "Golang-QA-2k",
        "path": "ExAi/Code-Golang-QA-2k",
        "split": "train",
        "streaming": True
    },
    {
        "name": "Golang-Coder",
        "path": "smcleod/golang-coder",
        "split": "train",
        "streaming": True
    },
])

# Conversation summarization
DATASET_CONFIGS["conversation"].append({
    "name": "Conversation-Summarization",
    "path": "abhi227070/converstion-to-summarization-dataset",
    "split": "train",
    "streaming": True
})

# Image-to-video preference datasets (has source image + prompt + video)
DATASET_CONFIGS["image_to_video"].extend([
    {
        "name": "I2V-Preference-Seedance",
        "path": "Rapidata/image-to-video-human-preference-seedance-1-pro",
        "split": "train",
        "streaming": True
    },
])

# === VOICE ENHANCEMENT DATASETS ===
# New categories for advanced voice capabilities
# All datasets verified to work with HuggingFace streaming (parquet format)

# Emotion Detection / Expressive Speech
# Using multi-speaker datasets with varied emotional content
DATASET_CONFIGS["voice_emotion"] = [
    # VoxPopuli - multi-speaker European Parliament recordings with varied emotions
    {
        "name": "VoxPopuli-EN",
        "path": "facebook/voxpopuli",
        "config": "en",
        "split": "train",
        "streaming": True,
        "description": "European Parliament speech with natural emotional variation"
    },
    # VCTK - multi-accent, multi-speaker (age/gender metadata for emotion inference)
    {
        "name": "VCTK-MultiSpeaker",
        "path": "sanchit-gandhi/vctk",
        "split": "train",
        "streaming": True,
        "description": "Multi-speaker multi-accent English speech"
    },
    # Peoples Speech - varied real-world speech with natural emotion
    {
        "name": "PeoplesSpeech-Clean",
        "path": "MLCommons/peoples_speech",
        "config": "clean",
        "split": "train",
        "streaming": True,
        "description": "Large-scale natural speech with emotional variety"
    },
]

# Singing Voice Synthesis / Music with Vocals
DATASET_CONFIGS["voice_singing"] = [
    # AudioSet - contains singing, music, and various vocal styles
    {
        "name": "AudioSet-Vocals",
        "path": "agkphysics/AudioSet",
        "split": "train",
        "streaming": True,
        "description": "Large-scale audio with singing and music labels"
    },
    # MusicCaps - music with detailed captions (singing descriptions)
    {
        "name": "MusicCaps",
        "path": "google/MusicCaps",
        "split": "train",
        "streaming": True,
        "description": "Music with detailed vocal/singing descriptions"
    },
]

# Vocal Sounds / Non-verbal Audio
DATASET_CONFIGS["voice_beatbox"] = [
    # AudioSet contains beatbox, vocal percussion, and sound effects
    {
        "name": "AudioSet-NonVerbal",
        "path": "agkphysics/AudioSet",
        "split": "train", 
        "streaming": True,
        "description": "Non-verbal sounds including beatbox, clicks, effects"
    },
]

# Speech Interaction / Turn-taking / Multi-speaker
DATASET_CONFIGS["voice_interaction"] = [
    # VoxPopuli has natural conversation/speech patterns
    {
        "name": "VoxPopuli-Interaction",
        "path": "facebook/voxpopuli",
        "config": "en",
        "split": "train",
        "streaming": True,
        "description": "Natural speech interaction patterns"
    },
    # VCTK for multi-speaker dialogue patterns
    {
        "name": "VCTK-Dialogue",
        "path": "sanchit-gandhi/vctk",
        "split": "train",
        "streaming": True,
        "description": "Multi-speaker dialogue for interaction modeling"
    },
]

# Expressive Speech / Prosody / TTS Quality
DATASET_CONFIGS["voice_expressive"] = [
    # Jenny TTS - high quality expressive TTS dataset
    {
        "name": "JennyTTS",
        "path": "reach-vb/jenny_tts_dataset",
        "split": "train",
        "streaming": True,
        "description": "High-quality expressive TTS recordings"
    },
    # VCTK multi-accent for prosodic variety
    {
        "name": "VCTK-Expressive",
        "path": "sanchit-gandhi/vctk",
        "split": "train",
        "streaming": True,
        "description": "Multi-accent expressive speech"
    },
    # MLS speaker descriptions for prosody training
    {
        "name": "MLS-SpeakerDescriptions",
        "path": "parler-tts/mls-eng-speaker-descriptions",
        "split": "train",
        "streaming": True,
        "description": "Speech with speaker style descriptions"
    },
]

# Update MODALITY_GROUPS to include new voice categories
MODALITY_GROUPS['audio'].extend([
    'voice_emotion', 'voice_singing', 'voice_beatbox',
    'voice_interaction', 'voice_expressive'
])

# Update CATEGORY_TO_MODALITY for new categories
for cat in ['voice_emotion', 'voice_singing', 'voice_beatbox', 
            'voice_interaction', 'voice_expressive']:
    CATEGORY_TO_MODALITY[cat] = 'audio'
