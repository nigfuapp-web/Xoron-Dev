# 📊 Data Module Documentation

The Data module handles dataset loading, processing, and formatting for multimodal training.

## 📁 File Structure

```
data/
├── __init__.py
├── dataset.py      # TrueStreamingDataset
├── formatters.py   # Format functions for each dataset type
└── processors.py   # Media processing utilities
```

---

## 🌊 TrueStreamingDataset

### Overview

`TrueStreamingDataset` is an `IterableDataset` that streams data from multiple sources without loading everything into memory.

### Key Features

- **True Streaming**: No data stored in memory
- **Round-Robin**: Iterates across all dataset sources
- **On-Demand Processing**: Media processed only when accessed
- **Resume Support**: Can resume from saved state
- **Multi-Source**: Supports HuggingFace and local JSONL files

### Initialization

```python
class TrueStreamingDataset(IterableDataset):
    def __init__(
        self,
        dataset_configs: Dict[str, List[Dict]],  # From config
        format_functions: Dict[str, Callable],    # Formatters
        tokenizer,                                # Tokenizer
        tokens: Dict[str, str],                   # Special tokens
        image_processor,                          # Vision processor
        max_length: int = 1024,                   # Max sequence length
        max_per_epoch: int = 2000,                # Samples per epoch
        max_per_dataset: int = 100,               # Per-dataset limit
        sample_repeat: int = 2,                   # Repeat factor
        voice_processor=None,                     # Audio processor
        max_video_frames: int = 32,               # Video frames
        video_size: int = 256,                    # Video resolution
        image_size: int = 256,                    # Image resolution
        audio_n_mels: int = 80,                   # Mel bins
        audio_max_length: int = 1000,             # Audio length
        audio_sample_rate: int = 16000,           # Sample rate
        resume_state_path: str = None,            # Resume state
        use_raw_waveform: bool = True,            # Raw audio mode
        modality_max_values: Dict[str, int] = None,  # Per-modality limits
    ):
```

### Streaming Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    TrueStreamingDataset                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Dataset Sources (initialized lazily)                          │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐              │
│  │ HF Dataset 1│ │ HF Dataset 2│ │ Local JSONL │ ...          │
│  │ (streaming) │ │ (streaming) │ │ (generator) │              │
│  └──────┬──────┘ └──────┬──────┘ └──────┬──────┘              │
│         │               │               │                      │
│         └───────────────┼───────────────┘                      │
│                         │                                      │
│                         ▼                                      │
│              ┌─────────────────────┐                           │
│              │   Round-Robin       │                           │
│              │   Iterator          │                           │
│              └──────────┬──────────┘                           │
│                         │                                      │
│                         ▼                                      │
│              ┌─────────────────────┐                           │
│              │   Format Function   │                           │
│              │   (per dtype)       │                           │
│              └──────────┬──────────┘                           │
│                         │                                      │
│                         ▼                                      │
│              ┌─────────────────────┐                           │
│              │   Media Processing  │                           │
│              │   (on-demand)       │                           │
│              └──────────┬──────────┘                           │
│                         │                                      │
│                         ▼                                      │
│              ┌─────────────────────┐                           │
│              │   Tokenization      │                           │
│              └──────────┬──────────┘                           │
│                         │                                      │
│                         ▼                                      │
│                   Yield Sample                                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Iterator Implementation

```python
def __iter__(self):
    """Yield samples in round-robin fashion."""
    unique_samples = 0
    total_yields = 0
    dataset_counts = {src['name']: 0 for src in self._dataset_sources}
    modality_counts = {'text': 0, 'image': 0, 'video': 0, 'audio': 0}
    
    # Initialize iterators for each source
    iterators = {}
    for src in self._dataset_sources:
        iterators[src['name']] = self._get_iterator(src)
    
    while unique_samples < self.max_per_epoch:
        # Round-robin through sources
        for src in self._dataset_sources:
            name = src['name']
            dtype = src['dtype']
            modality = self._get_modality(dtype)
            
            # Check per-dataset limit
            if dataset_counts[name] >= self.max_per_dataset:
                continue
            
            # Check per-modality limit (for dual training)
            if self.is_dual_training:
                if modality_counts[modality] >= self.modality_max_values.get(modality, float('inf')):
                    continue
            
            # Get next sample
            try:
                raw_sample = next(iterators[name])
            except StopIteration:
                # Reinitialize iterator
                iterators[name] = self._get_iterator(src)
                raw_sample = next(iterators[name])
            
            # Format sample
            format_fn = self.format_functions.get(dtype)
            if format_fn is None:
                continue
            
            formatted = format_fn(raw_sample, self.tokens)
            if formatted is None:
                continue
            
            # Process media (on-demand)
            processed = self._process_sample(formatted, dtype)
            if processed is None:
                continue
            
            # Tokenize
            tokenized = self._tokenize(processed)
            
            # Update counts
            dataset_counts[name] += 1
            modality_counts[modality] += 1
            unique_samples += 1
            
            # Yield with repeat
            for _ in range(self.sample_repeat):
                total_yields += 1
                yield tokenized
            
            if unique_samples >= self.max_per_epoch:
                break
```

### Media Processing

```python
def _process_sample(self, sample, dtype):
    """Process media in sample based on dtype."""
    
    # Image processing
    if 'image' in sample and sample['image'] is not None:
        image = sample['image']
        if isinstance(image, str):
            # URL or path
            image = self._load_image(image)
        
        # Resize and normalize
        image = self.image_processor(
            image,
            return_tensors='pt',
            size={'height': self.image_size, 'width': self.image_size},
        ).pixel_values.squeeze(0)
        sample['image'] = image
    
    # Video processing
    if 'video' in sample and sample['video'] is not None:
        video = sample['video']
        if isinstance(video, str):
            video = self._load_video(video)
        
        # Sample frames
        video = self._sample_frames(video, self.max_video_frames)
        
        # Resize each frame
        video = torch.stack([
            self.image_processor(
                frame,
                return_tensors='pt',
                size={'height': self.video_size, 'width': self.video_size},
            ).pixel_values.squeeze(0)
            for frame in video
        ])
        sample['video'] = video
    
    # Audio processing
    if 'audio' in sample and sample['audio'] is not None:
        audio = sample['audio']
        if isinstance(audio, dict):
            # HuggingFace audio format
            waveform = torch.tensor(audio['array'])
            sample_rate = audio['sampling_rate']
        elif isinstance(audio, str):
            # Path
            waveform, sample_rate = self._load_audio(audio)
        else:
            waveform = audio
            sample_rate = self.audio_sample_rate
        
        # Resample if needed
        if sample_rate != self.audio_sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, sample_rate, self.audio_sample_rate
            )
        
        if self.use_raw_waveform:
            # Raw waveform mode
            sample['audio'] = waveform
        else:
            # Mel spectrogram mode
            mel = self._compute_mel(waveform)
            sample['audio'] = mel
    
    return sample
```

### Tokenization

```python
def _tokenize(self, sample):
    """Tokenize text and create labels."""
    text = sample.get('text', '')
    
    # Tokenize
    encoding = self.tokenizer(
        text,
        max_length=self.max_length,
        truncation=True,
        padding='max_length',
        return_tensors='pt',
    )
    
    result = {
        'input_ids': encoding.input_ids.squeeze(0),
        'attention_mask': encoding.attention_mask.squeeze(0),
    }
    
    # Create labels (shift by 1 for causal LM)
    if sample.get('is_training', True):
        labels = encoding.input_ids.squeeze(0).clone()
        # Mask padding
        labels[labels == self.tokenizer.pad_token_id] = -100
        result['labels'] = labels
    
    # Add media
    if 'image' in sample:
        result['images'] = sample['image']
    if 'video' in sample:
        result['videos'] = sample['video']
    if 'audio' in sample:
        result['audio'] = sample['audio']
    
    # Add metadata
    result['dtype'] = sample.get('dtype', 'text')
    result['has_image_gen'] = sample.get('has_image_gen', False)
    result['has_video_gen'] = sample.get('has_video_gen', False)
    result['has_asr'] = sample.get('has_asr', False)
    result['has_tts'] = sample.get('has_tts', False)
    
    return result
```

### Resume Support

```python
def _save_streaming_state(self, path):
    """Save streaming state for resume."""
    state = {
        'epoch': self._streaming_state['epoch'],
        'unique_samples': self._streaming_state['unique_samples'],
        'total_yields': self._streaming_state['total_yields'],
        'dataset_positions': self._streaming_state['dataset_positions'],
        'modality_counts': self._streaming_state['modality_counts'],
    }
    with open(path, 'w') as f:
        json.dump(state, f)

def _load_streaming_state(self, path):
    """Load streaming state for resume."""
    with open(path, 'r') as f:
        state = json.load(f)
    self._streaming_state.update(state)
    
    # Skip already-seen samples
    for src in self._dataset_sources:
        name = src['name']
        skip_count = self._streaming_state['dataset_positions'].get(name, 0)
        if skip_count > 0:
            iterator = self._get_iterator(src)
            for _ in range(skip_count):
                try:
                    next(iterator)
                except StopIteration:
                    break
```

---

## 📝 Format Functions

### Overview

Format functions transform raw dataset samples into a standardized format.

### Code Format

```python
def format_code(sample, tokens):
    """Format code dataset sample."""
    # Extract fields
    instruction = sample.get('instruction', sample.get('prompt', ''))
    code = sample.get('output', sample.get('code', sample.get('response', '')))
    language = sample.get('language', 'python')
    
    # Build text
    text = (
        f"{tokens['system_start']}"
        f"You are an expert programmer. Write clean, efficient code."
        f"{tokens['system_end']}"
        f"{tokens['user_start']}"
        f"{instruction}"
        f"{tokens['user_end']}"
        f"{tokens['assistant_start']}"
        f"```{language}\n{code}\n```"
        f"{tokens['assistant_end']}"
    )
    
    return {'text': text, 'dtype': 'code'}
```

### Conversation Format

```python
def format_conversation(sample, tokens):
    """Format conversation dataset sample."""
    messages = sample.get('messages', sample.get('conversations', []))
    
    text_parts = []
    for msg in messages:
        role = msg.get('role', msg.get('from', ''))
        content = msg.get('content', msg.get('value', ''))
        
        if role in ['system', 'instruction']:
            text_parts.append(f"{tokens['system_start']}{content}{tokens['system_end']}")
        elif role in ['user', 'human']:
            text_parts.append(f"{tokens['user_start']}{content}{tokens['user_end']}")
        elif role in ['assistant', 'gpt', 'bot']:
            text_parts.append(f"{tokens['assistant_start']}{content}{tokens['assistant_end']}")
    
    return {'text': ''.join(text_parts), 'dtype': 'conversation'}
```

### Tool Use Format

```python
def format_tool_use(sample, tokens):
    """Format tool/function calling sample."""
    query = sample.get('query', sample.get('instruction', ''))
    tools = sample.get('tools', sample.get('functions', []))
    response = sample.get('response', sample.get('output', ''))
    
    # Format tools
    tools_str = json.dumps(tools, indent=2) if tools else ''
    
    text = (
        f"{tokens['system_start']}"
        f"You have access to the following tools:\n{tools_str}"
        f"{tokens['system_end']}"
        f"{tokens['user_start']}"
        f"{query}"
        f"{tokens['user_end']}"
        f"{tokens['assistant_start']}"
        f"{tokens['tool_call']}{response}{tokens['tool_call_end']}"
        f"{tokens['assistant_end']}"
    )
    
    return {'text': text, 'dtype': 'tool_use'}
```

### Vision Format

```python
def format_vision(sample, tokens):
    """Format image understanding sample."""
    image = sample.get('image')
    question = sample.get('question', sample.get('prompt', ''))
    answer = sample.get('answer', sample.get('response', ''))
    
    text = (
        f"{tokens['user_start']}"
        f"{tokens['image_start']}{tokens['image_end']}"
        f"{question}"
        f"{tokens['user_end']}"
        f"{tokens['assistant_start']}"
        f"{answer}"
        f"{tokens['assistant_end']}"
    )
    
    return {
        'text': text,
        'image': image,
        'dtype': 'vision',
    }
```

### Video Format

```python
def format_video(sample, tokens):
    """Format video understanding sample."""
    video = sample.get('video', sample.get('video_path'))
    question = sample.get('question', sample.get('prompt', ''))
    answer = sample.get('answer', sample.get('response', ''))
    
    text = (
        f"{tokens['user_start']}"
        f"{tokens['video_start']}{tokens['video_end']}"
        f"{question}"
        f"{tokens['user_end']}"
        f"{tokens['assistant_start']}"
        f"{answer}"
        f"{tokens['assistant_end']}"
    )
    
    return {
        'text': text,
        'video': video,
        'dtype': 'video',
    }
```

### ASR Format

```python
def format_voice_asr(sample, tokens):
    """Format speech-to-text sample."""
    audio = sample.get('audio')
    transcript = sample.get('text', sample.get('transcript', ''))
    
    text = (
        f"{tokens['user_start']}"
        f"{tokens['audio_start']}{tokens['audio_end']}"
        f"Transcribe this audio."
        f"{tokens['user_end']}"
        f"{tokens['assistant_start']}"
        f"{transcript}"
        f"{tokens['assistant_end']}"
    )
    
    return {
        'text': text,
        'audio': audio,
        'dtype': 'voice_asr',
        'has_asr': True,
    }
```

### TTS Format

```python
def format_voice_tts(sample, tokens):
    """Format text-to-speech sample."""
    text_input = sample.get('text', sample.get('transcript', ''))
    audio = sample.get('audio')
    speaker_ref = sample.get('speaker_ref', sample.get('reference_audio'))
    
    text = (
        f"{tokens['user_start']}"
        f"Read this text aloud: {text_input}"
        f"{tokens['user_end']}"
        f"{tokens['assistant_start']}"
        f"{tokens['gen_audio']}{tokens['gen_audio_end']}"
        f"{tokens['assistant_end']}"
    )
    
    return {
        'text': text,
        'target_audio': audio,
        'speaker_ref': speaker_ref,
        'dtype': 'voice_tts',
        'has_tts': True,
    }
```

### Generation Format

```python
def format_generation(sample, tokens):
    """Format image/video generation sample."""
    prompt = sample.get('prompt', sample.get('text', ''))
    image = sample.get('image')
    video = sample.get('video')
    
    if video is not None:
        text = (
            f"{tokens['user_start']}"
            f"Generate a video: {prompt}"
            f"{tokens['user_end']}"
            f"{tokens['assistant_start']}"
            f"{tokens['gen_video']}{tokens['gen_video_end']}"
            f"{tokens['assistant_end']}"
        )
        return {
            'text': text,
            'gen_videos': video,
            'gen_prompts': prompt,
            'dtype': 'generation',
            'has_video_gen': True,
        }
    else:
        text = (
            f"{tokens['user_start']}"
            f"Generate an image: {prompt}"
            f"{tokens['user_end']}"
            f"{tokens['assistant_start']}"
            f"{tokens['gen_image']}{tokens['gen_image_end']}"
            f"{tokens['assistant_end']}"
        )
        return {
            'text': text,
            'gen_images': image,
            'gen_prompts': prompt,
            'dtype': 'generation',
            'has_image_gen': True,
        }
```

---

## 🔄 Collate Function

```python
def collate_fn(batch):
    """Collate batch of samples."""
    result = {
        'input_ids': torch.stack([s['input_ids'] for s in batch]),
        'attention_mask': torch.stack([s['attention_mask'] for s in batch]),
    }
    
    # Labels
    if 'labels' in batch[0]:
        result['labels'] = torch.stack([s['labels'] for s in batch])
    
    # Images
    images = [s.get('images') for s in batch if s.get('images') is not None]
    if images:
        result['images'] = torch.stack(images)
    
    # Videos
    videos = [s.get('videos') for s in batch if s.get('videos') is not None]
    if videos:
        result['videos'] = torch.stack(videos)
    
    # Audio
    audios = [s.get('audio') for s in batch if s.get('audio') is not None]
    if audios:
        # Pad to same length
        max_len = max(a.shape[-1] for a in audios)
        padded = [F.pad(a, (0, max_len - a.shape[-1])) for a in audios]
        result['audio'] = torch.stack(padded)
    
    # Flags
    result['has_image_gen'] = any(s.get('has_image_gen', False) for s in batch)
    result['has_video_gen'] = any(s.get('has_video_gen', False) for s in batch)
    result['has_asr'] = any(s.get('has_asr', False) for s in batch)
    result['has_tts'] = any(s.get('has_tts', False) for s in batch)
    
    return result
```

---

## 📊 Dataset Statistics

### Supported Datasets

| Category | Count | Examples |
|----------|-------|----------|
| Code | 10+ | Code-Feedback, HumanEval |
| Conversation | 4+ | Dolly, OpenAssistant |
| Tool Use | 4+ | Function-Calling-ChatML |
| Agentic | 5+ | AgentInstruct, WildChat |
| Vision | 5+ | ScienceQA, WebSight |
| Video | 10+ | Video-MME, VideoInstruct |
| Generation | 15+ | Stable-Diffusion-Prompts |
| Audio | 4+ | LibriSpeech, LibriTTS |

### Synthetic Datasets

| Category | Datasets |
|----------|----------|
| Chain-of-Thought | Synth-CoT |
| Anti-Hallucination | Synth-IDK, Synth-FactCheck, Synth-Uncertainty |
| Git Operations | Synth-Commits, Synth-Diffs, Synth-Issues |
| Code Execution | Synth-Jupyter, Synth-Shell |
| Documents | Synth-Documents |

---

## 🔗 Related Documentation

- [Config Documentation](../config/README.md) - Dataset configuration
- [Training Documentation](../training/README.md) - How data is used in training
- [Synth Documentation](../synth/README.md) - Synthetic data generation
