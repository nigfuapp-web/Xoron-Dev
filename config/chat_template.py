"""
Custom chat template for Xoron-Dev multimodal model.

This template uses our special tokens instead of the pretrained tokenizer's template.
Supports all SOTA features:
- Multimodal inputs (images, videos, audio, documents)
- Chain-of-thought reasoning (think, plan, critique, analysis, observation, reflection)
- Tool/function calling with structured arguments
- Code execution (Jupyter, exec)
- Memory and context management
- Uncertainty/confidence markers
- Generation outputs (gen_image, gen_video)
- Voice/TTS with emotion and prosody markers
- FIM (Fill-in-the-Middle) for code completion
"""

from config.special_tokens import SPECIAL_TOKENS

# Full Jinja2 chat template using Xoron special tokens
# Includes support for all reasoning blocks, multimodal content, and generation
XORON_CHAT_TEMPLATE = """{%- set bos = '<|bos|>' -%}
{%- set eos = '<|eos|>' -%}
{%- set system_start = '<|system|>' -%}
{%- set system_end = '<|/system|>' -%}
{%- set user_start = '<|user|>' -%}
{%- set user_end = '<|/user|>' -%}
{%- set assistant_start = '<|assistant|>' -%}
{%- set assistant_end = '<|/assistant|>' -%}
{%- set image_start = '<|image|>' -%}
{%- set image_end = '<|/image|>' -%}
{%- set video_start = '<|video|>' -%}
{%- set video_end = '<|/video|>' -%}
{%- set audio_start = '<|audio|>' -%}
{%- set audio_end = '<|/audio|>' -%}
{%- set doc_start = '<|doc|>' -%}
{%- set doc_end = '<|/doc|>' -%}
{%- set tool_call_start = '<|tool_call|>' -%}
{%- set tool_call_end = '<|/tool_call|>' -%}
{%- set tool_result_start = '<|tool_result|>' -%}
{%- set tool_result_end = '<|/tool_result|>' -%}
{%- set tools_start = '<|tools|>' -%}
{%- set tools_end = '<|/tools|>' -%}
{%- set available_tools_start = '<|available_tools|>' -%}
{%- set available_tools_end = '<|/available_tools|>' -%}
{%- set function_name_start = '<|function_name|>' -%}
{%- set function_name_end = '<|/function_name|>' -%}
{%- set function_args_start = '<|function_args|>' -%}
{%- set function_args_end = '<|/function_args|>' -%}
{%- set think_start = '<|think|>' -%}
{%- set think_end = '<|/think|>' -%}
{%- set plan_start = '<|plan|>' -%}
{%- set plan_end = '<|/plan|>' -%}
{%- set critique_start = '<|critique|>' -%}
{%- set critique_end = '<|/critique|>' -%}
{%- set analysis_start = '<|analysis|>' -%}
{%- set analysis_end = '<|/analysis|>' -%}
{%- set observation_start = '<|observation|>' -%}
{%- set observation_end = '<|/observation|>' -%}
{%- set reflection_start = '<|reflection|>' -%}
{%- set reflection_end = '<|/reflection|>' -%}
{%- set conclusion_start = '<|conclusion|>' -%}
{%- set conclusion_end = '<|/conclusion|>' -%}
{%- set code_start = '<|code|>' -%}
{%- set code_end = '<|/code|>' -%}
{%- set exec_start = '<|exec|>' -%}
{%- set exec_end = '<|/exec|>' -%}
{%- set exec_result = '<|exec_result|>' -%}
{%- set exec_result_end = '<|/exec_result|>' -%}
{%- set jupyter_code = '<|jupyter_code|>' -%}
{%- set jupyter_code_end = '<|/jupyter_code|>' -%}
{%- set jupyter_output = '<|jupyter_output|>' -%}
{%- set jupyter_output_end = '<|/jupyter_output|>' -%}
{%- set gen_image_start = '<|gen_image|>' -%}
{%- set gen_image_end = '<|/gen_image|>' -%}
{%- set gen_video_start = '<|gen_video|>' -%}
{%- set gen_video_end = '<|/gen_video|>' -%}
{%- set speak_start = '<|speak|>' -%}
{%- set speak_end = '<|/speak|>' -%}
{%- set listen_start = '<|listen|>' -%}
{%- set listen_end = '<|/listen|>' -%}
{%- set memory_start = '<|memory|>' -%}
{%- set memory_end = '<|/memory|>' -%}
{%- set context_start = '<|context|>' -%}
{%- set context_end = '<|/context|>' -%}
{%- set uncertain_start = '<|uncertain|>' -%}
{%- set uncertain_end = '<|/uncertain|>' -%}
{%- set cite_start = '<|cite|>' -%}
{%- set cite_end = '<|/cite|>' -%}
{%- set eod = '<|eod|>' -%}

{{- bos -}}
{%- if messages[0]['role'] == 'system' -%}
    {{- system_start + messages[0]['content'] + system_end -}}
    {%- set messages = messages[1:] -%}
{%- endif -%}
{%- if available_tools is defined and available_tools -%}
    {{- available_tools_start + available_tools + available_tools_end -}}
{%- elif tools is defined and tools -%}
    {{- tools_start + tools + tools_end -}}
{%- endif -%}
{%- if memory is defined and memory -%}
    {{- memory_start + memory + memory_end -}}
{%- endif -%}
{%- if context is defined and context -%}
    {{- context_start + context + context_end -}}
{%- endif -%}
{%- for message in messages -%}
    {%- if message['role'] == 'system' -%}
        {{- system_start + message['content'] + system_end -}}
    {%- elif message['role'] == 'user' -%}
        {{- user_start -}}
        {%- if message.get('images') -%}
            {%- for img in message['images'] -%}
                {{- image_start + img + image_end -}}
            {%- endfor -%}
        {%- endif -%}
        {%- if message.get('videos') -%}
            {%- for vid in message['videos'] -%}
                {{- video_start + vid + video_end -}}
            {%- endfor -%}
        {%- endif -%}
        {%- if message.get('audio') -%}
            {%- for aud in message['audio'] -%}
                {{- audio_start + aud + audio_end -}}
            {%- endfor -%}
        {%- endif -%}
        {%- if message.get('documents') -%}
            {%- for doc in message['documents'] -%}
                {{- doc_start + doc + doc_end -}}
            {%- endfor -%}
        {%- endif -%}
        {{- message['content'] + user_end -}}
    {%- elif message['role'] == 'assistant' -%}
        {{- assistant_start -}}
        {%- if message.get('thinking') -%}
            {{- think_start + message['thinking'] + think_end -}}
        {%- endif -%}
        {%- if message.get('planning') -%}
            {{- plan_start + message['planning'] + plan_end -}}
        {%- endif -%}
        {%- if message.get('analysis') -%}
            {{- analysis_start + message['analysis'] + analysis_end -}}
        {%- endif -%}
        {%- if message.get('observation') -%}
            {{- observation_start + message['observation'] + observation_end -}}
        {%- endif -%}
        {%- if message.get('reflection') -%}
            {{- reflection_start + message['reflection'] + reflection_end -}}
        {%- endif -%}
        {%- if message.get('critique') -%}
            {{- critique_start + message['critique'] + critique_end -}}
        {%- endif -%}
        {%- if message.get('conclusion') -%}
            {{- conclusion_start + message['conclusion'] + conclusion_end -}}
        {%- endif -%}
        {%- if message.get('tool_calls') -%}
            {%- for tool in message['tool_calls'] -%}
                {{- tool_call_start -}}
                {%- if tool is mapping -%}
                    {{- function_name_start + tool.get('name', '') + function_name_end -}}
                    {{- function_args_start + (tool.get('arguments', '') | tojson if tool.get('arguments') is mapping else tool.get('arguments', '')) + function_args_end -}}
                {%- else -%}
                    {{- tool -}}
                {%- endif -%}
                {{- tool_call_end -}}
            {%- endfor -%}
        {%- endif -%}
        {%- if message.get('code') -%}
            {{- code_start + message['code'] + code_end -}}
        {%- endif -%}
        {%- if message.get('exec') -%}
            {{- exec_start + message['exec'] + exec_end -}}
        {%- endif -%}
        {%- if message.get('gen_image') -%}
            {{- gen_image_start + message['gen_image'] + gen_image_end -}}
        {%- endif -%}
        {%- if message.get('gen_video') -%}
            {{- gen_video_start + message['gen_video'] + gen_video_end -}}
        {%- endif -%}
        {%- if message.get('speak') -%}
            {{- speak_start + message['speak'] + speak_end -}}
        {%- endif -%}
        {%- if message.get('uncertain') -%}
            {{- uncertain_start + message['uncertain'] + uncertain_end -}}
        {%- endif -%}
        {%- if message.get('citation') -%}
            {{- cite_start + message['citation'] + cite_end -}}
        {%- endif -%}
        {{- message['content'] -}}
        {%- if not loop.last or add_generation_prompt is not defined or not add_generation_prompt -%}
            {{- assistant_end -}}
        {%- endif -%}
    {%- elif message['role'] == 'tool' -%}
        {{- tool_result_start + message['content'] + tool_result_end -}}
    {%- elif message['role'] == 'exec_result' -%}
        {{- exec_result + message['content'] + exec_result_end -}}
    {%- elif message['role'] == 'jupyter' -%}
        {{- jupyter_output + message['content'] + jupyter_output_end -}}
    {%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt is defined and add_generation_prompt -%}
    {{- assistant_start -}}
    {%- if enable_thinking is defined and enable_thinking -%}
        {{- think_start -}}
    {%- endif -%}
{%- endif -%}
"""

# Simpler version for basic chat (text-only, with optional reasoning)
XORON_CHAT_TEMPLATE_SIMPLE = """{%- set bos = '<|bos|>' -%}
{%- set eos = '<|eos|>' -%}
{%- set system_start = '<|system|>' -%}
{%- set system_end = '<|/system|>' -%}
{%- set user_start = '<|user|>' -%}
{%- set user_end = '<|/user|>' -%}
{%- set assistant_start = '<|assistant|>' -%}
{%- set assistant_end = '<|/assistant|>' -%}
{%- set think_start = '<|think|>' -%}
{%- set think_end = '<|/think|>' -%}
{%- set tool_call_start = '<|tool_call|>' -%}
{%- set tool_call_end = '<|/tool_call|>' -%}
{%- set tool_result_start = '<|tool_result|>' -%}
{%- set tool_result_end = '<|/tool_result|>' -%}
{%- set code_start = '<|code|>' -%}
{%- set code_end = '<|/code|>' -%}

{{- bos -}}
{%- for message in messages -%}
    {%- if message['role'] == 'system' -%}
        {{- system_start + message['content'] + system_end -}}
    {%- elif message['role'] == 'user' -%}
        {{- user_start + message['content'] + user_end -}}
    {%- elif message['role'] == 'assistant' -%}
        {{- assistant_start -}}
        {%- if message.get('thinking') -%}
            {{- think_start + message['thinking'] + think_end -}}
        {%- endif -%}
        {%- if message.get('tool_calls') -%}
            {%- for tool in message['tool_calls'] -%}
                {{- tool_call_start + tool + tool_call_end -}}
            {%- endfor -%}
        {%- endif -%}
        {%- if message.get('code') -%}
            {{- code_start + message['code'] + code_end -}}
        {%- endif -%}
        {{- message['content'] -}}
        {%- if not loop.last or add_generation_prompt is not defined or not add_generation_prompt -%}
            {{- assistant_end -}}
        {%- endif -%}
    {%- elif message['role'] == 'tool' -%}
        {{- tool_result_start + message['content'] + tool_result_end -}}
    {%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt is defined and add_generation_prompt -%}
    {{- assistant_start -}}
    {%- if enable_thinking is defined and enable_thinking -%}
        {{- think_start -}}
    {%- endif -%}
{%- endif -%}
"""


def get_chat_template(multimodal: bool = True) -> str:
    """
    Get the appropriate chat template.
    
    Args:
        multimodal: If True, returns template with full multimodal support.
                   If False, returns simpler text-only template with basic reasoning.
    
    Returns:
        Jinja2 chat template string
    """
    return XORON_CHAT_TEMPLATE if multimodal else XORON_CHAT_TEMPLATE_SIMPLE


def apply_chat_template_to_tokenizer(tokenizer, multimodal: bool = True):
    """
    Apply the Xoron chat template to a tokenizer.
    
    This replaces any existing chat_template from the pretrained tokenizer
    with our custom template that uses Xoron special tokens.
    
    Args:
        tokenizer: HuggingFace tokenizer instance
        multimodal: Whether to use multimodal template
    
    Returns:
        Modified tokenizer with custom chat_template
    """
    tokenizer.chat_template = get_chat_template(multimodal)
    
    # Also set the special tokens properly
    tokenizer.bos_token = SPECIAL_TOKENS['bos']
    tokenizer.eos_token = SPECIAL_TOKENS['eos']
    tokenizer.pad_token = SPECIAL_TOKENS['pad']
    
    return tokenizer


def format_chat_example(messages: list, tokenizer=None, **kwargs) -> str:
    """
    Format a list of messages using the chat template.
    
    Args:
        messages: List of message dicts with 'role' and 'content' keys
        tokenizer: Optional tokenizer (if None, uses template directly)
        **kwargs: Additional template variables (tools, memory, context, etc.)
    
    Returns:
        Formatted string
    
    Example:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!", "images": ["<image_placeholder>"]},
            {"role": "assistant", "content": "Hi there!", "thinking": "User greeted me."}
        ]
        
        # With tools
        tools = '[{"name": "search", "description": "Search the web"}]'
        format_chat_example(messages, tools=tools)
    """
    if tokenizer is not None and hasattr(tokenizer, 'apply_chat_template'):
        return tokenizer.apply_chat_template(messages, tokenize=False, **kwargs)
    
    # Manual formatting without tokenizer
    from jinja2 import Template
    template = Template(XORON_CHAT_TEMPLATE)
    return template.render(messages=messages, add_generation_prompt=False, **kwargs)


def format_multimodal_message(
    content: str,
    images: list = None,
    videos: list = None,
    audio: list = None,
    documents: list = None,
    role: str = "user"
) -> dict:
    """
    Helper to create a properly formatted multimodal message.
    
    Args:
        content: Text content of the message
        images: List of image placeholders or paths
        videos: List of video placeholders or paths
        audio: List of audio placeholders or paths
        documents: List of document contents
        role: Message role (user, assistant, system)
    
    Returns:
        Formatted message dict
    """
    message = {"role": role, "content": content}
    if images:
        message["images"] = images
    if videos:
        message["videos"] = videos
    if audio:
        message["audio"] = audio
    if documents:
        message["documents"] = documents
    return message


def format_assistant_response(
    content: str,
    thinking: str = None,
    planning: str = None,
    analysis: str = None,
    observation: str = None,
    reflection: str = None,
    critique: str = None,
    conclusion: str = None,
    tool_calls: list = None,
    code: str = None,
    gen_image: str = None,
    gen_video: str = None,
    speak: str = None,
    uncertain: str = None,
    citation: str = None,
) -> dict:
    """
    Helper to create a properly formatted assistant response with reasoning.
    
    Args:
        content: Main response content
        thinking: Chain-of-thought reasoning
        planning: Planning steps
        analysis: Analysis content
        observation: Observations
        reflection: Reflections
        critique: Self-critique
        conclusion: Conclusions
        tool_calls: List of tool calls (dicts with 'name' and 'arguments')
        code: Code to execute
        gen_image: Image generation prompt
        gen_video: Video generation prompt
        speak: TTS content
        uncertain: Uncertain content
        citation: Citation/source
    
    Returns:
        Formatted assistant message dict
    """
    message = {"role": "assistant", "content": content}
    if thinking:
        message["thinking"] = thinking
    if planning:
        message["planning"] = planning
    if analysis:
        message["analysis"] = analysis
    if observation:
        message["observation"] = observation
    if reflection:
        message["reflection"] = reflection
    if critique:
        message["critique"] = critique
    if conclusion:
        message["conclusion"] = conclusion
    if tool_calls:
        message["tool_calls"] = tool_calls
    if code:
        message["code"] = code
    if gen_image:
        message["gen_image"] = gen_image
    if gen_video:
        message["gen_video"] = gen_video
    if speak:
        message["speak"] = speak
    if uncertain:
        message["uncertain"] = uncertain
    if citation:
        message["citation"] = citation
    return message
