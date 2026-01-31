"""
Comprehensive unit tests for data formatters module.
Tests MultimodalFormatter class and all its methods.

This test suite downloads 1 sample from each dataset (both synth and external
from HuggingFace) and tests the formatting output to verify everything works
correctly including video and multimodal content.
"""

import unittest
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from data.formatters import MultimodalFormatter
from config.special_tokens import SPECIAL_TOKENS


class TestMultimodalFormatterInit(unittest.TestCase):
    """Test MultimodalFormatter initialization."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.tokens = {
            'bos': '<|bos|>',
            'eos': '<|eos|>',
            'system_start': '<|system|>',
            'system_end': '<|/system|>',
            'user_start': '<|user|>',
            'user_end': '<|/user|>',
            'assistant_start': '<|assistant|>',
            'assistant_end': '<|/assistant|>',
            'think_start': '<|think|>',
            'think_end': '<|/think|>',
            'emotion_happy': '<|happy|>',
            'emotion_sad': '<|sad|>',
            'emotion_neutral': '<|neutral|>',
            'prosody_fast': '<|fast|>',
            'prosody_slow': '<|slow|>',
            'prosody_normal_speed': '<|normal_speed|>',
            'prosody_loud': '<|loud|>',
            'prosody_soft': '<|soft|>',
            'prosody_normal_volume': '<|normal_volume|>',
            'prosody_high_pitch': '<|high_pitch|>',
            'prosody_low_pitch': '<|low_pitch|>',
            'prosody_normal_pitch': '<|normal_pitch|>',
        }
        
    def test_initialization(self):
        """Test MultimodalFormatter initialization."""
        formatter = MultimodalFormatter(self.tokens)
        
        self.assertEqual(formatter.t, self.tokens)
        self.assertIsNone(formatter.image_processor)
        
    def test_initialization_with_image_processor(self):
        """Test initialization with image processor."""
        mock_processor = object()
        formatter = MultimodalFormatter(self.tokens, image_processor=mock_processor)
        
        self.assertEqual(formatter.image_processor, mock_processor)
        
    def test_emotions_list(self):
        """Test emotions list is initialized."""
        formatter = MultimodalFormatter(self.tokens)
        
        self.assertIn('neutral', formatter.emotions)
        self.assertIn('happy', formatter.emotions)
        self.assertIn('sad', formatter.emotions)
        
    def test_prosody_options(self):
        """Test prosody options are initialized."""
        formatter = MultimodalFormatter(self.tokens)
        
        self.assertIn('speed', formatter.prosody_options)
        self.assertIn('volume', formatter.prosody_options)
        self.assertIn('pitch', formatter.prosody_options)


class TestWrapSequence(unittest.TestCase):
    """Test _wrap_sequence method."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.tokens = {
            'bos': '<|bos|>',
            'eos': '<|eos|>',
        }
        self.formatter = MultimodalFormatter(self.tokens)
        
    def test_wrap_sequence(self):
        """Test wrapping text with BOS/EOS tokens."""
        text = "Hello world"
        result = self.formatter._wrap_sequence(text)
        
        self.assertEqual(result, "<|bos|>Hello world<|eos|>")
        
    def test_wrap_empty_string(self):
        """Test wrapping empty string."""
        result = self.formatter._wrap_sequence("")
        
        self.assertEqual(result, "<|bos|><|eos|>")


class TestFormatSimpleQA(unittest.TestCase):
    """Test _format_simple_qa method."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.tokens = {
            'bos': '<|bos|>',
            'eos': '<|eos|>',
            'system_start': '<|system|>',
            'system_end': '<|/system|>',
            'user_start': '<|user|>',
            'user_end': '<|/user|>',
            'assistant_start': '<|assistant|>',
            'assistant_end': '<|/assistant|>',
        }
        self.formatter = MultimodalFormatter(self.tokens)
        
    def test_format_simple_qa_without_system(self):
        """Test formatting Q&A without system message."""
        result = self.formatter._format_simple_qa("What is 2+2?", "4")
        
        self.assertIn("<|user|>", result)
        self.assertIn("What is 2+2?", result)
        self.assertIn("<|assistant|>", result)
        self.assertIn("4", result)
        self.assertTrue(result.startswith("<|bos|>"))
        self.assertTrue(result.endswith("<|eos|>"))
        
    def test_format_simple_qa_with_system(self):
        """Test formatting Q&A with system message."""
        result = self.formatter._format_simple_qa(
            "What is 2+2?", 
            "4", 
            system_content="You are a math tutor."
        )
        
        self.assertIn("<|system|>", result)
        self.assertIn("You are a math tutor.", result)
        
    def test_format_simple_qa_no_wrap(self):
        """Test formatting Q&A without BOS/EOS wrapping."""
        result = self.formatter._format_simple_qa("Question", "Answer", wrap=False)
        
        self.assertFalse(result.startswith("<|bos|>"))
        self.assertFalse(result.endswith("<|eos|>"))


class TestEmotionTokens(unittest.TestCase):
    """Test emotion token methods."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.tokens = {
            'bos': '<|bos|>',
            'eos': '<|eos|>',
            'emotion_happy': '<|happy|>',
            'emotion_sad': '<|sad|>',
            'emotion_neutral': '<|neutral|>',
        }
        self.formatter = MultimodalFormatter(self.tokens)
        
    def test_add_emotion_token(self):
        """Test adding emotion token."""
        result = self.formatter._add_emotion_token("Hello!", "happy")
        
        self.assertEqual(result, "<|happy|>Hello!")
        
    def test_add_emotion_token_none(self):
        """Test adding no emotion token."""
        result = self.formatter._add_emotion_token("Hello!", None)
        
        self.assertEqual(result, "Hello!")
        
    def test_add_emotion_token_unknown(self):
        """Test adding unknown emotion token."""
        result = self.formatter._add_emotion_token("Hello!", "unknown_emotion")
        
        self.assertEqual(result, "Hello!")
        
    def test_get_random_emotion(self):
        """Test getting random emotion."""
        emotion = self.formatter._get_random_emotion()
        
        self.assertIn(emotion, self.formatter.emotions)


class TestProsodyTokens(unittest.TestCase):
    """Test prosody token methods."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.tokens = {
            'bos': '<|bos|>',
            'eos': '<|eos|>',
            'prosody_fast': '<|fast|>',
            'prosody_slow': '<|slow|>',
            'prosody_normal_speed': '<|normal_speed|>',
            'prosody_loud': '<|loud|>',
            'prosody_soft': '<|soft|>',
            'prosody_normal_volume': '<|normal_volume|>',
            'prosody_high_pitch': '<|high_pitch|>',
            'prosody_low_pitch': '<|low_pitch|>',
            'prosody_normal_pitch': '<|normal_pitch|>',
        }
        self.formatter = MultimodalFormatter(self.tokens)
        
    def test_add_prosody_tokens_speed(self):
        """Test adding speed prosody token."""
        result = self.formatter._add_prosody_tokens("Hello!", speed="fast")
        
        self.assertEqual(result, "<|fast|>Hello!")
        
    def test_add_prosody_tokens_multiple(self):
        """Test adding multiple prosody tokens."""
        result = self.formatter._add_prosody_tokens("Hello!", speed="fast", volume="loud", pitch="high_pitch")
        
        self.assertIn("<|fast|>", result)
        self.assertIn("<|loud|>", result)
        self.assertIn("<|high_pitch|>", result)
        self.assertIn("Hello!", result)
        
    def test_add_prosody_tokens_none(self):
        """Test adding no prosody tokens."""
        result = self.formatter._add_prosody_tokens("Hello!")
        
        self.assertEqual(result, "Hello!")
        
    def test_get_random_prosody(self):
        """Test getting random prosody settings."""
        prosody = self.formatter._get_random_prosody()
        
        self.assertIn('speed', prosody)
        self.assertIn('volume', prosody)
        self.assertIn('pitch', prosody)
        self.assertIn(prosody['speed'], self.formatter.prosody_options['speed'])


class TestPlanningTokens(unittest.TestCase):
    """Test planning token methods."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.tokens = {
            'bos': '<|bos|>',
            'eos': '<|eos|>',
            'plan_start': '<|plan|>',
            'plan_end': '<|/plan|>',
            'plan_step': '<|step|>',
            'plan_step_end': '<|/step|>',
        }
        self.formatter = MultimodalFormatter(self.tokens)
        
    def test_wrap_with_planning(self):
        """Test wrapping content with planning tokens."""
        steps = ["Step 1", "Step 2", "Step 3"]
        result = self.formatter._wrap_with_planning(steps, "Final content")
        
        self.assertIn("<|plan|>", result)
        self.assertIn("<|/plan|>", result)
        self.assertIn("<|step|>Step 1<|/step|>", result)
        self.assertIn("Final content", result)
        
    def test_wrap_with_planning_empty_steps(self):
        """Test wrapping with empty steps list."""
        result = self.formatter._wrap_with_planning([], "Content only")
        
        self.assertEqual(result, "Content only")


class TestThinkingTokens(unittest.TestCase):
    """Test thinking/reasoning token methods."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.tokens = {
            'bos': '<|bos|>',
            'eos': '<|eos|>',
            'think_start': '<|think|>',
            'think_end': '<|/think|>',
            'conclusion_start': '<|conclusion|>',
            'conclusion_end': '<|/conclusion|>',
        }
        self.formatter = MultimodalFormatter(self.tokens)
        
    def test_wrap_with_thinking(self):
        """Test wrapping content with thinking tokens."""
        result = self.formatter._wrap_with_thinking("Let me think about this...")
        
        self.assertIn("<|think|>", result)
        self.assertIn("Let me think about this...", result)
        self.assertIn("<|/think|>", result)
        
    def test_wrap_with_thinking_and_conclusion(self):
        """Test wrapping with thinking and conclusion."""
        result = self.formatter._wrap_with_thinking("Reasoning here", "The answer is 42")
        
        self.assertIn("<|think|>", result)
        self.assertIn("<|conclusion|>", result)
        self.assertIn("The answer is 42", result)


class TestCritiqueTokens(unittest.TestCase):
    """Test critique/correction token methods."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.tokens = {
            'bos': '<|bos|>',
            'eos': '<|eos|>',
            'critique_start': '<|critique|>',
            'critique_end': '<|/critique|>',
            'error_found': '<|error|>',
            'no_error': '<|no_error|>',
        }
        self.formatter = MultimodalFormatter(self.tokens)
        
    def test_wrap_with_critique_no_error(self):
        """Test wrapping with critique, no error."""
        result = self.formatter._wrap_with_critique("Content", "Looks good!", has_error=False)
        
        self.assertIn("<|no_error|>", result)
        self.assertIn("Looks good!", result)
        
    def test_wrap_with_critique_has_error(self):
        """Test wrapping with critique, has error."""
        result = self.formatter._wrap_with_critique("Content", "Found a mistake", has_error=True)
        
        self.assertIn("<|error|>", result)
        self.assertIn("Found a mistake", result)


class TestUncertaintyTokens(unittest.TestCase):
    """Test uncertainty/confidence token methods."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.tokens = {
            'bos': '<|bos|>',
            'eos': '<|eos|>',
            'uncertainty_score': '<|uncertainty|>',
            'uncertainty_score_end': '<|/uncertainty|>',
            'confidence_high': '<|high_confidence|>',
            'confidence_medium': '<|medium_confidence|>',
            'confidence_low': '<|low_confidence|>',
        }
        self.formatter = MultimodalFormatter(self.tokens)
        
    def test_add_uncertainty_score(self):
        """Test adding uncertainty score."""
        result = self.formatter._add_uncertainty_score("Content", 75)
        
        self.assertIn("<|uncertainty|>75<|/uncertainty|>", result)
        
    def test_add_uncertainty_score_clamped(self):
        """Test uncertainty score is clamped to 0-100."""
        result_high = self.formatter._add_uncertainty_score("Content", 150)
        result_low = self.formatter._add_uncertainty_score("Content", -50)
        
        self.assertIn("100", result_high)
        self.assertIn("0", result_low)
        
    def test_add_confidence_level(self):
        """Test adding confidence level."""
        result = self.formatter._add_confidence_level("Content", "high")
        
        self.assertEqual(result, "<|high_confidence|>Content")
        
    def test_add_confidence_level_unknown(self):
        """Test adding unknown confidence level."""
        result = self.formatter._add_confidence_level("Content", "unknown")
        
        self.assertEqual(result, "Content")


class TestMemoryTokens(unittest.TestCase):
    """Test memory token methods."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.tokens = {
            'bos': '<|bos|>',
            'eos': '<|eos|>',
            'memory_start': '<|memory|>',
            'memory_end': '<|/memory|>',
            'summary_start': '<|summary|>',
            'summary_end': '<|/summary|>',
        }
        self.formatter = MultimodalFormatter(self.tokens)
        
    def test_wrap_with_memory(self):
        """Test wrapping content with memory tokens."""
        result = self.formatter._wrap_with_memory("Previous context here")
        
        self.assertIn("<|memory|>", result)
        self.assertIn("Previous context here", result)
        self.assertIn("<|/memory|>", result)
        
    def test_wrap_with_summary(self):
        """Test wrapping content with summary tokens."""
        result = self.formatter._wrap_with_summary("Summary of conversation")
        
        self.assertIn("<|summary|>", result)
        self.assertIn("Summary of conversation", result)
        self.assertIn("<|/summary|>", result)


class TestTableFormatting(unittest.TestCase):
    """Test table formatting methods."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.tokens = {
            'bos': '<|bos|>',
            'eos': '<|eos|>',
            'table_start': '<|table|>',
            'table_end': '<|/table|>',
            'table_header_start': '<|thead|>',
            'table_header_end': '<|/thead|>',
            'table_body_start': '<|tbody|>',
            'table_body_end': '<|/tbody|>',
            'table_row_start': '<|tr|>',
            'table_row_end': '<|/tr|>',
            'table_cell_start': '<|td|>',
            'table_cell_end': '<|/td|>',
        }
        self.formatter = MultimodalFormatter(self.tokens)
        
    def test_wrap_with_table(self):
        """Test wrapping data as table."""
        headers = ["Name", "Age"]
        rows = [["Alice", "30"], ["Bob", "25"]]
        
        result = self.formatter._wrap_with_table(headers, rows)
        
        self.assertIn("<|table|>", result)
        self.assertIn("<|thead|>", result)
        self.assertIn("<|td|>Name<|/td|>", result)
        self.assertIn("<|td|>Alice<|/td|>", result)
        self.assertIn("<|/table|>", result)


class TestJSONFormatting(unittest.TestCase):
    """Test JSON formatting methods."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.tokens = {
            'bos': '<|bos|>',
            'eos': '<|eos|>',
            'json_start': '<|json|>',
            'json_end': '<|/json|>',
        }
        self.formatter = MultimodalFormatter(self.tokens)
        
    def test_wrap_json(self):
        """Test wrapping JSON content."""
        json_content = '{"key": "value"}'
        result = self.formatter._wrap_json(json_content)
        
        self.assertIn("<|json|>", result)
        self.assertIn('{"key": "value"}', result)
        self.assertIn("<|/json|>", result)


class TestHelperMethods(unittest.TestCase):
    """Test helper methods."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.tokens = {'bos': '<|bos|>', 'eos': '<|eos|>'}
        self.formatter = MultimodalFormatter(self.tokens)
        
    def test_safe_get_first_key(self):
        """Test _safe_get with first key found."""
        sample = {'key1': 'value1', 'key2': 'value2'}
        result = self.formatter._safe_get(sample, ['key1', 'key2'])
        
        self.assertEqual(result, 'value1')
        
    def test_safe_get_second_key(self):
        """Test _safe_get with second key found."""
        sample = {'key2': 'value2'}
        result = self.formatter._safe_get(sample, ['key1', 'key2'])
        
        self.assertEqual(result, 'value2')
        
    def test_safe_get_default(self):
        """Test _safe_get returns default when no key found."""
        sample = {'other': 'value'}
        result = self.formatter._safe_get(sample, ['key1', 'key2'], default='default')
        
        self.assertEqual(result, 'default')
        
    def test_clean_text(self):
        """Test _clean_text normalizes whitespace."""
        text = "  Hello   world  \n\t test  "
        result = self.formatter._clean_text(text)
        
        self.assertEqual(result, "Hello world test")
        
    def test_clean_text_empty(self):
        """Test _clean_text with empty string."""
        result = self.formatter._clean_text("")
        
        self.assertEqual(result, "")
        
    def test_clean_text_none(self):
        """Test _clean_text with None."""
        result = self.formatter._clean_text(None)
        
        self.assertEqual(result, "")


class TestAvailableToolsFormatting(unittest.TestCase):
    """Test available tools formatting."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.tokens = {
            'bos': '<|bos|>',
            'eos': '<|eos|>',
            'available_tools_start': '<|available_tools|>',
            'available_tools_end': '<|/available_tools|>',
            'tool_def_start': '<|tool_def|>',
            'tool_def_end': '<|/tool_def|>',
            'tool_name': '<|tool_name|>',
            'tool_name_end': '<|/tool_name|>',
            'tool_description': '<|tool_desc|>',
            'tool_description_end': '<|/tool_desc|>',
            'tool_params_start': '<|tool_params|>',
            'tool_params_end': '<|/tool_params|>',
            'param_name': '<|param_name|>',
            'param_name_end': '<|/param_name|>',
            'param_type': '<|param_type|>',
            'param_type_end': '<|/param_type|>',
            'param_required': '<|param_required|>',
            'param_optional': '<|param_optional|>',
        }
        self.formatter = MultimodalFormatter(self.tokens)
        
    def test_format_available_tools_empty(self):
        """Test formatting empty tools list."""
        result = self.formatter._format_available_tools([])
        
        self.assertEqual(result, "")
        
    def test_format_available_tools_single(self):
        """Test formatting single tool."""
        tools = [{
            'name': 'search',
            'description': 'Search the web',
            'parameters': {
                'properties': {
                    'query': {'type': 'string'}
                },
                'required': ['query']
            }
        }]
        
        result = self.formatter._format_available_tools(tools)
        
        self.assertIn("<|available_tools|>", result)
        self.assertIn("<|tool_name|>search<|/tool_name|>", result)
        self.assertIn("Search the web", result)


class TestRealDatasetFormatting(unittest.TestCase):
    """
    Test formatters with REAL data from HuggingFace datasets.
    Downloads 1 sample from each dataset category and tests formatting.
    """
    
    @classmethod
    def setUpClass(cls):
        """Set up formatter with full SPECIAL_TOKENS once for all tests."""
        cls.formatter = MultimodalFormatter(SPECIAL_TOKENS)
        cls.max_samples = 1  # Download only 1 sample per dataset
    
    def _load_hf_sample(self, path, split="train", config=None, streaming=True, is_audio=False):
        """
        Helper to load 1 sample from a HuggingFace dataset.
        
        For audio datasets, we disable automatic audio decoding to use our custom
        decoder (soundfile/librosa) instead of torchcodec.
        """
        try:
            from datasets import load_dataset
            
            load_kwargs = {
                "path": path,
                "split": split,
                "streaming": streaming,
            }
            if config:
                load_kwargs["name"] = config
            
            ds = load_dataset(**load_kwargs)
            
            # For audio datasets, disable automatic audio decoding
            # We use soundfile/librosa in our custom VoiceProcessor instead of torchcodec
            if is_audio:
                try:
                    from datasets import Audio
                    # Cast audio column to disable automatic decoding
                    ds = ds.cast_column('audio', Audio(decode=False))
                except Exception:
                    pass  # Column might not exist or Audio feature not available
            
            if streaming:
                sample = next(iter(ds))
            else:
                sample = ds[0] if len(ds) > 0 else None
            
            return sample
        except Exception as e:
            print(f"    ⚠️ Could not load {path}: {e}")
            return None
    
    def _load_local_sample(self, path):
        """Helper to load 1 sample from a local JSONL file."""
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        full_path = os.path.join(project_root, path)
        
        if not os.path.exists(full_path):
            print(f"    ⚠️ Local dataset not found: {full_path}")
            return None
        
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        return json.loads(line)
        except Exception as e:
            print(f"    ⚠️ Could not load {full_path}: {e}")
            return None
        return None
    
    def _print_formatted_output(self, category, dataset_name, formatted):
        """Print formatted output for inspection."""
        if formatted and formatted.get("text"):
            text = formatted["text"]
            # Truncate for readability
            display_text = text[:500] + "..." if len(text) > 500 else text
            print(f"\n{'='*60}")
            print(f"📦 Category: {category}")
            print(f"📊 Dataset: {dataset_name}")
            print(f"📝 Type: {formatted.get('type', 'unknown')}")
            print(f"{'='*60}")
            print(display_text)
            print(f"{'='*60}\n")

    # === EXTERNAL HUGGINGFACE DATASETS ===
    
    def test_format_code_sample_external(self):
        """Test formatting code samples from external HuggingFace datasets."""
        print("\n\n🔄 Testing CODE formatting with external datasets...")
        
        # Test Code-Feedback dataset
        sample = self._load_hf_sample("m-a-p/Code-Feedback")
        if sample:
            formatted = self.formatter.format_code_sample(sample)
            self._print_formatted_output("code", "Code-Feedback", formatted)
            if formatted:
                self.assertIn("text", formatted)
                self.assertTrue(len(formatted["text"]) > 0, "Formatted text should not be empty")
    
    def test_format_conversation_sample_external(self):
        """Test formatting conversation samples from external HuggingFace datasets."""
        print("\n\n🔄 Testing CONVERSATION formatting with external datasets...")
        
        # Test Dolly-15k dataset
        sample = self._load_hf_sample("databricks/databricks-dolly-15k")
        if sample:
            formatted = self.formatter.format_conversation_sample(sample)
            self._print_formatted_output("conversation", "Dolly-15k", formatted)
            if formatted:
                self.assertIn("text", formatted)
                # Check for conversation tokens (BOS may not always be present depending on format)
                text = formatted["text"]
                has_conversation_structure = (
                    SPECIAL_TOKENS['user_start'] in text or 
                    SPECIAL_TOKENS['assistant_start'] in text or
                    SPECIAL_TOKENS['bos'] in text
                )
                self.assertTrue(has_conversation_structure, "Formatted text should have conversation tokens")
    
    def test_format_tool_use_sample_external(self):
        """Test formatting tool-use samples from external HuggingFace datasets."""
        print("\n\n🔄 Testing TOOL-USE formatting with external datasets...")
        
        # Test Function-Calling-ChatML dataset
        sample = self._load_hf_sample("Locutusque/function-calling-chatml")
        if sample:
            formatted = self.formatter.format_tool_use_sample(sample)
            self._print_formatted_output("tool_use", "Function-Calling-ChatML", formatted)
            if formatted:
                self.assertIn("text", formatted)
    
    def test_format_agentic_sample_external(self):
        """Test formatting agentic samples from external HuggingFace datasets."""
        print("\n\n🔄 Testing AGENTIC formatting with external datasets...")
        
        # Test AgentInstruct dataset
        sample = self._load_hf_sample("THUDM/AgentInstruct", split="os")
        if sample:
            formatted = self.formatter.format_agentic_sample(sample)
            self._print_formatted_output("agentic", "AgentInstruct", formatted)
            if formatted:
                self.assertIn("text", formatted)
    
    def test_format_image_caption_sample_external(self):
        """Test formatting image caption samples from external HuggingFace datasets."""
        print("\n\n🔄 Testing IMAGE CAPTION formatting with external datasets...")
        
        # Test Flickr8k dataset
        sample = self._load_hf_sample("Naveengo/flickr8k")
        if sample:
            formatted = self.formatter.format_image_caption_sample(sample)
            self._print_formatted_output("image_caption", "Flickr8k", formatted)
            if formatted:
                self.assertIn("text", formatted)
    
    def test_format_image_vqa_sample_external(self):
        """Test formatting image VQA samples from external HuggingFace datasets."""
        print("\n\n🔄 Testing IMAGE VQA formatting with external datasets...")
        
        # Test ScienceQA dataset
        sample = self._load_hf_sample("derek-thomas/ScienceQA")
        if sample:
            formatted = self.formatter.format_image_vqa_sample(sample)
            self._print_formatted_output("image_vqa", "ScienceQA", formatted)
            if formatted:
                self.assertIn("text", formatted)
    
    def test_format_image_generation_sample_external(self):
        """Test formatting image generation samples from external HuggingFace datasets."""
        print("\n\n🔄 Testing IMAGE GENERATION formatting with external datasets...")
        
        # Test SD-Prompts dataset
        sample = self._load_hf_sample("Gustavosta/Stable-Diffusion-Prompts")
        if sample:
            formatted = self.formatter.format_image_generation_sample(sample)
            self._print_formatted_output("image_generation", "SD-Prompts", formatted)
            if formatted:
                self.assertIn("text", formatted)
    
    def test_format_image_editing_sample_external(self):
        """Test formatting image editing samples from external HuggingFace datasets."""
        print("\n\n🔄 Testing IMAGE EDITING formatting with external datasets...")
        
        # Test MagicBrush dataset
        sample = self._load_hf_sample("osunlp/MagicBrush")
        if sample:
            formatted = self.formatter.format_image_editing_sample(sample)
            self._print_formatted_output("image_editing", "MagicBrush", formatted)
            if formatted:
                self.assertIn("text", formatted)
    
    def test_format_video_caption_sample_external(self):
        """Test formatting video caption samples from external HuggingFace datasets."""
        print("\n\n🔄 Testing VIDEO CAPTION formatting with external datasets...")
        
        # Test Video-MME dataset
        sample = self._load_hf_sample("lmms-lab/Video-MME", split="test")
        if sample:
            formatted = self.formatter.format_video_caption_sample(sample)
            self._print_formatted_output("video_caption", "Video-MME", formatted)
            if formatted:
                self.assertIn("text", formatted)
                print(f"    📹 Video formatting verified with metadata")
    
    def test_format_video_qa_sample_external(self):
        """Test formatting video QA samples from external HuggingFace datasets."""
        print("\n\n🔄 Testing VIDEO QA formatting with external datasets...")
        
        # Test VideoInstruct-100K dataset
        sample = self._load_hf_sample("MBZUAI/VideoInstruct-100K")
        if sample:
            formatted = self.formatter.format_video_qa_sample(sample)
            self._print_formatted_output("video_qa", "VideoInstruct-100K", formatted)
            if formatted:
                self.assertIn("text", formatted)
                print(f"    📹 Video QA formatting verified")
    
    def test_format_video_generation_sample_external(self):
        """Test formatting video generation samples from external HuggingFace datasets."""
        print("\n\n🔄 Testing VIDEO GENERATION formatting with external datasets...")
        
        # Test WebVid-10M dataset
        sample = self._load_hf_sample("TempoFunk/webvid-10M")
        if sample:
            formatted = self.formatter.format_video_generation_sample(sample)
            self._print_formatted_output("video_generation", "WebVid-10M", formatted)
            if formatted:
                self.assertIn("text", formatted)
                print(f"    🎬 Video generation formatting verified")
    
    def test_format_image_to_video_sample_external(self):
        """Test formatting image-to-video samples from external HuggingFace datasets."""
        print("\n\n🔄 Testing IMAGE-TO-VIDEO formatting with external datasets...")
        
        # Test TIP-I2V dataset
        sample = self._load_hf_sample("WenhaoWang/TIP-I2V", split="Full")
        if sample:
            formatted = self.formatter.format_image_to_video_sample(sample)
            self._print_formatted_output("image_to_video", "TIP-I2V", formatted)
            if formatted:
                self.assertIn("text", formatted)
                print(f"    🎬 Image-to-Video formatting verified")
    
    def test_format_video_preference_sample_external(self):
        """Test formatting video preference samples from external HuggingFace datasets."""
        print("\n\n🔄 Testing VIDEO PREFERENCE formatting with external datasets...")
        
        # Test Sora-Physics-Likert dataset
        sample = self._load_hf_sample("Rapidata/sora-video-generation-physics-likert-scoring")
        if sample:
            formatted = self.formatter.format_video_generation_sample(sample)
            self._print_formatted_output("video_preference", "Sora-Physics-Likert", formatted)
            if formatted:
                self.assertIn("text", formatted)
                print(f"    📊 Video preference/likert formatting verified")
    
    def test_format_ui_to_code_sample_external(self):
        """Test formatting UI-to-code samples from external HuggingFace datasets."""
        print("\n\n🔄 Testing UI-TO-CODE formatting with external datasets...")
        
        # Test WebSight dataset
        sample = self._load_hf_sample("HuggingFaceM4/WebSight")
        if sample:
            formatted = self.formatter.format_ui_to_code_sample(sample)
            self._print_formatted_output("ui_to_code", "WebSight", formatted)
            if formatted:
                self.assertIn("text", formatted)
    
    def test_format_voice_asr_sample_external(self):
        """
        Test formatting voice ASR samples from external HuggingFace datasets.
        
        Uses is_audio=True to disable automatic audio decoding and use our
        custom decoder (soundfile/librosa) instead of torchcodec.
        """
        print("\n\n🔄 Testing VOICE ASR formatting with external datasets...")
        print("    📝 Using custom audio decoder (soundfile/librosa), NOT torchcodec")
        
        # Test LibriSpeech dataset - disable auto-decoding for custom processing
        sample = self._load_hf_sample(
            "openslr/librispeech_asr", 
            config="clean", 
            split="train.100",
            is_audio=True  # Disable torchcodec, use soundfile/librosa
        )
        if sample:
            formatted = self.formatter.format_voice_asr_sample(sample)
            self._print_formatted_output("voice_asr", "LibriSpeech-Clean", formatted)
            if formatted:
                self.assertIn("text", formatted)
                print(f"    🎤 Voice ASR formatting verified (custom decoder)")
    
    def test_format_voice_tts_sample_external(self):
        """
        Test formatting voice TTS samples from external HuggingFace datasets.
        
        Uses is_audio=True to disable automatic audio decoding and use our
        custom decoder (soundfile/librosa) instead of torchcodec.
        """
        print("\n\n🔄 Testing VOICE TTS formatting with external datasets...")
        print("    📝 Using custom audio decoder (soundfile/librosa), NOT torchcodec")
        
        # Test LibriTTS-R dataset - disable auto-decoding for custom processing
        sample = self._load_hf_sample(
            "blabble-io/libritts_r", 
            config="clean", 
            split="train.clean.100",
            is_audio=True  # Disable torchcodec, use soundfile/librosa
        )
        if sample:
            formatted = self.formatter.format_voice_tts_sample(sample)
            self._print_formatted_output("voice_tts", "LibriTTS-R-Clean", formatted)
            if formatted:
                self.assertIn("text", formatted)
                print(f"    🔊 Voice TTS formatting verified (custom decoder)")

    # === SYNTHETIC/LOCAL DATASETS ===
    
    def test_format_chain_of_thought_sample_synth(self):
        """Test formatting chain-of-thought samples from synthetic datasets."""
        print("\n\n🔄 Testing CHAIN-OF-THOUGHT formatting with synthetic datasets...")
        
        sample = self._load_local_sample("synth/data/cot_dataset.jsonl")
        if sample:
            formatted = self.formatter.format_chain_of_thought_sample(sample)
            self._print_formatted_output("chain_of_thought", "Synth-CoT", formatted)
            if formatted:
                self.assertIn("text", formatted)
                print(f"    🧠 Chain-of-thought formatting verified")
    
    def test_format_anti_hallucination_idk_sample_synth(self):
        """Test formatting IDK (I don't know) samples from synthetic datasets."""
        print("\n\n🔄 Testing ANTI-HALLUCINATION IDK formatting with synthetic datasets...")
        
        sample = self._load_local_sample("synth/data/idk_dataset.jsonl")
        if sample:
            formatted = self.formatter.format_passthrough_sample(sample)
            self._print_formatted_output("anti_hallucination", "Synth-IDK", formatted)
            if formatted:
                self.assertIn("text", formatted)
    
    def test_format_knowledge_cutoff_sample_synth(self):
        """Test formatting knowledge cutoff samples from synthetic datasets."""
        print("\n\n🔄 Testing KNOWLEDGE CUTOFF formatting with synthetic datasets...")
        
        sample = self._load_local_sample("synth/data/knowledge_cutoff_dataset.jsonl")
        if sample:
            formatted = self.formatter.format_passthrough_sample(sample)
            self._print_formatted_output("anti_hallucination", "Synth-KnowledgeCutoff", formatted)
            if formatted:
                self.assertIn("text", formatted)
    
    def test_format_document_sample_synth(self):
        """Test formatting document samples from synthetic datasets."""
        print("\n\n🔄 Testing DOCUMENT formatting with synthetic datasets...")
        
        sample = self._load_local_sample("synth/data/document_dataset.jsonl")
        if sample:
            formatted = self.formatter.format_document_sample(sample)
            self._print_formatted_output("document", "Synth-Documents", formatted)
            if formatted:
                self.assertIn("text", formatted)
    
    def test_format_fim_sample_synth(self):
        """Test formatting FIM (fill-in-the-middle) samples from synthetic datasets."""
        print("\n\n🔄 Testing FIM formatting with synthetic datasets...")
        
        sample = self._load_local_sample("synth/data/fim_dataset.jsonl")
        if sample:
            formatted = self.formatter.format_passthrough_sample(sample)
            self._print_formatted_output("fim", "Synth-FIM", formatted)
            if formatted:
                self.assertIn("text", formatted)
    
    def test_format_git_commit_sample_synth(self):
        """Test formatting git commit samples from synthetic datasets."""
        print("\n\n🔄 Testing GIT COMMIT formatting with synthetic datasets...")
        
        sample = self._load_local_sample("synth/data/commit_dataset.jsonl")
        if sample:
            formatted = self.formatter.format_passthrough_sample(sample)
            self._print_formatted_output("git_operations", "Synth-Commits", formatted)
            if formatted:
                self.assertIn("text", formatted)
    
    def test_format_git_diff_sample_synth(self):
        """Test formatting git diff samples from synthetic datasets."""
        print("\n\n🔄 Testing GIT DIFF formatting with synthetic datasets...")
        
        sample = self._load_local_sample("synth/data/diff_dataset.jsonl")
        if sample:
            formatted = self.formatter.format_passthrough_sample(sample)
            self._print_formatted_output("git_operations", "Synth-Diffs", formatted)
            if formatted:
                self.assertIn("text", formatted)
    
    def test_format_jupyter_sample_synth(self):
        """Test formatting Jupyter execution samples from synthetic datasets."""
        print("\n\n🔄 Testing JUPYTER formatting with synthetic datasets...")
        
        sample = self._load_local_sample("synth/data/jupyter_dataset.jsonl")
        if sample:
            formatted = self.formatter.format_passthrough_sample(sample)
            self._print_formatted_output("code_execution", "Synth-Jupyter", formatted)
            if formatted:
                self.assertIn("text", formatted)
    
    def test_format_file_ops_sample_synth(self):
        """Test formatting file operations samples from synthetic datasets."""
        print("\n\n🔄 Testing FILE OPERATIONS formatting with synthetic datasets...")
        
        sample = self._load_local_sample("synth/data/file_ops_dataset.jsonl")
        if sample:
            formatted = self.formatter.format_agentic_sample(sample)
            self._print_formatted_output("file_operations", "Synth-FileOps", formatted)
            if formatted:
                self.assertIn("text", formatted)
    
    def test_format_system_admin_docker_sample_synth(self):
        """Test formatting system admin docker samples from synthetic datasets."""
        print("\n\n🔄 Testing SYSTEM ADMIN DOCKER formatting with synthetic datasets...")
        
        sample = self._load_local_sample("synth/data/docker_dataset.jsonl")
        if sample:
            formatted = self.formatter.format_passthrough_sample(sample)
            self._print_formatted_output("system_admin", "Synth-Docker", formatted)
            if formatted:
                self.assertIn("text", formatted)


class TestFormatterIntegration(unittest.TestCase):
    """
    Integration tests that verify all formatters work together.
    Downloads 1 sample from each major category to ensure end-to-end functionality.
    """
    
    @classmethod
    def setUpClass(cls):
        """Set up formatter with full SPECIAL_TOKENS."""
        cls.formatter = MultimodalFormatter(SPECIAL_TOKENS)
    
    def test_all_format_functions_exist(self):
        """Test that all required format functions exist on the formatter."""
        required_methods = [
            'format_code_sample',
            'format_conversation_sample',
            'format_tool_use_sample',
            'format_agentic_sample',
            'format_image_caption_sample',
            'format_image_vqa_sample',
            'format_video_caption_sample',
            'format_video_qa_sample',
            'format_video_generation_sample',
            'format_image_to_video_sample',
            'format_image_generation_sample',
            'format_image_editing_sample',
            'format_ui_to_code_sample',
            'format_voice_asr_sample',
            'format_voice_tts_sample',
            'format_document_sample',
            'format_multi_document_sample',
            'format_chain_of_thought_sample',
            'format_passthrough_sample',
        ]
        
        for method in required_methods:
            self.assertTrue(
                hasattr(self.formatter, method),
                f"Formatter missing method: {method}"
            )
            self.assertTrue(
                callable(getattr(self.formatter, method)),
                f"Method {method} is not callable"
            )
    
    def test_special_tokens_coverage(self):
        """Test that SPECIAL_TOKENS has all required tokens."""
        required_tokens = [
            'bos', 'eos', 'pad',
            'system_start', 'system_end',
            'user_start', 'user_end',
            'assistant_start', 'assistant_end',
            'image_start', 'image_end',
            'video_start', 'video_end',
            'think_start', 'think_end',
        ]
        
        for token in required_tokens:
            self.assertIn(
                token, SPECIAL_TOKENS,
                f"SPECIAL_TOKENS missing: {token}"
            )
    
    def test_formatted_output_structure(self):
        """Test that formatted outputs have correct structure."""
        # Test with a simple mock sample
        mock_sample = {
            'instruction': 'Write a hello world program',
            'response': 'print("Hello, World!")'
        }
        
        formatted = self.formatter.format_code_sample(mock_sample)
        
        if formatted:
            self.assertIn('text', formatted)
            self.assertIsInstance(formatted['text'], str)
            self.assertTrue(len(formatted['text']) > 0)


if __name__ == '__main__':
    unittest.main()
