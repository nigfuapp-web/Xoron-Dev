"""Unit tests for config/special_tokens.py."""

import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config.special_tokens import (
    SPECIAL_TOKENS,
    REASONING_TOKENS,
    MEMORY_TOKENS,
    TEMPORAL_TOKENS,
    STRUCTURED_DATA_TOKENS,
    UNCERTAINTY_TOKENS,
    HIDDEN_TOKENS,
    get_special_tokens_list,
    get_reasoning_tokens,
    get_memory_tokens,
    get_temporal_tokens,
    get_structured_data_tokens,
    get_uncertainty_tokens,
    get_hidden_tokens,
    strip_hidden_tokens,
)


class TestSpecialTokens(unittest.TestCase):
    """Test cases for special tokens dictionary."""
    
    def test_special_tokens_is_dict(self):
        """Test SPECIAL_TOKENS is a dictionary."""
        self.assertIsInstance(SPECIAL_TOKENS, dict)
        
    def test_basic_tokens_exist(self):
        """Test basic sequence control tokens exist."""
        self.assertIn('bos', SPECIAL_TOKENS)
        self.assertIn('eos', SPECIAL_TOKENS)
        self.assertIn('pad', SPECIAL_TOKENS)
        
    def test_conversation_tokens_exist(self):
        """Test conversation tokens exist."""
        self.assertIn('system_start', SPECIAL_TOKENS)
        self.assertIn('system_end', SPECIAL_TOKENS)
        self.assertIn('user_start', SPECIAL_TOKENS)
        self.assertIn('user_end', SPECIAL_TOKENS)
        self.assertIn('assistant_start', SPECIAL_TOKENS)
        self.assertIn('assistant_end', SPECIAL_TOKENS)
        
    def test_multimodal_tokens_exist(self):
        """Test multimodal input tokens exist."""
        self.assertIn('image_start', SPECIAL_TOKENS)
        self.assertIn('image_end', SPECIAL_TOKENS)
        self.assertIn('video_start', SPECIAL_TOKENS)
        self.assertIn('video_end', SPECIAL_TOKENS)
        self.assertIn('audio_start', SPECIAL_TOKENS)
        self.assertIn('audio_end', SPECIAL_TOKENS)
        
    def test_thinking_tokens_exist(self):
        """Test thinking/reasoning tokens exist."""
        self.assertIn('think_start', SPECIAL_TOKENS)
        self.assertIn('think_end', SPECIAL_TOKENS)
        
    def test_tool_calling_tokens_exist(self):
        """Test tool calling tokens exist."""
        self.assertIn('tool_call_start', SPECIAL_TOKENS)
        self.assertIn('tool_call_end', SPECIAL_TOKENS)
        self.assertIn('tool_result_start', SPECIAL_TOKENS)
        self.assertIn('tool_result_end', SPECIAL_TOKENS)
        
    def test_code_tokens_exist(self):
        """Test code-related tokens exist."""
        self.assertIn('code_start', SPECIAL_TOKENS)
        self.assertIn('code_end', SPECIAL_TOKENS)
        self.assertIn('lang_python', SPECIAL_TOKENS)
        
    def test_generation_tokens_exist(self):
        """Test generation output tokens exist."""
        self.assertIn('gen_image_start', SPECIAL_TOKENS)
        self.assertIn('gen_image_end', SPECIAL_TOKENS)
        self.assertIn('gen_video_start', SPECIAL_TOKENS)
        self.assertIn('gen_video_end', SPECIAL_TOKENS)
        
    def test_token_format(self):
        """Test tokens have correct format."""
        for key, token in SPECIAL_TOKENS.items():
            self.assertIsInstance(token, str)
            # Most tokens should have delimiters
            if 'encoder' not in key and 'decoder' not in key and 'projection' not in key and 'state' not in key and 'modal_switch' not in key:
                self.assertTrue(
                    token.startswith('<|') or token.startswith('['),
                    f"Token {key}={token} doesn't have expected format"
                )


class TestReasoningTokens(unittest.TestCase):
    """Test cases for reasoning tokens."""
    
    def test_reasoning_tokens_is_dict(self):
        """Test REASONING_TOKENS is a dictionary."""
        self.assertIsInstance(REASONING_TOKENS, dict)
        
    def test_get_reasoning_tokens(self):
        """Test get_reasoning_tokens function."""
        tokens = get_reasoning_tokens()
        # Returns dict with token names as keys
        self.assertIsInstance(tokens, dict)
        self.assertGreater(len(tokens), 0)
        
    def test_reasoning_tokens_contain_think(self):
        """Test reasoning tokens contain think tokens."""
        tokens = get_reasoning_tokens()
        # Check keys or values for 'think'
        self.assertTrue(any('think' in k.lower() for k in tokens.keys()))


class TestMemoryTokens(unittest.TestCase):
    """Test cases for memory tokens."""
    
    def test_memory_tokens_is_dict(self):
        """Test MEMORY_TOKENS is a dictionary."""
        self.assertIsInstance(MEMORY_TOKENS, dict)
        
    def test_get_memory_tokens(self):
        """Test get_memory_tokens function."""
        tokens = get_memory_tokens()
        # Returns dict with token names as keys
        self.assertIsInstance(tokens, dict)
        
    def test_memory_tokens_contain_memory(self):
        """Test memory tokens contain memory-related tokens."""
        tokens = get_memory_tokens()
        self.assertTrue(any('memory' in k.lower() for k in tokens.keys()))


class TestTemporalTokens(unittest.TestCase):
    """Test cases for temporal tokens."""
    
    def test_temporal_tokens_is_dict(self):
        """Test TEMPORAL_TOKENS is a dictionary."""
        self.assertIsInstance(TEMPORAL_TOKENS, dict)
        
    def test_get_temporal_tokens(self):
        """Test get_temporal_tokens function."""
        tokens = get_temporal_tokens()
        # Returns dict with token names as keys
        self.assertIsInstance(tokens, dict)


class TestStructuredDataTokens(unittest.TestCase):
    """Test cases for structured data tokens."""
    
    def test_structured_data_tokens_is_dict(self):
        """Test STRUCTURED_DATA_TOKENS is a dictionary."""
        self.assertIsInstance(STRUCTURED_DATA_TOKENS, dict)
        
    def test_get_structured_data_tokens(self):
        """Test get_structured_data_tokens function."""
        tokens = get_structured_data_tokens()
        # Returns dict with token names as keys
        self.assertIsInstance(tokens, dict)


class TestUncertaintyTokens(unittest.TestCase):
    """Test cases for uncertainty tokens."""
    
    def test_uncertainty_tokens_is_dict(self):
        """Test UNCERTAINTY_TOKENS is a dictionary."""
        self.assertIsInstance(UNCERTAINTY_TOKENS, dict)
        
    def test_get_uncertainty_tokens(self):
        """Test get_uncertainty_tokens function."""
        tokens = get_uncertainty_tokens()
        # Returns dict with token names as keys
        self.assertIsInstance(tokens, dict)
        
    def test_uncertainty_tokens_contain_confidence(self):
        """Test uncertainty tokens contain confidence-related tokens."""
        tokens = get_uncertainty_tokens()
        self.assertTrue(any('confidence' in k.lower() or 'uncertain' in k.lower() for k in tokens.keys()))


class TestHiddenTokens(unittest.TestCase):
    """Test cases for hidden tokens."""
    
    def test_hidden_tokens_is_list(self):
        """Test HIDDEN_TOKENS is a list."""
        self.assertIsInstance(HIDDEN_TOKENS, list)
        
    def test_get_hidden_tokens(self):
        """Test get_hidden_tokens function."""
        tokens = get_hidden_tokens()
        # Returns dict with token names as keys
        self.assertIsInstance(tokens, dict)


class TestStripHiddenTokens(unittest.TestCase):
    """Test cases for strip_hidden_tokens function."""
    
    def test_strip_hidden_tokens_basic(self):
        """Test basic hidden token stripping."""
        text = "Hello world"
        result = strip_hidden_tokens(text)
        self.assertEqual(result, "Hello world")
        
    def test_strip_hidden_tokens_preserves_normal_text(self):
        """Test that normal text is preserved."""
        text = "This is a normal response without hidden tokens."
        result = strip_hidden_tokens(text)
        self.assertEqual(result, text)
        
    def test_strip_hidden_tokens_returns_string(self):
        """Test that strip_hidden_tokens returns a string."""
        result = strip_hidden_tokens("test")
        self.assertIsInstance(result, str)


class TestGetSpecialTokensList(unittest.TestCase):
    """Test cases for get_special_tokens_list function."""
    
    def test_returns_list(self):
        """Test that function returns a list."""
        tokens = get_special_tokens_list()
        self.assertIsInstance(tokens, list)
        
    def test_list_not_empty(self):
        """Test that returned list is not empty."""
        tokens = get_special_tokens_list()
        self.assertGreater(len(tokens), 0)
        
    def test_all_items_are_strings(self):
        """Test that all items in list are strings."""
        tokens = get_special_tokens_list()
        for token in tokens:
            self.assertIsInstance(token, str)
            
    def test_no_duplicates(self):
        """Test that there are no duplicate tokens."""
        tokens = get_special_tokens_list()
        self.assertEqual(len(tokens), len(set(tokens)))


if __name__ == '__main__':
    unittest.main()
