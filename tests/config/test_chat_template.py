"""Unit tests for config/chat_template.py."""

import sys
import os
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config.chat_template import (
    XORON_CHAT_TEMPLATE,
    XORON_CHAT_TEMPLATE_SIMPLE,
    get_chat_template,
    apply_chat_template_to_tokenizer,
    format_multimodal_message,
    format_assistant_response,
)


class TestChatTemplates(unittest.TestCase):
    """Test cases for chat templates."""
    
    def test_xoron_chat_template_is_string(self):
        """Test XORON_CHAT_TEMPLATE is a string."""
        self.assertIsInstance(XORON_CHAT_TEMPLATE, str)
        
    def test_xoron_chat_template_simple_is_string(self):
        """Test XORON_CHAT_TEMPLATE_SIMPLE is a string."""
        self.assertIsInstance(XORON_CHAT_TEMPLATE_SIMPLE, str)
        
    def test_template_contains_jinja_syntax(self):
        """Test template contains Jinja2 syntax."""
        self.assertIn('{%', XORON_CHAT_TEMPLATE)
        self.assertIn('%}', XORON_CHAT_TEMPLATE)
        self.assertIn('{{', XORON_CHAT_TEMPLATE)
        self.assertIn('}}', XORON_CHAT_TEMPLATE)
        
    def test_template_contains_special_tokens(self):
        """Test template contains special token definitions."""
        self.assertIn('<|bos|>', XORON_CHAT_TEMPLATE)
        self.assertIn('<|eos|>', XORON_CHAT_TEMPLATE)
        self.assertIn('<|system|>', XORON_CHAT_TEMPLATE)
        self.assertIn('<|user|>', XORON_CHAT_TEMPLATE)
        self.assertIn('<|assistant|>', XORON_CHAT_TEMPLATE)
        
    def test_template_contains_multimodal_tokens(self):
        """Test template contains multimodal tokens."""
        self.assertIn('<|image|>', XORON_CHAT_TEMPLATE)
        self.assertIn('<|video|>', XORON_CHAT_TEMPLATE)
        self.assertIn('<|audio|>', XORON_CHAT_TEMPLATE)
        
    def test_template_contains_thinking_tokens(self):
        """Test template contains thinking tokens."""
        self.assertIn('<|think|>', XORON_CHAT_TEMPLATE)
        self.assertIn('<|/think|>', XORON_CHAT_TEMPLATE)
        
    def test_template_contains_tool_tokens(self):
        """Test template contains tool calling tokens."""
        self.assertIn('<|tool_call|>', XORON_CHAT_TEMPLATE)
        self.assertIn('<|tool_result|>', XORON_CHAT_TEMPLATE)


class TestGetChatTemplate(unittest.TestCase):
    """Test cases for get_chat_template function."""
    
    def test_multimodal_true_returns_full_template(self):
        """Test multimodal=True returns full template."""
        template = get_chat_template(multimodal=True)
        self.assertEqual(template, XORON_CHAT_TEMPLATE)
        
    def test_multimodal_false_returns_simple_template(self):
        """Test multimodal=False returns simple template."""
        template = get_chat_template(multimodal=False)
        self.assertEqual(template, XORON_CHAT_TEMPLATE_SIMPLE)
        
    def test_default_returns_full_template(self):
        """Test default returns full template."""
        template = get_chat_template()
        self.assertEqual(template, XORON_CHAT_TEMPLATE)


class TestApplyChatTemplateToTokenizer(unittest.TestCase):
    """Test cases for apply_chat_template_to_tokenizer function."""
    
    def test_applies_template_to_tokenizer(self):
        """Test that template is applied to tokenizer."""
        mock_tokenizer = MagicMock()
        
        result = apply_chat_template_to_tokenizer(mock_tokenizer)
        
        self.assertEqual(result.chat_template, XORON_CHAT_TEMPLATE)
        
    def test_sets_special_tokens(self):
        """Test that special tokens are set."""
        mock_tokenizer = MagicMock()
        
        apply_chat_template_to_tokenizer(mock_tokenizer)
        
        self.assertEqual(mock_tokenizer.bos_token, '<|bos|>')
        self.assertEqual(mock_tokenizer.eos_token, '<|eos|>')
        self.assertEqual(mock_tokenizer.pad_token, '<|pad|>')
        
    def test_returns_tokenizer(self):
        """Test that function returns the tokenizer."""
        mock_tokenizer = MagicMock()
        
        result = apply_chat_template_to_tokenizer(mock_tokenizer)
        
        self.assertEqual(result, mock_tokenizer)
        
    def test_multimodal_false_uses_simple_template(self):
        """Test multimodal=False uses simple template."""
        mock_tokenizer = MagicMock()
        
        apply_chat_template_to_tokenizer(mock_tokenizer, multimodal=False)
        
        self.assertEqual(mock_tokenizer.chat_template, XORON_CHAT_TEMPLATE_SIMPLE)


class TestFormatMultimodalMessage(unittest.TestCase):
    """Test cases for format_multimodal_message function."""
    
    def test_basic_message(self):
        """Test basic message formatting."""
        message = format_multimodal_message("Hello world")
        
        self.assertEqual(message['role'], 'user')
        self.assertEqual(message['content'], 'Hello world')
        
    def test_message_with_images(self):
        """Test message with images."""
        message = format_multimodal_message(
            "Describe this image",
            images=['image1.jpg', 'image2.jpg']
        )
        
        self.assertEqual(message['images'], ['image1.jpg', 'image2.jpg'])
        
    def test_message_with_videos(self):
        """Test message with videos."""
        message = format_multimodal_message(
            "Describe this video",
            videos=['video1.mp4']
        )
        
        self.assertEqual(message['videos'], ['video1.mp4'])
        
    def test_message_with_audio(self):
        """Test message with audio."""
        message = format_multimodal_message(
            "Transcribe this audio",
            audio=['audio1.wav']
        )
        
        self.assertEqual(message['audio'], ['audio1.wav'])
        
    def test_message_with_documents(self):
        """Test message with documents."""
        message = format_multimodal_message(
            "Summarize this document",
            documents=['doc content']
        )
        
        self.assertEqual(message['documents'], ['doc content'])
        
    def test_custom_role(self):
        """Test message with custom role."""
        message = format_multimodal_message(
            "System prompt",
            role='system'
        )
        
        self.assertEqual(message['role'], 'system')
        
    def test_no_optional_fields_when_none(self):
        """Test that optional fields are not included when None."""
        message = format_multimodal_message("Hello")
        
        self.assertNotIn('images', message)
        self.assertNotIn('videos', message)
        self.assertNotIn('audio', message)
        self.assertNotIn('documents', message)


class TestFormatAssistantResponse(unittest.TestCase):
    """Test cases for format_assistant_response function."""
    
    def test_basic_response(self):
        """Test basic response formatting."""
        response = format_assistant_response("Hello!")
        
        self.assertEqual(response['role'], 'assistant')
        self.assertEqual(response['content'], 'Hello!')
        
    def test_response_with_thinking(self):
        """Test response with thinking."""
        response = format_assistant_response(
            "The answer is 42.",
            thinking="Let me think about this..."
        )
        
        self.assertEqual(response['thinking'], "Let me think about this...")
        
    def test_response_with_planning(self):
        """Test response with planning."""
        response = format_assistant_response(
            "Done!",
            planning="Step 1: Do X. Step 2: Do Y."
        )
        
        self.assertEqual(response['planning'], "Step 1: Do X. Step 2: Do Y.")
        
    def test_response_with_tool_calls(self):
        """Test response with tool calls."""
        tool_calls = [{'name': 'search', 'arguments': {'query': 'test'}}]
        response = format_assistant_response(
            "Let me search for that.",
            tool_calls=tool_calls
        )
        
        self.assertEqual(response['tool_calls'], tool_calls)
        
    def test_response_with_code(self):
        """Test response with code."""
        response = format_assistant_response(
            "Here's the code:",
            code="print('hello')"
        )
        
        self.assertEqual(response['code'], "print('hello')")
        
    def test_response_with_gen_image(self):
        """Test response with image generation."""
        response = format_assistant_response(
            "Here's the image:",
            gen_image="A beautiful sunset"
        )
        
        self.assertEqual(response['gen_image'], "A beautiful sunset")
        
    def test_response_with_uncertainty(self):
        """Test response with uncertainty."""
        response = format_assistant_response(
            "I'm not sure, but...",
            uncertain="This information may be outdated"
        )
        
        self.assertEqual(response['uncertain'], "This information may be outdated")
        
    def test_no_optional_fields_when_none(self):
        """Test that optional fields are not included when None."""
        response = format_assistant_response("Hello")
        
        self.assertNotIn('thinking', response)
        self.assertNotIn('planning', response)
        self.assertNotIn('tool_calls', response)
        self.assertNotIn('code', response)


if __name__ == '__main__':
    unittest.main()
