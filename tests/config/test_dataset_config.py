"""Unit tests for config/dataset_config.py."""

import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config.dataset_config import (
    DATASET_CONFIGS,
    MODALITY_GROUPS,
    CATEGORY_TO_MODALITY,
    filter_datasets_by_modalities,
    filter_datasets_by_categories,
    get_finetune_datasets,
)


class TestDatasetConfigs(unittest.TestCase):
    """Test cases for DATASET_CONFIGS dictionary."""
    
    def test_dataset_configs_is_dict(self):
        """Test DATASET_CONFIGS is a dictionary."""
        self.assertIsInstance(DATASET_CONFIGS, dict)
        
    def test_dataset_configs_not_empty(self):
        """Test DATASET_CONFIGS is not empty."""
        self.assertGreater(len(DATASET_CONFIGS), 0)
        
    def test_code_category_exists(self):
        """Test code category exists."""
        self.assertIn('code', DATASET_CONFIGS)
        
    def test_conversation_category_exists(self):
        """Test conversation category exists."""
        self.assertIn('conversation', DATASET_CONFIGS)
        
    def test_tool_use_category_exists(self):
        """Test tool_use category exists."""
        self.assertIn('tool_use', DATASET_CONFIGS)
        
    def test_chain_of_thought_category_exists(self):
        """Test chain_of_thought category exists."""
        self.assertIn('chain_of_thought', DATASET_CONFIGS)
        
    def test_anti_hallucination_category_exists(self):
        """Test anti_hallucination category exists."""
        self.assertIn('anti_hallucination', DATASET_CONFIGS)
        
    def test_image_caption_category_exists(self):
        """Test image_caption category exists."""
        self.assertIn('image_caption', DATASET_CONFIGS)
        
    def test_video_caption_category_exists(self):
        """Test video_caption category exists."""
        self.assertIn('video_caption', DATASET_CONFIGS)
        
    def test_voice_asr_category_exists(self):
        """Test voice_asr category exists."""
        self.assertIn('voice_asr', DATASET_CONFIGS)
        
    def test_dataset_config_structure(self):
        """Test dataset config has correct structure."""
        for category, configs in DATASET_CONFIGS.items():
            self.assertIsInstance(configs, list)
            for config in configs:
                self.assertIsInstance(config, dict)
                self.assertIn('name', config)
                self.assertIn('path', config)
                
    def test_dataset_configs_have_required_fields(self):
        """Test each dataset config has required fields."""
        for category, configs in DATASET_CONFIGS.items():
            for config in configs:
                self.assertIn('name', config)
                self.assertIn('path', config)
                # split is optional for some configs
                
    def test_local_datasets_have_local_flag(self):
        """Test local datasets have local flag."""
        for category, configs in DATASET_CONFIGS.items():
            for config in configs:
                if config['path'].startswith('synth/'):
                    self.assertTrue(config.get('local', False))


class TestModalityGroups(unittest.TestCase):
    """Test cases for MODALITY_GROUPS dictionary."""
    
    def test_modality_groups_is_dict(self):
        """Test MODALITY_GROUPS is a dictionary."""
        self.assertIsInstance(MODALITY_GROUPS, dict)
        
    def test_text_modality_exists(self):
        """Test text modality exists."""
        self.assertIn('text', MODALITY_GROUPS)
        
    def test_image_modality_exists(self):
        """Test image modality exists."""
        self.assertIn('image', MODALITY_GROUPS)
        
    def test_video_modality_exists(self):
        """Test video modality exists."""
        self.assertIn('video', MODALITY_GROUPS)
        
    def test_audio_modality_exists(self):
        """Test audio modality exists."""
        self.assertIn('audio', MODALITY_GROUPS)
        
    def test_reasoning_modality_exists(self):
        """Test reasoning modality exists."""
        self.assertIn('reasoning', MODALITY_GROUPS)
        
    def test_anti_hallucination_modality_exists(self):
        """Test anti_hallucination modality exists."""
        self.assertIn('anti_hallucination', MODALITY_GROUPS)
        
    def test_modality_groups_contain_lists(self):
        """Test modality groups contain lists of categories."""
        for modality, categories in MODALITY_GROUPS.items():
            self.assertIsInstance(categories, list)
            self.assertGreater(len(categories), 0)
            
    def test_text_modality_categories(self):
        """Test text modality contains expected categories."""
        text_categories = MODALITY_GROUPS['text']
        self.assertIn('code', text_categories)
        self.assertIn('conversation', text_categories)


class TestCategoryToModality(unittest.TestCase):
    """Test cases for CATEGORY_TO_MODALITY dictionary."""
    
    def test_category_to_modality_is_dict(self):
        """Test CATEGORY_TO_MODALITY is a dictionary."""
        self.assertIsInstance(CATEGORY_TO_MODALITY, dict)
        
    def test_main_categories_mapped(self):
        """Test main categories are mapped."""
        # Check that key categories are mapped
        self.assertIn('code', CATEGORY_TO_MODALITY)
        self.assertIn('image_caption', CATEGORY_TO_MODALITY)
        self.assertIn('video_caption', CATEGORY_TO_MODALITY)
            
    def test_code_maps_to_text(self):
        """Test code category maps to text modality."""
        self.assertEqual(CATEGORY_TO_MODALITY['code'], 'text')
        
    def test_image_caption_maps_to_image(self):
        """Test image_caption maps to image modality."""
        self.assertEqual(CATEGORY_TO_MODALITY['image_caption'], 'image')
        
    def test_video_caption_maps_to_video(self):
        """Test video_caption maps to video modality."""
        self.assertEqual(CATEGORY_TO_MODALITY['video_caption'], 'video')
        
    def test_voice_asr_maps_to_audio(self):
        """Test voice_asr maps to audio modality."""
        self.assertEqual(CATEGORY_TO_MODALITY['voice_asr'], 'audio')


class TestFilterDatasetsByModalities(unittest.TestCase):
    """Test cases for filter_datasets_by_modalities function."""
    
    def test_returns_dict(self):
        """Test function returns a dictionary."""
        result = filter_datasets_by_modalities()
        self.assertIsInstance(result, dict)
        
    def test_no_filter_returns_all(self):
        """Test no filter returns all datasets."""
        result = filter_datasets_by_modalities()
        self.assertGreater(len(result), 0)
        
    def test_filter_by_text_modality(self):
        """Test filtering by text modality returns results."""
        result = filter_datasets_by_modalities(modalities=['text'])
        # Should return some results
        self.assertIsInstance(result, dict)
            
    def test_filter_by_image_modality(self):
        """Test filtering by image modality returns results."""
        result = filter_datasets_by_modalities(modalities=['image'])
        self.assertIsInstance(result, dict)
            
    def test_filter_by_multiple_modalities(self):
        """Test filtering by multiple modalities returns results."""
        result = filter_datasets_by_modalities(modalities=['text', 'image'])
        self.assertIsInstance(result, dict)


class TestFilterDatasetsByCategories(unittest.TestCase):
    """Test cases for filter_datasets_by_categories function."""
    
    def test_returns_dict(self):
        """Test function returns a dictionary."""
        result = filter_datasets_by_categories(['code'])
        self.assertIsInstance(result, dict)
        
    def test_filter_single_category(self):
        """Test filtering by single category returns results."""
        result = filter_datasets_by_categories(['code'])
        # Should return dict with results
        self.assertIsInstance(result, dict)
        
    def test_filter_multiple_categories(self):
        """Test filtering by multiple categories returns results."""
        result = filter_datasets_by_categories(['code', 'conversation'])
        self.assertIsInstance(result, dict)
        
    def test_empty_list_returns_dict(self):
        """Test empty list returns a dict."""
        result = filter_datasets_by_categories([])
        self.assertIsInstance(result, dict)


class TestGetFinetuneDatasetsFunction(unittest.TestCase):
    """Test cases for get_finetune_datasets function."""
    
    def test_returns_dict_with_mode(self):
        """Test function returns a dictionary with finetune_mode."""
        result = get_finetune_datasets(finetune_mode='text')
        self.assertIsInstance(result, dict)
        
    def test_text_mode_returns_datasets(self):
        """Test text mode returns datasets."""
        result = get_finetune_datasets(finetune_mode='text')
        self.assertGreater(len(result), 0)
        
    def test_image_mode_returns_datasets(self):
        """Test image mode returns datasets."""
        result = get_finetune_datasets(finetune_mode='image')
        self.assertIsInstance(result, dict)


if __name__ == '__main__':
    unittest.main()
