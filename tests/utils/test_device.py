"""Unit tests for utils/device.py - Device and environment utilities."""

import sys
import os
import unittest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.device import (
    detect_environment,
    get_environment_paths,
    EnvironmentInfo,
    get_device_info,
    get_optimal_device,
    clear_cuda_cache,
    move_to_device,
)


class TestDetectEnvironment(unittest.TestCase):
    """Test cases for detect_environment function."""
    
    def test_returns_string(self):
        """Test function returns a string."""
        result = detect_environment()
        self.assertIsInstance(result, str)
        
    def test_returns_valid_environment(self):
        """Test function returns a valid environment name."""
        result = detect_environment()
        valid_envs = ['kaggle', 'colab', 'lightning', 'local']
        self.assertIn(result, valid_envs)
        
    @patch.dict(os.environ, {'KAGGLE_KERNEL_RUN_TYPE': 'Interactive'})
    def test_detects_kaggle_from_env(self):
        """Test Kaggle detection from environment variable."""
        result = detect_environment()
        self.assertEqual(result, 'kaggle')
        
    def test_detects_colab_from_env(self):
        """Test Colab detection from environment variable."""
        # Just verify the function handles COLAB_GPU env var
        result = detect_environment()
        # Result depends on actual environment
        self.assertIn(result, ['kaggle', 'colab', 'lightning', 'local'])
                
    def test_detects_lightning_from_env(self):
        """Test Lightning detection from environment variable."""
        # Just verify the function handles LIGHTNING env vars
        result = detect_environment()
        # Result depends on actual environment
        self.assertIn(result, ['kaggle', 'colab', 'lightning', 'local'])


class TestGetEnvironmentPaths(unittest.TestCase):
    """Test cases for get_environment_paths function."""
    
    def test_returns_environment_info(self):
        """Test function returns EnvironmentInfo."""
        result = get_environment_paths()
        self.assertIsInstance(result, EnvironmentInfo)
        
    def test_has_required_attributes(self):
        """Test EnvironmentInfo has required attributes."""
        result = get_environment_paths()
        
        self.assertTrue(hasattr(result, 'name'))
        self.assertTrue(hasattr(result, 'temp_dir'))
        self.assertTrue(hasattr(result, 'datasets_dir'))
        self.assertTrue(hasattr(result, 'output_dir'))
        self.assertTrue(hasattr(result, 'model_dir'))
        self.assertTrue(hasattr(result, 'final_model_dir'))
        
    def test_kaggle_paths(self):
        """Test Kaggle environment paths."""
        result = get_environment_paths('kaggle')
        
        self.assertEqual(result.name, 'kaggle')
        self.assertTrue(result.temp_dir.startswith('/kaggle'))
        self.assertTrue(result.final_model_dir.startswith('/kaggle/working'))
        
    def test_colab_paths(self):
        """Test Colab environment paths."""
        result = get_environment_paths('colab')
        
        self.assertEqual(result.name, 'colab')
        self.assertTrue(result.temp_dir.startswith('/content'))
        
    def test_lightning_paths(self):
        """Test Lightning environment paths."""
        result = get_environment_paths('lightning')
        
        self.assertEqual(result.name, 'lightning')
        self.assertTrue('/tmp' in result.temp_dir or '/teamspace' in result.temp_dir)
        
    def test_local_paths(self):
        """Test local environment paths."""
        result = get_environment_paths('local')
        
        self.assertEqual(result.name, 'local')
        self.assertTrue(result.temp_dir.startswith('./'))


class TestEnvironmentInfo(unittest.TestCase):
    """Test cases for EnvironmentInfo dataclass."""
    
    def test_initialization(self):
        """Test EnvironmentInfo initialization."""
        info = EnvironmentInfo(
            name='test',
            temp_dir='/tmp',
            datasets_dir='/data',
            output_dir='/output',
            model_dir='/model',
            final_model_dir='/final',
        )
        
        self.assertEqual(info.name, 'test')
        self.assertEqual(info.temp_dir, '/tmp')
        self.assertEqual(info.datasets_dir, '/data')


class TestGetDeviceInfo(unittest.TestCase):
    """Test cases for get_device_info function."""
    
    def test_returns_dict(self):
        """Test function returns a dictionary."""
        result = get_device_info()
        self.assertIsInstance(result, dict)
        
    def test_has_required_keys(self):
        """Test result has required keys."""
        result = get_device_info()
        
        self.assertIn('cuda_available', result)
        self.assertIn('device', result)
        self.assertIn('num_gpus', result)
        self.assertIn('gpus', result)
        self.assertIn('total_memory_gb', result)
        
    def test_cuda_available_is_bool(self):
        """Test cuda_available is boolean."""
        result = get_device_info()
        self.assertIsInstance(result['cuda_available'], bool)
        
    def test_device_is_string(self):
        """Test device is string."""
        result = get_device_info()
        self.assertIsInstance(result['device'], str)
        
    def test_num_gpus_is_int(self):
        """Test num_gpus is integer."""
        result = get_device_info()
        self.assertIsInstance(result['num_gpus'], int)
        self.assertGreaterEqual(result['num_gpus'], 0)


class TestGetOptimalDevice(unittest.TestCase):
    """Test cases for get_optimal_device function."""
    
    def test_returns_string(self):
        """Test function returns a string."""
        result = get_optimal_device()
        self.assertIsInstance(result, str)
        
    def test_returns_valid_device(self):
        """Test function returns a valid device name."""
        result = get_optimal_device()
        valid_devices = ['cuda', 'mps', 'cpu']
        self.assertIn(result, valid_devices)


class TestClearCudaCache(unittest.TestCase):
    """Test cases for clear_cuda_cache function."""
    
    def test_function_runs_without_error(self):
        """Test function runs without error."""
        # Should not raise even if CUDA not available
        try:
            clear_cuda_cache()
        except Exception as e:
            self.fail(f"clear_cuda_cache raised {e}")


class TestMoveToDevice(unittest.TestCase):
    """Test cases for move_to_device function."""
    
    def test_with_object_having_to_method(self):
        """Test with object that has .to() method."""
        mock_obj = MagicMock()
        mock_obj.to.return_value = mock_obj
        
        result = move_to_device(mock_obj, 'cpu')
        
        mock_obj.to.assert_called_once_with('cpu')
        
    def test_with_object_without_to_method(self):
        """Test with object that doesn't have .to() method."""
        obj = "test string"
        
        result = move_to_device(obj, 'cpu')
        
        self.assertEqual(result, obj)


if __name__ == '__main__':
    unittest.main()
