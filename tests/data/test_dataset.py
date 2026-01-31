"""Unit tests for data/dataset.py - Dataset classes."""

import sys
import os
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class TestTrueStreamingDatasetStructure(unittest.TestCase):
    """Test cases for TrueStreamingDataset structure."""
    
    def test_class_exists(self):
        """Test TrueStreamingDataset class exists."""
        from data.dataset import TrueStreamingDataset
        self.assertTrue(callable(TrueStreamingDataset))
        
    def test_class_is_dataset(self):
        """Test TrueStreamingDataset inherits from Dataset."""
        from data.dataset import TrueStreamingDataset
        from torch.utils.data import Dataset
        
        self.assertTrue(issubclass(TrueStreamingDataset, Dataset))


class TestDatasetModuleImports(unittest.TestCase):
    """Test cases for dataset module imports."""
    
    def test_module_imports(self):
        """Test dataset module can be imported."""
        try:
            from data import dataset
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Failed to import dataset module: {e}")
            
    def test_true_streaming_dataset_import(self):
        """Test TrueStreamingDataset can be imported."""
        try:
            from data.dataset import TrueStreamingDataset
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Failed to import TrueStreamingDataset: {e}")


if __name__ == '__main__':
    unittest.main()
