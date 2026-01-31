"""Unit tests for training/utils.py - Training utilities."""

import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class TestTrainingUtilsModule(unittest.TestCase):
    """Test cases for training utils module."""
    
    def test_module_imports(self):
        """Test training utils module can be imported."""
        try:
            from training import utils
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Failed to import training utils module: {e}")


if __name__ == '__main__':
    unittest.main()
