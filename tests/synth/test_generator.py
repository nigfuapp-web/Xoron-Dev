"""Unit tests for synth/generator.py - Synthetic data generator."""

import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class TestGeneratorModule(unittest.TestCase):
    """Test cases for generator module."""
    
    def test_module_imports(self):
        """Test generator module can be imported."""
        try:
            from synth import generator
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Failed to import generator module: {e}")


if __name__ == '__main__':
    unittest.main()
