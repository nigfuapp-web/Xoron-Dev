"""Unit tests for synth/quality_utils.py - Quality utilities."""

import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class TestQualityUtilsModule(unittest.TestCase):
    """Test cases for quality_utils module."""
    
    def test_module_imports(self):
        """Test quality_utils module can be imported."""
        try:
            from synth import quality_utils
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Failed to import quality_utils module: {e}")


if __name__ == '__main__':
    unittest.main()
