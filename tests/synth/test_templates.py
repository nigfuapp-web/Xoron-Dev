"""Unit tests for synth/templates.py - Synthetic data templates."""

import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class TestTemplatesModule(unittest.TestCase):
    """Test cases for templates module."""
    
    def test_module_imports(self):
        """Test templates module can be imported."""
        try:
            from synth import templates
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Failed to import templates module: {e}")


if __name__ == '__main__':
    unittest.main()
