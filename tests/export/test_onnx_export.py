"""Unit tests for export/onnx_export.py - ONNX export functionality."""

import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class TestOnnxExportModule(unittest.TestCase):
    """Test cases for ONNX export module."""
    
    def test_module_imports(self):
        """Test onnx_export module can be imported."""
        try:
            from export import onnx_export
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Failed to import onnx_export module: {e}")


if __name__ == '__main__':
    unittest.main()
