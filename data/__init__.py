"""Data module for Xoron-Dev."""

from data.formatters import MultimodalFormatter
from data.processors import VoiceProcessor

# TrueStreamingDataset requires torch, import conditionally
try:
    from data.dataset import TrueStreamingDataset
    __all__ = ['MultimodalFormatter', 'TrueStreamingDataset', 'VoiceProcessor']
except ImportError:
    TrueStreamingDataset = None
    __all__ = ['MultimodalFormatter', 'VoiceProcessor']
