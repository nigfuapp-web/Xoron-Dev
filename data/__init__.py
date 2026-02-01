"""Data module for Xoron-Dev."""

from data.formatters import MultimodalFormatter
from data.processors import VoiceProcessor

# TrueStreamingDataset requires torch, import conditionally
try:
    from data.dataset import TrueStreamingDataset, create_train_eval_datasets
    __all__ = ['MultimodalFormatter', 'TrueStreamingDataset', 'VoiceProcessor', 'create_train_eval_datasets']
except ImportError:
    TrueStreamingDataset = None
    create_train_eval_datasets = None
    __all__ = ['MultimodalFormatter', 'VoiceProcessor']
