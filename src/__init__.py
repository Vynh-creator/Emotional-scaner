"""Emotional Scanner - AI-powered emotion recognition system."""
__version__ = "1.0.0"
__author__ = "Emotional Scanner Team"
from .config import Config
from .core import ModelLoader, FaceDetector
from .services import VideoRecorder, AudioProcessor, EmotionAnalyzer
from .utils import DataPreprocessor, Visualizer
__all__ = [
    "Config",
    "ModelLoader", 
    "FaceDetector",
    "VideoRecorder", 
    "AudioProcessor", 
    "EmotionAnalyzer",
    "DataPreprocessor", 
    "Visualizer"
]