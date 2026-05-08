"""Configuration settings for Emotional Scanner."""
import os
from pathlib import Path
from typing import Dict, Any
from dataclasses import dataclass
@dataclass
class ModelConfig:
    """Model configuration settings."""
    emotion_cnn_path: str = "src/models/best_emotion_cnn.pth"
    involvement_path: str = "src/models/best_model_involvement.pth"
    drowsy_path: str = "src/models/best_model_drowsy.pth"
    vitals_path: str = "src/models/best_model_vitals.pth"
    audio_path: str = "src/models/best_model_audio.pth"
    pupil_path: str = "src/models/best_model_pupil.pth"
    yunet_path: str = "models/face_detection_yunet_2023mar.onnx"
    yunet_url: str = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
    input_size: tuple = (320, 320)
    score_threshold: float = 0.6
    nms_threshold: float = 0.3
    top_k: int = 5000
@dataclass
class ProcessingConfig:
    """Data processing configuration."""
    fps: int = 30
    buffer_size: int = 10
    sample_rate: int = 16000
    audio_chunk_size: int = 1024
    device: str = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
@dataclass
class UIConfig:
    """UI configuration settings."""
    window_title: str = "Emotional Scanner"
    window_width: int = 1200
    window_height: int = 800
    primary_color: str = "#2E86AB"
    secondary_color: str = "#A23B72"
    background_color: str = "#F18F01"
    text_color: str = "#C73E1D"
class Config:
    """Main configuration class."""
    def __init__(self):
        self.base_dir = Path(__file__).parent.parent.parent
        self.models = ModelConfig()
        self.processing = ProcessingConfig()
        self.ui = UIConfig()
        self._load_env()
    def _load_env(self):
        """Load configuration from environment variables."""
        env_file = self.base_dir / ".env"
        if env_file.exists():
            from dotenv import load_dotenv
            load_dotenv(env_file)
    def get_model_paths(self) -> Dict[str, str]:
        """Get all model paths with absolute paths."""
        return {
            "emotion_cnn": str(self.base_dir / self.models.emotion_cnn_path),
            "involvement": str(self.base_dir / self.models.involvement_path),
            "drowsy": str(self.base_dir / self.models.drowsy_path),
            "vitals": str(self.base_dir / self.models.vitals_path),
            "audio": str(self.base_dir / self.models.audio_path),
            "pupil": str(self.base_dir / self.models.pupil_path),
        }
    def get_yunet_path(self) -> str:
        """Get absolute path to YuNet model."""
        return str(self.base_dir / self.models.yunet_path)
config = Config()