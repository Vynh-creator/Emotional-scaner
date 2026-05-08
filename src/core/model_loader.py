"""Model loading and management functionality."""
import os
import urllib.request
import torch
import cv2
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from src.config.settings import config
from src.models.classes import (
    EmotionCNN, 
    VideoEngagementModel, 
    DualEyeResNet, 
    VitalSignsModel, 
    Model_MFCC_Wave2Vec_v2
)
class ModelLoader:
    """Handles loading and management of all ML models."""
    def __init__(self, device: str = None):
        self.device = device or config.processing.device
        self.config = config
        self._models = {}
        self._processor = None
        self._w2v_model = None
        self._face_detector = None
    def _ensure_yunet(self) -> str:
        """Ensure YuNet model is downloaded and return path."""
        path = config.get_yunet_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if not os.path.exists(path):
            print(f"Downloading YuNet model from {config.models.yunet_url}")
            urllib.request.urlretrieve(config.models.yunet_url, path)
        return path
    def _load_face_detector(self):
        """Load face detection model."""
        if self._face_detector is None:
            yunet_path = self._ensure_yunet()
            self._face_detector = cv2.FaceDetectorYN.create(
                model=yunet_path,
                config="",
                input_size=config.models.input_size,
                score_threshold=config.models.score_threshold,
                nms_threshold=config.models.nms_threshold,
                top_k=config.models.top_k,
            )
        return self._face_detector
    def _load_audio_models(self):
        """Load audio processing models."""
        if self._processor is None or self._w2v_model is None:
            self._processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
            self._w2v_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").to(self.device).eval()
    def _strip_prefix(self, state_dict, prefix="_orig_mod."):
        """Strip prefix from state dict keys."""
        if not isinstance(state_dict, dict):
            return state_dict
        if not any(k.startswith(prefix) for k in state_dict.keys()):
            return state_dict
        return { (k[len(prefix):] if k.startswith(prefix) else k): v for k, v in state_dict.items() }
    def load_all_models(self):
        """Load all models at once."""
        paths = config.get_model_paths()
        involvement = VideoEngagementModel(num_classes=4, unfreeze_last_block=True).to(self.device)
        emotion_cnn = EmotionCNN().to(self.device)
        involvement_state = torch.load(paths["involvement"], map_location=self.device)
        emotion_cnn_state = torch.load(paths["emotion_cnn"], map_location=self.device)
        involvement_state = self._strip_prefix(involvement_state)
        emotion_cnn_state = self._strip_prefix(emotion_cnn_state)
        involvement.load_state_dict(involvement_state, strict=True)
        emotion_cnn.load_state_dict(emotion_cnn_state, strict=True)
        involvement.eval()
        emotion_cnn.eval()
        self._models = {
            "emotion_cnn": emotion_cnn,
            "drowsy": torch.load(paths["drowsy"], map_location=self.device, weights_only=False).to(self.device).eval(),
            "involvement": involvement,
            "vitals": torch.load(paths["vitals"], map_location=self.device, weights_only=False).to(self.device).eval(),
            "audio": torch.load(paths["audio"], map_location=self.device, weights_only=False).to(self.device).eval(),
            "pupil": torch.load(paths["pupil"], map_location=self.device, weights_only=False).to(self.device).eval(),
        }
        self._load_audio_models()
        self._load_face_detector()
        return self._processor, self._w2v_model, self._models, self._face_detector
    def get_model(self, name: str):
        """Get a specific model by name."""
        if not self._models:
            self.load_all_models()
        return self._models.get(name)
    def get_audio_models(self):
        """Get audio processing models."""
        if not self._processor or not self._w2v_model:
            self.load_all_models()
        return self._processor, self._w2v_model
    def get_face_detector(self):
        """Get face detector model."""
        if self._face_detector is None:
            self.load_all_models()
        return self._face_detector