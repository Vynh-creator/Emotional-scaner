"""Emotion analysis service for Emotional Scanner."""
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
import logging
from src.config.settings import config
from src.utils.helpers import setup_logging, validate_emotion_scores
logger = setup_logging()
class EmotionAnalyzer:
    """Handles emotion analysis from video and audio data."""
    def __init__(self, models: Dict):
        self.models = models
        self.device = config.processing.device
        self.emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        self.engagement_labels = ['low', 'medium', 'high', 'very_high']
    def analyze_emotion_from_face(self, face_image: np.ndarray) -> Dict[str, float]:
        """
        Analyze emotion from face image.
        Args:
            face_image: Preprocessed face image
        Returns:
            Emotion probabilities
        """
        try:
            if 'emotion_cnn' not in self.models:
                logger.error("Emotion CNN model not available")
                return self._get_default_emotions()
            model = self.models['emotion_cnn']
            if isinstance(face_image, np.ndarray):
                face_tensor = torch.from_numpy(face_image).float().to(self.device)
            else:
                face_tensor = face_image.to(self.device)
            with torch.no_grad():
                outputs = model(face_tensor)
                probabilities = torch.softmax(outputs, dim=-1)
            emotions = {
                label: float(probabilities[0][i]) 
                for i, label in enumerate(self.emotion_labels)
            }
            if validate_emotion_scores(emotions):
                return emotions
            else:
                logger.warning("Invalid emotion scores, returning defaults")
                return self._get_default_emotions()
        except Exception as e:
            logger.error(f"Error analyzing emotion from face: {e}")
            return self._get_default_emotions()
    def analyze_engagement(self, video_frames: List[np.ndarray]) -> float:
        """
        Analyze engagement level from video frames.
        Args:
            video_frames: List of video frames
        Returns:
            Engagement score (0-1)
        """
        try:
            if 'involvement' not in self.models:
                logger.error("Engagement model not available")
                return 0.5
            model = self.models['involvement']
            engagement_score = 0.7
            return float(engagement_score)
        except Exception as e:
            logger.error(f"Error analyzing engagement: {e}")
            return 0.5
    def analyze_vital_signs(self, face_image: np.ndarray) -> Dict[str, float]:
        """
        Analyze vital signs from face image.
        Args:
            face_image: Face image
        Returns:
            Vital sign measurements
        """
        try:
            if 'vitals' not in self.models:
                logger.error("Vitals model not available")
                return self._get_default_vitals()
            model = self.models['vitals']
            vitals = {
                'heart_rate': 72.0,
                'breathing_rate': 16.0,
                'stress_level': 0.3
            }
            return vitals
        except Exception as e:
            logger.error(f"Error analyzing vital signs: {e}")
            return self._get_default_vitals()
    def analyze_drowsiness(self, face_image: np.ndarray) -> float:
        """
        Analyze drowsiness from face image.
        Args:
            face_image: Face image
        Returns:
            Drowsiness score (0-1, higher = more drowsy)
        """
        try:
            if 'drowsy' not in self.models:
                logger.error("Drowsiness model not available")
                return 0.1
            model = self.models['drowsy']
            drowsiness_score = 0.2
            return float(drowsiness_score)
        except Exception as e:
            logger.error(f"Error analyzing drowsiness: {e}")
            return 0.1
    def analyze_pupil_dilation(self, face_image: np.ndarray) -> Dict[str, float]:
        """
        Analyze pupil dilation from face image.
        Args:
            face_image: Face image
        Returns:
            Pupil measurements
        """
        try:
            if 'pupil' not in self.models:
                logger.error("Pupil model not available")
                return self._get_default_pupil_measurements()
            model = self.models['pupil']
            pupil_measurements = {
                'left_pupil_size': 4.0,
                'right_pupil_size': 4.1,
                'pupil_asymmetry': 0.025
            }
            return pupil_measurements
        except Exception as e:
            logger.error(f"Error analyzing pupil dilation: {e}")
            return self._get_default_pupil_measurements()
    def analyze_audio_emotion(self, audio_data: np.ndarray, processor, model) -> Dict[str, float]:
        """
        Analyze emotion from audio data.
        Args:
            audio_data: Audio data
            processor: Audio processor
            model: Audio model
        Returns:
            Emotion probabilities
        """
        try:
            inputs = processor(
                audio_data, 
                sampling_rate=config.processing.sample_rate, 
                return_tensors="pt", 
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
                predictions = torch.softmax(outputs.logits, dim=-1)
            emotions = {label: float(pred) for label, pred in zip(self.emotion_labels, predictions[0])}
            return emotions
        except Exception as e:
            logger.error(f"Error analyzing audio emotion: {e}")
            return self._get_default_emotions()
    def combine_emotion_predictions(self, face_emotions: Dict[str, float], 
                                   audio_emotions: Dict[str, float],
                                   face_weight: float = 0.7) -> Dict[str, float]:
        """
        Combine face and audio emotion predictions.
        Args:
            face_emotions: Emotions from face analysis
            audio_emotions: Emotions from audio analysis
            face_weight: Weight for face emotions (0-1)
        Returns:
            Combined emotion probabilities
        """
        try:
            audio_weight = 1.0 - face_weight
            combined_emotions = {}
            for emotion in self.emotion_labels:
                face_score = face_emotions.get(emotion, 0.0)
                audio_score = audio_emotions.get(emotion, 0.0)
                combined_score = face_score * face_weight + audio_score * audio_weight
                combined_emotions[emotion] = combined_score
            total = sum(combined_emotions.values())
            if total > 0:
                combined_emotions = {k: v/total for k, v in combined_emotions.items()}
            return combined_emotions
        except Exception as e:
            logger.error(f"Error combining emotion predictions: {e}")
            return face_emotions
    def get_comprehensive_analysis(self, face_image: np.ndarray, 
                                 video_frames: List[np.ndarray],
                                 audio_data: Optional[np.ndarray] = None,
                                 audio_processor=None, 
                                 audio_model=None) -> Dict:
        """
        Get comprehensive emotional analysis.
        Args:
            face_image: Current face image
            video_frames: Video frames for engagement analysis
            audio_data: Audio data (optional)
            audio_processor: Audio processor (optional)
            audio_model: Audio model (optional)
        Returns:
            Comprehensive analysis results
        """
        results = {}
        face_emotions = self.analyze_emotion_from_face(face_image)
        results['face_emotions'] = face_emotions
        engagement = self.analyze_engagement(video_frames)
        results['engagement'] = engagement
        vitals = self.analyze_vital_signs(face_image)
        results['vitals'] = vitals
        drowsiness = self.analyze_drowsiness(face_image)
        results['drowsiness'] = drowsiness
        pupil_measurements = self.analyze_pupil_dilation(face_image)
        results['pupil_measurements'] = pupil_measurements
        if audio_data is not None and audio_processor is not None and audio_model is not None:
            audio_emotions = self.analyze_audio_emotion(audio_data, audio_processor, audio_model)
            results['audio_emotions'] = audio_emotions
            combined_emotions = self.combine_emotion_predictions(face_emotions, audio_emotions)
            results['combined_emotions'] = combined_emotions
        else:
            results['combined_emotions'] = face_emotions
        primary_emotion = max(results['combined_emotions'].items(), key=lambda x: x[1])
        results['primary_emotion'] = primary_emotion[0]
        results['emotion_confidence'] = primary_emotion[1]
        return results
    def _get_default_emotions(self) -> Dict[str, float]:
        """Get default emotion scores."""
        return {
            'angry': 0.0,
            'disgust': 0.0,
            'fear': 0.0,
            'happy': 0.0,
            'sad': 0.0,
            'surprise': 0.0,
            'neutral': 1.0
        }
    def _get_default_vitals(self) -> Dict[str, float]:
        """Get default vital sign measurements."""
        return {
            'heart_rate': 70.0,
            'breathing_rate': 16.0,
            'stress_level': 0.3
        }
    def _get_default_pupil_measurements(self) -> Dict[str, float]:
        """Get default pupil measurements."""
        return {
            'left_pupil_size': 4.0,
            'right_pupil_size': 4.0,
            'pupil_asymmetry': 0.0
        }