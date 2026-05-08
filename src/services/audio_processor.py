"""Audio processing service for Emotional Scanner."""
import numpy as np
import torch
import sounddevice as sd
import librosa
from typing import Optional, Callable, Tuple
import threading
import queue
from src.config.settings import config
from src.utils.preprocessing import DataPreprocessor
from src.utils.helpers import setup_logging
logger = setup_logging()
class AudioProcessor:
    """Handles audio capture and processing."""
    def __init__(self, callback: Optional[Callable] = None):
        self.config = config.processing
        self.preprocessor = DataPreprocessor()
        self.callback = callback
        self.sample_rate = self.config.sample_rate
        self.chunk_size = self.config.audio_chunk_size
        self.is_recording = False
        self.audio_queue = queue.Queue()
        self.audio_model = None
        self.processor = None
    def set_models(self, audio_model, processor):
        """Set audio processing models."""
        self.audio_model = audio_model
        self.processor = processor
    def start_recording(self):
        """Start audio recording."""
        if self.is_recording:
            logger.warning("Audio recording already in progress")
            return
        self.is_recording = True
        self.audio_thread = threading.Thread(target=self._record_audio)
        self.audio_thread.daemon = True
        self.audio_thread.start()
        logger.info("Started audio recording")
    def stop_recording(self):
        """Stop audio recording."""
        if not self.is_recording:
            logger.warning("No audio recording in progress")
            return
        self.is_recording = False
        if hasattr(self, 'audio_thread'):
            self.audio_thread.join(timeout=1.0)
        logger.info("Stopped audio recording")
    def _record_audio(self):
        """Audio recording thread function."""
        def audio_callback(indata, frames, time, status):
            if status:
                logger.warning(f"Audio callback status: {status}")
            self.audio_queue.put(indata.copy())
        try:
            with sd.InputStream(
                callback=audio_callback,
                channels=1,
                samplerate=self.sample_rate,
                blocksize=self.chunk_size
            ):
                while self.is_recording:
                    sd.sleep(100)
        except Exception as e:
            logger.error(f"Error in audio recording: {e}")
            self.is_recording = False
    def process_audio_chunk(self, audio_data: np.ndarray) -> Optional[torch.Tensor]:
        """
        Process a chunk of audio data.
        Args:
            audio_data: Raw audio data
        Returns:
            Processed audio tensor
        """
        try:
            audio_tensor, _ = self.preprocessor.preprocess_audio(
                audio_data, self.sample_rate
            )
            return audio_tensor
        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}")
            return None
    def extract_features(self, audio_data: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract audio features for analysis.
        Args:
            audio_data: Raw audio data
        Returns:
            Audio features
        """
        try:
            mfcc_features = self.preprocessor.extract_mfcc_features(
                audio_data, self.sample_rate
            )
            return mfcc_features
        except Exception as e:
            logger.error(f"Error extracting audio features: {e}")
            return None
    def predict_emotion_from_audio(self, audio_data: np.ndarray) -> Optional[dict]:
        """
        Predict emotion from audio data.
        Args:
            audio_data: Raw audio data
        Returns:
            Emotion predictions
        """
        if self.audio_model is None or self.processor is None:
            logger.warning("Audio models not loaded")
            return None
        try:
            inputs = self.processor(
                audio_data, 
                sampling_rate=self.sample_rate, 
                return_tensors="pt", 
                padding=True
            )
            inputs = {k: v.to(self.audio_model.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.audio_model(**inputs)
                predictions = torch.softmax(outputs.logits, dim=-1)
            emotion_labels = ['neutral', 'happy', 'sad', 'angry', 'fear', 'disgust', 'surprise']
            emotions = {label: float(pred) for label, pred in zip(emotion_labels, predictions[0])}
            return emotions
        except Exception as e:
            logger.error(f"Error predicting emotion from audio: {e}")
            return None
    def get_audio_level(self, audio_data: np.ndarray) -> float:
        """
        Calculate audio level (RMS).
        Args:
            audio_data: Audio data
        Returns:
            RMS audio level
        """
        return float(np.sqrt(np.mean(audio_data**2)))
    def detect_silence(self, audio_data: np.ndarray, threshold: float = 0.01) -> bool:
        """
        Detect if audio segment is silent.
        Args:
            audio_data: Audio data
            threshold: Silence threshold
        Returns:
            True if silent, False otherwise
        """
        level = self.get_audio_level(audio_data)
        return level < threshold
    def process_queue(self) -> Optional[np.ndarray]:
        """
        Process all audio data in queue.
        Returns:
            Combined audio data
        """
        audio_chunks = []
        while not self.audio_queue.empty():
            try:
                chunk = self.audio_queue.get_nowait()
                audio_chunks.append(chunk)
            except queue.Empty:
                break
        if not audio_chunks:
            return None
        combined_audio = np.concatenate(audio_chunks, axis=0)
        return combined_audio
    def analyze_audio_stream(self, duration: float = 1.0) -> Optional[dict]:
        """
        Analyze audio stream for specified duration.
        Args:
            duration: Analysis duration in seconds
        Returns:
            Analysis results
        """
        frames_needed = int(duration * self.sample_rate / self.chunk_size)
        audio_chunks = []
        for _ in range(frames_needed):
            if not self.is_recording:
                break
            try:
                chunk = self.audio_queue.get(timeout=0.1)
                audio_chunks.append(chunk)
            except queue.Empty:
                continue
        if not audio_chunks:
            return None
        combined_audio = np.concatenate(audio_chunks, axis=0)
        features = self.extract_features(combined_audio)
        emotions = self.predict_emotion_from_audio(combined_audio)
        level = self.get_audio_level(combined_audio)
        results = {
            'audio_level': level,
            'is_silent': self.detect_silence(combined_audio),
            'features': features.tolist() if features is not None else None,
            'emotions': emotions
        }
        if self.callback:
            self.callback(results)
        return results