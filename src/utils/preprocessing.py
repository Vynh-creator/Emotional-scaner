"""Data preprocessing utilities for Emotional Scanner."""
import cv2
import numpy as np
import librosa
from typing import Tuple, List, Optional
import torch
from src.config.settings import config
class DataPreprocessor:
    """Handles preprocessing of video and audio data."""
    def __init__(self):
        self.config = config.processing
    def preprocess_video_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess a single video frame.
        Args:
            frame: Input frame in BGR format
        Returns:
            Preprocessed frame
        """
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame.astype(np.float32) / 255.0
        frame = np.transpose(frame, (2, 0, 1))
        return frame
    def preprocess_video_sequence(self, frames: List[np.ndarray], sequence_length: int = 10) -> np.ndarray:
        """
        Preprocess a sequence of video frames.
        Args:
            frames: List of video frames
            sequence_length: Target sequence length
        Returns:
            Preprocessed video sequence tensor
        """
        if len(frames) < sequence_length:
            frames.extend([frames[-1]] * (sequence_length - len(frames)))
        elif len(frames) > sequence_length:
            indices = np.linspace(0, len(frames) - 1, sequence_length, dtype=int)
            frames = [frames[i] for i in indices]
        processed_frames = [self.preprocess_video_frame(frame) for frame in frames]
        video_tensor = np.stack(processed_frames, axis=0)
        video_tensor = np.transpose(video_tensor, (1, 0, 2, 3))
        return torch.from_numpy(video_tensor).unsqueeze(0)
    def preprocess_audio(self, audio_data: np.ndarray, sample_rate: int = None) -> Tuple[torch.Tensor, int]:
        """
        Preprocess audio data for model input.
        Args:
            audio_data: Raw audio data
            sample_rate: Audio sample rate
        Returns:
            Preprocessed audio tensor and sample rate
        """
        if sample_rate is None:
            sample_rate = self.config.sample_rate
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        if sample_rate != self.config.sample_rate:
            audio_data = librosa.resample(
                audio_data, 
                orig_sr=sample_rate, 
                target_sr=self.config.sample_rate
            )
            sample_rate = self.config.sample_rate
        audio_data = librosa.util.normalize(audio_data)
        audio_tensor = torch.from_numpy(audio_data).float()
        return audio_tensor, sample_rate
    def extract_mfcc_features(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Extract MFCC features from audio.
        Args:
            audio_data: Audio data
            sample_rate: Sample rate
        Returns:
            MFCC features
        """
        mfcc = librosa.feature.mfcc(
            y=audio_data,
            sr=sample_rate,
            n_mfcc=13,
            n_fft=2048,
            hop_length=512
        )
        delta_mfcc = librosa.feature.delta(mfcc)
        delta2_mfcc = librosa.feature.delta(mfcc, order=2)
        features = np.concatenate([mfcc, delta_mfcc, delta2_mfcc], axis=0)
        return features
    def preprocess_for_engagement(self, frames: List[np.ndarray]) -> torch.Tensor:
        """
        Preprocess video frames for engagement model.
        Args:
            frames: List of video frames
        Returns:
            Preprocessed tensor for engagement model
        """
        resized_frames = []
        for frame in frames:
            frame = cv2.resize(frame, (224, 224))
            frame = self.preprocess_video_frame(frame)
            resized_frames.append(frame)
        video_tensor = np.stack(resized_frames, axis=0)
        video_tensor = np.transpose(video_tensor, (1, 0, 2, 3))
        return torch.from_numpy(video_tensor).unsqueeze(0)
    def create_sliding_windows(self, data: np.ndarray, window_size: int, step: int = 1) -> List[np.ndarray]:
        """
        Create sliding windows from sequential data.
        Args:
            data: Input sequential data
            window_size: Size of each window
            step: Step size between windows
        Returns:
            List of windows
        """
        windows = []
        for i in range(0, len(data) - window_size + 1, step):
            window = data[i:i + window_size]
            windows.append(window)
        return windows