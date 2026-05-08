"""Visualization utilities for Emotional Scanner."""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    sns = None
from src.config.settings import config
class Visualizer:
    """Handles visualization of results and data."""
    def __init__(self):
        self.config = config.ui
        self.emotion_colors = {
            'angry': (0, 0, 255),
            'disgust': (0, 128, 0),
            'fear': (0, 255, 255),
            'happy': (0, 255, 0),
            'sad': (255, 0, 0),
            'surprise': (0, 165, 255),
            'neutral': (128, 128, 128)
        }
    def draw_face_detections(self, image: np.ndarray, faces: List[np.ndarray], 
                            emotions: Optional[List[Dict]] = None) -> np.ndarray:
        """
        Draw face detection boxes and emotions on image.
        Args:
            image: Input image
            faces: List of face detections
            emotions: List of emotion predictions for each face
        Returns:
            Image with visualizations
        """
        vis_image = image.copy()
        for i, face in enumerate(faces):
            x, y, w, h = face[:4].astype(int)
            confidence = face[4]
            cv2.rectangle(vis_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(vis_image, f'{confidence:.2f}', (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            if emotions and i < len(emotions):
                emotion_data = emotions[i]
                if 'emotion' in emotion_data:
                    emotion = emotion_data['emotion']
                    color = self.emotion_colors.get(emotion, (255, 255, 255))
                    label = f"{emotion}: {emotion_data.get('confidence', 0):.2f}"
                    cv2.putText(vis_image, label, (x, y + h + 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return vis_image
    def draw_facial_landmarks(self, image: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        """
        Draw facial landmarks on image.
        Args:
            image: Input image
            landmarks: Facial landmarks
        Returns:
            Image with landmarks
        """
        vis_image = image.copy()
        if len(landmarks.shape) == 1:
            landmarks = landmarks.reshape(-1, 2)
        for i, (x, y) in enumerate(landmarks):
            cv2.circle(vis_image, (int(x), int(y)), 3, (0, 0, 255), -1)
            cv2.putText(vis_image, str(i), (int(x) + 5, int(y) - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        return vis_image
    def plot_emotion_distribution(self, emotions: Dict[str, float], 
                                title: str = "Emotion Distribution") -> plt.Figure:
        """
        Plot emotion distribution as bar chart.
        Args:
            emotions: Dictionary of emotion probabilities
            title: Plot title
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        emotions_list = list(emotions.keys())
        values = list(emotions.values())
        colors = [self.emotion_colors.get(emotion, (0.5, 0.5, 0.5)) for emotion in emotions_list]
        bars = ax.bar(emotions_list, values, color=colors)
        ax.set_title(title)
        ax.set_ylabel('Probability')
        ax.set_ylim(0, 1)
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.3f}', ha='center', va='bottom')
        plt.xticks(rotation=45)
        plt.tight_layout()
        return fig
    def plot_time_series(self, data: Dict[str, List[float]], 
                        title: str = "Metrics Over Time") -> plt.Figure:
        """
        Plot time series data.
        Args:
            data: Dictionary of metric names to values over time
            title: Plot title
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        for metric_name, values in data.items():
            ax.plot(values, label=metric_name, linewidth=2)
        ax.set_title(title)
        ax.set_xlabel('Time (frames)')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig
    def create_heatmap(self, data: np.ndarray, title: str = "Attention Heatmap") -> plt.Figure:
        """
        Create heatmap visualization.
        Args:
            data: 2D data for heatmap
            title: Plot title
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        if HAS_SEABORN:
            sns.heatmap(data, ax=ax, cmap='viridis', cbar=True)
        else:
            im = ax.imshow(data, cmap='viridis', aspect='auto')
            fig.colorbar(im, ax=ax)
        ax.set_title(title)
        plt.tight_layout()
        return fig
    def draw_metrics_overlay(self, image: np.ndarray, metrics: Dict[str, float]) -> np.ndarray:
        """
        Draw metrics overlay on image.
        Args:
            image: Input image
            metrics: Dictionary of metrics to display
        Returns:
            Image with metrics overlay
        """
        vis_image = image.copy()
        overlay = vis_image.copy()
        cv2.rectangle(overlay, (10, 10), (300, 150), (0, 0, 0), -1)
        vis_image = cv2.addWeighted(overlay, 0.7, vis_image, 0.3, 0)
        y_offset = 30
        for metric_name, value in metrics.items():
            text = f"{metric_name}: {value:.3f}"
            cv2.putText(vis_image, text, (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y_offset += 25
        return vis_image
    def save_frame(self, image: np.ndarray, filepath: str):
        """
        Save frame to file.
        Args:
            image: Image to save
            filepath: Output file path
        """
        cv2.imwrite(filepath, image)
    def show_frame(self, image: np.ndarray, window_name: str = "Emotional Scanner"):
        """
        Display frame in window.
        Args:
            image: Image to display
            window_name: Window title
        """
        cv2.imshow(window_name, image)
    def create_results_summary(self, results: Dict) -> np.ndarray:
        """
        Create a summary image of all results.
        Args:
            results: Dictionary containing all analysis results
        Returns:
            Summary image
        """
        canvas = np.ones((600, 800, 3), dtype=np.uint8) * 255
        cv2.putText(canvas, "Emotional Scanner Results", (50, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2)
        y_offset = 100
        if 'emotions' in results:
            cv2.putText(canvas, "Emotions:", (50, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            y_offset += 30
            for emotion, confidence in results['emotions'].items():
                text = f"  {emotion}: {confidence:.3f}"
                cv2.putText(canvas, text, (50, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
                y_offset += 25
        if 'engagement' in results:
            y_offset += 20
            cv2.putText(canvas, f"Engagement: {results['engagement']:.3f}", 
                       (50, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        if 'vitals' in results:
            y_offset += 30
            cv2.putText(canvas, "Vital Signs:", (50, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            y_offset += 30
            for vital, value in results['vitals'].items():
                text = f"  {vital}: {value:.3f}"
                cv2.putText(canvas, text, (50, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
                y_offset += 25
        return canvas