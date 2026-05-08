"""Helper functions for Emotional Scanner."""
import json
import csv
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging
def format_results(emotions: Dict[str, float], 
                 engagement: Optional[float] = None,
                 vitals: Optional[Dict[str, float]] = None,
                 timestamp: Optional[str] = None) -> Dict[str, Any]:
    """
    Format analysis results into a standardized dictionary.
    Args:
        emotions: Emotion probabilities
        engagement: Engagement score
        vitals: Vital sign measurements
        timestamp: Analysis timestamp
    Returns:
        Formatted results dictionary
    """
    if timestamp is None:
        timestamp = datetime.now().isoformat()
    results = {
        'timestamp': timestamp,
        'emotions': emotions,
        'primary_emotion': max(emotions.items(), key=lambda x: x[1])[0],
        'emotion_confidence': max(emotions.values())
    }
    if engagement is not None:
        results['engagement'] = engagement
    if vitals is not None:
        results['vitals'] = vitals
    return results
def save_results(results: Dict[str, Any], 
                filepath: str,
                format: str = 'json') -> None:
    """
    Save results to file.
    Args:
        results: Results dictionary
        filepath: Output file path
        format: Output format ('json' or 'csv')
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    if format.lower() == 'json':
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    elif format.lower() == 'csv':
        flattened = flatten_dict(results)
        with open(path, 'w', newline='', encoding='utf-8') as f:
            if path.exists() and path.stat().st_size > 0:
                writer = csv.DictWriter(f, fieldnames=flattened.keys())
                writer.writerow(flattened)
            else:
                writer = csv.DictWriter(f, fieldnames=flattened.keys())
                writer.writeheader()
                writer.writerow(flattened)
    else:
        raise ValueError(f"Unsupported format: {format}")
def flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '_') -> Dict[str, Any]:
    """
    Flatten nested dictionary.
    Args:
        d: Input dictionary
        parent_key: Parent key for nesting
        sep: Separator for nested keys
    Returns:
        Flattened dictionary
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)
def setup_logging(log_level: str = 'INFO', log_file: Optional[str] = None) -> logging.Logger:
    """
    Setup logging configuration.
    Args:
        log_level: Logging level
        log_file: Log file path (optional)
    Returns:
        Configured logger
    """
    logger = logging.getLogger('emotional_scanner')
    logger.setLevel(getattr(logging, log_level.upper()))
    logger.handlers.clear()
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger
def calculate_smoothed_values(values: List[float], window_size: int = 5) -> List[float]:
    """
    Calculate moving average smoothing for a list of values.
    Args:
        values: Input values
        window_size: Smoothing window size
    Returns:
        Smoothed values
    """
    if len(values) < window_size:
        return values
    smoothed = []
    for i in range(len(values)):
        start_idx = max(0, i - window_size // 2)
        end_idx = min(len(values), i + window_size // 2 + 1)
        window = values[start_idx:end_idx]
        smoothed.append(np.mean(window))
    return smoothed
def detect_peaks(data: List[float], threshold: float = 0.5, 
                min_distance: int = 5) -> List[int]:
    """
    Detect peaks in time series data.
    Args:
        data: Input data
        threshold: Minimum peak height
        min_distance: Minimum distance between peaks
    Returns:
        List of peak indices
    """
    peaks = []
    for i in range(1, len(data) - 1):
        if (data[i] > threshold and 
            data[i] > data[i-1] and 
            data[i] > data[i+1]):
            if not peaks or i - peaks[-1] >= min_distance:
                peaks.append(i)
    return peaks
def calculate_engagement_trend(engagement_values: List[float], 
                             window_size: int = 10) -> str:
    """
    Calculate engagement trend over time.
    Args:
        engagement_values: Engagement scores over time
        window_size: Analysis window size
    Returns:
        Trend description ('increasing', 'decreasing', 'stable')
    """
    if len(engagement_values) < window_size:
        return 'insufficient_data'
    recent_values = engagement_values[-window_size:]
    early_values = engagement_values[-window_size*2:-window_size] if len(engagement_values) >= window_size*2 else engagement_values[:window_size]
    recent_avg = np.mean(recent_values)
    early_avg = np.mean(early_values)
    diff = recent_avg - early_avg
    if diff > 0.1:
        return 'increasing'
    elif diff < -0.1:
        return 'decreasing'
    else:
        return 'stable'
def validate_emotion_scores(emotions: Dict[str, float]) -> bool:
    """
    Validate emotion scores format and values.
    Args:
        emotions: Emotion scores dictionary
    Returns:
        True if valid, False otherwise
    """
    required_emotions = {'angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'}
    if not required_emotions.issubset(set(emotions.keys())):
        return False
    for score in emotions.values():
        if not isinstance(score, (int, float)) or score < 0 or score > 1:
            return False
    total = sum(emotions.values())
    if abs(total - 1.0) > 0.01:
        return False
    return True
def create_session_id() -> str:
    """
    Create a unique session identifier.
    Returns:
        Session ID string
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")
def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human readable string.
    Args:
        seconds: Duration in seconds
    Returns:
        Formatted duration string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"