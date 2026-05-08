"""Face detection and processing functionality."""
import cv2
import numpy as np
from typing import List, Tuple, Optional
from src.config.settings import config
class FaceDetector:
    """Handles face detection and facial landmark extraction."""
    def __init__(self, detector=None):
        self.detector = detector
        self.config = config.models
    def detect_faces(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Detect faces in an image.
        Args:
            image: Input image in BGR format
        Returns:
            List of face detections, each containing [x, y, w, h, confidence, landmarks...]
        """
        if self.detector is None:
            raise ValueError("Face detector not initialized")
        height, width = image.shape[:2]
        if width != self.config.input_size[0] or height != self.config.input_size[1]:
            image = cv2.resize(image, self.config.input_size)
        faces = self.detector.detect(image)
        if faces[1] is not None:
            return faces[1]
        return []
    def extract_face_roi(self, image: np.ndarray, face: np.ndarray) -> np.ndarray:
        """
        Extract face region of interest from image.
        Args:
            image: Input image
            face: Face detection result
        Returns:
            Cropped face image
        """
        x, y, w, h = face[:4].astype(int)
        x = max(0, x)
        y = max(0, y)
        w = min(w, image.shape[1] - x)
        h = min(h, image.shape[0] - y)
        return image[y:y+h, x:x+w]
    def get_facial_landmarks(self, face: np.ndarray) -> np.ndarray:
        """
        Extract facial landmarks from face detection.
        Args:
            face: Face detection result
        Returns:
            Facial landmarks array
        """
        return face[4:]
    def align_face(self, image: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        """
        Align face based on eye landmarks.
        Args:
            image: Input face image
            landmarks: Facial landmarks
        Returns:
            Aligned face image
        """
        left_eye = landmarks[0]
        right_eye = landmarks[1]
        dy = right_eye[1] - left_eye[1]
        dx = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dy, dx))
        eye_center = ((left_eye[0] + right_eye[0]) / 2, (left_eye[1] + right_eye[1]) / 2)
        rotation_matrix = cv2.getRotationMatrix2D(eye_center, angle, 1.0)
        aligned_face = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))
        return aligned_face
    def preprocess_for_emotion(self, face_image: np.ndarray, target_size: tuple = (48, 48)) -> np.ndarray:
        """
        Preprocess face image for emotion recognition.
        Args:
            face_image: Input face image
            target_size: Target size for emotion model
        Returns:
            Preprocessed face image
        """
        if len(face_image.shape) == 3:
            face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        face_image = cv2.resize(face_image, target_size)
        face_image = face_image.astype(np.float32) / 255.0
        face_image = np.expand_dims(face_image, axis=0)
        face_image = np.expand_dims(face_image, axis=0)
        return face_image