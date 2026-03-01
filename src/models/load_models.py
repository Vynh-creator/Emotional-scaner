import os
import urllib.request
import cv2
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2Model

from src.models.classes import EmotionCNN, VideoEngagementModel, DualEyeResNet, VitalSignsModel, Model_MFCC_Wave2Vec_v2


def ensure_yunet(path="models/face_detection_yunet_2023mar.onnx"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        url = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
        urllib.request.urlretrieve(url, path)
    return path


def load_yunet(path_yunet):
    return cv2.FaceDetectorYN.create(
        model=path_yunet,
        config="",
        input_size=(320, 320),
        score_threshold=0.6,
        nms_threshold=0.3,
        top_k=5000,
    )

def _strip_prefix(state_dict, prefix="_orig_mod."):
    if not isinstance(state_dict, dict):
        return state_dict
    if not any(k.startswith(prefix) for k in state_dict.keys()):
        return state_dict
    return { (k[len(prefix):] if k.startswith(prefix) else k): v for k, v in state_dict.items() }


def load_all(device, paths):
    involvement = VideoEngagementModel(num_classes=4, unfreeze_last_block=True).to(device)
    emotion_cnn = EmotionCNN().to(device)

    involvement_state = torch.load(paths["involvement"], map_location=device)
    emotion_cnn_state = torch.load(paths["emotion_cnn"], map_location=device)

    involvement_state = _strip_prefix(involvement_state)
    emotion_cnn_state = _strip_prefix(emotion_cnn_state)

    involvement.load_state_dict(involvement_state, strict=True)
    emotion_cnn.load_state_dict(emotion_cnn_state, strict=True)

    involvement.eval()
    emotion_cnn.eval()

    models_out = {
        "emotion_cnn": emotion_cnn,
        "drowsy": torch.load(paths["drowsy"], map_location=device, weights_only=False).to(device).eval(),
        "involvement": involvement,
        "vitals": torch.load(paths["vitals"], map_location=device, weights_only=False).to(device).eval(),
        "audio": torch.load(paths["audio"], map_location=device, weights_only=False).to(device).eval(),
        "pupil": torch.load(paths["pupil"], map_location=device, weights_only=False).to(device).eval(),
    }

    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    w2v_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").to(device).eval()
    detector = load_yunet(ensure_yunet())

    return w2v_model, processor, models_out, detector
