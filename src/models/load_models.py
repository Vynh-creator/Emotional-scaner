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
    print("Loading models...")
    import sys
    original_modules = sys.modules.copy()
    sys.modules['models.classes'] = sys.modules.get('src.models.classes')
    sys.modules['models.load_models'] = sys.modules.get('src.models.load_models')
    print("Loading core models...")
    involvement = VideoEngagementModel(num_classes=4, unfreeze_last_block=True).to(device)
    emotion_cnn = EmotionCNN().to(device)
    try:
        involvement_state = torch.load(paths["involvement"], map_location=device, weights_only=False)
        emotion_cnn_state = torch.load(paths["emotion_cnn"], map_location=device, weights_only=False)
    except Exception as e:
        print(f"Error loading model states: {e}")
        involvement_state = torch.load(paths["involvement"], map_location=device, weights_only=False)
        emotion_cnn_state = torch.load(paths["emotion_cnn"], map_location=device, weights_only=False)
    involvement_state = _strip_prefix(involvement_state)
    emotion_cnn_state = _strip_prefix(emotion_cnn_state)
    involvement.load_state_dict(involvement_state, strict=True)
    emotion_cnn.load_state_dict(emotion_cnn_state, strict=True)
    involvement.eval()
    emotion_cnn.eval()
    models_out = {
        "emotion_cnn": emotion_cnn,
        "involvement": involvement,
    }
    model_keys = ["drowsy", "vitals", "audio", "pupil"]
    for key in model_keys:
        try:
            print(f"Loading {key} model...")
            models_out[key] = torch.load(paths[key], map_location=device, weights_only=False).to(device).eval()
        except Exception as e:
            print(f"Warning: Could not load {key} model: {e}")
            models_out[key] = None
    print("Loading audio models...")
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    w2v_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").to(device).eval()
    print("Loading face detector...")
    try:
        detector = load_yunet(ensure_yunet())
    except Exception as e:
        print(f"Error loading face detector: {e}")
        detector = None
    sys.modules.clear()
    sys.modules.update(original_modules)
    print("Models loaded successfully!")
    return w2v_model, processor, models_out, detector