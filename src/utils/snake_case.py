import cv2
import numpy as np
import torch
import torchaudio
import librosa
import mediapipe as mp

from PIL import Image
from torchvision import transforms


transform_drowsy = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

transform_involvement = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

transform_emotion_cnn = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

transform_vitals = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])


def preprocess_faces_array(frames_array, detector, device, num_faces=16):
    frames_array = np.asarray(frames_array)
    if frames_array.max() <= 1.0:
        frames_array = (frames_array * 255).astype(np.uint8)
    frames_array = frames_array.astype(np.uint8)

    n = len(frames_array)
    k = min(num_faces, n)
    start = max(0, n // 2 - k // 2)
    indices = range(start, start + k)

    faces_rgb_list = []

    for idx in indices:
        frame_bgr = frames_array[idx]
        h, w = frame_bgr.shape[:2]
        detector.setInputSize((w, h))

        _, faces = detector.detect(frame_bgr)
        if faces is None or len(faces) == 0:
            continue

        x, y, bw, bh = faces[0][:4].astype(int)
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(w, x + bw), min(h, y + bh)

        face_bgr = frame_bgr[y1:y2, x1:x2]
        if face_bgr.size == 0:
            continue

        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        faces_rgb_list.append(Image.fromarray(face_rgb))

    while len(faces_rgb_list) < num_faces:
        faces_rgb_list.append(
            faces_rgb_list[-1] if faces_rgb_list else Image.new("RGB", (224, 224), (128, 128, 128))
        )
    faces_rgb_list = faces_rgb_list[:num_faces]

    model1 = torch.stack([transform_drowsy(x) for x in faces_rgb_list]).float().to(device)
    model2 = torch.stack([transform_involvement(x) for x in faces_rgb_list]).float().unsqueeze(0).to(device)
    model3 = torch.stack([transform_emotion_cnn(x) for x in faces_rgb_list]).float().to(device)
    model4 = torch.stack([transform_vitals(x) for x in faces_rgb_list]).float().unsqueeze(0).to(device)

    return model1, model2, model3, model4


melkwargs = {"n_fft": 400, "n_mels": 40, "hop_length": 160}
transform_audio = torchaudio.transforms.MFCC(
    sample_rate=16000,
    n_mfcc=13,
    melkwargs=melkwargs
)

def preprocess_audio(audio, sr, device, processor, w2v_model):
    if isinstance(audio, torch.Tensor):
        audio = audio.detach().cpu().numpy()
    audio = np.asarray(audio, dtype=np.float32)

    if audio.ndim == 2:
        if audio.shape[1] in (1, 2):
            audio = audio.mean(axis=1)
        elif audio.shape[0] in (1, 2):
            audio = audio.mean(axis=0)
        else:
            audio = audio.reshape(-1)
    elif audio.ndim > 2:
        audio = audio.reshape(-1)

    target_sr = 16000
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

    audio_t = torch.from_numpy(audio)
    mfcc = transform_audio(audio_t)

    if mfcc.ndim == 3:
        mfcc = mfcc.squeeze(0)
    if mfcc.shape[0] == 13:
        mfcc = mfcc.transpose(0, 1).contiguous()

    inputs = processor(
        audio,
        sampling_rate=target_sr,
        return_tensors="pt",
        padding=True
    )

    input_values = inputs["input_values"].to(device)
    with torch.no_grad():
        outputs = w2v_model(
            input_values,
            output_hidden_states=True,
            return_dict=True
        )

    w2v = torch.stack(outputs.hidden_states[-4:], dim=0).mean(dim=0).squeeze(0)

    return mfcc.to(device), w2v.to(device)



def process_audio_chunk(audio_chunk, sample_rate, device, processor, w2v_model):
    try:
        mfcc, w2v = preprocess_audio(audio_chunk, sample_rate, device, processor, w2v_model)
        return mfcc, w2v
    except Exception as e:
        print(f"Error processing audio: {e}")
        return None, None


LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)


def middle_k_slice(n, k=8):
    if n <= 0:
        return slice(0, 0)
    k = min(k, n)
    start = max(0, n // 2 - k // 2)
    end = start + k
    if end > n:
        end = n
        start = max(0, end - k)
    return slice(start, end)


def extract_eye_32x16(frame_bgr, left=True, margin=5):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    if not results.multi_face_landmarks:
        return None

    h, w = frame_bgr.shape[:2]
    lm = results.multi_face_landmarks[0].landmark
    idxs = LEFT_EYE if left else RIGHT_EYE
    coords = np.array([(lm[i].x * w, lm[i].y * h) for i in idxs], dtype=np.float32)

    x_min, y_min = np.floor(coords.min(axis=0)).astype(int)
    x_max, y_max = np.ceil(coords.max(axis=0)).astype(int)

    x_min = max(0, x_min - margin)
    y_min = max(0, y_min - margin)
    x_max = min(w, x_max + margin)
    y_max = min(h, y_max + margin)

    crop = frame_bgr[y_min:y_max, x_min:x_max]
    if crop.size == 0:
        return None

    return cv2.resize(crop, (32, 16), interpolation=cv2.INTER_AREA)


def preprocess_pupil(frames_bgr, device, margin=5):
    s = middle_k_slice(len(frames_bgr), k=8)
    mid_frames = frames_bgr[s]

    left_eyes = []
    right_eyes = []

    for frame in mid_frames:
        left_eyes.append(extract_eye_32x16(frame, left=True, margin=margin))
        right_eyes.append(extract_eye_32x16(frame, left=False, margin=margin))

    def fill_none(img):
        if img is None:
            return np.zeros((16, 32, 3), dtype=np.uint8)
        return img

    left_eyes = np.stack([fill_none(x) for x in left_eyes], axis=0)
    right_eyes = np.stack([fill_none(x) for x in right_eyes], axis=0)

    left_t = torch.from_numpy(left_eyes).permute(0, 3, 1, 2).float().div(255.0).to(device)
    right_t = torch.from_numpy(right_eyes).permute(0, 3, 1, 2).float().div(255.0).to(device)

    return left_t, right_t
