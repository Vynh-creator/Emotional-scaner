from src.utils.snake_case import preprocess_faces_array, process_audio_chunk, preprocess_pupil
from dotenv import load_dotenv
import os
import requests
import torch
load_dotenv()
def preprocess_data(video_frames, audio_chunk,device, audio_sr, processor, w2v_model,detector):
    x_drowsy, x_involvement, x_emotion,x_vitals = preprocess_faces_array(
        frames_array=video_frames,
        detector=detector,
        device=device,
        num_faces=3
    )
    x_mfcc, x_w2v = process_audio_chunk(audio_chunk, audio_sr, device,processor,w2v_model)
    if x_mfcc is None or x_w2v is None:
        x_mfcc = torch.zeros((1, 13, 100), device=device)
        x_w2v = torch.zeros((1, 100, 768), device=device)
    left_eye, right_eye = preprocess_pupil(video_frames, device)
    return x_drowsy, x_involvement, x_emotion, (x_mfcc, x_w2v), (left_eye, right_eye),x_vitals
import torch
@torch.no_grad()
def get_predictions_arrays(models,x_drowsy, x_involvement, x_emotion, x_audio, x_pupil,x_vitals):
    out = {}
    if models['drowsy'] is not None:
        y_drowsy = models['drowsy'](x_drowsy)
        out["drowsy_logits"] = y_drowsy.detach()
    else:
        out["drowsy_logits"] = None
    if models['involvement'] is not None:
        y_involvement = models['involvement'](x_involvement)
        out["involvement_logits"] = y_involvement.detach()
    else:
        out["involvement_logits"] = None
    if models['emotion_cnn'] is not None:
        y_emotion = models['emotion_cnn'](x_emotion)
        out["emotion_logits"] = y_emotion.detach()
        out["emotion_probs"] = torch.softmax(y_emotion.detach(), dim=-1)
    else:
        out["emotion_logits"] = None
        out["emotion_probs"] = None
    mfcc, w2v = x_audio
    out["audio_mfcc"] = mfcc.detach()
    out["audio_w2v"] = w2v.detach()
    if models['audio'] is not None:
        y_audio = models['audio'](mfcc.unsqueeze(0), w2v.unsqueeze(0))
        out["audio_logits"] = y_audio.detach()
    else:
        out["audio_logits"] = None
    left_eye, right_eye = x_pupil
    if models['pupil'] is not None:
        y_pupil = models['pupil'](left_eye, right_eye)
        out["pupil_pred"] = y_pupil.detach()
    else:
        out["pupil_pred"] = None
    if models['vitals'] is not None:
        y_vitals = models['vitals'](x_vitals)
        out["vitals_pred"] = y_vitals.detach()
    else:
        out["vitals_pred"] = None
    return out
def deepseek_chat(prompt: str,
                 token: str | None = None,
                 model: str = "deepseek-chat",
                 system: str = "You are a helpful assistant.",
                 timeout: float = 30.0) -> str:
    api_key = token or os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("No DeepSeek API key. Set DEEPSEEK_API_KEY or pass token=...")
    url = "https://api.deepseek.com/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
    }
    r = requests.post(url, headers=headers, json=payload, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"]