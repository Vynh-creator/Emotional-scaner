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
    left_eye, right_eye = preprocess_pupil(video_frames, device)

    return x_drowsy, x_involvement, x_emotion, (x_mfcc, x_w2v), (left_eye, right_eye),x_vitals


import torch

@torch.no_grad()
def get_predictions_arrays(models,x_drowsy, x_involvement, x_emotion, x_audio, x_pupil,x_vitals):
    out = {}

    y_drowsy = models['drowsy'](x_drowsy)
    out["drowsy_logits"] = y_drowsy.detach()

    y_involvement = models['involvement'](x_involvement)
    out["involvement_logits"] = y_involvement.detach()

    y_emotion = models['emotion_cnn'](x_emotion)
    out["emotion_logits"] = y_emotion.detach()
    out["emotion_probs"] = torch.softmax(y_emotion.detach(), dim=-1)

    mfcc, w2v = x_audio
    out["audio_mfcc"] = mfcc.detach()
    out["audio_w2v"] = w2v.detach()

    y_audio = models['audio'](mfcc.unsqueeze(0), w2v.unsqueeze(0))
    out["audio_logits"] = y_audio.detach()

    left_eye, right_eye = x_pupil
    y_pupil = models['pupil'](left_eye, right_eye)
    out["pupil_pred"] = y_pupil.detach()

    y_vitals=models['vitals'](x_vitals)
    out["vitals_pred"] = y_vitals.detach()

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
