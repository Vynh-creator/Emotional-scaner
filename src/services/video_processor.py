import sys
import time
import asyncio
from threading import Thread, Lock

import cv2
import numpy as np
import sounddevice as sd
import torch

from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout,
    QPushButton, QLabel, QSpinBox, QHBoxLayout, QTextEdit
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap

from qasync import QEventLoop

from pathlib import Path

from src.utils.preprocessing_data import preprocess_data, deepseek_chat, get_predictions_arrays
from src.models.load_models import load_all


ROOT = Path(__file__).resolve().parents[1]
PATHS = {
    "drowsy": str(ROOT / "models" / "best_model_drowsy.pth"),
    "emotion_cnn": str(ROOT / "models" / "best_emotion_cnn.pth"),
    "audio": str(ROOT / "models" / "best_model_audio.pth"),
    "involvement": str(ROOT / "models" / "best_model_involvement.pth"),
    "pupil": str(ROOT / "models" / "best_model_pupil.pth"),
    "vitals": str(ROOT / "models" / "best_model_vitals.pth"),
}


def _fmt_tensor(t, max_items=30, prec=4):
    if t is None:
        return "None"
    arr = t.detach().cpu().flatten().tolist()[:max_items]
    return "[" + ", ".join(f"{x:.{prec}f}" for x in arr) + ("]" if len(arr) < max_items else ", ...]")


class DeepSeekWorker:
    def __init__(self, log_fn):
        self.queue = asyncio.Queue(maxsize=10)
        self._task = None
        self._log = log_fn
        self._stopping = False

    async def start(self):
        self._task = asyncio.create_task(self._run())

    async def stop(self):
        self._stopping = True
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except Exception:
                pass

    async def submit(self, prompt: str, tag: str):
        if self.queue.full():
            self._log(f"[{time.strftime('%H:%M:%S')}] DeepSeek queue full, drop: {tag}")
            return
        await self.queue.put((prompt, tag))

    async def _run(self):
        while not self._stopping:
            prompt, tag = await self.queue.get()
            try:
                answer = await asyncio.to_thread(deepseek_chat, prompt)
                self._log(f"[{time.strftime('%H:%M:%S')}] DeepSeek({tag}): {answer}")
            except Exception as e:
                self._log(f"[{time.strftime('%H:%M:%S')}] DeepSeek({tag}) ERROR: {type(e).__name__}: {e}")


class Pipeline:
    def __init__(self, device, w2v_model, processor, models, detector):
        self.device = device
        self.w2v_model = w2v_model
        self.processor = processor
        self.models = models
        self.detector = detector
        self.busy_lock = Lock()

    def infer_preds(self, video_frames, audio_blocks, sample_rate):
        x = preprocess_data(
            video_frames, audio_blocks,
            self.device, sample_rate,
            self.processor, self.w2v_model, self.detector
        )
        preds = get_predictions_arrays(self.models, *x)
        return preds

    def make_prompt(self, preds):
        return (
            "Сгенерируй короткое описание состояния человека по признакам.\n"
            "1) Сонливость (logits/score): " + _fmt_tensor(preds.get("drowsy_logits")) + "\n"
            "2) Вовлечённость (logits): " + _fmt_tensor(preds.get("involvement_logits")) + "\n"
            "3) Эмоции по видео (probs 7 классов): " + _fmt_tensor(preds.get("emotion_probs")) + "\n"
            "   Классы: 1=Angry,2=Disgust,3=Fear,4=Happy,5=Sad,6=Surprise,7=Neutral\n"
            "4) Эмоции по аудио (logits 8 классов): " + _fmt_tensor(preds.get("audio_logits")) + "\n"
            "   Классы: 01=neutral,02=calm,03=happy,04=sad,05=angry,06=fearful,07=disgust,08=surprised\n"
            "5) Зрачок (предсказания по 8 кадрам): " + _fmt_tensor(preds.get("pupil_pred")) + "\n"
            "6) Пульс по видео: " + _fmt_tensor(preds.get("vitals_pred")) + "\n"
            "Ответ: 3-6 предложений, без списков."
        )


class VideoRecorder(QWidget):
    log_signal = pyqtSignal(str)

    def __init__(self, device, w2v_model, processor, models, detector, aio_loop, camera_idx=0, sample_rate=44100):
        super().__init__()
        self.camera_idx = camera_idx
        self.sample_rate = sample_rate

        self.device = device
        self.w2v_model = w2v_model
        self.processor = processor
        self.models = models
        self.detector = detector

        self.aio_loop = aio_loop

        self.cap = None
        self.audio_stream = None

        self.video_buffer = []
        self.audio_buffer = []
        self.last_saved_video_chunk = []
        self.last_saved_audio_chunk = []

        self.is_recording = False
        self.buffer_lock = Lock()

        self.pipeline = Pipeline(self.device, self.w2v_model, self.processor, self.models, self.detector)

        self.init_ui()
        self.init_camera()

        self.log_signal.connect(self.ai_log.append)

        self.frame_timer = QTimer(self)
        self.frame_timer.timeout.connect(self.update_frame)
        self.frame_timer.start(30)

        self.chunk_timer = QTimer(self)
        self.chunk_timer.timeout.connect(self.on_chunk_tick)

        self.deepseek = DeepSeekWorker(log_fn=lambda s: self.log_signal.emit(s))

    def init_camera(self):
        self.cap = cv2.VideoCapture(self.camera_idx)
        if not self.cap.isOpened():
            self.preview_label.setText("❌ Ошибка: Камера не найдена")

    def init_ui(self):
        self.setWindowTitle("Emotion Analysis & Recorder")
        self.setMinimumWidth(700)
        layout = QVBoxLayout()

        self.preview_label = QLabel("Ожидание потока...")
        self.preview_label.setFixedSize(640, 480)
        self.preview_label.setStyleSheet("background: black; border: 2px solid #333;")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.preview_label)

        self.info_label = QLabel("Статус: Готов")
        layout.addWidget(self.info_label)

        layout.addWidget(QLabel("Анализ признаков:"))
        self.ai_log = QTextEdit()
        self.ai_log.setReadOnly(True)
        self.ai_log.setMaximumHeight(240)
        self.ai_log.setStyleSheet("background: #1e1e1e; color: #00ff00; font-family: Consolas;")
        layout.addWidget(self.ai_log)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Размер чанка (сек):"))
        self.interval_spin = QSpinBox()
        self.interval_spin.setRange(1, 10)
        self.interval_spin.setValue(3)
        controls.addWidget(self.interval_spin)
        layout.addLayout(controls)

        self.btn_toggle = QPushButton("🔴 Начать запись")
        self.btn_toggle.clicked.connect(self.toggle_recording)
        layout.addWidget(self.btn_toggle)

        self.btn_play = QPushButton("▶️ Просмотреть последний фрагмент")
        self.btn_play.clicked.connect(self.play_last_chunk)
        layout.addWidget(self.btn_play)

        self.setLayout(layout)

    def update_frame(self):
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                if self.is_recording:
                    with self.buffer_lock:
                        self.video_buffer.append(frame.copy())
                self.update_ui_preview(frame)

    def update_ui_preview(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        img = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        self.preview_label.setPixmap(
            QPixmap.fromImage(img).scaled(640, 480, Qt.AspectRatioMode.KeepAspectRatio)
        )

    def audio_callback(self, indata, frames, time_info, status):
        if status:
            print(f"Audio status: {status}")
        if self.is_recording:
            with self.buffer_lock:
                self.audio_buffer.append(indata.copy())

    def toggle_recording(self):
        if not self.is_recording:
            self.start_recording()
        else:
            self.stop_recording()

    def start_recording(self):
        self.is_recording = True
        with self.buffer_lock:
            self.video_buffer = []
            self.audio_buffer = []

        try:
            self.audio_stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=2,
                dtype="float32",
                blocksize=4096,
                latency="high",
                callback=self.audio_callback,
            )
            self.audio_stream.start()
            audio_status = "✓"
        except Exception as e:
            print(f"Audio error: {e}")
            audio_status = "✗"

        self.btn_toggle.setText("⏹️ Остановить")
        self.info_label.setText(f"Запись идет... (видео ✓ | аудио {audio_status})")
        self.log_signal.emit(f"[{time.strftime('%H:%M:%S')}] Начата запись")

        self.chunk_timer.start(self.interval_spin.value() * 1000)

    def stop_recording(self):
        self.is_recording = False

        if self.chunk_timer.isActive():
            self.chunk_timer.stop()

        if self.audio_stream:
            self.audio_stream.stop()
            self.audio_stream.close()
            self.audio_stream = None

        self.btn_toggle.setText("🔴 Начать запись")
        self.info_label.setText("Запись остановлена")
        self.log_signal.emit(f"[{time.strftime('%H:%M:%S')}] Запись остановлена")

    def snapshot_buffers(self, clear=False):
        with self.buffer_lock:
            v = list(self.video_buffer)
            a = list(self.audio_buffer)
            if clear:
                self.video_buffer = []
                self.audio_buffer = []
        return v, a

    def on_chunk_tick(self):
        if not self.is_recording:
            return

        interval_s = self.interval_spin.value()

        live_v, live_a = self.snapshot_buffers(clear=False)
        live_v = live_v[-int(max(1, interval_s * 30)):]
        live_a = live_a[-int(max(1, interval_s * 50)):]

        rec_v, rec_a = self.snapshot_buffers(clear=True)

        print(f"[{time.strftime('%H:%M:%S')}] CHUNK tick={interval_s}s | live: v={len(live_v)} a={len(live_a)} | rec: v={len(rec_v)} a={len(rec_a)}")

        if rec_v:
            self.last_saved_video_chunk = rec_v
            self.last_saved_audio_chunk = rec_a

        if len(live_v) >= 8:
            Thread(target=self._infer_and_send, args=("live", live_v, live_a), daemon=True).start()

        if len(rec_v) >= 8:
            Thread(target=self._infer_and_send, args=("rec", rec_v, rec_a), daemon=True).start()

    def _infer_and_send(self, tag, video_frames, audio_blocks):
        if not self.pipeline.busy_lock.acquire(blocking=False):
            return
        try:
            preds = self.pipeline.infer_preds(video_frames, audio_blocks, self.sample_rate)
            prompt = self.pipeline.make_prompt(preds)

            self.log_signal.emit(
                f"[{time.strftime('%H:%M:%S')}] {tag} preds ok | "
                f"drowsy={_fmt_tensor(preds.get('drowsy_logits'), 5)} "
                f"involv={_fmt_tensor(preds.get('involvement_logits'), 5)}"
            )

            asyncio.run_coroutine_threadsafe(
                self.deepseek.submit(prompt, tag),
                self.aio_loop
            )
        finally:
            self.pipeline.busy_lock.release()

    def play_last_chunk(self):
        if not self.last_saved_video_chunk:
            self.info_label.setText("❌ Сначала запишите фрагмент!")
            return
        self.info_label.setText("▶️ Воспроизведение...")
        Thread(target=self._play_video_and_audio, daemon=True).start()

    def _play_video_and_audio(self):
        audio_thread = None
        if self.last_saved_audio_chunk:
            audio_thread = Thread(target=self._play_audio, daemon=True)
            audio_thread.start()

        cv2.namedWindow("Playback", cv2.WINDOW_NORMAL)
        for frame in self.last_saved_video_chunk:
            cv2.imshow("Playback", frame)
            if cv2.waitKey(30) & 0xFF == ord("q"):
                break
        cv2.destroyWindow("Playback")

        if audio_thread:
            audio_thread.join()

    def _play_audio(self):
        if not self.last_saved_audio_chunk:
            return
        audio_data = np.concatenate(self.last_saved_audio_chunk, axis=0)
        sd.play(audio_data, self.sample_rate)
        sd.wait()

    def closeEvent(self, event):
        self.is_recording = False
        if self.chunk_timer:
            self.chunk_timer.stop()
        if self.audio_stream:
            self.audio_stream.stop()
            self.audio_stream.close()
        if self.cap and self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
        event.accept()


def get_working_camera_index():
    for idx in range(3):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            if ret and frame is not None:
                return idx
    return 0


async def run_app(app: QApplication):
    camera_idx = get_working_camera_index()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    w2v_model, processor, models_out, detector = load_all(device, PATHS)

    loop = asyncio.get_running_loop()

    window = VideoRecorder(
        device=device,
        w2v_model=w2v_model,
        processor=processor,
        models=models_out,
        detector=detector,
        aio_loop=loop,
        camera_idx=camera_idx,
        sample_rate=44100,
    )
    window.show()

    await window.deepseek.start()

    app_close_event = asyncio.Event()
    app.aboutToQuit.connect(app_close_event.set)
    await app_close_event.wait()

    await window.deepseek.stop()


def start():
    app = QApplication(sys.argv)
    loop = QEventLoop(app)
    asyncio.set_event_loop(loop)
    with loop:
        loop.run_until_complete(run_app(app))
