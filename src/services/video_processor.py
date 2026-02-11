import sys
import cv2
import time
import numpy as np
import sounddevice as sd
import asyncio
from PyQt6.QtWidgets import (QApplication, QWidget, QVBoxLayout,
                             QPushButton, QLabel, QSpinBox, QHBoxLayout, QTextEdit)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap
from threading import Thread, Lock


class VideoRecorder(QWidget):
    def __init__(self, camera_idx=0, sample_rate=44100):
        super().__init__()
        self.camera_idx = camera_idx
        self.sample_rate = sample_rate
        self.cap = None
        
        self.video_buffer = []
        self.audio_buffer = []
        self.last_saved_video_chunk = []
        self.last_saved_audio_chunk = []
        self.is_recording = False
        self.buffer_lock = Lock()
        
        self.audio_stream = None
        self.chunk_timer = None

        self.init_ui()
        self.init_camera()
        
        # Таймер для обновления кадров
        self.frame_timer = QTimer()
        self.frame_timer.timeout.connect(self.update_frame)
        self.frame_timer.start(30)  # ~30 FPS

    def init_camera(self):
        """Инициализация камеры"""
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

        layout.addWidget(QLabel("Анализ признаков (DeepSeek):"))
        self.ai_log = QTextEdit()
        self.ai_log.setReadOnly(True)
        self.ai_log.setMaximumHeight(150)
        self.ai_log.setPlaceholderText("Здесь появятся результаты анализа...")
        self.ai_log.setStyleSheet("background: #1e1e1e; color: #00ff00; font-family: Consolas;")
        layout.addWidget(self.ai_log)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Интервал захвата (сек):"))
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
        """Обновление кадра в UI"""
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                if self.is_recording:
                    with self.buffer_lock:
                        self.video_buffer.append(frame.copy())
                
                self.update_ui_preview(frame)

    def audio_callback(self, indata, frames, time_info, status):
        """Callback для захвата аудио"""
        if status:
            print(f"Audio status: {status}")
        if self.is_recording:
            with self.buffer_lock:
                self.audio_buffer.append(indata.copy())

    def update_ui_preview(self, frame):
        """Отображение кадра в превью"""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        img = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        self.preview_label.setPixmap(
            QPixmap.fromImage(img).scaled(640, 480, Qt.AspectRatioMode.KeepAspectRatio)
        )

    def toggle_recording(self):
        """Переключение записи"""
        if not self.is_recording:
            self.start_recording()
        else:
            self.stop_recording()

    def start_recording(self):
        """Начало записи"""
        self.is_recording = True
        with self.buffer_lock:
            self.video_buffer = []
            self.audio_buffer = []
        
        # Запуск аудио потока
        try:
            self.audio_stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=2,
                callback=self.audio_callback
            )
            self.audio_stream.start()
            audio_status = "✓"
        except Exception as e:
            print(f"Audio error: {e}")
            audio_status = "✗"
        
        self.btn_toggle.setText("⏹️ Остановить и сохранить")
        self.info_label.setText(f"Запись идет... (видео ✓ | аудио {audio_status})")
        self.ai_log.append(f"[{time.strftime('%H:%M:%S')}] Начата запись")
        
        # Запуск таймера для сохранения чанков
        self.chunk_timer = QTimer()
        self.chunk_timer.timeout.connect(self.save_chunk)
        self.chunk_timer.start(self.interval_spin.value() * 1000)

    def stop_recording(self):
        """Остановка записи"""
        self.is_recording = False
        
        if self.chunk_timer:
            self.chunk_timer.stop()
            self.chunk_timer = None
        
        if self.audio_stream:
            self.audio_stream.stop()
            self.audio_stream.close()
            self.audio_stream = None
        
        self.btn_toggle.setText("🔴 Начать запись")
        self.info_label.setText("Запись остановлена")
        self.ai_log.append(f"[{time.strftime('%H:%M:%S')}] Запись остановлена")

    def save_chunk(self):
        """Сохранение текущего чанка"""
        if not self.is_recording:
            return
        
        with self.buffer_lock:
            if self.video_buffer:
                self.last_saved_video_chunk = list(self.video_buffer)
                self.last_saved_audio_chunk = list(self.audio_buffer)
                
                video_frames = len(self.last_saved_video_chunk)
                audio_samples = sum(len(chunk) for chunk in self.last_saved_audio_chunk)
                
                self.video_buffer = []
                self.audio_buffer = []
                
                self.ai_log.append(
                    f"[{time.strftime('%H:%M:%S')}] Фрагмент сохранен: "
                    f"{video_frames} кадров, {audio_samples} аудио семплов"
                )

    def play_last_chunk(self):
        """Воспроизведение последнего фрагмента"""
        if not self.last_saved_video_chunk:
            self.info_label.setText("❌ Сначала запишите фрагмент!")
            return
        
        self.info_label.setText("▶️ Воспроизведение...")
        
        # Запуск в отдельном потоке
        Thread(target=self._play_video_and_audio, daemon=True).start()

    def _play_video_and_audio(self):
        """Воспроизведение видео и аудио"""
        # Запуск аудио в фоне
        audio_thread = None
        if self.last_saved_audio_chunk:
            audio_thread = Thread(target=self._play_audio, daemon=True)
            audio_thread.start()
        
        # Показ видео
        cv2.namedWindow("Playback", cv2.WINDOW_NORMAL)
        for frame in self.last_saved_video_chunk:
            cv2.imshow("Playback", frame)
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
        cv2.destroyWindow("Playback")
        
        if audio_thread:
            audio_thread.join()

    def _play_audio(self):
        """Воспроизведение аудио"""
        if self.last_saved_audio_chunk:
            audio_data = np.concatenate(self.last_saved_audio_chunk, axis=0)
            sd.play(audio_data, self.sample_rate)
            sd.wait()

    def closeEvent(self, event):
        """Закрытие приложения"""
        self.is_recording = False
        
        if self.frame_timer:
            self.frame_timer.stop()
        
        if self.audio_stream:
            self.audio_stream.stop()
            self.audio_stream.close()
        
        if self.cap and self.cap.isOpened():
            self.cap.release()
        
        cv2.destroyAllWindows()
        event.accept()


def get_working_camera_index():
    """Поиск рабочей камеры"""
    for idx in range(3):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            if ret and frame is not None:
                print(f"✓ Найдена камера: {idx}")
                return idx
    
    print("⚠ Камера не найдена, используется индекс 0")
    return 0


async def main():
    app = QApplication(sys.argv)
    
    camera_idx = get_working_camera_index()
    window = VideoRecorder(camera_idx)
    window.show()
    
    # Асинхронный цикл событий Qt
    while True:
        app.processEvents()
        await asyncio.sleep(0.005)


try:
    asyncio.run(main())
except KeyboardInterrupt:
    pass
except RuntimeError as e:
    if "Event loop is closed" not in str(e):
        raise
