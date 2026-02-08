import sys
import cv2
import asyncio
import time
import numpy as np
from PyQt6.QtWidgets import (QApplication, QWidget, QVBoxLayout,
                             QPushButton, QLabel, QSpinBox, QHBoxLayout, QTextEdit)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap
from concurrent.futures import ThreadPoolExecutor


class AsyncVideoRecorder(QWidget):
    def __init__(self, camera_idx=0):
        super().__init__()
        self.camera_idx = camera_idx
        self.cap = cv2.VideoCapture(self.camera_idx, cv2.CAP_DSHOW)

        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.video_buffer = []
        self.last_saved_chunk = []
        self.is_recording = False
        self.executor = ThreadPoolExecutor(max_workers=2)

        self.init_ui()
        asyncio.create_task(self.frame_loop())

    def init_ui(self):
        self.setWindowTitle("Emotion Analysis & Recorder")
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
        self.btn_play.clicked.connect(lambda: asyncio.create_task(self.play_last_chunk()))
        layout.addWidget(self.btn_play)

        self.setLayout(layout)

    async def frame_loop(self):
        loop = asyncio.get_running_loop()
        while True:
            ret, frame = await loop.run_in_executor(self.executor, self.cap.read)

            if ret:
                if self.is_recording:
                    self.video_buffer.append(frame.copy())
                self.update_ui_preview(frame)
            else:
                self.preview_label.setText("Ошибка: Камера не найдена или занята")
                await asyncio.sleep(1)

            await asyncio.sleep(0.01)

    def update_ui_preview(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        img = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        self.preview_label.setPixmap(QPixmap.fromImage(img).scaled(640, 480, Qt.AspectRatioMode.KeepAspectRatio))

    def toggle_recording(self):
        if not self.is_recording:
            self.is_recording = True
            self.video_buffer = []
            self.btn_toggle.setText("⏹️ Остановить и сохранить")
            self.info_label.setText("Запись идет...")
            asyncio.create_task(self.buffer_manager())
        else:
            self.is_recording = False
            self.btn_toggle.setText("🔴 Начать запись")
            self.info_label.setText("Запись остановлена")

    async def buffer_manager(self):
        while self.is_recording:
            await asyncio.sleep(self.interval_spin.value())
            if self.is_recording and self.video_buffer:
                self.last_saved_chunk = list(self.video_buffer)
                self.video_buffer = []
                self.ai_log.append(f"[{time.strftime('%H:%M:%S')}] Фрагмент сохранен для анализа.")

    async def play_last_chunk(self):
        if not self.last_saved_chunk:
            self.info_label.setText("Ошибка: Сначала запишите фрагмент!")
            return

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self.executor, self._show_opencv_window)

    def _show_opencv_window(self):
        cv2.namedWindow("Playback", cv2.WINDOW_NORMAL)
        for frame in self.last_saved_chunk:
            cv2.imshow("Playback", frame)
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
        cv2.destroyWindow("Playback")

    def closeEvent(self, event):
        self.is_recording = False
        if self.cap.isOpened():
            self.cap.release()
        event.accept()


async def main():
    app = QApplication(sys.argv)

    window = AsyncVideoRecorder(get_working_camera_index())
    window.show()

    while True:
        app.processEvents()
        await asyncio.sleep(0.005)


def get_working_camera_index():
    for idx in range(5):
        cap = cv2.VideoCapture(idx, cv2.CAP_MSMF)

        if cap.isOpened():
            non_black = 0
            for _ in range(5):
                ret, frame = cap.read()
                if ret and frame is not None:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
                    if np.mean(gray) > 20:
                        non_black += 1

            cap.release()

            if non_black >= 3:
                print(f"Найдена камера {idx} (MSMF)")
                return idx

    for idx in range(5):
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)

        if cap.isOpened():
            non_black = 0
            for _ in range(5):
                ret, frame = cap.read()
                if ret and frame is not None:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
                    if np.mean(gray) > 20:
                        non_black += 1

            cap.release()

            if non_black >= 3:
                print(f"Найдена камера {idx} (DSHOW)")
                return idx

    for idx in range(5):
        cap = cv2.VideoCapture(idx)

        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            if ret:
                print(f"Найдена камера {idx} (авто)")
                return idx

    return 0


try:
    asyncio.run(main())
except KeyboardInterrupt:
    pass
except RuntimeError as e:
    if "Event loop is closed" not in str(e):
        raise