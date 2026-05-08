# 🎯 Emotional Scanner

**Двухъязычная версия | Bilingual version:**  
🇷🇺 Русская версия ниже • 🇬🇧 English version below

---

## 🇬🇧 Project Overview

**Emotional Scanner** is a computer vision system based on deep learning that analyzes human emotional state from video.  
The system extracts visual signals (facial micro-expressions, gaze direction, head movement) and predicts emotional state.

### ✅ Key Features
- Emotion recognition from video
- Face detection and analysis
- PyTorch-based neural models
- Modular architecture
- Extendable dataset pipeline
- Team-ready structure (Git-based workflow)

---

## 🛠️ Tech Stack
| Category | Technologies |
|-----------|-------------|
| Language  | Python |
| Framework | PyTorch |
| CV Tools  | OpenCV |
| Utils     | NumPy, Pandas, tqdm |
| Project   | Git, GitHub, GitHub Actions |

---

## 🗂️ Project Structure
```
emotional-scanner/
│
├── src/                          # Main source code
│   ├── __init__.py              # Package initialization
│   ├── main.py                   # Application entry point
│   ├── config/                   # Configuration management
│   │   ├── __init__.py
│   │   └── settings.py           # Centralized configuration
│   ├── core/                     # Core functionality
│   │   ├── __init__.py
│   │   ├── model_loader.py       # Model loading and management
│   │   └── face_detector.py      # Face detection utilities
│   ├── services/                 # Service layer
│   │   ├── __init__.py
│   │   ├── video_processor.py    # Video processing service
│   │   ├── audio_processor.py    # Audio processing service
│   │   └── emotion_analyzer.py   # Emotion analysis service
│   ├── utils/                    # Utility functions
│   │   ├── __init__.py
│   │   ├── preprocessing.py      # Data preprocessing
│   │   ├── visualization.py      # Visualization utilities
│   │   └── helpers.py            # Helper functions
│   └── models/                   # ML model definitions
│       ├── __init__.py
│       ├── classes.py            # Model classes
│       └── load_models.py        # Model loading utilities
│
├── models/                       # Trained model files
├── tests/                        # Unit tests
├── docs/                         # Documentation
├── website/                      # GitHub Pages (project site)
├── .github/                      # GitHub workflow and PR templates
├── setup.py                      # Package setup configuration
├── requirements.txt              # Dependencies
├── .env                          # Environment variables
└── README.md                     # This file
```

---

## 👥 Team
| Name | Role |
|------|------|
| **Илья Злотников** | Team Lead |
| **Саллохидин Нурмахомедов** | Developer |
| **Дмитрий Филипов** | Developer |

---

---

## 🇷🇺 Описание проекта

**Emotional Scanner** — это система компьютерного зрения на основе нейронных сетей, которая анализирует эмоциональное состояние человека по видео.

### ✅ Возможности
- Распознавание эмоций по лицу
- Анализ мимики и микродвижений
- Поддержка видеопотока
- Архитектура для обучения моделей

---

## 🚀 Installation and Usage

### Prerequisites
- Python 3.8 or higher
- CUDA-capable GPU (recommended for better performance)
- Webcam for video input
- Microphone for audio input (optional)

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/emotional-scanner.git
cd emotional-scanner

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

### Running the Application

```bash
# Run the GUI application
python src/main.py

# Or using the installed package
emotional-scanner
```

### Configuration

Create a `.env` file in the root directory:

```env
# Device configuration
CUDA_VISIBLE_DEVICES=0

# Model settings
MODEL_DEVICE=cuda
INPUT_SIZE=320,320
SCORE_THRESHOLD=0.6

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/emotional_scanner.log
```

### Development Setup

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Code formatting
black src/
flake8 src/

# Type checking
mypy src/
```
