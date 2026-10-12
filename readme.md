# Air Canva — Real-Time Hand Gesture Drawing

Air Canva is a computer vision application built with **OpenCV** and **Google MediaPipe** that allows users to draw virtually on a digital canvas using hand gestures captured via webcam.

---

## ✨ Features

- **Real-Time Hand Tracking:** Fast, low-latency landmark tracking using MediaPipe's Hand Landmarker task.
- **Air Drawing:** Draw smoothly on the canvas using your index finger.
- **Gesture Shortcuts:** Toggle colors, clear canvas, or switch to eraser mode using customizable finger gestures.
- **Virtual Color Palette:** Visual UI overlays for selecting brush colors and stroke widths.

---

## 📁 Project Structure

```text
opencv/
│
├── aircanva.py              # Main application script
├── hand_landmarker.task     # MediaPipe hand landmark detection model
├── requirements.txt         # Project dependencies
├── .gitignore               # Ignored files (virtual environments, cache)
└── README.md                # Project documentation
```

---

## ⚙️ Prerequisites

- **Python 3.9 – 3.11** (recommended for MediaPipe compatibility)
- Working webcam
- Git (optional, for version control)

---

## 🚀 Quick Start Guide

### 1. Clone or Download the Repository

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

### 2. Set Up a Virtual Environment

Isolate your dependencies to avoid conflicts with global system packages:

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

> *Tip:* If you encounter a script execution policy error on PowerShell, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` and try activating again.

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

### 3. Install Dependencies

Install all required Python packages:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### Typical `requirements.txt` contents:
```text
opencv-python>=4.8.0
mediapipe>=0.10.0
numpy>=1.24.0
```

---

### 4. Download the Model File

The project requires Google's pre-trained `hand_landmarker.task` file in the project root:

**Windows (PowerShell):**
```powershell
curl.exe -L -o hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
```

**macOS / Linux:**
```bash
curl -L -o hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
```

---

### 5. Run the Application

Run the script using the activated environment:

```bash
python aircanva.py
```

Press **`q`** or **`Esc`** in the application window to quit.

---

## ✋ Controls & Gestures

| Gesture / Key | Action |
| :--- | :--- |
| **Index Finger Up** | Drawing mode (draws on canvas at fingertip position) |
| **Index & Middle Fingers Up** | Selection mode (hover over palette to pick colors / eraser) |
| **Fist / Closed Hand** | Neutral / pause drawing |
| **`q` / `Esc`** | Exit application |
| **`c`** | Clear canvas (if keyboard shortcut enabled) |

---

## 🛠️ Troubleshooting

- **`FileNotFoundError: Missing model file: .../hand_landmarker.task`**:
  Make sure `hand_landmarker.task` is in the same directory as `aircanva.py`, not inside `.venv`.
- **Camera does not open (`cv2.VideoCapture` failed)**:
  Ensure no other application (Zoom, Teams, Camera app) is using your webcam. If you have an external webcam, try changing `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)` inside `aircanva.py`.
- **Low FPS / Stuttering**:
  Lower the frame capture resolution inside `aircanva.py` to `640x480` for better frame rates on low-spec hardware.

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.