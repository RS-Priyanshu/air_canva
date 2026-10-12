"""
Real-Time Air Canvas & Gesture Drawing Assistant (MediaPipe Tasks API)
======================================================================

Gestures
--------
  Index only              -> DRAW   (paint a continuous stroke)
  Index + Middle          -> MOVE   (hover cursor, no stroke)
  Thumb only / pinch      -> COLOR_CHANGE (cycle brush color, debounced)
  All five fingers open   -> CLEAR  (wipe the canvas, flash message)
  Pinky only              -> ERASE  (paint in black with a wider brush)

Run
---
    pip install -r requirements.txt
    python air_canvas.py

Press 'q' or ESC to quit, 'c' to clear manually, 's' to save snapshot.
"""

import math
import os
import time
from collections import deque

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

CAM_WIDTH, CAM_HEIGHT = 1280, 720

COLOR_PALETTE = [
    ("Yellow", (0, 255, 255)),
    ("Green",  (0, 255, 0)),
    ("Red",    (0, 0, 255)),
    ("White",  (255, 255, 255)),
    ("Cyan",   (255, 255, 0)),
]

BRUSH_THICKNESS  = 6
ERASER_THICKNESS = 40
POINTER_RADIUS   = 10

SMOOTHING_ALPHA      = 0.45
SMOOTHING_BUFFER_LEN = 5

COLOR_DEBOUNCE_FRAMES = 15
CLEAR_DEBOUNCE_FRAMES = 20
CLEAR_FLASH_FRAMES    = 25

PINCH_DISTANCE_THRESHOLD = 0.045

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "hand_landmarker.task")


# --------------------------------------------------------------------------- #
# Hand connections (the new API doesn't ship drawing utils, so we define them)
# --------------------------------------------------------------------------- #

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (0, 17),                                  # palm base
]


# --------------------------------------------------------------------------- #
# Finger / gesture classification
# --------------------------------------------------------------------------- #

def get_finger_status(landmarks):
    """
    `landmarks` is a list of 21 objects with .x, .y (normalized 0..1).
    Returns dict[str, bool] with keys: thumb, index, middle, ring, pinky.
    """
    lm = landmarks

    thumb_tip, thumb_ip, pinky_mcp = lm[4], lm[3], lm[17]
    dist_tip = math.hypot(thumb_tip.x - pinky_mcp.x, thumb_tip.y - pinky_mcp.y)
    dist_ip  = math.hypot(thumb_ip.x  - pinky_mcp.x, thumb_ip.y  - pinky_mcp.y)

    fingers = {"thumb": dist_tip > dist_ip * 1.05}

    for name, tip_id, pip_id in (
        ("index",  8,  6),
        ("middle", 12, 10),
        ("ring",   16, 14),
        ("pinky",  20, 18),
    ):
        fingers[name] = lm[tip_id].y < lm[pip_id].y

    return fingers


def is_pinch(landmarks, threshold=PINCH_DISTANCE_THRESHOLD):
    """True when thumb tip (4) and index tip (8) are close together."""
    a, b = landmarks[4], landmarks[8]
    return math.hypot(a.x - b.x, a.y - b.y) < threshold


def classify_gesture(fingers, pinch_active):
    thumb  = fingers["thumb"]
    index  = fingers["index"]
    middle = fingers["middle"]
    ring   = fingers["ring"]
    pinky  = fingers["pinky"]

    if thumb and index and middle and ring and pinky:
        return "CLEAR"

    if pinch_active:
        return "COLOR_CHANGE"

    if index and middle and not ring and not pinky and not thumb:
        return "MOVE"

    if index and not middle and not ring and not pinky and not thumb:
        return "DRAW"

    if thumb and not index and not middle and not ring and not pinky:
        return "COLOR_CHANGE"

    if pinky and not thumb and not index and not middle and not ring:
        return "ERASE"

    return "IDLE"


# --------------------------------------------------------------------------- #
# Coordinate smoothing
# --------------------------------------------------------------------------- #

def smooth_coordinates(curr_x, curr_y, prev_x, prev_y, alpha=SMOOTHING_ALPHA):
    if prev_x is None or prev_y is None:
        return curr_x, curr_y
    smooth_x = int(alpha * curr_x + (1 - alpha) * prev_x)
    smooth_y = int(alpha * curr_y + (1 - alpha) * prev_y)
    return smooth_x, smooth_y


class PointBuffer:
    def __init__(self, maxlen=SMOOTHING_BUFFER_LEN):
        self.xs = deque(maxlen=maxlen)
        self.ys = deque(maxlen=maxlen)

    def push_and_average(self, x, y):
        self.xs.append(x)
        self.ys.append(y)
        return int(sum(self.xs) / len(self.xs)), int(sum(self.ys) / len(self.ys))

    def clear(self):
        self.xs.clear()
        self.ys.clear()


# --------------------------------------------------------------------------- #
# Hand drawing (since mp.solutions.drawing_utils was removed)
# --------------------------------------------------------------------------- #

def draw_hand_landmarks(frame, landmarks, color=(0, 255, 0)):
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], color, 2, cv2.LINE_AA)

    for (px, py) in pts:
        cv2.circle(frame, (px, py), 3, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, (px, py), 3, (0, 0, 0), 1, cv2.LINE_AA)


# --------------------------------------------------------------------------- #
# HUD rendering
# --------------------------------------------------------------------------- #

def draw_hud(frame, current_mode, current_color_name, current_color_bgr,
             flash_active=False):
    h, w = frame.shape[:2]
    bar_height = 90

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_height), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, dst=frame)

    cv2.putText(frame, f"Mode: {current_mode}", (20, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

    legend = ("Open Palm = Clear | Thumb/Pinch = Color | "
              "Index+Middle = Move | Index = Draw | Pinky = Erase")
    cv2.putText(frame, legend, (20, 64),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 220, 180), 1, cv2.LINE_AA)

    swatch_size = 36
    sx, sy = w - swatch_size - 20, 15
    cv2.rectangle(frame, (sx, sy), (sx + swatch_size, sy + swatch_size),
                  current_color_bgr, -1)
    cv2.rectangle(frame, (sx, sy), (sx + swatch_size, sy + swatch_size),
                  (255, 255, 255), 2)
    cv2.putText(frame, current_color_name,
                (sx - 10 - 8 * len(current_color_name), sy + 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    if flash_active:
        text = "Canvas Cleared!"
        (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.4, 3)
        tx, ty = (w - tw) // 2, h // 2
        cv2.putText(frame, text, (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3, cv2.LINE_AA)


# --------------------------------------------------------------------------- #
# Canvas / frame blending
# --------------------------------------------------------------------------- #

def blend_canvas_on_frame(frame, canvas):
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, inv_mask = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY_INV)
    inv_mask_bgr = cv2.cvtColor(inv_mask, cv2.COLOR_GRAY2BGR)
    frame = cv2.bitwise_and(frame, inv_mask_bgr)
    frame = cv2.bitwise_or(frame, canvas)
    return frame


# --------------------------------------------------------------------------- #
# Main application loop
# --------------------------------------------------------------------------- #

def main():
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(
            f"Missing model file: {MODEL_PATH}\n"
            "Download it from:\n"
            "  https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
            "hand_landmarker/float16/latest/hand_landmarker.task\n"
            "and place it next to this script."
        )

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

    if not cap.isOpened():
        raise RuntimeError(
            "Could not open webcam. Check that a camera is connected and accessible."
        )

    # ----- New Tasks API setup -----
    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    detector = vision.HandLandmarker.create_from_options(options)

    canvas = None
    point_buffer = PointBuffer()

    prev_x, prev_y = None, None
    color_index = 0
    current_mode = "IDLE"

    color_cooldown = 0
    clear_cooldown = 0
    clear_flash_counter = 0

    prev_time = time.time()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Warning: failed to read frame from webcam.")
                break

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            if canvas is None:
                canvas = np.zeros_like(frame)

            # MediaPipe Tasks expects an RGB mp.Image
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            timestamp_ms = int(time.time() * 1000)
            result = detector.detect_for_video(mp_image, timestamp_ms)

            # Cooldowns
            if color_cooldown > 0:      color_cooldown -= 1
            if clear_cooldown > 0:      clear_cooldown -= 1
            if clear_flash_counter > 0: clear_flash_counter -= 1

            color_name, color_bgr = COLOR_PALETTE[color_index]

            if result.hand_landmarks:
                landmarks = result.hand_landmarks[0]   # list of 21 NormalizedLandmark

                draw_hand_landmarks(frame, landmarks)

                fingers = get_finger_status(landmarks)
                pinch_active = is_pinch(landmarks)
                current_mode = classify_gesture(fingers, pinch_active)

                index_tip  = landmarks[8]
                middle_tip = landmarks[12]

                if current_mode == "MOVE":
                    cx = int((index_tip.x + middle_tip.x) / 2 * w)
                    cy = int((index_tip.y + middle_tip.y) / 2 * h)
                else:
                    cx = int(index_tip.x * w)
                    cy = int(index_tip.y * h)

                cx, cy = smooth_coordinates(cx, cy, prev_x, prev_y)
                cx, cy = point_buffer.push_and_average(cx, cy)

                if current_mode == "DRAW":
                    if prev_x is not None and prev_y is not None:
                        cv2.line(canvas, (prev_x, prev_y), (cx, cy),
                                 color_bgr, BRUSH_THICKNESS)
                    prev_x, prev_y = cx, cy
                    cv2.circle(frame, (cx, cy), POINTER_RADIUS, color_bgr, -1)

                elif current_mode == "ERASE":
                    if prev_x is not None and prev_y is not None:
                        cv2.line(canvas, (prev_x, prev_y), (cx, cy),
                                 (0, 0, 0), ERASER_THICKNESS)
                    prev_x, prev_y = cx, cy
                    cv2.circle(frame, (cx, cy), ERASER_THICKNESS // 2,
                               (0, 0, 0), 2)

                elif current_mode == "MOVE":
                    prev_x, prev_y = None, None
                    point_buffer.clear()
                    cv2.circle(frame, (cx, cy), POINTER_RADIUS,
                               (255, 255, 255), 2)

                elif current_mode == "COLOR_CHANGE":
                    prev_x, prev_y = None, None
                    point_buffer.clear()
                    if color_cooldown == 0:
                        color_index = (color_index + 1) % len(COLOR_PALETTE)
                        color_cooldown = COLOR_DEBOUNCE_FRAMES
                    color_name, color_bgr = COLOR_PALETTE[color_index]
                    cv2.circle(frame, (cx, cy), POINTER_RADIUS, color_bgr, 2)

                elif current_mode == "CLEAR":
                    prev_x, prev_y = None, None
                    point_buffer.clear()
                    if clear_cooldown == 0:
                        canvas = np.zeros_like(frame)
                        clear_cooldown = CLEAR_DEBOUNCE_FRAMES
                        clear_flash_counter = CLEAR_FLASH_FRAMES

                else:  # IDLE
                    prev_x, prev_y = None, None
                    point_buffer.clear()

            else:
                current_mode = "IDLE"
                prev_x, prev_y = None, None
                point_buffer.clear()

            frame = blend_canvas_on_frame(frame, canvas)
            draw_hud(frame, current_mode, color_name, color_bgr,
                     flash_active=clear_flash_counter > 0)

            now = time.time()
            fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now
            cv2.putText(frame, f"FPS: {fps:0.1f}", (w - 150, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200),
                        1, cv2.LINE_AA)

            cv2.imshow("Air Canvas - Gesture Drawing", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("c"):
                canvas = np.zeros_like(frame)
            elif key == ord("s"):
                filename = f"air_canvas_{int(time.time())}.png"
                cv2.imwrite(filename, canvas)
                print(f"Saved canvas snapshot to {filename}")

    finally:
        detector.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()