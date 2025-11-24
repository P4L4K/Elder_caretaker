"""
Real-time emotion and mood detection using OpenCV and DeepFace.

This module provides:
- get_emotion_from_frame(frame): Reusable function to analyze a single frame and return the dominant emotion and confidence.
- webcam_emotion_loop(...): Real-time webcam loop that overlays detected emotions on the video feed.

Notes:
- DeepFace must be installed in the environment. Install via: `pip install deepface opencv-python`.
- The code resizes frames for analysis and processes every Nth frame to reduce lag.
"""

from __future__ import annotations

import time
from typing import Dict, Optional, Tuple

import cv2
from deepface import DeepFace


# -----------------------------
# Configuration defaults
# -----------------------------
EMOTION_LABELS = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "sad",
    "surprise", 
    "neutral",
]


def _normalize_deepface_output(analysis) -> Tuple[Optional[str], Optional[float], Optional[Dict[str, float]], Optional[Dict]]:
    """
    Normalize DeepFace.analyze output across versions.

    Returns a tuple of (dominant_emotion, dominant_conf, emotions_dict, raw_dict)
    where emotions_dict maps emotion -> probability (0-100).
    """
    if analysis is None:
        return None, None, None, None

    # DeepFace may return a list of results if enforce_detection=True and multiple faces are found.
    # We take the first face's analysis by default.
    if isinstance(analysis, list) and len(analysis) > 0:
        analysis = analysis[0]

    if not isinstance(analysis, dict):
        return None, None, None, None

    emotions_dict = analysis.get("emotion") or analysis.get("emotions")
    dom = analysis.get("dominant_emotion")

    dom_conf = None
    if isinstance(emotions_dict, dict) and dom in emotions_dict:
        try:
            dom_conf = float(emotions_dict[dom])
        except Exception:
            dom_conf = None

    return dom, dom_conf, emotions_dict, analysis


def get_emotion_from_frame(
    frame,
    *,
    detector_backend: str = "opencv",
    enforce_detection: bool = False,
    align: bool = True,
    target_width: int = 640,
) -> Tuple[Optional[str], Optional[float], Optional[Dict[str, float]]]:
    """
    Analyze a single BGR frame (as from OpenCV) with DeepFace to detect emotions.

    Parameters:
    - frame: BGR image (numpy array) from cv2.VideoCapture.
    - detector_backend: DeepFace detector backend (e.g., 'opencv', 'mtcnn', 'retinaface').
    - enforce_detection: If True, raises error when no face is detected; we set False to skip gracefully.
    - align: Whether to align the face prior to analysis.
    - target_width: Resize width for faster inference; height is scaled accordingly.

    Returns:
    - (dominant_emotion, confidence_percent, emotions_dict)
      confidence_percent is 0-100 (float). Returns (None, None, None) if no face found or on error.
    """
    if frame is None or frame.size == 0:
        return None, None, None

    # Resize frame for faster analysis while preserving aspect ratio
    h, w = frame.shape[:2]
    if w > target_width:
        scale = target_width / float(w)
        resized = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    else:
        resized = frame

    try:
        analysis = DeepFace.analyze(
            img_path=resized,
            actions=["emotion"],
            detector_backend=detector_backend,
            enforce_detection=enforce_detection,
            align=align,
        )
    except Exception:
        # Gracefully skip frames where detection fails
        return None, None, None

    dominant, dom_conf, emotions_dict, _ = _normalize_deepface_output(analysis)
    if dominant is None:
        return None, None, None

    return dominant, dom_conf, emotions_dict


def _put_text(
    img,
    text: str,
    org: Tuple[int, int],
    color: Tuple[int, int, int] = (255, 255, 255),
    scale: float = 0.6,
    thickness: int = 2,
    bg: Optional[Tuple[int, int, int]] = (0, 0, 0),
):
    """
    Draw readable text with an optional background box.
    """
    if bg is not None:
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        x, y = org
        cv2.rectangle(img, (x - 2, y - th - 6), (x + tw + 2, y + baseline + 2), bg, -1)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _draw_emotions_overlay(
    frame,
    emotions_dict: Optional[Dict[str, float]],
    dominant: Optional[str],
    dom_conf: Optional[float],
    *,
    start_x: int = 10,
    start_y: int = 30,
    line_h: int = 24,
):
    """
    Overlay emotions and confidences on the frame.
    """
    if emotions_dict is None:
        _put_text(frame, "No face detected", (start_x, start_y), color=(0, 0, 255))
        return

    y = start_y
    for label in EMOTION_LABELS:
        val = emotions_dict.get(label)
        if val is None:
            continue
        is_dom = (label == dominant)
        color = (0, 255, 0) if is_dom else (255, 255, 255)
        text = f"{label}: {val:.1f}%"
        _put_text(frame, text, (start_x, y), color=color, bg=(0, 0, 0))
        y += line_h

    if dominant is not None and dom_conf is not None:
        _put_text(frame, f"Dominant: {dominant} ({dom_conf:.1f}%)", (start_x, y + 6), color=(0, 255, 0))


def webcam_emotion_loop(
    *,
    camera_index: int = 0,
    detection_interval: int = 3,
    target_width: int = 640,
    window_name: str = "Emotion Recognition",
    detector_backend: str = "opencv",
    enforce_detection: bool = False,
    align: bool = True,
    display: bool = True,
) -> Tuple[Optional[str], Optional[float]]:
    """
    Run real-time emotion detection from the webcam, overlay labels, and optionally display the feed.

    Processing efficiency:
    - Only analyze every `detection_interval`-th frame.
    - Resize frames for analysis using `target_width`.

    Returns the last detected (dominant_emotion, confidence_percent) when the loop exits.
    Press 'q' or ESC to quit when display=True.
    """
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Check camera_index or device permissions.")

    frame_count = 0
    last_dominant: Optional[str] = None
    last_conf: Optional[float] = None
    last_emotions: Optional[Dict[str, float]] = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                # If reading fails, wait briefly and try again
                time.sleep(0.01)
                continue

            # Run analysis only every N frames to reduce lag
            if frame_count % detection_interval == 0:
                dom, conf, emotions = get_emotion_from_frame(
                    frame,
                    detector_backend=detector_backend,
                    enforce_detection=enforce_detection,
                    align=align,
                    target_width=target_width,
                )
                if dom is not None:
                    last_dominant, last_conf, last_emotions = dom, conf, emotions

            # Draw overlay using the last available result (even if this frame wasn't analyzed)
            if display:
                _draw_emotions_overlay(frame, last_emotions, last_dominant, last_conf)
                cv2.imshow(window_name, frame)

                # Quit on 'q' or ESC
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    break

            frame_count += 1

            # If not displaying (headless), break after one successful read-analyze cycle to return a result
            if not display and frame_count >= detection_interval:
                break

    finally:
        cap.release()
        if display:
            cv2.destroyAllWindows()

    return last_dominant, last_conf


if __name__ == "__main__":
    """
    Standalone testing entry point.
    Opens the default webcam, runs real-time emotion detection, and prints the last detected emotion.
    Press 'q' or ESC to exit.
    """
    dom, conf = webcam_emotion_loop(
        camera_index=0,
        detection_interval=3,
        target_width=640,
        detector_backend="opencv",
        enforce_detection=False,
        align=True,
        display=True,
    )
    if dom is None:
        print("No emotion detected in the last analyzed frame.")
    else:
        print(f"Last detected emotion: {dom} ({conf:.1f}%)")

