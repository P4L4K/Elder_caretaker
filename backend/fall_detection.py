import cv2
import numpy as np
import time
from typing import Dict, Optional, Tuple

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None  # Will be checked at runtime


def _angle_deg(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    # Angle of vector p1->p2 relative to vertical axis (90=vertical, 0=horizontal)
    vx, vy = (p2[0] - p1[0], p2[1] - p1[1])
    norm = max(1e-6, (vx * vx + vy * vy) ** 0.5)
    # angle vs horizontal
    import math
    ang_h = abs(math.degrees(math.atan2(vy, vx)))  # 0 = horizontal
    # convert to vertical-based angle
    return 90.0 - min(ang_h, 180 - ang_h)


class FallDetector:
    def __init__(
        self,
        *,
        model_name: str = "yolov8n-pose.pt",
        conf: float = 0.25,
        angle_fall_threshold: float = 30.0,  # torso angle to vertical below this means horizontal
        speed_threshold: float = 40.0,       # downward hip speed px/sec
        cooldown_seconds: float = 3.0,
    ):
        if YOLO is None:
            raise RuntimeError("ultralytics is not installed. Please `pip install ultralytics`.")
        self.model = YOLO(model_name)
        self.conf = conf
        self.angle_fall_threshold = angle_fall_threshold
        self.speed_threshold = speed_threshold
        self.cooldown_seconds = cooldown_seconds

        self._prev_angle: Optional[float] = None
        self._prev_hip_y: Optional[float] = None
        self._prev_time: Optional[float] = None
        self._cooldown_until: float = 0.0

    def _get_keypoints(self, frame: np.ndarray) -> Optional[np.ndarray]:
        results = self.model.predict(frame, conf=self.conf, imgsz=640, verbose=False)
        if not results or results[0].keypoints is None:
            return None
        kps = results[0].keypoints.xy
        if kps is None or len(kps) == 0:
            return None
        return kps[0].cpu().numpy()  # (num_kpts, 2)

    def detect_fall(self, frame: np.ndarray) -> Dict[str, object]:
        ts = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        now = time.time()
        if frame is None or frame.size == 0:
            return {"fall_detected": False, "timestamp": ts}

        fall = False
        angle = None
        hip_y = None

        try:
            kps = self._get_keypoints(frame)
            if kps is not None and kps.shape[0] >= 12:
                # COCO format indices used by YOLOv8-Pose
                L_SHO = 5
                R_SHO = 6
                L_HIP = 11
                R_HIP = 12
                shoulder = ((kps[L_SHO][0] + kps[R_SHO][0]) / 2.0, (kps[L_SHO][1] + kps[R_SHO][1]) / 2.0)
                hip = ((kps[L_HIP][0] + kps[R_HIP][0]) / 2.0, (kps[L_HIP][1] + kps[R_HIP][1]) / 2.0)
                angle = _angle_deg(hip, shoulder)  # 90=vertical, 0=horizontal
                hip_y = hip[1]
        except Exception:
            angle = None
            hip_y = None

        if angle is not None and hip_y is not None:
            speed = 0.0
            if self._prev_time is not None and self._prev_hip_y is not None:
                dt = max(1e-3, now - self._prev_time)
                speed = (hip_y - self._prev_hip_y) / dt  # +ve means moving downward in image coords

            prev_angle = self._prev_angle if self._prev_angle is not None else 90.0
            standing_before = prev_angle > 45.0
            now_horizontal = angle < self.angle_fall_threshold

            if now >= self._cooldown_until and standing_before and now_horizontal and speed > self.speed_threshold:
                fall = True
                self._cooldown_until = now + self.cooldown_seconds

            # Debug to tune
            try:
                print(f"[PoseFallDebug] angle={angle:.1f} prev_angle={prev_angle:.1f} speed={speed:.1f}")
            except Exception:
                pass

            self._prev_angle = angle
            self._prev_hip_y = hip_y
            self._prev_time = now

        return {"fall_detected": fall, "timestamp": ts}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default="", help="Path to a video file. Leave empty to use camera.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index when not using --video")
    parser.add_argument("--frame-skip", type=int, default=2, help="Analyze every Nth frame")
    parser.add_argument("--display", action="store_true", help="Show live preview window")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video if args.video else args.camera)
    if not cap.isOpened():
        raise SystemExit("Could not open input source")

    det = FallDetector()
    i = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            if i % max(1, args.frame_skip) == 0:
                res = det.detect_fall(frame)
                if res.get("fall_detected"):
                    print(res, flush=True)
            if args.display:
                cv2.imshow("YOLOv8-Pose Fall Detection", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    break
            i += 1
    finally:
        cap.release()
        if args.display:
            cv2.destroyAllWindows()
