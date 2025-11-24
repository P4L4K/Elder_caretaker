from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional

import cv2
import numpy as np
from PIL import Image
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import sqlite3
import uuid
import os

# Try to import DeepFace; allow backend to start even if it's missing,
# and handle the error per-request.
try:
    from deepface import DeepFace  # type: ignore
except ModuleNotFoundError:
    DeepFace = None  # type: ignore

# Use a DB file inside backend folder
DB_PATH = os.path.join(os.path.dirname(__file__), "..", "emotion_users.db")
DB_PATH = os.path.abspath(DB_PATH)


@dataclass
class DetectionResult:
    user_id: Optional[str]
    dominant_emotion: Optional[str]
    confidence: Optional[float]
    emotions: Dict[str, float]
    face_location: Optional[Dict]
    timestamp: str


class EnhancedEmotionAnalyzer:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._init_database()

    def _init_database(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    reference_photo TEXT NOT NULL,
                    face_embedding BLOB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS emotion_logs (
                    log_id TEXT PRIMARY KEY,
                    user_id TEXT,
                    emotion TEXT,
                    confidence REAL,
                    emotions_json TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (user_id)
                )
                """
            )

    def register_user(self, name: str, email: str, reference_photo_base64: str) -> str:
        user_id = str(uuid.uuid4())
        try:
            image_data = base64.b64decode(
                reference_photo_base64.split(",")[1] if "," in reference_photo_base64 else reference_photo_base64
            )
            image = Image.open(io.BytesIO(image_data)).convert("RGB")
            frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

            embedding = self._extract_face_embedding(frame)
            if embedding is None:
                raise ValueError("No face detected in reference photo")

            embedding_bytes = embedding.tobytes()
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "INSERT INTO users (user_id, name, email, reference_photo, face_embedding) VALUES (?, ?, ?, ?, ?)",
                    (user_id, name, email, reference_photo_base64, embedding_bytes),
                )

            return user_id
        except Exception as e:
            raise ValueError(f"User registration failed: {str(e)}")

    def _extract_face_embedding(self, frame) -> Optional[np.ndarray]:
        if DeepFace is None:
            # DeepFace not available; no embedding can be computed
            return None
        try:
            analysis = DeepFace.represent(
                img_path=frame,
                model_name="VGG-Face",
                enforce_detection=True,
                detector_backend="opencv",
            )
            if isinstance(analysis, list) and len(analysis) > 0:
                return np.array(analysis[0]["embedding"], dtype=np.float64)
        except Exception:
            pass
        return None

    def identify_user(self, frame) -> Optional[str]:
        """Identify user from frame using cosine similarity.

        Uses a relaxed threshold (0.6) so matching is practical in
        real environments with lighting / pose changes, and supports
        multiple users.
        """
        try:
            current_embedding = self._extract_face_embedding(frame)
            if current_embedding is None:
                return None

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT user_id, face_embedding FROM users")
                users = cursor.fetchall()

            if not users:
                return None

            best_match_id: Optional[str] = None
            best_similarity = 0.4  # more relaxed threshold
            for user_id, embedding_bytes in users:
                if not embedding_bytes:
                    continue
                stored_embedding = np.frombuffer(embedding_bytes, dtype=np.float64)
                denom = np.linalg.norm(current_embedding) * np.linalg.norm(stored_embedding)
                if denom == 0:
                    continue
                similarity = float(np.dot(current_embedding, stored_embedding) / denom)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match_id = user_id
            return best_match_id
        except Exception:
            return None

    def _normalize_deepface_output(self, analysis):
        if analysis is None:
            return None, None, None, None
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

    def analyze_emotion(self, frame, user_id: Optional[str] = None) -> DetectionResult:
        if DeepFace is None:
            # Without DeepFace we cannot analyze emotions
            return DetectionResult(None, None, None, {}, None, datetime.now().isoformat())
        try:
            h, w = frame.shape[:2]
            if w > 640:
                scale = 640.0 / float(w)
                resized = cv2.resize(frame, (int(w * scale), int(h * scale)))
            else:
                resized = frame

            if user_id is None:
                user_id = self.identify_user(resized)
            if user_id is None:
                return DetectionResult(None, None, None, {}, None, datetime.now().isoformat())

            analysis = DeepFace.analyze(
                img_path=resized,
                actions=["emotion"],
                detector_backend="opencv",
                enforce_detection=True,
                align=True,
            )

            dominant, dom_conf, emotions_dict, raw_analysis = self._normalize_deepface_output(analysis)

            if dominant is not None:
                self._log_emotion(user_id, dominant, dom_conf or 0.0, emotions_dict or {})

            return DetectionResult(
                user_id=user_id,
                dominant_emotion=dominant,
                confidence=dom_conf,
                emotions=emotions_dict or {},
                face_location=raw_analysis.get("region") if raw_analysis else None,
                timestamp=datetime.now().isoformat(),
            )
        except Exception:
            return DetectionResult(None, None, None, {}, None, datetime.now().isoformat())

    def _log_emotion(self, user_id: Optional[str], emotion: str, confidence: float, emotions: Dict):
        try:
            log_id = str(uuid.uuid4())
            emotions_json = str(emotions)
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "INSERT INTO emotion_logs (log_id, user_id, emotion, confidence, emotions_json) VALUES (?, ?, ?, ?, ?)",
                    (log_id, user_id, emotion, confidence, emotions_json),
                )
        except Exception:
            pass


# ---------- FastAPI Router ----------
router = APIRouter(prefix="/emotion", tags=["emotion"])
_analyzer = EnhancedEmotionAnalyzer()


class RegisterRequest(BaseModel):
    name: str
    email: str
    reference_photo: str  # base64


class AnalyzeRequest(BaseModel):
    image: str
    user_id: Optional[str] = None


@router.post("/register")
async def register_user(req: RegisterRequest):
    try:
        user_id = _analyzer.register_user(req.name, req.email, req.reference_photo)
        return {"user_id": user_id, "message": "User registered successfully"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        raise HTTPException(status_code=500, detail="Registration failed")


@router.post("/analyze")
async def analyze_emotion(req: AnalyzeRequest):
    try:
        image_bytes = base64.b64decode(req.image.split(",")[1] if "," in req.image else req.image)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        result = _analyzer.analyze_emotion(frame, req.user_id)

        # Normalize numpy types to plain Python for JSON serialization
        def _to_float(val):
            try:
                return float(val) if val is not None else None
            except Exception:
                return None

        emotions_clean: Dict[str, float] = {}
        if result.emotions:
            for k, v in result.emotions.items():
                try:
                    emotions_clean[str(k)] = float(v)
                except Exception:
                    continue

        if result.user_id is None or result.dominant_emotion is None:
            return {
                "user_id": None,
                "dominant_emotion": None,
                "confidence": None,
                "emotions": emotions_clean,
                "face_location": None,
                "timestamp": result.timestamp,
            }

        return {
            "user_id": result.user_id,
            "dominant_emotion": result.dominant_emotion,
            "confidence": _to_float(result.confidence),
            "emotions": emotions_clean,
            "face_location": result.face_location,
            "timestamp": result.timestamp,
        }
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=500, detail="Analysis failed")


@router.get("/history/{user_id}")
async def get_emotion_history(user_id: str):
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.execute(
                "SELECT emotion, confidence, timestamp FROM emotion_logs WHERE user_id = ? ORDER BY timestamp DESC LIMIT 100",
                (user_id,),
            )
            history = cursor.fetchall()
        return [
            {"emotion": row[0], "confidence": row[1], "timestamp": row[2]}
            for row in history
        ]
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to fetch history")
