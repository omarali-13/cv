"""
cv_engine.py — Real-Time Fitness Tracking CV Module  (v4)
=========================================================
- Auto-detects which body side is visible & tracks only that side
- Uses 2D angles (x,y) for side-view accuracy (z from MediaPipe is noisy)
- Rich contextual voice coaching (specific form corrections)
- Automatic rep counting with real-time feedback

Dependencies:
    pip install mediapipe opencv-python numpy fastdtw scipy pyttsx3
"""

from __future__ import annotations

import logging
import os
import queue
import sys
import threading
import time
import urllib.request
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    PoseLandmarker,
    PoseLandmarkerOptions,
    RunningMode,
)

# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger("cv_engine")

# TTS
try:
    import pyttsx3
    _HAS_TTS = True
except ImportError:
    _HAS_TTS = False

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
_MODEL_DIR = Path(__file__).parent / "models"
_MODEL_PATH = _MODEL_DIR / "pose_landmarker_full.task"
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_full/float16/latest/"
    "pose_landmarker_full.task"
)

# ---------------------------------------------------------------------------
# Landmark indices
# ---------------------------------------------------------------------------
LM = {
    "L_SHOULDER": 11, "R_SHOULDER": 12,
    "L_ELBOW":    13, "R_ELBOW":    14,
    "L_WRIST":    15, "R_WRIST":    16,
    "L_HIP":      23, "R_HIP":      24,
    "L_KNEE":     25, "R_KNEE":     26,
    "L_ANKLE":    27, "R_ANKLE":    28,
    "L_FOOT":     31, "R_FOOT":     32,
}

_POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (15, 17), (15, 19), (15, 21), (16, 18), (16, 20), (16, 22),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (24, 26), (26, 28),
    (27, 29), (27, 31), (29, 31), (28, 30), (28, 32), (30, 32),
]


def _ensure_model() -> str:
    if _MODEL_PATH.exists():
        return str(_MODEL_PATH)
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n    [*] Downloading pose model (first time only)...")
    try:
        urllib.request.urlretrieve(_MODEL_URL, str(_MODEL_PATH))
        print("    [OK] Done.\n")
    except Exception as e:
        print(f"    [ERROR] {e}")
        sys.exit(1)
    return str(_MODEL_PATH)


# ===================================================================
# Audio Feedback
# ===================================================================
class AudioFeedback:
    """Non-blocking voice coach running in a background thread."""

    def __init__(self, enabled: bool = True):
        self._enabled = enabled
        self._queue: queue.Queue = queue.Queue()
        self._cooldowns: dict[str, float] = {}
        self._engine = None
        if enabled:
            t = threading.Thread(target=self._worker, daemon=True)
            t.start()

    def say(self, text: str, cooldown: float = 3.0, key: str = "") -> None:
        """
        Speak text with a per-key cooldown to avoid spam.
        'key' groups related messages (e.g. 'elbow_swing').
        """
        if not self._enabled:
            return
        k = key or text
        now = time.time()
        if now - self._cooldowns.get(k, 0) < cooldown:
            return
        self._cooldowns[k] = now
        # Drop old if backed up
        while self._queue.qsize() > 2:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        self._queue.put(text)

    def stop(self) -> None:
        """Stop any current speech and clear queue immediately."""
        if not self._enabled:
            return
        # Clear the queue
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        # Call stop on the pyttsx3 engine
        if self._engine is not None:
            try:
                self._engine.stop()
            except Exception:
                pass

    def shutdown(self):
        self.stop()
        self._queue.put(None)

    def _worker(self):
        if _HAS_TTS:
            try:
                # Initialize COM for the background thread on Windows
                import pythoncom
                pythoncom.CoInitialize()
            except Exception:
                pass

            try:
                engine = pyttsx3.init()
                engine.setProperty("rate", 170)
                voices = engine.getProperty("voices")
                for v in voices:
                    if "zira" in v.name.lower() or "female" in v.name.lower():
                        engine.setProperty("voice", v.id)
                        break
                self._engine = engine
            except Exception:
                self._worker_beep()
                return

            while True:
                try:
                    text = self._queue.get(timeout=1)
                    if text is None:
                        break
                    engine.say(text)
                    engine.runAndWait()
                except queue.Empty:
                    pass
                except Exception:
                    pass

            try:
                pythoncom.CoUninitialize()
            except Exception:
                pass
        else:
            self._worker_beep()

    def _worker_beep(self):
        try:
            import winsound
        except ImportError:
            return
        while True:
            try:
                text = self._queue.get(timeout=1)
                if text is None:
                    break
                winsound.Beep(800, 150)
            except queue.Empty:
                pass


# ===================================================================
# Data
# ===================================================================
class ExerciseType(Enum):
    BICEP_CURL = "bicep_curl"
    SQUAT = "squat"
    LUNGE = "lunge"
    PUSH_UP = "push_up"
    SUPERMAN = "superman"
    SHOULDER_PRESS = "shoulder_press"
    DIPS = "dips"
    PLANK = "plank"


@dataclass
class FormEvaluation:
    dtw_distance: float
    form_score: float
    is_correct: bool
    threshold: float
    feedback: str


@dataclass
class RepetitionState:
    angle_sequence: list = field(default_factory=list)
    rep_count: int = 0
    phase: str = "idle"
    prev_angle: float = 0.0
    min_angle: float = 180.0
    max_angle: float = 0.0
    shoulder_angles: list = field(default_factory=list)  # Track shoulder stability


# ===================================================================
# PoseAnalyzer
# ===================================================================
class PoseAnalyzer:
    """
    Main class: auto side-detection, 2D angles, contextual coaching.
    """

    # ---------- Per-side joint definitions ----------
    _JOINTS = {
        ExerciseType.BICEP_CURL: {
            "left": {
                "elbow":    (LM["L_SHOULDER"], LM["L_ELBOW"], LM["L_WRIST"]),
                "shoulder": (LM["L_HIP"],      LM["L_SHOULDER"], LM["L_ELBOW"]),
            },
            "right": {
                "elbow":    (LM["R_SHOULDER"], LM["R_ELBOW"], LM["R_WRIST"]),
                "shoulder": (LM["R_HIP"],      LM["R_SHOULDER"], LM["R_ELBOW"]),
            },
        },
        ExerciseType.SQUAT: {
            "left": {
                "hip":   (LM["L_SHOULDER"], LM["L_HIP"],  LM["L_KNEE"]),
                "knee":  (LM["L_HIP"],      LM["L_KNEE"], LM["L_ANKLE"]),
                "ankle": (LM["L_KNEE"],     LM["L_ANKLE"], LM["L_FOOT"]),
            },
            "right": {
                "hip":   (LM["R_SHOULDER"], LM["R_HIP"],  LM["R_KNEE"]),
                "knee":  (LM["R_HIP"],      LM["R_KNEE"], LM["R_ANKLE"]),
                "ankle": (LM["R_KNEE"],     LM["R_ANKLE"], LM["R_FOOT"]),
            },
        },
        ExerciseType.LUNGE: {
            "left": {
                "hip":   (LM["L_SHOULDER"], LM["L_HIP"],  LM["L_KNEE"]),
                "knee":  (LM["L_HIP"],      LM["L_KNEE"], LM["L_ANKLE"]),
                "ankle": (LM["L_KNEE"],     LM["L_ANKLE"], LM["L_FOOT"]),
            },
            "right": {
                "hip":   (LM["R_SHOULDER"], LM["R_HIP"],  LM["R_KNEE"]),
                "knee":  (LM["R_HIP"],      LM["R_KNEE"], LM["R_ANKLE"]),
                "ankle": (LM["R_KNEE"],     LM["R_ANKLE"], LM["R_FOOT"]),
            },
        },
        ExerciseType.PUSH_UP: {
            "left": {
                "elbow":    (LM["L_SHOULDER"], LM["L_ELBOW"], LM["L_WRIST"]),
                "shoulder": (LM["L_HIP"],      LM["L_SHOULDER"], LM["L_ELBOW"]),
                "hip":      (LM["L_SHOULDER"], LM["L_HIP"],  LM["L_KNEE"]),
            },
            "right": {
                "elbow":    (LM["R_SHOULDER"], LM["R_ELBOW"], LM["R_WRIST"]),
                "shoulder": (LM["R_HIP"],      LM["R_SHOULDER"], LM["R_ELBOW"]),
                "hip":      (LM["R_SHOULDER"], LM["R_HIP"],  LM["R_KNEE"]),
            },
        },
        ExerciseType.SUPERMAN: {
            "left": {
                "hip":      (LM["L_SHOULDER"], LM["L_HIP"],  LM["L_KNEE"]),
                "shoulder": (LM["L_ELBOW"],    LM["L_SHOULDER"], LM["L_HIP"]),
            },
            "right": {
                "hip":      (LM["R_SHOULDER"], LM["R_HIP"],  LM["R_KNEE"]),
                "shoulder": (LM["R_ELBOW"],    LM["R_SHOULDER"], LM["R_HIP"]),
            },
        },
        ExerciseType.SHOULDER_PRESS: {
            "left": {
                "elbow":    (LM["L_SHOULDER"], LM["L_ELBOW"], LM["L_WRIST"]),
                "shoulder": (LM["L_HIP"],      LM["L_SHOULDER"], LM["L_ELBOW"]),
            },
            "right": {
                "elbow":    (LM["R_SHOULDER"], LM["R_ELBOW"], LM["R_WRIST"]),
                "shoulder": (LM["R_HIP"],      LM["R_SHOULDER"], LM["R_ELBOW"]),
            },
        },
        ExerciseType.DIPS: {
            "left": {
                "elbow":    (LM["L_SHOULDER"], LM["L_ELBOW"], LM["L_WRIST"]),
                "shoulder": (LM["L_HIP"],      LM["L_SHOULDER"], LM["L_ELBOW"]),
                "hip":      (LM["L_SHOULDER"], LM["L_HIP"],  LM["L_KNEE"]),
            },
            "right": {
                "elbow":    (LM["R_SHOULDER"], LM["R_ELBOW"], LM["R_WRIST"]),
                "shoulder": (LM["R_HIP"],      LM["R_SHOULDER"], LM["R_ELBOW"]),
                "hip":      (LM["R_SHOULDER"], LM["R_HIP"],  LM["R_KNEE"]),
            },
        },
        ExerciseType.PLANK: {
            "left": {
                "hip":      (LM["L_SHOULDER"], LM["L_HIP"],  LM["L_KNEE"]),
                "shoulder": (LM["L_HIP"],      LM["L_SHOULDER"], LM["L_ELBOW"]),
                "knee":     (LM["L_HIP"],      LM["L_KNEE"], LM["L_ANKLE"]),
            },
            "right": {
                "hip":      (LM["R_SHOULDER"], LM["R_HIP"],  LM["R_KNEE"]),
                "shoulder": (LM["R_HIP"],      LM["R_SHOULDER"], LM["R_ELBOW"]),
                "knee":     (LM["R_HIP"],      LM["R_KNEE"], LM["R_ANKLE"]),
            },
        },
    }

    _REP_CONFIG = {
        ExerciseType.BICEP_CURL: {
            "primary": "elbow",
            "start": 140,
            "bottom": 60,
        },
        ExerciseType.SQUAT: {
            "primary": "knee",
            "start": 155,
            "bottom": 110,
        },
        ExerciseType.LUNGE: {
            "primary": "knee",
            "start": 155,
            "bottom": 110,
        },
        ExerciseType.PUSH_UP: {
            "primary": "elbow",
            "start": 150,
            "bottom": 95,
        },
        ExerciseType.SUPERMAN: {
            "primary": "hip",
            "start": 175,
            "bottom": 166,
        },
        ExerciseType.SHOULDER_PRESS: {
            "primary": "elbow",
            "start": 140,
            "bottom": 100,
        },
        ExerciseType.DIPS: {
            "primary": "elbow",
            "start": 150,
            "bottom": 95,
        },
        ExerciseType.PLANK: {
            "primary": "hip",
            "start": 165,
            "bottom": 150,
        },
    }

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        dtw_threshold: float = 25.0,
        enable_audio: bool = True,
    ):
        self._det_conf = min_detection_confidence
        self._trk_conf = min_tracking_confidence
        self.dtw_threshold = dtw_threshold
        self._landmarker: Optional[PoseLandmarker] = None
        self.audio = AudioFeedback(enabled=enable_audio)

    # ------------------------------------------------------------------
    def _get_landmarker(self) -> PoseLandmarker:
        if self._landmarker is None:
            opts = PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=_ensure_model()),
                running_mode=RunningMode.VIDEO,
                num_poses=1,
                min_pose_detection_confidence=self._det_conf,
                min_tracking_confidence=self._trk_conf,
                output_segmentation_masks=False,
            )
            self._landmarker = PoseLandmarker.create_from_options(opts)
        return self._landmarker

    def release(self):
        if self._landmarker:
            self._landmarker.close()
            self._landmarker = None
        self.audio.shutdown()

    # ------------------------------------------------------------------
    # Landmark extraction
    # ------------------------------------------------------------------
    def extract_landmarks(self, frame: np.ndarray, ts_ms: int) -> Optional[list]:
        if frame is None or frame.size == 0:
            return None
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self._get_landmarker().detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts_ms
        )
        if not result.pose_landmarks:
            return None
        return result.pose_landmarks[0]

    # ------------------------------------------------------------------
    # 2D angle (more accurate from side view)
    # ------------------------------------------------------------------
    @staticmethod
    def calc_angle_2d(a, b, c) -> float:
        """
        Angle at vertex b using ONLY x,y coordinates.

        Why 2D instead of 3D?
        MediaPipe's z-coordinate is an ESTIMATE based on the 2D image.
        It's noisy and unreliable. When standing sideways (which we
        instruct the user to do), all the movement happens in the
        camera's x-y plane, so x,y gives the MOST accurate angle.

        Formula: cos(theta) = (BA . BC) / (|BA| * |BC|)
        """
        ba = np.array([a[0] - b[0], a[1] - b[1]])
        bc = np.array([c[0] - b[0], c[1] - b[1]])
        n1 = np.linalg.norm(ba)
        n2 = np.linalg.norm(bc)
        if n1 < 1e-9 or n2 < 1e-9:
            return 0.0
        cos = np.clip(np.dot(ba, bc) / (n1 * n2), -1.0, 1.0)
        return float(np.degrees(np.arccos(cos)))

    @staticmethod
    def calculate_3d_angle(a, b, c) -> float:
        """
        Calculate 3D angle at vertex b using x, y, z coordinates.
        Formula: cos(theta) = (BA . BC) / (|BA| * |BC|)
        """
        ba = a - b
        bc = c - b
        n1 = np.linalg.norm(ba)
        n2 = np.linalg.norm(bc)
        if n1 < 1e-9 or n2 < 1e-9:
            return 0.0
        cos = np.clip(np.dot(ba, bc) / (n1 * n2), -1.0, 1.0)
        return float(np.degrees(np.arccos(cos)))

    # ------------------------------------------------------------------
    # Side detection
    # ------------------------------------------------------------------
    def detect_active_side(
        self, landmarks: list, exercise: ExerciseType
    ) -> str:
        """
        Determine which side is facing the camera using key shoulder and hip landmarks.
        Uses a combination of landmark visibility and Z depth (closer/more negative is better).
        """
        l_sh = landmarks[LM["L_SHOULDER"]]
        r_sh = landmarks[LM["R_SHOULDER"]]
        l_hip = landmarks[LM["L_HIP"]]
        r_hip = landmarks[LM["R_HIP"]]

        # Visibility score (higher is better)
        l_vis = (l_sh.visibility + l_hip.visibility) / 2.0
        r_vis = (r_sh.visibility + r_hip.visibility) / 2.0

        # Depth score (smaller/more negative is closer to camera, which is better)
        l_depth = (l_sh.z + l_hip.z) / 2.0
        r_depth = (r_sh.z + r_hip.z) / 2.0

        # Combined score: Vis - Depth (since negative depth means closer, subtracting it increases the score)
        l_score = l_vis - l_depth
        r_score = r_vis - r_depth

        return "left" if l_score >= r_score else "right"

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------
    def extract_features(
        self, landmarks: list, exercise: ExerciseType, side: str
    ) -> Optional[dict[str, float]]:
        """Extract 2D angles for the given side."""
        jmap = self._JOINTS.get(exercise, {}).get(side)
        if not jmap:
            return None

        features = {}
        for name, (ia, ib, ic) in jmap.items():
            la, lb, lc = landmarks[ia], landmarks[ib], landmarks[ic]
            # Accept lower visibility since we're only using x,y
            if min(la.visibility, lb.visibility, lc.visibility) < 0.15:
                return None
            a = [la.x, la.y]
            b = [lb.x, lb.y]
            c = [lc.x, lc.y]
            features[name] = self.calc_angle_2d(a, b, c)

        return features

    def extract_side_features_3d(
        self, landmarks: list, exercise_type: ExerciseType, side: str
    ) -> Optional[dict[str, float]]:
        """Extract 3D angles for a specific exercise and side."""
        jmap = self._JOINTS[exercise_type].get(side)
        if not jmap:
            return None

        features = {}
        for name, (ia, ib, ic) in jmap.items():
            la, lb, lc = landmarks[ia], landmarks[ib], landmarks[ic]
            if min(la.visibility, lb.visibility, lc.visibility) < 0.15:
                return None
            a = np.array([la.x, la.y, la.z])
            b = np.array([lb.x, lb.y, lb.z])
            c = np.array([lc.x, lc.y, lc.z])
            features[name] = self.calculate_3d_angle(a, b, c)

        return features

    # ------------------------------------------------------------------
    # Orientation
    # ------------------------------------------------------------------
    @staticmethod
    def detect_orientation(landmarks: list) -> str:
        spread = abs(landmarks[LM["L_SHOULDER"]].x - landmarks[LM["R_SHOULDER"]].x)
        return "front" if spread > 0.15 else "side"

    # ------------------------------------------------------------------
    # FORM COACHING — the interactive feedback brain
    # ------------------------------------------------------------------
    def coach_bicep_curl(
        self, landmarks: list, angles: dict[str, float],
        side: str, rep_state: RepetitionState,
    ) -> list[str]:
        """
        Analyse current form and return specific correction messages.
        """
        tips: list[str] = []

        elbow_angle = angles.get("elbow", 90)
        shoulder_angle = angles.get("shoulder", 20)

        # --- 1. Shoulder stability (upper arm should stay still) ---
        if shoulder_angle > 45:
            tips.append(("You are swinging your upper arm. Try to keep your elbow pinned to your side to isolate the bicep.", "swing"))
        elif shoulder_angle > 35:
            tips.append(("Pin your elbow to your ribs and keep it stable.", "elbow_pin"))

        # --- 2. Track shoulder angle over the rep to detect momentum ---
        if rep_state.phase in ("descending", "ascending"):
            rep_state.shoulder_angles.append(shoulder_angle)
            if len(rep_state.shoulder_angles) > 10:
                sh_range = max(rep_state.shoulder_angles) - min(rep_state.shoulder_angles)
                if sh_range > 20:
                    tips.append(("You are using momentum. Slow down and control the movement.", "momentum"))

        # --- 3. Range of motion at the TOP (full curl) ---
        if rep_state.phase == "ascending" and rep_state.min_angle > 75:
            tips.append(("Curl higher! Bring your hand closer to your shoulder to fully contract the bicep.", "curl_higher"))

        # --- 4. Range of motion at BOTTOM (full extension) ---
        if rep_state.phase == "idle" and rep_state.max_angle < 130 and rep_state.rep_count > 0:
            tips.append(("Lower the weight all the way down for a full stretch in your arm.", "extend"))

        # --- 5. Body lean check ---
        if side == "left":
            sh = landmarks[LM["L_SHOULDER"]]
            hp = landmarks[LM["L_HIP"]]
        else:
            sh = landmarks[LM["R_SHOULDER"]]
            hp = landmarks[LM["R_HIP"]]

        lean = sh.x - hp.x
        if abs(lean) > 0.06:
            tips.append(("Stand up straight and engage your core. Don't lean back.", "lean"))

        return tips

    def coach_squat(
        self, landmarks: list, angles: dict[str, float],
        side: str, rep_state: RepetitionState,
    ) -> list[tuple[str, str]]:
        """Analyse squat form from side view."""
        tips: list[tuple[str, str]] = []

        knee_angle = angles.get("knee", 160)
        hip_angle = angles.get("hip", 170)

        # --- 1. Depth check ---
        if rep_state.phase == "ascending" and rep_state.min_angle > 115:
            tips.append(("Go a little deeper! Try to bring your thighs parallel to the ground.", "depth"))

        # --- 2. Back angle (hip angle shouldn't drop too low = leaning forward) ---
        if hip_angle < 80:
            tips.append(("Keep your chest up. Avoid leaning too far forward.", "chest_up"))

        # --- 3. Forward knee travel (knees passing toes) ---
        if side == "left":
            kn = landmarks[LM["L_KNEE"]]
            an = landmarks[LM["L_ANKLE"]]
            sh = landmarks[LM["L_SHOULDER"]]
            hp = landmarks[LM["L_HIP"]]
        else:
            kn = landmarks[LM["R_KNEE"]]
            an = landmarks[LM["R_ANKLE"]]
            sh = landmarks[LM["R_SHOULDER"]]
            hp = landmarks[LM["R_HIP"]]
            
        torso_height = abs(sh.y - hp.y) or 0.3
        max_knee_forward = torso_height * 0.4  # Max horizontal offset allowed is 40% of torso length
        
        knee_travel = kn.x - an.x
        if abs(knee_travel) > max_knee_forward:
            tips.append(("Keep your weight on your heels. Don't let your knees travel too far forward.", "knee_travel"))

        # --- 4. Standing up fully ---
        if rep_state.phase == "idle" and rep_state.max_angle < 155 and rep_state.rep_count > 0:
            tips.append(("Stand up completely and squeeze your glutes at the top.", "stand_full"))

        return tips

    def coach_shoulder_press(
        self, landmarks: list, angles: dict[str, float],
        side: str, rep_state: RepetitionState,
    ) -> list[tuple[str, str]]:
        """Analyse shoulder press form from front view."""
        tips: list[tuple[str, str]] = []
        elbow_angle = angles.get("elbow", 160)
        shoulder_angle = angles.get("shoulder", 90)

        # --- 1. Symmetry check ---
        left_press = self.extract_side_features_3d(landmarks, ExerciseType.SHOULDER_PRESS, "left")
        right_press = self.extract_side_features_3d(landmarks, ExerciseType.SHOULDER_PRESS, "right")
        if left_press and right_press:
            asym = abs(left_press["elbow"] - right_press["elbow"])
            if asym > 25:
                tips.append(("Keep your arms symmetrical. Press both weights together.", "asymmetry"))

        # --- 2. Depth check (elbows dropping too low) ---
        if shoulder_angle < 45 and rep_state.phase == "descending":
            tips.append(("Do not drop your elbows too low. Keep them at 90 degrees.", "elbow_drop"))

        # --- 3. Full range of motion at top ---
        if rep_state.phase == "ascending" and rep_state.min_angle > 110:
            tips.append(("Press all the way up! Extend your arms overhead.", "press_up"))

        return tips

    def coach_push_up(
        self, landmarks: list, angles: dict[str, float],
        side: str, rep_state: RepetitionState,
    ) -> list[tuple[str, str]]:
        """Analyse push-up form from side view."""
        tips: list[tuple[str, str]] = []
        elbow_angle = angles.get("elbow", 150)
        hip_angle = angles.get("hip", 175)

        # --- 1. Hip sag / Spike (body straight alignment) ---
        if hip_angle < 155:
            tips.append(("Keep your body straight. Do not sag or spike your hips.", "hip_sag"))

        # --- 2. Push depth ---
        if rep_state.phase == "ascending" and rep_state.min_angle > 110:
            tips.append(("Go lower! Try to bring your chest closer to the floor.", "depth"))

        return tips

    def coach_lunge(
        self, landmarks: list, angles: dict[str, float],
        side: str, rep_state: RepetitionState,
    ) -> list[tuple[str, str]]:
        """Analyse lunge form from side view."""
        tips: list[tuple[str, str]] = []
        knee_angle = angles.get("knee", 160)
        hip_angle = angles.get("hip", 170)

        # --- 1. Knee forward position ---
        if side == "left":
            kn = landmarks[LM["L_KNEE"]]
            an = landmarks[LM["L_ANKLE"]]
            sh = landmarks[LM["L_SHOULDER"]]
            hp = landmarks[LM["L_HIP"]]
        else:
            kn = landmarks[LM["R_KNEE"]]
            an = landmarks[LM["R_ANKLE"]]
            sh = landmarks[LM["R_SHOULDER"]]
            hp = landmarks[LM["R_HIP"]]
        
        # Scale-invariant normalization using vertical torso height
        torso_height = abs(sh.y - hp.y) or 0.3
        max_knee_over = torso_height * 0.35  # Max horizontal offset allowed is 35% of torso length
        
        knee_over = kn.x - an.x
        if abs(knee_over) > max_knee_over:
            tips.append(("Do not push your front knee too far forward. Keep it above your ankle.", "knee_forward"))

        # --- 2. Lunge depth ---
        if rep_state.phase == "ascending" and rep_state.min_angle > 115:
            tips.append(("Step deeper into the lunge. Lower your back knee.", "depth"))

        return tips

    def coach_superman(
        self, landmarks: list, angles: dict[str, float],
        side: str, rep_state: RepetitionState,
    ) -> list[tuple[str, str]]:
        """Analyse superman lift form from side view."""
        tips: list[tuple[str, str]] = []
        hip_angle = angles.get("hip", 180)
        shoulder_angle = angles.get("shoulder", 180)

        if rep_state.phase == "ascending" and rep_state.min_angle > 172:
            tips.append(("Lift your chest and thighs higher off the ground.", "lift"))

        return tips

    def coach_dips(
        self, landmarks: list, angles: dict[str, float],
        side: str, rep_state: RepetitionState,
    ) -> list[tuple[str, str]]:
        """Analyse dips form from side view."""
        tips: list[tuple[str, str]] = []
        elbow_angle = angles.get("elbow", 150)
        shoulder_angle = angles.get("shoulder", 90)

        # --- 1. Dip depth ---
        if rep_state.phase == "ascending" and rep_state.min_angle > 110:
            tips.append(("Dip lower! Try to flex your elbows to 90 degrees.", "depth"))

        # --- 2. Body proximity to chair ---
        if side == "left":
            sh = landmarks[LM["L_SHOULDER"]]
            hp = landmarks[LM["L_HIP"]]
        else:
            sh = landmarks[LM["R_SHOULDER"]]
            hp = landmarks[LM["R_HIP"]]
        
        dist = abs(sh.x - hp.x)
        if dist > 0.08:
            tips.append(("Keep your back close to the bench or chair.", "bench_dist"))

        return tips

    def coach_plank(
        self, landmarks: list, angles: dict[str, float],
        side: str, rep_state: RepetitionState,
    ) -> list[tuple[str, str]]:
        """Analyse plank form from side view."""
        tips: list[tuple[str, str]] = []
        hip_angle = angles.get("hip", 175)

        if hip_angle < 155:
            tips.append(("Keep your hips straight. Avoid sagging or spiking your hips.", "hip_sag"))
        elif hip_angle > 185:
            tips.append(("Bring your hips down. Your body should form a straight line.", "hip_spike"))

        return tips

    # ------------------------------------------------------------------
    # DTW evaluation
    # ------------------------------------------------------------------
    @staticmethod
    def evaluate_form(
        user_seq: np.ndarray, ref_seq: np.ndarray, threshold: float = 25.0,
    ) -> FormEvaluation:
        user = np.atleast_2d(np.asarray(user_seq, dtype=np.float64))
        ref = np.atleast_2d(np.asarray(ref_seq, dtype=np.float64))
        if user.shape[0] < 2 or ref.shape[0] < 2:
            return FormEvaluation(float("inf"), 0, False, threshold, "Not enough movement frames captured.")

        dist, _ = fastdtw(user, ref, dist=euclidean)
        # Normalize by max length AND by square root of features to represent average degrees deviation
        norm = dist / (max(user.shape[0], ref.shape[0]) * np.sqrt(user.shape[1]))
        
        # Softer exponential decay: average 10 degrees difference maps to ~80% score
        score = float(np.clip(100.0 * np.exp(-0.022 * norm), 0, 100))
        ok = score >= 60.0

        if score >= 82:
            fb = f"Perfect form! Score {score:.0f}. Keep doing exactly like that."
        elif score >= 55:
            fb = f"That rep was decent, score {score:.0f}, but try to stabilize and control your tempo more."
        else:
            fb = f"Form needs correction, score {score:.0f}. Focus on full range of motion and slow down."

        return FormEvaluation(norm, score, ok, threshold, fb)

    # ------------------------------------------------------------------
    # Rep state machine
    # ------------------------------------------------------------------
    def _update_rep(
        self, state: RepetitionState, angle: float,
        angles: dict, config: dict, ref_seq, evals: list,
        exercise_type: ExerciseType,
    ) -> bool:
        """Returns True if a new rep was just completed."""
        if exercise_type == ExerciseType.PLANK:
            # Plank hold tracking: 5 seconds of correct posture (150 frames @ 30fps) = 1 rep
            hip = angles.get("hip", 175)
            if 155 <= hip <= 185:
                if not hasattr(state, 'plank_frames'):
                    state.plank_frames = 0
                state.plank_frames += 1
                if state.plank_frames >= 150:
                    state.rep_count += 1
                    state.plank_frames = 0
                    return True
            else:
                if hasattr(state, 'plank_frames'):
                    state.plank_frames = 0
            return False

        start = config["start"]
        bottom = config["bottom"]
        vec = list(angles.values())

        state.max_angle = max(state.max_angle, angle)

        if state.phase == "idle":
            if angle < start:
                state.phase = "descending"
                state.angle_sequence = [vec]
                state.min_angle = angle
                state.max_angle = angle
                state.shoulder_angles = []

        elif state.phase == "descending":
            state.angle_sequence.append(vec)
            state.min_angle = min(state.min_angle, angle)
            if angle <= bottom:
                state.phase = "ascending"

        elif state.phase == "ascending":
            state.angle_sequence.append(vec)
            if angle >= start:
                state.rep_count += 1
                state.phase = "idle"

                if ref_seq is not None and len(state.angle_sequence) > 3:
                    ev = self.evaluate_form(
                        np.array(state.angle_sequence), ref_seq, self.dtw_threshold,
                    )
                    evals.append(ev)

                state.angle_sequence = []
                state.shoulder_angles = []
                return True

        state.prev_angle = angle
        return False

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------
    def process_stream(
        self,
        source: str | int = 0,
        exercise_type: ExerciseType = ExerciseType.BICEP_CURL,
        reference_sequence: Optional[np.ndarray] = None,
        show_preview: bool = True,
        output_path: Optional[str] = None,
    ) -> dict:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise IOError(f"Cannot open: {source}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer = None
        if output_path:
            writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        if show_preview:
            self._show_instructions(cap, exercise_type)

        rep_state = RepetitionState()
        config = self._REP_CONFIG[exercise_type]
        evals: list[FormEvaluation] = []
        latest_eval: Optional[FormEvaluation] = None
        active_side = "left"
        orientation = "unknown"
        current_tips: list[str] = []  # Currently displayed coaching tips
        tip_display_text = ""

        ts = 0
        dt = int(1000 / fps)
        frame_n = 0
        coach_interval = 15  # Check form every N frames
        last_coach_frame = 0

        is_front = exercise_type in (ExerciseType.SHOULDER_PRESS,)
        if is_front:
            self.audio.say("Welcome to your training session! Please stand directly facing the camera with your full body visible, and begin the exercise when you are ready.", cooldown=0)
        else:
            self.audio.say("Welcome to your training session! Please stand sideways to the camera so that your side profile is visible, and begin the exercise when you are ready.", cooldown=0)

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame_n += 1
                ts += dt

                lm = self.extract_landmarks(frame, ts)
                angles: Optional[dict[str, float]] = None
                left_angles: Optional[dict[str, float]] = None
                right_angles: Optional[dict[str, float]] = None

                if lm is not None:
                    orientation = self.detect_orientation(lm)

                    if exercise_type in (ExerciseType.SHOULDER_PRESS,):
                        active_side = "both"
                        left_angles = self.extract_features(lm, exercise_type, "left")
                        right_angles = self.extract_features(lm, exercise_type, "right")

                        if left_angles is not None and right_angles is not None:
                            angles = {
                                "elbow": (left_angles["elbow"] + right_angles["elbow"]) / 2.0,
                                "shoulder": (left_angles["shoulder"] + right_angles["shoulder"]) / 2.0,
                            }
                        else:
                            angles = None
                    else:
                        # Auto-detect active side for Bicep Curl and other side-view exercises
                        detected_side = self.detect_active_side(lm, exercise_type)
                        active_side = detected_side
                        angles = self.extract_features(lm, exercise_type, active_side)

                    if angles is not None:
                        primary = config["primary"]
                        p_angle = angles.get(primary, 180.0)

                        # Update rep counter
                        new_rep = self._update_rep(
                            rep_state, p_angle, angles,
                            config, reference_sequence, evals, exercise_type
                        )

                        if new_rep:
                            rep_num = rep_state.rep_count
                            latest_eval = evals[-1] if evals else None
                            
                            if latest_eval:
                                score = latest_eval.form_score
                                if score >= 82:
                                    self.audio.say(f"Rep {rep_num} completed. Excellent form! Keep it up.", cooldown=0.1, key="count")
                                elif score >= 55:
                                    self.audio.say(f"Rep {rep_num} done. Nice effort, but try to control your speed and stabilize your movement a bit more.", cooldown=0.1, key="count")
                                else:
                                    self.audio.say(f"Rep {rep_num} recorded. However, your form needs correction. Please slow down and focus on control.", cooldown=0.1, key="count")
                            else:
                                self.audio.say(f"Rep {rep_num} completed.", cooldown=0.1, key="count")

                        # --- Contextual coaching (every N frames) ---
                        if frame_n - last_coach_frame >= coach_interval:
                            last_coach_frame = frame_n

                            if exercise_type == ExerciseType.BICEP_CURL:
                                tips = self.coach_bicep_curl(lm, angles, active_side, rep_state)
                            elif exercise_type == ExerciseType.SQUAT:
                                tips = self.coach_squat(lm, angles, active_side, rep_state)
                            elif exercise_type == ExerciseType.SHOULDER_PRESS:
                                tips = self.coach_shoulder_press(lm, angles, active_side, rep_state)
                            elif exercise_type == ExerciseType.PUSH_UP:
                                tips = self.coach_push_up(lm, angles, active_side, rep_state)
                            elif exercise_type == ExerciseType.LUNGE:
                                tips = self.coach_lunge(lm, angles, active_side, rep_state)
                            elif exercise_type == ExerciseType.SUPERMAN:
                                tips = self.coach_superman(lm, angles, active_side, rep_state)
                            elif exercise_type == ExerciseType.DIPS:
                                tips = self.coach_dips(lm, angles, active_side, rep_state)
                            elif exercise_type == ExerciseType.PLANK:
                                tips = self.coach_plank(lm, angles, active_side, rep_state)
                            else:
                                tips = []

                            # Orientation warning
                            if exercise_type in (ExerciseType.SHOULDER_PRESS,):
                                if orientation == "side":
                                    tips.append(("Please face the camera directly so that both of your knees and hips can be tracked from the front.", "orientation"))
                            else:
                                if orientation == "front":
                                    tips.append(("Please stand sideways to the camera so that your arm and joints are visible from the side.", "orientation"))

                            # Speak the most important tip
                            if tips:
                                msg, key = tips[0]
                                self.audio.say(msg, cooldown=5.0, key=key)
                                tip_display_text = msg
                            else:
                                tip_display_text = ""
                                # If the form is correct during the rep, speak positive reinforcement
                                if rep_state.phase in ("descending", "ascending"):
                                    positives = [
                                        "Perfect posture! Keep going.",
                                        "Excellent control. Stay steady.",
                                        "Great form! Keep pushing.",
                                        "Looking solid. Nice alignment.",
                                        "Perfect execution!"
                                    ]
                                    pos_msg = positives[frame_n % len(positives)]
                                    self.audio.say(pos_msg, cooldown=7.0, key="positive_reinforcement")

                    # Draw skeleton
                    self._draw_skeleton(frame, lm)
                    
                    # Draw angles on joints
                    if exercise_type in (ExerciseType.SHOULDER_PRESS,):
                        if left_angles and right_angles:
                            # Draw left side 3D angles
                            jmap_l = self._JOINTS[exercise_type]["left"]
                            for jname, (_, jidx, _) in jmap_l.items():
                                if jname in left_angles:
                                    self._draw_angle_on_joint(frame, lm, jidx, left_angles[jname])
                            
                            # Draw right side 3D angles
                            jmap_r = self._JOINTS[exercise_type]["right"]
                            for jname, (_, jidx, _) in jmap_r.items():
                                if jname in right_angles:
                                    self._draw_angle_on_joint(frame, lm, jidx, right_angles[jname])
                    else:
                        if angles:
                            jmap = self._JOINTS[exercise_type][active_side]
                            for jname, (_, jidx, _) in jmap.items():
                                if jname in angles:
                                    self._draw_angle_on_joint(frame, lm, jidx, angles[jname])

                # HUD
                self._draw_hud(
                    frame, angles, rep_state, latest_eval,
                    exercise_type, active_side, orientation, tip_display_text,
                )

                if writer:
                    writer.write(frame)
                if show_preview:
                    cv2.imshow("CV for Fit", frame)
                    k = cv2.waitKey(1) & 0xFF
                    if k == ord("q") or k == 27:
                        break
        finally:
            cap.release()
            if writer:
                writer.release()
            if show_preview:
                cv2.destroyAllWindows()
            total = rep_state.rep_count
            self.audio.say(f"Workout session completed! You have finished a total of {total} repetitions. Excellent effort today, keep up the good work!", cooldown=0)
            time.sleep(3.5)
            self.release()

        return {"reps": rep_state.rep_count, "evaluations": evals, "side": active_side}

    # ==================================================================
    # Drawing helpers
    # ==================================================================
    def _show_instructions(self, cap, exercise_type):
        title_text = exercise_type.value.replace("_", " ").upper()
        is_front = exercise_type in (ExerciseType.SQUAT, ExerciseType.SHOULDER_PRESS)

        if is_front:
            lines = [
                (f"{title_text} SETUP", True),
                ("", False),
                ("1. Stand FACING the camera", False),
                ("   (front view setup)", False),
                ("2. Keep your FULL BODY visible", False),
                ("3. Stand with feet shoulder-width apart", False),
                ("", False),
                ("The system will track both sides in 3D", False),
                ("and coach your posture in real time.", False),
                ("", False),
                ("Press SPACE to start", True),
            ]
            self.audio.say(f"Before we begin, please stand directly facing the camera so it can track both of your sides. Press the spacebar on your keyboard when you are ready to start.", cooldown=0)
        else:
            lines = [
                (f"{title_text} SETUP", True),
                ("", False),
                ("1. Stand SIDEWAYS to the camera", False),
                ("   (your side profile facing the camera)", False),
                ("2. Keep your FULL BODY visible", False),
                ("3. Keep your joints in clear view", False),
                ("", False),
                ("The system will auto-detect your active side", False),
                ("and coach your form in real time.", False),
                ("", False),
                ("Press SPACE to start", True),
            ]
            self.audio.say(f"Before we begin, please stand sideways to the camera so that your side profile is visible. Press the spacebar on your keyboard when you are ready to start.", cooldown=0)

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            h, w = frame.shape[:2]
            ov = frame.copy()
            cv2.rectangle(ov, (0, 0), (w, h), (15, 15, 15), -1)
            cv2.addWeighted(ov, 0.8, frame, 0.2, 0, frame)

            y = 70
            font = cv2.FONT_HERSHEY_SIMPLEX
            for text, highlight in lines:
                if not text:
                    y += 15
                    continue
                color = (0, 220, 255) if highlight else (200, 200, 200)
                size = 0.85 if highlight else 0.6
                thick = 2 if highlight else 1
                cv2.putText(frame, text, (40, y), font, size, color, thick, cv2.LINE_AA)
                y += 38

            cv2.imshow("CV for Fit", frame)
            k = cv2.waitKey(1) & 0xFF
            if k == ord(" "):
                self.audio.stop()
                break
            if k == ord("q") or k == 27:
                self.audio.stop()
                raise KeyboardInterrupt()

    # ------------------------------------------------------------------
    @staticmethod
    def _draw_skeleton(frame, lm):
        h, w = frame.shape[:2]
        for s, e in _POSE_CONNECTIONS:
            if s >= len(lm) or e >= len(lm):
                continue
            if lm[s].visibility < 0.15 or lm[e].visibility < 0.15:
                continue
            p1 = (int(lm[s].x * w), int(lm[s].y * h))
            p2 = (int(lm[e].x * w), int(lm[e].y * h))
            cv2.line(frame, p1, p2, (0, 220, 120), 2, cv2.LINE_AA)
        for m in lm:
            if m.visibility < 0.15:
                continue
            c = (int(m.x * w), int(m.y * h))
            cv2.circle(frame, c, 4, (0, 140, 255), -1, cv2.LINE_AA)

    @staticmethod
    def _draw_angle_on_joint(frame, lm, idx, val):
        h, w = frame.shape[:2]
        cx, cy = int(lm[idx].x * w), int(lm[idx].y * h)
        color = (0, 230, 120) if val > 140 else (0, 140, 255) if val < 70 else (0, 220, 255)
        txt = f"{int(val)}"
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(frame, (cx + 6, cy - th - 4), (cx + 12 + tw, cy + 4), (0, 0, 0), -1)
        cv2.putText(frame, txt, (cx + 8, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

    @staticmethod
    def _draw_hud(
        frame, angles, rep_state, latest_eval,
        exercise_type, active_side, orientation, tip_text,
    ):
        h, w = frame.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        white = (255, 255, 255)
        green = (0, 230, 120)
        yellow = (0, 220, 255)
        red = (60, 60, 255)
        cyan = (255, 200, 0)
        gray = (130, 130, 130)

        # Panel
        ov = frame.copy()
        panel_h = 210 if tip_text else 165
        cv2.rectangle(ov, (8, 8), (420, 8 + panel_h), (15, 15, 15), -1)
        cv2.addWeighted(ov, 0.72, frame, 0.28, 0, frame)

        y = 32

        # Title + side
        is_front = exercise_type in (ExerciseType.SHOULDER_PRESS,)
        if is_front:
            title = f"{exercise_type.value.replace('_',' ').title()} (Front View)"
        else:
            side_label = active_side.upper()
            title = f"{exercise_type.value.replace('_',' ').title()} ({side_label} side)"
        cv2.putText(frame, title, (16, y), font, 0.6, yellow, 2, cv2.LINE_AA)
        y += 30

        # Angles
        if angles:
            for name, val in angles.items():
                if name in ("elbow", "knee"):
                    c = green if val > 140 else cyan if val > 80 else yellow if val > 50 else red
                else:
                    c = white
                cv2.putText(frame, f"{name}: {val:.0f} deg", (16, y), font, 0.55, c, 1, cv2.LINE_AA)
                y += 24
        else:
            cv2.putText(frame, "Waiting for pose...", (16, y), font, 0.5, gray, 1, cv2.LINE_AA)
            y += 24

        y += 4

        # Rep counter (large)
        cv2.putText(
            frame, f"Reps: {rep_state.rep_count}",
            (16, y), font, 0.85, green, 2, cv2.LINE_AA,
        )

        # Phase
        phase = rep_state.phase
        if phase == "descending":
            pcolor, plabel = yellow, "WORKING"
        elif phase == "ascending":
            pcolor, plabel = cyan, "RETURNING"
        else:
            pcolor, plabel = gray, "READY"
        cv2.putText(frame, plabel, (220, y), font, 0.6, pcolor, 1, cv2.LINE_AA)
        y += 28

        # Orientation check
        if is_front:
            if orientation == "side":
                cv2.putText(frame, "! FACE THE CAMERA", (16, y), font, 0.5, red, 2, cv2.LINE_AA)
            else:
                cv2.putText(frame, "Front view OK", (16, y), font, 0.45, green, 1, cv2.LINE_AA)
        else:
            if orientation == "front":
                cv2.putText(frame, "! TURN SIDEWAYS", (16, y), font, 0.5, red, 2, cv2.LINE_AA)
            else:
                cv2.putText(frame, "Side view OK", (16, y), font, 0.45, green, 1, cv2.LINE_AA)
        y += 24

        # Coaching tip
        if tip_text:
            # Draw with a slight highlight background
            cv2.putText(frame, tip_text[:52], (16, y), font, 0.48, (100, 180, 255), 1, cv2.LINE_AA)
            y += 22

        # DTW score (bottom-left)
        if latest_eval:
            sc = latest_eval.form_score
            sc_c = green if latest_eval.is_correct else red
            cv2.putText(
                frame, f"Form: {sc:.0f}/100",
                (16, y), font, 0.5, sc_c, 1, cv2.LINE_AA,
            )

        # Big rep counter bottom-right
        total_txt = str(rep_state.rep_count)
        (tw, th), _ = cv2.getTextSize(total_txt, font, 2.5, 4)
        tx = w - tw - 30
        ty = h - 30
        cv2.putText(frame, total_txt, (tx, ty), font, 2.5, (40, 40, 40), 8, cv2.LINE_AA)
        cv2.putText(frame, total_txt, (tx, ty), font, 2.5, green, 3, cv2.LINE_AA)


# ===================================================================
# Synthetic reference
# ===================================================================
def generate_synthetic_reference(ex: ExerciseType, n: int = 60) -> np.ndarray:
    t = np.linspace(0, np.pi, n)
    if ex in (ExerciseType.BICEP_CURL, ExerciseType.SUPERMAN, ExerciseType.SHOULDER_PRESS):
        # 2 Features
        return np.column_stack([90 + 60 * np.abs(np.cos(t)), 45 + 40 * np.abs(np.cos(t))])
    else:
        # 3 Features
        return np.column_stack([100 + 60 * np.abs(np.cos(t)), 90 + 70 * np.abs(np.cos(t)), 80 + 30 * np.abs(np.cos(t))])


# ===================================================================
# CLI
# ===================================================================
def main():
    import argparse
    p = argparse.ArgumentParser(description="CV for Fit")
    p.add_argument("--source", default="0")
    p.add_argument(
        "--exercise", 
        choices=["bicep_curl", "squat", "lunge", "push_up", "superman", "shoulder_press", "dips", "plank"], 
        default="bicep_curl"
    )
    p.add_argument("--threshold", type=float, default=25.0)
    p.add_argument("--output", default=None)
    p.add_argument("--no-preview", action="store_true")
    p.add_argument("--no-audio", action="store_true")
    args = p.parse_args()

    try:
        src: int | str = int(args.source)
    except ValueError:
        src = args.source

    ex = ExerciseType(args.exercise)
    ref = generate_synthetic_reference(ex)
    analyzer = PoseAnalyzer(dtw_threshold=args.threshold, enable_audio=not args.no_audio)

    try:
        result = analyzer.process_stream(
            source=src, exercise_type=ex,
            reference_sequence=ref,
            show_preview=not args.no_preview,
            output_path=args.output,
        )
    except (IOError, KeyboardInterrupt) as e:
        logger.info("Stopped: %s", e)
        sys.exit(0)

    print("\n" + "=" * 55)
    print(f"  {ex.value.replace('_',' ').title()} - Session Summary")
    print("=" * 55)
    if ex == ExerciseType.SQUAT:
        print("  Tracking view: Front View")
    else:
        print(f"  Side tracked : {result['side'].capitalize()}")
    print(f"  Total reps   : {result['reps']}")
    evs = result.get("evaluations", [])
    if evs:
        avg = np.mean([e.form_score for e in evs])
        print(f"  Avg score    : {avg:.0f}/100")
        for i, e in enumerate(evs, 1):
            tag = "OK" if e.is_correct else "XX"
            print(f"    Rep {i:>2d} [{tag}]  {e.form_score:5.1f} - {e.feedback}")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    main()
