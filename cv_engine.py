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
VOICE_PHRASES = {
    # Natural, realistic human coach counts every 2 reps (2, 4, 6, 8, 10...)
    "rep_2": {"ar": "عدتان، ممتاز! حافظ على نفس الإيقاع والثبات.", "en": "Two reps down, great rhythm! Keep it steady."},
    "rep_4": {"ar": "أربع عدات، عاش! تنفس بانتظام واصل التركيز.", "en": "Four reps, strong form! Breathe steadily and stay focused."},
    "rep_6": {"ar": "ست عدات! أداء بطولي، استمر بنفس القوة والمدى الكامل.", "en": "Six reps! Excellent strength, maintain full range."},
    "rep_8": {"ar": "ثماني عدات! ثبات رائع وعضلاتك مشدودة، واصل يا بطل!", "en": "Eight reps! Great stamina and clean execution, keep pushing!"},
    "rep_10": {"ar": "عشر عدات كاملة! مجهود عظيم، واصل لآخر عدة!", "en": "Ten complete reps! Outstanding effort, finish strong!"},
    "rep_even": {"ar": "{num} عدات ممتازة! أداء احترافي، كمل!", "en": "{num} clean reps! Professional form, keep going!"},
    "rep_mistake": {"ar": "العدة السابقة غير مكتملة المدى الحركي، ركّز على العمق الصحيح.", "en": "Last rep had incomplete range, focus on reaching full depth."},

    # Welcome / Setup
    "welcome_front": {
        "ar": "أهلاً بك! يرجى الوقوف بمواجهة الكاميرا مباشرة، وابدأ عندما تكون مستعداً.",
        "en": "Welcome! Please stand directly facing the camera, and begin when ready."
    },
    "welcome_side": {
        "ar": "أهلاً بك! يرجى الوقوف بالجنب للكاميرا، وابدأ عندما تكون مستعداً.",
        "en": "Welcome! Please stand sideways to the camera, and begin when ready."
    },

    # Plank
    "plank_start": {"ar": "تم رصد البلانك! بدأ احتساب الثواني، اثبت!", "en": "Plank detected! Timer started. Keep holding."},
    "plank_pause": {"ar": "انتبه! توقف المؤقت، اضبط استقامة الظهر لاستئناف العد.", "en": "Posture lost! Timer paused. Straighten your body to resume."},
    "plank_10": {"ar": "ثبات ممتاز! 10 ثوانٍ مكتملة.", "en": "Great hold! 10 seconds completed."},
    "plank_20": {"ar": "20 ثانية، واصل الثبات!", "en": "20 seconds, stay steady!"},
    "plank_30": {"ar": "30 ثانية، مجهود بطولي!", "en": "30 seconds, heroic effort!"},
    "plank_45": {"ar": "45 ثانية، اقتربت من الهدف!", "en": "45 seconds, almost there!"},
    "plank_60": {"ar": "دقيقة كاملة! أداء أسطوري!", "en": "One full minute! Incredible hold!"},

    # Session completion
    "session_done_reps": {
        "ar": "انتهت الجلسة بنجاح! أنجزت {total} تكراراً، مجهود رائع اليوم!",
        "en": "Workout session completed! You finished a total of {total} reps. Excellent effort!"
    },
    "session_done_plank": {
        "ar": "انتهت الجلسة بنجاح! حققت ثباتاً لمدة {total} ثانية، مجهود رائع!",
        "en": "Workout session completed! You held the plank for {total} seconds. Excellent effort!"
    }
}

COACH_TIPS_I18N = {
    # Bicep Curl
    "curl_higher": {
        "ar": "ارفع يدك للأعلى أكثر لانقباض كامل لعضلة الباي.",
        "en": "Curl higher! Bring your hand closer to your shoulder to fully contract the bicep."
    },
    "extend": {
        "ar": "افرد ذراعك للأسفل بالكامل لتمديد العضلة.",
        "en": "Lower the weight all the way down for a full stretch in your arm."
    },
    "lean": {
        "ar": "قف مستقيماً وشد عضلات البطن، لا تمل للخلف.",
        "en": "Stand up straight and engage your core. Don't lean back."
    },
    "elbow_swing": {
        "ar": "ثبت كوعك بجانب خصرك وتجنب أرجحة الذراع.",
        "en": "Keep your upper arm still. Don't swing."
    },
    "swing": {
        "ar": "ثبت كوعك بجانب خصرك وتجنب أرجحة الذراع.",
        "en": "Keep your upper arm still. Don't swing."
    },
    "elbow_pin": {
        "ar": "ثبت كوعك بمحاذاة أضلاعك ولا تدعه يتحرك.",
        "en": "Pin your elbow to your ribs and keep it stable."
    },
    "momentum": {
        "ar": "تحكم بالوزن وتجنب استخدام قوة الاندفاع.",
        "en": "You are using momentum. Slow down and control the movement."
    },

    # Squat
    "depth": {
        "ar": "انزل أكثر! اجعل الفخذين موازيين للأرض.",
        "en": "Go a little deeper! Try to bring your thighs parallel to the ground."
    },
    "chest_up": {
        "ar": "ارفع صدرك للأعلى وتجنب الميل للأمام.",
        "en": "Keep your chest up. Avoid leaning too far forward."
    },
    "knee_travel": {
        "ar": "وزع وزنك على الكعبين، لا تدفع ركبتك للأمام.",
        "en": "Keep your weight on your heels. Don't let your knees travel too far forward."
    },
    "stand_full": {
        "ar": "اصعد وافرد ركبتيك بالكامل عند نهاية العدة.",
        "en": "Stand up completely and squeeze your glutes at the top."
    },

    # Shoulder Press
    "shoulder_press_elbow": {
        "ar": "أنزل أوزانك لمستوى الأذنين بزاوية 90 درجة.",
        "en": "Do not drop your elbows too low. Keep them at 90 degrees."
    },
    "elbow_drop": {
        "ar": "أنزل أوزانك لمستوى الأذنين بزاوية 90 درجة.",
        "en": "Do not drop your elbows too low. Keep them at 90 degrees."
    },
    "shoulder_press_up": {
        "ar": "ادفع الأوزان لأعلى وافرد ذراعيك فوق رأسك.",
        "en": "Press all the way up! Extend your arms overhead."
    },
    "press_up": {
        "ar": "ادفع الأوزان لأعلى وافرد ذراعيك فوق رأسك.",
        "en": "Press all the way up! Extend your arms overhead."
    },
    "shoulder_press_sym": {
        "ar": "حافظ على تماثل حركة الذراعين وارفعهما معاً.",
        "en": "Keep your arms symmetrical. Press both weights together."
    },
    "asymmetry": {
        "ar": "حافظ على تماثل حركة الذراعين وارفعهما معاً.",
        "en": "Keep your arms symmetrical. Press both weights together."
    },

    # Plank
    "hip_sag": {
        "ar": "ارفع حوضك قليلاً، لا تدع أسفل ظهرك يرتخي.",
        "en": "Lift your hips slightly! Don't let your lower back sag."
    },
    "hip_pike": {
        "ar": "أنزل حوضك، اجعل جسمك مستقيماً تماماً.",
        "en": "Lower your hips! Keep your body in a straight line."
    },
    "hip_arch": {
        "ar": "حافظ على استقامة الظهر وعضلات البطن مشدودة.",
        "en": "Straighten your core. Don't arch your back."
    },
    "knee_bend": {
        "ar": "افرد ركبتيك وشد عضلات الفخذ.",
        "en": "Keep your knees straight and engage your thighs."
    },
    "plank_posture": {
        "ar": "اتخذ وضعية البلانك الأفقية على الأرض.",
        "en": "Get into a horizontal plank position on the floor."
    },

    # Lunge
    "knee_forward": {
        "ar": "لا تدفع ركبتك الأمامية بعيداً، حافظ عليها فوق الكاحل.",
        "en": "Do not push your front knee too far forward. Keep it above your ankle."
    },
    "lunge_depth": {
        "ar": "انزل بركبتك الخلفية أكثر نحو الأرض.",
        "en": "Step deeper into the lunge. Lower your back knee."
    },

    # Push-up
    "pushup_depth": {
        "ar": "انزل بصدرك أكثر نحو الأرض لمدى حركي كامل.",
        "en": "Go lower! Try to bring your chest closer to the floor."
    },

    # Superman
    "lift": {
        "ar": "ارفع صدرك وفخذيك أعلى عن الأرض.",
        "en": "Lift your chest and thighs higher off the ground."
    },

    # Dips
    "dips_depth": {
        "ar": "انزل أكثر واثنِ كوعيك بزاوية 90 درجة.",
        "en": "Dip lower! Try to flex your elbows to 90 degrees."
    },
    "bench_dist": {
        "ar": "حافظ على ظهرك قريباً من المقعد أو الكرسي.",
        "en": "Keep your back close to the bench or chair."
    },

    # Orientation
    "orientation": {
        "ar": "يرجى تعديل وقفتك أمام الكاميرا حسب الإرشادات.",
        "en": "Please adjust your stance facing the camera as instructed."
    },
    "orientation_front": {
        "ar": "يرجى مواجهة الكاميرا مباشرة ليظهر جسمك بوضوح.",
        "en": "Please face the camera directly so that both sides can be tracked."
    },
    "orientation_side": {
        "ar": "يرجى الوقوف بالجنب للكاميرا لظهور تفاصيل الحركة.",
        "en": "Please stand sideways to the camera so that your joints are visible from the side."
    }
}

POSITIVE_PHRASES = {
    "ar": [
        "وضعية ممتازة! واصل.",
        "تحكم رائع، استمر بثبات.",
        "عاش! أداء احترافي.",
        "استقامة ممتازة، كمل!",
        "تنفيذ مثالي للحركة!"
    ],
    "en": [
        "Perfect posture! Keep going.",
        "Excellent control. Stay steady.",
        "Great form! Keep pushing.",
        "Looking solid. Nice alignment.",
        "Perfect execution!"
    ]
}

class AudioFeedback:
    """Non-blocking voice coach running in a background thread with bilingual support."""

    def __init__(self, enabled: bool = True, lang: str = "ar"):
        self._enabled = enabled
        self.lang = lang
        self._queue: queue.Queue = queue.Queue()
        self._cooldowns: dict[str, float] = {}
        self._engine = None
        self._lock = threading.Lock()
        if enabled:
            t = threading.Thread(target=self._worker, daemon=True)
            t.start()

    def set_language(self, lang: str):
        """Switch audio coach language dynamically."""
        if lang in ("ar", "en"):
            self.lang = lang
            self.stop()
            with self._lock:
                try:
                    if self._engine:
                        del self._engine
                except Exception:
                    pass
                self._engine = self._create_engine()

    def say(self, text: str, cooldown: float = 3.0, key: str = "") -> None:
        """Speak raw text with a per-key cooldown."""
        if not self._enabled:
            return
        k = key or text
        now = time.time()
        if now - self._cooldowns.get(k, 0) < cooldown:
            return
        self._cooldowns[k] = now
        while self._queue.qsize() > 2:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        self._queue.put(text)

    def say_phrase(self, phrase_key: str, cooldown: float = 3.0, key: str = "", **kwargs) -> None:
        """Speak a localized phrase from VOICE_PHRASES based on current language."""
        phrase_data = VOICE_PHRASES.get(phrase_key, {})
        text = phrase_data.get(self.lang) or phrase_data.get("en") or phrase_key
        if kwargs:
            try:
                text = text.format(**kwargs)
            except Exception:
                pass
        self.say(text, cooldown=cooldown, key=key or phrase_key)

    def stop(self) -> None:
        """Stop any current speech and clear queue immediately."""
        if not self._enabled:
            return
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        with self._lock:
            if self._engine is not None:
                try:
                    self._engine.stop()
                except Exception:
                    pass

    def shutdown(self):
        self.stop()
        self._queue.put(None)

    def _create_engine(self):
        try:
            eng = pyttsx3.init()
            eng.setProperty("rate", 165)
            voices = eng.getProperty("voices")
            
            # Select language voice if available
            chosen = None
            if self.lang == "ar":
                for v in voices:
                    v_name = v.name.lower()
                    if "arabic" in v_name or "hoda" in v_name or "naayf" in v_name or "maged" in v_name or "tarik" in v_name:
                        chosen = v.id
                        break
            if not chosen:
                for v in voices:
                    if "zira" in v.name.lower() or "female" in v.name.lower() or "david" in v.name.lower():
                        chosen = v.id
                        break
            if chosen:
                eng.setProperty("voice", chosen)
            return eng
        except Exception:
            return None

    def _worker(self):
        if _HAS_TTS:
            try:
                import pythoncom
                pythoncom.CoInitialize()
            except Exception:
                pass

            with self._lock:
                self._engine = self._create_engine()

            if self._engine is None:
                self._worker_beep()
                return

            while True:
                try:
                    text = self._queue.get(timeout=0.5)
                    if text is None:
                        break
                    
                    try:
                        self._engine.say(text)
                        self._engine.runAndWait()
                    except Exception:
                        pass

                    # Cleanly refresh engine instance after utterance so SAPI5 is never broken
                    with self._lock:
                        try:
                            del self._engine
                        except Exception:
                            pass
                        self._engine = self._create_engine()

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
    hold_seconds: float = 0.0                            # For Plank: seconds held with verified form
    is_holding: bool = False                             # For Plank: currently in correct posture
    progress_pct: float = 0.0                            # 0 to 100% progress for current rep / hold


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
        lang: str = "ar",
    ):
        self._det_conf = min_detection_confidence
        self._trk_conf = min_tracking_confidence
        self.dtw_threshold = dtw_threshold
        self.lang = lang
        self._last_ts_ms = -1
        self._landmarker: Optional[PoseLandmarker] = None
        self.audio = AudioFeedback(enabled=enable_audio, lang=lang)

    def set_language(self, lang: str):
        """Update language for both analyzer and its audio engine."""
        if lang in ("ar", "en"):
            self.lang = lang
            self.audio.set_language(lang)

    def get_tip_text(self, tip: tuple[str, str], lang: Optional[str] = None) -> str:
        """Translate tip tuple (text, key) according to given or default language."""
        target_lang = lang or self.lang
        raw_text, key = tip
        if key in COACH_TIPS_I18N:
            return COACH_TIPS_I18N[key].get(target_lang) or COACH_TIPS_I18N[key].get("en") or raw_text
        return raw_text

    def get_coaching_dict(self, tip: tuple[str, str]) -> dict[str, str]:
        """Return dict with both Arabic and English coaching tips."""
        raw_text, key = tip
        info = COACH_TIPS_I18N.get(key, {})
        return {
            "key": key,
            "ar": info.get("ar", raw_text),
            "en": info.get("en", raw_text),
        }

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

        # Guarantee strict monotonically increasing timestamp required by MediaPipe RunningMode.VIDEO
        if ts_ms <= self._last_ts_ms:
            ts_ms = self._last_ts_ms + 1
        self._last_ts_ms = ts_ms

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        try:
            result = self._get_landmarker().detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts_ms
            )
            if not result.pose_landmarks:
                return None
            return result.pose_landmarks[0]
        except Exception as e:
            logger.warning("MediaPipe inference error: %s", e)
            return None

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
        """Analyse plank form from side view and verify posture."""
        tips: list[tuple[str, str]] = []
        hip_angle = angles.get("hip", 175)
        knee_angle = angles.get("knee", 175)

        if side == "left":
            sh = landmarks[LM["L_SHOULDER"]]
            hp = landmarks[LM["L_HIP"]]
            kn = landmarks[LM["L_KNEE"]]
            an = landmarks[LM["L_ANKLE"]]
        else:
            sh = landmarks[LM["R_SHOULDER"]]
            hp = landmarks[LM["R_HIP"]]
            kn = landmarks[LM["R_KNEE"]]
            an = landmarks[LM["R_ANKLE"]]

        # Check if user is horizontal on the floor
        vertical_span = abs(sh.y - an.y)
        if vertical_span > 0.40:
            tips.append(("Get into a horizontal plank position on the floor.", "plank_posture"))
            rep_state.is_holding = False
            return tips

        is_sagging = False
        is_piked = False

        if hip_angle < 155:
            mid_y = (sh.y + kn.y) / 2.0
            if hp.y > mid_y:
                tips.append(("Lift your hips slightly! Don't let your lower back sag.", "hip_sag"))
                is_sagging = True
            else:
                tips.append(("Lower your hips! Keep your body in a straight line.", "hip_pike"))
                is_piked = True
        elif hip_angle > 185:
            tips.append(("Straighten your core. Don't arch your back.", "hip_arch"))
            is_sagging = True

        is_knee_bent = False
        if knee_angle < 145:
            tips.append(("Keep your knees straight and engage your thighs.", "knee_bend"))
            is_knee_bent = True

        # Valid hold only when posture is clean
        rep_state.is_holding = not (is_sagging or is_piked or is_knee_bent)
        return tips

    # ------------------------------------------------------------------
    # DTW evaluation
    # ------------------------------------------------------------------
    @staticmethod
    def evaluate_form(
        user_seq: np.ndarray, ref_seq: np.ndarray, threshold: float = 25.0, lang: str = "ar",
    ) -> FormEvaluation:
        user = np.atleast_2d(np.asarray(user_seq, dtype=np.float64))
        ref = np.atleast_2d(np.asarray(ref_seq, dtype=np.float64))
        if user.shape[0] < 2 or ref.shape[0] < 2:
            msg = "Not enough movement frames captured." if lang == "en" else "لم يتم التقاط كادرات كافية للحركة."
            return FormEvaluation(float("inf"), 0, False, threshold, msg)

        dist, _ = fastdtw(user, ref, dist=euclidean)
        # Normalize by max length AND by square root of features to represent average degrees deviation per joint
        norm = dist / (max(user.shape[0], ref.shape[0]) * np.sqrt(user.shape[1]))
        
        # Transparent, explainable biomechanical scoring:
        # Every 1 deg of average joint deviation across the cycle deducts 1.6% from 100%:
        # Average deviation of 3-5 deg gives 92% - 95% (Flawless execution)
        # Average deviation of 8-10 deg gives 84% - 87% (Optimal execution)
        # Average deviation of 15 deg gives 76% (Acceptable execution)
        # Average deviation of 25 deg gives 60% (Pass threshold)
        # Average deviation > 35 deg drops below 50%
        score = float(np.clip(100.0 - 1.6 * norm, 10.0, 100.0))
        ok = score >= 60.0

        if score >= 82:
            fb_ar = f"أداء متقن وممتاز! توافق حركي عالي بدرجة {score:.0f}%."
            fb_en = f"Optimal form! Score {score:.0f}%. High kinematic alignment."
        elif score >= 60:
            fb_ar = f"أداء جيد جداً بدرجة {score:.0f}%، حافظ على ثبات المفصل والتحكم في السرعة."
            fb_en = f"Good form, score {score:.0f}%, keep joints stabilized."
        else:
            fb_ar = f"يحتاج لتصحيح المسار، الدرجة {score:.0f}%. ركز على المدى الحركي الكامل وتثبيت الجذع."
            fb_en = f"Form needs correction, score {score:.0f}%. Focus on full range of motion."

        fb = fb_ar if lang == "ar" else fb_en
        return FormEvaluation(norm, score, ok, threshold, fb)

    # ------------------------------------------------------------------
    # Rep state machine
    # ------------------------------------------------------------------
    def _update_rep(
        self, state: RepetitionState, angle: float,
        angles: dict, config: dict, ref_seq, evals: list,
        exercise_type: ExerciseType, dt_sec: float = 0.033,
    ) -> bool:
        """Returns True if a new rep was just completed."""
        if exercise_type == ExerciseType.PLANK:
            if state.is_holding:
                was_holding = (state.phase == "HOLDING")
                state.phase = "HOLDING"
                state.hold_seconds += dt_sec
                # Smooth progress percentage based on 60-second target
                state.progress_pct = float(np.clip((state.hold_seconds / 60.0) * 100.0, 0.0, 100.0))
                state.rep_count = int(state.hold_seconds)

                sec = int(state.hold_seconds)
                if sec > 0 and sec % 10 == 0 and not getattr(state, f"_ann_{sec}", False):
                    setattr(state, f"_ann_{sec}", True)
                    if f"plank_{sec}" in VOICE_PHRASES:
                        self.audio.say_phrase(f"plank_{sec}", cooldown=3.0, key=f"plank_{sec}")
                    else:
                        self.audio.say_phrase("plank_10", cooldown=3.0, key="plank_time", sec=sec)
                elif not was_holding and state.hold_seconds < 1.0:
                    self.audio.say_phrase("plank_start", cooldown=4.0, key="plank_start")
            else:
                if state.phase == "HOLDING":
                    self.audio.say_phrase("plank_pause", cooldown=3.5, key="plank_pause")
                state.phase = "PAUSED"
            return False

        start = config["start"]
        bottom = config["bottom"]
        vec = list(angles.values())
        range_deg = abs(start - bottom) or 1.0

        state.max_angle = max(state.max_angle, angle)

        # Smooth, continuous Range of Motion depth (0% at starting point, 100% at peak contraction/bottom depth)
        target_rom = float(np.clip(((start - angle) / range_deg) * 100.0, 0.0, 100.0))
        # Continuous exponential smoothing for fluid visual feedback without jerky 50% split halves
        state.progress_pct = float(np.clip(0.60 * state.progress_pct + 0.40 * target_rom, 0.0, 100.0))

        if state.phase == "idle":
            if angle < start - 5:
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
            if angle >= start - 4:
                state.rep_count += 1
                state.phase = "idle"

                if ref_seq is not None and len(state.angle_sequence) > 3:
                    ev = self.evaluate_form(
                        np.array(state.angle_sequence), ref_seq, self.dtw_threshold, lang=self.lang,
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
            self.audio.say_phrase("welcome_front", cooldown=0)
        else:
            self.audio.say_phrase("welcome_side", cooldown=0)

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

                        # For Plank: evaluate posture on every frame so is_holding is real-time
                        if exercise_type == ExerciseType.PLANK:
                            tips = self.coach_plank(lm, angles, active_side, rep_state)
                            if tips:
                                tip_text = self.get_tip_text(tips[0])
                                tip_display_text = tip_text
                                self.audio.say(tip_text, cooldown=4.0, key=tips[0][1])
                            else:
                                tip_display_text = ""

                        # Update rep counter / hold seconds
                        new_rep = self._update_rep(
                            rep_state, p_angle, angles,
                            config, reference_sequence, evals, exercise_type,
                            dt_sec=1.0 / fps
                        )

                        if new_rep:
                            rep_num = rep_state.rep_count
                            latest_eval = evals[-1] if evals else None
                            
                            # Speak realistic encouraging count only after every 2 reps (2, 4, 6, 8, 10...)
                            if rep_num % 2 == 0:
                                if rep_num in (2, 4, 6, 8, 10):
                                    self.audio.say_phrase(f"rep_{rep_num}", cooldown=0.1, key="count")
                                else:
                                    self.audio.say_phrase("rep_even", cooldown=0.1, key="count", num=rep_num)
                            elif latest_eval and latest_eval.form_score < 55:
                                # Speak immediate correction if odd rep had flawed execution
                                self.audio.say_phrase("rep_mistake", cooldown=2.0, key="mistake_rep")

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
                                pass  # Evaluated continuously above
                            else:
                                tips = []

                            # Orientation warning
                            if exercise_type in (ExerciseType.SHOULDER_PRESS,):
                                if orientation == "side":
                                    tips.append(("Please face the camera directly so that both of your knees and hips can be tracked from the front.", "orientation_front"))
                            else:
                                if orientation == "front":
                                    tips.append(("Please stand sideways to the camera so that your arm and joints are visible from the side.", "orientation_side"))

                            # Speak the most important tip
                            if tips:
                                msg, key = tips[0]
                                tip_text = self.get_tip_text((msg, key))
                                self.audio.say(tip_text, cooldown=5.0, key=key)
                                tip_display_text = tip_text
                            else:
                                tip_display_text = ""
                                # If the form is correct during the rep, speak positive reinforcement
                                if rep_state.phase in ("descending", "ascending"):
                                    pos_list = POSITIVE_PHRASES.get(self.lang, POSITIVE_PHRASES["en"])
                                    pos_msg = pos_list[frame_n % len(pos_list)]
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
            if exercise_type == ExerciseType.PLANK:
                total_sec = int(rep_state.hold_seconds)
                self.audio.say_phrase("session_done_plank", cooldown=0, total=total_sec)
            else:
                total = rep_state.rep_count
                self.audio.say_phrase("session_done_reps", cooldown=0, total=total)
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
            self.audio.say_phrase("welcome_front", cooldown=0)
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
            self.audio.say_phrase("welcome_side", cooldown=0)

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
        panel_h = 240 if tip_text else 195
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

        # Rep / Hold counter (large)
        if exercise_type == ExerciseType.PLANK:
            mins = int(rep_state.hold_seconds) // 60
            secs = int(rep_state.hold_seconds) % 60
            counter_str = f"Hold: {mins:02d}:{secs:02d}"
            counter_color = green if rep_state.is_holding else yellow
        else:
            counter_str = f"Reps: {rep_state.rep_count}"
            counter_color = green

        cv2.putText(
            frame, counter_str,
            (16, y), font, 0.85, counter_color, 2, cv2.LINE_AA,
        )

        # Phase / Status
        phase = rep_state.phase
        if exercise_type == ExerciseType.PLANK:
            if rep_state.is_holding:
                pcolor, plabel = green, "HOLDING"
            else:
                pcolor, plabel = yellow, "PAUSED"
        elif phase == "descending":
            pcolor, plabel = yellow, "WORKING"
        elif phase == "ascending":
            pcolor, plabel = cyan, "RETURNING"
        else:
            pcolor, plabel = gray, "READY"
        cv2.putText(frame, plabel, (230, y), font, 0.6, pcolor, 1, cv2.LINE_AA)
        y += 24

        # Workout Progress Bar ("زي بار للتمرينه والعدات")
        bar_w = 340
        bar_h = 10
        cv2.rectangle(frame, (16, y), (16 + bar_w, y + bar_h), (45, 45, 45), -1)
        fill_w = int(bar_w * (np.clip(rep_state.progress_pct, 0.0, 100.0) / 100.0))
        bar_c = green if (exercise_type == ExerciseType.PLANK and rep_state.is_holding) or rep_state.progress_pct >= 95 else cyan
        if fill_w > 0:
            cv2.rectangle(frame, (16, y), (16 + fill_w, y + bar_h), bar_c, -1)
        cv2.rectangle(frame, (16, y), (16 + bar_w, y + bar_h), (80, 80, 80), 1)
        # Percentage text
        pct_text = f"{int(rep_state.progress_pct)}%"
        cv2.putText(frame, pct_text, (16 + bar_w + 8, y + 9), font, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
        y += 22

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

        # Big rep / hold counter bottom-right
        if exercise_type == ExerciseType.PLANK:
            total_txt = f"{int(rep_state.hold_seconds)}s"
        else:
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
    """
    Generate an anatomically accurate biomechanical reference trajectory for DTW comparison.
    Cosine bell curve represents a smooth, controlled repetition (concentric + eccentric).
    t ranges from 0 to 2*pi so that bell starts at 1, drops to 0 at mid-rep (depth), and returns to 1.
    """
    t = np.linspace(0, 2 * np.pi, n)
    bell = 0.5 + 0.5 * np.cos(t)  # 1.0 at start, 0.0 at peak contraction/bottom, 1.0 at finish

    if ex == ExerciseType.BICEP_CURL:
        # Features: [elbow, shoulder]
        # Elbow curls from 150 deg down to 55 deg and returns to 150 deg
        elbow = 55.0 + 95.0 * bell
        # Shoulder remains stable and pinned to torso at ~15 deg
        shoulder = np.full(n, 15.0)
        return np.column_stack([elbow, shoulder])

    elif ex == ExerciseType.SQUAT:
        # Features: [hip, knee, ankle]
        # Hip bends from 170 deg standing down to 90 deg and returns
        hip = 90.0 + 80.0 * bell
        # Knee flexes from 165 deg down to 100 deg (parallel) and returns
        knee = 100.0 + 65.0 * bell
        # Ankle dorsiflexes from 85 deg down to 75 deg and returns
        ankle = 75.0 + 10.0 * bell
        return np.column_stack([hip, knee, ankle])

    elif ex == ExerciseType.LUNGE:
        # Features: [hip, knee, ankle]
        hip = 100.0 + 65.0 * bell
        knee = 95.0 + 70.0 * bell
        ankle = 75.0 + 10.0 * bell
        return np.column_stack([hip, knee, ankle])

    elif ex == ExerciseType.PUSH_UP:
        # Features: [elbow, shoulder, hip]
        # Elbow flexes from 160 deg (extended) down to 85 deg (chest near floor) and returns
        elbow = 85.0 + 75.0 * bell
        # Shoulder flexes from 75 deg down to 45 deg and returns
        shoulder = 45.0 + 30.0 * bell
        # Hip remains rigid in plank posture at ~172 deg
        hip = np.full(n, 172.0)
        return np.column_stack([elbow, shoulder, hip])

    elif ex == ExerciseType.SHOULDER_PRESS:
        # Features: [elbow, shoulder]
        # Start at ears (elbow ~95 deg), press overhead (elbow ~165 deg), and return to ears
        inv_bell = 0.5 - 0.5 * np.cos(t)  # 0 at start, 1 at overhead peak, 0 at finish
        elbow = 95.0 + 70.0 * inv_bell
        shoulder = 90.0 + 65.0 * inv_bell
        return np.column_stack([elbow, shoulder])

    elif ex == ExerciseType.DIPS:
        # Features: [elbow, shoulder, hip]
        elbow = 90.0 + 65.0 * bell
        shoulder = 30.0 + 30.0 * (1.0 - bell)
        hip = np.full(n, 95.0)
        return np.column_stack([elbow, shoulder, hip])

    elif ex == ExerciseType.SUPERMAN:
        # Features: [hip, shoulder]
        hip = 162.0 + 16.0 * bell
        shoulder = 155.0 + 20.0 * bell
        return np.column_stack([hip, shoulder])

    elif ex == ExerciseType.PLANK:
        # Features: [hip, shoulder, knee]
        hip = np.full(n, 175.0)
        shoulder = np.full(n, 90.0)
        knee = np.full(n, 175.0)
        return np.column_stack([hip, shoulder, knee])

    else:
        return np.column_stack([60.0 + 90.0 * bell, np.full(n, 20.0)])


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
    p.add_argument("--lang", choices=["ar", "en"], default="ar", help="Voice coach language (ar or en)")
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
    analyzer = PoseAnalyzer(dtw_threshold=args.threshold, enable_audio=not args.no_audio, lang=args.lang)

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
