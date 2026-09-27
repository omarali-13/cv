import sys
import os
import time
import base64
from typing import Optional, Dict, Any, List
import numpy as np
import cv2

# Add parent directory to path so we can import cv_engine
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cv_engine import (
    PoseAnalyzer, ExerciseType, RepetitionState, FormEvaluation, LM,
    VOICE_PHRASES, COACH_TIPS_I18N
)

class LandmarkMock:
    """Mock class matching Mediapipe landmark structure for PoseAnalyzer."""
    def __init__(self, x: float, y: float, z: float, visibility: float = 1.0):
        self.x = x
        self.y = y
        self.z = z
        self.visibility = visibility

class RemoteSession:
    """Manages repetition state, angle kinematics, and form coaching for a WebSocket/API session."""

    def __init__(self, exercise: str, threshold: float = 25.0, lang: str = "ar"):
        self.exercise_type = ExerciseType(exercise)
        self.lang = lang
        self.analyzer = PoseAnalyzer(dtw_threshold=threshold, enable_audio=False, lang=lang)
        self.rep_state = RepetitionState()
        self.config = self.analyzer._REP_CONFIG[self.exercise_type]
        self.evals: List[FormEvaluation] = []
        self.latest_eval: Optional[FormEvaluation] = None
        self.active_side = "left"
        self.orientation = "unknown"
        self.reference_sequence: Optional[np.ndarray] = None
        self.frame_n = 0
        self.last_ts = time.time()
        self._last_announced_sec = 0

    def set_reference(self, reference_seq: np.ndarray):
        self.reference_sequence = reference_seq

    def set_language(self, lang: str):
        if lang in ("ar", "en"):
            self.lang = lang
            self.analyzer.set_language(lang)

    def reset(self):
        """Reset session counters and state for a fresh workout."""
        self.rep_state = RepetitionState()
        self.evals = []
        self.latest_eval = None
        self.frame_n = 0
        self.last_ts = time.time()
        self._last_announced_sec = 0

    def process_image(self, base64_data: str) -> Dict[str, Any]:
        """Decode base64 video frame and run landmark extraction + posture coaching."""
        self.frame_n += 1
        try:
            # Strip header if present
            if "," in base64_data:
                base64_data = base64_data.split(",", 1)[1]
            img_bytes = base64.b64decode(base64_data)
            nparr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is None:
                return {"detected": False, "error": "Failed to decode image frame"}
        except Exception as e:
            return {"detected": False, "error": f"Invalid image data: {e}"}

        ts_ms = int(time.time() * 1000)
        raw_lm = self.analyzer.extract_landmarks(frame, ts_ms)
        if raw_lm is None:
            return {
                "detected": False,
                "rep_count": self.rep_state.rep_count,
                "hold_seconds": round(self.rep_state.hold_seconds, 1),
                "is_holding": self.rep_state.is_holding,
                "progress_pct": round(self.rep_state.progress_pct, 1),
                "phase": self.rep_state.phase,
                "active_side": self.active_side,
                "orientation": self.orientation,
            }

        # Convert for internal processing
        landmarks = [
            LandmarkMock(p.x, p.y, p.z, getattr(p, "visibility", 1.0))
            for p in raw_lm
        ]

        result = self._evaluate_landmarks(landmarks)
        result["detected"] = True
        
        # Serialize landmarks so frontend can render glowing HUD skeleton
        result["landmarks"] = [
            {"x": float(p.x), "y": float(p.y), "z": float(p.z), "visibility": float(getattr(p, "visibility", 1.0))}
            for p in raw_lm
        ]
        return result

    def process_frame(self, landmark_list: List[Dict[str, float]]) -> Dict[str, Any]:
        """Process pre-extracted landmarks list from client-side tracker."""
        self.frame_n += 1
        landmarks = [
            LandmarkMock(lm["x"], lm["y"], lm.get("z", 0.0), lm.get("visibility", 1.0))
            for lm in landmark_list
        ]
        result = self._evaluate_landmarks(landmarks)
        result["detected"] = True
        return result

    def _evaluate_landmarks(self, landmarks: List[LandmarkMock]) -> Dict[str, Any]:
        """Core biomechanics analysis, angle extraction, and coaching generation."""
        self.orientation = self.analyzer.detect_orientation(landmarks)

        # Extract joint angles
        angles: Optional[dict[str, float]] = None
        left_angles: Optional[dict[str, float]] = None
        right_angles: Optional[dict[str, float]] = None

        is_front_view = self.exercise_type in (ExerciseType.SHOULDER_PRESS,)

        if is_front_view:
            self.active_side = "both"
            left_angles = self.analyzer.extract_features(landmarks, self.exercise_type, "left")
            right_angles = self.analyzer.extract_features(landmarks, self.exercise_type, "right")

            if left_angles is not None and right_angles is not None:
                angles = {
                    "elbow": (left_angles["elbow"] + right_angles["elbow"]) / 2.0,
                    "shoulder": (left_angles["shoulder"] + right_angles["shoulder"]) / 2.0,
                }
        else:
            self.active_side = self.analyzer.detect_active_side(landmarks, self.exercise_type)
            angles = self.analyzer.extract_features(landmarks, self.exercise_type, self.active_side)

        now = time.time()
        dt_sec = now - self.last_ts
        if dt_sec <= 0 or dt_sec > 0.5:
            dt_sec = 0.033
        self.last_ts = now

        coaching_tip_ar = ""
        coaching_tip_en = ""
        tip_key = ""
        announcement_ar = ""
        announcement_en = ""
        new_rep = False

        if angles is not None:
            primary = self.config["primary"]
            p_angle = angles.get(primary, 180.0)

            # Plank posture check
            tips = []
            if self.exercise_type == ExerciseType.PLANK:
                tips = self.analyzer.coach_plank(landmarks, angles, self.active_side, self.rep_state)

            # State machine update
            prev_holding = (self.rep_state.phase == "HOLDING")
            new_rep = self.analyzer._update_rep(
                self.rep_state, p_angle, angles,
                self.config, self.reference_sequence, self.evals, self.exercise_type,
                dt_sec=dt_sec
            )

            # Handle plank voice milestones
            if self.exercise_type == ExerciseType.PLANK:
                sec = int(self.rep_state.hold_seconds)
                if self.rep_state.is_holding:
                    if not prev_holding and self.rep_state.hold_seconds < 1.0:
                        announcement_ar = VOICE_PHRASES["plank_start"]["ar"]
                        announcement_en = VOICE_PHRASES["plank_start"]["en"]
                    elif sec > 0 and sec % 10 == 0 and sec != self._last_announced_sec:
                        self._last_announced_sec = sec
                        k = f"plank_{sec}" if f"plank_{sec}" in VOICE_PHRASES else "plank_10"
                        announcement_ar = VOICE_PHRASES[k]["ar"]
                        announcement_en = VOICE_PHRASES[k]["en"]
                else:
                    if prev_holding:
                        announcement_ar = VOICE_PHRASES["plank_pause"]["ar"]
                        announcement_en = VOICE_PHRASES["plank_pause"]["en"]

            # Handle rep completion voice announcement: speak every 2 reps (2, 4, 6, 8, 10...)
            if new_rep:
                rep_n = self.rep_state.rep_count
                if self.evals:
                    self.latest_eval = self.evals[-1]
                
                if rep_n % 2 == 0:
                    k = f"rep_{rep_n}" if f"rep_{rep_n}" in VOICE_PHRASES else "rep_even"
                    announcement_ar = VOICE_PHRASES[k]["ar"].format(num=rep_n)
                    announcement_en = VOICE_PHRASES[k]["en"].format(num=rep_n)
                elif self.latest_eval and self.latest_eval.form_score < 55:
                    announcement_ar = VOICE_PHRASES["rep_mistake"]["ar"]
                    announcement_en = VOICE_PHRASES["rep_mistake"]["en"]

            # Contextual coaching tips
            if self.exercise_type == ExerciseType.BICEP_CURL:
                tips = self.analyzer.coach_bicep_curl(landmarks, angles, self.active_side, self.rep_state)
            elif self.exercise_type == ExerciseType.SQUAT:
                tips = self.analyzer.coach_squat(landmarks, angles, self.active_side, self.rep_state)
            elif self.exercise_type == ExerciseType.SHOULDER_PRESS:
                tips = self.analyzer.coach_shoulder_press(landmarks, angles, self.active_side, self.rep_state)
            elif self.exercise_type == ExerciseType.PUSH_UP:
                tips = self.analyzer.coach_push_up(landmarks, angles, self.active_side, self.rep_state)
            elif self.exercise_type == ExerciseType.LUNGE:
                tips = self.analyzer.coach_lunge(landmarks, angles, self.active_side, self.rep_state)
            elif self.exercise_type == ExerciseType.SUPERMAN:
                tips = self.analyzer.coach_superman(landmarks, angles, self.active_side, self.rep_state)
            elif self.exercise_type == ExerciseType.DIPS:
                tips = self.analyzer.coach_dips(landmarks, angles, self.active_side, self.rep_state)

            # Orientation checks
            if is_front_view:
                if self.orientation == "side":
                    tips.append(("Please face the camera directly.", "orientation_front"))
            else:
                if self.orientation == "front":
                    tips.append(("Please stand sideways to the camera.", "orientation_side"))

            faulty_joint = ""
            if tips:
                tip_info = self.analyzer.get_coaching_dict(tips[0])
                tip_key = tip_info["key"]
                coaching_tip_ar = tip_info["ar"]
                coaching_tip_en = tip_info["en"]

                # Identify specific joint for visual canvas warning
                if any(x in tip_key for x in ("knee", "depth", "lunge")):
                    faulty_joint = "knee"
                elif any(x in tip_key for x in ("hip", "sag", "pike", "arch", "lean")):
                    faulty_joint = "hip"
                elif any(x in tip_key for x in ("elbow", "swing", "curl", "extend", "dips")):
                    faulty_joint = "elbow"
                elif any(x in tip_key for x in ("shoulder", "press", "asym")):
                    faulty_joint = "shoulder"
                else:
                    faulty_joint = "body"

                # Always speak correction if user made a form mistake
                if not announcement_ar and (time.time() - getattr(self, "_last_mistake_spoken", 0) > 4.0):
                    self._last_mistake_spoken = time.time()
                    announcement_ar = coaching_tip_ar
                    announcement_en = coaching_tip_en

        active_tip = coaching_tip_ar if self.lang == "ar" else coaching_tip_en
        active_ann = announcement_ar if self.lang == "ar" else announcement_en

        response = {
            "rep_count": self.rep_state.rep_count,
            "hold_seconds": round(self.rep_state.hold_seconds, 1),
            "is_holding": self.rep_state.is_holding,
            "progress_pct": round(self.rep_state.progress_pct, 1),
            "phase": self.rep_state.phase,
            "active_side": self.active_side,
            "angles": angles,
            "orientation": self.orientation,
            "coaching_tip": active_tip,
            "coaching_tip_ar": coaching_tip_ar,
            "coaching_tip_en": coaching_tip_en,
            "tip_key": tip_key,
            "announcement": active_ann,
            "announcement_ar": announcement_ar,
            "announcement_en": announcement_en,
            "faulty_joint": faulty_joint,
            "has_mistake": bool(faulty_joint),
            "new_rep": new_rep,
        }

        if new_rep and self.latest_eval:
            response["latest_score"] = round(self.latest_eval.form_score, 1)
            response["latest_feedback"] = self.latest_eval.feedback

        return response
