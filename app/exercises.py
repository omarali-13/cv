import sys
import os
from typing import Optional, Dict, Any, List

# Add parent directory to path so we can import cv_engine
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cv_engine import PoseAnalyzer, ExerciseType, RepetitionState, FormEvaluation, LM
import numpy as np

class LandmarkMock:
    """Mock class matching Mediapipe landmark structure for PoseAnalyzer."""
    def __init__(self, x: float, y: float, z: float, visibility: float):
        self.x = x
        self.y = y
        self.z = z
        self.visibility = visibility

class RemoteSession:
    """Manages repetition state and evaluation for an active remote connection."""
    def __init__(self, exercise: str, threshold: float = 25.0):
        self.exercise_type = ExerciseType(exercise)
        self.analyzer = PoseAnalyzer(dtw_threshold=threshold, enable_audio=False)
        self.rep_state = RepetitionState()
        self.config = self.analyzer._REP_CONFIG[self.exercise_type]
        self.evals: List[FormEvaluation] = []
        self.latest_eval: Optional[FormEvaluation] = None
        self.active_side = "left"
        self.orientation = "unknown"
        self.reference_sequence: Optional[np.ndarray] = None
        self.frame_n = 0

    def set_reference(self, reference_seq: np.ndarray):
        self.reference_sequence = reference_seq

    def process_frame(self, landmark_list: List[Dict[str, float]]) -> Dict[str, Any]:
        """Process a single frame from the stream."""
        self.frame_n += 1
        
        # Convert JSON landmarks to mock objects
        landmarks = [
            LandmarkMock(lm["x"], lm["y"], lm["z"], lm.get("visibility", 1.0))
            for lm in landmark_list
        ]
        
        # Detect orientation
        self.orientation = self.analyzer.detect_orientation(landmarks)
        
        # Extract angles
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

        coaching_tip = ""
        new_rep = False
        
        if angles is not None:
            primary = self.config["primary"]
            p_angle = angles.get(primary, 180.0)
            
            # Update rep state
            new_rep = self.analyzer._update_rep(
                self.rep_state, p_angle, angles,
                self.config, self.reference_sequence, self.evals, self.exercise_type
            )
            
            if new_rep and self.evals:
                self.latest_eval = self.evals[-1]

            # Coaching tip rules
            tips = []
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
            elif self.exercise_type == ExerciseType.PLANK:
                tips = self.analyzer.coach_plank(landmarks, angles, self.active_side, self.rep_state)

            # Check orientation warnings
            if is_front_view:
                if self.orientation == "side":
                    tips.append(("Please face the camera directly.", "orientation"))
            else:
                if self.orientation == "front":
                    tips.append(("Please stand sideways to the camera.", "orientation"))

            if tips:
                coaching_tip = tips[0][0] # Get text of primary feedback

        # Prepare response object
        response = {
            "rep_count": self.rep_state.rep_count,
            "phase": self.rep_state.phase,
            "active_side": self.active_side,
            "angles": angles,
            "orientation": self.orientation,
            "coaching_tip": coaching_tip,
            "new_rep": new_rep,
        }
        
        if new_rep and self.latest_eval:
            response["latest_score"] = self.latest_eval.form_score
            response["latest_feedback"] = self.latest_eval.feedback
            
        return response
