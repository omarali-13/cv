import os
import sys
import argparse
import numpy as np
import cv2
import mediapipe as mp

# Add parent directory to path to enable cv_engine import
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from cv_engine import PoseAnalyzer, ExerciseType, LM, _ensure_model
from app.dtw_evaluator import save_exercise_reference

# Mediapipe configuration
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

def extract_reference_sequence(video_path: str, exercise_type: ExerciseType) -> np.ndarray:
    """Analyze expert video and return sequence of landmark angles for DTW reference."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")
        
    model_path = _ensure_model()
    opts = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=False,
    )
    
    landmarker = PoseLandmarker.create_from_options(opts)
    analyzer = PoseAnalyzer(enable_audio=False)
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    dt = int(1000 / fps)
    ts = 0
    frame_n = 0
    
    sequence = []
    active_side = "left"
    
    print(f"\nProcessing {video_path} for {exercise_type.value}...")
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_n += 1
            ts += dt
            
            # Extract landmarks
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            res = landmarker.detect_for_video(mp_img, ts)
            
            if not res.pose_landmarks:
                continue
                
            lm = res.pose_landmarks[0]
            
            # Extract features depending on the exercise view
            is_front_view = exercise_type in (ExerciseType.SHOULDER_PRESS,)
            
            if is_front_view:
                left_angles = analyzer.extract_features(lm, exercise_type, "left")
                right_angles = analyzer.extract_features(lm, exercise_type, "right")
                
                if left_angles and right_angles:
                    vec = [
                        (left_angles["elbow"] + right_angles["elbow"]) / 2.0,
                        (left_angles["shoulder"] + right_angles["shoulder"]) / 2.0,
                    ]
                    sequence.append(vec)
            else:
                # Detect active side once at frame 10 or auto-detect
                if frame_n == 10 or active_side == "left":
                    active_side = analyzer.detect_active_side(lm, exercise_type)
                    
                angles = analyzer.extract_features(lm, exercise_type, active_side)
                if angles:
                    vec = list(angles.values())
                    sequence.append(vec)
                    
        print(f"Extraction completed. Processed {frame_n} frames, captured {len(sequence)} valid pose states.")
        return np.array(sequence, dtype=np.float64)
        
    finally:
        cap.release()
        landmarker.close()

def main():
    p = argparse.ArgumentParser(description="Extract expert references for DTW form validation")
    p.add_argument("--video", required=True, help="Path to the expert video file (.mp4)")
    p.add_argument(
        "--exercise", 
        required=True, 
        choices=["bicep_curl", "squat", "lunge", "push_up", "superman", "shoulder_press", "dips", "plank"],
        help="Exercise type"
    )
    args = p.parse_args()
    
    ex_type = ExerciseType(args.exercise)
    
    if not os.path.exists(args.video):
        print(f"[ERROR] Video file not found: {args.video}")
        sys.exit(1)
        
    try:
        seq = extract_reference_sequence(args.video, ex_type)
        if len(seq) == 0:
            print("[ERROR] No valid landmarks extracted from video. Try another video file.")
            sys.exit(1)
            
        success = save_exercise_reference(ex_type, seq)
        if success:
            print(f"[SUCCESS] Reference template registered and saved to app/references/{ex_type.value}_reference.npy")
        else:
            print("[ERROR] Failed to save reference template.")
            
    except Exception as e:
        print(f"[ERROR] An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
