import os
import sys
import numpy as np
from typing import Optional

# Add parent directory to path so we can import cv_engine
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cv_engine import ExerciseType, generate_synthetic_reference

REFERENCES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "references")
os.makedirs(REFERENCES_DIR, exist_ok=True)

def get_exercise_reference(exercise_type: ExerciseType) -> np.ndarray:
    """Retrieve template from .npy file or generate dynamic synthetic fallback."""
    ref_filename = f"{exercise_type.value}_reference.npy"
    ref_path = os.path.join(REFERENCES_DIR, ref_filename)
    
    if os.path.exists(ref_path):
        try:
            ref_seq = np.load(ref_path)
            # Validate shape dimensions
            expected_dim = 2 if exercise_type in (ExerciseType.BICEP_CURL, ExerciseType.SUPERMAN, ExerciseType.SHOULDER_PRESS) else 3
            if len(ref_seq.shape) == 2 and ref_seq.shape[1] == expected_dim:
                return ref_seq
            else:
                print(f"[WARNING] Reference template {ref_filename} has shape {ref_seq.shape}, expected (*, {expected_dim}). Generating synthetic fallback.")
        except Exception as e:
            print(f"[ERROR] Failed to load reference {ref_filename}: {e}. Generating synthetic fallback.")
            
    # Synthetic fallback
    return generate_synthetic_reference(exercise_type)

def save_exercise_reference(exercise_type: ExerciseType, sequence: np.ndarray) -> bool:
    """Save reference array to references folder."""
    try:
        ref_filename = f"{exercise_type.value}_reference.npy"
        ref_path = os.path.join(REFERENCES_DIR, ref_filename)
        np.save(ref_path, sequence)
        return True
    except Exception as e:
        print(f"[ERROR] Failed to save reference for {exercise_type.value}: {e}")
        return False
