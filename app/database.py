import os
import sqlite3
import logging
from typing import Optional
from datetime import date
try:
    from supabase import create_client, Client
    _HAS_SUPABASE = True
except ImportError:
    _HAS_SUPABASE = False
    Client = None
    create_client = None

logger = logging.getLogger("cv_fit_api")

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

supabase_client: Optional[Client] = None

if _HAS_SUPABASE and SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
        logger.info("Connected to Supabase client successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize Supabase client: {e}")
else:
    logger.info("Supabase library or credentials not found. Using local SQLite database (cv_fit_local.db).")

# Exercise ID mapping from user SQL database
EXERCISE_ID_MAP = {
    "push_up": 1,
    "squat": 3,
    "shoulder_press": 6,
    "bicep_curl": 8,
    "dips": 9,
    "plank": 10,
    "lunge": 11,
    "superman": 12,
}

# Local SQLite fallback setup
LOCAL_DB_PATH = "cv_fit_local.db"

def init_local_db():
    conn = sqlite3.connect(LOCAL_DB_PATH)
    cursor = conn.cursor()
    # Create tables matching user schema
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS pose_analysis (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            exercise_name TEXT,
            score INTEGER,
            feedback TEXT,
            date DATE
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS workout_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            exercise_id INTEGER,
            date DATE,
            sets_done INTEGER,
            reps_done INTEGER,
            notes TEXT
        )
    """)
    conn.commit()
    conn.close()

init_local_db()

def save_workout_session(session_id: str, user_id: str, exercise: str, reps: int, avg_score: float) -> bool:
    """Save workout logs and pose analysis to SQL tables: pose_analysis and workout_logs."""
    # Convert user_id safely to integer
    try:
        u_id = int(user_id)
    except ValueError:
        u_id = 1  # Fallback to Ahmed Ali (id=1) in sample data
        
    ex_id = EXERCISE_ID_MAP.get(exercise, 1) # Default to push-up if not found
    current_date = date.today().isoformat()
    feedback_text = f"Completed {reps} reps with average score of {avg_score:.1f}/100."
    
    pose_data = {
        "user_id": u_id,
        "exercise_name": exercise.replace("_", " ").title(),
        "score": int(avg_score),
        "feedback": feedback_text,
        "date": current_date
    }
    
    log_data = {
        "user_id": u_id,
        "exercise_id": ex_id,
        "date": current_date,
        "sets_done": 1,
        "reps_done": reps,
        "notes": feedback_text
    }
    
    saved_supabase = False
    
    if supabase_client:
        try:
            # Insert into pose_analysis
            supabase_client.table("pose_analysis").insert(pose_data).execute()
            # Insert into workout_logs
            supabase_client.table("workout_logs").insert(log_data).execute()
            logger.info(f"Workout data saved to Supabase (pose_analysis and workout_logs) for user {u_id}.")
            saved_supabase = True
        except Exception as e:
            logger.error(f"Failed to save to Supabase: {e}. Attempting SQLite fallback.")
            
    if not saved_supabase:
        # SQLite fallback
        try:
            conn = sqlite3.connect(LOCAL_DB_PATH)
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO pose_analysis (user_id, exercise_name, score, feedback, date) VALUES (?, ?, ?, ?, ?)",
                (u_id, pose_data["exercise_name"], pose_data["score"], pose_data["feedback"], current_date)
            )
            cursor.execute(
                "INSERT INTO workout_logs (user_id, exercise_id, date, sets_done, reps_done, notes) VALUES (?, ?, ?, ?, ?, ?)",
                (u_id, log_data["exercise_id"], current_date, 1, log_data["reps_done"], log_data["notes"])
            )
            conn.commit()
            conn.close()
            logger.info(f"Workout data saved to local SQLite (pose_analysis and workout_logs) for user {u_id}.")
            return True
        except Exception as sqlite_err:
            logger.error(f"Failed to save to SQLite: {sqlite_err}")
            return False
            
    return True
