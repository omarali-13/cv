import os
import sys
import uuid
import logging
from typing import Dict, Any, List
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add parent directory to path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.exercises import RemoteSession, ExerciseType
from app.dtw_evaluator import get_exercise_reference
from app.database import save_workout_session

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cv_fit_api")

app = FastAPI(
    title="CV for Fit API",
    description="Real-time Pose Analytics & Voice Coaching API using Computer Vision and fastDTW",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic schemas
class SessionSaveRequest(BaseModel):
    session_id: str
    user_id: str
    exercise_type: str
    rep_count: int
    avg_score: float

class ExerciseInfo(BaseModel):
    id: str
    name: str
    type: str
    view: str
    muscle: str

@app.get("/")
def read_root():
    return {"message": "Welcome to CV for Fit Backend Engine API", "status": "running"}

@app.get("/api/exercises", response_model=List[ExerciseInfo])
def get_supported_exercises():
    """List all 8 exercises supported by the CV engine."""
    return [
        ExerciseInfo(id="bicep_curl", name="Bicep Curl", type="reps", view="side", muscle="Biceps"),
        ExerciseInfo(id="squat", name="Squat", type="reps", view="front", muscle="Quads/Glutes"),
        ExerciseInfo(id="lunge", name="Lunge", type="reps", view="side", muscle="Quads/Hamstrings"),
        ExerciseInfo(id="push_up", name="Push Up", type="reps", view="side", muscle="Chest/Triceps"),
        ExerciseInfo(id="superman", name="Superman", type="reps", view="side", muscle="Lower Back"),
        ExerciseInfo(id="shoulder_press", name="Shoulder Press", type="reps", view="front", muscle="Shoulders"),
        ExerciseInfo(id="dips", name="Chair Dips", type="reps", view="side", muscle="Triceps"),
        ExerciseInfo(id="plank", name="Plank", type="hold", view="side", muscle="Core"),
    ]

@app.post("/api/session/complete")
def complete_session(request: SessionSaveRequest):
    """Save the completed session details to Supabase or SQLite database."""
    success = save_workout_session(
        session_id=request.session_id,
        user_id=request.user_id,
        exercise=request.exercise_type,
        reps=request.rep_count,
        avg_score=request.avg_score
    )
    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save workout session details to the database."
        )
    return {"status": "success", "message": "Workout session logged successfully."}

@app.websocket("/ws/workout")
async def websocket_workout_endpoint(
    websocket: WebSocket,
    exercise: str = Query(..., description="Exercise type ID"),
    userId: str = Query("guest", description="User identifier")
):
    """WebSocket endpoint for real-time landmark streaming and feedback."""
    await websocket.accept()
    logger.info(f"WebSocket client connected. User: {userId}, Exercise: {exercise}")
    
    try:
        # Validate exercise type
        ex_type = ExerciseType(exercise)
    except ValueError:
        logger.error(f"Invalid exercise requested: {exercise}")
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Invalid exercise type")
        return

    # Initialize processing session
    session = RemoteSession(exercise)
    # Load expert reference sequence
    ref_seq = get_exercise_reference(ex_type)
    session.set_reference(ref_seq)
    
    session_id = str(uuid.uuid4())
    await websocket.send_json({
        "status": "initialized",
        "session_id": session_id,
        "exercise": exercise,
        "orientation_required": "front" if ex_type in (ExerciseType.SQUAT, ExerciseType.SHOULDER_PRESS) else "side"
    })
    
    try:
        while True:
            # Receive frame landmarks from client
            # Expected format: {"landmarks": [{"x": float, "y": float, "z": float, "visibility": float}]}
            data = await websocket.receive_json()
            landmarks = data.get("landmarks")
            if not landmarks:
                await websocket.send_json({"error": "Missing landmarks key in payload"})
                continue
                
            # Process landmarks using PoseAnalyzer core
            result = session.process_frame(landmarks)
            
            # Send real-time feedback back to client
            await websocket.send_json(result)
            
    except WebSocketDisconnect:
        logger.info(f"WebSocket client disconnected gracefully. Session: {session_id}")
    except Exception as e:
        logger.error(f"Error in WebSocket handler: {e}")
        try:
            await websocket.send_json({"error": "Internal processing error"})
            await websocket.close()
        except:
            pass
