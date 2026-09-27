import os
import sys
import uuid
import time
import shutil
import tempfile
import threading
import logging
from typing import Dict, Any, List, Optional
import numpy as np
import cv2

from fastapi import (
    FastAPI, WebSocket, WebSocketDisconnect, HTTPException,
    Query, Form, File, UploadFile, status
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

# Add parent directory to path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.exercises import RemoteSession, ExerciseType, LandmarkMock
from app.dtw_evaluator import get_exercise_reference
from app.database import save_workout_session
from cv_engine import COACH_TIPS_I18N

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cv_fit_api")

app = FastAPI(
    title="CV for Fit - AI Biomechanics Engine",
    description="Real-time Pose Analytics, Biomechanics Telemetry, and Video Analysis API",
    version="2.5.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# =====================================================================
# Direct OpenCV Camera Manager (MJPEG Stream)
# =====================================================================
class CameraManager:
    """Manages direct local webcam capture, pose analysis, and rock-solid MJPEG web streaming."""
    def __init__(self):
        self.lock = threading.Lock()
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_running = False
        self.current_exercise = ExerciseType.BICEP_CURL
        self.lang = "ar"
        self.session: Optional[RemoteSession] = None
        self.latest_jpeg: Optional[bytes] = None
        self.worker_thread: Optional[threading.Thread] = None
        self.rep_count = 0
        self.hold_seconds = 0.0
        self.is_holding = False
        self.progress_pct = 0.0
        self.phase = "IDLE"
        self.angles: Dict[str, float] = {}
        self.active_side = "left"
        self.tip_text = ""
        self.latest_score: Optional[float] = None
        self.latest_feedback = ""
        self.latest_landmarks: Optional[List[Dict[str, float]]] = None
        self.faulty_joint = ""
        self.has_mistake = False
        self.pending_announcement = ""

    def start(self):
        with self.lock:
            if self.is_running:
                return
            self.is_running = True
            self._init_session()
            self.worker_thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.worker_thread.start()

    def _init_session(self):
        self.session = RemoteSession(self.current_exercise.value, lang=self.lang)
        ref_seq = get_exercise_reference(self.current_exercise)
        self.session.set_reference(ref_seq)

    def set_exercise(self, ex_str: str):
        try:
            ex = ExerciseType(ex_str)
            with self.lock:
                self.current_exercise = ex
                self._init_session()
                self.rep_count = 0
                self.hold_seconds = 0.0
                self.progress_pct = 0.0
                self.phase = "IDLE"
                self.angles = {}
                self.tip_text = ""
                self.latest_landmarks = None
                self.faulty_joint = ""
                self.has_mistake = False
                self.pending_announcement = ""
        except Exception as e:
            logger.error(f"Error switching exercise in CameraManager: {e}")

    def set_language(self, lang_str: str):
        if lang_str in ("ar", "en"):
            with self.lock:
                self.lang = lang_str
                if self.session:
                    self.session.set_language(lang_str)

    def reset(self):
        with self.lock:
            if self.session:
                self.session.reset()
            self.rep_count = 0
            self.hold_seconds = 0.0
            self.progress_pct = 0.0
            self.phase = "IDLE"
            self.latest_score = None
            self.latest_feedback = ""
            self.latest_landmarks = None
            self.faulty_joint = ""
            self.has_mistake = False
            self.pending_announcement = ""

    def get_status(self) -> Dict[str, Any]:
        with self.lock:
            ann = self.pending_announcement
            self.pending_announcement = ""  # Consumed by client
            return {
                "exercise": self.current_exercise.value,
                "lang": self.lang,
                "rep_count": self.rep_count,
                "hold_seconds": round(self.hold_seconds, 1),
                "is_holding": self.is_holding,
                "progress_pct": round(self.progress_pct, 1),
                "phase": self.phase,
                "angles": self.angles,
                "active_side": self.active_side,
                "coaching_tip": self.tip_text,
                "announcement": ann,
                "faulty_joint": self.faulty_joint,
                "has_mistake": self.has_mistake,
                "landmarks": self.latest_landmarks,
                "latest_score": self.latest_score,
                "latest_feedback": self.latest_feedback,
                "is_active": self.is_running
            }

    def _capture_loop(self):
        logger.info("Initializing CameraManager direct capture thread...")
        try:
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)

            while self.is_running:
                if not self.cap.isOpened():
                    placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(placeholder, "Connecting to camera / Please connect webcam...", (50, 240),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 100), 2)
                    _, buf = cv2.imencode('.jpg', placeholder)
                    with self.lock:
                        self.latest_jpeg = buf.tobytes()
                    time.sleep(0.5)
                    self.cap.open(0)
                    continue

                ret, frame = self.cap.read()
                if not ret:
                    time.sleep(0.03)
                    continue

                ts_ms = int(time.time() * 1000)
                with self.lock:
                    session = self.session

                if session is not None:
                    raw_lm = session.analyzer.extract_landmarks(frame, ts_ms)
                    if raw_lm is not None:
                        lm_mocks = [
                            LandmarkMock(p.x, p.y, p.z, getattr(p, "visibility", 1.0))
                            for p in raw_lm
                        ]
                        res = session._evaluate_landmarks(lm_mocks)
                        with self.lock:
                            self.rep_count = res.get("rep_count", 0)
                            self.hold_seconds = res.get("hold_seconds", 0.0)
                            self.is_holding = res.get("is_holding", False)
                            self.progress_pct = res.get("progress_pct", 0.0)
                            self.phase = res.get("phase", "IDLE")
                            self.angles = res.get("angles") or {}
                            self.active_side = res.get("active_side", "left")
                            self.tip_text = res.get("coaching_tip", "")
                            self.faulty_joint = res.get("faulty_joint", "")
                            self.has_mistake = res.get("has_mistake", False)
                            if res.get("announcement"):
                                self.pending_announcement = res.get("announcement")
                            if res.get("new_rep") and res.get("latest_score") is not None:
                                self.latest_score = res.get("latest_score")
                                self.latest_feedback = res.get("latest_feedback", "")

                            # Export normalized landmarks with rounded coordinates for fast JSON transfer
                            self.latest_landmarks = [
                                {
                                    "x": round(float(p.x), 4),
                                    "y": round(float(p.y), 4),
                                    "z": round(float(p.z), 4),
                                    "v": round(float(getattr(p, "visibility", 1.0)), 2)
                                }
                                for p in raw_lm
                            ]
                    else:
                        with self.lock:
                            self.latest_landmarks = None

                # Keep video clean: no pixelated OpenCV text or lines burned into the frame
                _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                with self.lock:
                    self.latest_jpeg = jpeg.tobytes()

                time.sleep(0.015)
        except Exception as e:
            logger.error(f"Error in CameraManager loop: {e}")
        finally:
            if self.cap:
                self.cap.release()

    def generate_mjpeg(self):
        self.start()
        while self.is_running:
            with self.lock:
                frame_bytes = self.latest_jpeg
            if frame_bytes is not None:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.033)

camera_manager = CameraManager()

# =====================================================================
# API Endpoints
# =====================================================================
class SessionSaveRequest(BaseModel):
    session_id: str
    user_id: str
    exercise_type: str
    rep_count: int
    avg_score: float

class ExerciseInfo(BaseModel):
    id: str
    name: str
    name_ar: str
    type: str
    view: str
    muscle: str
    muscle_ar: str

@app.get("/", response_class=HTMLResponse)
def read_root():
    index_file = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return HTMLResponse("<h2>Welcome to CV for Fit Backend Engine API</h2>")

@app.get("/video_feed")
def video_feed():
    """
    Direct MJPEG Stream:
    Captures webcam directly on PC via OpenCV, runs MediaPipe pose tracking,
    renders HUD, and streams frames directly to the browser with zero dropped connections.
    """
    return StreamingResponse(
        camera_manager.generate_mjpeg(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.get("/api/status")
def get_live_status():
    """Current live metrics from the CameraManager."""
    return camera_manager.get_status()

@app.post("/api/control")
def control_session(payload: Dict[str, Any]):
    """Update active exercise, language, or reset session."""
    if "exercise" in payload:
        camera_manager.set_exercise(payload["exercise"])
    if "lang" in payload:
        camera_manager.set_language(payload["lang"])
    if payload.get("action") == "reset":
        camera_manager.reset()
    return {"status": "success", "data": camera_manager.get_status()}

@app.get("/api/exercises", response_model=List[ExerciseInfo])
def get_supported_exercises():
    """List all 8 exercises supported by the CV engine with bilingual metadata."""
    return [
        ExerciseInfo(id="bicep_curl", name="Bicep Curl", name_ar="بايسبس كيرل (عضلات الذراع)", type="reps", view="side", muscle="Biceps", muscle_ar="البايسبس"),
        ExerciseInfo(id="squat", name="Squat", name_ar="سكوات (القرفصاء)", type="reps", view="side", muscle="Quads/Glutes", muscle_ar="الفخذ والمؤخرة"),
        ExerciseInfo(id="lunge", name="Lunge", name_ar="طعنات الساق (لونجز)", type="reps", view="side", muscle="Quads/Hamstrings", muscle_ar="أوتار الفخذ والساقين"),
        ExerciseInfo(id="push_up", name="Push Up", name_ar="تمرين الضغط", type="reps", view="side", muscle="Chest/Triceps", muscle_ar="الصدر والترايسبس"),
        ExerciseInfo(id="superman", name="Superman", name_ar="تمرين سوبرمان", type="reps", view="side", muscle="Lower Back", muscle_ar="أسفل الظهر والأكتاف"),
        ExerciseInfo(id="shoulder_press", name="Shoulder Press", name_ar="دفع الكتف العلوي", type="reps", view="front", muscle="Shoulders", muscle_ar="الأكتاف"),
        ExerciseInfo(id="dips", name="Chair Dips", name_ar="غطس المقعد (ديبس)", type="reps", view="side", muscle="Triceps", muscle_ar="الترايسبس"),
        ExerciseInfo(id="plank", name="Plank", name_ar="تمرين البلانك (ثبات)", type="hold", view="side", muscle="Core", muscle_ar="عضلات الجذع والبطن"),
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

@app.post("/api/video/analyze")
async def analyze_video(
    file: UploadFile = File(...),
    exercise: str = Form(...),
    lang: str = Form("ar"),
):
    """
    Video Lab Analysis Endpoint:
    Processes an uploaded workout video, tracks execution frame-by-frame,
    evaluates kinetic accuracy via DTW, and returns an extensive verdict, score,
    detected mistakes, and actionable improvement recommendations.
    """
    try:
        ex_type = ExerciseType(exercise)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Unsupported exercise: '{exercise}'")

    suffix = os.path.splitext(file.filename or "")[1] or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_path = tmp.name
        shutil.copyfileobj(file.file, tmp)

    cap = cv2.VideoCapture(tmp_path)
    if not cap.isOpened():
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise HTTPException(status_code=400, detail="Could not open video file.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps if total_frames > 0 else 0.0

    session = RemoteSession(exercise, lang=lang)
    ref_seq = get_exercise_reference(ex_type)
    session.set_reference(ref_seq)

    mistake_counts: Dict[str, int] = {}
    frame_idx = 0
    stride = 1 if total_frames < 600 else 2

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            if stride > 1 and (frame_idx % stride != 0):
                continue

            ts_ms = int((frame_idx / fps) * 1000)
            raw_lm = session.analyzer.extract_landmarks(frame, ts_ms)
            if raw_lm is not None:
                lm_mocks = [
                    LandmarkMock(p.x, p.y, p.z, getattr(p, "visibility", 1.0))
                    for p in raw_lm
                ]
                res = session._evaluate_landmarks(lm_mocks)
                k = res.get("tip_key")
                if k:
                    mistake_counts[k] = mistake_counts.get(k, 0) + 1
    finally:
        cap.release()
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass

    is_hold_exercise = (ex_type == ExerciseType.PLANK)
    total_reps = session.rep_state.rep_count
    hold_sec = session.rep_state.hold_seconds
    evals = session.evals

    if is_hold_exercise:
        effective_time = max(duration_sec, 1.0)
        posture_ratio = hold_sec / effective_time
        score = float(np.clip(posture_ratio * 100.0, 0.0, 100.0))
        is_pass = (hold_sec >= 10.0 and score >= 60.0) or (hold_sec >= 20.0)
    else:
        if evals:
            score = float(np.clip(np.mean([e.form_score for e in evals]), 0.0, 100.0))
            is_pass = (total_reps >= 1 and score >= 60.0)
        else:
            if total_reps > 0:
                score = 70.0
                is_pass = True
            else:
                score = 25.0
                is_pass = False

    if lang == "ar":
        verdict_text = "ناجح ومتقن" if is_pass else "يحتاج إلى تصحيح المسار"
        verdict_sub = "أظهرت الحركة توافقاً حركياً سليماً وثباتاً ملحوظاً." if is_pass else "تم رصد أخطاء في المدى الحركي أو استقامة المفاصل تؤثر على كفاءة التمرين."
    else:
        verdict_text = "PASS (Optimal Form)" if is_pass else "NEEDS CORRECTION"
        verdict_sub = "Biomechanical trajectory is aligned with reference standards." if is_pass else "Deviations detected in range of motion or joint posture."

    sorted_mistakes = sorted(mistake_counts.items(), key=lambda x: x[1], reverse=True)
    key_mistakes = []
    actionable_tips = []

    for k, count in sorted_mistakes[:4]:
        info = COACH_TIPS_I18N.get(k, {})
        text_ar = info.get("ar", k)
        text_en = info.get("en", k)
        mistake_text = text_ar if lang == "ar" else text_en
        key_mistakes.append({
            "key": k,
            "description": mistake_text,
            "occurrences": count
        })
        actionable_tips.append(mistake_text)

    if not key_mistakes:
        if lang == "ar":
            actionable_tips.append("أداء مثالي! حافظ على استمرارية هذا التوافق الحركي والتحكم بسرعة العدات.")
        else:
            actionable_tips.append("Flawless execution! Maintain this cadence and core stabilization.")

    rep_breakdown = []
    for i, ev in enumerate(evals, 1):
        rep_status_ar = "ممتاز" if ev.form_score >= 80 else ("جيد" if ev.form_score >= 60 else "يحتاج تركيز")
        rep_status_en = "Optimal" if ev.form_score >= 80 else ("Good" if ev.form_score >= 60 else "Needs Work")
        rep_breakdown.append({
            "rep_number": i,
            "score": round(ev.form_score, 1),
            "is_correct": ev.is_correct,
            "feedback": ev.feedback,
            "status": rep_status_ar if lang == "ar" else rep_status_en,
        })

    return {
        "exercise": exercise,
        "exercise_type": "hold" if is_hold_exercise else "reps",
        "duration_seconds": round(duration_sec, 1),
        "total_frames_analyzed": frame_idx,
        "reps_completed": total_reps,
        "hold_seconds": round(hold_sec, 1),
        "form_score": round(score, 1),
        "verdict": "PASS" if is_pass else "FAIL",
        "verdict_label": verdict_text,
        "verdict_sub": verdict_sub,
        "key_mistakes": key_mistakes,
        "actionable_tips": actionable_tips,
        "rep_breakdown": rep_breakdown,
    }

@app.websocket("/ws/workout")
async def websocket_workout_endpoint(
    websocket: WebSocket,
    exercise: str = Query(..., description="Exercise type ID"),
    userId: str = Query("guest", description="User identifier"),
    lang: str = Query("ar", description="Language preference ('ar' or 'en')")
):
    """Fallback WebSocket endpoint for browser-based video frames or landmark streaming."""
    await websocket.accept()
    logger.info(f"WebSocket client connected. User: {userId}, Exercise: {exercise}, Lang: {lang}")
    
    try:
        ex_type = ExerciseType(exercise)
    except ValueError:
        logger.error(f"Invalid exercise requested: {exercise}")
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Invalid exercise type")
        return

    session = RemoteSession(exercise, lang=lang)
    ref_seq = get_exercise_reference(ex_type)
    session.set_reference(ref_seq)
    
    session_id = str(uuid.uuid4())
    await websocket.send_json({
        "status": "initialized",
        "session_id": session_id,
        "exercise": exercise,
        "orientation_required": "front" if ex_type in (ExerciseType.SHOULDER_PRESS,) else "side",
        "lang": lang
    })
    
    try:
        while True:
            try:
                data = await websocket.receive_json()
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.warning(f"Error reading websocket message: {e}")
                continue

            try:
                if not isinstance(data, dict) and not isinstance(data, list):
                    continue

                if isinstance(data, dict):
                    action = data.get("action")
                    if action == "set_lang":
                        new_lang = data.get("lang", "ar")
                        session.set_language(new_lang)
                        await websocket.send_json({"status": "language_updated", "lang": new_lang})
                        continue
                    elif action == "reset":
                        session.reset()
                        await websocket.send_json({"status": "session_reset"})
                        continue

                    if "image" in data and data["image"]:
                        result = session.process_image(data["image"])
                        await websocket.send_json(result)
                        continue

                    landmarks = data.get("landmarks")
                else:
                    landmarks = data

                if not landmarks:
                    continue

                result = session.process_frame(landmarks)
                await websocket.send_json(result)
            except Exception as frame_err:
                logger.warning(f"Error processing frame payload: {frame_err}")
                continue
            
    except WebSocketDisconnect:
        logger.info(f"WebSocket client disconnected gracefully. Session: {session_id}")
    except Exception as e:
        logger.error(f"Error in WebSocket handler: {e}")
    finally:
        try:
            await websocket.close()
        except Exception:
            pass
